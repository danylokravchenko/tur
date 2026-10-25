"""Offline byte-preservation and rejection contracts for partial_model.py.

    python3 -I scripts/test_partial_model.py
"""

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

SPEC = importlib.util.spec_from_file_location("partial_model", Path(__file__).with_name("partial_model.py"))
PARTIAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARTIAL)


class PartialModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "subset"
        self.config = {
            "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_text", "num_hidden_layers": 4, "mtp_num_hidden_layers": 1,
                "tie_word_embeddings": True, "full_attention_interval": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            },
        }
        (self.source / "config.json").write_text(json.dumps(self.config))
        (self.source / "tokenizer.json").write_bytes(b'{"fixture":true}\n')
        (self.source / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}')
        self.values = {
            "model.language_model.embed_tokens.weight": 0.5,
            "model.language_model.norm.weight": 1.5,
            **{f"model.language_model.layers.{i}.mlp.up_proj.weight": i + 2.5 for i in range(4)},
            "model.visual.weight": 6.5,
            "mtp.layers.0.weight": 7.5,
        }
        header = {}
        payload = bytearray()
        for name, value in self.values.items():
            header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [len(payload), len(payload) + 4]}
            payload.extend(struct.pack("<f", value))
        encoded = json.dumps(header).encode()
        self.weights = self.source / "model.safetensors"
        self.weights.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)

    def test_export_preserves_retained_bytes_and_remaps_original_layers(self):
        before = self.weights.read_bytes()
        manifest = PARTIAL.export_model(self.source, self.output, "pinned", [1])
        raw = (self.output / "model.safetensors").read_bytes()
        length = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + length])
        actual = {
            name: raw[8 + length + info["data_offsets"][0]:8 + length + info["data_offsets"][1]]
            for name, info in header.items() if name != "__metadata__"
        }
        expected = {
            "model.embed_tokens.weight": struct.pack("<f", 0.5),
            "model.norm.weight": struct.pack("<f", 1.5),
            "model.layers.0.mlp.up_proj.weight": struct.pack("<f", 2.5),
            "model.layers.1.mlp.up_proj.weight": struct.pack("<f", 4.5),
            "model.layers.2.mlp.up_proj.weight": struct.pack("<f", 5.5),
        }
        self.assertEqual(actual, expected)
        self.assertEqual(manifest["layer_mapping"], {"0": 0, "1": 2, "2": 3})
        self.assertEqual(manifest["source_text_parameters"], 6)
        self.assertEqual(manifest["retained_text_parameters"], 5)
        config = json.loads((self.output / "config.json").read_bytes())
        self.assertEqual(config["architectures"], ["Qwen3_5ForCausalLM"])
        self.assertEqual(config["num_hidden_layers"], 3)
        self.assertEqual(config["mtp_num_hidden_layers"], 0)
        self.assertEqual(config["layer_types"], ["linear_attention", "linear_attention", "full_attention"])
        self.assertNotIn("full_attention_interval", config)
        self.assertEqual((self.output / "tokenizer.json").read_bytes(), b'{"fixture":true}\n')
        self.assertEqual(self.weights.read_bytes(), before)

    def test_calibration_ranks_original_ids_and_checks_provenance(self):
        scores = {
            "schema_version": 1, "source_revision": "pinned", "config": self.config,
            "scores": [{"layer": i, "influence": score} for i, score in [(1, 0.4), (2, 0.1), (3, 0.0)]],
        }
        manifest = PARTIAL.export_model(self.source, self.output, "pinned", scores=scores, remove=1)
        self.assertEqual(manifest["removed_layers"], [2])
        for revision, count in [("different", 1), ("pinned", 3), ("pinned", -1)]:
            with self.subTest(revision=revision, count=count), self.assertRaises(ValueError):
                PARTIAL.export_model(self.source, self.root / "invalid", revision, scores=scores, remove=count)
        scores["scores"][0]["influence"] = float("nan")
        with self.assertRaises(ValueError):
            PARTIAL.export_model(self.source, self.root / "invalid", "pinned", scores=scores, remove=1)
        self.assertFalse((self.root / "invalid").exists())

    def test_protected_duplicate_and_out_of_range_layers_are_rejected(self):
        for dropped in ([0], [3], [4], [-1], [1, 1]):
            with self.subTest(dropped=dropped), self.assertRaises(ValueError):
                PARTIAL.export_model(self.source, self.output, "pinned", dropped)
        self.assertFalse(self.output.exists())

    def test_invalid_payload_and_index_cannot_publish_checkpoint(self):
        valid = self.weights.read_bytes()
        self.weights.write_bytes(valid[:-1])
        with self.assertRaisesRegex(ValueError, "file length"):
            PARTIAL.export_model(self.source, self.output, "pinned", [1])
        self.weights.write_bytes(valid)
        index = self.source / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"stolen": "../other.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "filenames"):
            PARTIAL.export_model(self.source, self.output, "pinned", [1])
        index.write_text(json.dumps({"weight_map": {"missing": "model.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "index disagrees"):
            PARTIAL.export_model(self.source, self.output, "pinned", [1])
        self.assertFalse(self.output.exists())

    def test_existing_output_and_failed_export_preserve_user_files(self):
        self.output.mkdir()
        sentinel = self.output / "user.txt"
        sentinel.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "output must not exist"):
            PARTIAL.export_model(self.source, self.output, "pinned", [1])
        self.assertEqual(sentinel.read_text(), "keep me")
        (self.source / "tokenizer_config.json").write_text("{}")
        missing_template = self.root / "missing-template"
        with self.assertRaisesRegex(ValueError, "chat template"):
            PARTIAL.export_model(self.source, missing_template, "pinned", [1])
        self.assertFalse(missing_template.exists())
        self.assertFalse(list(self.root.glob(".tur-partial-*")))

    def width_fixture(self):
        self.config["text_config"].update(hidden_size=2, intermediate_size=4)
        (self.source / "config.json").write_text(json.dumps(self.config))
        header = {}
        payload = bytearray()
        for layer in range(4):
            for projection, start, shape in [("gate_proj", 1, [4, 2]), ("up_proj", 11, [4, 2]),
                                             ("down_proj", 21, [2, 4])]:
                name = f"model.language_model.layers.{layer}.mlp.{projection}.weight"
                header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(payload), len(payload) + 32]}
                payload.extend(struct.pack("<8f", *(layer * 100 + start + i for i in range(8))))
        for name in ["embed_tokens", "norm"]:
            header[f"model.language_model.{name}.weight"] = {
                "dtype": "F32", "shape": [1], "data_offsets": [len(payload), len(payload) + 4],
            }
            payload.extend(struct.pack("<f", 0.5))
        encoded = json.dumps(header).encode()
        self.weights.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        return {
            "schema_version": 1, "axis": "mlp_width", "source_revision": "pinned", "config": self.config,
            "scores": [{"layer": i, "importance": [0.1, 4.0, 0.2, 3.0]} for i in range(4)],
        }

    def test_width_selection_keeps_all_layers_and_couples_swiglu_matrices(self):
        scores = self.width_fixture()
        manifest = PARTIAL.export_model(self.source, self.output, "pinned", neuron_scores=scores, width=2)
        self.assertEqual(manifest["retained_layers"], [0, 1, 2, 3])
        self.assertEqual(manifest["removed_layers"], [])
        self.assertEqual(manifest["retained_neurons"], {str(i): [1, 3] for i in range(4)})
        self.assertEqual(manifest["source_text_parameters"], 98)
        self.assertEqual(manifest["retained_text_parameters"], 50)
        config = json.loads((self.output / "config.json").read_bytes())
        self.assertEqual(config["intermediate_size"], 2)
        self.assertEqual(config["layer_types"], self.config["text_config"]["layer_types"])
        raw = (self.output / "model.safetensors").read_bytes()
        length = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + length])
        for layer in range(4):
            for projection, values in [("gate_proj", [3, 4, 7, 8]), ("up_proj", [13, 14, 17, 18]),
                                       ("down_proj", [22, 24, 26, 28])]:
                info = header[f"model.layers.{layer}.mlp.{projection}.weight"]
                start, end = info["data_offsets"]
                actual = raw[8 + length + start:8 + length + end]
                self.assertEqual(actual, struct.pack("<4f", *(layer * 100 + n for n in values)))
                self.assertEqual(info["shape"], [2, 2])

    def test_invalid_width_calibration_cannot_publish_checkpoint(self):
        scores = self.width_fixture()
        for width in [0, -1, 5]:
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "width"):
                PARTIAL.export_model(self.source, self.output, "pinned", neuron_scores=scores, width=width)
        with self.assertRaisesRegex(ValueError, "separately"):
            PARTIAL.export_model(self.source, self.output, "pinned", [1], neuron_scores=scores, width=2)
        scores["scores"][0]["importance"] = [1.0]
        with self.assertRaisesRegex(ValueError, "importance"):
            PARTIAL.export_model(self.source, self.output, "pinned", neuron_scores=scores, width=2)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
