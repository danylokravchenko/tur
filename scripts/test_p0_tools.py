import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import p0_checkpoints
from p0_serving_baseline import percentile
from p0_trace_summary import summarize


class BaselineToolsTests(unittest.TestCase):
    def test_percentiles_interpolate_and_handle_empty_samples(self):
        self.assertIsNone(percentile([], .95))
        self.assertEqual(percentile([9, 1, 5], .5), 5)
        self.assertAlmostEqual(percentile([1, 5], .95), 4.8)
        self.assertEqual(percentile([7], .99), 7)

    def test_checkpoint_materialization_hashes_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "cache/models--Qwen--Qwen3-0.6B/snapshots/pinned"
            repo.mkdir(parents=True)
            source = repo / "tokenizer_config.json"
            source.write_text(json.dumps({"chat_template": "fixture"}))
            output = root / "isolated"
            with patch.object(p0_checkpoints, "CHECKPOINTS", [("Qwen/Qwen3-0.6B", "pinned", [source.name])]):
                manifest = p0_checkpoints.materialize(root / "cache", output)
                self.assertEqual(manifest["repositories"][0]["revision"], "pinned")
                self.assertEqual(manifest["repositories"][0]["files"][0]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertEqual((output / "hub/models--Qwen--Qwen3-0.6B/refs/main").read_text(), "pinned")
                with self.assertRaises(FileExistsError):
                    p0_checkpoints.materialize(root / "cache", output)

    def test_trace_summary_counts_referenced_labels_and_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            toc = root / "toc.xml"
            events = root / "events.xml"
            toc.write_text('<trace-toc><run><data><table schema="metal-application-intervals"/><table schema="metal-current-allocated-size"/></data></run></trace-toc>')
            events.write_text('''<trace-query-result>
              <node xpath="table[1]">
                <row><formatted-label id="1"><metal-object-label id="2">blit_to_cpu</metal-object-label></formatted-label><metal-event-name id="3">Encoding</metal-event-name><metal-command-buffer-id id="4">15</metal-command-buffer-id></row>
                <row><formatted-label ref="1"/><metal-event-name ref="3"/><metal-command-buffer-id ref="4"/></row>
              </node>
              <node xpath="table[2]"><row><size-in-bytes id="5">4096</size-in-bytes></row><row><size-in-bytes ref="5"/></row></node>
            </trace-query-result>''')
            result = summarize(toc, events)["tables"]
            self.assertEqual(result["metal-application-intervals"]["encoder_label_components"], {"blit_to_cpu": 2})
            self.assertEqual(result["metal-application-intervals"]["unique_command_buffer_ids"], 1)
            self.assertEqual(result["metal-current-allocated-size"]["peak_reported_bytes"], 4096)


if __name__ == "__main__":
    unittest.main()
