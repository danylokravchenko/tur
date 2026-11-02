"""Golden GGUF block decodings from ggml itself, for the host and device decoders.

Random bytes decode to valid values in every format, so each block comes from a fixed
generator and only its scale fields are forced to stay finite.
  python -I scripts/gguf_blocks_reference.py /opt/homebrew/lib/libggml-base.dylib \
      crates/tur/tests/fixtures/gguf/blocks.json
"""
import ctypes
import json
import random
import struct
import sys

# name: (ggml decoder, values per block, block bytes, offsets of F16 scales)
FORMATS = {
    "Q4_0": ("q4_0", 32, 18, [0]),
    "Q4_1": ("q4_1", 32, 20, [0, 2]),
    "Q5_0": ("q5_0", 32, 22, [0]),
    "Q5_1": ("q5_1", 32, 24, [0, 2]),
    "Q8_0": ("q8_0", 32, 34, [0]),
    "Q2_K": ("q2_K", 256, 84, [80, 82]),
    "Q3_K": ("q3_K", 256, 110, [108]),
    "Q4_K": ("q4_K", 256, 144, [0, 2]),
    "Q5_K": ("q5_K", 256, 176, [0, 2]),
    "Q6_K": ("q6_K", 256, 210, [208]),
    "IQ1_S": ("iq1_s", 256, 50, [0]),
    "IQ1_M": ("iq1_m", 256, 56, []),
    "IQ2_XXS": ("iq2_xxs", 256, 66, [0]),
    "IQ2_XS": ("iq2_xs", 256, 74, [0]),
    "IQ2_S": ("iq2_s", 256, 82, [0]),
    "IQ3_XXS": ("iq3_xxs", 256, 98, [0]),
    "IQ3_S": ("iq3_s", 256, 110, [0]),
    "IQ4_NL": ("iq4_nl", 32, 18, [0]),
    "IQ4_XS": ("iq4_xs", 256, 136, [0]),
    "MXFP4": ("mxfp4", 32, 17, []),
}
BLOCKS = 2


def block(rng, size, scales):
    data = bytearray(rng.randrange(256) for _ in range(size))
    for at in scales:
        # A positive half near 2^-8.
        data[at : at + 2] = struct.pack("<H", 0x2000 | rng.randrange(1 << 10))
    return data


def main():
    ggml = ctypes.CDLL(sys.argv[1])
    golden = {}
    for name, (decoder, values, size, scales) in FORMATS.items():
        rng = random.Random(name)
        blocks = bytearray()
        for _ in range(BLOCKS * 256 // values):
            part = block(rng, size, scales)
            if name == "IQ1_M":
                # The scale's high nibbles live in the top of four 16-bit words.
                part[49] = (part[49] & 0x0F) | 0x30
                part[51] &= 0x0F
                part[53] &= 0x0F
                part[55] &= 0x0F
            if name == "MXFP4":
                part[0] = 120 + rng.randrange(8)
            blocks += part
        count = len(blocks) // size * values
        out = (ctypes.c_float * count)()
        decode = getattr(ggml, f"dequantize_row_{decoder}")
        decode.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int64]
        decode(bytes(blocks), out, count)
        golden[name] = {
            "blocks": bytes(blocks).hex(),
            "values": [struct.pack("<f", v).hex() for v in out],
        }
    with open(sys.argv[2], "w") as file:
        json.dump(golden, file, separators=(",", ":"))


if __name__ == "__main__":
    main()
