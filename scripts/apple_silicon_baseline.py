"""Capture a synthetic Metal run and machine provenance without checkpoint downloads."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def capture(command, environment=None):
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("dev", "pre-release"), default="pre-release")
    parser.add_argument("--example", choices=("metal_baseline",), default="metal_baseline")
    parser.add_argument("example_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "checkpoint": None,
        "tokenizer": None,
        "template": None,
        "quantizer": None,
        "thermal_conditions": "not instrumented",
        "commands": {},
    }
    commands = {
        "revision": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short"],
        "rust": ["rustc", "--version", "--verbose"],
        "cargo": ["cargo", "--version"],
        "os": ["sw_vers"],
        "power_settings": ["pmset", "-g", "custom"],
        "power_source": ["pmset", "-g", "batt"],
        "thermal": ["pmset", "-g", "therm"],
        "xcode": ["xcodebuild", "-version"],
        "metal_compiler": ["xcrun", "metal", "--version"],
    }
    for name, command in commands.items():
        metadata["commands"][name] = capture(command)
    hardware = capture(["system_profiler", "-json", "SPHardwareDataType", "SPDisplaysDataType"])
    if hardware["exit_code"] == 0:
        data = json.loads(hardware["stdout"])
        # Keep performance-relevant fields; omit machine serials and UUIDs.
        metadata["hardware"] = [
            {key: item[key] for key in ("machine_model", "chip_type", "physical_memory", "number_processors") if key in item}
            for item in data.get("SPHardwareDataType", [])
        ]
        metadata["gpu"] = [
            {key: item[key] for key in ("sppci_model", "sppci_cores", "spdisplays_metal") if key in item}
            for item in data.get("SPDisplaysDataType", [])
        ]
    metadata["hardware_capture_exit_code"] = hardware["exit_code"]
    sources = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    sources += sorted((ROOT / "scripts").glob("*.py"))
    sources += sorted((ROOT / "crates").rglob("Cargo.toml"))
    sources += sorted((ROOT / "crates").rglob("*.rs"))
    sources += sorted((ROOT / "crates").rglob("*.metal"))
    metadata["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    environment = dict(os.environ)
    build = capture(["cargo", "build", "--locked", "--offline", "-p", "tur", "--example", args.example, "--profile", args.profile])
    (args.output / "build.stdout").write_text(build.pop("stdout"))
    (args.output / "build.stderr").write_text(build.pop("stderr"))
    metadata["build"] = build
    profile_dir = "debug" if args.profile == "dev" else args.profile
    example_args = args.example_args[1:] if args.example_args[:1] == ["--"] else args.example_args
    command = ["/usr/bin/time", "-l", str(ROOT / "target" / profile_dir / "examples" / args.example), *example_args]
    result = capture(command, environment) if build["exit_code"] == 0 else {"command": command, "exit_code": build["exit_code"], "stdout": "", "stderr": "build failed; execution not attempted"}
    (args.output / "run.stdout").write_text(result.pop("stdout"))
    (args.output / "run.stderr").write_text(result.pop("stderr"))
    metadata["run"] = result
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if result["exit_code"] != 0:
        raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
