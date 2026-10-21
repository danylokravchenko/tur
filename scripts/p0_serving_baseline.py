"""Measure the existing HTTP server with a pinned cache and retain every request result."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
import hashlib
from pathlib import Path
import subprocess
import threading
import time


def request(port, content, barrier=None, max_tokens=16):
    body = {"model": "p0", "messages": [{"role": "user", "content": content}],
            "stream": True, "max_tokens": max_tokens, "thinking": False}
    if barrier is not None:
        barrier.wait(timeout=30)
    start = time.perf_counter()
    result = {"request": body, "text_events_seconds": [], "text": "", "usage": None, "done": False}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        result["status"] = response.status
        if response.status != 200:
            raise RuntimeError(response.read().decode())
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                result["done"] = True
                break
            event = json.loads(data)
            if event.get("usage"):
                result["usage"] = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    result["text_events_seconds"].append(time.perf_counter() - start)
                    result["text"] += text
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
        if not result["done"] or not result["usage"]:
            raise RuntimeError("stream ended without DONE and usage")
    except Exception as error:
        result["error"] = str(error)
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start
        connection.close()
    return result


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19841)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--prefix-cache", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    command = ["/usr/bin/time", "-l", str(args.binary.resolve()), "--model", "p0:Qwen/Qwen3-0.6B:Q4_K_M",
               "--port", str(args.port), "--temperature", "0", "--seed", "299792458", "--repeat-penalty", "1",
               "--max-batch-size", "4"]
    if args.prefix_cache:
        command.append("--prefix-cache")
    environment = dict(os.environ, HF_HOME=str(args.hf_home.resolve()), HF_ENDPOINT="http://127.0.0.1:9", RUST_LOG="warn")
    report = {"schema_version": 1, "command": command, "prefix_cache": args.prefix_cache,
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checkpoint_manifest": json.loads((args.hf_home / "checkpoints.json").read_text()),
              "activation_dtype": "server default BF16", "seed": 299792458, "temperature": 0,
              "repeat_penalty": 1, "queueing": "included in client latency; not separately exposed",
              "interval_semantics": "SSE text-event intervals; events can contain several tokens",
              "runs": []}
    started = time.perf_counter()
    with (args.output / "server.stdout").open("w") as stdout, (args.output / "server.stderr").open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError("server exited during startup")
                try:
                    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=1)
                    connection.request("GET", "/v1/models")
                    ready = connection.getresponse().status == 200
                    connection.close()
                    if ready:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server did not listen within 60 seconds")
                time.sleep(0.1)
            report["listen_seconds"] = time.perf_counter() - started
            report["warmup"] = request(args.port, "Explain Rust ownership briefly.", max_tokens=2)
            if "error" in report["warmup"]:
                raise RuntimeError(f"warmup failed: {report['warmup']['error']}")
            for concurrency in (1, 2, 4):
                for iteration in range(args.iterations):
                    barrier = threading.Barrier(concurrency)
                    prompts = ["Explain Rust ownership briefly." if i % 2 == 0 else "Rust " * 100 + "Explain ownership."
                               for i in range(concurrency)]
                    start = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        results = list(executor.map(lambda text: request(args.port, text, barrier), prompts))
                    elapsed = time.perf_counter() - start
                    successful = [item for item in results if "error" not in item]
                    ttft = [item["text_events_seconds"][0] for item in successful if item["text_events_seconds"]]
                    intervals = [later - earlier for item in successful
                                 for earlier, later in zip(item["text_events_seconds"], item["text_events_seconds"][1:])]
                    report["runs"].append({
                        "concurrency": concurrency, "iteration": iteration, "elapsed_seconds": elapsed,
                        "successful_requests": len(successful),
                        "completion_tokens_per_second": sum(item["usage"]["completion_tokens"] for item in successful) / elapsed,
                        "ttft_seconds": {name: percentile(ttft, quantile) for name, quantile in (("p50", .5), ("p95", .95), ("p99", .99))},
                        "text_event_intervals_seconds": {name: percentile(intervals, quantile) for name, quantile in (("p50", .5), ("p95", .95), ("p99", .99))},
                        "requests": results,
                    })
        except Exception as error:
            report["error"] = str(error)
        finally:
            # Interrupt the measurement's own server so time(1) writes peak resident memory.
            children = subprocess.run(["pgrep", "-P", str(process.pid)], capture_output=True, text=True, check=False)
            import signal
            for child in children.stdout.split():
                try:
                    os.kill(int(child), signal.SIGINT)
                except ProcessLookupError:
                    pass
            if not children.stdout.strip() and process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
            report["server_exit_code"] = process.returncode
    report["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if "error" in report or any("error" in item for run in report["runs"] for item in run["requests"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
