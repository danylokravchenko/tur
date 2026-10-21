"""Summarize selected xctrace Metal tables without publishing machine identifiers."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def summarize(toc_path, events_path):
    toc = ET.parse(toc_path).getroot()
    events = ET.parse(events_path).getroot()
    definitions = {element.attrib["id"]: element for element in events.iter() if "id" in element.attrib}

    def resolve(element):
        if element is None:
            return None
        return definitions[element.attrib["ref"]] if "ref" in element.attrib else element

    def text(element):
        element = resolve(element)
        return element.text if element is not None else None

    tables = toc.findall("./run/data/table")
    target = toc.find("./run/info/target/process")
    target_pid = target.attrib.get("pid") if target is not None else None
    report = {"tables": {}, "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                               for path in (toc_path, events_path)}}
    for node in events:
        match = re.search(r"table\[(\d+)\]", node.attrib["xpath"])
        if not match:
            continue
        name = tables[int(match.group(1)) - 1].attrib["schema"]
        rows = node.findall("row")
        if target_pid:
            rows = [row for row in rows if resolve(row.find("process")) is not None
                    and text(resolve(row.find("process")).find("pid")) == target_pid]
        result = {"rows": len(rows)}
        if name == "metal-current-allocated-size":
            result["peak_reported_bytes"] = max((int(text(row.find("size-in-bytes"))) for row in rows), default=0)
        elif name == "metal-application-intervals":
            result["events"] = dict(Counter(text(row.find("metal-event-name")) for row in rows))
            labels = [resolve(row.find("formatted-label")) for row in rows]
            components = Counter()
            for label in labels:
                if label is not None:
                    name_text = text(label.find("metal-object-label"))
                    if name_text and not name_text.startswith("Command Buffer"):
                        components.update(name_text.split(" & "))
            result["encoder_label_components"] = dict(components)
            result["unique_command_buffer_ids"] = len({text(row.find("metal-command-buffer-id")) for row in rows
                                                       if row.find("metal-command-buffer-id") is not None})
        elif name == "metal-resource-allocations":
            result["events"] = dict(Counter(text(row.find("gpu-memory-event-name")) for row in rows))
        elif name == "metal-driver-event-intervals":
            result["events"] = dict(Counter(text(row.findall("gpu-driver-name")[-1]) for row in rows
                                             if row.findall("gpu-driver-name")))
        elif name == "metal-gpu-intervals":
            result["unique_command_buffer_ids"] = len({text(row.find("metal-command-buffer-id")) for row in rows
                                                       if row.find("metal-command-buffer-id") is not None})
        report["tables"][name] = result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toc", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.toc, args.events), indent=2) + "\n")


if __name__ == "__main__":
    main()
