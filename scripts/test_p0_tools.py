import json
from pathlib import Path
import tempfile
import unittest

from p0_trace_summary import summarize


class BaselineToolsTests(unittest.TestCase):
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
