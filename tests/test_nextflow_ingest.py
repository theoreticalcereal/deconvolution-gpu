from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_NF = REPO_ROOT / "workflow" / "main.nf"


class NextflowIngestTest(unittest.TestCase):
    def test_astrocyte_input_uses_nextflow_from_path(self):
        main_nf = MAIN_NF.read_text()

        self.assertIn("Channel.fromPath(params.input, checkIfExists: true)", main_nf)
        self.assertNotIn("normalizeInputPatterns", main_nf)
        self.assertNotIn("Channel\n            .fromList", main_nf)

    def test_selected_inputs_are_collected_before_staging(self):
        main_nf = MAIN_NF.read_text()

        self.assertRegex(
            main_nf,
            re.compile(
                r"input_tiffs_ch\s*=\s*Channel\.fromPath"
                r"\(params\.input,\s*checkIfExists:\s*true\)\s*"
                r"\.collect\(\)",
                re.MULTILINE,
            ),
        )


if __name__ == "__main__":
    unittest.main()
