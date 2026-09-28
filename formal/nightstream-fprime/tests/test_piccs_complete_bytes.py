import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "check_piccs_complete_bytes", Path(__file__).with_name("check_piccs_complete_bytes.py"))
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class PiCCSCompleteBytesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        eval_k = [[[0, 0] for _ in range(54)] for _ in range(17)]
        eval_a = [[[[0, 0] for _ in range(54)] for _ in range(14)] for _ in range(17)]
        value = [2, [0] * 1188, [0] * 270,
                 [[[0, 0] for _ in range(10)] for _ in range(28)], eval_k, eval_a, []]
        phase = [1] + [0] * 11 + [eval_k, eval_a, [0] * 8]
        cls.input = checker.canonical(value)
        cls.phase = checker.canonical(phase) + b"\n"
        cls.words = checker.canonical(checker.proof_words(value)) + b"\n"

    def test_input_accepts_zero_or_one_final_lf_on_either_side(self):
        for lean_suffix in (b"", b"\n"):
            for rust_suffix in (b"", b"\n"):
                with self.subTest(lean=lean_suffix, rust=rust_suffix):
                    checker.compare(self.input + lean_suffix, self.phase, self.words,
                                    self.input + rust_suffix, self.phase)

    def test_input_rejects_other_formatting_even_when_both_files_match(self):
        variants = (self.input + b"\n\n", self.input + b"\r\n", self.input + b" ",
                    b" " + self.input, self.input.replace(b",", b", ", 1))
        for raw in variants:
            with self.subTest(format=raw[:20], suffix=raw[-4:]):
                with self.assertRaisesRegex(ValueError, "noncanonical PiCCS input encoding"):
                    checker.compare(raw, self.phase, self.words, raw, self.phase)

    def test_changed_input_value_is_not_just_a_newline_difference(self):
        changed = self.input.replace(b"[2,[0,", b"[2,[1,", 1) + b"\n"
        with self.assertRaisesRegex(ValueError, "complete PiCCS input bytes differ"):
            checker.compare(self.input, self.phase, self.words, changed, self.phase)

    def test_phase_and_proof_word_encodings_remain_exact(self):
        with self.assertRaisesRegex(ValueError, "complete PiCCS phase bytes differ"):
            checker.compare(self.input, self.phase, self.words, self.input, self.phase[:-1])
        with self.assertRaisesRegex(ValueError, "complete proof-input encoding differs"):
            checker.compare(self.input, self.phase, self.words[:-1], self.input, self.phase)


if __name__ == "__main__":
    unittest.main()
