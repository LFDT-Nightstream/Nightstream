import importlib.util
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('premises', PACKAGE / 'scripts' / 'check_premises.py')
premises = importlib.util.module_from_spec(spec)
spec.loader.exec_module(premises)

SPEC = ('### Algorithm 4. Fold verifiers\n'
        '`Claims.Fold.one` (`0123456789abcdef`), `Claims.Fold.two` (`fedcba9876543210`)\n')
ONE = 'Claims.Fold.one\taccepted\tEq\n'
TWO = 'Claims.Fold.two\taccepted\tFalse\n'


def page(lines):
    return (f'{premises.BEGIN}\n| Premise | Head | Lines | Class | Evidence |\n'
            '| --- | --- | --- | --- | --- |\n'
            f'| `accepted` | `*` | {lines} | game | `SPEC.md` Algorithm 4 |\n{premises.END}\n')


def errors(model, lines, spec_text=SPEC):
    context = {'assumptions': set(), 'tests': set(), 'model': model, 'spec': spec_text}
    return premises.check(model, lines, context)[0]


class PremiseGateTests(unittest.TestCase):
    def test_a_row_with_its_line_count_passes(self):
        self.assertEqual(errors(page(1), ONE), [])
        self.assertEqual(errors(page(2), ONE + TWO), [])

    def test_a_new_hypothesis_with_the_name_of_a_row_fails(self):
        found = errors(page(1), ONE + TWO)
        self.assertEqual(len(found), 1)
        self.assertIn('gives 1 lines, but it covers 2', found[0])

    def test_a_row_without_a_line_count_fails(self):
        found = errors(page(1).replace('| 1 | game', '| game'), ONE)
        self.assertEqual(len(found), 1)
        self.assertIn('has 0 rows in the premise table', found[0])

    def test_a_claim_without_a_pin_fails(self):
        found = errors(page(1), ONE, SPEC.replace(' (`0123456789abcdef`)', ''))
        self.assertEqual(found, ['`Claims.Fold.one` has no pin in SPEC.md'])


if __name__ == '__main__':
    unittest.main()
