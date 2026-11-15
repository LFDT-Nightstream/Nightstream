import importlib.util
from pathlib import Path
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('assumptions', PACKAGE / 'scripts' / 'check_assumptions.py')
assumptions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assumptions)

PAGE = '''# Security model

## Assumptions

<!-- assumptions:begin -->
- `P2-PERM`: the Poseidon2 permutation is a uniform permutation.
- `SPONGE`: the sponge theorem, as a random oracle on padded inputs.
<!-- assumptions:end -->

## What you trust

1. **The hash [P2-PERM] [SPONGE].** Collisions follow from the list.
2. **Gap: Rust and Lean agree on recorded inputs only.** Tests check fixed inputs.

## Out of scope

- Quantum adversaries.
'''


class AssumptionGateTests(unittest.TestCase):
    def test_the_security_model_passes(self):
        self.assertEqual(assumptions.check((PACKAGE / 'SECURITY_MODEL.md').read_text()), [])

    def test_a_page_with_listed_assumptions_passes(self):
        self.assertEqual(assumptions.check(PAGE), [])

    def test_an_assumption_off_the_list_fails(self):
        page = PAGE.replace('<!-- assumptions:end -->',
                            '- `P2-HASH`: the Poseidon2 hash is collision free.\n<!-- assumptions:end -->')
        self.assertIn('not on the allowed list', '\n'.join(assumptions.check(page)))

    def test_a_derived_random_oracle_outside_the_list_fails(self):
        page = PAGE.replace('Collisions follow from the list.',
                            'The transcript is a random oracle.')
        self.assertIn('random oracle on a derived object', '\n'.join(assumptions.check(page)))

    def test_collision_resistance_outside_the_list_fails(self):
        page = PAGE.replace('Quantum adversaries.', 'The state hash needs Poseidon2\ncollision resistance.')
        self.assertIn('collision resistance', '\n'.join(assumptions.check(page)))

    def test_a_citation_of_an_unlisted_id_fails(self):
        page = PAGE.replace('[SPONGE].**', '[SPONGE] [ETA-RO].**')
        self.assertIn('[ETA-RO] is not an assumption', '\n'.join(assumptions.check(page)))

    def test_a_trust_item_without_an_assumption_fails(self):
        page = PAGE.replace('**Gap: Rust and Lean', '**Rust and Lean')
        self.assertIn('cites no assumption', '\n'.join(assumptions.check(page)))

    def test_a_missing_list_fails(self):
        page = PAGE.replace('<!-- assumptions:begin -->\n', '')
        self.assertIn('needs one list', '\n'.join(assumptions.check(page)))


if __name__ == '__main__':
    unittest.main()
