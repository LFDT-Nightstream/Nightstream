import copy
import importlib.util
import json
from pathlib import Path
import unittest

project = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location('identity', project / 'scripts/check_identity.py')
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.expected = json.loads((project / 'artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1-binding-v1.json').read_text())
        self.rust = (project.parents[1] / 'crates/nightstream-fprime/src/identity.rs').read_text()

    def test_committed_authorities_agree(self):
        identity.check(self.expected, self.expected, self.rust)

    def test_changed_preimage_is_rejected_even_with_unchanged_digests(self):
        changed = copy.deepcopy(self.expected)
        changed[3][0] += 1
        with self.assertRaisesRegex(ValueError, 'differs from the committed fixture'):
            identity.check(changed, self.expected, self.rust)

    def test_a_matching_fixture_cannot_override_rust_pins(self):
        changed = copy.deepcopy(self.expected)
        changed[1][0] += 1
        with self.assertRaisesRegex(ValueError, 'differs from Rust pin'):
            identity.check(changed, changed, self.rust)

    def test_boolean_schema_is_rejected(self):
        changed = copy.deepcopy(self.expected)
        changed[0] = True
        with self.assertRaisesRegex(ValueError, 'schema-1'):
            identity.check(changed, self.expected, self.rust)

    def test_commented_pin_cannot_hide_a_changed_live_definition(self):
        name = identity.PINS[1]
        old = self.rust[self.rust.index('pub const ' + name):]
        old = old[:old.index('];') + 2]
        changed = self.rust.replace(old, '/* ' + old + ' */\npub const ' + name + ': [u64; 4] = [0; 4];')
        with self.assertRaisesRegex(ValueError, 'Expected one literal Rust pin'):
            identity.check(self.expected, self.expected, changed)

    def test_strings_and_nested_comments_cannot_supply_pins(self):
        decoys = '/* outer /* pub const BAD = [1]; */ outer */\nr##"a " pub const BAD = [2];"##\n'
        identity.check(self.expected, self.expected, decoys + self.rust)


if __name__ == '__main__':
    unittest.main()
