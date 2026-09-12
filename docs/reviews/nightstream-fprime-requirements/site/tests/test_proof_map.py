import copy
import json
import unittest
from pathlib import Path

from proof_map import resolve_map, export_map

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'requirements.json').read_text())
SOURCE = json.loads((ROOT / 'proof-map.json').read_text())


class ProofMapTests(unittest.TestCase):
    def test_every_proved_connection_resolves_to_current_evidence(self):
        result = resolve_map(SOURCE, DATA)
        for edge in result['edges']:
            if edge['kind'] in ['uses', 'supplies']:
                self.assertTrue(edge['code'])
                self.assertTrue(all(ref['path'].endswith('.lean') for ref in edge['code']))
        text = export_map(result, DATA)
        self.assertIn('proof-map:deployed', text)
        self.assertIn('history_probability_bound', text)
        self.assertEqual(sum(line.startswith('| ') for line in text.splitlines()), len(result['edges']) + 2)
        nodes = {n['id']: n for n in result['nodes']}
        self.assertEqual(nodes['deployed']['kind'], 'open')
        self.assertEqual(nodes['deployed']['statuses']['connection'], 'open')

    def test_missing_or_misnamed_evidence_fails(self):
        source = copy.deepcopy(SOURCE)
        source['edges'][0]['refs'] = []
        with self.assertRaisesRegex(ValueError, 'Proved connection needs evidence'):
            resolve_map(source, DATA)
        source = copy.deepcopy(SOURCE)
        source['edges'][0]['refs'][0]['symbol'] = 'missing_theorem'
        with self.assertRaisesRegex(ValueError, 'must resolve once'):
            resolve_map(source, DATA)

    def test_layers_cannot_hide_or_duplicate_a_result(self):
        resolve_map(SOURCE, DATA)
        for replacement in [[], ['deployed'], ['unknown']]:
            source = copy.deepcopy(SOURCE)
            source['layers'][1]['rows'][0] = replacement
            with self.assertRaisesRegex(ValueError, 'every node exactly once'):
                resolve_map(source, DATA)


if __name__ == '__main__':
    unittest.main()
