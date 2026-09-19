import copy
import json
import unittest
from pathlib import Path

from protocol_flow import resolve_flow, export_flow
from site_model import counts


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'requirements.json').read_text())
SOURCE = json.loads((ROOT / 'protocol-flow.json').read_text())


class ProtocolFlowTests(unittest.TestCase):
    def test_all_requirements_have_one_owner_and_keep_the_same_totals(self):
        flow = resolve_flow(SOURCE, DATA)
        by_id = {n['id']: n for n in DATA['nodes']}
        records = [by_id[key] for item in flow['items'] for key in item['records']]
        leaves = [n for n in DATA['nodes'] if n['kind'] == 'leaf' and n['origin'] != 'out_of_scope']
        self.assertEqual(len(records), len(leaves))
        for axis in ['proof', 'connection', 'rust']:
            self.assertEqual(counts(records, axis), counts(leaves, axis))
        self.assertEqual(flow['owners']['R.sampler.seed'], 'ccs-rlc')
        self.assertEqual(flow['owners']['D.connection.parent'], 'rlc-dec')
        self.assertEqual(flow['owners']['C.sumcheck.round'], 'rounds')
        self.assertEqual(flow['owners']['N.security.error_budget'], 'fold-security')

    def test_missing_and_duplicate_evidence_are_rejected(self):
        missing = copy.deepcopy(SOURCE)
        missing['items'] = [item for item in missing['items'] if item['id'] != 'rounds']
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            resolve_flow(missing, DATA)
        duplicate = copy.deepcopy(SOURCE)
        duplicate['items'][0]['records'].append('C.sumcheck.round')
        with self.assertRaisesRegex(ValueError, 'counted twice'):
            resolve_flow(duplicate, DATA)

    def test_circuit_boundary_base_bypass_and_selected_assumptions(self):
        flow = resolve_flow(SOURCE, DATA)
        self.assertEqual(len(flow['sections']), 6)
        circuit = set(flow['circuit']['items'])
        self.assertTrue({'circuit-ccs', 'circuit-rlc', 'circuit-dec', 'next-state'} <= circuit)
        self.assertFalse({'next-witness', 'next-commitment', 'repeat', 'terminal'} & circuit)
        self.assertIn({'from': 'next-state', 'to': 'next-witness', 'kind': 'encode',
                       'label': 'Encode after F′ returns'}, flow['edges'])
        self.assertEqual([(e['from'], e['to']) for e in flow['edges'] if e['kind'] == 'feedback'],
                         [('repeat', 'fresh'), ('repeat', 'running')])
        self.assertEqual([(e['from'], e['to']) for e in flow['edges'] if e['kind'] == 'base'],
                         [('initialize', 'next-state')])
        self.assertEqual(flow['owners']['C.connection.actual'], 'circuit-ccs')
        self.assertEqual(flow['owners']['R.connection.challenges'], 'circuit-rlc')
        self.assertEqual(flow['owners']['D.connection.phase'], 'circuit-dec')
        self.assertEqual(flow['owners']['P.delivery.backend'], 'compression')
        markdown = export_flow(flow, DATA)
        self.assertIn('subgraph recursive_circuit', markdown)
        self.assertIn('  next_state --> |"Encode after F′ returns"| next_witness', markdown.splitlines())
        for condition in flow['conditions']:
            self.assertIn('assumptions.md#assumption-' + condition['id'], markdown)
