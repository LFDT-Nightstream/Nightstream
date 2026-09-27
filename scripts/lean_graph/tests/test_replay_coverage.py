from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.lean_graph import records
from scripts.lean_graph.policy import gate_order, gate_scope, load_policy, validate
from scripts.lean_graph.snapshot import EvidenceError


class ReplayCoverageTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()

    def test_each_retained_kernel_has_a_checked_closure_registration(self):
        targets = {
            'pirlc-witness-kernel': 'PiRLCWitnessReplay',
            'pidec-witness-kernel': 'PiDECWitnessReplay',
            'pidec-commitment-kernel': 'PiDECCommitmentReplay',
            'pidec-evaluation-kernel': 'PiDECChildEvaluationReplay',
            'fresh-witness-kernel': 'FreshWitnessKernels',
            'recursive-loop-kernel': 'CheckedRecursiveReplay',
        }
        for name, suffix in targets.items():
            with self.subTest(kernel=name):
                target = 'LeanGraph.Targets.' + suffix
                obligation = self.policy['obligations'][name]
                self.assertEqual(obligation['target'], target)
                self.assertEqual(obligation['tier'], 'Compiler')
                self.assertIn('decomposition', obligation['reviews'])
                closures = {}
                for gate in gate_order(self.policy, obligation['gates']):
                    for command in self.policy['gates'][gate]['commands']:
                        closures.update(command['completion'].get('closures', {}))
                self.assertIn(target, closures)
                self.assertIn('independent-generation', obligation['gap'])

    def test_policy_cannot_declare_a_closed_status(self):
        policy = deepcopy(self.policy)
        policy['obligations']['independent-generation']['status'] = 'Conformance-closed'
        with self.assertRaisesRegex(EvidenceError, 'status is derived'):
            validate(policy)
        policy = deepcopy(self.policy)
        policy['schema'] = 1
        with self.assertRaisesRegex(EvidenceError, 'unsupported obligation-map schema'):
            validate(policy)

    def test_golden_contract_captures_its_actual_coordinators(self):
        gates = self.policy['obligations']['golden-conformance']['gates']
        scope = gate_scope(self.policy, gates)
        self.assertIn('golden', scope['sources'])
        self.assertIn('rust', scope['sources'])
        roots = self.policy['sources']['golden']['roots']
        self.assertIn('scripts/golden_conformance_ci.py', roots)
        self.assertIn('scripts/GOLDEN_CONFORMANCE.md', roots)
        for root in roots:
            self.assertTrue((Path(__file__).resolve().parents[3] / root).is_file())
        self.assertIn('does not independently generate',
                      self.policy['obligations']['golden-conformance']['gap'])

    def test_passed_contract_checks_cannot_close_unexecuted_generation(self):
        policy = deepcopy(self.policy)
        names = ['golden-conformance', 'independent-generation']
        policy['obligations'] = {name: policy['obligations'][name] for name in names}
        passed = {name: {'pass': True, 'completed': True,
                        'shown': {'outcome': 'pass', 'trusted': True},
                        'freshness': 'current', 'basis': 'source',
                        'prerequisites': [], 'missing_inputs': []}
                  for name in policy['gates']}
        with patch.object(records, 'read_runs', return_value=([], [])), \
             patch.object(records, 'gate_results', return_value=(passed, [])), \
             patch.object(records, 'review_results',
                          return_value={name: True for name in policy['reviews']}), \
             patch.object(records, 'decomposition_results', return_value={}):
            result = records.report(policy, {'sources': {}, 'inputs': {}},
                                    Path('/unused'), object())
        outcomes = {item['id']: item for item in result['obligations']}
        for name in names:
            self.assertFalse(outcomes[name]['closed'])
            self.assertEqual(outcomes[name]['status'], 'Open')
            self.assertEqual(outcomes[name]['tier'], 'Conformance')
            self.assertTrue(policy['obligations'][name]['open_requirements'])
        self.assertIn('implemented closing gate',
                      outcomes['independent-generation']['missing'])
        self.assertTrue(any('unit tests cannot close' in reason
                            for reason in outcomes['golden-conformance']['missing']))


if __name__ == '__main__':
    unittest.main()
