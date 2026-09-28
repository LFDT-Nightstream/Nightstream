from copy import deepcopy
from pathlib import Path
import re
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

    def test_independent_contract_captures_producers_and_the_feedback_check(self):
        gates = self.policy['obligations']['independent-generation']['gates']
        self.assertEqual(gates, ['independent-coordinator-contract'])
        scope = gate_scope(self.policy, gates)
        for source in ('independent', 'lean', 'rust', 'checker'):
            self.assertIn(source, scope['sources'])
        for root in self.policy['sources']['independent']['roots']:
            self.assertTrue((Path(__file__).resolve().parents[3] / root).is_file())
        patterns = self.policy['gates'][gates[0]]['commands'][0]['completion']['patterns']
        self.assertTrue(any('second_fold_reads_only_first_lean_outputs' in pattern for pattern in patterns))

    def test_complete_norm_registration_uses_current_carrier(self):
        root = Path(__file__).resolve().parents[3]
        source = (root / 'formal/nightstream-fprime/scripts/project_replay_sources.py').read_text()
        geometry = re.search(r'^D, BLOCKS, LOGICAL, CHILDREN, PUBLIC, MATRICES = ([0-9, ]+)$',
                             source, re.MULTILINE)
        self.assertIsNotNone(geometry)
        degree, blocks, _, children, _, _ = map(int, geometry[1].split(','))
        count = (degree * blocks + 3) // 4
        payload = (children + 1) * count
        commands = self.policy['gates']['piccs-norm-prefix-values']['commands']
        complete = [command for command in commands
                    if '{output:norm-prefix-complete}' in command['argv']]
        self.assertEqual(len(complete), 3)
        for command in complete:
            self.assertIn(str(count), command['argv'])
        self.assertTrue(any(f'"payload_bytes":{payload}' in pattern
                            for pattern in complete[0]['completion']['patterns']))
        self.assertTrue(any(f'"compared_bytes":{payload * 16}' in pattern
                            for pattern in complete[1]['completion']['patterns']))

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
        self.assertTrue(any('unit tests cannot close' in reason
                            for reason in outcomes['independent-generation']['missing']))
        self.assertTrue(any('unit tests cannot close' in reason
                            for reason in outcomes['golden-conformance']['missing']))


if __name__ == '__main__':
    unittest.main()
