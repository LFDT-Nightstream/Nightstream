import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from reference_check import check_references, publication_record
from site_model import counts, error_scenario, validate_data


SITE = Path(__file__).resolve().parents[1]
DATA = json.loads((SITE / 'requirements.json').read_text())


class AssuranceTests(unittest.TestCase):
    def test_counts_preserve_every_in_scope_leaf(self):
        leaves = [n for n in DATA['nodes'] if n['kind'] == 'leaf']
        included = [n for n in leaves if n['origin'] != 'out_of_scope']
        for axis in ['proof', 'connection', 'rust']:
            self.assertEqual(sum(counts(leaves, axis).values()), len(included))
        self.assertEqual(counts(leaves, 'proof')['assumed'], sum(n['proof'] == 'assumption' for n in included))
        self.assertEqual(counts(leaves, 'rust')['tested'], sum(n['rust'] == 'tested_scoped' for n in included))

    def test_new_ledger_and_readiness_links_must_resolve(self):
        validate_data(DATA)
        for target, message in [('records', 'Unknown assumption record'), ('depends_on', 'Unknown assumption dependency')]:
            changed = copy.deepcopy(DATA)
            changed['assumptions'][0][target] = ['missing']
            with self.assertRaisesRegex(ValueError, message):
                validate_data(changed)
        changed = copy.deepcopy(DATA)
        changed['assumptions'] = []
        with self.assertRaisesRegex(ValueError, 'Assumption missing from ledger|Unknown assumption'):
            validate_data(changed)
        changed = copy.deepcopy(DATA)
        changed['readiness'][0]['items'][0]['records'] = ['missing']
        with self.assertRaisesRegex(ValueError, 'Unknown readiness record'):
            validate_data(changed)

    def test_budget_is_conditional_and_uses_exact_field_size(self):
        budget = DATA['error_budget']
        one = error_scenario(budget, '1')
        many = error_scenario(budget, '100000000')
        self.assertEqual(one['numerator'], '13257')
        self.assertEqual(one['denominator'], str(18446744069414584321 ** 2))
        self.assertAlmostEqual(many['bound_bits'], 87.7301085, places=6)
        self.assertEqual(error_scenario(budget, str(10 ** 100))['bound'], 1)
        for invalid in ['0', '-1', '1.5', '', 1.5, '1e8']:
            with self.assertRaises(ValueError):
                error_scenario(budget, invalid)
        self.assertIn('not a full verifier', budget['not_a_total_bound'])
        self.assertTrue(all(p['value'] is None for p in budget['deployment_parameters']))

    def test_full_conformance_and_binding_keep_their_evidence_scope(self):
        nodes = {n['id']: n for n in DATA['nodes']}
        self.assertEqual(nodes['N.conformance.chain']['rust'], 'tested_scoped')
        self.assertEqual(nodes['P.assurance.full_native_chain']['connection'], 'open')
        self.assertEqual(nodes['N.security.binding']['connection'], 'partial')
        self.assertEqual(nodes['N.security.binding']['proof'], 'assumption')
        for node_id in ['N.native.order', 'N.native.openings', 'N.native.output']:
            self.assertEqual(nodes[node_id]['proof'], 'not_required')

    def test_publish_requires_exact_committed_inputs(self):
        root = SITE
        commit = 'a' * 40

        def committed_git(_root, *args):
            if args == ('rev-parse', '--show-toplevel'):
                return str(root).encode()
            if args == ('rev-parse', 'HEAD'):
                return commit.encode()
            return (SITE / args[1].split(':', 1)[1]).read_bytes()

        with patch('reference_check.git', side_effect=committed_git):
            self.assertEqual(publication_record(SITE, True)['map_commit'], commit)

        def dirty_git(root, *args):
            if args == ('show', commit + ':requirements.json'):
                return b'{"old":true}'
            return committed_git(root, *args)

        with patch('reference_check.git', side_effect=dirty_git):
            self.assertIsNone(publication_record(SITE)['map_commit'])
            with self.assertRaisesRegex(ValueError, 'Commit the exact site inputs'):
                publication_record(SITE, True)

    def test_reference_failures_block_publication(self):
        sample = {'provenance': {'code_commit': 'a' * 40}, 'nodes': [
            {'id': 'test', 'code': [{'path': 'test.lean', 'line': 2}], 'paper': []}]}
        with patch('reference_check.git', return_value=b'theorem named : True := trivial\n'):
            with self.assertRaisesRegex(ValueError, 'outside the source'):
                check_references(sample, SITE, None)
            sample['nodes'][0]['code'][0]['line'] = 1
            sample['error_budget'] = {'source_hashes': {'test.lean': 'wrong'}}
            with self.assertRaisesRegex(ValueError, 'Error-budget source changed'):
                check_references(sample, SITE, None)


if __name__ == '__main__':
    unittest.main()
