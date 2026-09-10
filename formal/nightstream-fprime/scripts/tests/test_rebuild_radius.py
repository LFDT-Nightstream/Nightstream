import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('radius', Path(__file__).parents[1] / 'rebuild_radius.py')
radius = importlib.util.module_from_spec(spec)
spec.loader.exec_module(radius)


class RadiusTests(unittest.TestCase):
    def test_comments_and_strings_cannot_add_edges(self):
        source = '''/- import False.One /- import False.Two -/ -/
public import Actual.One Actual.Two -- import False.Three
def text := "ignore\nimport False.Four\n"
meta import Actual.Three
'''
        self.assertEqual(radius.imports(source), ['Actual.One', 'Actual.Two', 'Actual.Three'])

    def test_reachability_counts_shared_dependents_once(self):
        edges = {'leaf': set(), 'a': {'leaf'}, 'b': {'leaf'}, 'top': {'a', 'b'}, 'other': set()}
        self.assertEqual(radius.dependents(edges, 'leaf'), ['a', 'b', 'top'])
        self.assertEqual(radius.dependents(edges, 'other'), [])
        with self.assertRaisesRegex(ValueError, 'Unknown module'):
            radius.dependents(edges, 'missing')


if __name__ == '__main__':
    unittest.main()
