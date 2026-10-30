import copy
import json
import unittest

from scripts import fprime_breakdown as fb


def totals(groups):
    return sum(value for _, sections in groups for _, leaves in sections for _, value, _ in leaves)


def application_leaves(groups):
    return [leaf for group, sections in groups if group == fb.APPLICATION for _, leaves in sections for leaf in leaves]


class BreakdownTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(fb.MANIFEST.read_text())
        self.reference = fb.Size(*self.manifest["reference"])

    def test_reference_totals_match_the_manifest_forms(self):
        rows, coordinates = fb.breakdown(self.manifest, fb.Application("reference", self.reference, ()))
        geometry = self.manifest["geometry"]
        self.assertEqual(totals(rows), fb.evaluate(geometry["logical_rows"], self.reference))
        width = fb.evaluate(geometry["logical_width"], self.reference)
        self.assertEqual(totals(coordinates) % geometry["ring_degree"], 0)
        self.assertLess(totals(coordinates) - width, geometry["ring_degree"])

    def test_application_numbers_move_only_the_application_section(self):
        larger = fb.Size(self.reference.witness_words + 10, self.reference.local_words + 20, self.reference.rows + 30)
        base_rows, _ = fb.breakdown(self.manifest, fb.Application("a", self.reference, ()))
        rows, coordinates = fb.breakdown(self.manifest, fb.Application("a", larger, ()))
        self.assertEqual(totals(rows) - totals(base_rows), 30)
        self.assertEqual(sum(value for _, value, _ in application_leaves(coordinates)),
                         self.manifest["geometry"]["field_slot_width"] * (larger.witness_words + larger.local_words))

    def test_parts_must_sum_to_the_application(self):
        half = fb.Size(2, 2740, 2742)
        rows, _ = fb.breakdown(self.manifest, fb.Application("a", self.reference, (("x", half), ("y", half))))
        self.assertEqual([value for _, value, _ in application_leaves(rows)], [2742, 2742])
        with self.assertRaises(ValueError):
            fb.breakdown(self.manifest, fb.Application("a", self.reference, (("x", half),)))

    def test_a_changed_manifest_is_refused(self):
        renamed = copy.deepcopy(self.manifest)
        renamed["children"][0]["id"] = "something_new"
        with self.assertRaises(fb.ManifestError):
            fb.breakdown(renamed, fb.Application("a", self.reference, ()))
        moved = copy.deepcopy(self.manifest)
        moved["assignment_blocks"][0]["slot_count"][0] += 1
        with self.assertRaises(fb.ManifestError):
            fb.breakdown(moved, fb.Application("a", self.reference, ()))


if __name__ == "__main__":
    unittest.main()
