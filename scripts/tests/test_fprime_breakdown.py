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
        self.facts = fb.lean_facts()
        self.reference = fb.Size(*self.manifest["reference"])

    def test_reference_totals_match_the_manifest_forms(self):
        rows, coordinates = fb.breakdown(self.manifest, fb.Application("reference", self.reference, ()), self.facts)
        geometry = self.manifest["geometry"]
        self.assertEqual(totals(rows), fb.evaluate(geometry["logical_rows"], self.reference))
        width = fb.evaluate(geometry["logical_width"], self.reference)
        self.assertEqual(totals(coordinates) % geometry["ring_degree"], 0)
        self.assertLess(totals(coordinates) - width, geometry["ring_degree"])

    def test_application_numbers_move_only_the_application_section(self):
        larger = fb.Size(self.reference.witness_words + 10, self.reference.local_words + 20, self.reference.rows + 30)
        base_rows, _ = fb.breakdown(self.manifest, fb.Application("a", self.reference, ()), self.facts)
        rows, coordinates = fb.breakdown(self.manifest, fb.Application("a", larger, ()), self.facts)
        self.assertEqual(totals(rows) - totals(base_rows), 30)
        self.assertEqual(sum(value for _, value, _ in application_leaves(coordinates)),
                         self.manifest["geometry"]["field_slot_width"] * (larger.witness_words + larger.local_words))

    def test_parts_must_sum_to_the_application(self):
        half = fb.Size(2, 2740, 2742)
        rows, _ = fb.breakdown(self.manifest, fb.Application("a", self.reference, (("x", half), ("y", half))), self.facts)
        self.assertEqual([value for _, value, _ in application_leaves(rows)], [2742, 2742])
        with self.assertRaises(ValueError):
            fb.breakdown(self.manifest, fb.Application("a", self.reference, (("x", half),)), self.facts)

    def test_a_changed_manifest_is_refused(self):
        renamed = copy.deepcopy(self.manifest)
        renamed["children"][0]["id"] = "something_new"
        with self.assertRaises(fb.ManifestError):
            fb.breakdown(renamed, fb.Application("a", self.reference, ()), self.facts)
        moved = copy.deepcopy(self.manifest)
        moved["assignment_blocks"][0]["slot_count"][0] += 1
        with self.assertRaises(fb.ManifestError):
            fb.breakdown(moved, fb.Application("a", self.reference, ()), self.facts)

    def test_a_changed_lean_count_is_refused(self):
        for key in ("statement_permutations", "eval_a_products", "wide_new_bits", "running_mux_words"):
            facts = dict(self.facts, **{key: self.facts[key] + 1})
            with self.subTest(key=key), self.assertRaises(fb.ManifestError):
                fb.breakdown(self.manifest, fb.Application("a", self.reference, ()), facts)

    def test_sections_come_from_part_names(self):
        parts = (("Memory / sponge", fb.Size(0, 5480, 5000)), ("Memory / words", fb.Size(4, 0, 0)),
                 ("Own rows", fb.Size(0, 0, 484)))
        rows, coordinates = fb.breakdown(self.manifest, fb.Application("App", self.reference, parts), self.facts)
        sections = [section for group, entries in coordinates if group == fb.APPLICATION for section, _ in entries]
        self.assertEqual(sections, ["Memory"], "a part with no coordinates has no coordinate leaf")
        row_sections = [section for group, entries in rows if group == fb.APPLICATION for section, _ in entries]
        self.assertEqual(row_sections, ["Memory", "App"])


if __name__ == "__main__":
    unittest.main()
