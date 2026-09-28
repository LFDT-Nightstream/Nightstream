import copy
import tempfile
import unittest
from pathlib import Path

from scripts.lean_graph.metadata import analyze
from scripts.lean_graph.snapshot import file_entry, write_json


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.lean = self.root / "formal/nightstream-fprime"
        (self.lean / "tests").mkdir(parents=True)
        write_json(self.lean / "lake-manifest.json", {"packages": []})
        (self.lean / "lean-toolchain").write_text("preserved-test-toolchain\n")
        self.source = self.lean / "tests/Fixture.lean"
        self.source.write_text("def Meaning := 2\ndef Target : Prop := Meaning = Meaning\n")
        self.runtime = self.root / "runtime"
        (self.runtime / "lib/lean/Init").mkdir(parents=True)
        external = self.runtime / "lib/lean/Init/Prelude.olean"
        external.write_bytes(b"preserved runtime metadata fixture")
        self.manifest = {"schema": 1, "sources": {"lean": {
            "source/formal/nightstream-fprime/tests/Fixture.lean": file_entry(self.source)}}, "inputs": {}}
        self.document = {"schema": 1, "root": "Target", "complete": True,
                         "runtime": str(self.runtime), "nodes": [
            self.node("Target", "theorem", ["Meaning"], ["Helper"], "Meaning = Meaning", "first proof"),
            self.node("Meaning", "definition", ["Nat"], [], "2", None),
            self.node("Helper", "theorem", ["Nat"], [], "2 = 2", "helper proof"),
            {"name": "Nat", "kind": "inductive", "module": "Init.Prelude",
             "origin": str(external), "local": False, "meaning": None, "shape": None,
             "meaning_dependencies": [], "proof_dependencies": [], "proof": None}]}

    def node(self, name, kind, meaning_dependencies, proof_dependencies, meaning, proof):
        return {"name": name, "kind": kind, "module": "tests.Fixture", "origin": str(self.source),
                "local": True, "meaning": meaning, "shape": None,
                "meaning_dependencies": meaning_dependencies,
                "proof_dependencies": proof_dependencies, "proof": proof}

    def result(self, document=None):
        return analyze(document or self.document, self.lean, self.manifest)

    def test_changed_definition_changes_meaning_with_unchanged_target_type(self):
        old = self.result()
        self.document["nodes"][1]["meaning"] = "3"
        new = self.result()
        self.assertTrue(old["complete"] and new["complete"])
        self.assertNotEqual(old["meaning_key"], new["meaning_key"])

    def test_proof_change_preserves_meaning_and_changes_proof_key(self):
        old = self.result()
        self.document["nodes"][0]["proof"] = "a different proof of the same proposition"
        new = self.result()
        self.assertEqual(old["meaning_key"], new["meaning_key"])
        self.assertNotEqual(old["proof_key"], new["proof_key"])
        self.assertNotIn("Helper", old["meaning"])
        self.assertIn("Helper", old["proof"])

    def test_constructor_change_changes_meaning(self):
        self.document["nodes"][1]["kind"] = "inductive"
        self.document["nodes"][1]["shape"] = ["first constructor"]
        old = self.result()
        self.document["nodes"][1]["shape"].append("another constructor")
        self.assertNotEqual(old["meaning_key"], self.result()["meaning_key"])

    def test_unknown_origin_cannot_be_approved_by_namespace(self):
        unknown = self.root / "unapproved/Init/Prelude.olean"
        unknown.parent.mkdir(parents=True)
        unknown.write_bytes(b"unapproved source")
        self.document["nodes"][-1]["origin"] = str(unknown)
        result = self.result()
        self.assertFalse(result["complete"])
        self.assertIsNone(result["meaning_key"])

    def test_missing_dependency_and_unsupported_form_are_incomplete(self):
        for mutate in (lambda doc: doc["nodes"].pop(),
                       lambda doc: doc["nodes"][0].update(kind="unknown")):
            document = copy.deepcopy(self.document)
            mutate(document)
            result = self.result(document)
            self.assertFalse(result["complete"])
            self.assertIsNone(result["meaning_key"])

    def test_recursive_definition_edges_do_not_require_an_acyclic_import_graph(self):
        self.document["nodes"][1]["meaning_dependencies"].append("Meaning")
        self.assertTrue(self.result()["complete"])

    def test_display_preserves_types_without_changing_semantic_keys(self):
        old = self.result()
        self.document["nodes"][0].update(statement="∀ n : Nat, Represents n → n = n",
                                          proposition="", type_expression={"root": 0, "nodes": []})
        new = self.result()
        self.assertEqual(old["meaning_key"], new["meaning_key"])
        self.assertEqual(old["proof_key"], new["proof_key"])
        self.assertIn("Represents", new["proof"]["Target"]["statement"])

    def test_graph_queries_show_dependencies_paths_premises_and_evidence(self):
        from scripts.lean_graph.queries import markdown, query
        self.document["nodes"][0]["statement"] = "∀ n : Nat, Represents n → n = n"
        graph = self.result()
        exports = {"metadata": {"graphs": {"Target": graph}, "record": "/checked/metadata/result.json"}}
        policy = {"gates": {"closure": {"commands": [{"completion": {"closures": {"TargetProp": "Target"}}}]}}}
        status = {"obligations": [{"id": "owner", "target": "TargetProp", "closed": False,
                  "missing": ["review"], "gates": ["closure"]}],
                  "gates": {"closure": {"execution": "passed", "freshness": "current", "checker": "diagnostic",
                                         "record": "/checked/closure/result.json"}}}
        answer = query(exports, "requires", "Target", None, policy, status, self.root)
        self.assertEqual(answer["meaning_dependencies"], ["Meaning"])
        self.assertEqual(answer["proof_dependencies"], ["Helper"])
        self.assertIn("Represents", markdown(answer))
        reverse = query(exports, "used-by", "Helper", None, policy, status, self.root)
        self.assertEqual(reverse["dependents"], [{"declaration": "Target", "edge": "proof"}])
        self.assertEqual(reverse["obligations"][0]["id"], "owner")
        path = query(exports, "path", "Nat", "Target", policy, status, self.root)
        self.assertTrue(path["connected"])
        self.assertEqual(path["path"][0]["dependency"], "Nat")
        self.assertEqual(path["path"][-1]["consumer"], "Target")
