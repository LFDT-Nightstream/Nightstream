import contextlib
import copy
import io

from scripts.lean_graph.checkpoint import checkpoint
from scripts.lean_graph.evidence import main
from scripts.lean_graph.reviews import (create_request, decomposition_results, record_review,
                                       validate_decision, validate_proposal)
from scripts.lean_graph.snapshot import EvidenceError, read_json, write_json
from scripts.lean_graph.tests.test_evidence import EvidenceFixture


class ReviewTests(EvidenceFixture):
    def setUp(self):
        super().setUp()
        self.policy["reviews"]["decomposition"] = {"scope": "Exact statement, premises, argument, and parent use."}
        self.policy["obligations"]["compiler"]["reviews"].append("decomposition")
        self.authorize()
        _, self.manifest, self.directory = self.snapshot()
        self.proposal = {"author": "worker", "statement": "RowsHold a implies Target a",
            "premises": ["The selected rows hold for the actual assignment."],
            "argument": "Decode the selected columns, derive the leaf contract, then apply the parent theorem.",
            "parent_use": "Supply the required decoded contract to Test.Target.",
            "dependencies": ["Test.decode", "Test.leaf", "Test.parent"]}
        self.request = create_request("compiler", self.proposal, self.policy, self.manifest, self.store)

    def decision(self, outcome="pass", at="2026-09-06T00:00:00+00:00"):
        record = copy.deepcopy(self.request["response_template"])
        record.update(reviewer="independent reviewer", reviewed_at=at, outcome=outcome)
        for name, assessment in record["assessments"].items():
            assessment.update(outcome="pass", reason="The reviewed argument establishes " + name + ".")
        if outcome == "fail":
            record["assessments"]["premises"] = {"outcome": "fail", "reason": "A required value equality is assumed."}
        return record

    def import_decision(self, record, approved=True):
        path = self.root / "response.json"
        write_json(path, self.authority.sign(record) if approved else record)
        return record_review(self.request["request"], path, self.store,
                             self.authority if approved else None)

    def test_request_preserves_exact_contract_and_resubmission_reuses_it(self):
        second = create_request("compiler", self.proposal, self.policy, self.manifest, self.store)
        self.assertEqual(self.request, second)
        self.assertEqual(self.request["content"]["target"], "Test.Target")
        self.assertEqual(self.request["content"]["proposal"], self.proposal)
        self.assertEqual(list((self.store / "review-requests").iterdir()), [self.store / "review-requests" /
                         (self.request["request"] + ".json")])

    def test_incomplete_proposal_and_missing_target_are_rejected(self):
        for field in ("author", "statement", "premises", "argument", "parent_use", "dependencies"):
            changed = copy.deepcopy(self.proposal)
            changed.pop(field)
            with self.subTest(field=field), self.assertRaises(EvidenceError):
                validate_proposal(changed)
        self.policy["obligations"]["compiler"]["target"] = None
        with self.assertRaisesRegex(EvidenceError, "exact Lean target"):
            create_request("compiler", self.proposal, self.policy, self.manifest, self.store)

    def test_approved_checkpoint_requires_review_and_diagnostics_cannot_grant_closure(self):
        with self.assertRaisesRegex(EvidenceError, "decomposition review is missing"):
            checkpoint("compiler", self.policy, self.manifest, self.directory, self.store, self.authority)
        self.assertFalse((self.store / "runs").exists())
        with contextlib.redirect_stderr(io.StringIO()):
            result = checkpoint("compiler", self.policy, self.manifest, self.directory, self.store)
        self.assertEqual(result["execution"], "passed")
        self.import_decision(self.decision(), approved=False)
        diagnostic = decomposition_results(self.policy, self.manifest, self.store)["compiler"]
        self.assertEqual(diagnostic["state"], "passed")
        self.assertFalse(diagnostic["accepted"])
        self.assertFalse(self.status(self.manifest)["statuses"]["Compiler-closed"])

    def test_valid_review_still_needs_lean_gate_and_other_reviews(self):
        self.import_decision(self.decision())
        self.assertFalse(self.status(self.manifest)["statuses"]["Compiler-closed"])
        self.run_validity()
        self.assertFalse(self.status(self.manifest)["statuses"]["Compiler-closed"])
        self.review(self.manifest)
        self.assertTrue(self.status(self.manifest)["statuses"]["Compiler-closed"])

    def test_changed_bindings_self_review_and_unfinished_assessments_are_rejected(self):
        original = self.decision()
        for field in ("target", "request", "snapshot", "policy", "checker", "scope", "obligation"):
            changed = {**original, field: "different"}
            with self.subTest(field=field), self.assertRaises(EvidenceError):
                validate_decision(changed, self.request["content"], self.request["request"])
        for altered in ({**original, "reviewer": " WORKER "},
                        {**original, "reviewed_at": "2026-09-06T00:00:00"},
                        {**original, "assessments": {}},
                        {**original, "assessments": self.request["response_template"]["assessments"]}):
            with self.assertRaises(EvidenceError):
                validate_decision(altered, self.request["content"], self.request["request"])
        changed = self.decision("fail")
        changed["outcome"] = "pass"
        with self.assertRaisesRegex(EvidenceError, "disagrees"):
            self.import_decision(changed)

    def test_importer_cannot_sign_or_accept_an_untrusted_review(self):
        path = self.root / "unsigned.json"
        write_json(path, {"record": self.decision()})
        with self.assertRaisesRegex(EvidenceError, "untrusted"):
            record_review(self.request["request"], path, self.store, self.authority)

    def test_later_failure_revokes_approval_and_blocks_checkpoint(self):
        self.import_decision(self.decision())
        self.run_validity()
        self.review(self.manifest)
        self.assertTrue(self.status(self.manifest)["statuses"]["Compiler-closed"])
        self.import_decision(self.decision("fail", at="2026-09-06T00:01:00+00:00"))
        state = self.status(self.manifest)
        self.assertEqual(state["obligations"][0]["reviews"]["decomposition"], "failed")
        self.assertFalse(state["statuses"]["Compiler-closed"])
        with self.assertRaisesRegex(EvidenceError, "decomposition review is failed"):
            checkpoint("compiler", self.policy, self.manifest, self.directory, self.store, self.authority)

    def test_changed_source_is_stale_and_the_old_snapshot_remains_valid(self):
        self.import_decision(self.decision())
        (self.source / "src/Meaning.lean").write_text("def Meaning := 3\n")
        _, changed, _ = self.snapshot()
        current = decomposition_results(self.policy, changed, self.store, self.authority)["compiler"]
        self.assertEqual(current["state"], "stale")
        self.assertFalse(current["accepted"])
        self.assertTrue(decomposition_results(self.policy, self.manifest, self.store, self.authority)
                        ["compiler"]["accepted"])

    def test_request_tampering_invalidates_its_review(self):
        self.import_decision(self.decision())
        path = self.store / "review-requests" / (self.request["request"] + ".json")
        changed = read_json(path)
        changed["proposal"]["premises"] = ["Represents a"]
        write_json(path, changed)
        state = decomposition_results(self.policy, self.manifest, self.store, self.authority)["compiler"]
        self.assertFalse(state["accepted"])
        self.assertTrue(state["reasons"])

    def test_cli_creates_request_and_imports_only_a_diagnostic_response(self):
        policy, inputs, proposal, response = [self.root / (name + ".json")
                                              for name in ("policy", "inputs", "proposal", "response")]
        for path, value in ((policy, self.policy), (inputs, self.inputs), (proposal, self.proposal),
                            (response, self.decision())):
            write_json(path, value)
        base = ["--source", str(self.source), "--store", str(self.store), "--policy", str(policy)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(base + ["--inputs", str(inputs), "review-request", "compiler",
                                          "--proposal", str(proposal)]), 0)
            self.assertEqual(main(base + ["record-review", self.request["request"], str(response)]), 0)
        self.assertFalse(decomposition_results(self.policy, self.manifest, self.store)["compiler"]["accepted"])
