"""Fault-injection checks for the protocol-contract data model."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


CONTRACT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRACT_DIR))

import contract_assurance as assurance  # noqa: E402
import contract_checks as checks  # noqa: E402
import contract_migration as migration  # noqa: E402
import contract_model as model  # noqa: E402
import contract_protocol as protocol  # noqa: E402


class ContractModelTests(unittest.TestCase):
    def test_current_model_and_generated_views_are_current(self) -> None:
        current = model.load_model(repository_mode=False)
        self.assertEqual(len(current.requirements), 104)
        self.assertEqual(len(current.protocol["events"]), 12)
        model.check_generated(current)

    def test_generated_contract_has_banner_and_replacement(self) -> None:
        text = (CONTRACT_DIR / "superneo-v1.md").read_text()
        self.assertIn("Generated reading view", text)
        self.assertIn(
            "`NS-PICCS-VARIANT` replaces `SN-PICCS-IDENTITY`", text
        )

    def test_package_manifest_ignores_only_ephemeral_cache_files(self) -> None:
        cache_dir = CONTRACT_DIR / "tests" / "__pycache__"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / "contract-test.pyc"
        cache_file.write_bytes(b"local cache")
        try:
            current = model.load_model(repository_mode=False)
            model.check_generated(current)
            checks.check_package_manifest()
        finally:
            cache_file.unlink(missing_ok=True)
            try:
                cache_dir.rmdir()
            except OSError:
                pass

    def test_sealed_migration_audit_does_not_freeze_live_semantics(self) -> None:
        current = model.load_model(repository_mode=False)
        changed = copy.deepcopy(current)
        changed.rules["NS-ENC-CONTAINER"]["text"] += "\nSynthetic later revision.\n"
        migration.validate_lossless_import(changed)

    def test_duplicate_json_key_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONTRACT_DIR / "tests") as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"id":"A","id":"B"}\n')
            with self.assertRaisesRegex(model.ModelError, "duplicate JSON key"):
                model.load_json(path)

    def test_duplicate_decision_text_fails(self) -> None:
        rows = (
            '{"id":"NSD-A-001","class":"profile","statement":"Same decision.",'
            '"reason":"first","owner":"owner-a","selection_state":"open",'
            '"integration_state":"unreviewed"}\n'
            '{"id":"NSD-B-001","class":"profile","statement":"Same decision.",'
            '"reason":"second","owner":"owner-b","selection_state":"open",'
            '"integration_state":"unreviewed"}\n'
        )
        with tempfile.TemporaryDirectory(dir=CONTRACT_DIR / "tests") as directory:
            path = Path(directory) / "decisions.jsonl"
            path.write_text(rows)
            with self.assertRaisesRegex(model.ModelError, "duplicate decision text"):
                model.parse_decisions(path)

    def test_cycle_detector_returns_the_cycle(self) -> None:
        cycle = model.find_cycle({"A": ["B"], "B": ["C"], "C": ["A"]})
        self.assertIsNotNone(cycle)
        self.assertEqual(cycle[0], cycle[-1])

    def test_release_derivation_has_a_success_path(self) -> None:
        claims = {
            "ROOT": {
                "id": "ROOT",
                "kind": "release",
                "applicability": "required",
                "evidence_state": "complete",
                "depends_on": [],
                "evidence": ["protocol-contract/README.md"],
                "blocker_ids": [],
            }
        }
        contract_hash = "1" * 64
        profile_hash = "2" * 64
        reviews = {
            "REV-ROOT": {
                "id": "REV-ROOT",
                "claim_id": "ROOT",
                "reviewer_role": "protocol-review",
                "reviewer": "test",
                "reviewed_at": "2026-08-03T00:00:00Z",
                "method": "synthetic closure test",
                "conclusion": "accepted",
                "contract_sha256": contract_hash,
                "profile_sha256": profile_hash,
                "evidence_sha256": assurance.claim_evidence_digest(
                    claims["ROOT"], False
                ),
            }
        }
        status = assurance._derive_claim_status(
            claims, {}, {}, reviews, contract_hash, profile_hash, False
        )
        gates, release = assurance._derive_release(
            {
                "root_claim": "ROOT",
                "implementation_ready_gate": "G0",
                "gates": [{"id": "G0", "requires": ["ROOT"]}],
            },
            status,
        )
        self.assertEqual(gates[0]["closure_state"], "closed")
        self.assertTrue(release["eligible"])
        self.assertIsNone(release["next_gate"])

    def test_live_model_can_reach_implementation_ready_without_g2(self) -> None:
        current = model.load_model(repository_mode=False)
        changed = copy.deepcopy(current)
        changed.issues["ISSUE-INDEPENDENT-SEMANTIC-REVIEW"]["state"] = "resolved"
        changed.claims["SOURCE-NORMALIZATION-REVIEW"]["evidence_state"] = "complete"
        claim_ids = {
            claim_id
            for gate in changed.assurance_graph["gates"][:3]
            for claim_id in gate["requires"]
        }
        changed.reviews = {}
        for claim_id in sorted(claim_ids):
            claim = changed.claims[claim_id]
            evidence_hash = assurance.claim_evidence_digest(claim, False)
            self.assertIsNotNone(evidence_hash, claim_id)
            changed.reviews[f"REV-{claim_id}"] = {
                "id": f"REV-{claim_id}",
                "claim_id": claim_id,
                "reviewer_role": "protocol-review",
                "reviewer": "test",
                "reviewed_at": "2026-08-03T00:00:00Z",
                "method": "synthetic implementation-ready review",
                "conclusion": "accepted",
                "contract_sha256": changed.contract_hash,
                "profile_sha256": changed.profile_hash,
                "evidence_sha256": evidence_hash,
            }
        model.validate_model(changed, repository_mode=False)
        status = assurance._derive_claim_status(
            changed.claims,
            changed.issues,
            changed.decisions,
            changed.reviews,
            changed.contract_hash,
            changed.profile_hash,
            False,
        )
        gates, release = assurance._derive_release(changed.assurance_graph, status)
        self.assertTrue(all(row["closure_state"] == "closed" for row in gates[:3]))
        self.assertEqual(gates[3]["closure_state"], "blocked")
        self.assertTrue(release["implementation_ready"])
        self.assertFalse(release["eligible"])

    def test_unresolved_issue_blocks_a_complete_leaf(self) -> None:
        claims = {
            "ROOT": {
                "id": "ROOT",
                "kind": "release",
                "applicability": "required",
                "evidence_state": "complete",
                "depends_on": [],
                "evidence": ["protocol-contract/README.md"],
                "blocker_ids": ["ISSUE-X"],
            }
        }
        issues = {"ISSUE-X": {"id": "ISSUE-X", "state": "open"}}
        status = assurance._derive_claim_status(
            claims, issues, {}, {}, "1" * 64, "2" * 64, False
        )
        self.assertEqual(status["ROOT"]["closure_state"], "open")
        self.assertEqual(status["ROOT"]["blocker_state"], "blocked")

    def test_stale_review_receipt_reopens_claim(self) -> None:
        claim = {
            "id": "ROOT",
            "kind": "source",
            "applicability": "required",
            "evidence_state": "complete",
            "depends_on": [],
            "evidence": ["protocol-contract/README.md"],
            "blocker_ids": [],
        }
        reviews = {
            "REV-ROOT": {
                "id": "REV-ROOT",
                "claim_id": "ROOT",
                "reviewer_role": "protocol-review",
                "reviewer": "test",
                "reviewed_at": "2026-08-03T00:00:00Z",
                "method": "stale receipt test",
                "conclusion": "accepted",
                "contract_sha256": "0" * 64,
                "profile_sha256": "2" * 64,
                "evidence_sha256": assurance.claim_evidence_digest(claim, False),
            }
        }
        status = assurance._derive_claim_status(
            {"ROOT": claim}, {}, {}, reviews, "1" * 64, "2" * 64, False
        )
        self.assertEqual(status["ROOT"]["freshness"], "stale")
        self.assertEqual(status["ROOT"]["closure_state"], "open")

    def test_query_joins_requirement_and_evidence(self) -> None:
        current = model.load_model(repository_mode=False)
        result = model.query(current, "NS-PICCS-NORM-BINDING")
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "requirement")
        self.assertIn("lean", result["evidence"])
        self.assertIn("downstream_dependents", result)

    def test_requirement_cycle_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.requirements["SN-FND-FIELD"]["depends_on"] = [
            "NS-RELEASE-PRODUCTION"
        ]
        with self.assertRaisesRegex(model.ModelError, "requirement dependency cycle"):
            model.validate_model(mutated, repository_mode=False)

    def test_redundant_requirement_edge_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        found = False
        for rule_id, item in mutated.requirements.items():
            direct = set(item["depends_on"])
            for dependency in list(direct):
                ancestors = model.dependency_closure(
                    {
                        key: value["depends_on"]
                        for key, value in mutated.requirements.items()
                    },
                    dependency,
                )
                extra = next((value for value in ancestors if value not in direct), None)
                if extra is not None:
                    item["depends_on"].append(extra)
                    found = True
                    break
            if found:
                break
        self.assertTrue(found)
        with self.assertRaisesRegex(
            model.ModelError, "redundant transitive requirement edge"
        ):
            model.validate_model(mutated, repository_mode=False)

    def test_redundant_claim_edge_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        graph = {key: value["depends_on"] for key, value in mutated.claims.items()}
        found = False
        for claim_id, claim in mutated.claims.items():
            direct = set(claim["depends_on"])
            for dependency in list(direct):
                ancestors = model.dependency_closure(graph, dependency)
                extra = next((value for value in ancestors if value not in direct), None)
                if extra is not None:
                    claim["depends_on"].append(extra)
                    found = True
                    break
            if found:
                break
        self.assertTrue(found)
        with self.assertRaisesRegex(model.ModelError, "redundant transitive claim edge"):
            model.validate_model(mutated, repository_mode=False)

    def test_unknown_review_flag_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.requirements["SN-FND-FIELD"]["review_flags"].append(
            "misspelled-flag"
        )
        with self.assertRaisesRegex(model.ModelError, "unknown review flags"):
            model.validate_model(mutated, repository_mode=False)

    def test_literal_paper_citation_requires_its_source_file(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.requirements["SN-GLOBAL-STRONG-SET"]["source_ids"].remove(
            "SRC-PAPER-12"
        )
        with self.assertRaisesRegex(model.ModelError, "omits a source file"):
            model.validate_model(mutated, repository_mode=False)

    def test_atomic_keyword_limit_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.rules["SN-FND-FIELD"]["text"] += "\n" + " ".join(
            ["It MUST fail."] * 5
        )
        with self.assertRaisesRegex(model.ModelError, "invalid atomicity count"):
            model.validate_model(mutated, repository_mode=False)

    def test_paper_rule_cannot_mix_decision_authority(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.rules["SN-FND-FIELD"]["text"] += "\nDecision: NSD-DOMAIN-001.\n"
        with self.assertRaisesRegex(model.ModelError, "mixes decision authority"):
            model.validate_model(mutated, repository_mode=False)

    def test_duplicate_normative_clause_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        sentence = (
            "### SN-FND-RING — Cyclotomic rings\n\n"
            "For a selected degree-`d` cyclotomic polynomial `Phi`, the rings MUST be\n"
            "`R_F=F[X]/(Phi)` and `R_K=K_ext[X]/(Phi)`."
        )
        mutated.rules["SN-FND-FIELD"]["text"] += f"\n{sentence}\n"
        with self.assertRaisesRegex(model.ModelError, "duplicate normative clause"):
            model.validate_model(mutated, repository_mode=False)

    def test_decision_claim_omission_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.claims["DEC-CARRIER"]["blocker_ids"].remove(
            "NSD-COLUMN-MAP-001"
        )
        with self.assertRaisesRegex(
            model.ModelError, "do not cover every canonical decision"
        ):
            model.validate_model(mutated, repository_mode=False)

    def test_duplicate_decision_claim_coverage_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.claims["DEC-CARRIER"]["blocker_ids"].append("NSD-SPLIT-001")
        with self.assertRaisesRegex(model.ModelError, "decision claim coverage"):
            model.validate_model(mutated, repository_mode=False)

    def test_global_id_collision_fails(self) -> None:
        with self.assertRaisesRegex(checks.ContractError, "global semantic ID collisions"):
            checks.check_global_id_uniqueness(
                {"source": {"COLLISION-ID"}, "protocol-event": {"COLLISION-ID"}}
            )

    def test_unknown_assurance_blocker_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.claims["SOURCE-NORMALIZATION-REVIEW"]["blocker_ids"].append(
            "ISSUE-UNKNOWN"
        )
        with self.assertRaisesRegex(model.ModelError, "unknown claim blockers"):
            assurance.validate_assurance(
                mutated,
                repository_mode=False,
                rule_ids=set(mutated.rules),
                decision_ids=set(mutated.decisions),
            )

    def test_open_decision_cannot_be_fully_integrated(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.decisions["NSD-NORM-BINDING-001"]["selection_state"] = "open"
        mutated.decisions["NSD-NORM-BINDING-001"]["integration_state"] = "complete"
        with self.assertRaisesRegex(model.ModelError, "unselected decision is integrated"):
            model.validate_model(mutated, repository_mode=False)

    def test_profile_hash_input_duplicate_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        first = mutated.bundle["authored"]["profile_hash_inputs"][0]
        mutated.bundle["authored"]["profile_hash_inputs"].append(first)
        with self.assertRaisesRegex(model.ModelError, "duplicate profile hash input paths"):
            model.validate_model(mutated, repository_mode=False)

    def test_event_with_unknown_state_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.protocol["events"][0]["from_state"] = "ST-UNKNOWN"
        with self.assertRaisesRegex(model.ModelError, "event has unknown state"):
            protocol.validate_protocol(mutated)

    def test_event_with_unknown_rejection_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.protocol["events"][0]["reject_conditions"][0] = "REJECT-TYPO"
        with self.assertRaisesRegex(model.ModelError, "unknown rejection code"):
            protocol.validate_protocol(mutated)

    def test_unused_rejection_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        mutated.protocol["rejections"].append(
            {
                "id": "REJECT-UNUSED",
                "description": "Synthetic unused failure.",
                "rule_ids": ["NS-ENC-BASE"],
            }
        )
        with self.assertRaisesRegex(model.ModelError, "registry and event use differ"):
            protocol.validate_protocol(mutated)

    def test_challenge_cannot_precede_transcript_dependency(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        for challenge in mutated.protocol["challenges"]:
            if challenge["id"] == "CH-PIRLC-CANDIDATE":
                challenge["after_events"] = ["EV-FOLD-FINALIZE"]
        with self.assertRaisesRegex(
            model.ModelError, "used before its transcript predecessor"
        ):
            protocol.validate_protocol(mutated)

    def test_repetition_cycle_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        rows = {item["id"]: item for item in mutated.protocol["repetitions"]}
        rows["REP-RHO-SOURCES"]["parent_id"] = "REP-RHO-COEFFICIENTS"
        with self.assertRaisesRegex(model.ModelError, "repetition dependency cycle"):
            protocol.validate_protocol(mutated)

    def test_transcript_schedule_repeat_nesting_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        source_loop = next(
            step
            for step in mutated.protocol["schedule"]["steps"]
            if step["id"] == "LOOP-RHO-SOURCES"
        )
        source_loop["body"][0]["repetition_id"] = "REP-RHO-ATTEMPTS"
        with self.assertRaisesRegex(
            model.ModelError, "transcript repeat nesting differs"
        ):
            protocol.validate_protocol(mutated)

    def test_transcript_schedule_profile_count_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        next(
            step
            for step in mutated.protocol["schedule"]["steps"]
            if step["id"] == "FR-STATEMENT"
        )["base_field_count"] += 1
        with self.assertRaisesRegex(
            model.ModelError, "transcript frame differs from the selected profile"
        ):
            protocol.check_protocol_profile_alignment(mutated, checks.load_config())

    def test_lifecycle_digest_cannot_be_next_fold_authority(self) -> None:
        current = model.load_model(repository_mode=False)
        config = copy.deepcopy(checks.load_config())
        config["lifecycle_profile"]["fold_transcript_digest_next_fold_input"] = True
        with self.assertRaisesRegex(model.ModelError, "digest authority differs"):
            protocol.check_protocol_profile_alignment(current, config)

    def test_exact_repetition_bound_mismatch_fails(self) -> None:
        current = model.load_model(repository_mode=False)
        mutated = copy.deepcopy(current)
        for repetition in mutated.protocol["repetitions"]:
            if repetition["id"] == "REP-PICCS-ROUNDS":
                repetition["minimum"] = 23
        with self.assertRaisesRegex(model.ModelError, "exact repetition has unequal bounds"):
            protocol.validate_protocol(mutated)

    def test_profile_section_count_fault_fails(self) -> None:
        config = copy.deepcopy(checks.load_config())
        config["container_sections"]["proof_section_field_counts"][0] += 2
        with self.assertRaisesRegex(checks.ContractError, "proof field census differs"):
            checks.check_profile_consistency(config)

    def test_profile_container_byte_count_fault_fails(self) -> None:
        config = copy.deepcopy(checks.load_config())
        config["container_sections"]["statement_total_bytes"] += 8
        with self.assertRaisesRegex(checks.ContractError, "statement byte census differs"):
            checks.check_profile_consistency(config)

    def test_commitment_orientation_fault_fails(self) -> None:
        config = copy.deepcopy(checks.load_config())
        config["commitment_profile"]["orientation"] = "right-vector-matrix"
        with self.assertRaisesRegex(checks.ContractError, "orientation"):
            checks.check_profile_consistency(config)

    def test_carried_source_mapping_fault_fails(self) -> None:
        config = copy.deepcopy(checks.load_config())
        config["nightstream_candidate"]["carried_local_to_global_offset"] = 0
        with self.assertRaisesRegex(checks.ContractError, "carried source mapping differs"):
            checks.check_security_census(config)

    def test_oracle_query_limit_covers_protocol_schedule(self) -> None:
        config = copy.deepcopy(checks.load_config())
        config["security_accounting"]["maximum_adaptive_oracle_queries"] = 128
        config["security_policy"]["maximum_adaptive_oracle_queries"] = 128
        with self.assertRaisesRegex(
            checks.ContractError, "oracle-query limit excludes"
        ):
            checks.check_security_census(config)

    def test_ajtai_setup_stream_fault_fails(self) -> None:
        config = copy.deepcopy(checks.load_config())
        config["ajtai_setup_v1"]["test_first_64_u32"][0] ^= 1
        with self.assertRaisesRegex(checks.ContractError, "initial test vector differs"):
            checks.check_profile_consistency(config)

    def test_structure_encoding_fault_fails(self) -> None:
        config = copy.deepcopy(checks.load_config())
        config["structure_encoding_v1"]["identity_variant_code"] = 2
        with self.assertRaisesRegex(checks.ContractError, "identity variant differs"):
            checks.check_profile_consistency(config)

    def test_verifier_key_digest_layout_fault_fails(self) -> None:
        config = copy.deepcopy(checks.load_config())
        config["verifier_key_digest_v1"]["preimage_layout"].pop()
        with self.assertRaisesRegex(checks.ContractError, "preimage layout differs"):
            checks.check_profile_consistency(config)

    def test_public_image_layout_fault_fails(self) -> None:
        config = copy.deepcopy(checks.load_config())
        config["public_image_v1"]["public_field_order"][0] = "profile_version_tag"
        with self.assertRaisesRegex(checks.ContractError, "public-image output layout differs"):
            checks.check_profile_consistency(config)

    def test_duplicate_transcript_tag_fails(self) -> None:
        config = copy.deepcopy(checks.load_config())
        config["transcript_tags"]["piccs_gamma"] = config["transcript_tags"][
            "piccs_alpha"
        ]
        with self.assertRaisesRegex(checks.ContractError, "duplicate transcript tag"):
            checks.check_profile_consistency(config)

    def test_poseidon2_test_vector_fault_fails(self) -> None:
        config = copy.deepcopy(checks.load_config())
        config["poseidon2_goldilocks_v1"]["test_zero_output"][0] ^= 1
        with self.assertRaisesRegex(checks.ContractError, "zero test vector differs"):
            checks.check_profile_consistency(config)


if __name__ == "__main__":
    unittest.main()
