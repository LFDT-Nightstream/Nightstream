#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use bellpepper_core::test_cs::TestConstraintSystem;
use bellpepper_core::ConstraintSystem;
use neo_fold_next::nightstream::rv64im::audit::{
    circuit_derive_phase0_point, circuit_enforce_phase0_commitment_root_and_opened_object_digest,
    circuit_enforce_phase0_payload_eq, circuit_enforce_phase0_point_eq,
    circuit_evaluate_phase0_payload_from_packed_rows,
};
use neo_fold_next::nightstream::rv64im::debug_check_rv64im_side_spartan_circuit;
use neo_fold_next::rv64im::main_relation_spartan::debug_check_rv64im_spartan2_decider_circuit;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, prove_rv64im_public_proof_and_published_seam_with_perf, FamilyEvalSchemaId,
    Rv64imProofInput,
};
use neo_transcript::{Poseidon2Transcript, Transcript};
use spartan2::provider::goldi::F as SpartanF;

const RV64IM_N2_TOTAL_CONSTRAINT_BUDGET: usize = 3_072;
const RV64IM_N2_REPLAY_CONSTRAINT_BUDGET: usize = 1_024;
const RV64IM_N2_CARRIER_CONSTRAINT_BUDGET: usize = 2_048;
const RV64IM_N2_SIDE_CONSTRAINT_BUDGET: usize = 2_048;

fn alloc_digest_vars(
    cs: &mut TestConstraintSystem<SpartanF>,
    digest: [u8; 32],
    label: &str,
) -> [bellpepper_core::num::AllocatedNum<SpartanF>; 4] {
    let values = [
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[0..8].try_into().expect("digest limb 0"))),
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[8..16].try_into().expect("digest limb 1"))),
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[16..24].try_into().expect("digest limb 2"))),
        SpartanF::from_canonical_u64(u64::from_le_bytes(digest[24..32].try_into().expect("digest limb 3"))),
    ];
    core::array::from_fn(|idx| {
        bellpepper_core::num::AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(values[idx]))
            .expect("alloc digest vars")
    })
}

#[test]
fn rv64im_side_relation_n2_debug_satisfiable() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build N=2 fixture");
    let (statement, witness) = fixture
        .build_side_debug_inputs()
        .expect("build side debug inputs");
    debug_check_rv64im_side_spartan_circuit(&statement, &witness).expect("debug check side spartan");
}

#[test]
#[ignore = "manual debug canary for the N=2 mixed-opcode main decider circuit"]
fn rv64im_main_relation_n2_debug_satisfiable() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build N=2 fixture");
    debug_check_rv64im_spartan2_decider_circuit(&fixture.final_statement, &fixture.final_proof)
        .expect("debug check main spartan decider");
}

#[test]
#[ignore = "manual debug canary for the N=2 mixed-opcode published-seam main decider circuit"]
fn rv64im_main_relation_n2_published_seam_debug_satisfiable() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let ((_proof, seam), _perf) =
        prove_rv64im_public_proof_and_published_seam_with_perf(&input).expect("build N=2 published seam");
    debug_check_rv64im_spartan2_decider_circuit(&seam.final_statement, &seam.final_proof)
        .expect("debug check main spartan decider on published seam");
}

#[test]
fn rv64im_phase0_transcript_parity_n2() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build N=2 fixture");
    let (_statement, witness) = fixture
        .build_side_audit_inputs()
        .expect("build side debug inputs");
    let claim_witness = witness
        .phase0_claim_witnesses
        .iter()
        .find(|claim| claim.claim.payload.schema == FamilyEvalSchemaId::Stage2RegisterWrites)
        .expect("stage2 register-writes phase0 claim witness");

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let (opened_object_digest_vars, packed_column_matrix_entries) =
        circuit_enforce_phase0_commitment_root_and_opened_object_digest(
            cs.namespace(|| "phase0_commitment"),
            claim_witness.claim.payload.schema,
            &claim_witness.claim.commitment_context,
            &claim_witness.witness.packed_columns,
            &claim_witness.witness.commitment_vector,
            &claim_witness.claim.opened_object,
            "phase0_commitment",
        )
        .expect("enforce phase0 commitment root and opened-object digest");
    let binding_digest_vars = alloc_digest_vars(&mut cs, claim_witness.claim.binding_digest, "phase0_binding_digest");
    let (point_vars, point_values) = circuit_derive_phase0_point(
        cs.namespace(|| "phase0_point"),
        &opened_object_digest_vars,
        claim_witness.claim.opened_object.digest,
        &claim_witness.claim.commitment_context,
        claim_witness.claim.payload.schema,
        claim_witness.claim.id.slot,
        &binding_digest_vars,
        claim_witness.claim.binding_digest,
        claim_witness.claim.opened_object.row_domain_log_size as usize,
        "phase0_point",
    )
    .expect("derive phase0 point");
    circuit_enforce_phase0_point_eq(
        &mut cs.namespace(|| "phase0_point_eq"),
        &point_vars,
        &claim_witness.claim.point,
        "phase0_point_eq",
    )
    .expect("enforce phase0 point eq");
    assert_eq!(point_values, claim_witness.claim.point);

    let expected_payload = claim_witness
        .claim
        .payload
        .column_evals
        .iter()
        .map(|column| column.coeffs.to_vec())
        .collect::<Vec<_>>();
    let (payload_vars, payload_values) = circuit_evaluate_phase0_payload_from_packed_rows(
        &mut cs.namespace(|| "phase0_payload"),
        &claim_witness.witness.packed_columns,
        &packed_column_matrix_entries,
        &point_vars,
        &point_values,
        "phase0_payload",
    )
    .expect("evaluate phase0 payload");
    circuit_enforce_phase0_payload_eq(
        &mut cs.namespace(|| "phase0_payload_eq"),
        &payload_vars,
        &expected_payload,
        "phase0_payload_eq",
    )
    .expect("enforce phase0 payload eq");
    assert_eq!(payload_values, expected_payload);
    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

#[test]
#[ignore = "manual redline canary for the paper-faithful cut; faster than tests/perf.rs and should eventually go green"]
fn rv64im_main_relation_n2_counting_budget() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build N=2 fixture");
    let metrics = fixture
        .measure_main_relation()
        .expect("measure main relation");
    let replay_relation = metrics
        .phase_rollup
        .iter()
        .filter(|bucket| {
            matches!(
                bucket.phase.as_str(),
                "transcript_core"
                    | "transcript_bind"
                    | "challenge_sampling"
                    | "initial_sum"
                    | "sumcheck_fe"
                    | "sumcheck_nc"
                    | "terminal_fe"
                    | "terminal_nc"
                    | "fold_digest"
                    | "other"
            )
        })
        .map(|bucket| bucket.constraint_count)
        .sum::<usize>();
    let carrier_relation = metrics
        .phase_rollup
        .iter()
        .filter(|bucket| {
            matches!(
                bucket.phase.as_str(),
                "carrier_outputs" | "carrier_parent" | "carrier_children"
            )
        })
        .map(|bucket| bucket.constraint_count)
        .sum::<usize>();
    assert!(
        metrics.constraint_count <= RV64IM_N2_TOTAL_CONSTRAINT_BUDGET
            && replay_relation <= RV64IM_N2_REPLAY_CONSTRAINT_BUDGET
            && carrier_relation <= RV64IM_N2_CARRIER_CONSTRAINT_BUDGET,
        "N=2 paper-boundary budget exceeded: total={}, replay={}, carrier={}, target_total={}, target_replay={}, target_carrier={}",
        metrics.constraint_count,
        replay_relation,
        carrier_relation,
        RV64IM_N2_TOTAL_CONSTRAINT_BUDGET,
        RV64IM_N2_REPLAY_CONSTRAINT_BUDGET,
        RV64IM_N2_CARRIER_CONSTRAINT_BUDGET,
    );
}

#[test]
#[ignore = "manual redline canary for fixed theorem-surface fanout on tiny traces"]
fn rv64im_main_relation_n2_surface_proportionality() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build N=2 fixture");
    let real_rows = fixture.real_rows();
    let child_claim_count = fixture.child_claim_count();
    let packaged_final_main_claims_total = fixture.packaged_final_main_claims_total();
    let packaged_dec_children_total = fixture.packaged_dec_children_total();
    assert!(
        child_claim_count <= 2 * real_rows
            && packaged_final_main_claims_total <= 2 * real_rows
            && packaged_dec_children_total <= 2 * real_rows,
        "N=2 surface proportionality violated: real_rows={}, child_claims={}, packaged_final_main_claims_total={}, packaged_dec_children_total={}",
        real_rows,
        child_claim_count,
        packaged_final_main_claims_total,
        packaged_dec_children_total,
    );
}

#[test]
#[ignore = "manual redline canary for theorem-facing side-boundary baggage"]
fn rv64im_side_relation_n2_public_boundary() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build N=2 fixture");
    let (statement, witness) = fixture
        .build_side_debug_inputs()
        .expect("build side debug inputs");
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/authoritative_side/statement");
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/authoritative_side/statement/version",
        b"v1",
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/authoritative_side/statement/nightstream_statement_core_digest",
        &fixture.nightstream_statement.core_digest(),
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/authoritative_side/statement/public_instance_digest",
        &witness.digest,
    );
    assert_eq!(
        statement.digest(),
        tr.digest32(),
        "side theorem boundary still carries baggage beyond the Nightstream statement core and authoritative public instance digest",
    );
}

#[test]
#[ignore = "manual redline canary for the tiny-N side theorem circuit"]
fn rv64im_side_relation_n2_counting_budget() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build N=2 fixture");
    let constraint_count = fixture
        .measure_side_relation_constraints()
        .expect("measure side relation");
    assert!(
        constraint_count <= RV64IM_N2_SIDE_CONSTRAINT_BUDGET,
        "N=2 side-theorem budget exceeded: total={}, target_total={}",
        constraint_count,
        RV64IM_N2_SIDE_CONSTRAINT_BUDGET,
    );
}
