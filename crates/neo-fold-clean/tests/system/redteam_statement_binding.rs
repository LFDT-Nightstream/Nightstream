//! Retained red-team regressions for verifier-owned public statements.

#[path = "../support/mod.rs"]
mod support;

use neo_ajtai::Commitment;
use neo_math::{D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_fold_clean::frontends::f_prime::compiler::chunk_digest_for_shape;
use neo_fold_clean::frontends::f_prime::recursive_plan::build_semantic_state_preimage_fields;
use neo_fold_clean::frontends::r1cs_f_prime::{self, R1csChainBuilder};
use neo_fold_clean::paper::construction2::ProofState;
use neo_fold_clean::paper::digest::{digest_fields_as_digest32, terminal_ce_public_digest, terminal_children_digest};
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::terminal_ce::TerminalCePublic;

use support::r1cs_compiler_fixtures::{
    assignment_one_product_with_extras, make_tiny_stateful_lifecycle_plan_with_anchor, one_product_r1cs, tiny_params,
};

fn semantic_digest(value: u64) -> [u8; 32] {
    let fields = [F::from_u64(value)];
    digest_fields_as_digest32(encode_poseidon_trace(&build_semantic_state_preimage_fields(&fields)).digest_native)
}

/// HyperNova's IVC verifier is parameterized by the public start/end
/// statement. A verifier intending final semantic state `H(43)` must not
/// accept a proof for `H(44)` merely because the latter is internally valid.
///
/// The application relation deliberately leaves `z[7]` nondeterministic so
/// both outputs are legal executions. This isolates statement ownership from
/// circuit satisfaction: the expected final state must come from the verifier,
/// not from the proof being checked.
#[test]
fn verify_uncompressed_rejects_proof_for_unexpected_final_semantic_state() {
    let r1cs = one_product_r1cs();
    let initial = semantic_digest(42);
    let expected_final = semantic_digest(43);
    let attacker_final = semantic_digest(44);
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(r1cs.m(), r1cs.m_in, vec![6], vec![7], Some(initial));
    let prep = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0x5A7E_0001)
        .expect("stateful preprocessing");

    let mut chain = R1csChainBuilder::new(&prep).expect("start stateful chain");
    chain
        .append_assignment(assignment_one_product_with_extras(1, 1, &[(6, 42), (7, 44)]))
        .expect("prove attacker-selected final state");
    let audit = chain.finish_with_audit().expect("finalize one-step chain");

    assert_eq!(audit.proof.state.semantic_state_digest, attacker_final);
    assert_ne!(audit.proof.state.semantic_state_digest, expected_final);

    let result = neo_fold_clean::verify_uncompressed(&prep.prep, &audit.proof);
    assert!(
        result.is_err(),
        "soundness failure: terminal verifier accepted a proof for attacker-selected H(44) \
         when the verifier's public statement requires H(43); verify_uncompressed has no \
         verifier-owned expected-final-state input"
    );
}

/// HyperNova's base case is the default satisfying accumulator at `i = 0`.
/// The public lifecycle already constructs this state for an empty batch
/// iterator, so finalization and verification must preserve perfect
/// completeness instead of producing an artifact no verifier accepts.
#[test]
fn verify_uncompressed_accepts_honest_zero_step_base_case() {
    let prep = support::toy_preprocessing();
    let audit = neo_fold_clean::prove(&prep, std::iter::empty::<Vec<neo_fold_clean::CcsInstance>>())
        .expect("construct honest zero-step base state");
    let proof = neo_fold_clean::finish_uncompressed(&prep, audit).expect("finalize honest zero-step base state");

    assert!(
        neo_fold_clean::verify_uncompressed(&prep, &proof).is_ok(),
        "completeness failure: the public lifecycle constructs and finalizes an honest i=0 base proof that verify_uncompressed cannot accept"
    );
}

/// `FinalFoldProof::terminal_inputs` is part of the public terminal proof and
/// is the snapshot consumed by `verify_uncompressed`.  The audit verifier
/// must not silently accept a contradictory value for that same wire field
/// merely because it reconstructs equivalent fold inputs from its separate
/// `public_batches` trail.
#[test]
fn audit_verifier_rejects_contradictory_terminal_input_snapshot() {
    let prep = support::toy_preprocessing();
    let audit =
        neo_fold_clean::prove(&prep, [vec![support::toy_instance(&prep, 28)]]).expect("construct one-batch audit");
    let mut finalized = neo_fold_clean::finish_uncompressed_with_audit(&prep, audit).expect("finalize one-batch audit");
    neo_fold_clean::verify_uncompressed_audit(&prep, &finalized)
        .expect("honest audit verifies before the wire snapshot is changed");

    finalized
        .proof
        .final_fold
        .as_mut()
        .expect("one-batch proof carries a terminal fold")
        .terminal_inputs
        .latest
        .instances[0]
        .claim
        .x[0] += F::ONE;

    assert!(
        neo_fold_clean::verify_uncompressed(&prep, &finalized.proof).is_err(),
        "attack precondition: the terminal verifier must consume and reject the contradictory snapshot"
    );
    assert!(
        neo_fold_clean::verify_uncompressed_audit(&prep, &finalized).is_err(),
        "soundness failure: the audit verifier ignored contradictory final_fold.terminal_inputs accepted by its public proof type"
    );
}

/// The recorded running parent is part of the terminal accumulator authority.
/// Every one of its carried fields must either be re-derived from the terminal
/// fold or included in the accumulator digest.  Otherwise the production
/// terminal verifier and audit replay accept different wire languages.
#[test]
fn terminal_verifier_rejects_unbound_parent_y_zcol() {
    let prep = support::toy_preprocessing();
    let audit =
        neo_fold_clean::prove(&prep, [vec![support::toy_instance(&prep, 31)]]).expect("construct one-batch audit");
    let mut finalized = neo_fold_clean::finish_uncompressed_with_audit(&prep, audit).expect("finalize one-batch audit");
    neo_fold_clean::verify_uncompressed(&prep, &finalized.proof)
        .expect("honest terminal proof verifies before mutation");
    neo_fold_clean::verify_uncompressed_audit(&prep, &finalized).expect("honest audit verifies before mutation");

    let neo_fold_clean::paper::construction2::ProofState::Active { running, .. } = &mut finalized.proof.state.proof
    else {
        panic!("finalized proof must be active");
    };
    running
        .as_materialized_mut()
        .expect("fixture uses a materialized final running accumulator")
        .parent_authority
        .as_mut()
        .expect("nonempty final running carries Pi_RLC parent authority")
        .y_zcol[0] += K::ONE;

    assert!(
        neo_fold_clean::verify_uncompressed_audit(&prep, &finalized).is_err(),
        "attack precondition: audit replay must bind and reject the changed parent authority"
    );
    assert!(
        neo_fold_clean::verify_uncompressed(&prep, &finalized.proof).is_err(),
        "soundness failure: terminal verifier accepted a mutated running.parent_authority.y_zcol that is absent from both its derived-state comparison and accumulator digest"
    );
}

/// Inactive X columns are required to be zero by the native Π_RLC/Π_DEC
/// language.  The separately recorded terminal parent must not be able to
/// smuggle data into those digest-skipped columns after the verifier has
/// derived the honest parent and children.
#[test]
fn terminal_verifier_rejects_unbound_parent_inactive_x() {
    let structure = CcsStructure::new(vec![Mat::zero(1, 2, F::ZERO)], SparsePoly::<F>::new(1, Vec::new()))
        .expect("two-input zero relation");
    let params = neo_fold_clean::config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape params");
    support::install_ajtai_module(&params, &structure);
    let prep = neo_fold_clean::preprocess(params, structure, Some(2)).expect("two-public-input preprocessing");
    let instance = neo_fold_clean::CcsInstance::from_low_norm_assignment(
        &prep.params,
        &prep.log,
        prep.structure(),
        &[F::ZERO, F::ONE],
        2,
    )
    .expect("satisfying two-input instance");
    let audit = neo_fold_clean::prove(&prep, [vec![instance]]).expect("construct one-batch audit");
    let mut finalized = neo_fold_clean::finish_uncompressed_with_audit(&prep, audit).expect("finalize one-batch audit");
    neo_fold_clean::verify_uncompressed(&prep, &finalized.proof)
        .expect("honest terminal proof verifies before mutation");
    neo_fold_clean::verify_uncompressed_audit(&prep, &finalized).expect("honest audit verifies before mutation");

    let ProofState::Active { running, .. } = &mut finalized.proof.state.proof else {
        panic!("finalized proof must be active");
    };
    let parent = running
        .as_materialized_mut()
        .expect("fixture uses a materialized final accumulator")
        .parent_authority
        .as_mut()
        .expect("nonempty final running carries Pi_RLC parent authority");
    assert_eq!(parent.m_in, 2);
    assert_eq!(parent.X.cols(), 2);
    parent.X[(0, 1)] = F::ONE;

    assert!(
        neo_fold_clean::verify_uncompressed_audit(&prep, &finalized).is_err(),
        "attack precondition: audit replay must reject nonzero inactive parent X data"
    );
    assert!(
        neo_fold_clean::verify_uncompressed(&prep, &finalized.proof).is_err(),
        "soundness failure: terminal verifier accepted a nonzero inactive column in recorded running.parent_authority.X"
    );
}

/// Parent commitment dimensions are integer metadata, but the accumulator
/// digest encodes them as one Goldilocks element. On a 64-bit target, adding
/// the field modulus therefore leaves the digest unchanged. The terminal
/// verifier must compare the exact derived parent rather than accept this
/// numeric alias while audit replay rejects it.
#[cfg(target_pointer_width = "64")]
#[test]
fn terminal_verifier_rejects_unbound_parent_commitment_dimension_alias() {
    let prep = support::toy_preprocessing();
    let audit =
        neo_fold_clean::prove(&prep, [vec![support::toy_instance(&prep, 35)]]).expect("construct one-batch audit");
    let mut finalized = neo_fold_clean::finish_uncompressed_with_audit(&prep, audit).expect("finalize one-batch audit");
    neo_fold_clean::verify_uncompressed(&prep, &finalized.proof)
        .expect("honest terminal proof verifies before mutation");
    neo_fold_clean::verify_uncompressed_audit(&prep, &finalized).expect("honest audit verifies before mutation");

    let ProofState::Active { running, .. } = &mut finalized.proof.state.proof else {
        panic!("finalized proof must be active");
    };
    let parent = running
        .as_materialized_mut()
        .expect("fixture uses a materialized final accumulator")
        .parent_authority
        .as_mut()
        .expect("nonempty final running carries Pi_RLC parent authority");
    let original_d = parent.c.d;
    parent.c.d = original_d
        .checked_add(F::ORDER_U64 as usize)
        .expect("Goldilocks modulus plus D fits 64-bit usize");
    assert_eq!(
        F::from_u64(parent.c.d as u64),
        F::from_u64(original_d as u64),
        "attack precondition: commitment dimensions alias in the accumulator digest"
    );

    assert!(
        neo_fold_clean::verify_uncompressed_audit(&prep, &finalized).is_err(),
        "attack precondition: audit replay must compare and reject the changed parent dimension"
    );
    assert!(
        neo_fold_clean::verify_uncompressed(&prep, &finalized.proof).is_err(),
        "differential-verification failure: terminal verifier accepted parent commitment dimension d+p because its accumulator digest reduced the integer modulo Goldilocks"
    );
}

/// The Construction-2 chunk digest is the active per-step shape domain
/// separator. Its integer dimensions must be encoded injectively rather than
/// reduced modulo the Goldilocks field.
#[cfg(target_pointer_width = "64")]
#[test]
fn f_prime_chunk_shape_digest_binds_full_integer_dimensions() {
    let d = neo_math::D;
    let kappa = 18usize;
    let m_in = 1usize;
    let modulus = F::ORDER_U64 as usize;

    let canonical = chunk_digest_for_shape(0, d, kappa, m_in);
    let d_alias = chunk_digest_for_shape(
        0,
        d.checked_add(modulus)
            .expect("Goldilocks modulus plus D fits 64-bit usize"),
        kappa,
        m_in,
    );

    assert_ne!(
        canonical, d_alias,
        "distinct Construction-2 chunk shapes must not alias when d differs by the Goldilocks modulus"
    );
}

/// `TerminalFoldInputs` is documented as a claims-only wire snapshot: the
/// private latest witness is replaced by empty placeholders, and the
/// pre-final running carries no witness matrices.  The verifier must enforce
/// that boundary instead of accepting an equivalent proof with attacker-
/// controlled private payloads appended to it.
#[test]
fn terminal_verifier_rejects_unstripped_witness_payloads() {
    let prep = support::toy_preprocessing();
    let audit =
        neo_fold_clean::prove(&prep, [vec![support::toy_instance(&prep, 32)]]).expect("construct one-batch audit");
    let honest = neo_fold_clean::finish_uncompressed(&prep, audit).expect("finalize one-batch proof");
    neo_fold_clean::verify_uncompressed(&prep, &honest).expect("honest stripped terminal proof verifies");

    let mut with_latest_witness = honest.clone();
    let latest_witness = &mut with_latest_witness
        .final_fold
        .as_mut()
        .expect("one-batch proof carries a terminal fold")
        .terminal_inputs
        .latest
        .instances[0]
        .witness;
    latest_witness.w = vec![F::ONE];
    latest_witness.Z = Mat::zero(neo_math::D, 1, F::ONE);
    let latest_result = neo_fold_clean::verify_uncompressed(&prep, &with_latest_witness);

    let mut with_running_witness = honest;
    with_running_witness
        .final_fold
        .as_mut()
        .expect("one-batch proof carries a terminal fold")
        .terminal_inputs
        .pre_final_running
        .witnesses
        .push(Mat::zero(neo_math::D, 1, F::ONE));
    let running_result = neo_fold_clean::verify_uncompressed(&prep, &with_running_witness);

    assert!(
        latest_result.is_err() && running_result.is_err(),
        "proof-boundary failure: terminal verifier accepted witness-bearing snapshots despite the claims-only wire contract (latest={latest_result:?}, running={running_result:?})"
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn terminal_ce_public_digest_binds_full_claim_count() {
    let zero = [F::ZERO; 4];
    let empty = terminal_ce_public_digest(zero, zero, zero, zero, 0);
    let aliased = terminal_ce_public_digest(zero, zero, zero, zero, F::ORDER_U64 as usize);

    assert_ne!(
        empty, aliased,
        "distinct terminal CE claim counts must not alias modulo Goldilocks"
    );
}

/// `fold_digest` is a 32-byte public transcript handle, but terminal-CE
/// hashing interprets it as four Goldilocks lanes. The native public-statement
/// constructor must reject byte limbs outside the canonical field encoding;
/// otherwise distinct transcript handles alias before the compact proof sees
/// them.
#[test]
fn terminal_ce_public_rejects_noncanonical_fold_digest_alias() {
    let prep = support::toy_preprocessing();
    let d_pad = D.next_power_of_two();
    let ell_n = prep
        .structure()
        .n
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let ell_m = prep
        .structure()
        .m
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let t = prep.structure().t();
    let kappa = prep.params.kappa() as usize;

    let canonical_child = neo_fold_clean::CeClaim {
        c: Commitment {
            d: D,
            kappa,
            data: vec![F::ZERO; D * kappa],
        },
        X: Mat::zero(D, 1, F::ZERO),
        r: vec![K::ZERO; ell_n],
        s_col: vec![K::ZERO; ell_m],
        y_ring: vec![vec![K::ZERO; d_pad]; t],
        ct: vec![K::ZERO; t],
        aux_openings: Vec::new(),
        y_zcol: vec![K::ZERO; d_pad],
        m_in: 1,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    };
    let canonical_public =
        TerminalCePublic::from_terminal_children(&prep.params, prep.structure(), &[canonical_child.clone()])
            .expect("canonical terminal child");

    let mut noncanonical_child = canonical_child.clone();
    noncanonical_child.fold_digest[..8].copy_from_slice(&F::ORDER_U64.to_le_bytes());
    assert_ne!(
        canonical_child.fold_digest, noncanonical_child.fold_digest,
        "attack precondition: the child transcript handles are byte-distinct"
    );
    assert_eq!(
        terminal_children_digest(&[canonical_child]),
        terminal_children_digest(&[noncanonical_child.clone()]),
        "attack precondition: digest32_as_fields reduces the modulus byte limb to canonical zero"
    );

    let noncanonical_public =
        TerminalCePublic::from_terminal_children(&prep.params, prep.structure(), &[noncanonical_child]);
    if let Ok(alias) = &noncanonical_public {
        assert_eq!(
            alias, &canonical_public,
            "attack precondition: the full compact public statements alias"
        );
    }
    assert!(
        noncanonical_public.is_err(),
        "compact-statement soundness failure: TerminalCePublic accepted a noncanonical fold_digest whose distinct bytes alias the canonical-zero child and public digest"
    );
}
