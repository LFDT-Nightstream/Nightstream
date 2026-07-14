//! Isolated coverage for the terminal CE-relation gadget.
//!
//! Each test hits one obligation (`commit / X / low-norm / y_ring / ct`
//! and the implementation-carried NC sidecar) by mutating exactly the
//! data that feeds that gate, holding the other obligations valid.
//! Replaces the older end-to-end "tamper Z, bypass preflight" test,
//! which couldn't isolate which obligation was load-bearing.
//!
//! Two fixtures:
//! - The toy `(n=1, m=1)` running plucked from a real lifecycle proof
//!   — easy baseline + commit/X tampering.
//! - A non-trivial `(n=4, m=16, log_n=2)` R1CS-derived fixture built
//!   from scratch via `r1cs_to_ccs` + a synthetic CE claim. This is
//!   the only path that actually exercises the in-circuit `chi_r`
//!   tensor unfold for `log_n ≥ 2` and the low-norm rows in isolation.

#![allow(non_snake_case)]

#[path = "../support/mod.rs"]
mod support;

use neo_ajtai::{has_global_pp_for_dims, set_global_pp_seeded};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::utils::tensor_point;
use neo_ccs::{r1cs_to_ccs, Mat};
use neo_fold_clean::config;
use neo_fold_clean::engine::decider::__test_isolation::{
    enforce_ce_relations_against, enforce_ce_relations_many_against, enforce_ce_relations_with_wires_against,
};
use neo_fold_clean::engine::r1cs_circuit::builder::Var;
use neo_fold_clean::paper::construction2::ProofState;
use neo_fold_clean::paper::construction2::{NebulaConfig, StackShape};
use neo_fold_clean::paper::relations::{LaneRanges, LaneScheme};
use neo_fold_clean::{preprocess, CeClaim, Params, Preprocessing, Structure};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::common::{compute_y_from_Z_and_r, project_x_from_witness_mat};
use p3_field::PrimeCharacteristicRing;

fn k_c1_one() -> K {
    K::from_coeffs([F::ZERO, F::ONE])
}

fn assert_only_fold_digest_unconstrained(
    builder: &neo_fold_clean::engine::r1cs_circuit::R1csBuilder,
    fold_digest_fields: &[[Var; 4]],
    label: &str,
) {
    let unconstrained = builder.unconstrained_columns();
    let mut allowed: Vec<usize> = fold_digest_fields
        .iter()
        .flat_map(|digest| digest.iter().map(|var| var.col()))
        .collect();
    allowed.sort_unstable();
    assert!(
        unconstrained == allowed,
        "{label} left unexpected terminal CE wires unconstrained: got {unconstrained:?}, \
         expected only fold_digest metadata wires {allowed:?}"
    );
}

/// Build an honest finished proof, pluck its final running's first
/// `(claim, witness)` pair, and assert the CE-relation gadget alone
/// (no chain replay, no public-image pins) accepts it. Sanity baseline
/// for the tampering tests below.
#[test]
fn decider_ce_isolation_accepts_honest_terminal_pair() {
    let (prep, claim, witness) = honest_terminal_pair();
    let output = enforce_ce_relations_with_wires_against(&prep, &claim, &witness).expect("synthesis");
    let builder = &output.builder;
    assert!(
        builder.is_satisfied(),
        "honest terminal pair must satisfy the CE-relation gadget alone; \
         first unsatisfied row: {:?}",
        builder.first_unsatisfied_row()
    );
    assert_only_fold_digest_unconstrained(builder, &output.fold_digest_fields, "honest toy terminal CE pair");
}

/// **Claim/witness pairing isolation.** Every terminal CE child must have
/// exactly one opened `Z`. This catches the classic iterator-zip footgun:
/// without an explicit count check, an extra claim or extra witness would
/// be silently skipped and left outside the CE relation.
#[test]
fn decider_ce_isolation_rejects_claim_witness_count_mismatch() {
    let (prep, claim, witness) = honest_terminal_pair();

    let cases = [
        ("extra claim", vec![claim.clone(), claim.clone()], vec![witness.clone()]),
        ("extra witness", vec![claim], vec![witness.clone(), witness]),
    ];

    for (name, claims, witnesses) in cases {
        let err = enforce_ce_relations_many_against(&prep, &claims, &witnesses)
            .err()
            .unwrap_or_else(|| panic!("{name} must abort CE-relation synthesis"));
        assert!(
            err.contains("claim/witness count mismatch"),
            "expected count-mismatch error for {name}, got: {err}"
        );
    }
}

/// **y_ring isolation.** Mutate one `y_ring` entry on the terminal
/// claim while leaving the witness, `c`, `X`, low-norm bounds, and
/// `r` unchanged. The Ajtai opening, X projection, and balanced-digit
/// rows still hold; only the CE-evaluation closure can fail.
///
/// This is the test the previous end-to-end "tamper Z" couldn't be —
/// that one mutated `Z`, which broke commit / X / low-norm and
/// y_ring simultaneously, so it proved the witness was constrained
/// but not which row caught it.
#[test]
fn decider_ce_isolation_rejects_y_ring_inconsistent_with_m_z_at_r() {
    let (prep, mut claim, witness) = honest_terminal_pair();
    assert!(
        !claim.y_ring.is_empty() && !claim.y_ring[0].is_empty(),
        "test setup must have a non-empty y_ring"
    );
    let original = claim.y_ring[0][0];
    claim.y_ring[0][0] = original + K::ONE;
    assert_ne!(claim.y_ring[0][0], original, "mutation must change y_ring[0][0]");

    let builder = enforce_ce_relations_against(&prep, &claim, &witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted y_ring inconsistent with M·Z at r — the y_ring \
         evaluation rows must be load-bearing in isolation (commit / X / low-norm rows \
         alone are not sufficient)"
    );
}

/// **commit isolation.** Mutate one `c.data` lane on the terminal
/// claim. The Ajtai opening row binds `commit(Z) == c.data`; flipping
/// `c.data` breaks the opening even though `Z` still satisfies the
/// other four obligations.
#[test]
fn decider_ce_isolation_rejects_commitment_not_opened_by_z() {
    let (prep, mut claim, witness) = honest_terminal_pair();
    assert!(!claim.c.data.is_empty(), "test setup must have a non-empty commitment");
    let original = claim.c.data[0];
    claim.c.data[0] = original + F::ONE;
    assert_ne!(claim.c.data[0], original, "mutation must change c.data[0]");

    let builder = enforce_ce_relations_against(&prep, &claim, &witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted a commitment that Z does not open — the Ajtai \
         opening rows must be load-bearing"
    );
}

#[test]
fn decider_ce_isolation_accepts_honest_nebula_lane_openings() {
    let fixture = non_trivial_nebula_fixture();
    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        builder.is_satisfied(),
        "honest Nebula lane commitments must open against their terminal witness slices; first bad row: {:?}",
        builder.first_unsatisfied_row()
    );
}

#[test]
fn decider_ce_isolation_rejects_nebula_lane_not_opened_by_z() {
    let mut fixture = non_trivial_nebula_fixture();
    fixture.claim.adv.as_mut().expect("Nebula adv").ops.data[0] += F::ONE;

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "terminal CE relation accepted an adv.ops commitment not opened by the designated witness slice"
    );
}

/// **Ajtai shape isolation.** Claim one more commitment column than the
/// registered Ajtai setup actually has, and pad `c.data` so the local
/// commitment shape is self-consistent. Without an explicit
/// `pp.m_rows.len() == c.kappa` guard, the opening gadget only emits rows
/// for the setup's columns and leaves the extra commitment column
/// unconstrained.
#[test]
fn decider_ce_isolation_rejects_ajtai_kappa_mismatch() {
    let (prep, mut claim, witness) = honest_terminal_pair();
    let original_kappa = claim.c.kappa;
    claim.c.kappa += 1;
    claim
        .c
        .data
        .extend(std::iter::repeat(F::ZERO).take(claim.c.d));
    assert_eq!(
        claim.c.data.len(),
        claim.c.d * claim.c.kappa,
        "test mutation keeps the commitment shape internally consistent"
    );

    let err = enforce_ce_relations_against(&prep, &claim, &witness)
        .err()
        .expect("Ajtai kappa mismatch must abort CE-relation synthesis");
    assert!(
        err.contains("Ajtai kappa"),
        "expected an `Ajtai kappa` shape-mismatch error for original κ={original_kappa}, got: {err}"
    );
}

/// **X isolation.** Mutate the claim's public-input matrix `X` in
/// an active column. Witness Z still opens c and is low-norm; only
/// the projection row should fail.
#[test]
fn decider_ce_isolation_rejects_x_not_projected_from_z() {
    let (prep, mut claim, witness) = honest_terminal_pair();
    let m_in = claim.m_in;
    let required_cols = m_in.div_ceil(neo_math::D);
    assert!(required_cols > 0, "test fixture must expose active X columns");
    let original = claim.X[(0, 0)];
    claim.X[(0, 0)] = original + F::ONE;
    assert_ne!(claim.X[(0, 0)], original, "mutation must change X[0, 0]");

    let builder = enforce_ce_relations_against(&prep, &claim, &witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted X that does not project from Z — the X projection \
         rows must be load-bearing"
    );
}

/// **Packed X tail isolation.** `m_in` counts scalar public field lanes, but
/// SuperNeo's `L_in(Z)` projects whole active ring columns. For `m_in = 1`,
/// row 0 of column 0 is the scalar public input while rows `1..D` in the
/// same column are still active ring coordinates. Mutate one of those tail
/// rows, leaving the scalar public lane untouched. A scalar-only projection
/// check would accept this; the terminal CE closure must reject it by binding
/// the full active packed column to `Z`.
#[test]
fn decider_ce_isolation_rejects_active_x_tail_not_projected_from_z() {
    let mut fixture = non_trivial_fixture_with_m_in(1);
    assert_eq!(fixture.claim.m_in, 1, "fixture must expose one scalar public lane");
    assert!(
        fixture.claim.X.rows() > 1 && fixture.claim.X.cols() == 1,
        "fixture must have active packed-tail X lanes"
    );

    let original = fixture.claim.X[(1, 0)];
    fixture.claim.X[(1, 0)] = original + F::ONE;
    assert_ne!(
        fixture.claim.X[(1, 0)],
        original,
        "mutation must change the active packed tail lane without touching X[0,0]"
    );

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted an active packed X tail lane that does not project from Z; \
         terminal X projection must bind full active ring columns, not only scalar public lanes"
    );
}

/// **Packed witness tail isolation.** Production F' uses logical CCS width
/// `m = 257`, represented by five complete `D = 54` witness blocks. The final
/// thirteen packed lanes are not logical CCS variables, but they remain part
/// of the ring element consumed by `compute_y_from_Z_and_r`. Mutating one of
/// those lanes and recomputing the Ajtai commitment must therefore invalidate
/// a stale `y_ring/ct` pair even though `X` and every logical witness entry are
/// unchanged.
#[test]
fn decider_ce_isolation_rejects_stale_y_after_packed_witness_tail_mutation() {
    let mut fixture = non_trivial_fixture_with_load_bearing_final_block(257, 1, Some(1));
    let packed_width = fixture.witness.cols() * D;
    assert_eq!(
        packed_width, 270,
        "production-width fixture must have five packed blocks"
    );
    assert_eq!(
        packed_width - fixture.prep.structure().m,
        13,
        "production-width fixture must expose the real thirteen-lane packed tail"
    );

    let tail_col = packed_width - 1;
    let tail_off = tail_col % D;
    let tail_block = tail_col / D;
    assert_eq!(fixture.witness[(tail_off, tail_block)], F::ZERO);
    fixture.witness[(tail_off, tail_block)] = F::ONE;

    // Keep every non-evaluation obligation honest for the mutated witness.
    fixture.claim.c = fixture.prep.log.commit(&fixture.witness);
    fixture.claim.X = project_x_from_witness_mat(&fixture.witness, fixture.prep.structure().m, fixture.claim.m_in)
        .expect("X projection after packed-tail mutation");

    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let (updated_y_ring, updated_ct) = compute_y_from_Z_and_r(
        fixture.prep.structure(),
        &fixture.witness,
        &fixture.claim.r,
        ell_d,
        fixture.prep.params.b(),
    );
    assert_ne!(
        updated_y_ring, fixture.claim.y_ring,
        "the final packed lane must be load-bearing in the native ring evaluation"
    );

    let stale = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !stale.is_satisfied(),
        "CE-relation gadget accepted stale y_ring/ct after a load-bearing packed-tail mutation"
    );

    fixture.claim.y_ring = updated_y_ring;
    fixture.claim.ct = updated_ct;
    let repaired = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        repaired.is_satisfied(),
        "CE-relation gadget must accept the same packed-tail witness once y_ring/ct are recomputed; \
         first unsatisfied row: {:?}",
        repaired.first_unsatisfied_row()
    );
}

/// **m_in relabel isolation.** Keep the terminal witness and every CE
/// value locally self-consistent, but relabel the claim as having no
/// public input. For a zero witness this makes `X = L_in(Z)` vacuously
/// true under the smaller projection, so commit / X / low-norm / y_ring
/// / ct can all pass unless the gadget pins `m_in` to the verifier-owned
/// `Preprocessing.public_input_len`.
#[test]
fn decider_ce_isolation_rejects_m_in_relabel_below_preprocessing_public_input_len() {
    let (prep, mut claim, witness) = honest_terminal_pair();
    assert_eq!(
        prep.public_input_len,
        Some(1),
        "toy preprocessing fixes a one-element public input"
    );
    assert!(
        witness.as_slice().iter().all(|&entry| entry == F::ZERO),
        "toy fixture must exercise a self-consistent zero-witness relabel"
    );

    claim.m_in = 0;
    claim.X = Mat::zero(D, 0, F::ZERO);

    let err = enforce_ce_relations_against(&prep, &claim, &witness)
        .err()
        .expect("m_in relabel must abort CE-relation synthesis");
    assert!(
        err.contains("m_in vs prep.public_input_len"),
        "expected an `m_in vs prep.public_input_len` shape-mismatch error, got: {err}"
    );
}

/// Same relabel attack as above, but on a non-zero, non-trivial witness.
/// This avoids the comforting but weaker "zero witness" case: c, low-norm,
/// y_ring, and ct all remain valid for the same `Z`; only the verifier-owned
/// program shape (`prep.public_input_len`) says the claim cannot shrink
/// `m_in` to zero.
#[test]
fn decider_ce_isolation_rejects_nonzero_witness_m_in_relabel_below_preprocessing_public_input_len() {
    let mut fixture = non_trivial_fixture();
    assert_eq!(
        fixture.prep.public_input_len,
        Some(1),
        "non-trivial fixture fixes a one-element public input"
    );
    assert!(
        fixture
            .witness
            .as_slice()
            .iter()
            .any(|&entry| entry != F::ZERO),
        "test setup must carry a non-zero witness"
    );

    fixture.claim.m_in = 0;
    fixture.claim.X = Mat::zero(D, 0, F::ZERO);

    let err = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness)
        .err()
        .expect("non-zero witness m_in relabel must abort CE-relation synthesis");
    assert!(
        err.contains("m_in vs prep.public_input_len"),
        "expected an `m_in vs prep.public_input_len` shape-mismatch error, got: {err}"
    );
}

/// **m_in structure-bound isolation.** Leave `public_input_len` unfixed,
/// set the CCS width to `m = 2D - 1`, and then relabel the CE claim as
/// `m_in = 2D`. That extra public lane lands exactly in the final packed
/// padding slot of `Z`, so commit / X / low-norm / y_ring / ct can all be
/// locally self-consistent unless the gadget rejects `m_in > structure.m`.
#[test]
fn decider_ce_isolation_rejects_m_in_exceeding_structure_width_when_public_input_len_unfixed() {
    let fixture = non_trivial_fixture_with_shape(2 * D - 1, 2 * D, None);
    assert_eq!(fixture.prep.public_input_len, None, "fixture leaves m_in unfixed");
    assert_eq!(
        fixture.prep.structure().m,
        2 * D - 1,
        "fixture has one packed padding lane"
    );
    assert_eq!(fixture.claim.m_in, 2 * D, "claim overstates m_in by one scalar lane");
    assert_eq!(
        fixture.witness.cols(),
        fixture.claim.m_in.div_ceil(D),
        "overstated m_in still fits the existing packed witness columns"
    );

    let err = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness)
        .err()
        .expect("m_in exceeding structure.m must abort CE-relation synthesis");
    assert!(
        err.contains("m_in vs structure.m"),
        "expected an `m_in vs structure.m` shape-mismatch error, got: {err}"
    );
}

/// **Unsupported sidecar isolation.** `aux_openings`, Pattern-A coordinates,
/// and `u_offset/u_len` are accumulator-digested CE-claim fields, but this
/// clean SplitNc/NIFS path does not implement their algebra. The reference
/// terminal CE gadget must reject them structurally rather than accept a
/// CE-valid `(claim, Z)` with extra authoritative-but-unconstrained metadata.
#[test]
fn decider_ce_isolation_rejects_unsupported_accumulator_sidecars() {
    expect_unsupported_sidecar("aux_openings", |claim| {
        claim.aux_openings.push(K::ONE);
    });
    expect_unsupported_sidecar("c_step_coords", |claim| {
        claim.c_step_coords.push(F::ONE);
    });
    expect_unsupported_sidecar("u_offset", |claim| {
        claim.u_offset = 1;
    });
    expect_unsupported_sidecar("u_len", |claim| {
        claim.u_len = 1;
    });
}

/// **inactive-X isolation.** Native `project_x_from_witness_mat`
/// returns a `D × m_in` matrix and leaves packed columns
/// `ceil(m_in / D)..m_in` as structural zeros. The terminal CE gadget
/// must enforce those inactive columns locally rather than relying on
/// upstream callers. Here we keep the active column honest, add one
/// inactive X column, and make it non-zero. A gadget that checks only
/// the active prefix accepts this; the fixed gadget rejects.
#[test]
fn decider_ce_isolation_rejects_inactive_x_not_zero() {
    let mut fixture = non_trivial_fixture_with_m_in(2);
    let mut x = Mat::zero(D, 2, F::ZERO);
    for row in 0..D {
        x[(row, 0)] = fixture.claim.X[(row, 0)];
    }
    x[(0, 1)] = F::ONE;
    fixture.claim.X = x;

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted a non-zero inactive X column; inactive X columns \
         must be constrained to zero to match native project_x_from_witness_mat"
    );
}

/// **Honest baseline, log_n=2.** Builds a non-trivial CCS (n=4,
/// m=16) from scratch via `r1cs_to_ccs`, synthesises a satisfying
/// `Z`, computes the honest CE claim consistently, and asserts the
/// in-circuit gadget accepts. This is the only fixture that actually
/// exercises `enforce_y_ring_from_z_at_r`'s tensor unfold for
/// `log_n ≥ 2` and multi-block `m`. If the gadget's `chi_r` ordering
/// disagrees with native `tensor_point`, this test fails at the
/// y_ring rows (the helper-only tensor parity test cannot catch a
/// regression in the gadget itself).
#[test]
fn decider_ce_isolation_accepts_honest_pair_log_n_2() {
    let fixture = non_trivial_fixture();
    let output =
        enforce_ce_relations_with_wires_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    let builder = &output.builder;
    assert!(
        builder.is_satisfied(),
        "honest non-trivial (n=4, m=16, log_n=2) CE pair must satisfy the gadget; \
         first unsatisfied row: {:?}",
        builder.first_unsatisfied_row()
    );
    assert_only_fold_digest_unconstrained(
        builder,
        &output.fold_digest_fields,
        "honest non-trivial terminal CE pair",
    );
}

#[test]
fn decider_ce_isolation_seeded_ajtai_native_commit_matches_circuit_opening() {
    let fixture = non_trivial_fixture();
    assert_eq!(
        fixture.prep.log.commit(&fixture.witness),
        fixture.claim.c,
        "fixture claim must use the same seeded Ajtai setup as native preprocessing"
    );

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        builder.is_satisfied(),
        "circuit Ajtai opening rows must accept the native commitment under the seeded setup; \
         first unsatisfied row: {:?}",
        builder.first_unsatisfied_row()
    );
}

/// **y_ring isolation on the non-trivial fixture.** Mutate one y_ring
/// lane on a claim whose `r ∈ K²` exercises the in-circuit tensor
/// unfold non-trivially. Catches both (a) the y_ring eval row missing
/// AND (b) `chi_r` tensor ordering regressions that would make
/// y_ring(r) compute against the wrong row weights.
#[test]
fn decider_ce_isolation_rejects_y_ring_tamper_log_n_2() {
    let mut fixture = non_trivial_fixture();
    let original = fixture.claim.y_ring[0][0];
    fixture.claim.y_ring[0][0] = original + K::ONE;
    assert_ne!(
        fixture.claim.y_ring[0][0], original,
        "mutation must change y_ring[0][0]"
    );

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted y_ring inconsistent with M·Z at r on a log_n=2 fixture — \
         the y_ring evaluation rows OR the chi_r tensor unfold is broken"
    );
}

#[test]
fn decider_ce_isolation_rejects_y_ring_c1_limb_tamper_log_n_2() {
    // Same obligation as the y_ring tamper above, but perturb only the
    // extension-field c1 limb. This catches a half-bound KVar regression
    // where the gadget constrains y_ring.c0 but accidentally leaves c1 free.
    let mut fixture = non_trivial_fixture();
    let original = fixture.claim.y_ring[0][0];
    fixture.claim.y_ring[0][0] = original + k_c1_one();
    assert_eq!(
        fixture.claim.y_ring[0][0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave the c0 limb unchanged"
    );
    assert_ne!(
        fixture.claim.y_ring[0][0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change only the c1 limb"
    );

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted a c1-only y_ring tamper; both K limbs of y_ring \
         must be bound to M·Z(r)"
    );
}

/// **same-shape r tamper.** Keep the witness, commitment, X projection,
/// low-norm alphabet, y_ring shape, and ct shape all intact, but mutate the
/// evaluation point `r` in-place. This is the adversarial case the shape
/// checks do not prove: the old y_ring was computed at the honest `r`, so
/// the circuit must reject by recomputing `M·Z(r')` at the tampered point.
#[test]
fn decider_ce_isolation_rejects_same_shape_r_tamper() {
    let mut fixture = non_trivial_fixture();
    assert_eq!(
        fixture.claim.r.len(),
        2,
        "non-trivial fixture has log_n = 2, so this test keeps r's shape"
    );
    let original = fixture.claim.r[0];
    fixture.claim.r[0] = original + K::ONE;
    assert_eq!(
        fixture.claim.r.len(),
        2,
        "mutation must preserve the r vector length; this is not a shape test"
    );
    assert_ne!(fixture.claim.r[0], original, "mutation must change r[0]");

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted a same-shape r tamper. Shape checks are only hygiene; \
         y_ring must be recomputed from the opened Z at the exact terminal-child r"
    );
}

#[test]
fn decider_ce_isolation_rejects_same_shape_r_c1_limb_tamper() {
    // Same attack as the r-value relabel above, but mutate only the
    // extension-field c1 limb. This catches a bad chi(r) implementation
    // that accidentally feeds only r.c0 into the multilinear tensor.
    let mut fixture = non_trivial_fixture();
    assert_eq!(
        fixture.claim.r.len(),
        2,
        "non-trivial fixture has log_n = 2, so this test keeps r's shape"
    );
    let original = fixture.claim.r[0];
    fixture.claim.r[0] = original + k_c1_one();
    assert_eq!(
        fixture.claim.r[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave the c0 limb unchanged"
    );
    assert_ne!(
        fixture.claim.r[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change only the c1 limb"
    );

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted a same-shape c1-only r tamper. Shape checks and \
         c0-only challenge binding are not enough; the full K-valued evaluation point \
         must feed M·Z(r)"
    );
}

/// **ct isolation.** Mutate `claim.ct[0]` while leaving `y_ring`,
/// `Z`, `c`, `X`, low-norm all valid. The CE-relation gadget must
/// reject because `enforce_ct_from_y_ring` binds `ct[j] ==
/// y_ring[j][lane=0]` per SuperNeo paper Theorem 5. Without this
/// constraint, the circuit would silently accept `ct` mismatched
/// from the constant-term lane of `y_ring`, while the native
/// `verify_uncompressed` already enforces the same obligation.
#[test]
fn decider_ce_isolation_rejects_ct_inconsistent_with_y_ring() {
    let mut fixture = non_trivial_fixture();
    assert!(!fixture.claim.ct.is_empty(), "test setup must have non-empty ct");
    let original = fixture.claim.ct[0];
    fixture.claim.ct[0] = original + K::ONE;
    assert_ne!(fixture.claim.ct[0], original, "mutation must change ct[0]");

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted ct inconsistent with y_ring lane-0; \
         enforce_ct_from_y_ring must bind ct[j] == y_ring[j][lane=0] per Paper Theorem 5"
    );
}

#[test]
fn decider_ce_isolation_rejects_ct_c1_limb_inconsistent_with_y_ring() {
    // `ct` is a K element. A c0-only equality would still pass this
    // mutation, so this directly guards the second limb of
    // `ct == y_ring[j][lane 0]`.
    let mut fixture = non_trivial_fixture();
    assert!(!fixture.claim.ct.is_empty(), "test setup must have non-empty ct");
    let original = fixture.claim.ct[0];
    fixture.claim.ct[0] = original + k_c1_one();
    assert_eq!(
        fixture.claim.ct[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave the c0 limb unchanged"
    );
    assert_ne!(
        fixture.claim.ct[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change only the c1 limb"
    );

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted a c1-only ct tamper; both K limbs of ct must be \
         constrained to y_ring[j][lane 0]"
    );
}

/// **r-shape isolation.** Shrink `claim.r` by one element so it no
/// longer equals `log2(next_pow2(structure.n))` (the fixture has
/// `log_n = 2`). The y_ring gadget builds a `chi_r` tensor of `2^|r|`
/// leaves; an off-shape `r` would evaluate `M · Z` against the wrong-size
/// tensor, so the gadget must abort synthesis up front instead of
/// silently computing on a truncated point. Mirrors the native
/// `check_ce_relation` guard. Unlike the mutation tests above (which
/// produce an *unsatisfiable* builder), an off-shape `r` is a structural
/// error, so `enforce_ce_relations_against` returns `Err`.
#[test]
fn decider_ce_isolation_rejects_r_length_mismatch() {
    let mut fixture = non_trivial_fixture();
    assert_eq!(
        fixture.claim.r.len(),
        2,
        "non-trivial fixture has log_n = 2, so honest r ∈ K²"
    );
    fixture.claim.r.pop();

    // `enforce_ce_relations_against` returns `Result<R1csBuilder, String>`
    // and `R1csBuilder` isn't `Debug`, so reach the error via `.err()`.
    let err = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness)
        .err()
        .expect("off-shape r must abort CE-relation synthesis");
    assert!(
        err.contains("claim.r length"),
        "expected a `claim.r length` shape-mismatch error, got: {err}"
    );
}

/// **y_ring inner-length isolation.** Append one extra `K` element to a
/// claim's `y_ring[0]` so the flattened wire row is `(d_pad+1)·K_LIMBS`
/// long instead of the canonical `d_pad·K_LIMBS` (`d_pad =
/// D.next_power_of_two()`). `alloc_ce_claim` flattens the row faithfully,
/// so the extra limbs would land as *allocated but unconstrained* wires.
/// The native `check_ce_relation` rejects off-length rows via exact `Vec`
/// equality; the in-circuit gadget must match that contract by rejecting
/// the off-shape inner length up front — a `<`-only guard left the
/// trailing wires free. Structural error → `Err`, like the r-length test.
#[test]
fn decider_ce_isolation_rejects_y_ring_inner_length_mismatch() {
    let mut fixture = non_trivial_fixture();
    fixture.claim.y_ring[0].push(K::ONE);

    let err = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness)
        .err()
        .expect("off-length y_ring[0] must abort CE-relation synthesis");
    assert!(
        err.contains("y_ring[j] inner length"),
        "expected a `y_ring[j] inner length` shape-mismatch error, got: {err}"
    );
}

/// **y_ring padding-lane isolation.** Native `compute_y_from_Z_and_r`
/// returns the `D` real ring coefficients padded up to `d_pad =
/// D.next_power_of_two()` K-lanes with zeros. A `<`-only guard plus a
/// `0..D` eval loop left lanes `D..d_pad` unconstrained — even in honest
/// proofs. Set the first padding lane (index `D`) to a non-zero value:
/// commit / X / low-norm and every real eval lane still hold, so only the
/// padding-lane zero-binding can reject. The builder must be unsatisfied.
/// Catches a regression where the gadget stops constraining the padding
/// tail (re-opening the unconstrained-wire gap on the in-circuit boundary).
#[test]
fn decider_ce_isolation_rejects_nonzero_y_ring_padding_lane() {
    let mut fixture = non_trivial_fixture();
    assert!(
        fixture.claim.y_ring[0].len() > neo_math::D,
        "fixture y_ring must carry zero-padding lanes beyond the D real coefficients"
    );
    // Lane `D` is the first zero-pad lane; make it non-zero.
    fixture.claim.y_ring[0][neo_math::D] = K::ONE;

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted a non-zero y_ring padding lane; the padding lanes \
         (D..d_pad) must be bound to zero to match native exact Vec-equality"
    );
}

/// **NC-channel isolation.** `s_col/y_zcol` are not part of the paper CE
/// tuple, but this implementation carries them in the accumulator digest
/// and continuity wiring. If they are authoritative, the terminal
/// in-circuit closure must bind `y_zcol = Z · chi(s_col)`. Mutate only
/// `y_zcol`: commit / X / low-norm / y_ring / ct all remain valid, so only
/// the NC-channel row can reject.
#[test]
fn decider_ce_isolation_rejects_y_zcol_inconsistent_with_z_at_s_col() {
    let mut fixture = non_trivial_fixture();
    attach_nc_channel(&mut fixture);
    assert!(!fixture.claim.y_zcol.is_empty(), "fixture must carry y_zcol");

    let honest_output =
        enforce_ce_relations_with_wires_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    let honest = &honest_output.builder;
    assert!(
        honest.is_satisfied(),
        "honest non-trivial CE pair with NC channel must satisfy the gadget; first bad row: {:?}",
        honest.first_unsatisfied_row()
    );
    assert_only_fold_digest_unconstrained(
        honest,
        &honest_output.fold_digest_fields,
        "honest terminal CE pair with NC channel",
    );

    let original = fixture.claim.y_zcol[0];
    fixture.claim.y_zcol[0] = original + K::ONE;
    assert_ne!(fixture.claim.y_zcol[0], original, "mutation must change y_zcol[0]");

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted y_zcol inconsistent with Z · chi(s_col). Because y_zcol \
         is carried in the accumulator digest, it must be recomputed from the opened terminal Z \
         or treated as non-authoritative."
    );
}

#[test]
fn decider_ce_isolation_rejects_y_zcol_c1_limb_inconsistent_with_z_at_s_col() {
    // Mutate only the c1 limb of y_zcol. This is the NC-channel analogue
    // of the y_ring c1 test above: the sidecar must be a full K-valued
    // evaluation, not just a c0 scalar.
    let mut fixture = non_trivial_fixture();
    attach_nc_channel(&mut fixture);
    assert!(!fixture.claim.y_zcol.is_empty(), "fixture must carry y_zcol");

    let honest = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        honest.is_satisfied(),
        "honest non-trivial CE pair with NC channel must satisfy the gadget; first bad row: {:?}",
        honest.first_unsatisfied_row()
    );

    let original = fixture.claim.y_zcol[0];
    fixture.claim.y_zcol[0] = original + k_c1_one();
    assert_eq!(
        fixture.claim.y_zcol[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave the c0 limb unchanged"
    );
    assert_ne!(
        fixture.claim.y_zcol[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change only the c1 limb"
    );

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted a c1-only y_zcol tamper; both K limbs of the \
         NC-channel sidecar must be bound to Z·chi(s_col)"
    );
}

#[test]
fn decider_ce_isolation_rejects_s_col_c1_limb_inconsistent_with_y_zcol() {
    // Keep y_zcol honest for the original column point, then mutate only
    // the c1 limb of s_col. If the chi(s_col) tensor or its equality rows
    // accidentally bind only c0, the sidecar evaluation would accept a
    // relabelled K-valued point.
    let mut fixture = non_trivial_fixture();
    attach_nc_channel(&mut fixture);
    assert!(!fixture.claim.s_col.is_empty(), "fixture must carry s_col");

    let honest = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        honest.is_satisfied(),
        "honest non-trivial CE pair with NC channel must satisfy the gadget; first bad row: {:?}",
        honest.first_unsatisfied_row()
    );

    let original = fixture.claim.s_col[0];
    fixture.claim.s_col[0] = original + k_c1_one();
    assert_eq!(
        fixture.claim.s_col[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave the c0 limb unchanged"
    );
    assert_ne!(
        fixture.claim.s_col[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change only the c1 limb"
    );

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted a c1-only s_col tamper; both K limbs of the \
         NC-channel point must feed chi(s_col)"
    );
}

/// **NC-channel completeness isolation.** Carry `y_zcol` while deleting
/// the matching `s_col` point. Because `y_zcol` is accumulator-digested
/// sidecar data, the terminal CE closure must either recompute it from a
/// concrete column point or reject the claim structurally. A regression
/// that treats empty `s_col` as "no NC channel" would leave `y_zcol`
/// authoritative but unconstrained.
#[test]
fn decider_ce_isolation_rejects_y_zcol_without_s_col() {
    let mut fixture = non_trivial_fixture();
    attach_nc_channel(&mut fixture);
    assert!(!fixture.claim.y_zcol.is_empty(), "fixture must carry y_zcol");
    fixture.claim.s_col.clear();

    let err = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness)
        .err()
        .expect("one-sided NC channel must abort CE-relation synthesis");
    assert!(
        err.contains("incomplete NC channel"),
        "expected an `incomplete NC channel` shape-mismatch error, got: {err}"
    );
}

/// Same completeness invariant, opposite direction: carry an NC column
/// point without the evaluation sidecar. This prevents a future guard
/// from treating missing `y_zcol` as "nothing to bind" while still
/// letting `s_col` participate in accumulator-authority data.
#[test]
fn decider_ce_isolation_rejects_s_col_without_y_zcol() {
    let mut fixture = non_trivial_fixture();
    attach_nc_channel(&mut fixture);
    assert!(!fixture.claim.s_col.is_empty(), "fixture must carry s_col");
    fixture.claim.y_zcol.clear();

    let err = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness)
        .err()
        .expect("one-sided NC channel must abort CE-relation synthesis");
    assert!(
        err.contains("incomplete NC channel"),
        "expected an `incomplete NC channel` shape-mismatch error, got: {err}"
    );
}

/// **NC-channel shape isolation.** Drop one `s_col` challenge while
/// keeping `y_zcol` in the canonical shape. A short point changes the
/// `chi(s_col)` tensor domain, so the gadget must reject it structurally
/// instead of evaluating the sidecar against a truncated column point.
#[test]
fn decider_ce_isolation_rejects_s_col_length_mismatch() {
    let mut fixture = non_trivial_fixture();
    attach_nc_channel(&mut fixture);
    assert!(
        !fixture.claim.s_col.is_empty(),
        "fixture must carry a non-empty NC column point"
    );
    fixture.claim.s_col.pop();

    let err = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness)
        .err()
        .expect("off-shape s_col must abort CE-relation synthesis");
    assert!(
        err.contains("claim.s_col length"),
        "expected a `claim.s_col length` shape-mismatch error, got: {err}"
    );
}

/// **NC-channel y_zcol length isolation.** Append one extra K-lane to
/// `y_zcol`. `alloc_ce_claim` allocates the extra limbs faithfully; an
/// in-circuit closure that only checks the active prefix would leave the
/// tail as authoritative-but-unconstrained sidecar data.
#[test]
fn decider_ce_isolation_rejects_y_zcol_inner_length_mismatch() {
    let mut fixture = non_trivial_fixture();
    attach_nc_channel(&mut fixture);
    fixture.claim.y_zcol.push(K::ONE);

    let err = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness)
        .err()
        .expect("off-length y_zcol must abort CE-relation synthesis");
    assert!(
        err.contains("claim.y_zcol length"),
        "expected a `claim.y_zcol length` shape-mismatch error, got: {err}"
    );
}

/// **NC-channel padding-lane isolation.** `y_zcol` is padded from the
/// `D` real ring coefficients to `d_pad = D.next_power_of_two()` lanes,
/// and the padding lanes must be zero. Mutating only the first padding
/// lane leaves the real `Z · chi(s_col)` coefficients valid, so only the
/// padding zero rows can reject.
#[test]
fn decider_ce_isolation_rejects_nonzero_y_zcol_padding_lane() {
    let mut fixture = non_trivial_fixture();
    attach_nc_channel(&mut fixture);
    assert!(
        fixture.claim.y_zcol.len() > neo_math::D,
        "fixture y_zcol must carry zero-padding lanes beyond the D real coefficients"
    );
    fixture.claim.y_zcol[neo_math::D] = K::ONE;

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted a non-zero y_zcol padding lane; the NC sidecar \
         padding lanes (D..d_pad) must be bound to zero just like y_ring padding"
    );
}

/// **Low-norm isolation.** Build a `Z_bad` with one entry OUTSIDE the
/// alphabet `{-(b-1), …, +(b-1)}`. Recompute `c = Commit(Z_bad)`,
/// `X = L_in(Z_bad)`, and `y_ring = eval(M · Z_bad, r)` consistently
/// from that bad `Z` — so Ajtai opening, X projection, and y_ring
/// evaluation all PASS, and only the balanced-alphabet rows can
/// reject. Without the alphabet rows the gadget would accept this
/// witness, breaking the Ajtai-binding norm bound.
#[test]
fn decider_ce_isolation_rejects_out_of_alphabet_witness() {
    let mut fixture = non_trivial_fixture();
    let b = fixture.prep.params.b();
    // Put `b` (= b, just outside `{-(b-1), …, +(b-1)}`) at Z[0, 0].
    fixture.witness[(0, 0)] = F::from_u64(b as u64);
    // Recompute c / X / y_ring consistently from the bad Z so the
    // other four obligations all pass.
    let new_c = fixture.prep.log.commit(&fixture.witness);
    let new_x = project_x_from_witness_mat(&fixture.witness, fixture.prep.structure().m, fixture.claim.m_in)
        .expect("X projection");
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let (new_y_ring, new_ct) =
        compute_y_from_Z_and_r(fixture.prep.structure(), &fixture.witness, &fixture.claim.r, ell_d, b);
    fixture.claim.c = new_c;
    fixture.claim.X = new_x;
    fixture.claim.y_ring = new_y_ring;
    fixture.claim.ct = new_ct;

    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        !builder.is_satisfied(),
        "CE-relation gadget accepted a witness with an entry OUTSIDE the balanced-digit alphabet \
         while commit / X / y_ring all match the bad witness — the alphabet rows must be \
         load-bearing in isolation (and without them, the Ajtai norm bound is meaningless)"
    );
}

// ── Fixture helper ──────────────────────────────────────────────────────

/// Honest `(prep, claim, witness)` triple for the gadget tests.
/// Builds a one-step proof under the existing toy preprocessing,
/// finalizes it, and plucks the final running's first
/// `(claim, witness)` pair. The pair is what the terminal CE-relation
/// gadget receives in production.
fn honest_terminal_pair() -> (
    neo_fold_clean::Preprocessing,
    neo_fold_clean::CeClaim,
    neo_fold_clean::paper::relations::WitnessMat,
) {
    let prep = support::toy_preprocessing();
    let in_flight = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 1)]]).expect("prove");
    let finished = neo_fold_clean::finish_uncompressed(&prep, in_flight).expect("finish");
    let running = match finished.state.proof {
        ProofState::Active { running, .. } => running,
        ProofState::Initial => panic!("finished proof must be Active"),
    };
    let running = running
        .into_materialized()
        .expect("terminal pair running materialization");
    let claim = running
        .claims
        .first()
        .cloned()
        .expect("non-empty terminal running");
    let witness = running
        .witnesses
        .first()
        .cloned()
        .expect("witness paired with claim");
    (prep, claim, witness)
}

/// Non-trivial CE fixture: 4-row R1CS over m=16 variables.
///
/// - **R1CS**: 4 independent multiplication constraints `z[col_c] = z[col_a] * z[col_b]`
///   chosen so the assignment stays inside `{-1, 0, 1}` (the typical
///   SuperNeo alphabet for `b=2`). This gives a satisfying low-norm Z.
/// - **Structure**: `r1cs_to_ccs(A, B, C)` → CCS with 3 matrices and
///   `f(a, b, c) = a * b - c`. `n = 4`, `m = 16`, so `log_n = 2` and
///   `m / D = 2` blocks per packed witness column.
/// - **Claim**: built from `(c = Commit(Z), X = L_in(Z), r = random K²,
///   y_ring = eval(M · Z, r))` so the gadget accepts.
///
/// This is the only fixture in this file that genuinely exercises
/// (a) `chi_r` tensor unfolding for `log_n ≥ 2`, (b) multi-block `m`,
/// and (c) the low-norm rows on a witness whose other four
/// obligations all hold simultaneously.
struct NonTrivialFixture {
    prep: Preprocessing,
    claim: CeClaim,
    witness: neo_fold_clean::paper::relations::WitnessMat,
}

fn non_trivial_fixture() -> NonTrivialFixture {
    non_trivial_fixture_with_m_in(1)
}

fn non_trivial_nebula_fixture() -> NonTrivialFixture {
    let mut fixture = non_trivial_fixture_with_shape(3 * D, 1, Some(1));
    let scheme = LaneScheme::from_seeds(
        fixture.prep.params.kappa() as usize,
        LaneRanges {
            ops: 0..1,
            is: 1..2,
            fs: 2..3,
        },
        [0xA5; 32],
        [0x5A; 32],
    )
    .expect("test lane scheme");
    fixture.claim.adv = Some(scheme.commit(&fixture.witness).expect("lane commitments"));
    fixture.prep = fixture.prep.with_nebula(NebulaConfig {
        scheme,
        steps_per_segment: 1,
        seg_max: 1,
        stacks: StackShape::NONE,
        plan_digest: [F::ZERO; 4],
        d_init: [F::ZERO; 4],
    });
    fixture
}

fn non_trivial_fixture_with_m_in(m_in: usize) -> NonTrivialFixture {
    non_trivial_fixture_with_shape(2 * D, m_in, Some(m_in))
}

fn non_trivial_fixture_with_shape(m: usize, m_in: usize, public_input_len: Option<usize>) -> NonTrivialFixture {
    non_trivial_fixture_with_shape_and_last_row_base(m, m_in, public_input_len, None)
}

fn non_trivial_fixture_with_load_bearing_final_block(
    m: usize,
    m_in: usize,
    public_input_len: Option<usize>,
) -> NonTrivialFixture {
    non_trivial_fixture_with_shape_and_last_row_base(m, m_in, public_input_len, Some(m - 3))
}

fn non_trivial_fixture_with_shape_and_last_row_base(
    m: usize,
    m_in: usize,
    public_input_len: Option<usize>,
    last_row_base: Option<usize>,
) -> NonTrivialFixture {
    let n = 4;
    assert!(m >= n * 3, "non-trivial fixture needs three witness slots per row");

    // 4 independent multiplications, layout `[z_a₀, z_b₀, z_c₀, z_a₁, z_b₁, z_c₁, …]`.
    let mut a = Mat::zero(n, m, F::ZERO);
    let mut b = Mat::zero(n, m, F::ZERO);
    let mut c = Mat::zero(n, m, F::ZERO);
    for row in 0..n {
        let base = if row == n - 1 {
            last_row_base.unwrap_or(row * 3)
        } else {
            row * 3
        };
        a[(row, base)] = F::ONE;
        b[(row, base + 1)] = F::ONE;
        c[(row, base + 2)] = F::ONE;
    }
    let structure = r1cs_to_ccs(a, b, c);
    let params = config::r1cs_params(n, m).expect("non-trivial fixture params");
    install_ajtai_module_local(&params, &structure);
    let prep = preprocess(params, structure, public_input_len).expect("non-trivial preprocessing");

    // Satisfying assignment with all entries in `{-1, 0, 1}`:
    // `z[3k+2] = z[3k] * z[3k+1]` for `k = 0..4`. Index 12..16 stay
    // zero (padding to a multiple of D).
    let neg_one = F::ZERO - F::ONE;
    let row_values = [
        (F::ONE, F::ONE, F::ONE),
        (F::ONE, neg_one, neg_one),
        (neg_one, F::ONE, neg_one),
        (neg_one, neg_one, F::ONE),
    ];
    let mut z_pattern = vec![F::ZERO; m];
    for (row, (z_a, z_b, z_c)) in row_values.into_iter().enumerate() {
        let base = if row == n - 1 {
            last_row_base.unwrap_or(row * 3)
        } else {
            row * 3
        };
        z_pattern[base] = z_a;
        z_pattern[base + 1] = z_b;
        z_pattern[base + 2] = z_c;
    }

    // Pack into the SuperNeo D × ceil(m/D) layout: Z[c % D, c / D] = z[c].
    let cols = m.div_ceil(D);
    let mut witness = Mat::zero(D, cols, F::ZERO);
    for (col, value) in z_pattern.iter().enumerate() {
        witness[(col % D, col / D)] = *value;
    }

    let c_data = prep.log.commit(&witness);
    let X = if m_in <= prep.structure().m {
        project_x_from_witness_mat(&witness, prep.structure().m, m_in).expect("X projection")
    } else {
        project_x_from_witness_mat_lenient_for_overclaimed_padding(&witness, m_in)
    };
    // r ∈ K^{log_n} = K². Pick a non-trivial point to exercise the
    // tensor unfold meaningfully.
    let r: Vec<K> = vec![
        K::from_coeffs([F::from_u64(7), F::from_u64(11)]),
        K::from_coeffs([F::from_u64(13), F::from_u64(17)]),
    ];
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let (y_ring, ct) = compute_y_from_Z_and_r(prep.structure(), &witness, &r, ell_d, prep.params.b());

    let claim = CeClaim {
        adv: None,
        c: c_data,
        X,
        r,
        s_col: Vec::new(),
        y_ring,
        ct,
        aux_openings: Vec::new(),
        y_zcol: Vec::new(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    };

    NonTrivialFixture { prep, claim, witness }
}

fn project_x_from_witness_mat_lenient_for_overclaimed_padding(
    witness: &neo_fold_clean::paper::relations::WitnessMat,
    m_in: usize,
) -> Mat<F> {
    let required_cols = m_in.div_ceil(D);
    assert!(
        required_cols <= witness.cols(),
        "overclaimed-padding fixture must still fit the witness packing"
    );
    let mut X = Mat::zero(D, m_in, F::ZERO);
    for col in 0..required_cols {
        for row in 0..D {
            X[(row, col)] = witness[(row, col)];
        }
    }
    X
}

fn attach_nc_channel(fixture: &mut NonTrivialFixture) {
    let ell_m = fixture
        .prep
        .structure()
        .m
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    fixture.claim.s_col = (0..ell_m)
        .map(|i| K::from_coeffs([F::from_u64((19 + 2 * i) as u64), F::from_u64((23 + 2 * i) as u64)]))
        .collect();
    fixture.claim.y_zcol =
        compute_linear_y_zcol_for_fixture(fixture.prep.structure().m, &fixture.witness, &fixture.claim.s_col);
}

fn compute_linear_y_zcol_for_fixture(
    expected_m: usize,
    witness: &neo_fold_clean::paper::relations::WitnessMat,
    s_col: &[K],
) -> Vec<K> {
    let d_pad = D.next_power_of_two();
    let chi_s = tensor_point::<K>(s_col);
    let mut y_zcol = vec![K::ZERO; d_pad];
    for logical_col in 0..expected_m {
        let w = chi_s.get(logical_col).copied().unwrap_or(K::ZERO);
        if w == K::ZERO {
            continue;
        }
        let off = logical_col % D;
        let block = logical_col / D;
        y_zcol[off] += K::from(witness[(off, block)]) * w;
    }
    y_zcol
}

// ── Ajtai setup + mixers (copy of the toy support fixture).
// Kept local so the non-trivial fixture's m=16 setup can register a
// separate (D, cols=2) Ajtai global PP without colliding with the
// toy support's (D, cols=1).

fn install_ajtai_module_local(params: &Params, structure: &Structure) {
    let cols = structure.m.div_ceil(D);
    if !has_global_pp_for_dims(D, cols) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x4e45_4f46_4f4c_4432_u64.to_le_bytes());
        match set_global_pp_seeded(D, params.kappa() as usize, cols, seed) {
            Ok(()) => {}
            Err(_err) if has_global_pp_for_dims(D, cols) => {}
            Err(err) => panic!("Ajtai global setup: {err}"),
        }
    }
}

fn expect_unsupported_sidecar<FN>(field: &'static str, mutate: FN)
where
    FN: FnOnce(&mut CeClaim),
{
    let mut fixture = non_trivial_fixture();
    mutate(&mut fixture.claim);
    let err = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness)
        .err()
        .unwrap_or_else(|| panic!("unsupported sidecar {field} must abort CE-relation synthesis"));
    assert!(
        err.contains(field),
        "expected a `{field}` shape-mismatch error, got: {err}"
    );
}
