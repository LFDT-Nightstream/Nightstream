//! Isolated coverage for the terminal CE-relation gadget.
//!
//! Each test hits one obligation (`commit / X / low-norm / y_ring / ct`)
//! by mutating exactly the data that feeds that gate, holding the
//! other four valid. Replaces the older end-to-end "tamper Z, bypass
//! preflight" test, which couldn't isolate which obligation was
//! load-bearing.
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

use neo_ajtai::{has_global_pp_for_dims, s_mul_add, scale_commitment_add_inplace, set_global_pp_seeded, Commitment};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{r1cs_to_ccs, Mat};
use neo_fold_clean::config;
use neo_fold_clean::engine::decider::__test_isolation::enforce_ce_relations_against;
use neo_fold_clean::paper::construction2::ProofState;
use neo_fold_clean::{preprocess, CeClaim, DecMixer, Params, Preprocessing, RlcMixer, Structure};
use neo_math::ring::{cf_inv, Rq as RqEl};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::common::{compute_y_from_Z_and_r, project_x_from_witness_mat};
use p3_field::PrimeCharacteristicRing;

/// Build an honest finished proof, pluck its final running's first
/// `(claim, witness)` pair, and assert the CE-relation gadget alone
/// (no chain replay, no public-image pins) accepts it. Sanity baseline
/// for the tampering tests below.
#[test]
fn decider_ce_isolation_accepts_honest_terminal_pair() {
    let (prep, claim, witness) = honest_terminal_pair();
    let builder = enforce_ce_relations_against(&prep, &claim, &witness).expect("synthesis");
    assert!(
        builder.is_satisfied(),
        "honest terminal pair must satisfy the CE-relation gadget alone; \
         first unsatisfied row: {:?}",
        builder.first_unsatisfied_row()
    );
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
    if required_cols == 0 {
        // Toy structures with m_in = 0 have no active X columns; skip
        // the X-isolation check for those (commit / y_ring still cover
        // the witness binding).
        return;
    }
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

/// **inactive-X isolation.** Native `project_x_from_witness_mat`
/// returns a `D × m_in` matrix and leaves packed columns
/// `ceil(m_in / D)..m_in` as structural zeros. The terminal CE gadget
/// must enforce those inactive columns locally rather than relying on
/// upstream callers. Here we keep the active column honest, add one
/// inactive X column, and make it non-zero. A gadget that checks only
/// the active prefix accepts this; the fixed gadget rejects.
#[test]
fn decider_ce_isolation_rejects_inactive_x_not_zero() {
    let mut fixture = non_trivial_fixture();
    let mut x = Mat::zero(D, 2, F::ZERO);
    for row in 0..D {
        x[(row, 0)] = fixture.claim.X[(row, 0)];
    }
    x[(0, 1)] = F::ONE;
    fixture.claim.X = x;
    fixture.claim.m_in = 2;

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
    let builder = enforce_ce_relations_against(&fixture.prep, &fixture.claim, &fixture.witness).expect("synthesis");
    assert!(
        builder.is_satisfied(),
        "honest non-trivial (n=4, m=16, log_n=2) CE pair must satisfy the gadget; \
         first unsatisfied row: {:?}",
        builder.first_unsatisfied_row()
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
    let (new_y_ring, _ct) =
        compute_y_from_Z_and_r(fixture.prep.structure(), &fixture.witness, &fixture.claim.r, ell_d, b);
    fixture.claim.c = new_c;
    fixture.claim.X = new_x;
    fixture.claim.y_ring = new_y_ring;

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
    let m = 2 * D; // = 16 under Goldilocks D=8 → ceil(m/D) = 2 blocks.
    let n = 4;
    let m_in = 1;

    // 4 independent multiplications, layout `[z_a₀, z_b₀, z_c₀, z_a₁, z_b₁, z_c₁, …]`.
    let mut a = Mat::zero(n, m, F::ZERO);
    let mut b = Mat::zero(n, m, F::ZERO);
    let mut c = Mat::zero(n, m, F::ZERO);
    for row in 0..n {
        let base = row * 3;
        a[(row, base)] = F::ONE;
        b[(row, base + 1)] = F::ONE;
        c[(row, base + 2)] = F::ONE;
    }
    let structure = r1cs_to_ccs(a, b, c);
    let params = config::r1cs_params(n, m).expect("non-trivial fixture params");
    install_ajtai_module_local(&params, &structure);
    let prep = preprocess(
        params,
        structure,
        mix_rhos_commits as RlcMixer,
        combine_b_pows as DecMixer,
        Some(m_in),
    )
    .expect("non-trivial preprocessing");

    // Satisfying assignment with all entries in `{-1, 0, 1}`:
    // `z[3k+2] = z[3k] * z[3k+1]` for `k = 0..4`. Index 12..16 stay
    // zero (padding to a multiple of D).
    let neg_one = F::ZERO - F::ONE;
    let z: [F; 16] = [
        F::ONE,
        F::ONE,
        F::ONE, // 1 * 1 = 1
        F::ONE,
        neg_one,
        neg_one, // 1 * -1 = -1
        neg_one,
        F::ONE,
        neg_one, // -1 * 1 = -1
        neg_one,
        neg_one,
        F::ONE, // -1 * -1 = 1
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    ];

    // Pack into the SuperNeo D × ceil(m/D) layout: Z[c % D, c / D] = z[c].
    let cols = m.div_ceil(D);
    let mut witness = Mat::zero(D, cols, F::ZERO);
    for (col, value) in z.iter().enumerate() {
        witness[(col % D, col / D)] = *value;
    }

    let c_data = prep.log.commit(&witness);
    let X = project_x_from_witness_mat(&witness, prep.structure().m, m_in).expect("X projection");
    // r ∈ K^{log_n} = K². Pick a non-trivial point to exercise the
    // tensor unfold meaningfully.
    let r: Vec<K> = vec![
        K::from_coeffs([F::from_u64(7), F::from_u64(11)]),
        K::from_coeffs([F::from_u64(13), F::from_u64(17)]),
    ];
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let (y_ring, ct) = compute_y_from_Z_and_r(prep.structure(), &witness, &r, ell_d, prep.params.b());

    let claim = CeClaim {
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

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Commitment]) -> Commitment {
    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    for (rho, c) in rhos.iter().zip(cs.iter()) {
        let rq = rot_matrix_to_rq(rho);
        s_mul_add(&mut acc, &rq, c);
    }
    acc
}

fn combine_b_pows(cs: &[Commitment], b: u32) -> Commitment {
    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    let base = F::from_u64(b as u64);
    let mut pow = F::ONE;
    for c in cs {
        scale_commitment_add_inplace(&mut acc, pow, c);
        pow *= base;
    }
    acc
}
