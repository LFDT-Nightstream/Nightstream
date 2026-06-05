//! SplitNcV1 — Π_CCS.V composition: hard-gate parity test.
//!
//! This is the gating test for sub-step J: a real native `pi_ccs::prove`
//! proof must satisfy the composed in-circuit SplitNc Π_CCS.V verifier
//! (`enforce_split_nc_pi_ccs_v`), and targeted mutations of the proof must
//! cause `R1csBuilder::is_satisfied()` to return false.
//!
//! Without this, all the FE/NC/digest/transcript-binding sub-gadget parity
//! tests only prove the *pieces* work; the *composition* is unverified.
//!
//! Tests:
//! - `split_nc_pi_ccs_v_accepts_native_proof`
//! - `split_nc_pi_ccs_v_rejects_tampered_fe_round`
//! - `split_nc_pi_ccs_v_rejects_tampered_output_commitment`
//! - `split_nc_pi_ccs_v_rejects_self_consistent_fresh_output_commitment_forgery`
//! - `split_nc_pi_ccs_v_rejects_tampered_output_active_x`
//! - `split_nc_pi_ccs_v_rejects_self_consistent_fresh_output_x_forgery`
//! - `split_nc_pi_ccs_v_rejects_tampered_output_inactive_x`
//! - `split_nc_pi_ccs_v_rejects_tampered_output_r`
//! - `split_nc_pi_ccs_v_rejects_tampered_output_r_c1_limb`
//! - `split_nc_pi_ccs_v_rejects_tampered_output_s_col`
//! - `split_nc_pi_ccs_v_rejects_tampered_running_output_y_ring_non_ct_lane`
//! - `split_nc_pi_ccs_v_rejects_tampered_fresh_output_y_ring_padding_lane`
//! - `split_nc_pi_ccs_v_rejects_tampered_output_ct`
//! - `split_nc_pi_ccs_v_rejects_tampered_parent_authority_r`
//! - `split_nc_pi_ccs_v_rejects_tampered_parent_authority_y_ring`
//! - `split_nc_pi_ccs_v_rejects_tampered_parent_authority_y_ring_c1_limb`
//! - `split_nc_pi_ccs_v_rejects_tampered_parent_authority_ct`
//! - `split_nc_pi_ccs_v_rejects_tampered_parent_authority_s_col`
//! - `split_nc_pi_ccs_v_accepts_tampered_parent_authority_y_zcol_non_authority`
//! - `split_nc_pi_ccs_v_accepts_tampered_parent_authority_y_zcol_c1_limb_non_authority`
//! - `split_nc_pi_ccs_v_rejects_tampered_parent_authority_fold_digest`
//! - `split_nc_pi_ccs_v_rejects_tampered_nc_y_zcol`
//! - `split_nc_pi_ccs_v_rejects_tampered_nc_y_zcol_c1_limb`
//! - `split_nc_pi_ccs_v_rejects_tampered_header_digest`
//! - `split_nc_pi_ccs_v_rejects_output_m_in_mismatch`
//! - `split_nc_pi_ccs_v_rejects_nonempty_running_without_parent_authority`
//! - `split_nc_pi_ccs_v_rejects_empty_running_with_parent_authority`

#![allow(non_snake_case)]

use std::collections::BTreeSet;

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, TranscriptGadget};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::reductions::pi_ccs;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::{
    enforce_split_nc_pi_ccs_v, Error, SplitNcPiCcsVConfig, SplitNcPiCcsVDerived, SplitNcPiCcsVMessages,
};
use neo_fold_clean::paper::relations::{superneo_public_x_cols, CcsClaim};
use neo_math::ring::D;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

// The paper-layer `Transcript::session()` initializes its inner sponge with
// `b"neo.fold.clean/session/v1"`. The in-circuit `TranscriptGadget` must use
// the same label so the prove- and verify-side sponge states stay in sync.
const SESSION_LABEL: &[u8] = b"neo.fold.clean/session/v1";

fn k_c1_one() -> K {
    K::from_coeffs([F::ZERO, F::ONE])
}

fn running_y_zcol_sidecar_columns(derived: &SplitNcPiCcsVDerived) -> BTreeSet<usize> {
    let mut allowed: BTreeSet<usize> = derived
        .running
        .iter()
        .flat_map(|running| running.y_zcol.iter())
        .flat_map(|k| [k.c0.col(), k.c1.col()])
        .collect();
    if let Some(parent) = &derived.running_parent_authority {
        allowed.extend(parent.y_zcol.iter().flat_map(|k| [k.c0.col(), k.c1.col()]));
    }
    allowed
}

fn fresh_output_deferred_ce_columns(derived: &SplitNcPiCcsVDerived) -> BTreeSet<usize> {
    let mut allowed = BTreeSet::new();
    let k_mcs = derived.fresh_x.len();

    for output in derived.outputs.iter().take(k_mcs) {
        // Native `validate_mcs_output_x_recomposition` pins only scalar
        // public lanes `x[c]` at `(row=c%D, col=c/D)`. The remaining lanes
        // in the active packed ring columns are L_in sidecars carried to the
        // terminal CE relation, where X is bound to the opened Z.
        let active_x_cols = superneo_public_x_cols(output.m_in);
        for col in 0..active_x_cols.min(output.x_cols) {
            for row in 0..output.x_rows {
                if col * D + row >= output.m_in {
                    allowed.insert(output.x[row * output.x_cols + col].col());
                }
            }
        }

        // For fresh CCS outputs, Π_CCS.V's FE terminal identity consumes
        // only `ct(y_j) = y_ring[j][0]` through the CCS polynomial
        // `f(ct(y_0), ..., ct(y_{t-1}))`. Non-ct ring coefficients are not
        // immediate Π_CCS verifier authority; they are carried forward and
        // bound by the terminal CE relation's `y_ring = M·Z(r)` check.
        for row in &output.y_ring {
            for lane in row.iter().take(D).skip(1) {
                insert_kvar_columns(&mut allowed, *lane);
            }
        }
    }

    allowed
}

fn insert_kvar_columns(cols: &mut BTreeSet<usize>, v: neo_fold_clean::engine::r1cs_circuit::field_ext::KVar) {
    cols.insert(v.c0.col());
    cols.insert(v.c1.col());
}

fn insert_output_columns(
    cols: &mut BTreeSet<usize>,
    wires: &neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsOutputWires,
) {
    cols.insert(wires.c_d_var.col());
    cols.insert(wires.c_kappa_var.col());
    cols.extend(wires.c_data.iter().map(|v| v.col()));
    cols.extend(wires.x.iter().map(|v| v.col()));
    cols.insert(wires.x_rows_var.col());
    cols.insert(wires.x_cols_var.col());
    cols.insert(wires.m_in_var.col());
    for v in &wires.r {
        insert_kvar_columns(cols, *v);
    }
    for v in &wires.s_col {
        insert_kvar_columns(cols, *v);
    }
    for row in &wires.y_ring {
        for v in row {
            insert_kvar_columns(cols, *v);
        }
    }
    for v in &wires.ct {
        insert_kvar_columns(cols, *v);
    }
    for v in &wires.y_zcol {
        insert_kvar_columns(cols, *v);
    }
    cols.extend(wires.fold_digest_fields.iter().map(|v| v.col()));
}

fn output_column_categories(
    prefix: &str,
    wires: &neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsOutputWires,
) -> Vec<(String, BTreeSet<usize>)> {
    let mut categories = Vec::new();

    let mut shape = BTreeSet::new();
    shape.insert(wires.c_d_var.col());
    shape.insert(wires.c_kappa_var.col());
    shape.insert(wires.x_rows_var.col());
    shape.insert(wires.x_cols_var.col());
    shape.insert(wires.m_in_var.col());
    categories.push((format!("{prefix}.shape"), shape));

    let mut c_data = BTreeSet::new();
    c_data.extend(wires.c_data.iter().map(|v| v.col()));
    categories.push((format!("{prefix}.c_data"), c_data));

    let mut x = BTreeSet::new();
    x.extend(wires.x.iter().map(|v| v.col()));
    categories.push((format!("{prefix}.x"), x));

    let mut r = BTreeSet::new();
    for v in &wires.r {
        insert_kvar_columns(&mut r, *v);
    }
    categories.push((format!("{prefix}.r"), r));

    let mut s_col = BTreeSet::new();
    for v in &wires.s_col {
        insert_kvar_columns(&mut s_col, *v);
    }
    categories.push((format!("{prefix}.s_col"), s_col));

    let mut y_ring = BTreeSet::new();
    for row in &wires.y_ring {
        for v in row {
            insert_kvar_columns(&mut y_ring, *v);
        }
    }
    categories.push((format!("{prefix}.y_ring"), y_ring));

    let mut ct = BTreeSet::new();
    for v in &wires.ct {
        insert_kvar_columns(&mut ct, *v);
    }
    categories.push((format!("{prefix}.ct"), ct));

    let mut y_zcol = BTreeSet::new();
    for v in &wires.y_zcol {
        insert_kvar_columns(&mut y_zcol, *v);
    }
    categories.push((format!("{prefix}.y_zcol"), y_zcol));

    let mut fold_digest = BTreeSet::new();
    fold_digest.extend(wires.fold_digest_fields.iter().map(|v| v.col()));
    categories.push((format!("{prefix}.fold_digest"), fold_digest));

    categories
}

fn split_nc_unconstrained_column_summary(derived: &SplitNcPiCcsVDerived, unexpected: &BTreeSet<usize>) -> String {
    let mut categories: Vec<(&str, BTreeSet<usize>)> = Vec::new();

    let mut r_prime = BTreeSet::new();
    for v in &derived.r_prime {
        insert_kvar_columns(&mut r_prime, *v);
    }
    categories.push(("r_prime", r_prime));

    let mut s_col_prime = BTreeSet::new();
    for v in &derived.s_col_prime {
        insert_kvar_columns(&mut s_col_prime, *v);
    }
    categories.push(("s_col_prime", s_col_prime));

    let mut running = BTreeSet::new();
    for r in &derived.running {
        insert_output_columns(&mut running, r);
    }
    categories.push(("running", running));

    let mut parent = BTreeSet::new();
    if let Some(p) = &derived.running_parent_authority {
        insert_output_columns(&mut parent, p);
    }
    categories.push(("running_parent_authority", parent));

    let mut fresh_x = BTreeSet::new();
    for row in &derived.fresh_x {
        fresh_x.extend(row.iter().map(|v| v.col()));
    }
    categories.push(("fresh_x", fresh_x));

    let mut running_c_data = BTreeSet::new();
    for row in &derived.running_c_data {
        running_c_data.extend(row.iter().map(|v| v.col()));
    }
    categories.push(("running_c_data", running_c_data));

    let mut running_acc_digest = BTreeSet::new();
    running_acc_digest.extend(derived.running_acc_digest.iter().map(|v| v.col()));
    categories.push(("running_acc_digest", running_acc_digest));

    let mut out = Vec::new();
    for (name, cols) in categories {
        let hit: Vec<_> = unexpected.intersection(&cols).copied().collect();
        if !hit.is_empty() {
            out.push(format!("{name}: {hit:?}"));
        }
    }
    for (idx, output) in derived.outputs.iter().enumerate() {
        for (name, cols) in output_column_categories(&format!("outputs[{idx}]"), output) {
            let hit: Vec<_> = unexpected.intersection(&cols).copied().collect();
            if !hit.is_empty() {
                out.push(format!("{name}: {hit:?}"));
            }
        }
    }
    out.join("; ")
}

// ── R1CS fixture: z[0]·(z[1] + z[2]) = z[3] (three-term addition) ────────

/// One-constraint R1CS: `(z[1] + z[2]) · z[0] = z[3]`. With `z[0] = 1` this
/// degenerates into `z[1] + z[2] = z[3]`, satisfied by any `(a, b, a+b)`.
fn three_term_addition() -> R1cs {
    let m = D;
    let mut a = Mat::zero(1, m, F::ZERO);
    a.set(0, 1, F::ONE);
    a.set(0, 2, F::ONE);

    let mut b = Mat::zero(1, m, F::ZERO);
    b.set(0, 0, F::ONE);

    let mut c = Mat::zero(1, m, F::ZERO);
    c.set(0, 3, F::ONE);

    R1cs { a, b, c, m_in: 3 }
}

/// Satisfying assignment `z = [1, a, b, a+b, 0, ..., 0]`.
fn assignment(a: u64, b: u64) -> Vec<F> {
    let mut z = vec![F::ZERO; D];
    z[0] = F::ONE;
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[3] = F::from_u64(a + b);
    z
}

// ── Fixture: native NIFS step + standalone Π_CCS proof ────────────────────

/// Test fixture. `running` is the running accumulator after one NIFS fold,
/// and `proof` is a fresh `pi_ccs::prove` output that the in-circuit verifier
/// must accept.
struct Fixture {
    prep: neo_fold_clean::Preprocessing,
    fresh_claims: Vec<CcsClaim>,
    running: RunningInstance,
    proof: pi_ccs::Proof,
}

fn build_fixture() -> Fixture {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");

    // Step 1: seed the running accumulator with one NIFS fold so that the
    // second Π_CCS proof has a non-empty `running`. Without this step the
    // verifier path skips the eq(α', α)·eq(r', r_in)·γ^k_total·eval_sum
    // branch in the FE terminal identity, which we want to exercise.
    let first = direct_ccs::build_instance(&prep, &r1cs, &assignment(1, 0)).expect("first instance");
    let mut first_tr = Transcript::session();
    let (running, _) = neo_fold_clean::paper::nifs::prove(
        &mut first_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![first],
        &RunningInstance::default(),
    )
    .expect("first NIFS.P");

    // Step 2: standalone Π_CCS proof. This is what the in-circuit verifier
    // mirrors. Uses a fresh session transcript with the same label as the
    // in-circuit `TranscriptGadget::new(...)` will use.
    let second = direct_ccs::build_instance(&prep, &r1cs, &assignment(0, 1)).expect("second instance");
    let fresh_claims = vec![second.claim.clone()];

    let mut tr = Transcript::session();
    let proof = pi_ccs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        vec![second],
        &running,
    )
    .expect("pi_ccs.prove");

    Fixture {
        prep,
        fresh_claims,
        running,
        proof,
    }
}

// ── Verifier driver: emit the SplitNc Π_CCS.V circuit on the fixture ─────

/// Build the `SplitNcPiCcsVConfig` for this fixture by recomputing the
/// engine's dims + matrix digest + header bundle from the same params and
/// structure the prover used. Mirrors the native verifier wrapper exactly.
fn split_nc_config<'a>(prep: &'a neo_fold_clean::Preprocessing) -> SplitNcPiCcsVConfig<'a> {
    // The paper-layer `Params` keeps its `NeoParams` private; reconstruct
    // it from the same shape the production `r1cs_params` derives.
    let raw_params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs_with(
        prep.structure().n.max(prep.structure().m),
        neo_fold_clean::config::MIN_EFFECTIVE_LAMBDA,
        neo_fold_clean::config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("raw params reconstruction");

    let dims =
        neo_reductions::engines::utils::build_dims_and_policy(&raw_params, prep.structure()).expect("engine dims");
    let mat_digest = neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(prep.structure(), None);
    let header_bundle = neo_reductions::engines::utils::pi_ccs_header_bundle_digest_fields(
        &raw_params,
        prep.structure(),
        dims,
        &mat_digest,
    )
    .expect("header bundle digest");

    SplitNcPiCcsVConfig {
        params: &prep.params,
        structure: prep.structure(),
        header_bundle,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
    }
}

/// Build a fresh R1cs circuit, allocate the SplitNc Π_CCS.V composition for
/// the fixture's proof, and return the populated `R1csBuilder`. Caller
/// inspects `builder.is_satisfied()`.
fn emit_verifier(f: &Fixture) -> Result<R1csBuilder, Error> {
    emit_verifier_with_derived(f).map(|(builder, _derived)| builder)
}

fn emit_verifier_with_derived(f: &Fixture) -> Result<(R1csBuilder, SplitNcPiCcsVDerived), Error> {
    let mut builder = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut builder, SESSION_LABEL);
    let cfg = split_nc_config(&f.prep);

    let derived = enforce_split_nc_pi_ccs_v(
        &mut builder,
        &mut tr,
        &cfg,
        &SplitNcPiCcsVMessages {
            fresh: &f.fresh_claims,
            running: &f.running.claims,
            running_parent_authority: f.running.parent_authority.as_ref(),
            outputs: &f.proof.outputs,
            sumcheck_rounds_fe: &f.proof.sumcheck.sumcheck_rounds,
            sumcheck_rounds_nc: &f.proof.sumcheck.sumcheck_rounds_nc,
            header_digest: &f.proof.sumcheck.header_digest,
        },
    )?;
    Ok((builder, derived))
}

fn prove_pi_ccs_from_empty_running(
    prep: &neo_fold_clean::Preprocessing,
    r1cs: &R1cs,
) -> (Vec<CcsClaim>, pi_ccs::Proof) {
    let fresh = direct_ccs::build_instance(prep, r1cs, &assignment(0, 1)).expect("empty-running fresh instance");
    let fresh_claims = vec![fresh.claim.clone()];
    let mut tr = Transcript::session();
    let proof = pi_ccs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        vec![fresh],
        &RunningInstance::default(),
    )
    .expect("empty-running pi_ccs.prove");
    (fresh_claims, proof)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn split_nc_pi_ccs_v_accepts_native_proof() {
    let fixture = build_fixture();
    let builder = emit_verifier(&fixture).expect("emit verifier");

    assert!(
        builder.is_satisfied(),
        "native pi_ccs::prove proof must satisfy SplitNc Π_CCS.V circuit; first bad row {:?}",
        builder.first_unsatisfied_row()
    );
}

#[test]
fn split_nc_pi_ccs_v_leaves_only_documented_deferred_ce_sidecars_unconstrained() {
    // SplitNc owns the full Π_CCS verifier for fresh/output authority and
    // the running accumulator handle used by HyperNova's recursive link.
    // The intentionally floating columns are narrow and paper-grounded:
    // running-side y_zcol is a Π_DEC sidecar, and fresh-output CE lanes that
    // Π_CCS.V only carries to the terminal CE relation (packed X sidecars and
    // non-ct y_ring coefficients) are not consumed by this standalone
    // verifier. Output y_zcol is not exempt; it is consumed by the NC
    // terminal identity.
    let fixture = build_fixture();
    let (builder, derived) = emit_verifier_with_derived(&fixture).expect("emit verifier");

    assert!(
        builder.is_satisfied(),
        "native pi_ccs::prove proof must satisfy before unconstrained-column audit"
    );

    let unconstrained: BTreeSet<_> = builder.unconstrained_columns().into_iter().collect();
    let mut allowed = running_y_zcol_sidecar_columns(&derived);
    allowed.extend(fresh_output_deferred_ce_columns(&derived));
    let unexpected: BTreeSet<_> = unconstrained.difference(&allowed).copied().collect();
    let summary = split_nc_unconstrained_column_summary(&derived, &unexpected);
    assert!(
        unconstrained == allowed,
        "SplitNc Π_CCS.V left unexpected unconstrained columns: got {unconstrained:?}, \
         expected exactly documented deferred CE sidecars {allowed:?}; unexpected categories: {summary}"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_nonempty_running_without_parent_authority() {
    let fixture = build_fixture();
    let mut builder = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut builder, SESSION_LABEL);
    let cfg = split_nc_config(&fixture.prep);

    let err = enforce_split_nc_pi_ccs_v(
        &mut builder,
        &mut tr,
        &cfg,
        &SplitNcPiCcsVMessages {
            fresh: &fixture.fresh_claims,
            running: &fixture.running.claims,
            running_parent_authority: None,
            outputs: &fixture.proof.outputs,
            sumcheck_rounds_fe: &fixture.proof.sumcheck.sumcheck_rounds,
            sumcheck_rounds_nc: &fixture.proof.sumcheck.sumcheck_rounds_nc,
            header_digest: &fixture.proof.sumcheck.header_digest,
        },
    )
    .err()
    .expect("non-empty running accumulator must carry its Pi_RLC parent authority");

    assert!(
        err.to_string()
            .contains("non-empty running accumulator missing Pi_RLC parent authority"),
        "expected missing-parent-authority shape error, got {err}"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_empty_running_with_parent_authority() {
    let fixture = build_fixture();
    let r1cs = three_term_addition();
    let (fresh_claims, proof) = prove_pi_ccs_from_empty_running(&fixture.prep, &r1cs);
    let parent = fixture
        .running
        .parent_authority
        .as_ref()
        .expect("fixture running must carry parent authority");

    let mut builder = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut builder, SESSION_LABEL);
    let cfg = split_nc_config(&fixture.prep);

    let err = enforce_split_nc_pi_ccs_v(
        &mut builder,
        &mut tr,
        &cfg,
        &SplitNcPiCcsVMessages {
            fresh: &fresh_claims,
            running: &[],
            running_parent_authority: Some(parent),
            outputs: &proof.outputs,
            sumcheck_rounds_fe: &proof.sumcheck.sumcheck_rounds,
            sumcheck_rounds_nc: &proof.sumcheck.sumcheck_rounds_nc,
            header_digest: &proof.sumcheck.header_digest,
        },
    )
    .err()
    .expect("empty running accumulator must not carry a parent authority");

    assert!(
        err.to_string()
            .contains("running parent authority present while running is empty"),
        "expected unexpected-parent-authority shape error, got {err}"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_fe_round() {
    let mut fixture = build_fixture();
    // Bump the leading coeff of the first FE round. The chain identity
    // `g(0) + g(1) == claim_q` and downstream sumcheck challenges diverge.
    fixture.proof.sumcheck.sumcheck_rounds[0][0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered FE round must be rejected");
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_output_commitment() {
    // Output commitments are not just transcript decoration: for fresh
    // outputs they must equal the fresh CCS claim commitment wire-to-wire,
    // and later Π_RLC folds them into the next CE parent.
    let mut fixture = build_fixture();
    assert!(
        !fixture.proof.outputs[0].c.data.is_empty(),
        "fixture output must have c.data"
    );
    fixture.proof.outputs[0].c.data[0] += F::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered SplitNc output commitment must be rejected"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_self_consistent_fresh_output_commitment_forgery() {
    // Sneakier than a one-sided output tamper: mutate the fresh CCS
    // commitment and the corresponding SplitNc output commitment together.
    // That preserves the output→fresh wire equality. The rejection must
    // come from the recomputed public-instance digest absorbed into the
    // Π_CCS transcript; otherwise fresh.c would be transcript-free
    // authority.
    let mut fixture = build_fixture();
    assert!(
        !fixture.fresh_claims[0].c.data.is_empty() && !fixture.proof.outputs[0].c.data.is_empty(),
        "fixture must expose fresh/output commitment lanes"
    );
    fixture.fresh_claims[0].c.data[0] += F::ONE;
    fixture.proof.outputs[0].c.data[0] += F::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "SplitNc Π_CCS.V accepted a self-consistent forged fresh/output commitment; \
         fresh.c must be bound through the instance digest transcript"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_output_active_x() {
    // Fresh output X encodes the fresh CCS public input. Mutating an
    // active packed lane must break the output→fresh binding.
    let mut fixture = build_fixture();
    assert!(fixture.proof.outputs[0].X.rows() > 0 && fixture.proof.outputs[0].X.cols() > 0);
    let old = fixture.proof.outputs[0].X[(0, 0)];
    fixture.proof.outputs[0].X.set(0, 0, old + F::ONE);

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered SplitNc output X must be rejected");
}

#[test]
fn split_nc_pi_ccs_v_rejects_self_consistent_fresh_output_x_forgery() {
    // Preserve the output→fresh X equality by mutating the fresh CCS
    // public input and the matching packed output X lane together. The
    // only load-bearing rejection path left is the recomputed fresh CCS
    // digest inside the Π_CCS transcript. This guards against treating
    // fresh.x as wire-equality-only metadata.
    let mut fixture = build_fixture();
    assert!(
        !fixture.fresh_claims[0].x.is_empty() && fixture.proof.outputs[0].X.rows() > 0,
        "fixture must expose fresh/output public input lanes"
    );
    fixture.fresh_claims[0].x[0] += F::ONE;
    let old = fixture.proof.outputs[0].X[(0, 0)];
    fixture.proof.outputs[0].X.set(0, 0, old + F::ONE);

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "SplitNc Π_CCS.V accepted a self-consistent forged fresh/output X lane; \
         fresh.x must be bound through the instance digest transcript"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_output_inactive_x() {
    // Native Π_CCS.V rejects output CE claims whose structural inactive X
    // columns are non-zero. The circuit digest also hashes only active X
    // columns, so the standalone SplitNc verifier must pin inactive output
    // columns to zero instead of relying on a later Π_RLC consumer.
    let mut fixture = build_fixture();
    let active_cols = superneo_public_x_cols(fixture.proof.outputs[0].m_in);
    assert!(
        active_cols < fixture.proof.outputs[0].X.cols(),
        "fixture must expose an inactive output X column"
    );
    fixture.proof.outputs[0].X.set(0, active_cols, F::ONE);

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "SplitNc Π_CCS.V accepted a non-zero inactive output X column"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_output_r() {
    // Every output CE claim must carry the FE sumcheck's r_prime. If this
    // point is unbound, later CE evaluations can be labelled at the wrong
    // point.
    let mut fixture = build_fixture();
    assert!(!fixture.proof.outputs[0].r.is_empty(), "fixture output must have r");
    fixture.proof.outputs[0].r[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered SplitNc output r must be rejected");
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_output_r_c1_limb() {
    // The FE output point is K-valued. This catches a verifier that pins
    // only the c0 limb of r_prime into output.r.
    let mut fixture = build_fixture();
    assert!(!fixture.proof.outputs[0].r.is_empty(), "fixture output must have r");
    let original = fixture.proof.outputs[0].r[0];
    fixture.proof.outputs[0].r[0] = original + k_c1_one();
    assert_eq!(
        fixture.proof.outputs[0].r[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        fixture.proof.outputs[0].r[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "SplitNc Π_CCS.V accepted a c1-only output.r tamper"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_output_s_col() {
    // Every output CE claim must carry the NC sumcheck's s_col_prime.
    // Leaving it free would disconnect the NC terminal identity from the
    // claim passed to Π_RLC/Π_DEC.
    let mut fixture = build_fixture();
    assert!(
        !fixture.proof.outputs[0].s_col.is_empty(),
        "fixture output must have s_col"
    );
    fixture.proof.outputs[0].s_col[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered SplitNc output s_col must be rejected"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_running_output_y_ring_non_ct_lane() {
    // Use lane 1 so this does not rely only on the ct == y_ring[0]
    // denormalization guard. For carried CE outputs (idx >= K fresh), the
    // FE terminal identity's Eval block consumes all y_ring lanes through
    // the SuperNeo evaluation relation.
    let mut fixture = build_fixture();
    let idx = fixture.fresh_claims.len();
    assert!(idx < fixture.proof.outputs.len(), "fixture must have a running output");
    assert!(
        !fixture.proof.outputs[idx].y_ring.is_empty() && fixture.proof.outputs[idx].y_ring[0].len() > 1,
        "fixture running output must have a non-ct y_ring lane"
    );
    fixture.proof.outputs[idx].y_ring[0][1] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered SplitNc running output y_ring lane must be rejected"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_fresh_output_y_ring_padding_lane() {
    // Fresh outputs' FE identity consumes ct(y) for the CCS polynomial, and
    // Π_RLC rotates only the real D ring lanes. The padded representation
    // still must be canonical: native SuperNeo pads y_ring[D..d_pad] with
    // zeros, so the verifier circuit must not leave those witness lanes free.
    let mut fixture = build_fixture();
    let d_pad = D.next_power_of_two();
    assert!(d_pad > D, "fixture must have padded y_ring lanes");
    assert!(
        !fixture.proof.outputs[0].y_ring.is_empty() && fixture.proof.outputs[0].y_ring[0].len() == d_pad,
        "fixture must expose full padded y_ring rows"
    );
    fixture.proof.outputs[0].y_ring[0][D] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered SplitNc fresh-output y_ring padding lane must be rejected"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_output_ct() {
    // ct is cached from y_ring's constant term and feeds scalar folding.
    // It must be constrained back to y_ring immediately after allocation.
    let mut fixture = build_fixture();
    assert!(!fixture.proof.outputs[0].ct.is_empty(), "fixture output must have ct");
    fixture.proof.outputs[0].ct[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered SplitNc output ct must be rejected");
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_parent_authority_r() {
    // The running parent authority is part of HyperNova's carried U_i
    // handle. Its r point must be transcript-bound, not merely allocated.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("fixture must have running parent authority");
    assert!(!parent.r.is_empty(), "parent authority must have r");
    parent.r[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running parent-authority r must be rejected"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_parent_authority_y_ring() {
    // This targets the exact old failure mode: a commitment-only handle
    // would miss y_ring. The current running-accumulator authority handle
    // should absorb it and reroute the verifier transcript.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("fixture must have running parent authority");
    assert!(
        !parent.y_ring.is_empty() && parent.y_ring[0].len() > 1,
        "parent authority must have non-ct y_ring lanes"
    );
    parent.y_ring[0][1] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running parent-authority y_ring must be rejected"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_parent_authority_y_ring_c1_limb() {
    // The parent-authority y_ring is part of the full HyperNova U_i
    // handle. A c1-only mutation would pass if the handle or ct guard
    // accidentally absorbed only the first K limb.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("fixture must have running parent authority");
    assert!(
        !parent.y_ring.is_empty() && parent.y_ring[0].len() > 1,
        "parent authority must have non-ct y_ring lanes"
    );
    let original = parent.y_ring[0][1];
    parent.y_ring[0][1] = original + k_c1_one();
    assert_eq!(
        parent.y_ring[0][1].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        parent.y_ring[0][1].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running parent-authority y_ring.c1 must be rejected"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_parent_authority_ct() {
    // ct is a denormalized scalar view of parent.y_ring. It must be tied
    // back to y_ring before the parent authority is used as part of the
    // running-accumulator authority handle.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("fixture must have running parent authority");
    assert!(!parent.ct.is_empty(), "parent authority must have ct");
    parent.ct[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running parent-authority ct must be rejected"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_parent_authority_s_col() {
    // s_col is not part of the paper CE tuple, but it is carried by the
    // SplitNc implementation and included in the full accumulator handle.
    // It must not be a free metadata field.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("fixture must have running parent authority");
    assert!(!parent.s_col.is_empty(), "parent authority must have s_col");
    parent.s_col[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running parent-authority s_col must be rejected"
    );
}

#[test]
fn split_nc_pi_ccs_v_accepts_tampered_parent_authority_y_zcol_non_authority() {
    // Parent-authority y_zcol is an NC sidecar, not part of SuperNeo's CE
    // tuple. Π_DEC children cannot prove a verifier-checkable radix-b
    // y_zcol recomposition equation, so the recursive accumulator handle
    // deliberately omits it. Terminal CE verification binds final y_zcol
    // against the opened witness instead.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("fixture must have running parent authority");
    assert!(!parent.y_zcol.is_empty(), "parent authority must have y_zcol");
    parent.y_zcol[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        builder.is_satisfied(),
        "SplitNc Π_CCS.V should not treat parent-authority y_zcol as recursive accumulator authority"
    );
}

#[test]
fn split_nc_pi_ccs_v_accepts_tampered_parent_authority_y_zcol_c1_limb_non_authority() {
    // Same non-authority boundary as the c0-limb test, but perturb only c1
    // so an accidental limb-selective absorb would be caught by the digest
    // regression tests rather than hidden here.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("fixture must have running parent authority");
    assert!(!parent.y_zcol.is_empty(), "parent authority must have y_zcol");
    let original = parent.y_zcol[0];
    parent.y_zcol[0] = original + k_c1_one();
    assert_eq!(
        parent.y_zcol[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        parent.y_zcol[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        builder.is_satisfied(),
        "SplitNc Π_CCS.V should not treat c1 of parent-authority y_zcol as recursive accumulator authority"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_parent_authority_fold_digest() {
    // fold_digest is the carried transcript authority for the parent. It
    // must be absorbed into the running-accumulator authority handle, not
    // trusted as self-consistent metadata.
    let mut fixture = build_fixture();
    let parent = fixture
        .running
        .parent_authority
        .as_mut()
        .expect("fixture must have running parent authority");
    parent.fold_digest[0] ^= 1;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "tampered running parent-authority fold_digest must be rejected"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_nc_y_zcol() {
    let mut fixture = build_fixture();
    // Mutate one y_zcol lane of the first output. The NC terminal identity
    // recomputes `⟨y_zcol, χ_{α'}⟩` from this wire, so its pin-to-rhs_nc
    // must break.
    fixture.proof.outputs[0].y_zcol[0] += K::ONE;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered y_zcol must be rejected");
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_nc_y_zcol_c1_limb() {
    // y_zcol is K-valued in the NC terminal identity. Perturb only c1 so
    // this fails only if the verifier really consumes both limbs.
    let mut fixture = build_fixture();
    assert!(
        !fixture.proof.outputs[0].y_zcol.is_empty(),
        "fixture output must have y_zcol"
    );
    let original = fixture.proof.outputs[0].y_zcol[0];
    fixture.proof.outputs[0].y_zcol[0] = original + k_c1_one();
    assert_eq!(
        fixture.proof.outputs[0].y_zcol[0].as_coeffs()[0],
        original.as_coeffs()[0],
        "mutation must leave c0 unchanged"
    );
    assert_ne!(
        fixture.proof.outputs[0].y_zcol[0].as_coeffs()[1],
        original.as_coeffs()[1],
        "mutation must change c1"
    );

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(
        !builder.is_satisfied(),
        "SplitNc Π_CCS.V accepted a c1-only y_zcol tamper"
    );
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_header_digest() {
    let mut fixture = build_fixture();
    // Flip one byte of the captured header digest. The catch-up squeeze
    // computes the real digest and pins each lane to the recorded value,
    // so any byte flip breaks at least one lane's pin.
    fixture.proof.sumcheck.header_digest[0] ^= 1;

    let builder = emit_verifier(&fixture).expect("emit verifier");
    assert!(!builder.is_satisfied(), "tampered header_digest must be rejected");
}

#[test]
fn split_nc_pi_ccs_v_rejects_output_m_in_mismatch() {
    // `m_in` is a structural field; the verifier rejects with `Err(Shape)`
    // *before* emitting any constraints when it disagrees with the input
    // claim's m_in. (Mirrors native `validate_me_outputs_against_inputs`.)
    let mut fixture = build_fixture();
    fixture.proof.outputs[0].m_in += 1;

    let err = match emit_verifier(&fixture) {
        Ok(_) => panic!("m_in mismatch must surface as Err(Shape)"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(msg.contains("m_in"), "expected 'm_in' in error, got: {msg}");
}

#[test]
fn split_nc_pi_ccs_v_rejects_tampered_output_shape_metadata_wire() {
    // Shape metadata is represented as scalar wires and pinned
    // output→input by rows, not just checked by Rust before synthesis.
    // This simulates the Spartan setting: an honest circuit is emitted,
    // then a prover tries to change a metadata witness column.
    let fixture = build_fixture();
    let (mut builder, derived) = emit_verifier_with_derived(&fixture).expect("emit verifier");
    assert!(builder.is_satisfied(), "baseline must satisfy");

    let target_col = derived.outputs[0].m_in_var.col();
    let tampered = builder.witness()[target_col] + F::ONE;
    builder.tamper_witness(target_col, tampered);

    assert!(
        !builder.is_satisfied(),
        "SplitNc Π_CCS.V accepted output.m_in metadata wire after it diverged from the input"
    );
}
