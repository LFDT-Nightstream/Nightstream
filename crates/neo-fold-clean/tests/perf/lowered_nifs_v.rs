//! Measurement snapshot: the complete authoritative NIFS.V circuit,
//! low-norm lowered.
//!
//! This is the shell-vs-lowering decision datapoint. The manual projection
//! shell prices a hand-typed image (14,040,452 bits/step, incomplete
//! semantics); the alternative Road A completion lowers the field-native
//! NIFS.V circuit — the relation that already IS the authority — into the
//! b = 2 bit-backed CCS mechanically. Until now the lowering ratio was only
//! measured on the 3-field C14 toy (10,532 cols → 661,445 bits, ≈ 62.8
//! bits/col). This snapshot measures it on the real object: the full
//! Π_CCS.V → Π_RLC.V (projection census) → Π_DEC.V verifier circuit over an
//! honest two-fold chain at the small direct-CCS app shape, exact rows,
//! satisfiability checked on both sides of the lowering.
//!
//! Extrapolation to production is deliberately left to the reader of the
//! printed ratios: production field-native wire counts come from the decider
//! shape snapshot (`perf_fibonacci_bits`), and multiplying them by the
//! bits-per-col ratio printed here is an estimate, not a measurement.

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::engine::r1cs_circuit::TranscriptGadget;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::frontends::r1cs_f_prime::lowering::{lower_field_r1cs, lower_sparse_r1cs_to_low_norm};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::nifs::circuit::{
    enforce_nifs_v_circuit_with_transcript, NifsVCircuitConfig, NifsVCircuitMessages, NifsVOutputs,
};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_fold_clean::paper::relations::CcsClaim;
use neo_fold_clean::{CeClaim, Preprocessing};
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const SESSION_LABEL: &[u8] = b"neo.fold.clean/session/v1";

struct Fixture {
    prep: Preprocessing,
    fresh_claims: Vec<CcsClaim>,
    running: RunningInstance,
    proof: NifsProof,
    combined: CeClaim,
    children: Vec<CeClaim>,
}

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

fn assignment(a: u64, b: u64) -> Vec<F> {
    let mut z = vec![F::ZERO; D];
    z[0] = F::ONE;
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[3] = F::from_u64(a + b);
    z
}

fn build_honest_fixture() -> Fixture {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");

    let first = direct_ccs::build_instance(&prep, &r1cs, &assignment(1, 0)).expect("first instance");
    let mut first_tr = Transcript::session();
    let (running, _first_proof) = nifs::prove(
        &mut first_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![first],
        &RunningInstance::default(),
    )
    .expect("first NIFS.P");

    let second = direct_ccs::build_instance(&prep, &r1cs, &assignment(0, 1)).expect("second instance");
    let fresh_claims = vec![second.claim.clone()];
    let mut second_tr = Transcript::session();
    let (next_running, proof) = nifs::prove(
        &mut second_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![second],
        &running,
    )
    .expect("second NIFS.P");

    Fixture {
        prep,
        fresh_claims,
        running,
        combined: proof.pi_rlc.combined.clone(),
        children: next_running.claims,
        proof,
    }
}

fn pi_ccs_config(prep: &Preprocessing) -> SplitNcPiCcsVConfig<'_> {
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

fn emit_verifier(f: &Fixture) -> (R1csBuilder, NifsVOutputs) {
    let mut builder = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut builder, SESSION_LABEL);
    let cfg = NifsVCircuitConfig {
        pi_ccs: pi_ccs_config(&f.prep),
    };
    let outputs = enforce_nifs_v_circuit_with_transcript(
        &mut builder,
        &f.prep.params,
        &cfg,
        &mut tr,
        &NifsVCircuitMessages {
            fresh: &f.fresh_claims,
            running: &f.running.claims,
            running_parent_authority: f.running.parent_authority.as_ref(),
            pi_ccs: &f.proof.pi_ccs,
            combined: &f.combined,
            children: &f.children,
        },
    )
    .expect("NIFS.V synthesis");
    (builder, outputs)
}

#[test]
#[ignore = "measurement snapshot: run with --ignored --nocapture; prints the field-native vs lowered cost of the complete NIFS.V circuit"]
fn lowered_authoritative_nifs_v_measurement() {
    let fixture = build_honest_fixture();
    let (builder, outputs) = emit_verifier(&fixture);
    assert!(builder.is_satisfied(), "honest NIFS.V must satisfy before lowering");

    let field_rows = builder.rows();
    let field_cols = builder.cols();
    let field_nnz = builder.nonzero_entries();

    let lowered = lower_field_r1cs(builder, &outputs.projection_beta).expect("field lowering");
    let (shape, field_assignment) = lowered.into_parts();
    let encoded = lower_sparse_r1cs_to_low_norm(&shape, &field_assignment).expect("low-norm lowering");
    assert!(
        encoded.is_satisfied(encoded.assignment()),
        "the lowered honest witness must satisfy the bit-backed relation"
    );

    let low_rows = encoded.structure().n;
    let low_bits = encoded.structure().m;
    eprintln!("== authoritative NIFS.V circuit, low-norm lowered (two-fold direct-CCS fixture) ==");
    eprintln!("field-native   rows {field_rows:>12}  cols {field_cols:>12}  nnz {field_nnz:>12}");
    eprintln!("low-norm       rows {low_rows:>12}  committed_bits {low_bits:>12}");
    eprintln!(
        "ratios         bits/col {:>8.2}  rows/row {:>8.2}",
        low_bits as f64 / field_cols as f64,
        low_rows as f64 / field_rows as f64,
    );
}
