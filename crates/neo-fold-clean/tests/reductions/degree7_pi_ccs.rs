//! Degree-7 Π_CCS prove/verify hard gate.
//!
//! F' uses degree-7 Poseidon S-box rows, so the engine must demonstrably
//! accept a degree-7 CCS instance through a real `pi_ccs::prove` →
//! `pi_ccs::verify` cycle. This test pins that degree-flexibility claim
//! against an end-to-end measurement instead of an inference from reading
//! code.
//!
//! Structure under test: `t = 2`, `f(X_1, X_2) = X_1^7 − X_2`, one
//! row that enforces `z[0]^7 = z[1]`. The honest witness is
//! `(x, y) = (1, 1)`, which is low-norm under `b = 2`.

use neo_ajtai::{has_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule};
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term};
use neo_fold_clean::config;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::reductions::pi_ccs;
use neo_fold_clean::{CcsInstance, Params, Structure};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

const TEST_SEED_TAG: u64 = 0xD7D7_D7D7_D7D7_D7D7;
const LABEL: &[u8] = b"neo.fold.clean/test/degree7_pi_ccs/v1";

/// CCS structure that enforces `z[0]^7 = z[1]` at row 0. Matrix shape
/// is `1 × m`; the witness vector is length `m` (padded with zeros
/// beyond the two used cells).
fn degree7_structure(m: usize) -> Structure {
    assert!(m >= 2, "need at least 2 columns for [x, y]");
    let mut m1 = Mat::zero(1, m, F::ZERO);
    let mut m2 = Mat::zero(1, m, F::ZERO);
    m1.set(0, 0, F::ONE); // M_1 · z = z[0]
    m2.set(0, 1, F::ONE); // M_2 · z = z[1]
    let f = SparsePoly::new(
        2,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![7, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 1],
            },
        ],
    );
    CcsStructure::new(vec![m1, m2], f).expect("degree-7 structure")
}

fn install_ajtai(params: &Params, cols: usize) {
    if !has_global_pp_for_dims(D, cols) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&TEST_SEED_TAG.to_le_bytes());
        match set_global_pp_seeded(D, params.kappa() as usize, cols, seed) {
            Ok(()) => {}
            Err(_e) if has_global_pp_for_dims(D, cols) => {}
            Err(e) => panic!("Ajtai PP install for degree-7 test: {e}"),
        }
    }
}

#[test]
fn degree7_ccs_pi_ccs_prove_verify_accepts_honest_instance() {
    // Use m = 2 (the minimum). cols = 1 ring column under D = 54.
    let structure = degree7_structure(2);
    assert_eq!(structure.f.max_degree(), 7, "polynomial must have degree 7");
    let params = config::r1cs_params(structure.n, structure.m).expect("production-core params");
    let cols = structure.m.div_ceil(D);
    install_ajtai(&params, cols);
    let log = AjtaiSModule::from_global_for_dims(D, cols).expect("Ajtai log for degree-7 test");

    // Honest low-norm witness: x = 1, y = 1^7 = 1. Both |x|, |y| < b = 2.
    let mut z = vec![F::ZERO; structure.m];
    z[0] = F::ONE;
    z[1] = F::ONE;

    // m_in = 0: no public input. The instance digest absorbs the full claim
    // through `ccs_claim_digest` regardless.
    let instance =
        CcsInstance::from_low_norm_assignment(&params, &log, &structure, &z, 0).expect("honest degree-7 instance");

    let running = RunningInstance::default();
    let cache = neo_reductions::optimized_engine::OptimizedStructureCache::build(&structure).expect("cache build");

    // Prove with one transcript, verify with a matching one.
    let mut tr_prove = Transcript::with_label(LABEL);
    let proof = pi_ccs::prove(
        &mut tr_prove,
        &params,
        &structure,
        &cache,
        &log,
        vec![instance.clone()],
        &running,
    )
    .expect("degree-7 Π_CCS prove succeeds");

    let mut tr_verify = Transcript::with_label(LABEL);
    let outputs = pi_ccs::verify(
        &mut tr_verify,
        &params,
        &structure,
        &cache,
        &[instance.claim],
        &running,
        &proof,
    )
    .expect("degree-7 Π_CCS verify accepts honest proof");

    assert_eq!(
        outputs.len(),
        1,
        "Π_CCS output count should be |fresh| + |running| = 1 + 0 = 1"
    );
}
