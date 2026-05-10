//! Diagnostic: NIFS.P → NIFS.V directly on an R1CS-derived CCS structure
//! with non-trivial polynomial f, fresh transcripts on each side, no
//! lifecycle plumbing.
//!
//! This isolates whether the bug we hit in the e2e tests is:
//! - In our lifecycle layer (then this passes).
//! - In our NIFS / engine wrappers (then this fails).

use neo_ccs::matrix::Mat as NeoMat;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs;

fn three_term_addition() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    a[(0, 2)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 0)] = F::ONE;
    let mut c = NeoMat::zero(1, m, F::default());
    c[(0, 3)] = F::ONE;
    R1cs { a, b, c, m_in: 3 }
}

#[test]
fn nifs_round_trip_on_r1cs_structure() {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, /* seed = */ 42).expect("preprocess");

    // (a, b, c) = (1, 0, 1)
    let mut z = vec![F::default(); prep.structure.m];
    z[0] = F::ONE;
    z[1] = F::ONE;
    z[2] = F::ZERO;
    z[3] = F::ONE;
    let instance = direct_ccs::build_instance(&prep, &r1cs, &z).expect("build");
    let fresh_claims = vec![instance.claim.clone()];
    let running = RunningInstance::default();

    // Prover
    let mut prover_tr = Transcript::session();
    let (next_running, proof) = nifs::prove(
        &mut prover_tr,
        &prep.params,
        &prep.structure,
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        vec![instance],
        &running,
    )
    .expect("NIFS.P");

    // Verifier
    let mut verifier_tr = Transcript::session();
    let verified = nifs::verify(
        &mut verifier_tr,
        &prep.params,
        &prep.structure,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        &fresh_claims,
        &[],
        &proof,
    )
    .expect("NIFS.V");

    assert_eq!(verified, next_running.claims);
}
