#![cfg(feature = "cuda")]

//! CUDA-selected NIFS crosscheck boundary.
//!
//! CUDA currently selects the optimized CPU implementation of the one-joint
//! proof. This test checks the complete wrapper and proof boundary. It becomes
//! an evaluator-parity gate when a one-joint CUDA evaluator replaces that
//! delegation; it does not claim that such a kernel exists today.

use neo_ccs::Mat;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::nifs;
use neo_fold_clean::RunningInstance;
use neo_math::{D, F};
use neo_prover_cuda::CudaNifsProver;
use p3_field::PrimeCharacteristicRing;

fn relation(columns: usize) -> R1cs {
    let mut a = Mat::zero(1, columns, F::ZERO);
    a.set(0, 1, F::ONE);
    a.set(0, 2, F::ONE);
    let mut b = Mat::zero(1, columns, F::ZERO);
    b.set(0, 0, F::ONE);
    let mut c = Mat::zero(1, columns, F::ZERO);
    c.set(0, 3, F::ONE);
    R1cs { a, b, c, m_in: D }
}

fn assignment(columns: usize) -> Vec<F> {
    let mut values = vec![F::ZERO; columns];
    values[0] = F::ONE;
    values[1] = F::ONE;
    values[3] = F::ONE;
    values
}

#[test]
#[ignore = "requires CUDA hardware and the pinned cuda-oxide toolchain"]
fn cuda_selected_nifs_matches_optimized_cpu() {
    let r1cs = relation(2 * D);
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0x4355_4441_4e49_4653).expect("preprocess");
    let fresh = direct_ccs::build_instance(&prep, &r1cs, &assignment(2 * D)).expect("fresh instance");
    let mut prover = CudaNifsProver::new().expect("CUDA adapter").crosschecked();
    let mut transcript = Transcript::session();

    nifs::prove_with_adapter(
        &mut prover,
        &mut transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &RunningInstance::default(),
    )
    .expect("CUDA-selected NIFS matches optimized CPU");
}
