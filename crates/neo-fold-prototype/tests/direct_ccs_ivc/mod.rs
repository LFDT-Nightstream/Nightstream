use neo_ajtai::{
    has_global_pp_for_dims, s_mul_add, scale_commitment_add_inplace, set_global_pp_seeded, AjtaiSModule, Commitment,
};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{poly::SparsePoly, poly::Term, CcsClaim, CcsMatrix, CcsStructure, CcsWitness, CscMat, Mat};
use neo_fold_prototype::core::ivc::build_superneo_ivc_relations_with_initial_carry_accumulator_handle_perf;
use neo_fold_prototype::core::proof::{Carry, FoldSchedule, StepInput};
use neo_fold_prototype::core::prover::CommitmentMixers;
use neo_fold_prototype::direct_ccs::{
    direct_ccs_program_from_sparse_r1cs_with_public_input_len, direct_ccs_step_from_low_norm_full_witness,
    verify_direct_ccs_statement, DirectCcsCompactFPrimeImage, DirectCcsFPrimeLowNormSourceR1cs,
    DirectCcsIvcPublicImage, DirectCcsIvcSnark, DirectCcsIvcSnarkVerifierKey, DirectCcsIvcState,
    DirectCcsNativeFPrimeAdvice, DirectCcsProgram, DirectCcsRecursiveIvcPublicImage, DirectCcsRecursiveIvcState,
    DirectCcsStatement, DirectCcsStep, DirectSparseR1csExport,
};
use neo_fold_prototype::{
    extend_direct_ccs, preprocess_direct_ccs, prove_direct_ccs, verify_direct_ccs, DirectCcsCommitmentOps,
};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

mod adapter;
mod f_prime_source;
mod recursive_authority;
mod recursive_compression;
mod state;

fn assert_public_verify_rejects(
    vk: &DirectCcsIvcSnarkVerifierKey,
    snark: &DirectCcsIvcSnark,
    public_image: DirectCcsIvcPublicImage,
    label: &str,
) {
    assert!(
        verify_direct_ccs_statement(vk, &public_image.statement(), snark.proof()).is_err(),
        "direct CCS public verifier accepted tampered {label}"
    );
}
fn assert_statement_verify_rejects(
    vk: &DirectCcsIvcSnarkVerifierKey,
    snark: &DirectCcsIvcSnark,
    statement: DirectCcsStatement,
    label: &str,
) {
    assert!(
        verify_direct_ccs_statement(vk, &statement, snark.proof()).is_err(),
        "direct CCS statement verifier accepted tampered {label}"
    );
}
fn fibonacci_ccs() -> CcsStructure<F> {
    let mut m = Mat::zero(1, D, F::ZERO);
    m[(0, 0)] = F::ONE;
    m[(0, 1)] = F::ONE;
    m[(0, 2)] = -F::ONE;
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    CcsStructure::new(vec![m], f).expect("valid Fibonacci CCS")
}
fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}
fn ajtai_mixers() -> CommitmentMixers<fn(&[Mat<F>], &[Commitment]) -> Commitment, fn(&[Commitment], u32) -> Commitment>
{
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

    CommitmentMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn make_ajtai_module(params: &NeoParams) -> AjtaiSModule {
    make_ajtai_module_for_cols(params, 1)
}

fn make_ajtai_module_for_cols(params: &NeoParams, cols: usize) -> AjtaiSModule {
    if !has_global_pp_for_dims(D, cols) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x5355_5045_524e_454f_u64.to_le_bytes());
        match set_global_pp_seeded(D, params.kappa as usize, cols, seed) {
            Ok(()) => {}
            Err(_err) if has_global_pp_for_dims(D, cols) => {}
            Err(err) => panic!("Ajtai global setup: {err}"),
        }
    }
    AjtaiSModule::from_global_for_dims(D, cols).expect("Ajtai global module")
}

fn step(log: &AjtaiSModule, label: &str, a: u64, b: u64, c: u64) -> StepInput {
    let mut z = vec![F::ZERO; D];
    z[0] = F::from_u64(a);
    z[1] = F::from_u64(b);
    z[2] = F::from_u64(c);
    let mut z_mat = Mat::zero(D, 1, F::ZERO);
    for (idx, value) in z.iter().copied().enumerate() {
        z_mat[(idx, 0)] = value;
    }
    let m_in = 3;
    StepInput {
        label: label.to_string(),
        mcs: CcsClaim {
            c: log.commit(&z_mat),
            x: z[..m_in].to_vec(),
            m_in,
        },
        witness: CcsWitness {
            w: z[m_in..].to_vec(),
            Z: z_mat,
        },
    }
}
