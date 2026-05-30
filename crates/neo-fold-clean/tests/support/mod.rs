// Shared test fixtures: each test binary inlines this module via `#[path =
// "../support/mod.rs"] mod support;`, and only uses the subset it needs.
// `-Dwarnings` turns the resulting `dead_code` and `unused_imports` warnings
// into errors per binary, so opt them all out at the module root.
#![allow(dead_code, unused_imports)]

pub mod fibonacci_f_prime;
pub mod r1cs_compiler_fixtures;

use neo_ajtai::{has_global_pp_for_dims, s_mul_add, scale_commitment_add_inplace, set_global_pp_seeded, Commitment};
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_fold_clean::{config, preprocess, CcsInstance, DecMixer, Params, Preprocessing, RlcMixer, Structure};
use neo_math::ring::{cf_inv, Rq as RqEl};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

pub fn toy_preprocessing() -> Preprocessing {
    let structure = toy_structure();
    let params = config::r1cs_params(structure.n, structure.m).expect("production-core toy params");
    install_ajtai_module(&params, &structure);
    preprocess(
        params,
        structure,
        mix_rhos_commits as RlcMixer,
        combine_b_pows as DecMixer,
        Some(1),
    )
    .expect("toy preprocessing")
}

pub fn toy_instance(prep: &Preprocessing, _seed: u64) -> CcsInstance {
    let z = vec![F::ZERO; prep.structure().m];
    CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &z, 1)
        .expect("toy low-norm CCS instance")
}

#[allow(dead_code)]
pub fn mutate_ce_claim(claim: &mut neo_fold_clean::CeClaim) {
    claim.c.data[0] += F::ONE;
}

fn toy_structure() -> Structure {
    CcsStructure::new(vec![Mat::identity(1)], SparsePoly::new(1, vec![])).expect("toy CCS structure")
}

fn install_ajtai_module(params: &Params, structure: &Structure) {
    let cols = structure.m.div_ceil(D);
    if !has_global_pp_for_dims(D, cols) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x4e45_4f46_4f4c_4431_u64.to_le_bytes());
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
