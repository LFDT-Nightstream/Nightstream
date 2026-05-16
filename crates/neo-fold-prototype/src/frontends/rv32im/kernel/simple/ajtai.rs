//! Owns the Ajtai commitment mixer callbacks used by the RV32IM simple-kernel flow.

use crate::prover::CommitmentMixers;
use neo_ajtai::{s_mul_add_from_rot_col, scale_commitment_add_inplace, Commitment};
use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

fn first_rot_col_from_matrix(mat: &Mat<F>) -> [F; D] {
    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    coeffs
}

fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Commitment]) -> Commitment {
    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    for (rho, c) in rhos.iter().zip(cs.iter()) {
        let first_rot_col = first_rot_col_from_matrix(rho);
        s_mul_add_from_rot_col(&mut acc, &first_rot_col, c);
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

pub fn rv32im_ajtai_mixers(
) -> CommitmentMixers<fn(&[Mat<F>], &[Commitment]) -> Commitment, fn(&[Commitment], u32) -> Commitment> {
    CommitmentMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}
