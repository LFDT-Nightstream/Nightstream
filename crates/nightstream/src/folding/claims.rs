use super::{superneo_public_x_cols, CcsClaim, CcsWitness, CeClaim, Params, Structure};
use neo_ajtai::{s_mul_add, scale_commitment_add_inplace, Commitment};
use neo_ccs::Mat;
use neo_math::ring::{cf_inv, Rq};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

pub(crate) const EVALUATION_WIDTH: usize = D.next_power_of_two();

pub(crate) fn is_canonical_evaluation(values: &[K]) -> bool {
    values.len() == EVALUATION_WIDTH && values[D..].iter().all(|lane| *lane == K::ZERO)
}

pub(crate) fn has_canonical_evaluations(claim: &CeClaim, matrices: usize) -> bool {
    claim.eval_a.len() == matrices
        && is_canonical_evaluation(&claim.eval_k)
        && claim.eval_a.iter().all(|row| is_canonical_evaluation(row))
}

#[derive(Clone, Debug)]
pub(crate) struct CcsInstance {
    pub claim: CcsClaim,
    pub witness: CcsWitness,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RunningInstance {
    pub claims: Vec<CeClaim>,
    pub witnesses: Vec<Mat<F>>,
}
impl RunningInstance {
    pub(crate) fn new(claims: Vec<CeClaim>, witnesses: Vec<Mat<F>>) -> Self {
        Self { claims, witnesses }
    }
    pub(crate) fn claims_only(&self) -> Self {
        Self::new(self.claims.clone(), Vec::new())
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.claims.is_empty() && self.witnesses.is_empty()
    }
    pub(crate) fn prover_shape_is_valid(&self) -> bool {
        self.claims.len() == self.witnesses.len()
    }
    pub(crate) fn canonical_zero(pp: &Params, s: &Structure, m_in: usize) -> Result<Self, &'static str> {
        let Some(x_cols) = superneo_public_x_cols(m_in).filter(|_| m_in <= s.m) else {
            return Err("canonical zero public input shape");
        };
        let ell = s.cube_variables();
        let claim = CeClaim {
            c: Commitment::zeros(D, pp.kappa() as usize),
            X: Mat::virtual_constant(D, x_cols, F::ZERO),
            r: vec![K::ZERO; ell],
            eval_k: vec![K::ZERO; EVALUATION_WIDTH],
            eval_a: vec![vec![K::ZERO; EVALUATION_WIDTH]; s.t()],
            m_in,
            fold_digest: [0; 32],
            adv: None,
        };
        Ok(Self::new(
            vec![claim; pp.k_rho() as usize],
            vec![Mat::virtual_constant(D, s.m.div_ceil(D), F::ZERO); pp.k_rho() as usize],
        ))
    }
}
pub(crate) fn superneo_has_canonical_x_shape(x: &Mat<F>, m_in: usize) -> bool {
    x.rows() == D && superneo_public_x_cols(m_in) == Some(x.cols())
}
pub(crate) fn ajtai_rlc_mixer(rhos: &[Mat<F>], commits: &[Commitment]) -> Commitment {
    debug_assert!(!commits.is_empty(), "commit_mix: empty commitments");
    debug_assert_eq!(rhos.len(), commits.len(), "commit_mix: |rhos| != |commits|");
    let mut acc = Commitment::zeros(commits[0].d, commits[0].kappa);
    for (rho, c) in rhos.iter().zip(commits.iter()) {
        let rq = rot_matrix_to_rq(rho);
        s_mul_add(&mut acc, &rq, c);
    }
    acc
}
pub(crate) fn ajtai_dec_mixer(commits: &[Commitment], b: u32) -> Commitment {
    debug_assert!(!commits.is_empty(), "dec_mix: empty commitments");
    let mut acc = Commitment::zeros(commits[0].d, commits[0].kappa);
    let base = F::from_u64(b as u64);
    let mut pow = F::ONE;
    for c in commits {
        scale_commitment_add_inplace(&mut acc, pow, c);
        pow *= base;
    }
    acc
}
fn rot_matrix_to_rq(mat: &Mat<F>) -> Rq {
    let mut coeffs = [F::default(); D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}
