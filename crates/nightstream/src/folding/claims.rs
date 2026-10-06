use super::{superneo_public_x_cols, CcsClaim, CcsWitness, CeClaim, Params, Structure};
use neo_ajtai::{s_mul_add, scale_commitment_add_inplace, Commitment};
use neo_ccs::Mat;
use neo_math::ring::{cf_inv, Rq};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

/// Evaluation vectors use the padded ring degree; the lanes from `D` on are zero.
pub(crate) const EVALUATION_WIDTH: usize = D.next_power_of_two();

fn zero_padding(values: &[K]) -> bool {
    values.iter().skip(D).all(|lane| *lane == K::ZERO)
}

/// The one canonical encoding of an `Eval_K` vector or an `Eval_A` row.
pub(crate) fn is_canonical_evaluation(values: &[K]) -> bool {
    values.len() == EVALUATION_WIDTH && zero_padding(values)
}

/// `Eval_K` and `matrices` rows of `Eval_A`, each of the padded width.
pub(crate) fn has_evaluation_shape(claim: &CeClaim, matrices: usize) -> bool {
    claim.eval_k.len() == EVALUATION_WIDTH
        && claim.eval_a.len() == matrices
        && claim.eval_a.iter().all(|row| row.len() == EVALUATION_WIDTH)
}

/// Every lane from `D` on is zero in `Eval_K` and in every `Eval_A` row.
pub(crate) fn has_zero_evaluation_padding(claim: &CeClaim) -> bool {
    zero_padding(&claim.eval_k) && claim.eval_a.iter().all(|row| zero_padding(row))
}

/// The canonical evaluation families of one claim.
pub(crate) fn has_canonical_evaluations(claim: &CeClaim, matrices: usize) -> bool {
    has_evaluation_shape(claim, matrices) && has_zero_evaluation_padding(claim)
}

#[derive(Clone, Debug)]
pub struct CcsInstance {
    pub claim: CcsClaim,
    pub witness: CcsWitness,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunningInstance {
    pub claims: Vec<CeClaim>,
    pub witnesses: Vec<Mat<F>>,
    pub parent_authority: Option<CeClaim>,
}
impl RunningInstance {
    pub fn new(claims: Vec<CeClaim>, witnesses: Vec<Mat<F>>, parent_authority: Option<CeClaim>) -> Self {
        Self {
            claims,
            witnesses,
            parent_authority,
        }
    }
    pub fn claims_only(&self) -> Self {
        Self::new(self.claims.clone(), Vec::new(), self.parent_authority.clone())
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.claims.is_empty() && self.witnesses.is_empty() && self.parent_authority.is_none()
    }
    pub(crate) fn prover_shape_is_valid(&self) -> bool {
        self.claims.len() == self.witnesses.len() && self.claims.is_empty() == self.parent_authority.is_none()
    }
    pub(crate) fn canonical_zero(pp: &Params, s: &Structure, m_in: usize) -> Result<Self, &'static str> {
        if m_in > s.m || m_in % D != 0 {
            return Err("canonical zero public input shape");
        }
        let ell = s
            .domain_rows()
            .max(neo_reductions::common::superneo_carrier_width(s.m))
            .next_power_of_two()
            .max(2)
            .trailing_zeros() as usize;
        let claim = CeClaim {
            c: Commitment::zeros(D, pp.kappa() as usize),
            X: Mat::virtual_constant(D, superneo_public_x_cols(m_in), F::ZERO),
            r: vec![K::ZERO; ell],
            eval_k: vec![K::ZERO; EVALUATION_WIDTH],
            eval_a: vec![vec![K::ZERO; EVALUATION_WIDTH]; s.t()],
            m_in,
            fold_digest: [0; 32],
            adv: None,
        };
        Ok(Self::new(
            vec![claim.clone(); pp.k_rho() as usize],
            vec![Mat::virtual_constant(D, s.m.div_ceil(D), F::ZERO); pp.k_rho() as usize],
            Some(claim),
        ))
    }
}
pub(crate) fn superneo_has_canonical_x_shape(x: &Mat<F>, m_in: usize) -> bool {
    m_in % D == 0 && x.rows() == D && x.cols() == superneo_public_x_cols(m_in)
}
pub fn ajtai_rlc_mixer(rhos: &[Mat<F>], commits: &[Commitment]) -> Commitment {
    debug_assert!(!commits.is_empty(), "commit_mix: empty commitments");
    debug_assert_eq!(rhos.len(), commits.len(), "commit_mix: |rhos| != |commits|");
    let mut acc = Commitment::zeros(commits[0].d, commits[0].kappa);
    for (rho, c) in rhos.iter().zip(commits.iter()) {
        let rq = rot_matrix_to_rq(rho);
        s_mul_add(&mut acc, &rq, c);
    }
    acc
}
pub fn ajtai_dec_mixer(commits: &[Commitment], b: u32) -> Commitment {
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
