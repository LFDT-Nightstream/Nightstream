//! Neutral wire types for the `Pi_CCS` protocol.
//!
//! This module owns protocol messages and their canonical serialization. It
//! does not own either the reference or optimized computation.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{KExtensions, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};

use crate::error::PiCcsError;

/// Fiat--Shamir challenges used by a `Pi_CCS` proof.
///
/// For `PaperRectangularV1`, `beta_r` is the row equality point, `beta_m` is
/// the column equality point, and `gamma` mixes the paper terms. `alpha` and
/// `beta_a` are empty because the canonical protocol has no coefficient-lane
/// SumCheck axis.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Challenges {
    pub alpha: Vec<K>,
    pub beta_a: Vec<K>,
    pub beta_r: Vec<K>,
    pub beta_m: Vec<K>,
    pub gamma: K,
}

impl Challenges {
    pub fn paper_rectangular(beta_r: Vec<K>, beta_m: Vec<K>, gamma: K) -> Self {
        Self {
            alpha: Vec::new(),
            beta_a: Vec::new(),
            beta_r,
            beta_m,
            gamma,
        }
    }

    pub fn is_paper_rectangular(&self) -> bool {
        self.alpha.is_empty() && self.beta_a.is_empty()
    }
}

#[inline]
pub fn gamma_power(gamma: K, exponent: usize) -> K {
    let mut power = K::ONE;
    for _ in 0..exponent {
        power *= gamma;
    }
    power
}

/// Absolute paper exponent `2K+k+I(i,j,l)` with zero-based coordinates.
pub fn carried_gamma_exponent(
    fresh_count: usize,
    running_count: usize,
    matrix_count: usize,
    running: usize,
    matrix: usize,
    coefficient: usize,
) -> usize {
    2 * fresh_count + running_count + running + running_count * matrix + running_count * matrix_count * coefficient
}

#[inline]
pub fn equality(point: &[K], target: &[K]) -> K {
    assert_eq!(point.len(), target.len(), "equality point length mismatch");
    point
        .iter()
        .zip(target)
        .fold(K::ONE, |acc, (&left, &right)| {
            acc * ((K::ONE - left) * (K::ONE - right) + left * right)
        })
}

/// Reject public `X` data outside the active packed-input prefix.
///
/// `m_in` counts scalar inputs. One packed ring column carries `D` scalars,
/// so columns after `ceil(m_in / D)` are structural zeros.
pub fn validate_inactive_x_zero<Ff>(label: &str, claim: &CeClaim<Cmt, Ff, K>) -> Result<(), PiCcsError>
where
    Ff: Field,
{
    let active_columns = claim.m_in.div_ceil(D);
    if claim.X.rows() != D || claim.X.cols() != claim.m_in {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: X has shape {}x{}, expected {D}x{}",
            claim.X.rows(),
            claim.X.cols(),
            claim.m_in
        )));
    }
    for column in active_columns..claim.X.cols() {
        for row in 0..claim.X.rows() {
            if claim.X[(row, column)] != Ff::ZERO {
                return Err(PiCcsError::InvalidInput(format!(
                    "{label}: inactive X entry ({row},{column}) must be zero"
                )));
            }
        }
    }
    Ok(())
}

#[inline]
fn range_product<Ff>(value: K, base: u32) -> K
where
    Ff: Field + PrimeCharacteristicRing,
    K: From<Ff>,
{
    let mut product = K::ONE;
    for integer in -((base as i64) - 1)..=((base as i64) - 1) {
        product *= value - K::from(Ff::from_i64(integer));
    }
    product
}

/// Corrected absolute target `T` from the public running claims.
pub fn fe_initial_claim<Ff>(
    structure: &CcsStructure<Ff>,
    challenges: &Challenges,
    fresh_count: usize,
    running: &[CeClaim<Cmt, Ff, K>],
) -> Result<K, PiCcsError>
where
    Ff: Field,
{
    let running_count = running.len();
    let mut target = K::ZERO;
    for (running_index, claim) in running.iter().enumerate() {
        if claim.y_ring.len() != structure.t() {
            return Err(PiCcsError::InvalidInput(format!(
                "running claim {running_index} has {} matrix images, expected {}",
                claim.y_ring.len(),
                structure.t()
            )));
        }
        for (matrix, coefficients) in claim.y_ring.iter().enumerate() {
            if coefficients.len() < D {
                return Err(PiCcsError::InvalidInput(format!(
                    "running claim {running_index} matrix {matrix} has {} coefficients, expected at least {D}",
                    coefficients.len()
                )));
            }
            for (coefficient, &value) in coefficients.iter().take(D).enumerate() {
                let exponent = carried_gamma_exponent(
                    fresh_count,
                    running_count,
                    structure.t(),
                    running_index,
                    matrix,
                    coefficient,
                );
                target += gamma_power(challenges.gamma, exponent) * value;
            }
        }
    }
    Ok(target)
}

/// Verifier terminal for the row-domain FE polynomial.
pub fn fe_terminal<Ff>(
    structure: &CcsStructure<Ff>,
    challenges: &Challenges,
    fresh_count: usize,
    prior_point: Option<&[K]>,
    row_point: &[K],
    outputs: &[CeClaim<Cmt, Ff, K>],
) -> Result<K, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if outputs.len() < fresh_count {
        return Err(PiCcsError::InvalidInput("too few rectangular FE outputs".into()));
    }
    let mut fresh_ccs = K::ZERO;
    for (source, output) in outputs.iter().take(fresh_count).enumerate() {
        if output.ct.len() < structure.t() {
            return Err(PiCcsError::InvalidInput("rectangular FE output ct is too short".into()));
        }
        fresh_ccs += gamma_power(challenges.gamma, source) * structure.f.eval_in_ext::<K>(&output.ct[..structure.t()]);
    }
    let running_count = outputs.len() - fresh_count;
    let mut carried = K::ZERO;
    for (running, output) in outputs.iter().skip(fresh_count).enumerate() {
        if output.y_ring.len() != structure.t() {
            return Err(PiCcsError::InvalidInput(
                "rectangular FE output matrix count mismatch".into(),
            ));
        }
        for (matrix, coefficients) in output.y_ring.iter().enumerate() {
            if coefficients.len() < D {
                return Err(PiCcsError::InvalidInput(
                    "rectangular FE output coefficient row is too short".into(),
                ));
            }
            for (coefficient, &value) in coefficients.iter().take(D).enumerate() {
                let exponent =
                    carried_gamma_exponent(fresh_count, running_count, structure.t(), running, matrix, coefficient);
                carried += gamma_power(challenges.gamma, exponent) * value;
            }
        }
    }
    let carried_gate = prior_point.map_or(K::ZERO, |prior| equality(row_point, prior));
    Ok(equality(row_point, &challenges.beta_r) * fresh_ccs + carried_gate * carried)
}

/// Verifier terminal for the column-domain NC polynomial.
pub fn nc_terminal<Ff>(
    params: &neo_params::NeoParams,
    challenges: &Challenges,
    fresh_count: usize,
    column_point: &[K],
    outputs: &[CeClaim<Cmt, Ff, K>],
) -> Result<K, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let mut norm = K::ZERO;
    for (source, output) in outputs.iter().enumerate() {
        if output.y_zcol.len() < D {
            return Err(PiCcsError::InvalidInput(
                "rectangular NC output opening is too short".into(),
            ));
        }
        let value = output.y_zcol.iter().take(D).copied().sum();
        norm += gamma_power(challenges.gamma, fresh_count + source) * range_product::<Ff>(value, params.b);
    }
    Ok(equality(column_point, &challenges.beta_m) * norm)
}

/// Proof format selected before any public challenge is sampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PiCcsProofVariant {
    /// Previous row/lane plus column/lane split. This is not paper-exact.
    SplitNcV1,
    /// Previous production block/lane protocol. This is not paper-exact.
    BlockLaneNcDelayedV1,
    /// Corrected paper algebra with row-domain FE and column-domain NC.
    PaperRectangularV1,
}

/// Public proof message for `Pi_CCS`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PiCcsProof {
    pub variant: PiCcsProofVariant,
    pub sumcheck_rounds: Vec<Vec<K>>,
    pub sc_initial_sum: Option<K>,
    pub sumcheck_challenges: Vec<K>,
    pub sumcheck_rounds_nc: Vec<Vec<K>>,
    pub sc_initial_sum_nc: Option<K>,
    pub sumcheck_challenges_nc: Vec<K>,
    pub challenges_public: Challenges,
    pub sumcheck_final: K,
    pub sumcheck_final_nc: K,
    pub header_digest: Vec<u8>,
    pub _extra: Option<Vec<u8>>,
}

impl PiCcsProof {
    pub fn new(sumcheck_rounds: Vec<Vec<K>>, sc_initial_sum: Option<K>) -> Self {
        Self {
            variant: PiCcsProofVariant::PaperRectangularV1,
            sumcheck_rounds,
            sc_initial_sum,
            sumcheck_challenges: Vec::new(),
            sumcheck_rounds_nc: Vec::new(),
            sc_initial_sum_nc: None,
            sumcheck_challenges_nc: Vec::new(),
            challenges_public: Challenges::paper_rectangular(Vec::new(), Vec::new(), K::ZERO),
            sumcheck_final: K::ZERO,
            sumcheck_final_nc: K::ZERO,
            header_digest: Vec::new(),
            _extra: None,
        }
    }

    /// Normalize field representatives before serialization.
    pub fn canonicalize(&mut self) {
        canonicalize_rounds(&mut self.sumcheck_rounds);
        canonicalize_rounds(&mut self.sumcheck_rounds_nc);
        canonicalize_vec(&mut self.sumcheck_challenges);
        canonicalize_vec(&mut self.sumcheck_challenges_nc);
        canonicalize_vec(&mut self.challenges_public.alpha);
        canonicalize_vec(&mut self.challenges_public.beta_a);
        canonicalize_vec(&mut self.challenges_public.beta_r);
        canonicalize_vec(&mut self.challenges_public.beta_m);
        self.challenges_public.gamma = canonical_k(self.challenges_public.gamma);
        self.sc_initial_sum = self.sc_initial_sum.map(canonical_k);
        self.sc_initial_sum_nc = self.sc_initial_sum_nc.map(canonical_k);
        self.sumcheck_final = canonical_k(self.sumcheck_final);
        self.sumcheck_final_nc = canonical_k(self.sumcheck_final_nc);
    }

    /// Canonical bytes used for exact engine cross-checks and transport.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        let mut proof = self.clone();
        proof.canonicalize();
        bincode::serialize(&proof)
    }
}

fn canonicalize_rounds(rounds: &mut [Vec<K>]) {
    for round in rounds {
        canonicalize_vec(round);
    }
}

fn canonicalize_vec(values: &mut [K]) {
    for value in values {
        *value = canonical_k(*value);
    }
}

fn canonical_k(value: K) -> K {
    let (c0, c1) = value.to_limbs_u64();
    neo_math::from_complex(F::from_u64(c0), F::from_u64(c1))
}
