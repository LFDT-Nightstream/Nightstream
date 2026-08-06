//! Neutral contract data for the one-joint padded-row PiCCS protocol.
//!
//! This module owns dimensions, tags, and audit-trace types. It does not own
//! transcript execution, polynomial evaluation, SumCheck, or proof assembly.

use neo_ccs::CcsStructure;
use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use crate::error::PiCcsError;

pub const PUBLIC_INPUT_TAG: u64 = 40;
pub const PROTOCOL_VERSION: u64 = 2;
pub const STATEMENT_TAG: u64 = 41;
pub const ALPHA_TAG: u64 = 42;
pub const GAMMA_TAG: u64 = 43;
pub const ROUND_TAG: u64 = 45;
pub const ROUND_CHALLENGE_TAG: u64 = 46;
pub const COMPACT_BINDING_TAG: u64 = 47;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointDims {
    pub assignment_width: usize,
    pub row_count: usize,
    pub variables: usize,
    pub matrix_count: usize,
    pub degree: usize,
}

pub fn build_joint_dims(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_count: usize,
    running_count: usize,
) -> Result<JointDims, PiCcsError> {
    if structure.n == 0 || structure.m == 0 {
        return Err(PiCcsError::InvalidInput(
            "PaddedRowIdentity requires nonzero CCS dimensions".into(),
        ));
    }
    if fresh_count == 0 {
        return Err(PiCcsError::InvalidInput(
            "PaddedRowIdentity requires at least one fresh source".into(),
        ));
    }
    if structure
        .matrices
        .iter()
        .flat_map(|matrix| matrix.seeded_phi81_blocks())
        .any(|block| block.has_superneo_transformed_columns())
    {
        return Err(PiCcsError::InvalidInput(
            "PaddedRowIdentity requires original, untransformed CCS matrices".into(),
        ));
    }

    let zero = vec![F::ZERO; structure.t()];
    if structure.f.eval(&zero) != F::ZERO {
        return Err(PiCcsError::InvalidInput(
            "the selected zero-row padding requires f(0,...,0)=0".into(),
        ));
    }

    build_joint_dims_for_shape(
        params,
        structure.n,
        structure.m,
        structure.t(),
        structure.max_degree(),
        fresh_count,
        running_count,
    )
}

/// Build the selected joint geometry from a matrix-independent relation
/// header. The caller must separately validate the concrete matrices and the
/// zero-padding identity `f(0,...,0)=0`.
pub fn build_joint_dims_for_shape(
    params: &NeoParams,
    rows: usize,
    columns: usize,
    matrix_count_without_identity: usize,
    max_degree: u32,
    fresh_count: usize,
    running_count: usize,
) -> Result<JointDims, PiCcsError> {
    if params.q != F::ORDER_U64 {
        return Err(PiCcsError::InvalidInput(format!(
            "PaddedRowIdentity parameter modulus {} does not match the Goldilocks field modulus {}",
            params.q,
            F::ORDER_U64,
        )));
    }
    if rows == 0 || columns == 0 {
        return Err(PiCcsError::InvalidInput(
            "PaddedRowIdentity requires nonzero CCS dimensions".into(),
        ));
    }
    if fresh_count == 0 {
        return Err(PiCcsError::InvalidInput(
            "PaddedRowIdentity requires at least one fresh source".into(),
        ));
    }
    if fresh_count > neo_params::goldilocks_paper_b2::MAX_FRESH_K as usize {
        return Err(PiCcsError::InvalidInput(format!(
            "PaddedRowIdentity fresh source count {fresh_count} exceeds {}",
            neo_params::goldilocks_paper_b2::MAX_FRESH_K
        )));
    }
    if running_count > params.k_rho as usize {
        return Err(PiCcsError::InvalidInput(format!(
            "PaddedRowIdentity running source count {running_count} exceeds k_rho={}",
            params.k_rho
        )));
    }

    let assignment_width = crate::common::superneo_carrier_width(columns);
    let row_count = rows
        .max(assignment_width)
        .checked_next_power_of_two()
        .ok_or_else(|| PiCcsError::InvalidInput("PaddedRowIdentity row domain overflows usize".into()))?
        .max(2);
    let variables = row_count.trailing_zeros() as usize;
    let matrix_count = matrix_count_without_identity
        .checked_add(1)
        .ok_or_else(|| PiCcsError::InvalidInput("matrix count overflow".into()))?;
    let ccs_degree = (max_degree as usize)
        .checked_add(1)
        .ok_or_else(|| PiCcsError::InvalidInput("PaddedRowIdentity degree overflows usize".into()))?;
    let norm_degree = 2usize
        .checked_mul(params.b as usize)
        .ok_or_else(|| PiCcsError::InvalidInput("PaddedRowIdentity norm degree overflows usize".into()))?;
    let degree = ccs_degree.max(norm_degree).max(2);

    params
        .padded_row_security_check_for_shape(
            rows,
            columns,
            matrix_count_without_identity,
            max_degree,
            neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )
        .map_err(|error| PiCcsError::ExtensionPolicyFailed(error.to_string()))?;

    Ok(JointDims {
        assignment_width,
        row_count,
        variables,
        matrix_count,
        degree,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceEvent {
    Absorb(Vec<F>),
    Challenge {
        label: u64,
        index: Option<usize>,
        value: K,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProtocolTrace {
    pub events: Vec<TraceEvent>,
    pub initial_claim: K,
    pub rounds: Vec<Vec<K>>,
    pub round_challenges: Vec<K>,
    pub terminal_claim: K,
    pub final_digest: [u8; 32],
}

#[inline]
pub fn gamma_power(gamma: K, exponent: usize) -> K {
    let mut power = K::ONE;
    for _ in 0..exponent {
        power *= gamma;
    }
    power
}

#[inline]
pub fn equality(point: &[K], target: &[K]) -> K {
    assert_eq!(point.len(), target.len(), "equality point length mismatch");
    point
        .iter()
        .zip(target)
        .fold(K::ONE, |product, (&left, &right)| {
            product * ((K::ONE - left) * (K::ONE - right) + left * right)
        })
}

/// Zero-based form of `2K+k+I(i,j,l)`.
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

pub fn range_product<Ff>(value: K, base: u32) -> K
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
