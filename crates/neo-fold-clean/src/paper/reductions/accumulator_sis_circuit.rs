//! Domain-separated SIS/Ajtai binding compression for fixed field sequences.
//!
//! Inputs reuse the relation's 41-digit balanced-ternary encoding, pack it
//! row-major into an Ajtai message matrix, and commit under a domain-specific
//! rank-2 seeded map. An independent rank-1 map compresses that short
//! commitment envelope before one Poseidon2 digest enters Fiat–Shamir.
//! Carried accumulator chains remain outside this module.
//!
//! Owns: canonical field-to-message encoding and both verifier-recomputed SIS
//! binding layers.
//!
//! Does not own: which claim fields are authoritative or transcript scheduling.
//!
//! Emits constraints: yes.
//!
//! Authority boundary: fields are inputs; commitments and digests are derived.
//!
//! | Constraint family | Equation/obligation | Per source field |
//! |---|---|---:|
//! | Digit alphabet | `d(d-1)=2q`, `q(d+1)=0` | 82 rows |
//! | Reconstruction | `x = sum d_i 3^i` | 1 row |
//! | Canonical borrow | shifted word is below Goldilocks `p` | 41 rows |
//! | Seeded Φ81 map | `A_seed · digits = commitment` | `D·kappa` rows per map |
//! | Digest envelope | Poseidon2 of domain/shape/short binding | input-dependent |

use std::ops::Range;

use neo_ajtai::{commit_row_major_seeded, seeded_pp_chunk_seeds, Commitment};
use neo_ccs::{Mat, SeededPhi81LinearBlock};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::r1cs_circuit::builder::{ShiftedTernaryCanonicalTrace, BALANCED_TERNARY_DIGITS};
use crate::engine::r1cs_circuit::{enforce_poseidon2_hash, BalancedTernaryOpeningTraceEntry, Lc, R1csBuilder, Var};
use crate::paper::digest::pack_bytes_as_fields;
use crate::paper::relations::product_commitment_circuit::{alloc_commitment, CommitmentWires};

const SIS_ACCUMULATOR_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/accumulator/sis/digest/v4";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SisAccumulatorConfig {
    pub seed: [u8; 32],
    pub kappa: usize,
    pub domain: u64,
}

/// Rank selected for protocol-binding compression maps. At the formal
/// security-model ceiling, rank 2 requires BKZ block size 495 under the
/// selected post-quantum profile.
pub const PROTOCOL_BINDING_KAPPA: usize = 2;

/// Largest rank-2 message accepted by `formal/ajtai-lean/Ajtai/EstimatorModel.lean`.
pub const PROTOCOL_BINDING_MAX_MESSAGE_COLS: usize = 50_371;

/// Largest rank-1 message covered by the pinned short-map estimate.
pub const DIGEST_COMPRESSION_MAX_MESSAGE_COLS: usize = 82;

/// Independent short-message map used between the rank-2 binding and
/// Poseidon2. Its input is at most one rank-2 commitment plus metadata, not
/// the original witness-proportional sequence.
pub const SIS_DIGEST_COMPRESSION_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC6; 32],
    kappa: 1,
    domain: 0x5349_535F_4447_5354,
};

pub const CCS_CLAIM_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC1; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x4343_535F_434C_4149,
};

pub const CE_CLAIM_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC2; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x4345_5F43_4C41_494D,
};

/// Construction-2 accumulator child binding. This stays distinct from the
/// narrower PiCCS CE transcript digest even though both start from CE fields.
pub const ACCUMULATOR_CE_CLAIM_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC7; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x4143_4345_5F43_4C4D,
};

/// One complete fixed-profile pending accumulator family. This map is
/// independent of the conservative per-child CE map because the two
/// serializers have different authority and field-order contracts.
pub const PENDING_ACCUMULATOR_FAMILY_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC8; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x5046_414D_5F41_4343,
};

pub const PI_CCS_OUTPUTS_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC3; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x5049_4343_535F_4F55,
};

pub const PI_RLC_PROJECTION_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC4; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x5049_524C_435F_5052,
};

pub const NEBULA_LEAF_SIS_CONFIG: SisAccumulatorConfig = SisAccumulatorConfig {
    seed: [0xC5; 32],
    kappa: PROTOCOL_BINDING_KAPPA,
    domain: 0x4E42_4C41_5F4C_4546,
};

#[derive(Debug, Error)]
pub enum SisAccumulatorError {
    #[error("SIS accumulator requires at least one input field")]
    EmptyInput,
    #[error("SIS accumulator kappa must be nonzero")]
    ZeroKappa,
    #[error("protocol-binding SIS maps support only kappa 1 or 2, got {kappa}")]
    UnsupportedKappa { kappa: usize },
    #[error(
        "rank-{kappa} SIS message has {field_count} fields; at most {max_field_count} fields fit the \
         security-model ceiling of {max_message_cols} ring columns"
    )]
    MessageTooWide {
        kappa: usize,
        field_count: usize,
        max_field_count: usize,
        max_message_cols: usize,
    },
}

pub struct SisAccumulatorWires {
    pub commitment: CommitmentWires,
    pub digest_compression: CommitmentWires,
    pub digest: [Var; 4],
    pub layout: SisAccumulatorCircuitLayout,
}

/// Exact source-R1CS frontier occupied by one SIS phase.
///
/// This is diagnostic provenance. It does not establish that the rows are
/// sound or authorize their removal; the source relation and its validated
/// encoding trace remain authoritative for those claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SisCircuitSpan {
    rows: Range<usize>,
    columns: Range<usize>,
    balanced_ternary_openings: Range<usize>,
}

impl SisCircuitSpan {
    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }

    pub fn columns(&self) -> Range<usize> {
        self.columns.clone()
    }

    pub fn balanced_ternary_openings(&self) -> Range<usize> {
        self.balanced_ternary_openings.clone()
    }
}

/// Three non-overlapping phases of the complete SIS-to-Poseidon2 binding.
///
/// | Phase | Mathematical obligation | Input authority |
/// |---|---|---|
/// | `input_binding` | canonical openings and seeded Φ81 binding of the caller's fields | caller-owned fields |
/// | `digest_compression` | independent short SIS binding of the first commitment | derived commitment |
/// | `envelope` | domain/shape envelope and Poseidon2 hash | derived digest-compression output |
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SisAccumulatorCircuitLayout {
    input_binding: SisCircuitSpan,
    digest_compression: SisCircuitSpan,
    envelope: SisCircuitSpan,
}

impl SisAccumulatorCircuitLayout {
    pub fn input_binding(&self) -> &SisCircuitSpan {
        &self.input_binding
    }

    pub fn digest_compression(&self) -> &SisCircuitSpan {
        &self.digest_compression
    }

    pub fn envelope(&self) -> &SisCircuitSpan {
        &self.envelope
    }
}

#[derive(Clone, Copy)]
struct CircuitFrontier {
    row: usize,
    column: usize,
    balanced_ternary_opening: usize,
}

impl CircuitFrontier {
    fn capture(builder: &R1csBuilder) -> Self {
        Self {
            row: builder.rows(),
            column: builder.cols(),
            balanced_ternary_opening: builder.encoding_trace().balanced_ternary_openings().len(),
        }
    }

    fn until(self, end: Self) -> SisCircuitSpan {
        SisCircuitSpan {
            rows: self.row..end.row,
            columns: self.column..end.column,
            balanced_ternary_openings: self.balanced_ternary_opening..end.balanced_ternary_opening,
        }
    }
}

pub fn accumulator_digest(config: SisAccumulatorConfig, fields: &[F]) -> Result<[F; 4], SisAccumulatorError> {
    let binding = commit_fields(config, fields)?;
    let digest_compression = commit_fields(SIS_DIGEST_COMPRESSION_CONFIG, &binding.data)?;
    Ok(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&digest_envelope(
        config,
        fields.len(),
        &binding,
        &digest_compression,
    )))
}

/// Column-contiguous encoding of the canonical SIS signed-unit message.
///
/// [`commit_fields`] fills the logical matrix row-major. Accelerator Ajtai
/// kernels consume one ring column at a time, so this helper performs that
/// exact layout transpose and pads the last column with zeroes.
#[doc(hidden)]
pub fn accelerator_balanced_ternary_message(fields: &[F]) -> Vec<i8> {
    let message_cols = (fields.len() * BALANCED_TERNARY_DIGITS).div_ceil(D);
    let mut message = vec![0i8; D * message_cols];
    for (field_index, &field) in fields.iter().enumerate() {
        for (digit_index, digit) in balanced_ternary_digits(field).into_iter().enumerate() {
            let index = field_index * BALANCED_TERNARY_DIGITS + digit_index;
            let row = index / message_cols;
            let column = index % message_cols;
            message[column * D + row] = if digit == F::ONE {
                1
            } else if digit == -F::ONE {
                -1
            } else {
                0
            };
        }
    }
    message
}

pub fn commit_fields(config: SisAccumulatorConfig, fields: &[F]) -> Result<Commitment, SisAccumulatorError> {
    validate(config, fields.len())?;
    let message = balanced_ternary_message(fields);
    Ok(commit_row_major_seeded(
        config.seed,
        D,
        config.kappa,
        message.cols(),
        &message,
    ))
}

pub fn enforce_commit_fields(
    builder: &mut R1csBuilder,
    config: SisAccumulatorConfig,
    fields: &[Var],
) -> Result<CommitmentWires, SisAccumulatorError> {
    validate(config, fields.len())?;
    let values: Vec<F> = fields
        .iter()
        .map(|field| builder.witness()[field.col()])
        .collect();
    let native = commit_fields(config, &values)?;
    let commitment = alloc_commitment(builder, &native);
    let digit_words: Vec<[Var; BALANCED_TERNARY_DIGITS]> = fields
        .iter()
        .map(|&field| decompose_var_to_balanced_ternary(builder, field))
        .collect();
    let word_starts = digit_words
        .iter()
        .map(|digits| {
            let start = digits[0].col();
            assert!(digits
                .iter()
                .enumerate()
                .all(|(digit, var)| var.col() == start + digit));
            start
        })
        .collect::<Vec<_>>();
    let message_cols = (fields.len() * BALANCED_TERNARY_DIGITS).div_ceil(D);
    let (chunk_size, chunk_seeds) = seeded_pp_chunk_seeds(config.seed, config.kappa, message_cols);
    let block = SeededPhi81LinearBlock::new_with_word_width(
        builder.rows(),
        word_starts,
        BALANCED_TERNARY_DIGITS,
        config.kappa,
        message_cols,
        chunk_size,
        chunk_seeds,
    )
    .expect("fixed seeded SIS geometry");
    builder.enforce_seeded_phi81_a_block(block, &commitment.data);
    Ok(commitment)
}

pub fn enforce_accumulator_digest(
    builder: &mut R1csBuilder,
    config: SisAccumulatorConfig,
    fields: &[Var],
) -> Result<SisAccumulatorWires, SisAccumulatorError> {
    let input_binding_start = CircuitFrontier::capture(builder);
    let commitment = enforce_commit_fields(builder, config, fields)?;
    let digest_compression_start = CircuitFrontier::capture(builder);
    builder.record_column_family("r1cs.sis_accumulator.input_binding", input_binding_start.column);
    let digest_compression = enforce_commit_fields(builder, SIS_DIGEST_COMPRESSION_CONFIG, &commitment.data)?;
    let envelope_start = CircuitFrontier::capture(builder);
    builder.record_column_family(
        "r1cs.sis_accumulator.digest_compression",
        digest_compression_start.column,
    );
    let digest_preimage = digest_envelope_wires(builder, config, fields.len(), &commitment, &digest_compression);
    let digest = enforce_poseidon2_hash(builder, &digest_preimage);
    let end = CircuitFrontier::capture(builder);
    builder.record_column_family("r1cs.sis_accumulator.envelope", envelope_start.column);
    Ok(SisAccumulatorWires {
        commitment,
        digest_compression,
        digest,
        layout: SisAccumulatorCircuitLayout {
            input_binding: input_binding_start.until(digest_compression_start),
            digest_compression: digest_compression_start.until(envelope_start),
            envelope: envelope_start.until(end),
        },
    })
}

fn digest_envelope(
    config: SisAccumulatorConfig,
    field_count: usize,
    binding: &Commitment,
    digest_compression: &Commitment,
) -> Vec<F> {
    let mut envelope = accumulator_digest_envelope_prefix(config, field_count);
    debug_assert_eq!(binding.kappa, config.kappa);
    envelope.extend_from_slice(&digest_compression.data);
    envelope
}

/// Verifier-owned constant prefix placed before the short rank-1 commitment.
/// Accelerator backends and physical audits use this canonical definition
/// instead of copying protocol-domain constants.
#[doc(hidden)]
pub fn accumulator_digest_envelope_prefix(config: SisAccumulatorConfig, field_count: usize) -> Vec<F> {
    let mut envelope = pack_bytes_as_fields(SIS_ACCUMULATOR_DIGEST_DOMAIN);
    envelope.push(F::from_u64(config.domain));
    envelope.push(F::from_u64(field_count as u64));
    envelope.push(F::from_u64(config.kappa as u64));
    envelope
}

fn digest_envelope_wires(
    builder: &mut R1csBuilder,
    config: SisAccumulatorConfig,
    field_count: usize,
    binding: &CommitmentWires,
    digest_compression: &CommitmentWires,
) -> Vec<Var> {
    debug_assert_eq!(binding.data.len(), D * config.kappa);
    let mut envelope = accumulator_digest_envelope_prefix(config, field_count)
        .into_iter()
        .map(|value| alloc_constant(builder, value))
        .collect::<Vec<_>>();
    envelope.extend_from_slice(&digest_compression.data);
    envelope
}

fn balanced_ternary_message(fields: &[F]) -> Mat<F> {
    let message_cols = (fields.len() * BALANCED_TERNARY_DIGITS).div_ceil(D);
    let mut message = Mat::zero(D, message_cols, F::ZERO);
    for (field_index, field) in fields.iter().enumerate() {
        for (digit, value) in balanced_ternary_digits(*field).into_iter().enumerate() {
            let index = field_index * BALANCED_TERNARY_DIGITS + digit;
            message[(index / message_cols, index % message_cols)] = value;
        }
    }
    message
}

fn decompose_var_to_balanced_ternary(builder: &mut R1csBuilder, field: Var) -> [Var; BALANCED_TERNARY_DIGITS] {
    if let Some(digits) = builder.balanced_ternary_decomposition(field) {
        return digits;
    }
    let values = balanced_ternary_digits(builder.witness()[field.col()]);
    let digits = values.map(|value| builder.alloc(value));
    let digit_rows_start = builder.rows();
    let negative_indicators = digits.map(|digit| {
        let value = builder.witness()[digit.col()];
        let is_negative = value == -F::ONE;
        debug_assert!(is_negative || value == F::ZERO || value == F::ONE);
        let negative = builder.alloc(if is_negative { F::ONE } else { F::ZERO });
        builder.record_boolean(negative);

        // `q = d(d-1)/2` is the exact indicator for `d = -1` on the
        // centered alphabet. The second row completes `d(d-1)(d+1) = 0`.
        let mut minus_one = Lc::from_var(digit);
        minus_one.add_constant(-F::ONE);
        let twice_negative = Lc::zero().add_scaled(&Lc::from_var(negative), F::from_u64(2));
        builder.enforce(&Lc::from_var(digit), &minus_one, &twice_negative);
        let mut plus_one = Lc::from_var(digit);
        plus_one.add_constant(F::ONE);
        builder.enforce(&Lc::from_var(negative), &plus_one, &Lc::zero());
        builder.record_centered_unit(digit);
        negative
    });
    let mut reconstruction = Lc::zero();
    let mut power = F::ONE;
    for digit in digits {
        reconstruction.add_term(digit, power);
        power *= F::from_u64(3);
    }
    let digit_rows_end = builder.rows();
    let reconstruction_row = builder.rows();
    builder.enforce_eq(&Lc::from_var(field), &reconstruction);
    let transition_rows_start = builder.rows();
    let borrows = enforce_shifted_base3_canonical(builder, &digits, &negative_indicators);
    let transition_rows_end = builder.rows();
    builder.record_shifted_ternary_canonical_trace(ShiftedTernaryCanonicalTrace {
        field_column: field.col(),
        digit_columns_start: digits[0].col(),
        negative_columns_start: negative_indicators[0].col(),
        borrow_columns_start: borrows[0].col(),
        digit_rows_start,
        reconstruction_row,
        transition_rows_start,
    });
    builder.record_balanced_ternary_decomposition(field, digits);
    builder.record_balanced_ternary_opening_encoding(BalancedTernaryOpeningTraceEntry {
        field_col: field.col(),
        digit_cols: digits.map(Var::col),
        negative_cols: negative_indicators.map(Var::col),
        borrow_cols: borrows.map(Var::col),
        digit_rows: digit_rows_start..digit_rows_end,
        reconstruction_row,
        transition_rows: transition_rows_start..transition_rows_end,
    });
    digits
}

/// Select one low-norm opening for every field residue without a 64-bit
/// comparator. Let `M = (3^41-1)/2` and `t_i = d_i+1 ∈ {0,1,2}`. Native
/// encoding chooses the ordinary base-3 expansion
///
/// `N = (x + M) mod p = sum_i t_i 3^i`, with `0 <= N < p`.
///
/// Reconstruction already proves `x = sum_i d_i 3^i (mod p)`. The borrow
/// chain below proves `N <= p-1`, so ordinary base-3 uniqueness rules out the
/// alternative `N+p` opening. Each recurrence also inductively pins its next
/// borrow to a bit; the recorded Boolean width is therefore derived, not
/// trusted as a protocol conclusion.
fn enforce_shifted_base3_canonical(
    builder: &mut R1csBuilder,
    digits: &[Var; BALANCED_TERNARY_DIGITS],
    negative_indicators: &[Var; BALANCED_TERNARY_DIGITS],
) -> [Var; BALANCED_TERNARY_DIGITS - 1] {
    let mut bound = F::ORDER_U64 - 1;
    let mut borrow_value = false;
    let mut borrow = Lc::zero();
    let mut borrow_vars = Vec::with_capacity(BALANCED_TERNARY_DIGITS - 1);

    for index in 0..BALANCED_TERNARY_DIGITS {
        let bound_digit = bound % 3;
        bound /= 3;

        let digit_value = builder.witness()[digits[index].col()];
        let trit = if digit_value == -F::ONE {
            0
        } else if digit_value == F::ZERO {
            1
        } else {
            debug_assert_eq!(digit_value, F::ONE);
            2
        };
        let next_borrow_value = trit + u64::from(borrow_value) > bound_digit;
        let next_borrow_var = if index + 1 == BALANCED_TERNARY_DIGITS {
            debug_assert!(!next_borrow_value, "honest shifted-base-3 opening must be below p");
            None
        } else {
            let next = builder.alloc(if next_borrow_value { F::ONE } else { F::ZERO });
            builder.record_boolean(next);
            borrow_vars.push(next);
            Some(next)
        };
        let next_borrow = next_borrow_var.map_or_else(Lc::zero, Lc::from_var);

        let negative = Lc::from_var(negative_indicators[index]);
        let positive = Lc::from_var(digits[index]).add_scaled(&negative, F::ONE);
        let zero = Lc::from_const(F::ONE)
            .add_scaled(&Lc::from_var(digits[index]), -F::ONE)
            .add_scaled(&negative, -F::from_u64(2));

        match bound_digit {
            // b' = 1 unless t=0 and b=0.
            0 => {
                let one_minus_borrow = Lc::from_const(F::ONE).add_scaled(&borrow, -F::ONE);
                let one_minus_next = Lc::from_const(F::ONE).add_scaled(&next_borrow, -F::ONE);
                builder.enforce(&negative, &one_minus_borrow, &one_minus_next);
            }
            // b' = isPositive(t) + isZero(t) * b.
            1 => {
                let rhs = next_borrow.clone().add_scaled(&positive, -F::ONE);
                builder.enforce(&zero, &borrow, &rhs);
            }
            // b' = isPositive(t) * b.
            2 => builder.enforce(&positive, &borrow, &next_borrow),
            _ => unreachable!("base-3 digit"),
        }

        borrow = next_borrow;
        borrow_value = next_borrow_value;
    }
    debug_assert_eq!(bound, 0, "41 base-3 digits must cover p-1");
    borrow_vars
        .try_into()
        .expect("one borrow variable per non-terminal ternary digit")
}

fn balanced_ternary_digits(value: F) -> [F; BALANCED_TERNARY_DIGITS] {
    let modulus = F::ORDER_U64 as u128;
    let shift = (3u128.pow(BALANCED_TERNARY_DIGITS as u32) - 1) / 2;
    let mut remaining = (value.as_canonical_u64() as u128 + shift) % modulus;
    let digits = core::array::from_fn(|_| {
        let trit = remaining % 3;
        remaining /= 3;
        match trit {
            0 => -F::ONE,
            1 => F::ZERO,
            2 => F::ONE,
            _ => unreachable!("base-3 digit"),
        }
    });
    assert_eq!(remaining, 0, "shifted Goldilocks residue must fit in 41 base-3 digits");
    digits
}

fn validate(config: SisAccumulatorConfig, field_count: usize) -> Result<(), SisAccumulatorError> {
    if field_count == 0 {
        return Err(SisAccumulatorError::EmptyInput);
    }
    if config.kappa == 0 {
        return Err(SisAccumulatorError::ZeroKappa);
    }
    let max_message_cols = match config.kappa {
        1 => DIGEST_COMPRESSION_MAX_MESSAGE_COLS,
        PROTOCOL_BINDING_KAPPA => PROTOCOL_BINDING_MAX_MESSAGE_COLS,
        kappa => return Err(SisAccumulatorError::UnsupportedKappa { kappa }),
    };
    let max_field_count = max_message_cols * D / BALANCED_TERNARY_DIGITS;
    if field_count > max_field_count {
        return Err(SisAccumulatorError::MessageTooWide {
            kappa: config.kappa,
            field_count,
            max_field_count,
            max_message_cols,
        });
    }
    Ok(())
}

fn alloc_constant(builder: &mut R1csBuilder, value: F) -> Var {
    let wire = builder.alloc(value);
    builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(value));
    wire
}
