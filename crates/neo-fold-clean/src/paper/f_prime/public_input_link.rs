//! Native ownership of the F' public-input shape and delayed-link checker.
//!
//! Owns the logical/physical carrier dimensions, the exact typed native
//! verifier schedule, and its source interpreter. It does not own R1CS row
//! emission, application-suffix semantics, or `x_out` computation.

use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::poseidon2::DIGEST_LEN;
use crate::paper::construction2::EncInst;

/// Canonical bits per `x_out` digest lane. Goldilocks canonical form fits
/// in 64 bits.
pub const X_OUT_BITS_PER_LANE: usize = 64;

/// Number of `enc_inst(x_out)` bits — the bit-decomposed digest body.
pub const F_PRIME_ENC_INST_BITS: usize = DIGEST_LEN * X_OUT_BITS_PER_LANE;

/// Index of the constant-one slot in the F' CCS public input.
pub const F_PRIME_PUBLIC_ONE_OFFSET: usize = 0;

/// First index of the `enc_inst(x_out)` body inside the F' public input.
pub const F_PRIME_ENC_INST_OFFSET: usize = 1;

/// Logical F' public-input length: `[1, enc_inst(x_out)…]`.
///
/// `enc_inst` is the public-instance encoding boundary. Internal F' field
/// values remain ordinary field values; only the next fresh CCS input is
/// bit-decomposed so the public input is low-norm under `b = 2`.
pub const F_PRIME_PUBLIC_INPUT_LEN: usize = 1 + F_PRIME_ENC_INST_BITS;

/// Complete plain SuperNeo carrier width for the logical F' public input.
pub const F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN: usize = (F_PRIME_PUBLIC_INPUT_LEN + D - 1) / D * D;

/// Verifier-owned public-input shape for one F' relation.
///
/// Plain F' uses no suffix. A composed application may append public step
/// data after `enc_inst(x_out)`; the next recursive step receives those
/// coordinates without treating them as part of the hash link itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FPrimePublicInputLayout {
    suffix_len: usize,
}

impl FPrimePublicInputLayout {
    pub const fn plain() -> Self {
        Self { suffix_len: 0 }
    }

    pub const fn with_suffix(suffix_len: usize) -> Self {
        Self { suffix_len }
    }

    /// Logical fields carried before the verifier-fixed ring padding.
    pub const fn logical_len(self) -> usize {
        F_PRIME_PUBLIC_INPUT_LEN + self.suffix_len
    }

    /// Complete public carrier consumed by SuperNeo.
    pub const fn total_len(self) -> usize {
        let logical = self.logical_len();
        (logical + D - 1) / D * D
    }

    pub const fn carrier_padding_len(self) -> usize {
        self.total_len() - self.logical_len()
    }

    pub const fn suffix_len(self) -> usize {
        self.suffix_len
    }

    pub const fn suffix_offset(self) -> usize {
        F_PRIME_PUBLIC_INPUT_LEN
    }

    pub const fn suffix_end(self) -> usize {
        self.suffix_offset() + self.suffix_len
    }

    pub const fn carrier_padding_offset(self) -> usize {
        self.logical_len()
    }
}

/// Typed verifier schedule for one delayed F' public-input link.
///
/// Production evaluation and the Rust-to-Lean exporter consume the same
/// six-instruction value. Range instructions retain coordinate order and
/// length without materializing one instruction per bit.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FPrimePublicInputLinkInstruction {
    ExpectedPublicInputLen {
        expected: usize,
    },
    ClaimMIn {
        expected: usize,
    },
    ClaimXLen {
        expected: usize,
    },
    AffineOne {
        claim_index: usize,
    },
    BodyRange {
        expected_offset: usize,
        claim_offset: usize,
        len: usize,
    },
    PaddingZeroRange {
        claim_offset: usize,
        len: usize,
    },
}

impl FPrimePublicInputLinkInstruction {
    /// Definitional number of scalar checks represented by this instruction.
    #[doc(hidden)]
    pub const fn scalar_obligation_count(self) -> usize {
        match self {
            Self::ExpectedPublicInputLen { .. }
            | Self::ClaimMIn { .. }
            | Self::ClaimXLen { .. }
            | Self::AffineOne { .. } => 1,
            Self::BodyRange { len, .. } | Self::PaddingZeroRange { len, .. } => len,
        }
    }

    fn matches(
        self,
        expected_bits: &[u8; F_PRIME_ENC_INST_BITS],
        expected_public_input_len: usize,
        claim_m_in: usize,
        claim_x: &[F],
    ) -> bool {
        match self {
            Self::ExpectedPublicInputLen { expected } => expected_public_input_len == expected,
            Self::ClaimMIn { expected } => claim_m_in == expected,
            Self::ClaimXLen { expected } => claim_x.len() == expected,
            Self::AffineOne { claim_index } => claim_x
                .get(claim_index)
                .is_some_and(|value| *value == F::ONE),
            Self::BodyRange {
                expected_offset,
                claim_offset,
                len,
            } => {
                let Some(expected_end) = expected_offset.checked_add(len) else {
                    return false;
                };
                let Some(claim_end) = claim_offset.checked_add(len) else {
                    return false;
                };
                let Some(expected) = expected_bits.get(expected_offset..expected_end) else {
                    return false;
                };
                let Some(actual) = claim_x.get(claim_offset..claim_end) else {
                    return false;
                };
                expected.iter().zip(actual).all(|(&bit, &value)| {
                    let expected_value = if bit == 0 { F::ZERO } else { F::ONE };
                    value == expected_value
                })
            }
            Self::PaddingZeroRange { claim_offset, len } => claim_x
                .get(claim_offset..)
                .is_some_and(|padding| padding.len() == len && padding.iter().all(|value| *value == F::ZERO)),
        }
    }
}

/// Exact ordered program interpreted by
/// [`f_prime_public_input_link_matches`].
#[doc(hidden)]
pub fn f_prime_public_input_link_program(layout: FPrimePublicInputLayout) -> [FPrimePublicInputLinkInstruction; 6] {
    let total_len = layout.total_len();
    [
        FPrimePublicInputLinkInstruction::ExpectedPublicInputLen { expected: total_len },
        FPrimePublicInputLinkInstruction::ClaimMIn { expected: total_len },
        FPrimePublicInputLinkInstruction::ClaimXLen { expected: total_len },
        FPrimePublicInputLinkInstruction::AffineOne {
            claim_index: F_PRIME_PUBLIC_ONE_OFFSET,
        },
        FPrimePublicInputLinkInstruction::BodyRange {
            expected_offset: 0,
            claim_offset: F_PRIME_ENC_INST_OFFSET,
            len: F_PRIME_ENC_INST_BITS,
        },
        FPrimePublicInputLinkInstruction::PaddingZeroRange {
            claim_offset: layout.carrier_padding_offset(),
            len: layout.carrier_padding_len(),
        },
    ]
}

/// Three-phase row-emission schema for every claim in a terminal fresh batch.
///
/// The production decider expands this exact value in claim-major order. A
/// range is one typed instruction while its scalar cost is its exact length.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FPrimeTerminalLinkInstruction {
    AffineOne {
        claim_offset: usize,
    },
    BodyRange {
        claim_offset: usize,
        producer_offset: usize,
        len: usize,
    },
    PaddingZeroRange {
        claim_offset: usize,
        len: usize,
    },
}

impl FPrimeTerminalLinkInstruction {
    #[doc(hidden)]
    pub const fn scalar_obligation_count(self) -> usize {
        match self {
            Self::AffineOne { .. } => 1,
            Self::BodyRange { len, .. } | Self::PaddingZeroRange { len, .. } => len,
        }
    }
}

/// Exact claim-local terminal-link program interpreted by the R1CS emitter.
#[doc(hidden)]
pub fn f_prime_terminal_link_program(layout: FPrimePublicInputLayout) -> [FPrimeTerminalLinkInstruction; 3] {
    [
        FPrimeTerminalLinkInstruction::AffineOne {
            claim_offset: F_PRIME_PUBLIC_ONE_OFFSET,
        },
        FPrimeTerminalLinkInstruction::BodyRange {
            claim_offset: F_PRIME_ENC_INST_OFFSET,
            producer_offset: 0,
            len: F_PRIME_ENC_INST_BITS,
        },
        FPrimeTerminalLinkInstruction::PaddingZeroRange {
            claim_offset: layout.carrier_padding_offset(),
            len: layout.carrier_padding_len(),
        },
    ]
}

/// Native verifier predicate for one delayed F' public link.
///
/// The logical prefix is `[1 | enc_inst(expected)]`. A composed profile owns
/// any application suffix separately; this predicate checks the shape,
/// affine coordinate, ordered body, and trailing ring-carrier padding.
/// Accepting a typed [`EncInst`] prevents callers from substituting a free raw
/// bit vector after computing `x_out`.
#[doc(hidden)]
pub fn f_prime_public_input_link_matches(
    layout: FPrimePublicInputLayout,
    expected: &EncInst,
    expected_public_input_len: usize,
    claim_m_in: usize,
    claim_x: &[F],
) -> bool {
    let expected_bits = expected.bits();
    f_prime_public_input_link_program(layout)
        .into_iter()
        .all(|instruction| instruction.matches(&expected_bits, expected_public_input_len, claim_m_in, claim_x))
}
