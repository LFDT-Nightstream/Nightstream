//! Owns relation-neutral HyperNova Construction-2 public-image primitives.
//!
//! This is not a SuperNeo folding stage. It is the `enc_inst(hash(...))`
//! image used by recursive F' carriers to bind one iteration to the next.

use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

pub const CONSTRUCTION2_ENC_INST_BITS: usize = 256;
pub const CONSTRUCTION2_ENC_INST_RING_DEGREE: usize = D;
pub const CONSTRUCTION2_ENC_INST_RING_SLOTS: usize =
    (CONSTRUCTION2_ENC_INST_BITS + CONSTRUCTION2_ENC_INST_RING_DEGREE - 1) / CONSTRUCTION2_ENC_INST_RING_DEGREE;

/// Canonical Construction-2 encoded public input image.
///
/// The semantic `enc_inst` image is the digest bit-decomposition in little-
/// endian bit order, one low-norm field element per bit:
/// `bit_j = (digest_bytes[j / 8] >> (j % 8)) & 1`.
///
/// Public IO surfaces serialize the digest bytes as four canonical field limbs:
/// `limb_j = u64::from_le_bytes(digest_bytes[8*j .. 8*(j+1)])`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Construction2EncodedPublicInput {
    digest_bytes: [u8; 32],
}

fn enc_inst_bit_image_le(digest_bytes: [u8; 32]) -> [u8; CONSTRUCTION2_ENC_INST_BITS] {
    core::array::from_fn(|bit_index| {
        let byte = digest_bytes[bit_index / 8];
        (byte >> (bit_index % 8)) & 1
    })
}

impl Construction2EncodedPublicInput {
    pub fn from_digest_bytes(digest_bytes: [u8; 32]) -> Self {
        Self { digest_bytes }
    }

    pub fn bytes(&self) -> [u8; 32] {
        self.digest_bytes
    }

    pub fn bytes_mut(&mut self) -> &mut [u8; 32] {
        &mut self.digest_bytes
    }

    pub fn bit_image(&self) -> [u8; CONSTRUCTION2_ENC_INST_BITS] {
        enc_inst_bit_image_le(self.digest_bytes)
    }

    pub fn field_image(&self) -> [F; CONSTRUCTION2_ENC_INST_BITS] {
        self.bit_image().map(|bit| F::from_u64(bit as u64))
    }

    pub fn ring_image(&self) -> [[F; CONSTRUCTION2_ENC_INST_RING_DEGREE]; CONSTRUCTION2_ENC_INST_RING_SLOTS] {
        let field_image = self.field_image();
        core::array::from_fn(|ring_slot| {
            core::array::from_fn(|coeff_index| {
                field_image
                    .get(ring_slot * CONSTRUCTION2_ENC_INST_RING_DEGREE + coeff_index)
                    .copied()
                    .unwrap_or(F::ZERO)
            })
        })
    }

    pub fn is_binary_low_norm(&self) -> bool {
        self.bit_image().into_iter().all(|bit| bit <= 1)
    }
}
