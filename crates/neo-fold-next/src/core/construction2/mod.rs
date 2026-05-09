//! Owns relation-neutral HyperNova Construction-2 public-image primitives.
//!
//! This is not a SuperNeo folding stage. It is the `enc_inst(hash(...))`
//! image used by recursive F' carriers to bind one iteration to the next.

use neo_ajtai::Commitment;
use neo_math::{D, F};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

pub(crate) mod terminal;

pub const CONSTRUCTION2_ENC_INST_BITS: usize = 256;
pub const CONSTRUCTION2_ENC_INST_RING_DEGREE: usize = D;
pub const CONSTRUCTION2_ENC_INST_RING_SLOTS: usize =
    (CONSTRUCTION2_ENC_INST_BITS + CONSTRUCTION2_ENC_INST_RING_DEGREE - 1) / CONSTRUCTION2_ENC_INST_RING_DEGREE;
pub const CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG: u64 = 0x6e66_7332_7075_6269;
pub const CONSTRUCTION2_COMMITMENT_RAW_TAG: u64 = 0x6e66_7332_636f_6d6d;

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Construction2Commitment(Commitment);

impl Construction2Commitment {
    pub fn commitment(&self) -> &Commitment {
        &self.0
    }

    pub fn from_commitment(commitment: Commitment) -> Self {
        Self(commitment)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Construction2FreshInstance {
    c_i: Construction2Commitment,
    x_i: Construction2EncodedPublicInput,
}

impl Construction2FreshInstance {
    pub fn from_parts(c_i: Construction2Commitment, x_i: Construction2EncodedPublicInput) -> Self {
        Self { c_i, x_i }
    }

    pub fn x_only_placeholder(x_i: Construction2EncodedPublicInput) -> Self {
        Self {
            c_i: Construction2Commitment::from_commitment(Commitment::zeros(D, 1)),
            x_i,
        }
    }

    pub fn canonical_zero(kappa: usize, x_i: Construction2EncodedPublicInput) -> Self {
        Self {
            c_i: Construction2Commitment::from_commitment(Commitment::zeros(D, kappa)),
            x_i,
        }
    }

    pub fn is_x_only_placeholder_for(&self, x_i: &Construction2EncodedPublicInput) -> bool {
        let commitment = self.c_i.commitment();
        self.x_i == *x_i
            && commitment.d == D
            && commitment.kappa == 1
            && commitment.data.len() == D
            && commitment.data.iter().all(|value| *value == F::ZERO)
    }

    pub fn is_canonical_zero_for(&self, kappa: usize, x_i: &Construction2EncodedPublicInput) -> bool {
        let commitment = self.c_i.commitment();
        self.x_i == *x_i
            && kappa != 0
            && commitment.d == D
            && commitment.kappa == kappa
            && commitment.data.len() == D * kappa
            && commitment.data.iter().all(|value| *value == F::ZERO)
    }

    pub fn from_public_boundary(boundary: &Construction2PublicBoundary) -> Result<Self, String> {
        if !boundary.has_canonical_commitment_shape() {
            return Err("Construction-2 public boundary commitment shape is not canonical".into());
        }
        if boundary.commitment_digest != boundary.expected_commitment_digest() {
            return Err("Construction-2 public boundary commitment digest is stale".into());
        }
        if boundary.fresh_instance_digest != boundary.expected_fresh_instance_digest() {
            return Err("Construction-2 public boundary fresh-instance digest is stale".into());
        }
        Ok(Self {
            c_i: Construction2Commitment::from_commitment(Commitment {
                d: boundary.commitment_d as usize,
                kappa: boundary.commitment_kappa as usize,
                data: boundary.commitment_data.clone(),
            }),
            x_i: boundary.x_i.clone(),
        })
    }

    pub fn commitment(&self) -> &Construction2Commitment {
        &self.c_i
    }

    pub fn x_i(&self) -> &Construction2EncodedPublicInput {
        &self.x_i
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let encoded = bincode::serialize(self).expect("construction2 fresh instance encodes");
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/construction2/fresh_instance");
        tr.append_message(b"neo.fold.next/construction2/fresh_instance/version", b"v1");
        tr.append_message(b"neo.fold.next/construction2/fresh_instance/encoded", &encoded);
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Construction2PublicBoundary {
    pub fresh_instance_digest: [u8; 32],
    pub commitment_digest: [u8; 32],
    pub commitment_d: u64,
    pub commitment_kappa: u64,
    pub commitment_data: Vec<F>,
    pub x_i: Construction2EncodedPublicInput,
}

impl Construction2PublicBoundary {
    pub fn from_fresh_instance(instance: &Construction2FreshInstance) -> Self {
        let commitment_digest = construction2_commitment_digest(instance.commitment());
        let commitment = instance.commitment().commitment();
        Self {
            fresh_instance_digest: construction2_public_boundary_fresh_instance_digest(
                commitment_digest,
                instance.x_i(),
            ),
            commitment_digest,
            commitment_d: commitment.d as u64,
            commitment_kappa: commitment.kappa as u64,
            commitment_data: commitment.data.clone(),
            x_i: instance.x_i().clone(),
        }
    }

    pub fn expected_commitment_digest(&self) -> [u8; 32] {
        construction2_commitment_digest_from_parts(self.commitment_d, self.commitment_kappa, &self.commitment_data)
    }

    pub fn has_canonical_commitment_shape(&self) -> bool {
        self.commitment_d == D as u64
            && self.commitment_kappa != 0
            && self
                .commitment_d
                .checked_mul(self.commitment_kappa)
                .is_some_and(|len| self.commitment_data.len() as u64 == len)
    }

    pub fn expected_fresh_instance_digest(&self) -> [u8; 32] {
        construction2_public_boundary_fresh_instance_digest(self.expected_commitment_digest(), &self.x_i)
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/construction2/public_boundary");
        tr.append_message(b"neo.fold.next/construction2/public_boundary/version", b"v1");
        tr.append_message(
            b"neo.fold.next/construction2/public_boundary/fresh_instance_digest",
            &self.fresh_instance_digest,
        );
        tr.append_message(
            b"neo.fold.next/construction2/public_boundary/commitment_digest",
            &self.commitment_digest,
        );
        tr.append_fields(
            b"neo.fold.next/construction2/public_boundary/commitment_shape",
            &[F::from_u64(self.commitment_d), F::from_u64(self.commitment_kappa)],
        );
        tr.append_fields(
            b"neo.fold.next/construction2/public_boundary/commitment_data",
            &self.commitment_data,
        );
        tr.append_message(b"neo.fold.next/construction2/public_boundary/x_i", &self.x_i.bytes());
        tr.digest32()
    }
}

fn construction2_commitment_digest(commitment: &Construction2Commitment) -> [u8; 32] {
    let commitment = commitment.commitment();
    construction2_commitment_digest_from_parts(commitment.d as u64, commitment.kappa as u64, &commitment.data)
}

pub fn construction2_commitment_digest_from_parts(d: u64, kappa: u64, data: &[F]) -> [u8; 32] {
    let mut preimage = Vec::with_capacity(3 + data.len());
    preimage.push(F::from_u64(CONSTRUCTION2_COMMITMENT_RAW_TAG));
    preimage.push(F::from_u64(d));
    preimage.push(F::from_u64(kappa));
    preimage.extend_from_slice(data);
    crate::finalize::digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}

pub fn construction2_public_boundary_fresh_instance_digest(
    commitment_digest: [u8; 32],
    x_i: &Construction2EncodedPublicInput,
) -> [u8; 32] {
    let mut preimage = Vec::with_capacity(9);
    preimage.push(F::from_u64(CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG));
    preimage.extend(crate::finalize::digest32_as_fields(commitment_digest));
    preimage.extend(crate::finalize::digest32_as_fields(x_i.bytes()));
    crate::finalize::digest_fields_as_digest32(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage))
}
