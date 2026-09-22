//! Canonical field layout of the `Pi_CCS` output message bound before `Pi_RLC`.
//!
//! Owns: the protocol -> source -> vector -> lane -> limb ordering, exact
//! field counts, and the inverse map from a pre-SIS field index to one leaf.
//!
//! Does not own: output truth, SIS/Poseidon2 binding, transcript placement,
//! constraint emission, or permission to remove rows.
//!
//! Emits constraints: no.
//!
//! Authority boundary: this is lossless layout metadata. A field path names
//! where a value came from; it does not prove that the named output is valid.
//!
//! | Stage path | Mathematical object | R1CS input owner |
//! |---|---|---|
//! | `nifs.pi_ccs.output_message_hashes.digest.preimage.outer_header` | outer domain and exact source count | verifier-owned constant |
//! | `nifs.pi_ccs.output_message_hashes.digest.preimage.source_headers` | per-source domain and matrix count | verifier-owned constant |
//! | `nifs.pi_ccs.output_message_hashes.digest.preimage.y_ring` | identity-first, matrix-major active `K` vectors | accepted one-joint output wire |

use neo_math::ring::D;

pub const OUTPUTS_DOMAIN: &[u8] = b"neo.fold.clean/pi_ccs_outputs_digest/v3";
pub const OUTPUT_MESSAGE_DOMAIN: &[u8] = b"neo.fold.clean/pi_ccs_output_message_digest/v3";

pub const ACTIVE_F_PRIME_SOURCE_COUNT: usize = 15;
pub const ACTIVE_F_PRIME_MATRIX_COUNT: usize = 14;
pub const ACTIVE_F_PRIME_FIELD_COUNT: usize = 23_033;
pub const LEGACY_THREE_MATRIX_FIELD_COUNT: usize = 5_048;

const K_LIMBS: usize = 2;
const PACKED_BYTES_PER_FIELD: usize = 7;

const fn packed_domain_field_count(bytes: &[u8]) -> usize {
    1 + bytes.len().div_ceil(PACKED_BYTES_PER_FIELD)
}

pub const OUTPUTS_DOMAIN_FIELD_COUNT: usize = packed_domain_field_count(OUTPUTS_DOMAIN);
pub const OUTPUT_MESSAGE_DOMAIN_FIELD_COUNT: usize = packed_domain_field_count(OUTPUT_MESSAGE_DOMAIN);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KLimb {
    C0,
    C1,
}

impl KLimb {
    const fn from_index(index: usize) -> Self {
        match index {
            0 => Self::C0,
            1 => Self::C1,
            _ => unreachable!(),
        }
    }
}

/// The R1CS surface that owns one serializer input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum R1csInputOwner {
    VerifierShape,
    YRingOutput,
}

/// One leaf in the exact pre-SIS field order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldPath {
    OutputsDomain {
        field: usize,
    },
    SourceCount,
    SourceDomain {
        source: usize,
        field: usize,
    },
    MatrixCount {
        source: usize,
    },
    YRingWidth {
        source: usize,
        matrix: usize,
    },
    YRingLimb {
        source: usize,
        matrix: usize,
        lane: usize,
        limb: KLimb,
    },
}

impl FieldPath {
    pub const fn r1cs_input_owner(self) -> R1csInputOwner {
        match self {
            Self::OutputsDomain { .. }
            | Self::SourceCount
            | Self::SourceDomain { .. }
            | Self::MatrixCount { .. }
            | Self::YRingWidth { .. } => R1csInputOwner::VerifierShape,
            Self::YRingLimb { .. } => R1csInputOwner::YRingOutput,
        }
    }
}

/// Shape of one complete output message. The active lane count is the fixed
/// Phi81 ring degree `D`; padded implementation lanes are not serialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    source_count: usize,
    matrix_count: usize,
}

impl Profile {
    pub const fn new(source_count: usize, matrix_count: usize) -> Self {
        Self {
            source_count,
            matrix_count,
        }
    }

    pub const fn active_f_prime() -> Self {
        Self::new(ACTIVE_F_PRIME_SOURCE_COUNT, ACTIVE_F_PRIME_MATRIX_COUNT)
    }

    pub const fn source_count(self) -> usize {
        self.source_count
    }

    pub const fn matrix_count(self) -> usize {
        self.matrix_count
    }

    pub const fn lane_count(self) -> usize {
        D
    }

    pub const fn k_vector_field_count(self) -> usize {
        1 + K_LIMBS * D
    }

    pub const fn source_field_count(self) -> usize {
        OUTPUT_MESSAGE_DOMAIN_FIELD_COUNT + 1 + self.matrix_count * self.k_vector_field_count()
    }

    pub const fn field_count(self) -> usize {
        OUTPUTS_DOMAIN_FIELD_COUNT + 1 + self.source_count * self.source_field_count()
    }

    /// Inverse of the serializer order. Every index below `field_count()`
    /// resolves to exactly one path and every other index is rejected.
    pub fn decode(self, index: usize) -> Option<FieldPath> {
        let mut offset = index;
        if offset < OUTPUTS_DOMAIN_FIELD_COUNT {
            return Some(FieldPath::OutputsDomain { field: offset });
        }
        offset -= OUTPUTS_DOMAIN_FIELD_COUNT;

        if offset == 0 {
            return Some(FieldPath::SourceCount);
        }
        offset -= 1;

        let source_width = self.source_field_count();
        let source = offset / source_width;
        if source >= self.source_count {
            return None;
        }
        offset %= source_width;

        if offset < OUTPUT_MESSAGE_DOMAIN_FIELD_COUNT {
            return Some(FieldPath::SourceDomain { source, field: offset });
        }
        offset -= OUTPUT_MESSAGE_DOMAIN_FIELD_COUNT;

        if offset == 0 {
            return Some(FieldPath::MatrixCount { source });
        }
        offset -= 1;

        let vector_width = self.k_vector_field_count();
        let vector = offset / vector_width;
        let vector_offset = offset % vector_width;
        if vector >= self.matrix_count {
            return None;
        }

        if vector_offset == 0 {
            Some(FieldPath::YRingWidth { source, matrix: vector })
        } else {
            let limb_offset = vector_offset - 1;
            Some(FieldPath::YRingLimb {
                source,
                matrix: vector,
                lane: limb_offset / K_LIMBS,
                limb: KLimb::from_index(limb_offset % K_LIMBS),
            })
        }
    }
}

const _: () = assert!(OUTPUTS_DOMAIN_FIELD_COUNT == 7);
const _: () = assert!(OUTPUT_MESSAGE_DOMAIN_FIELD_COUNT == 8);
const _: () = assert!(Profile::active_f_prime().field_count() == ACTIVE_F_PRIME_FIELD_COUNT);
const _: () = assert!(Profile::new(ACTIVE_F_PRIME_SOURCE_COUNT, 3).field_count() == LEGACY_THREE_MATRIX_FIELD_COUNT);
