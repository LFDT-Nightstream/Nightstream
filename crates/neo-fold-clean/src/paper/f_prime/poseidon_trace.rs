//! Bit-backed Poseidon2 trace layout + encoder + decoder for the
//! Phase 1 `enc(F')` source/witness image.
//!
//! Owns: the contract between the bit-backed
//! [`crate::engine::ccs_native::poseidon2`] builder and the F' source
//! image — how many bits each Poseidon2 hash invocation contributes,
//! where the digest output lanes live, and how to decode them back to
//! `[F; 4]`.
//!
//! Does not own: per-call-site preimage construction (e.g.
//! `state_x_out`'s exact byte layout), nor any F' region offsets beyond
//! this one trace.
//!
//! Phase 1.1-mini-1 wires this up for `state_x_out`. Phase 1.1-mini-2
//! will reuse it verbatim for the parent_authority CE digest trace.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::ccs_native::poseidon2::{
    build_bit_backed_poseidon2_hash, POSEIDON2_DIGEST_LEN, POSEIDON2_GOLDILOCKS_BITS, POSEIDON2_RATE, POSEIDON2_WIDTH,
};

pub use crate::engine::ccs_native::poseidon2::{BITS_PER_PERMUTATION, BIT_BACKED_PERMUTATION_WORDS};

/// Bit-offset layout for one Poseidon2 hash trace as it appears in the
/// F' source/witness image.
///
/// The bit-backed hash uses `ceil(preimage_len / RATE) + 1` permutations
/// (one per absorb chunk plus the padding permutation). Each
/// permutation contributes [`BITS_PER_PERMUTATION`] committed bits laid
/// out contiguously. The post-final-permutation state's
/// `POSEIDON2_WIDTH = 8` words occupy the last `WIDTH * 64` bits of the
/// trace; the digest output is the first `DIGEST_LEN = 4` of those 8
/// lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoseidonTraceLayout {
    /// Index of the CCS constant-one slot (always `0`).
    pub constant_slot: usize,
    /// First bit-index of the trace data.
    pub trace_start: usize,
    /// Trace length in bits.
    pub trace_len: usize,
    /// Number of Poseidon2 absorb permutations.
    pub absorbs: usize,
}

impl PoseidonTraceLayout {
    /// Layout for a hash over `preimage_len` field-valued inputs.
    pub fn from_preimage_len(preimage_len: usize) -> Self {
        let absorbs = preimage_len.div_ceil(POSEIDON2_RATE) + 1;
        let trace_len = absorbs * BITS_PER_PERMUTATION;
        Self {
            constant_slot: 0,
            trace_start: 1,
            trace_len,
            absorbs,
        }
    }

    /// One past the last bit-index used by this trace.
    pub fn end(&self) -> usize {
        self.trace_start + self.trace_len
    }

    /// First bit-index of the post-final-permutation state's 8 words.
    pub fn final_state_start(&self) -> usize {
        self.end() - POSEIDON2_WIDTH * POSEIDON2_GOLDILOCKS_BITS
    }

    /// First bit-index of the `lane`-th digest output lane. Lanes
    /// `[0, DIGEST_LEN)` are the digest; lanes `[DIGEST_LEN, WIDTH)`
    /// are sponge capacity, not output.
    pub fn digest_lane_start(&self, lane: usize) -> usize {
        assert!(lane < POSEIDON2_DIGEST_LEN);
        self.final_state_start() + lane * POSEIDON2_GOLDILOCKS_BITS
    }
}

/// Bit-backed witness for one Poseidon2 hash invocation, packaged with
/// its layout and the digest the builder computed natively. Phase 1
/// callers use `values` as a contiguous slice in the larger F' source
/// image and `digest_native` as the parity-test reference.
pub struct PoseidonTraceImage {
    pub layout: PoseidonTraceLayout,
    /// Bit-backed witness vector. `values[0] = F::ONE` (CCS constant
    /// slot). Every other entry is in `{0, 1}`.
    pub values: Vec<F>,
    /// The `[F; 4]` digest the bit-backed builder computed natively.
    pub digest_native: [F; POSEIDON2_DIGEST_LEN],
}

/// Encode a single Poseidon2 hash invocation as a bit-backed trace.
/// Wraps [`build_bit_backed_poseidon2_hash`] and pins the trace into a
/// [`PoseidonTraceLayout`]. Panics if the builder and layout disagree
/// on length (regression guard against a private-API drift in
/// `ccs_native::poseidon2`).
pub fn encode_poseidon_trace(preimage: &[F]) -> PoseidonTraceImage {
    let bundle = build_bit_backed_poseidon2_hash(preimage);
    let layout = PoseidonTraceLayout::from_preimage_len(preimage.len());
    assert_eq!(
        bundle.z.len(),
        layout.end(),
        "bit-backed z length {} must match layout end {}",
        bundle.z.len(),
        layout.end(),
    );
    PoseidonTraceImage {
        layout,
        values: bundle.z,
        digest_native: bundle.digest,
    }
}

/// Decode the four digest-lane bit groups back to `[F; 4]` via
/// `Σ 2^i · bit_i` per lane. Asserts every read bit is `{0, 1}`.
pub fn decode_digest_lanes(image: &PoseidonTraceImage) -> [F; POSEIDON2_DIGEST_LEN] {
    let mut out = [F::ZERO; POSEIDON2_DIGEST_LEN];
    for lane in 0..POSEIDON2_DIGEST_LEN {
        let start = image.layout.digest_lane_start(lane);
        let mut acc = F::ZERO;
        let mut pow = F::ONE;
        for bit in 0..POSEIDON2_GOLDILOCKS_BITS {
            let v = image.values[start + bit];
            assert!(
                v == F::ZERO || v == F::ONE,
                "trace bit out of range: lane={lane} bit={bit} value={v:?}"
            );
            if v == F::ONE {
                acc += pow;
            }
            pow *= F::from_u64(2);
        }
        out[lane] = acc;
    }
    out
}

/// Assert the low-norm `b = 2` invariant on a bit-backed image:
/// `values[0] == F::ONE` (CCS constant slot) and `values[i] ∈ {0, 1}`
/// for every `i ≥ 1`.
pub fn assert_committed_coords_are_bits(values: &[F]) {
    assert_eq!(
        values.first().copied(),
        Some(F::ONE),
        "CCS constant slot z[0] must be ONE"
    );
    for (i, v) in values.iter().enumerate().skip(1) {
        assert!(
            *v == F::ZERO || *v == F::ONE,
            "z[{i}] must be in {{0, 1}} for b=2 low-norm; got {v:?}"
        );
    }
}
