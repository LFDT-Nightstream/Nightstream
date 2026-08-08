//! Bit-backed Poseidon2 trace layout and native encoding for `enc(F')`.
//!
//! Owns: per-hash trace width, final-state digest offsets, native trace encoding,
//! and digest decoding.
//!
//! Does not own: call-site preimages, outer image offsets, Poseidon2 constraint
//! emission, or digest authority.
//!
//! Emits constraints: no.
//!
//! Authority boundary: an encoded trace is prover data until the Poseidon2
//! relation verifies it and the call site binds both preimage and digest.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Trace layout | [`PoseidonTraceLayout`] | no | Poseidon2 width and rate constants |
//! | Native encoding | [`encode_poseidon_trace`] | no | Supplied preimage |
//! | Digest view | [`decode_digest_lanes`] | no | Verified final-state lanes |

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::ccs_native::poseidon2::{
    build_bit_backed_poseidon2_hash_values, POSEIDON2_DIGEST_LEN, POSEIDON2_GOLDILOCKS_BITS, POSEIDON2_RATE,
    POSEIDON2_WIDTH,
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
/// its layout and the digest the builder computed natively. Callers use
/// `values` as a contiguous slice in the larger F' source
/// image and `digest_native` as the parity-test reference.
///
/// `Clone` so a chunk-shared trace (one that does not depend on a
/// step's app public input) can be computed once per SuperNeo chunk and
/// cloned into each step's encoder input — cheaper than recomputing the
/// Poseidon permutations K times. See
/// `frontends::f_prime::compiler::assemble_shared_chunk_traces`.
#[derive(Clone)]
pub struct PoseidonTraceImage {
    pub layout: PoseidonTraceLayout,
    /// Bit-backed witness vector. `values[0] = F::ONE` (CCS constant
    /// slot). Every other entry is in `{0, 1}`.
    pub values: Vec<F>,
    /// The `[F; 4]` digest the bit-backed builder computed natively.
    pub digest_native: [F; POSEIDON2_DIGEST_LEN],
}

/// Encode a single Poseidon2 hash invocation as a bit-backed trace.
/// Uses the witness-only value walk (bit-identical to
/// `build_bit_backed_poseidon2_hash` without assembling its discarded
/// constraint structure) and pins the trace into a
/// [`PoseidonTraceLayout`]. Panics if the builder and layout disagree
/// on length (regression guard against a private-API drift in
/// `ccs_native::poseidon2`).
pub fn encode_poseidon_trace(preimage: &[F]) -> PoseidonTraceImage {
    let (values, digest_native) = build_bit_backed_poseidon2_hash_values(preimage);
    let layout = PoseidonTraceLayout::from_preimage_len(preimage.len());
    assert_eq!(
        values.len(),
        layout.end(),
        "bit-backed z length {} must match layout end {}",
        values.len(),
        layout.end(),
    );
    PoseidonTraceImage {
        layout,
        values,
        digest_native,
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
