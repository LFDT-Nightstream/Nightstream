//! Low-norm source image for F' boundary data.
//!
//! SuperNeo's fresh CCS relation commits a full assignment `z = [x, w]`
//! and requires `‖z‖_∞ < b`. For `b = 2`, any value placed *directly* into
//! the fresh CCS public input `x` must be bit-valued. That's the only
//! reason F' needs bit encoding for the `x_out` digest — not because
//! Poseidon2 itself "needs low norm".
//!
//! The raw F' hash output `h = state_x_out_digest(...)` is allowed to be
//! a normal Goldilocks field value while it is only an intermediate
//! computation. It becomes low-norm-relevant only at the `enc_inst(h)`
//! boundary, because `enc_inst(h)` is the public input of the *next*
//! fresh CCS instance.
//!
//! This module therefore owns source-image bits for those boundary
//! values. It is **not** a generic lowering pass, and it should not be
//! used to route every internal digest or transcript lane through a bit
//! image by default.
//!
//! The unresolved larger design is `enc(F')`: how to encode the private
//! F' execution witness `w` as a low-norm assignment suitable for
//! `CcsInstance::from_low_norm_assignment`. That is separate from
//! `enc_inst(h)`, which handles only the public-instance boundary. See
//! `encoding.md` for the distinction.
//!
//! ## Current scope
//!
//! At this stage, this image backs:
//!
//! - `enc_inst(prior_x_out)` (input recursive link),
//! - `enc_inst(current_x_out)` (output recursive link),
//! - selected u64 boundary counters (`chunk_count_in`, `step_count_in`, `pc`).
//!
//! It does **not** yet encode the full private F' execution witness.
//!
//! ## Layout
//!
//! Bits are appended sequentially via `push_*` methods. Each push returns
//! a typed handle (`BitRange`, [`Word64Image`], [`DigestImage`]) that
//! `SourceImageWires` later resolves to specific allocated R1CS vars.
//! Handles are offset-by-construction; they cannot collide with each
//! other within a single source image.

use neo_math::{KExtensions, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::paper::f_prime::r1cs::encode_x_out_public_bits;

/// Canonical bit width of a Goldilocks element.
pub const GOLDILOCKS_BITS: usize = 64;

/// Number of digest lanes (`DIGEST_LEN` from the Poseidon2 gadget).
pub const DIGEST_LANES: usize = 4;

/// Bits in an `enc_inst(x_out)` body — the 256-bit canonical encoding of
/// the four Goldilocks digest lanes.
pub const ENC_INST_BITS: usize = DIGEST_LANES * GOLDILOCKS_BITS;

/// Total bits in the F' CCS public input: `[1, enc_inst(x_out)…]`.
pub const F_PRIME_PUBLIC_INPUT_BITS: usize = 1 + ENC_INST_BITS;

/// A contiguous range `[start, start + len)` of bit indices in an
/// [`FPrimeSourceImage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitRange {
    start: usize,
    len: usize,
}

impl BitRange {
    pub fn new(start: usize, len: usize) -> Self {
        Self { start, len }
    }

    pub fn start(self) -> usize {
        self.start
    }

    pub fn len(self) -> usize {
        self.len
    }

    pub fn end(self) -> usize {
        self.start + self.len
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// Typed handle for a 64-bit Goldilocks word that was appended to the
/// source image. The bits are stored little-endian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Word64Image {
    bits: BitRange,
}

impl Word64Image {
    pub fn bits(self) -> BitRange {
        self.bits
    }
}

/// Typed handle for a four-lane digest (`[F; 4]`) appended to the source
/// image as `4 × 64` canonical bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DigestImage {
    lanes: [Word64Image; DIGEST_LANES],
}

impl DigestImage {
    pub fn lanes(self) -> [Word64Image; DIGEST_LANES] {
        self.lanes
    }
}

/// Typed handle for one quadratic-extension element `K = c0 + c1·X`
/// (X² = W) committed as two consecutive [`Word64Image`]s in the source
/// image. The W constant is owned by
/// [`crate::engine::r1cs_circuit::field_ext`] — this image only stores
/// canonical bits of each Goldilocks limb.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KWordImage {
    c0: Word64Image,
    c1: Word64Image,
}

impl KWordImage {
    /// Compose a `KWordImage` from two previously pushed Goldilocks
    /// words. Used by tests that need to inject a noncanonical limb
    /// before letting the circuit reject it; production code should
    /// always go through [`FPrimeSourceImage::push_k`].
    pub fn from_limbs(c0: Word64Image, c1: Word64Image) -> Self {
        Self { c0, c1 }
    }

    pub fn c0(self) -> Word64Image {
        self.c0
    }

    pub fn c1(self) -> Word64Image {
        self.c1
    }

    pub fn limbs(self) -> [Word64Image; 2] {
        [self.c0, self.c1]
    }
}

/// Typed handle for the three Karatsuba intermediates of one K-mul:
/// `p = a0·b0`, `q = a1·b1`, `r = (a0+a1)·(b0+b1)`, each committed as
/// a canonical Goldilocks [`Word64Image`]. They live in the source
/// image so the K-mul gadget never allocates raw product witnesses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KMulImage {
    p: Word64Image,
    q: Word64Image,
    r: Word64Image,
}

impl KMulImage {
    pub fn p(self) -> Word64Image {
        self.p
    }

    pub fn q(self) -> Word64Image {
        self.q
    }

    pub fn r(self) -> Word64Image {
        self.r
    }

    pub fn words(self) -> [Word64Image; 3] {
        [self.p, self.q, self.r]
    }
}

/// Append-only buffer of `{0, 1}` field values. Every coordinate that
/// will eventually be committed in the F' CCS instance lives here.
#[derive(Clone, Debug, Default)]
pub struct FPrimeSourceImage {
    values: Vec<F>,
}

impl FPrimeSourceImage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn range(&self, range: BitRange) -> &[F] {
        &self.values[range.start()..range.end()]
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Append one bit. Returns its index in the source image.
    pub fn push_bit(&mut self, bit: bool) -> usize {
        let idx = self.values.len();
        self.values.push(if bit { F::ONE } else { F::ZERO });
        idx
    }

    /// Append raw little-endian 64-bit data. If the word represents a
    /// Goldilocks element, the circuit side must also enforce canonicality
    /// (`< p = 2^64 - 2^32 + 1`) via
    /// [`crate::paper::f_prime::source_image_circuit::enforce_goldilocks_word_canonical`].
    pub fn push_u64_le(&mut self, value: u64) -> Word64Image {
        let start = self.values.len();
        for i in 0..GOLDILOCKS_BITS {
            self.push_bit(((value >> i) & 1) != 0);
        }
        Word64Image {
            bits: BitRange::new(start, GOLDILOCKS_BITS),
        }
    }

    /// Convenience: append `value.as_canonical_u64()` as a 64-bit word.
    /// The circuit side still has to call
    /// [`crate::paper::f_prime::source_image_circuit::enforce_goldilocks_word_canonical`]
    /// to bind the bit pattern to a canonical Goldilocks representative.
    pub fn push_goldilocks(&mut self, value: F) -> Word64Image {
        self.push_u64_le(value.as_canonical_u64())
    }

    /// Append one quadratic-extension element `value ∈ K = F[X]/(X² − W)`
    /// as two consecutive canonical Goldilocks words `(c0, c1)`. The
    /// circuit side enforces canonicality on each limb plus the K-mul
    /// law via [`crate::paper::f_prime::source_image_circuit::enforce_k_word_mul`].
    pub fn push_k(&mut self, value: K) -> KWordImage {
        let [c0, c1] = value.as_coeffs();
        let c0_word = self.push_goldilocks(c0);
        let c1_word = self.push_goldilocks(c1);
        KWordImage {
            c0: c0_word,
            c1: c1_word,
        }
    }

    /// Append the three Karatsuba intermediates of `a · b` in K. These
    /// are the prover's commitments to `p = a0·b0`, `q = a1·b1`, and
    /// `r = (a0+a1)·(b0+b1)`, each stored as a canonical Goldilocks
    /// word. The circuit then enforces the products against the
    /// bit-decoded LCs — there is no raw witness Var anywhere in the
    /// K-mul gadget.
    pub fn push_k_mul_witness(&mut self, a: K, b: K) -> KMulImage {
        let [a0, a1] = a.as_coeffs();
        let [b0, b1] = b.as_coeffs();
        let p = a0 * b0;
        let q = a1 * b1;
        let r = (a0 + a1) * (b0 + b1);
        KMulImage {
            p: self.push_goldilocks(p),
            q: self.push_goldilocks(q),
            r: self.push_goldilocks(r),
        }
    }

    /// Append a four-lane digest as `4 × 64` canonical bits.
    pub fn push_digest_lanes(&mut self, lanes: [F; DIGEST_LANES]) -> DigestImage {
        DigestImage {
            lanes: [
                self.push_u64_le(lanes[0].as_canonical_u64()),
                self.push_u64_le(lanes[1].as_canonical_u64()),
                self.push_u64_le(lanes[2].as_canonical_u64()),
                self.push_u64_le(lanes[3].as_canonical_u64()),
            ],
        }
    }

    /// Append `enc_inst(x_out)` — the `ENC_INST_BITS`-bit body that
    /// `paper::f_prime::r1cs` uses to carry an `x_out` digest as a
    /// low-norm CCS public input.
    pub fn push_enc_inst(&mut self, x_out: [F; DIGEST_LANES]) -> BitRange {
        let start = self.values.len();
        self.values.extend(encode_x_out_public_bits(x_out));
        BitRange::new(start, ENC_INST_BITS)
    }

    /// Append the full F' CCS public input `[1, enc_inst(x_out)…]`.
    pub fn push_f_prime_public_input(&mut self, x_out: [F; DIGEST_LANES]) -> BitRange {
        let start = self.values.len();
        self.push_bit(true);
        self.push_enc_inst(x_out);
        BitRange::new(start, F_PRIME_PUBLIC_INPUT_BITS)
    }

    /// Overwrite a single coordinate to `0` or `1`. Used by tests that
    /// craft a malformed source image (e.g. a tampered enc_inst bit) to
    /// exercise rejection paths in the F' verifier.
    pub fn set_bit(&mut self, idx: usize, bit: bool) {
        self.values[idx] = if bit { F::ONE } else { F::ZERO };
    }

    /// Overwrite a coordinate with an arbitrary `F` value. Distinct from
    /// [`Self::set_bit`] so callers explicitly opt into producing a
    /// non-binary source image (e.g. to verify the circuit's bitness
    /// constraint rejects the tamper).
    pub fn set_raw(&mut self, idx: usize, value: F) {
        self.values[idx] = value;
    }

    /// True iff every appended value is `F::ZERO` or `F::ONE`.
    pub fn is_binary(&self) -> bool {
        self.values.iter().all(|v| *v == F::ZERO || *v == F::ONE)
    }
}
