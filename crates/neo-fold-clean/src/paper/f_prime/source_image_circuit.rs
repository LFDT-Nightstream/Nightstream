//! Circuit view of [`FPrimeSourceImage`].
//!
//! Allocates one bit-valued witness wire per source-image coordinate,
//! enforces bitness on each, and exposes decoded `u64`/Goldilocks words
//! as *linear combinations* over those bits.
//!
//! Use this for values that are intentionally part of a low-norm
//! **boundary** encoding — e.g., the `enc_inst(x_out)` bits that become a
//! fresh CCS instance's public input, or the u64 counters that flow into
//! the same instance. Do **not** treat this module as permission to
//! bit-route every internal F' field value. Internal field values may
//! remain ordinary computed values until the separate `enc(F')` design
//! (see `encoding.md`) says how the private F' witness is represented as
//! a low-norm CCS assignment.
//!
//! Goldilocks canonicality (`v < p = 2^64 - 2^32 + 1`) is enforced
//! separately via [`enforce_goldilocks_word_canonical`].

use neo_math::{Fq, KExtensions, F, K};
use p3_field::extension::BinomiallyExtendable;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::paper::f_prime::source_image::{
    BitRange, DigestImage, FPrimeSourceImage, KMulImage, KWordImage, Word64Image, GOLDILOCKS_BITS,
};

/// One R1CS wire per source-image coordinate, all bit-constrained.
#[derive(Clone, Debug)]
pub struct SourceImageWires {
    bits: Vec<Var>,
}

impl SourceImageWires {
    /// Allocate one bit-valued witness wire per coordinate of `image` and
    /// emit a bitness constraint (`b · (b - 1) == 0`) for each.
    pub fn alloc(builder: &mut R1csBuilder, image: &FPrimeSourceImage) -> Self {
        let mut bits = Vec::with_capacity(image.len());
        for &value in image.values() {
            let bit = builder.alloc(value);
            enforce_bit(builder, bit);
            bits.push(bit);
        }
        Self { bits }
    }

    pub fn bits(&self) -> &[Var] {
        &self.bits
    }

    pub fn range(&self, range: BitRange) -> &[Var] {
        &self.bits[range.start()..range.end()]
    }

    pub fn bit_lc(&self, idx: usize) -> Lc {
        Lc::from_var(self.bits[idx])
    }

    /// Decode a [`Word64Image`] as the linear combination
    /// `Σ_{i=0..64} 2^i · bit_i`. No fresh witness wire is allocated for
    /// the decoded value — F' gadgets consume the `Lc` directly.
    pub fn word64_lc(&self, word: Word64Image) -> Lc {
        let range = word.bits();
        assert_eq!(range.len(), GOLDILOCKS_BITS, "Word64Image must have 64 bits");
        let mut lc = Lc::zero();
        let mut coeff = F::ONE;
        for &bit in self.range(range) {
            lc.add_term(bit, coeff);
            coeff = coeff + coeff;
        }
        lc
    }

    /// Decode a [`DigestImage`] as four `Lc`s, one per Goldilocks lane.
    pub fn digest_lcs(&self, digest: DigestImage) -> [Lc; 4] {
        let lanes = digest.lanes();
        [
            self.word64_lc(lanes[0]),
            self.word64_lc(lanes[1]),
            self.word64_lc(lanes[2]),
            self.word64_lc(lanes[3]),
        ]
    }

    /// Decode a [`KWordImage`] as `[c0_lc, c1_lc]`, two Goldilocks
    /// linear combinations over the committed limb bits. No fresh
    /// witness wire is allocated for either limb.
    pub fn k_word_lcs(&self, value: KWordImage) -> [Lc; 2] {
        let [c0, c1] = value.limbs();
        [self.word64_lc(c0), self.word64_lc(c1)]
    }
}

/// Enforce that a 64-bit word represents a canonical Goldilocks element.
///
/// `p = 2^64 - 2^32 + 1`, so the invalid 64-bit encodings are exactly
/// those with `hi == 0xFFFF_FFFF` and `lo >= 1`. We allocate an indicator
/// bit `hi_is_max` constrained to be 1 iff `hi == 0xFFFF_FFFF`, then
/// require `hi_is_max · lo == 0`.
pub fn enforce_goldilocks_word_canonical(builder: &mut R1csBuilder, wires: &SourceImageWires, word: Word64Image) {
    let range = word.bits();
    assert_eq!(range.len(), GOLDILOCKS_BITS, "Goldilocks word must have 64 bits");
    let bits = wires.range(range);

    // lo_lc = Σ_{i=0..32} 2^i · bit_i; hi_lc similarly over bits[32..].
    let mut lo_lc = Lc::zero();
    let mut lo_coeff = F::ONE;
    for &bit in &bits[..32] {
        lo_lc.add_term(bit, lo_coeff);
        lo_coeff = lo_coeff + lo_coeff;
    }
    let mut hi_lc = Lc::zero();
    let mut hi_coeff = F::ONE;
    for &bit in &bits[32..] {
        hi_lc.add_term(bit, hi_coeff);
        hi_coeff = hi_coeff + hi_coeff;
    }

    let hi_max = F::from_u64(0xFFFF_FFFF);
    let hi_val = builder.eval(&hi_lc);
    let hi_is_max_val = if hi_val == hi_max { F::ONE } else { F::ZERO };
    let hi_is_max = builder.alloc(hi_is_max_val);
    enforce_bit(builder, hi_is_max);

    let diff_lc = hi_lc.clone().add_scaled(&Lc::from_const(hi_max), -F::ONE);
    let diff_val = builder.eval(&diff_lc);
    let inv_val = if diff_val == F::ZERO {
        F::ZERO
    } else {
        diff_val.inverse()
    };
    let inv = builder.alloc(inv_val);

    // hi_is_max = 1 ⇒ diff = 0.
    builder.enforce(&Lc::from_var(hi_is_max), &diff_lc, &Lc::zero());

    // diff · inv = 1 - hi_is_max  (combined with the row above, pins
    // hi_is_max ↔ (diff == 0)).
    let mut one_minus = Lc::from_const(F::ONE);
    one_minus.add_term(hi_is_max, -F::ONE);
    builder.enforce(&diff_lc, &Lc::from_var(inv), &one_minus);

    // Canonicality: hi_is_max · lo == 0.
    builder.enforce(&Lc::from_var(hi_is_max), &lo_lc, &Lc::zero());
}

/// Enforce `decode(a) · decode(b) == decode(out)` over Goldilocks `F`.
///
/// This is the first `enc(F')` arithmetic primitive: source-image-native
/// multiplication. All three operands are source-image bit ranges. **No
/// raw output field witness is allocated** — the product is committed
/// only as `out`'s bits. Goldilocks's modular wrap (`p = 2^64 - 2^32 + 1`)
/// is implicit in field arithmetic; canonicality on `out` pins it to the
/// unique low-norm representative.
///
/// This is the pattern every nonlinear `enc(F')` gadget must follow:
/// ```text
/// a_bits, b_bits, out_bits        // committed witness coordinates
/// decode(a) · decode(b) = decode(out)
/// ```
/// `out` is *not* the output of `R1csBuilder::alloc_mul` — that would
/// introduce a raw Goldilocks Var into the committed assignment.
pub fn enforce_word64_mul(
    builder: &mut R1csBuilder,
    wires: &SourceImageWires,
    a: Word64Image,
    b: Word64Image,
    out: Word64Image,
) {
    enforce_goldilocks_word_canonical(builder, wires, a);
    enforce_goldilocks_word_canonical(builder, wires, b);
    enforce_goldilocks_word_canonical(builder, wires, out);

    let a_lc = wires.word64_lc(a);
    let b_lc = wires.word64_lc(b);
    let out_lc = wires.word64_lc(out);

    builder.enforce(&a_lc, &b_lc, &out_lc);
}

/// Enforce `decode(a) · decode(b) == decode(out)` over the quadratic
/// extension `K = F[X]/(X² − W)`.
///
/// Each operand and each Karatsuba intermediate is committed as
/// source-image bits — there are **no raw Goldilocks witness Vars
/// anywhere in this gadget**. The committed witness contains:
///
/// ```text
/// a.c0, a.c1, b.c0, b.c1, out.c0, out.c1     // operands (6 × 64 bits)
/// mul.p, mul.q, mul.r                          // Karatsuba (3 × 64 bits)
/// ```
///
/// Canonicality is enforced on every word, then the K-mul law is
/// applied against the bit-decoded linear combinations only:
///
/// ```text
/// a0 · b0 = p
/// a1 · b1 = q
/// (a0 + a1) · (b0 + b1) = r
/// out.c0 = p + W · q
/// out.c1 = r − p − q
/// ```
///
/// The `W` constant is read at runtime from
/// `<Fq as BinomiallyExtendable<2>>::W` to stay byte-identical with
/// the in-circuit `enforce_k_mul` and native K arithmetic.
pub fn enforce_k_word_mul(
    builder: &mut R1csBuilder,
    wires: &SourceImageWires,
    a: KWordImage,
    b: KWordImage,
    out: KWordImage,
    mul: KMulImage,
) {
    for word in a
        .limbs()
        .into_iter()
        .chain(b.limbs())
        .chain(out.limbs())
        .chain(mul.words())
    {
        enforce_goldilocks_word_canonical(builder, wires, word);
    }

    let [a0, a1] = wires.k_word_lcs(a);
    let [b0, b1] = wires.k_word_lcs(b);
    let [out0, out1] = wires.k_word_lcs(out);

    let p = wires.word64_lc(mul.p());
    let q = wires.word64_lc(mul.q());
    let r = wires.word64_lc(mul.r());

    // Three Karatsuba product constraints, each pinned to a bit-decoded
    // Word64Image — no `alloc_mul`, no raw product witnesses.
    builder.enforce(&a0, &b0, &p);
    builder.enforce(&a1, &b1, &q);
    let a_sum = a0.clone().add_scaled(&a1, F::ONE);
    let b_sum = b0.clone().add_scaled(&b1, F::ONE);
    builder.enforce(&a_sum, &b_sum, &r);

    // Two linear closures: out.c0 = p + W·q, out.c1 = r − p − q.
    let w: F = <Fq as BinomiallyExtendable<2>>::W;
    let expected_out0 = p.clone().add_scaled(&q, w);
    let expected_out1 = r.clone().add_scaled(&p, -F::ONE).add_scaled(&q, -F::ONE);
    builder.enforce_eq(&out0, &expected_out0);
    builder.enforce_eq(&out1, &expected_out1);
}

// ── K-extension linear glue ───────────────────────────────────────────────

/// Enforce `decode(lhs) == decode(rhs)` over K, limb-by-limb. Both sides
/// are pinned to canonical Goldilocks ranges first. No multiplication
/// constraints, no fresh witnesses — the helper is pure linear glue.
pub fn enforce_k_word_eq(builder: &mut R1csBuilder, wires: &SourceImageWires, lhs: KWordImage, rhs: KWordImage) {
    for word in lhs.limbs().into_iter().chain(rhs.limbs()) {
        enforce_goldilocks_word_canonical(builder, wires, word);
    }
    let [lhs0, lhs1] = wires.k_word_lcs(lhs);
    let [rhs0, rhs1] = wires.k_word_lcs(rhs);
    builder.enforce_eq(&lhs0, &rhs0);
    builder.enforce_eq(&lhs1, &rhs1);
}

/// Enforce the general affine identity
/// `decode(out) == a_coeff · decode(a) + b_coeff · decode(b) + constant`
/// over K, where `a_coeff`, `b_coeff`, `constant ∈ K` are compile-time
/// scalars baked into the constraint coefficients.
///
/// In K = F[X]/(X² − W), multiplication by a constant `(c0 + c1·X)`
/// expands a value `(x0 + x1·X)` to:
/// ```text
/// (x0 + x1·X)(c0 + c1·X) = (x0·c0 + W·x1·c1) + (x0·c1 + x1·c0)·X
/// ```
/// All committed witness coordinates (`a`, `b`, `out` limbs) are
/// source-image bits. No fresh raw Vars are allocated beyond the
/// canonicality helpers shared by every gadget in this module.
pub fn enforce_k_word_affine2(
    builder: &mut R1csBuilder,
    wires: &SourceImageWires,
    out: KWordImage,
    a: KWordImage,
    a_coeff: K,
    b: KWordImage,
    b_coeff: K,
    constant: K,
) {
    for word in out.limbs().into_iter().chain(a.limbs()).chain(b.limbs()) {
        enforce_goldilocks_word_canonical(builder, wires, word);
    }

    let [out0, out1] = wires.k_word_lcs(out);
    let [a0, a1] = wires.k_word_lcs(a);
    let [b0, b1] = wires.k_word_lcs(b);

    let [ac0, ac1] = a_coeff.as_coeffs();
    let [bc0, bc1] = b_coeff.as_coeffs();
    let [cc0, cc1] = constant.as_coeffs();
    let w: F = <Fq as BinomiallyExtendable<2>>::W;

    // rhs0 = cc0 + ac0·a0 + W·ac1·a1 + bc0·b0 + W·bc1·b1
    let rhs0 = Lc::from_const(cc0)
        .add_scaled(&a0, ac0)
        .add_scaled(&a1, w * ac1)
        .add_scaled(&b0, bc0)
        .add_scaled(&b1, w * bc1);

    // rhs1 = cc1 + ac1·a0 + ac0·a1 + bc1·b0 + bc0·b1
    let rhs1 = Lc::from_const(cc1)
        .add_scaled(&a0, ac1)
        .add_scaled(&a1, ac0)
        .add_scaled(&b0, bc1)
        .add_scaled(&b1, bc0);

    builder.enforce_eq(&out0, &rhs0);
    builder.enforce_eq(&out1, &rhs1);
}

/// Convenience: `decode(out) == decode(a) + decode(b)` in K.
pub fn enforce_k_word_add(
    builder: &mut R1csBuilder,
    wires: &SourceImageWires,
    out: KWordImage,
    a: KWordImage,
    b: KWordImage,
) {
    enforce_k_word_affine2(builder, wires, out, a, K::ONE, b, K::ONE, K::ZERO);
}

/// Convenience: `decode(out) == decode(a) − decode(b)` in K.
pub fn enforce_k_word_sub(
    builder: &mut R1csBuilder,
    wires: &SourceImageWires,
    out: KWordImage,
    a: KWordImage,
    b: KWordImage,
) {
    enforce_k_word_affine2(builder, wires, out, a, K::ONE, b, -K::ONE, K::ZERO);
}
