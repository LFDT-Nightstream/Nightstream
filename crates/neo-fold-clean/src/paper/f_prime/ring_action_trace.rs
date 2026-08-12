//! Bit-backed native trace for one SuperNeo ring-action multiplication.
//!
//! Owns: low-norm lane encoding, trace offsets, native product materialization,
//! and output decoding for `Rq(rho) * Rq(c)`.
//!
//! Does not own: ring-action constraint emission, transcript derivation of rho,
//! or binding operands to protocol claims.
//!
//! Emits constraints: no.
//!
//! Authority boundary: decomposition and product lanes are witness data until a
//! consuming relation constrains the multiplication and binds both operands.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Lane encoding | [`LowNormEncoding`] | no | Supplied field value |
//! | Trace layout | [`RingActionTraceLayout`] | no | Fixed ring degree and encoding |
//! | Product trace | [`encode_ring_action_trace`] | no | Supplied rho and claim element |

use neo_math::ring::{Rq, D};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Low-norm encoding of a single F-valued lane as `{0, 1}` bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum LowNormEncoding {
    /// Canonical unsigned 64-bit decomposition. Decoded value =
    /// `Σ 2^i · b_i`, in `[0, 2^64)`. Round-trip works for any F.
    U64,
    /// 2's-complement signed-digit decomposition. `bits` includes the
    /// sign bit. Decoded value =
    /// `Σ_{i<bits-1} 2^i · b_i − 2^(bits−1) · b_{bits−1}`, in
    /// `[−2^(bits−1), 2^(bits−1) − 1]`. `decompose` panics if the
    /// signed representative of the input is outside that range.
    SignedDigit { bits: u8 },
}

impl LowNormEncoding {
    /// Panic if this encoding carries an invalid parameter. Called by
    /// every consuming method so direct construction of
    /// `SignedDigit { bits: 0 }` (or `bits > 64`) is caught at first
    /// use instead of corrupting downstream offsets.
    fn assert_valid(&self) {
        if let Self::SignedDigit { bits } = self {
            assert!(
                *bits >= 1 && *bits <= 64,
                "SignedDigit::bits must be in 1..=64, got {bits}"
            );
        }
    }

    pub fn limb_count(self) -> usize {
        self.assert_valid();
        match self {
            Self::U64 => 64,
            Self::SignedDigit { bits } => bits as usize,
        }
    }

    /// Decode coefficient for limb index `i`. For `U64` this is `2^i`;
    /// for `SignedDigit{n}` the top bit (index `n-1`) carries
    /// coefficient `-2^(n-1)`, others `2^i`.
    pub fn limb_coef(self, i: usize) -> F {
        self.assert_valid();
        let n = self.limb_count();
        assert!(i < n, "limb index {i} out of range for {n}-limb encoding");
        match self {
            Self::U64 => F::from_u64(1u64 << i),
            Self::SignedDigit { .. } if i + 1 == n => F::ZERO - F::from_u64(1u64 << (n - 1)),
            Self::SignedDigit { .. } => F::from_u64(1u64 << i),
        }
    }

    /// Decompose `value` into a vector of `{0, 1}` field elements.
    pub fn decompose(self, value: F) -> Vec<F> {
        self.assert_valid();
        match self {
            Self::U64 => {
                let v = value.as_canonical_u64();
                (0..64).map(|i| F::from_u64((v >> i) & 1)).collect()
            }
            Self::SignedDigit { bits } => {
                let n = bits as usize;
                let signed = signed_repr(value);
                let two_n: i128 = 1i128 << n;
                let half: i128 = 1i128 << (n - 1);
                assert!(
                    (signed as i128) >= -half && (signed as i128) < half,
                    "SignedDigit{{bits:{n}}}: value {signed} out of range [-{half},{half})",
                );
                let masked = (((signed as i128) + two_n) as u128 & (two_n as u128 - 1)) as u64;
                (0..n).map(|i| F::from_u64((masked >> i) & 1)).collect()
            }
        }
    }
}

/// Goldilocks signed representative in `(-p/2, p/2]`.
fn signed_repr(value: F) -> i64 {
    let p: u128 = (1u128 << 64) - (1u128 << 32) + 1;
    let v = value.as_canonical_u64() as u128;
    if v <= p / 2 {
        v as i64
    } else {
        -((p - v) as i64)
    }
}

/// Bit-offset layout for one ring-action gadget invocation inside the
/// F' source/witness image. Subregions:
///
/// - **ρ** — `D` limbs of `rho_enc.limb_count()` bits each.
/// - **c** — `D` limbs of `c_enc.limb_count()` bits each.
/// - **prod** — `D² = 2916` products `ρ[i]·c[j]`, each
///   `prod_enc.limb_count()` bits.
/// - **out** — `D` output lanes from `Rq::mul`, each
///   `out_enc.limb_count()` bits.
///
/// Plus the CCS constant-one slot at index 0. Total `end =
/// 1 + D·(rho+c+out) + D²·prod` bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct RingActionTraceLayout {
    pub rho_enc: LowNormEncoding,
    pub c_enc: LowNormEncoding,
    pub prod_enc: LowNormEncoding,
    pub out_enc: LowNormEncoding,
    pub constant_slot: usize,
    pub rho_offset: usize,
    pub c_offset: usize,
    pub prod_offset: usize,
    pub out_offset: usize,
    pub end: usize,
}

impl RingActionTraceLayout {
    pub fn new(
        rho_enc: LowNormEncoding,
        c_enc: LowNormEncoding,
        prod_enc: LowNormEncoding,
        out_enc: LowNormEncoding,
    ) -> Self {
        let constant_slot = 0;
        let rho_offset = 1;
        let c_offset = rho_offset + D * rho_enc.limb_count();
        let prod_offset = c_offset + D * c_enc.limb_count();
        let out_offset = prod_offset + D * D * prod_enc.limb_count();
        let end = out_offset + D * out_enc.limb_count();
        Self {
            rho_enc,
            c_enc,
            prod_enc,
            out_enc,
            constant_slot,
            rho_offset,
            c_offset,
            prod_offset,
            out_offset,
            end,
        }
    }

    pub fn rho_limb_start(&self, i: usize) -> usize {
        assert!(i < D);
        self.rho_offset + i * self.rho_enc.limb_count()
    }
    pub fn c_limb_start(&self, j: usize) -> usize {
        assert!(j < D);
        self.c_offset + j * self.c_enc.limb_count()
    }
    pub fn prod_limb_start(&self, i: usize, j: usize) -> usize {
        assert!(i < D);
        assert!(j < D);
        self.prod_offset + (i * D + j) * self.prod_enc.limb_count()
    }
    pub fn out_lane_start(&self, m: usize) -> usize {
        assert!(m < D);
        self.out_offset + m * self.out_enc.limb_count()
    }
}

/// Bit-backed witness + native output for one ring-action invocation.
#[derive(Clone, Debug)]
pub struct RingActionTraceImage {
    pub layout: RingActionTraceLayout,
    /// `values[0] = F::ONE` (CCS constant slot); all later entries are
    /// in `{0, 1}` (the low-norm `b = 2` invariant).
    pub values: Vec<F>,
    /// `Rq::mul(Rq(ρ), Rq(c)).0` — the native output for parity.
    pub output_native: [F; D],
}

/// Encode one `Rq(rho).mul(&Rq(c))` invocation into a bit-backed source
/// image under the chosen per-subregion encodings. Populates ρ, c, all
/// `D²` products, and all `D` output lanes.
pub fn encode_ring_action_trace(
    rho_vals: &[F; D],
    c_vals: &[F; D],
    layout: RingActionTraceLayout,
) -> RingActionTraceImage {
    let output_native = Rq(*rho_vals).mul(&Rq(*c_vals)).0;

    let mut values = Vec::with_capacity(layout.end);
    values.push(F::ONE);

    for &v in rho_vals.iter() {
        values.extend(layout.rho_enc.decompose(v));
    }
    for &v in c_vals.iter() {
        values.extend(layout.c_enc.decompose(v));
    }
    for i in 0..D {
        for j in 0..D {
            values.extend(layout.prod_enc.decompose(rho_vals[i] * c_vals[j]));
        }
    }
    for &v in output_native.iter() {
        values.extend(layout.out_enc.decompose(v));
    }
    assert_eq!(
        values.len(),
        layout.end,
        "encoded image length {} must match layout end {}",
        values.len(),
        layout.end,
    );

    RingActionTraceImage {
        layout,
        values,
        output_native,
    }
}

/// Decode the `D` output lanes from a ring-action trace image. Returns
/// `[F; D]` for direct comparison against `Rq::mul`'s output. Asserts
/// every read bit is `{0, 1}`.
pub fn decode_ring_action_output(image: &RingActionTraceImage) -> [F; D] {
    let mut out = [F::ZERO; D];
    for m in 0..D {
        let start = image.layout.out_lane_start(m);
        let mut acc = F::ZERO;
        for i in 0..image.layout.out_enc.limb_count() {
            let bit = image.values[start + i];
            assert!(
                bit == F::ZERO || bit == F::ONE,
                "trace bit out of range at output lane {m}, limb {i}: {bit:?}",
            );
            if bit == F::ONE {
                acc += image.layout.out_enc.limb_coef(i);
            }
        }
        out[m] = acc;
    }
    out
}
