//! Phase 0D — labeled low-norm prototype for one R-26 ring-action lane.
//!
//! The prototype emits **one** ring-action gadget call (`ρ · c` in
//! `R_F = F[X]/(X^54 + X^27 + 1)`) under three witness encodings:
//!
//! 1. **full-field baseline** — current production gadget, no LN binding.
//!    Calls [`enforce_ring_mul`] directly.
//! 2. **U64 bit-backed** — every ρ, c, product, output lane is bridged
//!    to a canonical 64-bit decomposition in a source image of `{0, 1}`
//!    coords.
//! 3. **SignedDigit** — same shape, but each region uses a width matched
//!    to its actual norm bound (ρ 5 bits, c 8, prod 12, out 20).
//!
//! For each encoding the test prints: original gadget rows, source-image
//! coords (bits), bitness rows, bridge rows, total rows, total cols, and
//! the extrapolation to `κ · k_total = 18 · 16 = 288` ring-action lane
//! pairs per F' step.
//!
//! The prototype does NOT touch lifecycle code, NOT start the F' encoder
//! tree, and the three failing `ivc_invariants` tests stay red. It is
//! pure measurement, gated by the catalog §B.5 acceptance criteria.

use std::collections::HashMap;

use neo_fold_clean::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use neo_fold_clean::engine::r1cs_circuit::ring_action::enforce_ring_mul;
use neo_math::ring::D;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

// ── Scaffold: encoding enum, source binding, labeled builder ──────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LowNormEncoding {
    Unbound,
    U64,
    /// 2's-complement signed digits, top bit has coefficient `-2^(bits-1)`.
    /// Encodes values in `[-2^(bits-1), 2^(bits-1) - 1]`.
    SignedDigit {
        bits: u8,
    },
}

impl LowNormEncoding {
    fn limb_count(self) -> usize {
        match self {
            Self::Unbound => 0,
            Self::U64 => 64,
            Self::SignedDigit { bits } => bits as usize,
        }
    }

    fn limb_coef(self, i: usize) -> F {
        match self {
            Self::U64 => F::from_u64(1u64 << i),
            Self::SignedDigit { bits } if i + 1 == bits as usize => {
                // top bit has negative coefficient: -2^(bits-1)
                F::ZERO - F::from_u64(1u64 << (bits as usize - 1))
            }
            Self::SignedDigit { .. } => F::from_u64(1u64 << i),
            Self::Unbound => F::ZERO,
        }
    }

    fn decompose(self, value: F) -> Vec<F> {
        match self {
            Self::Unbound => Vec::new(),
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

/// Goldilocks signed representative in (-p/2, p/2].
fn signed_repr(value: F) -> i64 {
    // Goldilocks p = 2^64 - 2^32 + 1
    let p: u128 = (1u128 << 64) - (1u128 << 32) + 1;
    let v = value.as_canonical_u64() as u128;
    if v <= p / 2 {
        v as i64
    } else {
        -((p - v) as i64)
    }
}

#[derive(Default)]
struct SourceBinding {
    map: HashMap<String, (usize, LowNormEncoding)>,
}

impl SourceBinding {
    fn add(&mut self, label: String, offset: usize, enc: LowNormEncoding) {
        self.map.insert(label, (offset, enc));
    }
    fn get(&self, label: &str) -> Option<(usize, LowNormEncoding)> {
        self.map.get(label).copied()
    }
}

/// Wrapper that emits one bridge constraint per labeled alloc when the
/// label is bound. Mirrors `neo-fold-prototype`'s
/// `DirectSourceWitnessLinkingCs` (`terminal/committed/source_linking.rs`).
struct LabeledR1csBuilder<'a> {
    inner: &'a mut R1csBuilder,
    source_bits: &'a [Var],
    bindings: &'a SourceBinding,
    bridge_rows: usize,
    labeled_allocs: usize,
}

impl<'a> LabeledR1csBuilder<'a> {
    fn new(inner: &'a mut R1csBuilder, source_bits: &'a [Var], bindings: &'a SourceBinding) -> Self {
        Self {
            inner,
            source_bits,
            bindings,
            bridge_rows: 0,
            labeled_allocs: 0,
        }
    }

    fn maybe_bridge(&mut self, label: &str, var: Var) {
        self.labeled_allocs += 1;
        let Some((offset, enc)) = self.bindings.get(label) else {
            return;
        };
        let mut combo = Lc::zero();
        for i in 0..enc.limb_count() {
            combo.add_term(self.source_bits[offset + i], enc.limb_coef(i));
        }
        self.inner.enforce_eq(&Lc::from_var(var), &combo);
        self.bridge_rows += 1;
    }

    fn alloc(&mut self, label: &str, value: F) -> Var {
        let v = self.inner.alloc(value);
        self.maybe_bridge(label, v);
        v
    }

    fn alloc_mul(&mut self, label: &str, a: &Lc, b: &Lc) -> Var {
        let v = self.inner.alloc_mul(a, b);
        self.maybe_bridge(label, v);
        v
    }
}

// ── ϕ_81 reduction table (local copy to avoid exposing private surface) ─

const TABLE_LEN: usize = 2 * D - 1;
const PHI_MID_DEGREE: usize = neo_math::ring::PHI_MID_DEGREE;

fn build_phi_table() -> Vec<[F; D]> {
    let mut t = vec![[F::ZERO; D]; TABLE_LEN];
    for (k, row) in t.iter_mut().enumerate().take(D) {
        row[k] = F::ONE;
    }
    for k in D..TABLE_LEN {
        let mut tail = [F::ZERO; TABLE_LEN];
        tail[k] = F::ONE;
        reduce_in_place_local(&mut tail);
        t[k].copy_from_slice(&tail[..D]);
    }
    t
}

fn reduce_in_place_local(coeffs: &mut [F; TABLE_LEN]) {
    for i in (D..TABLE_LEN).rev() {
        let t = coeffs[i];
        if t == F::ZERO {
            continue;
        }
        coeffs[i] = F::ZERO;
        coeffs[i - D] -= t;
        let idx_27 = i - PHI_MID_DEGREE;
        if idx_27 < D {
            coeffs[idx_27] -= t;
        } else {
            coeffs[idx_27 - D] += t;
            if idx_27 - PHI_MID_DEGREE < D {
                coeffs[idx_27 - PHI_MID_DEGREE] += t;
            }
        }
    }
}

// ── Deterministic test values ──────────────────────────────────────────

fn signed_to_field(x: i64) -> F {
    if x >= 0 {
        F::from_u64(x as u64)
    } else {
        F::ZERO - F::from_u64((-x) as u64)
    }
}

/// ρ values in [-3, 3] — small alphabet-sampling regime.
fn make_rho_values() -> [F; D] {
    let raw: [i64; D] = std::array::from_fn(|i| ((i as i64 * 7 + 3) % 7) - 3);
    raw.map(signed_to_field)
}

/// c values in [-50, 50] — commitment-data regime under small norm.
fn make_c_values() -> [F; D] {
    let raw: [i64; D] = std::array::from_fn(|i| ((i as i64 * 13 + 1).rem_euclid(101)) - 50);
    raw.map(signed_to_field)
}

// ── Measurement ────────────────────────────────────────────────────────

#[derive(Debug)]
struct Measurement {
    name: &'static str,
    rho_enc: LowNormEncoding,
    c_enc: LowNormEncoding,
    prod_enc: LowNormEncoding,
    out_enc: LowNormEncoding,
    source_coords: usize,
    bitness_rows: usize,
    bridge_rows: usize,
    gadget_rows: usize,
    total_rows: usize,
    total_cols: usize,
}

/// Allocate one bit-Var per entry in `bit_values`, enforce bitness, and
/// return the Vars and the number of bitness rows emitted.
fn alloc_source_image(builder: &mut R1csBuilder, bit_values: &[F]) -> (Vec<Var>, usize) {
    let mut vars = Vec::with_capacity(bit_values.len());
    let mut rows = 0usize;
    for &v in bit_values {
        debug_assert!(v == F::ZERO || v == F::ONE, "source-image entries must be {{0,1}}");
        let var = builder.alloc(v);
        // bit · (bit - 1) = 0
        let mut bit_minus_one = Lc::from_var(var);
        bit_minus_one.add_constant(F::ZERO - F::ONE);
        builder.enforce(&Lc::from_var(var), &bit_minus_one, &Lc::zero());
        rows += 1;
        vars.push(var);
    }
    (vars, rows)
}

/// Full-field baseline: call the production gadget directly.
fn measure_full_field_baseline() -> Measurement {
    let mut b = R1csBuilder::new();
    let rho_vals = make_rho_values();
    let c_vals = make_c_values();
    let rho: [Var; D] = std::array::from_fn(|i| b.alloc(rho_vals[i]));
    let c: [Var; D] = std::array::from_fn(|j| b.alloc(c_vals[j]));
    let pre = b.rows();
    let _out = enforce_ring_mul(&mut b, &rho, &c);
    assert!(b.is_satisfied(), "full-field ring_mul must be satisfied");
    Measurement {
        name: "full-field baseline (no LN)",
        rho_enc: LowNormEncoding::Unbound,
        c_enc: LowNormEncoding::Unbound,
        prod_enc: LowNormEncoding::Unbound,
        out_enc: LowNormEncoding::Unbound,
        source_coords: 0,
        bitness_rows: 0,
        bridge_rows: 0,
        gadget_rows: b.rows() - pre,
        total_rows: b.rows(),
        total_cols: b.cols(),
    }
}

/// Labeled low-norm run: every committed coord is bridged to a region in
/// the source image under the chosen encoding.
fn measure_labeled_low_norm(
    name: &'static str,
    rho_enc: LowNormEncoding,
    c_enc: LowNormEncoding,
    prod_enc: LowNormEncoding,
    out_enc: LowNormEncoding,
) -> Measurement {
    let table = build_phi_table();
    let rho_vals = make_rho_values();
    let c_vals = make_c_values();

    // Native witness fill: prod[i][j] = rho[i] * c[j], out[m] from table.
    let mut prod_vals = vec![[F::ZERO; D]; D];
    for i in 0..D {
        for j in 0..D {
            prod_vals[i][j] = rho_vals[i] * c_vals[j];
        }
    }
    let mut out_vals = [F::ZERO; D];
    for m in 0..D {
        for i in 0..D {
            for j in 0..D {
                let coef = table[i + j][m];
                if coef != F::ZERO {
                    out_vals[m] += coef * prod_vals[i][j];
                }
            }
        }
    }

    // ── Source image layout ─────────────────────────────────────────────
    let mut bit_values: Vec<F> = Vec::new();
    let mut bindings = SourceBinding::default();

    let mut rho_offsets = [0usize; D];
    for i in 0..D {
        rho_offsets[i] = bit_values.len();
        bit_values.extend(rho_enc.decompose(rho_vals[i]));
        bindings.add(format!("ring_mul/rho[{i}]"), rho_offsets[i], rho_enc);
    }
    let mut c_offsets = [0usize; D];
    for j in 0..D {
        c_offsets[j] = bit_values.len();
        bit_values.extend(c_enc.decompose(c_vals[j]));
        bindings.add(format!("ring_mul/c[{j}]"), c_offsets[j], c_enc);
    }
    let mut prod_offsets = vec![[0usize; D]; D];
    for i in 0..D {
        for j in 0..D {
            prod_offsets[i][j] = bit_values.len();
            bit_values.extend(prod_enc.decompose(prod_vals[i][j]));
            bindings.add(format!("ring_mul/prod[{i}][{j}]"), prod_offsets[i][j], prod_enc);
        }
    }
    let mut out_offsets = [0usize; D];
    for m in 0..D {
        out_offsets[m] = bit_values.len();
        bit_values.extend(out_enc.decompose(out_vals[m]));
        bindings.add(format!("ring_mul/out[{m}]"), out_offsets[m], out_enc);
    }

    // ── R1CS emission ───────────────────────────────────────────────────
    let mut b = R1csBuilder::new();
    let (source_vars, bitness_rows) = alloc_source_image(&mut b, &bit_values);
    let pre_gadget = b.rows();

    let mut lab = LabeledR1csBuilder::new(&mut b, &source_vars, &bindings);

    // Labeled allocs for ρ and c — bridges emitted.
    let rho: [Var; D] = std::array::from_fn(|i| {
        let label = format!("ring_mul/rho[{i}]");
        lab.alloc(&label, rho_vals[i])
    });
    let c: [Var; D] = std::array::from_fn(|j| {
        let label = format!("ring_mul/c[{j}]");
        lab.alloc(&label, c_vals[j])
    });

    // D² alloc_muls for products — bridges emitted.
    let mut prod_vars = vec![[Var::ONE; D]; D];
    for i in 0..D {
        for j in 0..D {
            let label = format!("ring_mul/prod[{i}][{j}]");
            prod_vars[i][j] = lab.alloc_mul(&label, &Lc::from_var(rho[i]), &Lc::from_var(c[j]));
        }
    }

    // D output-lane allocs + linear-equality constraints. The output Var
    // is labeled (one bridge); the equality is the gadget body row.
    for m in 0..D {
        let label = format!("ring_mul/out[{m}]");
        let out_var = lab.alloc(&label, out_vals[m]);
        let mut combo = Lc::zero();
        for i in 0..D {
            for j in 0..D {
                let coef = table[i + j][m];
                if coef != F::ZERO {
                    combo.add_term(prod_vars[i][j], coef);
                }
            }
        }
        lab.inner.enforce_eq(&Lc::from_var(out_var), &combo);
    }

    let bridge_rows = lab.bridge_rows;
    let _labeled_allocs = lab.labeled_allocs;
    let total_rows = b.rows();
    let total_cols = b.cols();
    let gadget_rows = total_rows - pre_gadget;

    assert!(b.is_satisfied(), "{name}: labeled low-norm ring_mul must be satisfied");

    Measurement {
        name,
        rho_enc,
        c_enc,
        prod_enc,
        out_enc,
        source_coords: bit_values.len(),
        bitness_rows,
        bridge_rows,
        gadget_rows,
        total_rows,
        total_cols,
    }
}

// ── Printout ───────────────────────────────────────────────────────────

const KAPPA: usize = 18;
const K_TOTAL: usize = 16;
const LANE_PAIRS_PER_F_PRIME_STEP: usize = KAPPA * K_TOTAL;

fn fmt_enc(e: LowNormEncoding) -> String {
    match e {
        LowNormEncoding::Unbound => "—".to_string(),
        LowNormEncoding::U64 => "U64".to_string(),
        LowNormEncoding::SignedDigit { bits } => format!("SignedDigit{{{bits}b}}"),
    }
}

fn print_measurement(m: &Measurement) {
    eprintln!("─── {} ───", m.name);
    eprintln!(
        "  encodings: ρ={}  c={}  prod={}  out={}",
        fmt_enc(m.rho_enc),
        fmt_enc(m.c_enc),
        fmt_enc(m.prod_enc),
        fmt_enc(m.out_enc)
    );
    eprintln!("  source-image coords (committed bits):  {}", m.source_coords);
    eprintln!("  bitness rows:                          {}", m.bitness_rows);
    eprintln!("  bridge rows:                           {}", m.bridge_rows);
    eprintln!("  gadget rows (mults + lane eqs):        {}", m.gadget_rows);
    eprintln!("  total rows:                            {}", m.total_rows);
    eprintln!("  total cols (= committed coords + 1):   {}", m.total_cols);
    eprintln!(
        "  extrapolated × κ·k_total ({} pairs/step):",
        LANE_PAIRS_PER_F_PRIME_STEP
    );
    eprintln!(
        "    rows ≈ {}  ({:.2}M)",
        m.total_rows * LANE_PAIRS_PER_F_PRIME_STEP,
        (m.total_rows * LANE_PAIRS_PER_F_PRIME_STEP) as f64 / 1.0e6
    );
    eprintln!(
        "    cols ≈ {}  ({:.2}M)",
        m.total_cols * LANE_PAIRS_PER_F_PRIME_STEP,
        (m.total_cols * LANE_PAIRS_PER_F_PRIME_STEP) as f64 / 1.0e6
    );
}

// ── Test entry ─────────────────────────────────────────────────────────

#[test]
fn ring_action_low_norm_prototype() {
    eprintln!();
    eprintln!("Phase 0D — one R-26 ring-action lane under three encodings");
    eprintln!(
        "  D = {}, κ = {}, k_total = {}, lane pairs / F' step = {}",
        D, KAPPA, K_TOTAL, LANE_PAIRS_PER_F_PRIME_STEP
    );

    let baseline = measure_full_field_baseline();
    let u64_run = measure_labeled_low_norm(
        "LN bit-backed (U64 everywhere)",
        LowNormEncoding::U64,
        LowNormEncoding::U64,
        LowNormEncoding::U64,
        LowNormEncoding::U64,
    );
    let signed_run = measure_labeled_low_norm(
        "LN signed-digit (norm-matched widths)",
        LowNormEncoding::SignedDigit { bits: 5 },  // ρ ∈ [-3, 3]
        LowNormEncoding::SignedDigit { bits: 8 },  // c ∈ [-50, 50]
        LowNormEncoding::SignedDigit { bits: 12 }, // prod ∈ [-150, 150]
        LowNormEncoding::SignedDigit { bits: 20 }, // out ∈ [-~500k, ~500k]
    );

    print_measurement(&baseline);
    print_measurement(&u64_run);
    print_measurement(&signed_run);

    // Comparison summary.
    eprintln!();
    eprintln!("─── per-pair growth factors vs full-field baseline ───");
    for run in [&u64_run, &signed_run] {
        let r = run.total_rows as f64 / baseline.total_rows as f64;
        let c = run.total_cols as f64 / baseline.total_cols as f64;
        eprintln!("  {}:  rows × {:.2},  cols × {:.2}", run.name, r, c);
    }

    // Invariants we want to lock in.
    // 1. Baseline matches what ring_action.rs documents: D² mul rows +
    //    D linear rows = 2916 + 54 = 2970, plus 2D = 108 input allocs (0
    //    rows). Cols: 1 (constant-one) + 2D inputs + D² products + D outs.
    let expected_baseline_rows = D * D + D;
    let expected_baseline_cols = 1 + 2 * D + D * D + D;
    assert_eq!(
        baseline.total_rows, expected_baseline_rows,
        "baseline rows must match documented gadget shape"
    );
    assert_eq!(
        baseline.total_cols, expected_baseline_cols,
        "baseline cols must match documented gadget shape"
    );

    // 2. Bridge-row counts equal the number of labeled allocs.
    //    Labeled allocs per pair = 2D (ρ, c) + D² (prods) + D (outs)
    //                            = D² + 3D = 2916 + 162 = 3078.
    let expected_bridge_rows = D * D + 3 * D;
    assert_eq!(
        u64_run.bridge_rows, expected_bridge_rows,
        "U64 bridge rows must equal labeled-alloc count"
    );
    assert_eq!(
        signed_run.bridge_rows, expected_bridge_rows,
        "SignedDigit bridge rows must equal labeled-alloc count"
    );

    // 3. LN runs are strictly larger than baseline (no free lunch).
    assert!(u64_run.total_rows > baseline.total_rows);
    assert!(u64_run.total_cols > baseline.total_cols);
    assert!(signed_run.total_rows > baseline.total_rows);
    assert!(signed_run.total_cols > baseline.total_cols);

    // 4. SignedDigit is strictly cheaper than U64 (the whole point).
    assert!(
        signed_run.total_rows < u64_run.total_rows,
        "SignedDigit must beat U64 in rows"
    );
    assert!(
        signed_run.total_cols < u64_run.total_cols,
        "SignedDigit must beat U64 in cols"
    );
}
