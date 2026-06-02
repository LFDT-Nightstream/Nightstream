//! Minimal R1CS builder for in-circuit verifier gadgets.
//!
//! Owns: variable allocation, linear-combination accumulation, and
//! `(A·z) ⊙ (B·z) = (C·z)` triplet emission for Π_DEC.V / Π_RLC.V / Π_CCS.V
//! circuits used by F'.
//!
//! Does not own: matrix sparse-format conversion (caller does this once at
//! the end), public/private split (caller decides which `Var`s are public),
//! or any paper-level math.
//!
//! ## Convention
//!
//! Column 0 is always the constant `F::ONE`. `R1csBuilder::new()` allocates
//! it eagerly. Linear combinations use `(col, coeff)` term lists; the
//! `constant` field is folded into column 0 at constraint-emission time.
//!
//! Every `enforce(a, b, c)` call appends one row to A, B, C. A "linear
//! constraint" `lhs == rhs` is encoded as `(lhs - rhs) · 1 = 0`.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// A witness column index. Allocated by [`R1csBuilder::alloc`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(usize);

impl Var {
    /// The constant-1 wire. Always column 0 in any builder.
    pub const ONE: Var = Var(0);

    pub fn col(self) -> usize {
        self.0
    }
}

/// A linear combination over witness variables: `Σ coeff_i · z[col_i] + constant`.
#[derive(Clone, Debug, Default)]
pub struct Lc {
    pub terms: Vec<(usize, F)>,
    pub constant: F,
}

impl Lc {
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn from_var(v: Var) -> Self {
        Self {
            terms: vec![(v.col(), F::ONE)],
            constant: F::ZERO,
        }
    }

    pub fn from_const(c: F) -> Self {
        Self {
            terms: Vec::new(),
            constant: c,
        }
    }

    /// `self + scalar · other`.
    pub fn add_scaled(mut self, other: &Lc, scalar: F) -> Self {
        self.terms.reserve(other.terms.len());
        for &(col, coeff) in &other.terms {
            let scaled = coeff * scalar;
            if scaled != F::ZERO {
                self.terms.push((col, scaled));
            }
        }
        self.constant += other.constant * scalar;
        self
    }

    /// Append one term `coeff · v` to `self`.
    pub fn add_term(&mut self, v: Var, coeff: F) {
        if coeff != F::ZERO {
            self.terms.push((v.col(), coeff));
        }
    }

    /// Add a constant offset.
    pub fn add_constant(&mut self, c: F) {
        self.constant += c;
    }
}

/// One ring-multiplication's audit trail: the input wires, the
/// `D²`-product matrix, and the `D` output lanes. Recorded by
/// `enforce_ring_mul_with_products` when the builder's audit trail is
/// enabled. Used by Phase 1.3d-coverage to walk every ring_mul the F'
/// emitter actually invoked.
#[derive(Clone, Debug)]
pub struct RingMulAuditEntry {
    pub rho: Vec<Var>,
    pub c: Vec<Var>,
    pub output: Vec<Var>,
    pub products: Vec<Vec<Var>>,
}

/// R1CS builder: appends rows to (A, B, C) triplet form.
///
/// Construct via [`R1csBuilder::new`], allocate variables and emit constraints,
/// then call [`R1csBuilder::is_satisfied`] (or
/// [`R1csBuilder::first_unsatisfied_row`]) to check the witness.
///
/// **Audit trail** (opt-in via [`R1csBuilder::enable_audit_trail`]): when
/// enabled, every `enforce_k_mul_with_intermediates` and
/// `enforce_ring_mul_with_products` invocation pushes a record to
/// per-gadget lists, so callers can walk the full set of K-mul / ring-mul
/// wires the embedded gadgets allocated. Disabled by default — zero
/// memory overhead in normal builds.
pub struct R1csBuilder {
    a_trips: Vec<(usize, usize, F)>,
    b_trips: Vec<(usize, usize, F)>,
    c_trips: Vec<(usize, usize, F)>,
    /// `witness[col]` is the value for column `col`. Column 0 is `F::ONE`.
    witness: Vec<F>,
    rows: usize,
    audit_enabled: bool,
    audit_k_muls: Vec<[Var; 3]>,
    audit_ring_muls: Vec<RingMulAuditEntry>,
}

impl Default for R1csBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl R1csBuilder {
    pub fn new() -> Self {
        Self {
            a_trips: Vec::new(),
            b_trips: Vec::new(),
            c_trips: Vec::new(),
            witness: vec![F::ONE], // column 0 = ONE
            rows: 0,
            audit_enabled: false,
            audit_k_muls: Vec::new(),
            audit_ring_muls: Vec::new(),
        }
    }

    /// Enable the K-mul / ring-mul audit trail. Subsequent calls to
    /// `enforce_k_mul_with_intermediates` and
    /// `enforce_ring_mul_with_products` will record their internal wires
    /// in `audit_k_muls()` / `audit_ring_muls()`. Must be called before
    /// the gadgets run; pre-existing constraint rows are NOT
    /// retroactively audited.
    pub fn enable_audit_trail(&mut self) {
        self.audit_enabled = true;
    }

    /// Witness wires for every K-mul invocation recorded since the
    /// audit trail was enabled. Each `[p, q, r]` is the witness column
    /// of the corresponding Karatsuba intermediate.
    pub fn audit_k_muls(&self) -> &[[Var; 3]] {
        &self.audit_k_muls
    }

    /// Witness wires for every ring-mul invocation recorded since the
    /// audit trail was enabled.
    pub fn audit_ring_muls(&self) -> &[RingMulAuditEntry] {
        &self.audit_ring_muls
    }

    pub(crate) fn audit_trail_enabled(&self) -> bool {
        self.audit_enabled
    }

    /// Push one K-mul intermediate set. No-op when the audit trail is
    /// disabled. `pub(crate)` so only the K-mul gadget can call it.
    pub(crate) fn record_k_mul(&mut self, p: Var, q: Var, r: Var) {
        if self.audit_enabled {
            self.audit_k_muls.push([p, q, r]);
        }
    }

    /// Push one ring-mul invocation's wires. No-op when the audit trail
    /// is disabled. `pub(crate)` so only the ring-action gadget can call it.
    pub(crate) fn record_ring_mul(&mut self, entry: RingMulAuditEntry) {
        if self.audit_enabled {
            self.audit_ring_muls.push(entry);
        }
    }

    /// Allocate a private witness variable and bind it to `value`.
    pub fn alloc(&mut self, value: F) -> Var {
        let col = self.witness.len();
        self.witness.push(value);
        Var(col)
    }

    /// Allocate a vector of variables in order.
    pub fn alloc_vec(&mut self, values: &[F]) -> Vec<Var> {
        values.iter().copied().map(|v| self.alloc(v)).collect()
    }

    /// Append constraint `(a) · (b) = (c)`.
    ///
    /// Each `Lc` becomes one row's worth of triplets in A, B, C respectively.
    /// Constants are folded into column 0 (the constant-ONE wire).
    pub fn enforce(&mut self, a: &Lc, b: &Lc, c: &Lc) {
        self.push_lc_to_trips(&mut Self::pick_a, a);
        self.push_lc_to_trips(&mut Self::pick_b, b);
        self.push_lc_to_trips(&mut Self::pick_c, c);
        self.rows += 1;
    }

    fn pick_a(&mut self) -> &mut Vec<(usize, usize, F)> {
        &mut self.a_trips
    }
    fn pick_b(&mut self) -> &mut Vec<(usize, usize, F)> {
        &mut self.b_trips
    }
    fn pick_c(&mut self) -> &mut Vec<(usize, usize, F)> {
        &mut self.c_trips
    }

    fn push_lc_to_trips<P>(&mut self, picker: &mut P, lc: &Lc)
    where
        P: FnMut(&mut Self) -> &mut Vec<(usize, usize, F)>,
    {
        let row = self.rows;
        let trips = picker(self);
        for &(col, coeff) in &lc.terms {
            trips.push((row, col, coeff));
        }
        if lc.constant != F::ZERO {
            trips.push((row, Var::ONE.col(), lc.constant));
        }
    }

    /// Convenience: enforce `lhs == rhs` (i.e., `(lhs - rhs) · 1 = 0`).
    pub fn enforce_eq(&mut self, lhs: &Lc, rhs: &Lc) {
        let diff = lhs.clone().add_scaled(rhs, -F::ONE);
        let one = Lc::from_var(Var::ONE);
        let zero = Lc::zero();
        self.enforce(&diff, &one, &zero);
    }

    /// Convenience: enforce that `lc` evaluates to zero.
    pub fn enforce_zero(&mut self, lc: &Lc) {
        let one = Lc::from_var(Var::ONE);
        let zero = Lc::zero();
        self.enforce(lc, &one, &zero);
    }

    /// Allocate `out = a · b` and constrain it. Returns the new var.
    pub fn alloc_mul(&mut self, a: &Lc, b: &Lc) -> Var {
        let av = eval_lc(a, &self.witness);
        let bv = eval_lc(b, &self.witness);
        let out = self.alloc(av * bv);
        let out_lc = Lc::from_var(out);
        self.enforce(a, b, &out_lc);
        out
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.witness.len()
    }

    pub fn witness(&self) -> &[F] {
        &self.witness
    }

    /// Allocated witness columns that do not appear in any A/B/C row.
    ///
    /// This is an audit helper, not a proof of semantic binding: a column
    /// can appear in rows and still be under-constrained. It catches the
    /// narrower but dangerous class where a gadget allocates an authoritative
    /// value and never references it at all.
    pub fn unconstrained_columns(&self) -> Vec<usize> {
        let mut used = vec![false; self.witness.len()];
        used[Var::ONE.col()] = true;
        for &(_, col, _) in self
            .a_trips
            .iter()
            .chain(self.b_trips.iter())
            .chain(self.c_trips.iter())
        {
            if col < used.len() {
                used[col] = true;
            }
        }
        used.into_iter()
            .enumerate()
            .filter_map(|(col, is_used)| (!is_used).then_some(col))
            .collect()
    }

    /// Evaluate a linear combination against the current witness.
    pub fn eval(&self, lc: &Lc) -> F {
        eval_lc(lc, &self.witness)
    }

    /// Mutate a witness column. Used by tests to inject tamper.
    /// **Auditor**: not used in any prove/verify path. Tests only.
    pub fn tamper_witness(&mut self, col: usize, value: F) {
        self.witness[col] = value;
    }

    /// Index of the first row that fails `(A·z)[r] · (B·z)[r] = (C·z)[r]`,
    /// or `None` if all rows hold.
    pub fn first_unsatisfied_row(&self) -> Option<usize> {
        let z = &self.witness;
        let az = sparse_matvec(&self.a_trips, self.rows, z);
        let bz = sparse_matvec(&self.b_trips, self.rows, z);
        let cz = sparse_matvec(&self.c_trips, self.rows, z);
        for r in 0..self.rows {
            if az[r] * bz[r] != cz[r] {
                return Some(r);
            }
        }
        None
    }

    pub fn is_satisfied(&self) -> bool {
        self.first_unsatisfied_row().is_none()
    }
}

fn eval_lc(lc: &Lc, witness: &[F]) -> F {
    let mut acc = lc.constant;
    for &(col, coeff) in &lc.terms {
        acc += coeff * witness[col];
    }
    acc
}

fn sparse_matvec(trips: &[(usize, usize, F)], rows: usize, z: &[F]) -> Vec<F> {
    let mut out = vec![F::ZERO; rows];
    for &(r, c, v) in trips {
        out[r] += v * z[c];
    }
    out
}
