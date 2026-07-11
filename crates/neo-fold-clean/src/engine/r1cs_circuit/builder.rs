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

use std::collections::HashMap;

use neo_ccs::SeededPhi81LinearBlock;
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

/// One canonical Goldilocks decomposition emitted by the u64 gadget.
///
/// The field-native relation still contains every decomposition constraint.
/// Low-norm lowering may additionally use this trusted synthesis metadata to
/// place each one-bit child directly in the corresponding bit of `field_col`
/// instead of committing the same bit twice.
#[derive(Clone, Debug)]
pub(crate) struct CanonicalU64Decomposition {
    pub(crate) field_col: usize,
    pub(crate) bit_cols: [usize; 64],
}

pub(crate) const BALANCED_TERNARY_DIGITS: usize = 41;

#[derive(Clone, Debug)]
pub(crate) struct BalancedTernaryDecomposition {
    pub(crate) field_col: usize,
    pub(crate) digit_cols: [usize; BALANCED_TERNARY_DIGITS],
}

/// One Poseidon2 S-box in a selectively lowerable permutation trace.
#[derive(Clone, Debug)]
pub(crate) struct Poseidon2SboxTrace {
    pub(crate) input: Lc,
    pub(crate) output_col: usize,
}

/// Exact high-degree view of one Poseidon2 permutation.
///
/// The ordinary R1CS rows remain authoritative and testable. Road A's
/// low-norm compiler may replace precisely these rows with `x^7` gates and
/// final linear-output rows, eliminating only the listed temporary columns.
#[derive(Clone, Debug)]
pub(crate) struct Poseidon2PermutationTrace {
    pub(crate) row_start: usize,
    pub(crate) row_end: usize,
    pub(crate) input_cols: [usize; 8],
    pub(crate) allocated_columns: Vec<usize>,
    pub(crate) sboxes: Vec<Poseidon2SboxTrace>,
    pub(crate) output_cols: [usize; 8],
    pub(crate) output_linear_forms: [Lc; 8],
}

/// Compact assurance view of one exact production Poseidon2 invocation.
///
/// The isolated artifact numbers its eight inputs as columns 1..8 and its
/// fresh columns from 9 onward.  A call site is therefore identified by its
/// eight input columns and first fresh column; the remaining renaming is
/// affine.  Row hashes remain the drift authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poseidon2PermutationAudit {
    pub row_start: usize,
    pub row_end: usize,
    pub input_cols: [usize; 8],
    pub first_allocated_col: usize,
    pub allocated_col_count: usize,
    pub output_cols: [usize; 8],
}

#[derive(Clone, Debug)]
pub(crate) enum Poseidon2HashRoundKind {
    Absorb { chunk_cols: Vec<usize> },
    Pad,
}

#[derive(Clone, Debug)]
pub(crate) struct Poseidon2HashRoundTrace {
    pub(crate) kind: Poseidon2HashRoundKind,
    pub(crate) state_before_cols: [usize; 8],
    pub(crate) permutation_input_cols: [usize; 8],
    pub(crate) defining_rows: Vec<usize>,
    pub(crate) permutation_output_cols: [usize; 8],
}

#[derive(Clone, Debug)]
pub(crate) struct Poseidon2HashTrace {
    pub(crate) row_start: usize,
    pub(crate) row_end: usize,
    pub(crate) input_cols: Vec<usize>,
    pub(crate) zero_col: usize,
    pub(crate) zero_row: usize,
    pub(crate) rounds: Vec<Poseidon2HashRoundTrace>,
    pub(crate) output_cols: [usize; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Poseidon2HashRoundAuditKind {
    Absorb { chunk_cols: Vec<usize> },
    Pad,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poseidon2HashRoundAudit {
    pub kind: Poseidon2HashRoundAuditKind,
    pub state_before_cols: [usize; 8],
    pub permutation_input_cols: [usize; 8],
    pub defining_rows: Vec<usize>,
    pub permutation_output_cols: [usize; 8],
}

/// Compact semantic trace for one production Poseidon2 sponge invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poseidon2HashAudit {
    pub row_start: usize,
    pub row_end: usize,
    pub input_cols: Vec<usize>,
    pub zero_col: usize,
    pub zero_row: usize,
    pub rounds: Vec<Poseidon2HashRoundAudit>,
    pub output_cols: [usize; 4],
}

/// Column-renaming data for one canonical-u64 decomposition call.
///
/// This is read-only artifact metadata.  The exact emitted rows remain the
/// authority, and Lean rechecks the renamed isolated gadget rows against the
/// enclosing program before reusing its soundness theorem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalU64Audit {
    pub field_col: usize,
    pub bit_cols: [usize; 64],
}

/// Direct CCS view of one `p(beta)` evaluation in the projection checker.
#[derive(Clone, Debug)]
pub(crate) struct PolynomialEvaluationTrace {
    pub(crate) row_start: usize,
    pub(crate) row_end: usize,
    pub(crate) allocated_columns: Vec<usize>,
    pub(crate) coefficient_cols: Vec<usize>,
    pub(crate) power_cols: Vec<[usize; 2]>,
    pub(crate) output_cols: [usize; 2],
}

/// One identity inside a selectively lowered batch: `result = sum(a_i*b_i)`.
#[derive(Clone, Debug)]
pub(crate) struct ProductSumIdentityTrace {
    pub(crate) factors: Vec<ProductFactorTrace>,
    pub(crate) result: Lc,
}

/// One scaled product in a selectively lowered product sum.
#[derive(Clone, Debug)]
pub(crate) struct ProductFactorTrace {
    pub(crate) left: Lc,
    pub(crate) right: Lc,
    pub(crate) coefficient: F,
}

#[derive(Clone, Debug)]
pub(crate) struct CenteredUnitTrace {
    pub(crate) row_start: usize,
    pub(crate) row_end: usize,
    pub(crate) allocated_columns: Vec<usize>,
    pub(crate) value_col: usize,
}

/// A group of product-sum identities whose ordinary R1CS rows are contiguous.
#[derive(Clone, Debug)]
pub(crate) struct ProductSumBatchTrace {
    pub(crate) row_start: usize,
    pub(crate) row_end: usize,
    pub(crate) allocated_columns: Vec<usize>,
    pub(crate) retained_columns: Vec<usize>,
    pub(crate) identities: Vec<ProductSumIdentityTrace>,
}

/// Non-authoritative row ownership marker for assurance tooling.
///
/// A marker only names a contiguous interval already emitted by the builder;
/// it never changes or replaces the underlying R1CS rows. Formal manifests use
/// these ranges to make large generators reviewable and fail closed on drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowFamilyRange {
    pub name: &'static str,
    pub row_start: usize,
    pub row_end: usize,
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
    record_structure: bool,
    a_trips: Vec<(usize, usize, F)>,
    b_trips: Vec<(usize, usize, F)>,
    c_trips: Vec<(usize, usize, F)>,
    /// `witness[col]` is the value for column `col`. Column 0 is `F::ONE`.
    witness: Vec<F>,
    rows: usize,
    audit_enabled: bool,
    audit_k_muls: Vec<[Var; 3]>,
    audit_ring_muls: Vec<RingMulAuditEntry>,
    canonical_u64_decompositions: Vec<CanonicalU64Decomposition>,
    canonical_u64_decomposition_by_field: HashMap<usize, [Var; 64]>,
    balanced_ternary_decompositions: Vec<BalancedTernaryDecomposition>,
    balanced_ternary_decomposition_by_field: HashMap<usize, [Var; BALANCED_TERNARY_DIGITS]>,
    boolean_columns: Vec<usize>,
    centered_unit_columns: Vec<usize>,
    seeded_phi81_a_blocks: Vec<SeededPhi81LinearBlock>,
    poseidon2_traces: Vec<Poseidon2PermutationTrace>,
    poseidon2_hash_traces: Vec<Poseidon2HashTrace>,
    polynomial_evaluation_traces: Vec<PolynomialEvaluationTrace>,
    product_sum_batch_traces: Vec<ProductSumBatchTrace>,
    centered_unit_traces: Vec<CenteredUnitTrace>,
    equality_pairs: Vec<(usize, usize, usize)>,
    row_family_ranges: Vec<RowFamilyRange>,
}

/// Immutable output of one completed R1CS synthesis.
///
/// This stays crate-private: frontends may translate the exact rows and
/// witness into their committed representation, but protocol callers should
/// never construct or mutate matrix triplets directly.
pub(crate) struct R1csSynthesis {
    pub(crate) a_trips: Vec<(usize, usize, F)>,
    pub(crate) b_trips: Vec<(usize, usize, F)>,
    pub(crate) c_trips: Vec<(usize, usize, F)>,
    pub(crate) witness: Vec<F>,
    pub(crate) rows: usize,
    pub(crate) canonical_u64_decompositions: Vec<CanonicalU64Decomposition>,
    pub(crate) balanced_ternary_decompositions: Vec<BalancedTernaryDecomposition>,
    pub(crate) boolean_columns: Vec<usize>,
    pub(crate) centered_unit_columns: Vec<usize>,
    pub(crate) seeded_phi81_a_blocks: Vec<SeededPhi81LinearBlock>,
    pub(crate) poseidon2_traces: Vec<Poseidon2PermutationTrace>,
    pub(crate) polynomial_evaluation_traces: Vec<PolynomialEvaluationTrace>,
    pub(crate) product_sum_batch_traces: Vec<ProductSumBatchTrace>,
    pub(crate) centered_unit_traces: Vec<CenteredUnitTrace>,
    pub(crate) equality_pairs: Vec<(usize, usize, usize)>,
    pub(crate) row_family_ranges: Vec<RowFamilyRange>,
}

impl Default for R1csBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl R1csBuilder {
    pub fn new() -> Self {
        Self::with_structure_recording(true)
    }

    /// Evaluate a preprocessed fixed-shape circuit and retain only its witness.
    pub(crate) fn new_witness_only() -> Self {
        Self::with_structure_recording(false)
    }

    fn with_structure_recording(record_structure: bool) -> Self {
        Self {
            record_structure,
            a_trips: Vec::new(),
            b_trips: Vec::new(),
            c_trips: Vec::new(),
            witness: vec![F::ONE], // column 0 = ONE
            rows: 0,
            audit_enabled: false,
            audit_k_muls: Vec::new(),
            audit_ring_muls: Vec::new(),
            canonical_u64_decompositions: Vec::new(),
            canonical_u64_decomposition_by_field: HashMap::new(),
            balanced_ternary_decompositions: Vec::new(),
            balanced_ternary_decomposition_by_field: HashMap::new(),
            boolean_columns: Vec::new(),
            centered_unit_columns: Vec::new(),
            seeded_phi81_a_blocks: Vec::new(),
            poseidon2_traces: Vec::new(),
            poseidon2_hash_traces: Vec::new(),
            polynomial_evaluation_traces: Vec::new(),
            product_sum_batch_traces: Vec::new(),
            centered_unit_traces: Vec::new(),
            equality_pairs: Vec::new(),
            row_family_ranges: Vec::new(),
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

    pub(crate) fn record_canonical_u64_decomposition(&mut self, field: Var, bits: [Var; 64]) {
        self.canonical_u64_decomposition_by_field
            .insert(field.col(), bits);
        if self.record_structure {
            self.canonical_u64_decompositions
                .push(CanonicalU64Decomposition {
                    field_col: field.col(),
                    bit_cols: bits.map(Var::col),
                });
        }
    }

    pub(crate) fn canonical_u64_decomposition(&self, field: Var) -> Option<[Var; 64]> {
        self.canonical_u64_decomposition_by_field
            .get(&field.col())
            .copied()
    }

    pub(crate) fn record_balanced_ternary_decomposition(&mut self, field: Var, digits: [Var; BALANCED_TERNARY_DIGITS]) {
        self.balanced_ternary_decomposition_by_field
            .insert(field.col(), digits);
        if self.record_structure {
            self.balanced_ternary_decompositions
                .push(BalancedTernaryDecomposition {
                    field_col: field.col(),
                    digit_cols: digits.map(Var::col),
                });
        }
    }

    pub(crate) fn balanced_ternary_decomposition(&self, field: Var) -> Option<[Var; BALANCED_TERNARY_DIGITS]> {
        self.balanced_ternary_decomposition_by_field
            .get(&field.col())
            .copied()
    }

    pub(crate) fn record_boolean(&mut self, value: Var) {
        if self.record_structure {
            self.boolean_columns.push(value.col());
        }
    }

    pub(crate) fn record_centered_unit(&mut self, value: Var) {
        if self.record_structure {
            self.centered_unit_columns.push(value.col());
        }
    }

    pub(crate) fn record_centered_unit_trace(&mut self, trace: CenteredUnitTrace) {
        if self.record_structure {
            self.centered_unit_columns.push(trace.value_col);
            self.centered_unit_traces.push(trace);
        }
    }

    /// Record ownership of rows emitted since `row_start`.
    ///
    /// Ranges may be nested (for example, the F' NIFS range contains PiCCS,
    /// PiRLC, and PiDEC subranges). Consumers must still validate exact row
    /// hashes; this metadata is not semantic authority.
    pub fn record_row_family(&mut self, name: &'static str, row_start: usize) {
        assert!(row_start <= self.rows, "row-family start exceeds builder cursor");
        if self.record_structure {
            self.row_family_ranges.push(RowFamilyRange {
                name,
                row_start,
                row_end: self.rows,
            });
        }
    }

    /// Read-only assurance view of recorded row ownership.
    pub fn row_family_ranges(&self) -> &[RowFamilyRange] {
        &self.row_family_ranges
    }

    /// Exact field/bit column maps for emitted canonical-u64 gadgets.
    #[doc(hidden)]
    pub fn canonical_u64_audits(&self) -> Vec<CanonicalU64Audit> {
        self.canonical_u64_decompositions
            .iter()
            .map(|decomposition| CanonicalU64Audit {
                field_col: decomposition.field_col,
                bit_cols: decomposition.bit_cols,
            })
            .collect()
    }

    /// Exact column-renaming certificates for all emitted Poseidon2 calls.
    #[doc(hidden)]
    pub fn poseidon2_permutation_audits(&self) -> Vec<Poseidon2PermutationAudit> {
        self.poseidon2_traces
            .iter()
            .map(|trace| {
                let first_allocated_col = trace
                    .allocated_columns
                    .first()
                    .copied()
                    .expect("Poseidon2 permutation allocates fresh columns");
                assert!(
                    trace
                        .allocated_columns
                        .iter()
                        .copied()
                        .eq(first_allocated_col..first_allocated_col + trace.allocated_columns.len()),
                    "Poseidon2 fresh columns must remain contiguous",
                );
                Poseidon2PermutationAudit {
                    row_start: trace.row_start,
                    row_end: trace.row_end,
                    input_cols: trace.input_cols,
                    first_allocated_col,
                    allocated_col_count: trace.allocated_columns.len(),
                    output_cols: trace.output_cols,
                }
            })
            .collect()
    }

    #[doc(hidden)]
    pub fn poseidon2_hash_audits(&self) -> Vec<Poseidon2HashAudit> {
        self.poseidon2_hash_traces
            .iter()
            .map(|trace| Poseidon2HashAudit {
                row_start: trace.row_start,
                row_end: trace.row_end,
                input_cols: trace.input_cols.clone(),
                zero_col: trace.zero_col,
                zero_row: trace.zero_row,
                rounds: trace
                    .rounds
                    .iter()
                    .map(|round| Poseidon2HashRoundAudit {
                        kind: match &round.kind {
                            Poseidon2HashRoundKind::Absorb { chunk_cols } => Poseidon2HashRoundAuditKind::Absorb {
                                chunk_cols: chunk_cols.clone(),
                            },
                            Poseidon2HashRoundKind::Pad => Poseidon2HashRoundAuditKind::Pad,
                        },
                        state_before_cols: round.state_before_cols,
                        permutation_input_cols: round.permutation_input_cols,
                        defining_rows: round.defining_rows.clone(),
                        permutation_output_cols: round.permutation_output_cols,
                    })
                    .collect(),
                output_cols: trace.output_cols,
            })
            .collect()
    }

    pub(crate) fn enforce_seeded_phi81_a_block(&mut self, block: SeededPhi81LinearBlock, outputs: &[Var]) {
        assert_eq!(
            block.row_start(),
            self.rows,
            "seeded Phi81 rows must append at the builder cursor"
        );
        assert_eq!(outputs.len(), neo_math::D * block.kappa(), "seeded Phi81 output width");
        if self.record_structure {
            block
                .validate_matrix_shape(self.rows + outputs.len(), self.witness.len())
                .expect("seeded Phi81 block must fit the synthesized matrix");
            self.seeded_phi81_a_blocks.push(block);
            for &output in outputs {
                self.b_trips.push((self.rows, Var::ONE.col(), F::ONE));
                self.c_trips.push((self.rows, output.col(), F::ONE));
                self.rows += 1;
            }
        } else {
            self.rows += outputs.len();
        }
    }

    pub(crate) fn record_poseidon2_permutation(&mut self, trace: Poseidon2PermutationTrace) {
        if self.record_structure {
            debug_assert_eq!(trace.row_end, self.rows);
            self.poseidon2_traces.push(trace);
        }
    }

    pub(crate) fn record_poseidon2_hash(&mut self, trace: Poseidon2HashTrace) {
        if self.record_structure {
            debug_assert_eq!(trace.row_end, self.rows);
            self.poseidon2_hash_traces.push(trace);
        }
    }

    pub(crate) fn record_polynomial_evaluation(&mut self, trace: PolynomialEvaluationTrace) {
        if self.record_structure {
            debug_assert_eq!(trace.row_end, self.rows);
            self.polynomial_evaluation_traces.push(trace);
        }
    }

    pub(crate) fn record_product_sum_batch(&mut self, trace: ProductSumBatchTrace) {
        if self.record_structure {
            debug_assert_eq!(trace.row_end, self.rows);
            self.product_sum_batch_traces.push(trace);
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
        if self.record_structure {
            self.push_lc_to_trips(&mut Self::pick_a, a);
            self.push_lc_to_trips(&mut Self::pick_b, b);
            self.push_lc_to_trips(&mut Self::pick_c, c);
        }
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
        if !self.record_structure {
            self.rows += 1;
            return;
        }
        let single_var = |value: &Lc| {
            if value.constant == F::ZERO && value.terms.len() == 1 && value.terms[0].1 == F::ONE {
                Some(value.terms[0].0)
            } else {
                None
            }
        };
        if let (Some(lhs), Some(rhs)) = (single_var(lhs), single_var(rhs)) {
            if lhs != rhs {
                self.equality_pairs.push((self.rows, lhs, rhs));
            }
        }
        let diff = lhs.clone().add_scaled(rhs, -F::ONE);
        let one = Lc::from_var(Var::ONE);
        let zero = Lc::zero();
        self.enforce(&diff, &one, &zero);
    }

    /// Convenience: enforce that `lc` evaluates to zero.
    pub fn enforce_zero(&mut self, lc: &Lc) {
        if !self.record_structure {
            self.rows += 1;
            return;
        }
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

    /// Total nonzero coefficients currently stored across A, B, and C.
    /// Useful for cost audits of dense linear gadgets.
    pub fn nonzero_entries(&self) -> usize {
        self.a_trips.len() + self.b_trips.len() + self.c_trips.len()
    }

    pub fn witness(&self) -> &[F] {
        &self.witness
    }

    /// Read-only view of the exact `(A, B, C)` sparse triplets, in emission
    /// order. Audit/export surface (e.g. the Lean artifact exporter); not
    /// used by prove/verify paths.
    pub fn sparse_triplets(&self) -> (&[(usize, usize, F)], &[(usize, usize, F)], &[(usize, usize, F)]) {
        (&self.a_trips, &self.b_trips, &self.c_trips)
    }

    /// Finish synthesis and transfer the exact sparse rows plus witness to a
    /// frontend-owned lowering pass.
    pub(crate) fn into_synthesis(self) -> R1csSynthesis {
        assert!(
            self.record_structure,
            "witness-only builder cannot export R1CS structure"
        );
        R1csSynthesis {
            a_trips: self.a_trips,
            b_trips: self.b_trips,
            c_trips: self.c_trips,
            witness: self.witness,
            rows: self.rows,
            canonical_u64_decompositions: self.canonical_u64_decompositions,
            balanced_ternary_decompositions: self.balanced_ternary_decompositions,
            boolean_columns: self.boolean_columns,
            centered_unit_columns: self.centered_unit_columns,
            seeded_phi81_a_blocks: self.seeded_phi81_a_blocks,
            poseidon2_traces: self.poseidon2_traces,
            polynomial_evaluation_traces: self.polynomial_evaluation_traces,
            product_sum_batch_traces: self.product_sum_batch_traces,
            centered_unit_traces: self.centered_unit_traces,
            equality_pairs: self.equality_pairs,
            row_family_ranges: self.row_family_ranges,
        }
    }

    /// Allocated witness columns that do not appear in any A/B/C row.
    ///
    /// This is an audit helper, not a proof of semantic binding: a column
    /// can appear in rows and still be under-constrained. It catches the
    /// narrower but dangerous class where a gadget allocates an authoritative
    /// value and never references it at all.
    pub fn unconstrained_columns(&self) -> Vec<usize> {
        assert!(self.record_structure, "witness-only builder has no constraint metadata");
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
        for block in &self.seeded_phi81_a_blocks {
            for &start in block.word_starts() {
                used[start..start + block.word_width()].fill(true);
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
        assert!(self.record_structure, "witness-only builder has no constraint metadata");
        let z = &self.witness;
        let mut az = sparse_matvec(&self.a_trips, self.rows, z);
        for block in &self.seeded_phi81_a_blocks {
            block.add_mul_into::<F, F>(z, &mut az, self.rows);
        }
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
