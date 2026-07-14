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

use super::encoding_trace::{
    KMulTraceEntry, PoseidonHashTraceEntry, PoseidonPermutationTraceEntry, R1csEncodingTrace, R1csStageCheckpoint,
    RingMulToom3TraceEntry, Sbox7TraceEntry,
};
pub use super::relation::{R1csRelation, R1csSnapshot};

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

/// Read-only assurance view of one balanced-ternary decomposition.
///
/// The emitted rows remain authoritative. Artifact exporters use this map to
/// recover the source field for an exact SeededPhi81 word without relying on
/// witness-allocation offsets.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BalancedTernaryAudit {
    pub field_col: usize,
    pub digit_cols: [usize; BALANCED_TERNARY_DIGITS],
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

/// Compact schedule for one shifted-base-3 canonical field opening.
///
/// The source R1CS remains authoritative. The selective compiler uses this
/// schedule to replace the indicator-heavy alphabet and borrow rows with
/// equivalent degree-eight CCS rows while retaining every borrow state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShiftedTernaryCanonicalTrace {
    pub(crate) digit_columns_start: usize,
    pub(crate) negative_columns_start: usize,
    pub(crate) borrow_columns_start: usize,
    pub(crate) digit_rows_start: usize,
    pub(crate) transition_rows_start: usize,
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

/// SSA-normalization boundary for one generated row family.
///
/// Columns below `first_allocated_column` predate the family. Assurance
/// exporters use this to distinguish verifier inputs from deterministic
/// columns allocated by the family; it never affects emitted constraints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramRangeAudit {
    pub name: &'static str,
    pub row_start: usize,
    pub row_end: usize,
    pub first_allocated_column: usize,
}

/// Exact wire schedule for one claimed-chain SumCheck round.
///
/// This is read-only assurance metadata.  The emitted R1CS rows remain the
/// authority; conformance exporters reconstruct the round program from these
/// columns and compare it to the exact sparse row interval before using the
/// corresponding Lean compiler theorem.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckRoundAudit {
    pub row_start: usize,
    pub row_end: usize,
    pub first_allocated_column: usize,
    pub coefficient_cols: Vec<[usize; 2]>,
    pub challenge_cols: [usize; 2],
    pub claim_in_cols: [usize; 2],
    pub claim_out_cols: [usize; 2],
}

/// Exact wire schedule for one commitment coordinate consumed by strict
/// PiDEC. This is assurance metadata only: exporters must reconstruct the
/// expected rows from the schedule and compare them with the emitted matrix.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiDecCommitmentAudit {
    pub d_col: usize,
    pub kappa_col: usize,
    pub data_cols: Vec<usize>,
}

/// The optional three-coordinate Nebula commitment carried by a CE claim.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiDecAdvAudit {
    pub ops: PiDecCommitmentAudit,
    pub is: PiDecCommitmentAudit,
    pub fs: PiDecCommitmentAudit,
}

/// Exact input-wire layout for one CE claim consumed by strict PiDEC.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiDecClaimAudit {
    pub commitment: PiDecCommitmentAudit,
    pub adv: Option<PiDecAdvAudit>,
    pub x_cols: Vec<usize>,
    pub x_rows: usize,
    pub x_width: usize,
    pub x_rows_col: usize,
    pub x_width_col: usize,
    pub m_in: usize,
    pub m_in_col: usize,
    pub y_ring_cols: Vec<Vec<usize>>,
    pub ct_cols: Vec<[usize; 2]>,
    pub r_cols: Vec<[usize; 2]>,
    pub s_col_cols: Vec<[usize; 2]>,
    pub fold_digest_cols: [usize; 4],
}

/// Complete strict-PiDEC input schedule for one emitted verifier invocation.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiDecStrictAudit {
    pub row_start: usize,
    pub row_end: usize,
    pub first_allocated_column: usize,
    pub radix: u32,
    pub parent: PiDecClaimAudit,
    pub children: Vec<PiDecClaimAudit>,
}

/// Exact input-wire ownership for one direct terminal-CE claim program.
///
/// This is read-only assurance metadata. It carries no validity bit and does
/// not replace any row: artifact exporters use it only to decode the witness,
/// claim, evaluation point, and sidecar columns consumed by the six exact
/// terminal-CE row phases.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalCeClaimAudit {
    pub row_start: usize,
    pub row_end: usize,
    pub first_allocated_column: usize,
    pub norm_bound: u32,
    pub expected_public_width: Option<usize>,
    pub structure_rows: usize,
    pub structure_columns: usize,
    pub witness_rows: usize,
    pub witness_columns: usize,
    pub witness_cols: Vec<usize>,
    pub norm_first_allocated_column: usize,
    pub commitment_cols: Vec<usize>,
    pub commitment_d: usize,
    pub commitment_kappa: usize,
    pub public_cols: Vec<usize>,
    pub public_rows: usize,
    pub public_width: usize,
    pub public_input_len: usize,
    pub point_cols: Vec<[usize; 2]>,
    pub evaluation_cols: Vec<Vec<usize>>,
    pub constant_term_cols: Vec<[usize; 2]>,
    pub nc_point_cols: Vec<[usize; 2]>,
    pub nc_evaluation_cols: Vec<usize>,
    pub nc_evaluation_lanes: usize,
}

/// Compact wire schedule for one generated beta-power ladder.
///
/// This is read-only assurance metadata. The exact emitted rows remain the
/// authority and artifact exporters must re-derive and compare every row
/// represented by this schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionLadderAudit {
    pub row_start: usize,
    pub row_end: usize,
    pub beta_columns: [usize; 2],
    pub power_columns: Vec<[usize; 2]>,
}

/// Semantic owner of one production PiRLC projection identity.
///
/// The variants follow the native verifier's projection schedule. `Standalone`
/// is reserved for isolated gadget calls; the NIFS compiler replaces it with
/// one of the protocol roles before exposing its audit trail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionIdentityRole {
    Standalone,
    CommitmentLane {
        lane: usize,
    },
    NebulaCommitmentLane {
        coordinate: ProjectionNebulaCoordinate,
        lane: usize,
    },
    ActiveXColumn {
        column: usize,
    },
    YRingLimb {
        row: usize,
        limb: usize,
    },
    YZColLimb {
        limb: usize,
    },
}

/// Native order of the three optional Nebula commitment coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionNebulaCoordinate {
    Ops,
    Is,
    Fs,
}

/// Semantic owner of affine rows emitted between projection identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionGlueRole {
    InactiveXZero,
    YRingPaddingZero { row: usize },
    YZColPaddingZero,
}

/// Exact contiguous range of affine projection glue rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionGlueAudit {
    pub role: ProjectionGlueRole,
    pub row_start: usize,
    pub row_end: usize,
}

/// Compact wire schedule for one complete projected ring-action identity.
/// Shared beta-ladder and rho-evaluation rows are referenced by columns; the
/// identity's own contiguous row range is recorded separately.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionIdentityAudit {
    pub role: ProjectionIdentityRole,
    pub row_start: usize,
    pub row_end: usize,
    pub power_columns: Vec<[usize; 2]>,
    pub rho_columns: Vec<Vec<usize>>,
    pub rho_evaluation_outputs: Vec<[usize; 2]>,
    pub input_columns: Vec<Vec<usize>>,
    pub input_evaluation_outputs: Vec<[usize; 2]>,
    pub pair_product_outputs: Vec<[usize; 2]>,
    pub output_columns: Vec<usize>,
    pub quotient_columns: Vec<usize>,
    pub output_evaluation: [usize; 2],
    pub quotient_evaluation: [usize; 2],
    pub quotient_phi_product: [usize; 2],
}

/// Indexed ownership marker for repeated constraint blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedRowFamilyRange {
    pub name: &'static str,
    pub index: usize,
    pub row_start: usize,
    pub row_end: usize,
}

/// Non-authoritative ownership marker for a contiguous allocation range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnFamilyRange {
    pub name: &'static str,
    pub column_start: usize,
    pub column_end: usize,
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
    boolean_constraint_rows: Vec<(usize, usize)>,
    centered_unit_columns: Vec<usize>,
    seeded_phi81_a_blocks: Vec<SeededPhi81LinearBlock>,
    poseidon2_traces: Vec<Poseidon2PermutationTrace>,
    poseidon2_hash_traces: Vec<Poseidon2HashTrace>,
    polynomial_evaluation_traces: Vec<PolynomialEvaluationTrace>,
    product_sum_batch_traces: Vec<ProductSumBatchTrace>,
    centered_unit_traces: Vec<CenteredUnitTrace>,
    shifted_ternary_canonical_traces: Vec<ShiftedTernaryCanonicalTrace>,
    equality_pairs: Vec<(usize, usize, usize)>,
    row_family_ranges: Vec<RowFamilyRange>,
    program_range_audits: Vec<ProgramRangeAudit>,
    sumcheck_round_audits: Vec<SumcheckRoundAudit>,
    pi_dec_strict_audits: Vec<PiDecStrictAudit>,
    terminal_ce_claim_audits: Vec<TerminalCeClaimAudit>,
    projection_ladder_audits: Vec<ProjectionLadderAudit>,
    projection_identity_audits: Vec<ProjectionIdentityAudit>,
    projection_glue_audits: Vec<ProjectionGlueAudit>,
    indexed_row_family_ranges: Vec<IndexedRowFamilyRange>,
    column_family_ranges: Vec<ColumnFamilyRange>,
    encoding_trace_enabled: bool,
    encoding_trace: R1csEncodingTrace,
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
    pub(crate) boolean_constraint_rows: Vec<(usize, usize)>,
    pub(crate) centered_unit_columns: Vec<usize>,
    pub(crate) seeded_phi81_a_blocks: Vec<SeededPhi81LinearBlock>,
    pub(crate) poseidon2_traces: Vec<Poseidon2PermutationTrace>,
    pub(crate) polynomial_evaluation_traces: Vec<PolynomialEvaluationTrace>,
    pub(crate) product_sum_batch_traces: Vec<ProductSumBatchTrace>,
    pub(crate) centered_unit_traces: Vec<CenteredUnitTrace>,
    pub(crate) shifted_ternary_canonical_traces: Vec<ShiftedTernaryCanonicalTrace>,
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
    #[doc(hidden)]
    pub fn new_witness_only() -> Self {
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
            boolean_constraint_rows: Vec::new(),
            centered_unit_columns: Vec::new(),
            seeded_phi81_a_blocks: Vec::new(),
            poseidon2_traces: Vec::new(),
            poseidon2_hash_traces: Vec::new(),
            polynomial_evaluation_traces: Vec::new(),
            product_sum_batch_traces: Vec::new(),
            centered_unit_traces: Vec::new(),
            shifted_ternary_canonical_traces: Vec::new(),
            equality_pairs: Vec::new(),
            row_family_ranges: Vec::new(),
            program_range_audits: Vec::new(),
            sumcheck_round_audits: Vec::new(),
            pi_dec_strict_audits: Vec::new(),
            terminal_ce_claim_audits: Vec::new(),
            projection_ladder_audits: Vec::new(),
            projection_identity_audits: Vec::new(),
            projection_glue_audits: Vec::new(),
            indexed_row_family_ranges: Vec::new(),
            column_family_ranges: Vec::new(),
            encoding_trace_enabled: false,
            encoding_trace: R1csEncodingTrace::default(),
        }
    }

    /// Record high-level gadget provenance for low-norm compilation.
    ///
    /// This does not alter the emitted R1CS. It must be enabled before the
    /// relevant gadgets run; an encoder rejects incomplete provenance when a
    /// traced temporary escapes its recorded row range.
    pub fn enable_encoding_trace(&mut self) {
        self.encoding_trace_enabled = true;
    }

    pub fn encoding_trace(&self) -> &R1csEncodingTrace {
        &self.encoding_trace
    }

    /// Begin a named diagnostic stage at the current row/column frontier.
    /// No-op unless encoding provenance is enabled.
    pub fn begin_encoding_stage(&mut self, label: &'static str) {
        if self.encoding_trace_enabled {
            self.encoding_trace.push_stage(R1csStageCheckpoint {
                label,
                row: self.rows(),
                col: self.cols(),
            });
        }
    }

    pub(crate) fn encoding_trace_enabled(&self) -> bool {
        self.encoding_trace_enabled
    }

    pub(crate) fn record_sbox7_encoding(&mut self, entry: Sbox7TraceEntry) {
        if self.encoding_trace_enabled {
            self.encoding_trace.push_sbox7(entry);
        }
    }

    pub(crate) fn record_k_mul_encoding(&mut self, entry: KMulTraceEntry) {
        if self.encoding_trace_enabled {
            self.encoding_trace.push_k_mul(entry);
        }
    }

    pub(crate) fn record_poseidon_permutation_encoding(&mut self, entry: PoseidonPermutationTraceEntry) {
        if self.encoding_trace_enabled {
            self.encoding_trace.push_poseidon_permutation(entry);
        }
    }

    pub(crate) fn record_poseidon_hash_encoding(&mut self, entry: PoseidonHashTraceEntry) {
        if self.encoding_trace_enabled {
            self.encoding_trace.push_poseidon_hash(entry);
        }
    }

    pub(crate) fn record_ring_mul_toom3_encoding(&mut self, entry: RingMulToom3TraceEntry) {
        if self.encoding_trace_enabled {
            self.encoding_trace.push_ring_mul_toom3(entry);
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

    /// Assurance view of the exact balanced-ternary witness columns reused
    /// by SIS/Ajtai bindings. The field rows remain authoritative; this
    /// accessor exists so adversarial tests and artifact exporters can mutate
    /// or audit the same columns without depending on allocation offsets.
    pub fn balanced_ternary_digit_columns(&self, field: Var) -> Option<Vec<usize>> {
        self.balanced_ternary_decomposition(field)
            .map(|digits| digits.into_iter().map(Var::col).collect())
    }

    /// Exact source-field to digit-word maps recorded by production synthesis.
    #[doc(hidden)]
    pub fn balanced_ternary_audits(&self) -> Vec<BalancedTernaryAudit> {
        self.balanced_ternary_decompositions
            .iter()
            .map(|decomposition| BalancedTernaryAudit {
                field_col: decomposition.field_col,
                digit_cols: decomposition.digit_cols,
            })
            .collect()
    }

    pub(crate) fn record_boolean(&mut self, value: Var) {
        if self.record_structure {
            self.boolean_columns.push(value.col());
        }
    }

    pub(crate) fn record_boolean_constraint(&mut self, value: Var, row: usize) {
        if self.record_structure {
            self.boolean_constraint_rows.push((value.col(), row));
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

    pub(crate) fn record_shifted_ternary_canonical_trace(&mut self, trace: ShiftedTernaryCanonicalTrace) {
        if self.record_structure {
            self.shifted_ternary_canonical_traces.push(trace);
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

    /// Read-only SSA boundaries for exact generated owner programs.
    #[doc(hidden)]
    pub fn program_range_audits(&self) -> &[ProgramRangeAudit] {
        &self.program_range_audits
    }

    /// Exact wire schedules for generated claimed-chain SumCheck rounds.
    #[doc(hidden)]
    pub fn sumcheck_round_audits(&self) -> &[SumcheckRoundAudit] {
        &self.sumcheck_round_audits
    }

    /// Exact input schedules for strict PiDEC compiler invocations.
    #[doc(hidden)]
    pub fn pi_dec_strict_audits(&self) -> &[PiDecStrictAudit] {
        &self.pi_dec_strict_audits
    }

    /// Exact wire ownership for direct terminal-CE claim compilers.
    #[doc(hidden)]
    pub fn terminal_ce_claim_audits(&self) -> &[TerminalCeClaimAudit] {
        &self.terminal_ce_claim_audits
    }

    pub(crate) fn record_program_range(&mut self, name: &'static str, row_start: usize, first_allocated_column: usize) {
        if self.record_structure {
            self.program_range_audits.push(ProgramRangeAudit {
                name,
                row_start,
                row_end: self.rows,
                first_allocated_column,
            });
        }
    }

    pub(crate) fn record_sumcheck_round(&mut self, audit: SumcheckRoundAudit) {
        if self.record_structure {
            debug_assert_eq!(audit.row_end, self.rows);
            self.sumcheck_round_audits.push(audit);
        }
    }

    pub(crate) fn record_pi_dec_strict(&mut self, audit: PiDecStrictAudit) {
        if self.record_structure {
            debug_assert_eq!(audit.row_end, self.rows);
            self.pi_dec_strict_audits.push(audit);
        }
    }

    pub(crate) fn record_terminal_ce_claim(&mut self, audit: TerminalCeClaimAudit) {
        if self.record_structure {
            debug_assert_eq!(audit.row_end, self.rows);
            self.terminal_ce_claim_audits.push(audit);
        }
    }

    /// Exact wire schedules for generated beta-power ladders.
    #[doc(hidden)]
    pub fn projection_ladder_audits(&self) -> &[ProjectionLadderAudit] {
        &self.projection_ladder_audits
    }

    /// Exact wire schedules for generated projected ring-action identities.
    #[doc(hidden)]
    pub fn projection_identity_audits(&self) -> &[ProjectionIdentityAudit] {
        &self.projection_identity_audits
    }

    /// Exact affine row ranges between production projection identities.
    #[doc(hidden)]
    pub fn projection_glue_audits(&self) -> &[ProjectionGlueAudit] {
        &self.projection_glue_audits
    }

    pub(crate) fn record_projection_ladder(&mut self, audit: ProjectionLadderAudit) {
        if self.record_structure {
            debug_assert_eq!(audit.row_end, self.rows);
            self.projection_ladder_audits.push(audit);
        }
    }

    pub(crate) fn record_projection_identity(&mut self, audit: ProjectionIdentityAudit) {
        if self.record_structure {
            debug_assert_eq!(audit.row_end, self.rows);
            self.projection_identity_audits.push(audit);
        }
    }

    pub(crate) fn assign_projection_identity_roles(&mut self, first: usize, roles: &[ProjectionIdentityRole]) {
        if !self.record_structure {
            return;
        }
        let identities = self
            .projection_identity_audits
            .get_mut(first..)
            .expect("projection identity role start exceeds audit count");
        assert_eq!(identities.len(), roles.len(), "projection identity role census");
        for (identity, role) in identities.iter_mut().zip(roles) {
            identity.role = *role;
        }
    }

    pub(crate) fn record_projection_glue(&mut self, role: ProjectionGlueRole, row_start: usize) {
        assert!(row_start <= self.rows, "projection glue start exceeds builder cursor");
        if self.record_structure && row_start != self.rows {
            self.projection_glue_audits.push(ProjectionGlueAudit {
                role,
                row_start,
                row_end: self.rows,
            });
        }
    }

    /// Record one repeated row block, identified by its verifier-fixed index.
    pub fn record_indexed_row_family(&mut self, name: &'static str, index: usize, row_start: usize) {
        assert!(
            row_start <= self.rows,
            "indexed row-family start exceeds builder cursor"
        );
        if self.record_structure {
            self.indexed_row_family_ranges.push(IndexedRowFamilyRange {
                name,
                index,
                row_start,
                row_end: self.rows,
            });
        }
    }

    pub fn indexed_row_family_ranges(&self) -> &[IndexedRowFamilyRange] {
        &self.indexed_row_family_ranges
    }

    /// Record ownership of columns allocated since `column_start`.
    pub fn record_column_family(&mut self, name: &'static str, column_start: usize) {
        assert!(
            column_start <= self.witness.len(),
            "column-family start exceeds builder cursor"
        );
        if self.record_structure {
            self.column_family_ranges.push(ColumnFamilyRange {
                name,
                column_start,
                column_end: self.witness.len(),
            });
        }
    }

    pub fn column_family_ranges(&self) -> &[ColumnFamilyRange] {
        &self.column_family_ranges
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

    /// Read-only view of the explicitly stored `(A, B, C)` sparse triplets,
    /// in emission order.
    ///
    /// Compact seeded Phi81 contributions to `A` are intentionally absent
    /// from the first slice. Audit/export consumers that need the complete
    /// matrix must also consume [`Self::seeded_phi81_a_blocks`].
    pub fn sparse_triplets(&self) -> (&[(usize, usize, F)], &[(usize, usize, F)], &[(usize, usize, F)]) {
        (&self.a_trips, &self.b_trips, &self.c_trips)
    }

    /// Compact seeded Phi81 blocks contributing to the `A` matrix.
    ///
    /// This is a read-only assurance surface. Each block is part of the exact
    /// generated relation even though its coefficients are not materialized
    /// in [`Self::sparse_triplets`].
    #[doc(hidden)]
    pub fn seeded_phi81_a_blocks(&self) -> &[SeededPhi81LinearBlock] {
        &self.seeded_phi81_a_blocks
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
            boolean_constraint_rows: self.boolean_constraint_rows,
            centered_unit_columns: self.centered_unit_columns,
            seeded_phi81_a_blocks: self.seeded_phi81_a_blocks,
            poseidon2_traces: self.poseidon2_traces,
            polynomial_evaluation_traces: self.polynomial_evaluation_traces,
            product_sum_batch_traces: self.product_sum_batch_traces,
            centered_unit_traces: self.centered_unit_traces,
            shifted_ternary_canonical_traces: self.shifted_ternary_canonical_traces,
            equality_pairs: self.equality_pairs,
            row_family_ranges: self.row_family_ranges,
        }
    }

    /// Freeze the current relation and witness for deterministic encoding.
    /// Duplicate terms in a linear combination are coalesced, so equivalent
    /// builder call patterns yield the same row representation.
    pub fn snapshot(&self) -> R1csSnapshot {
        assert!(
            self.record_structure,
            "witness-only builder cannot snapshot R1CS structure"
        );
        R1csSnapshot::from_builder_parts(
            &self.a_trips,
            &self.b_trips,
            &self.c_trips,
            &self.seeded_phi81_a_blocks,
            self.rows,
            self.witness.clone(),
        )
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
