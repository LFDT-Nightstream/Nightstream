//! Gadget provenance needed by the low-norm `enc(F')` compiler.
//!
//! The field R1CS remains the semantic authority. This trace only records
//! which consecutive R1CS rows came from algebraically stronger gadgets, so
//! the encoder can replace their temporary product wires with exact CCS gates.
//!
//! Owns: immutable row/column provenance for exact lowering validators.
//!
//! Does not own: source semantics, row acceptance, or protocol authority.
//!
//! Emits constraints: no.
//!
//! | Trace family | Mathematical program named | Replacement allowed only after |
//! |---|---|---|
//! | Balanced ternary | 41 digit openings and canonical borrow | all 124 rows match exactly |
//! | Canonical u64 | one field, 64 bits, flag, inverse, and 69 rows | every role and row matches exactly |
//! | Poseidon2 | `x^7` permutations and hashes | every product row matches |
//! | Mod-5 chunk | 16-bit chunk decomposition and residue range | all 20 rows and 19 columns match |
//! | Product sum | bounded mixed SSA identities | identity and retained-rank checks pass |
//! | Ring/selection | Toom-3 and first-accepted products | exact row and escape checks pass |

use std::ops::Range;

use neo_math::F;

use super::builder::{Lc, ProductSumBatchTrace, Var, BALANCED_TERNARY_DIGITS};

/// Native order of the three optional Nebula commitment coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionNebulaCoordinate {
    Ops,
    Is,
    Fs,
}

/// Semantic owner of one production projected ring-action identity.
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
}

/// Deliberate balanced-ternary provenance corruptions used by fail-closed
/// integration tests.
///
/// The source relation is never changed. Each mutation must therefore be
/// rejected before the lowering shares or projects any source column.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum BalancedTernaryTraceTestMutation {
    FieldColumn { column: usize },
    DigitColumn { index: usize, column: usize },
    NegativeColumn { index: usize, column: usize },
    BorrowColumn { index: usize, column: usize },
    DigitRows { rows: Range<usize> },
    ReconstructionRow { row: usize },
    TransitionRows { rows: Range<usize> },
}

/// Deliberate canonical-u64 provenance corruptions used by fail-closed tests.
///
/// These mutations never alter authoritative R1CS rows.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum CanonicalU64TraceTestMutation {
    FieldColumn { column: usize },
    BitColumn { index: usize, column: usize },
    HighIsMaxColumn { column: usize },
    InverseColumn { column: usize },
    SourceRows { rows: Range<usize> },
}

/// Deliberate packed-mod-5 provenance corruptions used by fail-closed tests.
///
/// These mutations never alter the authoritative source R1CS.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum Mod5TraceTestMutation {
    SourceRowEnd { row_end: usize },
    AllocatedColumnEnd { column_end: usize },
    ChunkBitColumn { index: usize, column: usize },
    IndexColumn { column: usize },
    QuotientBitColumn { index: usize, column: usize },
    IndexProductColumn { index: usize, column: usize },
}

/// Deliberate chunk-acceptance provenance corruptions used by fail-closed
/// lowering tests. These mutations never alter authoritative source rows.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum AcceptanceTraceTestMutation {
    SourceRowEnd { row_end: usize },
    AllocatedColumnEnd { column_end: usize },
    ChunkBitColumn { index: usize, column: usize },
    AcceptColumn { column: usize },
    InverseColumn { column: usize },
}

/// Deliberate trace corruptions used by fail-closed integration tests.
///
/// This mutates provenance only; the source R1CS remains unchanged. The
/// gadget-native compiler must reject every variant before replacing rows.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum ProductSumTraceTestMutation {
    RowEnd {
        row_end: usize,
    },
    AllocatedColumn {
        offset: usize,
        column: usize,
    },
    RetainedColumns {
        columns: Vec<usize>,
    },
    CopyIdentity {
        from: usize,
        to: usize,
    },
    FactorCoefficient {
        identity: usize,
        factor: usize,
        coefficient: F,
    },
    ClearResult {
        identity: usize,
    },
}

/// Deliberate projection-identity provenance corruptions for fail-closed tests.
/// These never mutate the authoritative R1CS rows.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum ProjectionIdentityTraceTestMutation {
    Role {
        role: ProjectionIdentityRole,
    },
    SourceRowEnd {
        row_end: usize,
    },
    AllocatedColumnEnd {
        column_end: usize,
    },
    InputEvaluation {
        offset: usize,
        evaluation: usize,
    },
    PairProduct {
        offset: usize,
        product: usize,
    },
    FinalLimbRowEnd {
        row_end: usize,
    },
    InputColumn {
        pair: usize,
        coefficient: usize,
        column: usize,
    },
}

/// Deliberate polynomial-evaluation provenance corruptions for fail-closed
/// tests. The source rows remain unchanged.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum PolynomialEvaluationTraceTestMutation {
    RowEnd {
        row_end: usize,
    },
    AllocatedColumn {
        offset: usize,
        column: usize,
    },
    CoefficientColumn {
        offset: usize,
        column: usize,
    },
    PowerColumn {
        coefficient: usize,
        limb: usize,
        column: usize,
    },
    OutputColumn {
        limb: usize,
        column: usize,
    },
}

/// Deliberate Poseidon2 S-box provenance corruptions for fail-closed tests.
/// These never mutate the authoritative source R1CS.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum Sbox7TraceTestMutation {
    InputColumn { offset: usize, column: usize },
    IntermediateColumn { index: usize, column: usize },
    OutputColumn { column: usize },
    SourceRows { rows: Range<usize> },
}

/// Deliberate Poseidon2 permutation provenance corruptions for fail-closed
/// tests. These never mutate the authoritative source R1CS.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum PoseidonPermutationTraceTestMutation {
    InputColumn { lane: usize, column: usize },
    AllocatedColumns { columns: Range<usize> },
    OutputColumn { lane: usize, column: usize },
    SourceRows { rows: Range<usize> },
}

/// Deliberate Poseidon2 sponge provenance corruptions for fail-closed tests.
/// These never mutate the authoritative source R1CS.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum PoseidonHashTraceTestMutation {
    InputLen { input_len: usize },
    InputColumn { offset: usize, column: usize },
    ZeroColumn { column: usize },
    OutputColumn { lane: usize, column: usize },
    PermutationRange { range: Range<usize> },
    SourceRows { rows: Range<usize> },
}

/// One Poseidon2 `x -> x^7` expansion.
#[derive(Clone, Debug)]
pub struct Sbox7TraceEntry {
    pub input: Lc,
    pub intermediates: [Var; 3],
    pub output: Var,
    pub source_rows: Range<usize>,
}

/// One complete WIDTH-8 Poseidon2 permutation.
#[derive(Clone, Debug)]
pub struct PoseidonPermutationTraceEntry {
    pub input_columns: [usize; 8],
    pub allocated_columns: Range<usize>,
    pub output_columns: [usize; 8],
    pub source_rows: Range<usize>,
}

/// One variable-length Poseidon2 hash and its nested permutation range.
#[derive(Clone, Debug)]
pub struct PoseidonHashTraceEntry {
    pub input_len: usize,
    pub input_columns: Vec<usize>,
    pub zero_column: usize,
    pub output_columns: [usize; 4],
    pub permutation_range: Range<usize>,
    pub source_rows: Range<usize>,
}

/// One multiplication in `K = F[X]/(X^2 - W)`.
#[derive(Clone, Debug)]
pub struct KMulTraceEntry {
    pub a: [Lc; 2],
    pub b: [Lc; 2],
    pub intermediates: [Var; 3],
    pub output: [Var; 2],
    pub source_rows: Range<usize>,
}

/// One exact polynomial evaluation emitted by `enforce_eval_at_beta`.
///
/// The source rows remain authoritative. Compact lowerings must validate the
/// full row/column interval, ordered coefficient and power columns, and every
/// use of the allocated temporaries before replacing it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolynomialEvaluationTraceEntry {
    pub row_start: usize,
    pub row_end: usize,
    pub allocated_columns: Vec<usize>,
    pub coefficient_cols: Vec<usize>,
    pub power_cols: Vec<[usize; 2]>,
    pub output_cols: [usize; 2],
}

/// Exact production ownership of one complete projected ring-action identity.
///
/// Indices point into this same [`R1csEncodingTrace`]. They are diagnostic
/// provenance only: a lowering must revalidate the referenced source rows,
/// topological dependencies, final checks, and non-escape before use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionIdentityTraceEntry {
    pub role: ProjectionIdentityRole,
    pub source_rows: Range<usize>,
    pub allocated_columns: Range<usize>,
    pub power_columns: Vec<[usize; 2]>,
    pub rho_columns: Vec<Vec<usize>>,
    pub rho_evaluation_outputs: Vec<[usize; 2]>,
    pub input_columns: Vec<Vec<usize>>,
    pub output_columns: Vec<usize>,
    pub quotient_columns: Vec<usize>,
    pub input_evaluations: Range<usize>,
    pub pair_products: Range<usize>,
    pub output_evaluation: usize,
    pub quotient_evaluation: usize,
    pub quotient_phi_product: usize,
    pub final_limb_rows: Range<usize>,
}

/// One length-18 schoolbook convolution used by the 3-way ring product.
#[derive(Clone, Debug)]
pub struct Toom3ConvolutionTrace {
    pub lhs: Vec<Lc>,
    pub rhs: Vec<Lc>,
    /// Row-major `lhs[i] * rhs[j]` product wires.
    pub products: Vec<Var>,
}

/// One complete production Toom-3 ring multiplication.
#[derive(Clone, Debug)]
pub struct RingMulToom3TraceEntry {
    pub rho: Vec<Var>,
    pub c: Vec<Var>,
    pub convolutions: Vec<Toom3ConvolutionTrace>,
    /// Reduced output expressions in terms of the product wires.
    pub reduced_output_lcs: Vec<Lc>,
    pub output: Vec<Var>,
    pub source_rows: Range<usize>,
}

/// Three temporary products allocated for one candidate in a first-accepted
/// selection row.
#[derive(Clone, Copy, Debug)]
pub struct FirstAcceptedSelectionProducts {
    pub symbol: Var,
    pub accepted: Var,
    pub prefix: Var,
}

/// One bounded candidate-selector block for a single accepted-output position.
///
/// The trace covers only the product definitions and their three aggregate
/// bindings. Booleanity and `sum(one_hot) = 1` remain ordinary source rows.
#[derive(Clone, Debug)]
pub struct FirstAcceptedSelectionTraceEntry {
    pub one_hot: Vec<Var>,
    pub symbols: Vec<Var>,
    pub accepts: Vec<Var>,
    pub prefixes: Vec<Var>,
    pub products: Vec<FirstAcceptedSelectionProducts>,
    pub output: Var,
    pub position: usize,
    /// Unclaimed source rows proving bitness and `sum(one_hot) = 1`.
    pub one_hot_rows: Range<usize>,
    pub product_rows: Range<usize>,
    pub bind_rows: Range<usize>,
}

/// Exact source program for one canonical balanced-ternary field opening.
///
/// The source R1CS remains authoritative. The low-norm compiler re-derives
/// and compares all two digit rows, the reconstruction row, and every borrow
/// transition before sharing field/digit coordinates or projecting negative
/// indicators.
#[derive(Clone, Debug)]
pub struct BalancedTernaryOpeningTraceEntry {
    pub field_col: usize,
    pub digit_cols: [usize; BALANCED_TERNARY_DIGITS],
    pub negative_cols: [usize; BALANCED_TERNARY_DIGITS],
    pub borrow_cols: [usize; BALANCED_TERNARY_DIGITS - 1],
    pub digit_rows: Range<usize>,
    pub reconstruction_row: usize,
    pub transition_rows: Range<usize>,
}

/// Exact source program emitted by one canonical Goldilocks decomposition.
///
/// This is provenance only. A lowering must validate all 69 source rows under
/// these roles before it may use the trace for planning or cost attribution.
#[derive(Clone, Debug)]
pub struct CanonicalU64TraceEntry {
    pub field: Var,
    pub bits: [Var; 64],
    pub high_is_max: Var,
    pub inverse: Var,
    pub source_rows: Range<usize>,
}

/// Exact four-row source program for one 16-bit chunk acceptance decision.
///
/// Production allocates `accept` and `inverse` consecutively. The fourth row
/// canonicalizes the rejected inverse to zero, making projection invertible.
#[derive(Clone, Debug)]
pub struct AcceptanceTraceEntry {
    pub chunk_bits: [Var; 16],
    pub accept: Var,
    pub inverse: Var,
    pub source_rows: Range<usize>,
    pub allocated_columns: Range<usize>,
}

/// Exact source program for one 16-bit sampler chunk's mod-5 block.
///
/// Production allocates 19 consecutive columns in the order `index`,
/// `quotient`, three index-polynomial products, then fourteen quotient bits.
/// It emits four index-polynomial rows, fourteen bit rows, one quotient
/// reconstruction row, and one chunk-decomposition row.
#[derive(Clone, Debug)]
pub struct Mod5TraceEntry {
    pub chunk_bits: [Var; 16],
    pub index: Var,
    pub quotient: Var,
    pub index_products: [Var; 3],
    pub quotient_bits: [Var; 14],
    pub source_rows: Range<usize>,
    pub allocated_columns: Range<usize>,
}

/// Start of one named, sequential circuit-emission stage.
///
/// Checkpoints are diagnostic provenance only. The following checkpoint ends
/// the stage; the final `complete` checkpoint closes the last stage.
#[derive(Clone, Debug)]
pub struct R1csStageCheckpoint {
    pub label: &'static str,
    pub row: usize,
    pub col: usize,
}

/// Append-only high-level provenance for one R1CS emission.
#[derive(Clone, Debug, Default)]
pub struct R1csEncodingTrace {
    sbox7: Vec<Sbox7TraceEntry>,
    poseidon_permutations: Vec<PoseidonPermutationTraceEntry>,
    poseidon_hashes: Vec<PoseidonHashTraceEntry>,
    k_muls: Vec<KMulTraceEntry>,
    ring_muls_toom3: Vec<RingMulToom3TraceEntry>,
    first_accepted_selections: Vec<FirstAcceptedSelectionTraceEntry>,
    canonical_u64_decompositions: Vec<CanonicalU64TraceEntry>,
    acceptance_chunks: Vec<AcceptanceTraceEntry>,
    mod5_chunks: Vec<Mod5TraceEntry>,
    balanced_ternary_openings: Vec<BalancedTernaryOpeningTraceEntry>,
    product_sum_batches: Vec<ProductSumBatchTrace>,
    polynomial_evaluations: Vec<PolynomialEvaluationTraceEntry>,
    projection_identities: Vec<ProjectionIdentityTraceEntry>,
    stages: Vec<R1csStageCheckpoint>,
}

impl R1csEncodingTrace {
    pub fn sbox7(&self) -> &[Sbox7TraceEntry] {
        &self.sbox7
    }

    pub fn k_muls(&self) -> &[KMulTraceEntry] {
        &self.k_muls
    }

    pub fn poseidon_permutations(&self) -> &[PoseidonPermutationTraceEntry] {
        &self.poseidon_permutations
    }

    pub fn poseidon_hashes(&self) -> &[PoseidonHashTraceEntry] {
        &self.poseidon_hashes
    }

    pub fn ring_muls_toom3(&self) -> &[RingMulToom3TraceEntry] {
        &self.ring_muls_toom3
    }

    pub fn first_accepted_selections(&self) -> &[FirstAcceptedSelectionTraceEntry] {
        &self.first_accepted_selections
    }

    pub fn canonical_u64_decompositions(&self) -> &[CanonicalU64TraceEntry] {
        &self.canonical_u64_decompositions
    }

    pub fn acceptance_chunks(&self) -> &[AcceptanceTraceEntry] {
        &self.acceptance_chunks
    }

    pub fn mod5_chunks(&self) -> &[Mod5TraceEntry] {
        &self.mod5_chunks
    }

    pub fn balanced_ternary_openings(&self) -> &[BalancedTernaryOpeningTraceEntry] {
        &self.balanced_ternary_openings
    }

    pub(crate) fn product_sum_batches(&self) -> &[ProductSumBatchTrace] {
        &self.product_sum_batches
    }

    pub fn polynomial_evaluations(&self) -> &[PolynomialEvaluationTraceEntry] {
        &self.polynomial_evaluations
    }

    pub fn projection_identities(&self) -> &[ProjectionIdentityTraceEntry] {
        &self.projection_identities
    }

    pub fn stages(&self) -> &[R1csStageCheckpoint] {
        &self.stages
    }

    /// Corrupt one product-sum trace without touching authoritative R1CS rows.
    #[doc(hidden)]
    pub fn apply_product_sum_trace_test_mutation(&mut self, batch: usize, mutation: ProductSumTraceTestMutation) {
        let batch = &mut self.product_sum_batches[batch];
        match mutation {
            ProductSumTraceTestMutation::RowEnd { row_end } => batch.row_end = row_end,
            ProductSumTraceTestMutation::AllocatedColumn { offset, column } => {
                batch.allocated_columns[offset] = column;
            }
            ProductSumTraceTestMutation::RetainedColumns { columns } => batch.retained_columns = columns,
            ProductSumTraceTestMutation::CopyIdentity { from, to } => {
                let identity = batch.identities[from].clone();
                batch.identities[to] = identity;
            }
            ProductSumTraceTestMutation::FactorCoefficient {
                identity,
                factor,
                coefficient,
            } => batch.identities[identity].factors[factor].coefficient = coefficient,
            ProductSumTraceTestMutation::ClearResult { identity } => {
                batch.identities[identity].result = Lc::zero();
            }
        }
    }

    #[doc(hidden)]
    pub fn apply_projection_identity_trace_test_mutation(
        &mut self,
        identity: usize,
        mutation: ProjectionIdentityTraceTestMutation,
    ) {
        let identity = &mut self.projection_identities[identity];
        match mutation {
            ProjectionIdentityTraceTestMutation::Role { role } => {
                identity.role = role;
            }
            ProjectionIdentityTraceTestMutation::SourceRowEnd { row_end } => {
                identity.source_rows.end = row_end;
            }
            ProjectionIdentityTraceTestMutation::AllocatedColumnEnd { column_end } => {
                identity.allocated_columns.end = column_end;
            }
            ProjectionIdentityTraceTestMutation::InputEvaluation { offset, evaluation } => {
                assert!(offset < identity.input_evaluations.len());
                if offset == 0 {
                    identity.input_evaluations.start = evaluation;
                } else {
                    identity.input_evaluations.end = evaluation + 1;
                }
            }
            ProjectionIdentityTraceTestMutation::PairProduct { offset, product } => {
                assert!(offset < identity.pair_products.len());
                if offset == 0 {
                    identity.pair_products.start = product;
                } else {
                    identity.pair_products.end = product + 1;
                }
            }
            ProjectionIdentityTraceTestMutation::FinalLimbRowEnd { row_end } => {
                identity.final_limb_rows.end = row_end;
            }
            ProjectionIdentityTraceTestMutation::InputColumn {
                pair,
                coefficient,
                column,
            } => identity.input_columns[pair][coefficient] = column,
        }
    }

    #[doc(hidden)]
    pub fn apply_polynomial_evaluation_trace_test_mutation(
        &mut self,
        evaluation: usize,
        mutation: PolynomialEvaluationTraceTestMutation,
    ) {
        let evaluation = &mut self.polynomial_evaluations[evaluation];
        match mutation {
            PolynomialEvaluationTraceTestMutation::RowEnd { row_end } => evaluation.row_end = row_end,
            PolynomialEvaluationTraceTestMutation::AllocatedColumn { offset, column } => {
                evaluation.allocated_columns[offset] = column;
            }
            PolynomialEvaluationTraceTestMutation::CoefficientColumn { offset, column } => {
                evaluation.coefficient_cols[offset] = column;
            }
            PolynomialEvaluationTraceTestMutation::PowerColumn {
                coefficient,
                limb,
                column,
            } => evaluation.power_cols[coefficient][limb] = column,
            PolynomialEvaluationTraceTestMutation::OutputColumn { limb, column } => {
                evaluation.output_cols[limb] = column;
            }
        }
    }

    /// Corrupt one balanced-ternary trace without touching authoritative rows.
    #[doc(hidden)]
    pub fn apply_balanced_ternary_trace_test_mutation(
        &mut self,
        opening: usize,
        mutation: BalancedTernaryTraceTestMutation,
    ) {
        let opening = &mut self.balanced_ternary_openings[opening];
        match mutation {
            BalancedTernaryTraceTestMutation::FieldColumn { column } => opening.field_col = column,
            BalancedTernaryTraceTestMutation::DigitColumn { index, column } => {
                opening.digit_cols[index] = column;
            }
            BalancedTernaryTraceTestMutation::NegativeColumn { index, column } => {
                opening.negative_cols[index] = column;
            }
            BalancedTernaryTraceTestMutation::BorrowColumn { index, column } => {
                opening.borrow_cols[index] = column;
            }
            BalancedTernaryTraceTestMutation::DigitRows { rows } => opening.digit_rows = rows,
            BalancedTernaryTraceTestMutation::ReconstructionRow { row } => opening.reconstruction_row = row,
            BalancedTernaryTraceTestMutation::TransitionRows { rows } => opening.transition_rows = rows,
        }
    }

    /// Corrupt one canonical-u64 trace without touching authoritative rows.
    #[doc(hidden)]
    pub fn apply_canonical_u64_trace_test_mutation(
        &mut self,
        decomposition: usize,
        mutation: CanonicalU64TraceTestMutation,
    ) {
        let decomposition = &mut self.canonical_u64_decompositions[decomposition];
        match mutation {
            CanonicalU64TraceTestMutation::FieldColumn { column } => {
                decomposition.field = Var::from_column_for_trace(column);
            }
            CanonicalU64TraceTestMutation::BitColumn { index, column } => {
                decomposition.bits[index] = Var::from_column_for_trace(column);
            }
            CanonicalU64TraceTestMutation::HighIsMaxColumn { column } => {
                decomposition.high_is_max = Var::from_column_for_trace(column);
            }
            CanonicalU64TraceTestMutation::InverseColumn { column } => {
                decomposition.inverse = Var::from_column_for_trace(column);
            }
            CanonicalU64TraceTestMutation::SourceRows { rows } => decomposition.source_rows = rows,
        }
    }

    /// Corrupt one mod-5 trace without touching authoritative source rows.
    #[doc(hidden)]
    pub fn apply_mod5_trace_test_mutation(&mut self, chunk: usize, mutation: Mod5TraceTestMutation) {
        let chunk = &mut self.mod5_chunks[chunk];
        match mutation {
            Mod5TraceTestMutation::SourceRowEnd { row_end } => chunk.source_rows.end = row_end,
            Mod5TraceTestMutation::AllocatedColumnEnd { column_end } => {
                chunk.allocated_columns.end = column_end;
            }
            Mod5TraceTestMutation::ChunkBitColumn { index, column } => {
                chunk.chunk_bits[index] = Var::from_column_for_trace(column);
            }
            Mod5TraceTestMutation::IndexColumn { column } => {
                chunk.index = Var::from_column_for_trace(column);
            }
            Mod5TraceTestMutation::QuotientBitColumn { index, column } => {
                chunk.quotient_bits[index] = Var::from_column_for_trace(column);
            }
            Mod5TraceTestMutation::IndexProductColumn { index, column } => {
                chunk.index_products[index] = Var::from_column_for_trace(column);
            }
        }
    }

    /// Corrupt one acceptance trace without touching authoritative source rows.
    #[doc(hidden)]
    pub fn apply_acceptance_trace_test_mutation(&mut self, chunk: usize, mutation: AcceptanceTraceTestMutation) {
        let chunk = &mut self.acceptance_chunks[chunk];
        match mutation {
            AcceptanceTraceTestMutation::SourceRowEnd { row_end } => chunk.source_rows.end = row_end,
            AcceptanceTraceTestMutation::AllocatedColumnEnd { column_end } => {
                chunk.allocated_columns.end = column_end;
            }
            AcceptanceTraceTestMutation::ChunkBitColumn { index, column } => {
                chunk.chunk_bits[index] = Var::from_column_for_trace(column);
            }
            AcceptanceTraceTestMutation::AcceptColumn { column } => {
                chunk.accept = Var::from_column_for_trace(column);
            }
            AcceptanceTraceTestMutation::InverseColumn { column } => {
                chunk.inverse = Var::from_column_for_trace(column);
            }
        }
    }

    /// Corrupt one S-box trace without touching authoritative source rows.
    #[doc(hidden)]
    pub fn apply_sbox7_trace_test_mutation(&mut self, sbox: usize, mutation: Sbox7TraceTestMutation) {
        let sbox = &mut self.sbox7[sbox];
        match mutation {
            Sbox7TraceTestMutation::InputColumn { offset, column } => {
                sbox.input.terms[offset].0 = column;
            }
            Sbox7TraceTestMutation::IntermediateColumn { index, column } => {
                sbox.intermediates[index] = Var::from_column_for_trace(column);
            }
            Sbox7TraceTestMutation::OutputColumn { column } => {
                sbox.output = Var::from_column_for_trace(column);
            }
            Sbox7TraceTestMutation::SourceRows { rows } => sbox.source_rows = rows,
        }
    }

    /// Corrupt one permutation trace without touching authoritative rows.
    #[doc(hidden)]
    pub fn apply_poseidon_permutation_trace_test_mutation(
        &mut self,
        permutation: usize,
        mutation: PoseidonPermutationTraceTestMutation,
    ) {
        let permutation = &mut self.poseidon_permutations[permutation];
        match mutation {
            PoseidonPermutationTraceTestMutation::InputColumn { lane, column } => {
                permutation.input_columns[lane] = column;
            }
            PoseidonPermutationTraceTestMutation::AllocatedColumns { columns } => {
                permutation.allocated_columns = columns;
            }
            PoseidonPermutationTraceTestMutation::OutputColumn { lane, column } => {
                permutation.output_columns[lane] = column;
            }
            PoseidonPermutationTraceTestMutation::SourceRows { rows } => permutation.source_rows = rows,
        }
    }

    /// Corrupt one sponge trace without touching authoritative source rows.
    #[doc(hidden)]
    pub fn apply_poseidon_hash_trace_test_mutation(&mut self, hash: usize, mutation: PoseidonHashTraceTestMutation) {
        let hash = &mut self.poseidon_hashes[hash];
        match mutation {
            PoseidonHashTraceTestMutation::InputLen { input_len } => hash.input_len = input_len,
            PoseidonHashTraceTestMutation::InputColumn { offset, column } => {
                hash.input_columns[offset] = column;
            }
            PoseidonHashTraceTestMutation::ZeroColumn { column } => hash.zero_column = column,
            PoseidonHashTraceTestMutation::OutputColumn { lane, column } => {
                hash.output_columns[lane] = column;
            }
            PoseidonHashTraceTestMutation::PermutationRange { range } => hash.permutation_range = range,
            PoseidonHashTraceTestMutation::SourceRows { rows } => hash.source_rows = rows,
        }
    }

    /// Duplicate one S-box provenance entry without duplicating source rows.
    #[doc(hidden)]
    pub fn duplicate_sbox7_trace_for_test(&mut self, sbox: usize) {
        self.sbox7.insert(sbox, self.sbox7[sbox].clone());
    }

    /// Remove one S-box provenance entry without removing source rows.
    #[doc(hidden)]
    pub fn remove_sbox7_trace_for_test(&mut self, sbox: usize) {
        self.sbox7.remove(sbox);
    }

    /// Duplicate one permutation provenance entry without duplicating rows.
    #[doc(hidden)]
    pub fn duplicate_poseidon_permutation_trace_for_test(&mut self, permutation: usize) {
        self.poseidon_permutations
            .insert(permutation, self.poseidon_permutations[permutation].clone());
    }

    /// Remove one permutation provenance entry without removing source rows.
    #[doc(hidden)]
    pub fn remove_poseidon_permutation_trace_for_test(&mut self, permutation: usize) {
        self.poseidon_permutations.remove(permutation);
    }

    /// Reorder two permutation provenance entries without reordering rows.
    #[doc(hidden)]
    pub fn swap_poseidon_permutation_traces_for_test(&mut self, left: usize, right: usize) {
        self.poseidon_permutations.swap(left, right);
    }

    /// Duplicate provenance without duplicating source rows.
    #[doc(hidden)]
    pub fn duplicate_canonical_u64_trace_for_test(&mut self, decomposition: usize) {
        self.canonical_u64_decompositions
            .push(self.canonical_u64_decompositions[decomposition].clone());
    }

    /// Duplicate one opening trace without touching authoritative rows.
    #[doc(hidden)]
    pub fn duplicate_balanced_ternary_trace_for_test(&mut self, opening: usize) {
        self.balanced_ternary_openings
            .push(self.balanced_ternary_openings[opening].clone());
    }

    pub(crate) fn push_sbox7(&mut self, entry: Sbox7TraceEntry) {
        self.sbox7.push(entry);
    }

    pub(crate) fn push_k_mul(&mut self, entry: KMulTraceEntry) {
        self.k_muls.push(entry);
    }

    pub(crate) fn push_poseidon_permutation(&mut self, entry: PoseidonPermutationTraceEntry) {
        self.poseidon_permutations.push(entry);
    }

    pub(crate) fn push_poseidon_hash(&mut self, entry: PoseidonHashTraceEntry) {
        self.poseidon_hashes.push(entry);
    }

    pub(crate) fn push_ring_mul_toom3(&mut self, entry: RingMulToom3TraceEntry) {
        self.ring_muls_toom3.push(entry);
    }

    pub(crate) fn push_first_accepted_selection(&mut self, entry: FirstAcceptedSelectionTraceEntry) {
        self.first_accepted_selections.push(entry);
    }

    pub(crate) fn push_canonical_u64_decomposition(&mut self, entry: CanonicalU64TraceEntry) {
        self.canonical_u64_decompositions.push(entry);
    }

    pub(crate) fn push_acceptance_chunk(&mut self, entry: AcceptanceTraceEntry) {
        self.acceptance_chunks.push(entry);
    }

    pub(crate) fn push_mod5_chunk(&mut self, entry: Mod5TraceEntry) {
        self.mod5_chunks.push(entry);
    }

    pub(crate) fn push_balanced_ternary_opening(&mut self, entry: BalancedTernaryOpeningTraceEntry) {
        self.balanced_ternary_openings.push(entry);
    }

    pub(crate) fn push_product_sum_batch(&mut self, entry: ProductSumBatchTrace) {
        self.product_sum_batches.push(entry);
    }

    pub(crate) fn push_polynomial_evaluation(&mut self, entry: PolynomialEvaluationTraceEntry) {
        self.polynomial_evaluations.push(entry);
    }

    pub(crate) fn push_projection_identity(&mut self, entry: ProjectionIdentityTraceEntry) {
        self.projection_identities.push(entry);
    }

    pub(crate) fn assign_projection_identity_roles(&mut self, first: usize, roles: &[ProjectionIdentityRole]) {
        let identities = self
            .projection_identities
            .get_mut(first..)
            .expect("projection identity trace role start exceeds count");
        assert_eq!(identities.len(), roles.len(), "projection identity trace role census");
        for (identity, role) in identities.iter_mut().zip(roles) {
            identity.role = *role;
        }
    }

    pub(crate) fn push_stage(&mut self, checkpoint: R1csStageCheckpoint) {
        self.stages.push(checkpoint);
    }
}
