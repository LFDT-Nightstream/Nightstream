import Nightstream.Implementation.R1CS.Core.SeededPhi81
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericRowPadding

/-!
Contract: handwritten data schema for one compact selective-CCS matrix
bundle before it is interpreted as finite numeric matrices.

Owns: raw CSC arrays, seeded-Phi81 block geometry and coefficient-source
metadata, geometric row runs, enclosing dimensions, and the exact thirteen
matrix ports consumed by the independent selective polynomial.

Does not own: generated production values, Rust-to-Lean serialization
equality, equality with any emitted relation, row-family ownership,
constraint counts, or permission to remove rows.

Emits constraints: no.

Authority boundary: `Bundle.Valid` rejects out-of-range indices and the wrong
matrix count. It says only that handwritten or generated data can be decoded
without truncation. A future generated-artifact module must separately prove
that decoding the Rust matrix variant and its Lean-only sampler certificate
produces these components and the same matrix action; this schema does not
claim literal equality with Rust's enum payload.

| Stage path | Compact source | Validated obligation | Interpretation owner |
|---|---|---|---|
| `f_prime.selective_ccs.artifact.csc` | `colPtr`, `rowIdx`, and `vals` | canonical bounded CSC storage | `Artifact.Interpreter` |
| `f_prime.selective_ccs.artifact.seeded_phi81` | seeds plus block geometry | sampler, row span, and word spans are valid | `Artifact.Interpreter` |
| `f_prime.selective_ccs.artifact.geometric` | one contiguous geometric run | nonempty in-range row and column interval | `Artifact.Interpreter` |
| `f_prime.selective_ccs.artifact.bundle` | compact matrices | positive dimensions and exactly thirteen ports | `ValidatedBundle.interpretRelation` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Schema

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.Implementation.R1CS

/-- Exact matrix arity, owned by the independent selective polynomial rather
than duplicated by the artifact schema. -/
def portCount : Nat :=
  Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports.portCount

/-- Type-level binding from the artifact port count to the independently
specified gate polynomial. No artifact may substitute another polynomial. -/
abbrev gatePolynomial : ConstraintPolynomial F portCount :=
  Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.polynomial

/-- Raw arrays of Rust's compressed-sparse-column payload. No semantic
triplet list is accepted at this boundary. -/
structure CscPayload where
  colPtr : List Nat
  rowIdx : List Nat
  vals : List F
deriving DecidableEq, Repr

def CscPayload.nnz (payload : CscPayload) : Nat := payload.vals.length

def CscPayload.pointerAt (payload : CscPayload) (index : Nat) : Nat :=
  payload.colPtr.getD index 0

/-- Cardinality of Rust's `u32` index carrier. Raw CSC indices must remain
strictly below this bound even though the model uses `Nat` to make malformed
wire values representable and rejectable. -/
def cscU32Cardinality : Nat := 2 ^ 32

/-- The raw row-index slice owned by one CSC column. This is a validation
helper over the serialized arrays, not an alternative semantic-entry payload. -/
private def CscPayload.columnRows
    (payload : CscPayload) (column : Nat) : List Nat :=
  let start := payload.pointerAt column
  let stop := payload.pointerAt (column + 1)
  (payload.rowIdx.drop start).take (stop - start)

/-- Fail-closed canonical CSC conditions. Monotone pointers plus the zero
start and terminal `nnz` pointer also force every column range inside both
parallel entry arrays. -/
def CscPayload.Valid
    (payload : CscPayload) (rows columns : Nat) : Prop :=
  payload.colPtr.length = columns + 1 ∧
  payload.colPtr.head? = some 0 ∧
  payload.colPtr.all
      (fun pointer => decide (pointer < cscU32Cardinality)) = true ∧
  payload.colPtr.Pairwise (fun left right => left ≤ right) ∧
  payload.pointerAt columns = payload.nnz ∧
  payload.rowIdx.length = payload.vals.length ∧
  payload.rowIdx.all
      (fun row => decide (row < cscU32Cardinality)) = true ∧
  payload.rowIdx.all (fun row => decide (row < rows)) = true ∧
  (∀ column : Fin columns,
    (payload.columnRows column.val).Pairwise
      (fun left right => left < right)) ∧
  payload.vals.all (fun value => decide (value ≠ 0)) = true

instance (payload : CscPayload) (rows columns : Nat) :
    Decidable (payload.Valid rows columns) := by
  unfold CscPayload.Valid CscPayload.columnRows
  infer_instance

/-- Compact source for one seeded Phi81 linear block. `schedule.rejectionFuel`
is an interpreter certificate bound, not a field in Rust's compact block. A
later sampler refinement must prove that successful bounded expansion equals
the unbounded first-accepted coefficient stream used by production. -/
structure SeededBlock where
  rowStart : Nat
  wordStarts : List Nat
  wordWidth : Nat
  kappa : Nat
  messageColumns : Nat
  schedule : SeededPhi81.SeedSchedule
  transformedColumns : Bool
deriving DecidableEq, Repr

/-- Adapter into the existing independent seeded-Phi81 sampler semantics.
The output identifiers are derived row names and are not artifact data. The
adapter is deliberately untransformed; the optional bar transform is applied
by the matrix interpreter after original coefficients are expanded. -/
def SeededBlock.samplerBlock (block : SeededBlock) : SeededPhi81.Block where
  rowStart := block.rowStart
  wordStarts := block.wordStarts
  wordWidth := block.wordWidth
  kappa := block.kappa
  messageCols := block.messageColumns
  outputColumns :=
    (List.range (ringDegree * block.kappa)).map
      (fun offset => block.rowStart + offset)
  superneoTransformedColumns := false
  schedule := block.schedule

structure SeededBlock.Valid
    (rows columns : Nat) (block : SeededBlock) : Prop where
  inputNonempty : block.wordStarts ≠ []
  samplerValid : block.samplerBlock.Valid
  rowsFit : block.rowStart + ringDegree * block.kappa ≤ rows
  wordsFit : ∀ start ∈ block.wordStarts,
    start + block.wordWidth ≤ columns
  transformedWidth : block.transformedColumns = true →
    columns % ringDegree = 0

instance (rows columns : Nat) (block : SeededBlock) :
    Decidable (block.Valid rows columns) := by
  apply decidable_of_iff
    (block.wordStarts ≠ [] ∧
      block.samplerBlock.Valid ∧
      block.rowStart + ringDegree * block.kappa ≤ rows ∧
      (∀ start ∈ block.wordStarts,
        start + block.wordWidth ≤ columns) ∧
      (block.transformedColumns = true → columns % ringDegree = 0))
  constructor
  · intro fields
    exact
      { inputNonempty := fields.1
        samplerValid := fields.2.1
        rowsFit := fields.2.2.1
        wordsFit := fields.2.2.2.1
        transformedWidth := fields.2.2.2.2 }
  · intro valid
    exact ⟨valid.inputNonempty, valid.samplerValid, valid.rowsFit,
      valid.wordsFit, valid.transformedWidth⟩

/-- Exact metadata of Rust's `GeometricRowRun`: the row coefficients are
`initial * ratio^offset` over the declared half-open column interval. -/
structure GeometricRowRun where
  row : Nat
  columnStart : Nat
  length : Nat
  initial : F
  ratio : F
deriving DecidableEq, Repr

structure GeometricRowRun.Valid
    (rows columns : Nat) (run : GeometricRowRun) : Prop where
  nonempty : 0 < run.length
  rowInRange : run.row < rows
  columnsFit : run.columnStart + run.length ≤ columns

instance (rows columns : Nat) (run : GeometricRowRun) :
    Decidable (run.Valid rows columns) := by
  apply decidable_of_iff
    (0 < run.length ∧ run.row < rows ∧
      run.columnStart + run.length ≤ columns)
  constructor
  · intro fields
    exact
      { nonempty := fields.1
        rowInRange := fields.2.1
        columnsFit := fields.2.2 }
  · intro valid
    exact ⟨valid.nonempty, valid.rowInRange, valid.columnsFit⟩

/-- The three additive components of `CscWithSeededPhi81`. Contributions may
overlap; interpretation sums them just as matrix evaluation does. -/
structure CompactMatrix where
  csc : CscPayload
  seededBlocks : List SeededBlock
  geometricRuns : List GeometricRowRun
deriving DecidableEq, Repr

structure CompactMatrix.Valid
    (rows columns : Nat) (matrix : CompactMatrix) : Prop where
  cscValid : matrix.csc.Valid rows columns
  seededBlocksValid : ∀ block ∈ matrix.seededBlocks,
    block.Valid rows columns
  geometricRunsValid : ∀ run ∈ matrix.geometricRuns,
    run.Valid rows columns

instance (rows columns : Nat) (matrix : CompactMatrix) :
    Decidable (matrix.Valid rows columns) := by
  apply decidable_of_iff
    (matrix.csc.Valid rows columns ∧
      (∀ block ∈ matrix.seededBlocks, block.Valid rows columns) ∧
      (∀ run ∈ matrix.geometricRuns, run.Valid rows columns))
  constructor
  · intro fields
    exact
      { cscValid := fields.1
        seededBlocksValid := fields.2.1
        geometricRunsValid := fields.2.2 }
  · intro valid
    exact ⟨valid.cscValid, valid.seededBlocksValid,
      valid.geometricRunsValid⟩

def SeededBlock.rowEnd (block : SeededBlock) : Nat :=
  block.rowStart + ringDegree * block.kappa

/-- Compact seeded blocks may share inputs but must not own the same matrix
row in a production-shaped bundle. -/
def SeededBlock.RowsDisjoint (left right : SeededBlock) : Prop :=
  left.rowEnd ≤ right.rowStart ∨ right.rowEnd ≤ left.rowStart

instance (left right : SeededBlock) :
    Decidable (SeededBlock.RowsDisjoint left right) := by
  unfold SeededBlock.RowsDisjoint
  infer_instance

/-- Nondecreasing lexicographic order used by Rust's geometric-run sort key
`(row, columnStart, length)`. Equal keys remain permitted. -/
def GeometricRowRun.CanonicalBefore
    (left right : GeometricRowRun) : Prop :=
  left.row < right.row ∨
  (left.row = right.row ∧
    (left.columnStart < right.columnStart ∨
      (left.columnStart = right.columnStart ∧ left.length ≤ right.length)))

instance (left right : GeometricRowRun) :
    Decidable (GeometricRowRun.CanonicalBefore left right) := by
  unfold GeometricRowRun.CanonicalBefore
  infer_instance

/-- Stronger production-shape conditions are kept separate from semantic
decodability. They still make no claim that Rust emitted the data. -/
structure CompactMatrix.ProductionValid
    (rows columns : Nat) (matrix : CompactMatrix) : Prop where
  valid : matrix.Valid rows columns
  seededRowsDisjoint :
    matrix.seededBlocks.Pairwise SeededBlock.RowsDisjoint
  geometricRunsCanonical :
    matrix.geometricRuns.Pairwise GeometricRowRun.CanonicalBefore

instance (rows columns : Nat) (matrix : CompactMatrix) :
    Decidable (matrix.ProductionValid rows columns) := by
  apply decidable_of_iff
    (matrix.Valid rows columns ∧
      matrix.seededBlocks.Pairwise SeededBlock.RowsDisjoint ∧
      matrix.geometricRuns.Pairwise GeometricRowRun.CanonicalBefore)
  constructor
  · intro fields
    exact
      { valid := fields.1
        seededRowsDisjoint := fields.2.1
        geometricRunsCanonical := fields.2.2 }
  · intro valid
    exact ⟨valid.valid, valid.seededRowsDisjoint,
      valid.geometricRunsCanonical⟩

/-- Untyped artifact envelope. Keeping dimensions and the matrix list as data
allows a generated module to expose malformed values for rejection rather
than hiding bounds in `Fin` constructors. -/
structure Bundle where
  rows : Nat
  columns : Nat
  matrices : List CompactMatrix
deriving DecidableEq, Repr

structure Bundle.Valid (bundle : Bundle) : Prop where
  rowsPositive : 0 < bundle.rows
  columnsPositive : 0 < bundle.columns
  matrixCount : bundle.matrices.length = portCount
  matricesValid : ∀ matrix ∈ bundle.matrices,
    matrix.Valid bundle.rows bundle.columns

instance (bundle : Bundle) : Decidable bundle.Valid := by
  apply decidable_of_iff
    (0 < bundle.rows ∧ 0 < bundle.columns ∧
      bundle.matrices.length = portCount ∧
      (∀ matrix ∈ bundle.matrices,
        matrix.Valid bundle.rows bundle.columns))
  constructor
  · intro fields
    exact
      { rowsPositive := fields.1
        columnsPositive := fields.2.1
        matrixCount := fields.2.2.1
        matricesValid := fields.2.2.2 }
  · intro valid
    exact ⟨valid.rowsPositive, valid.columnsPositive, valid.matrixCount,
      valid.matricesValid⟩

/-- Production-shape supplement for every compact matrix. This remains
model-level until a generated artifact is bound to the Rust compiler output. -/
structure Bundle.ProductionValid (bundle : Bundle) : Prop where
  valid : bundle.Valid
  matricesProductionValid : ∀ matrix ∈ bundle.matrices,
    matrix.ProductionValid bundle.rows bundle.columns

instance (bundle : Bundle) : Decidable bundle.ProductionValid := by
  apply decidable_of_iff
    (bundle.Valid ∧ ∀ matrix ∈ bundle.matrices,
      matrix.ProductionValid bundle.rows bundle.columns)
  constructor
  · intro fields
    exact
      { valid := fields.1
        matricesProductionValid := fields.2 }
  · intro valid
    exact ⟨valid.valid, valid.matricesProductionValid⟩

/-- Proof-carrying boundary required by the total typed interpreter. This is
not a claim that `raw` came from Rust. -/
structure ValidatedBundle where
  raw : Bundle
  valid : raw.Valid

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Schema
