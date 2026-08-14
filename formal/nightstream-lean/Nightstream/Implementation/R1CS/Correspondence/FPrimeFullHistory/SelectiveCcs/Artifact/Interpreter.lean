import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Schema
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.RelationProfile
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

/-!
Contract: total semantic interpreter from a validated compact selective-CCS
bundle to one typed selective relation with exactly thirteen finite numeric
matrices and the independently fixed gate polynomial.

Owns: direct column-range decoding of raw CSC arrays, original seeded-Phi81
coefficients, the optional blockwise Phi81 bar transform, geometric row runs,
proof-directed indexing of the thirteen matrix ports, and decoding into the
independently owned finite relation surface.

Does not own: any concrete artifact value, Rust byte-decoder conformance, Rust
sampler conformance, generated-artifact drift, CCS satisfaction, equality with
an emitted production relation, or row removal.

Emits constraints: no.

| Stage path | Mathematical interpretation | Assurance tier |
|---|---|---|
| `f_prime.selective_ccs.artifact.csc.decode` | walk one validated raw `colPtr` range and select its `rowIdx`/`vals` cell | model-level artifact schema |
| `f_prime.selective_ccs.artifact.seeded_phi81.decode` | expand verifier-visible seeds through the independent sampler | model-level artifact schema |
| `f_prime.selective_ccs.artifact.geometric.decode` | `initial * ratio^(column-columnStart)` | model-level artifact schema |
| `f_prime.selective_ccs.artifact.bundle.decode` | all named role matrices plus definitionally fixed `gatePolynomial` | model-level artifact schema |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Schema
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

private def contributionSum {Item : Type}
    (items : List Item) (contribution : Item → F) : F :=
  items.foldl (fun total item => total + contribution item) 0

/-- Canonical field interpretation of an artifact coefficient word. Reduction
makes the raw interpreter total; `SeededBlock.Valid.samplerValid` proves the
successful sampler path is already canonical. -/
def fieldOfNat (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

/-- Interpret one matrix cell directly from the serialized CSC arrays. The
validated caller guarantees that both pointers exist, delimit a range inside
the parallel arrays, and contain strictly increasing in-range row indices.
Defaults make this helper total without creating a second semantic-entry
representation. -/
def CscPayload.valueAt
    {rows columns : Nat} (payload : CscPayload)
    (row : Fin rows) (column : Fin columns) : F :=
  let start := payload.pointerAt column.val
  let stop := payload.pointerAt (column.val + 1)
  contributionSum (List.range (stop - start)) fun offset =>
    let entry := start + offset
    if payload.rowIdx.getD entry rows = row.val then
      payload.vals.getD entry 0
    else
      0

def GeometricRowRun.valueAt
    {rows columns : Nat} (run : GeometricRowRun)
    (row : Fin rows) (column : Fin columns) : F :=
  if run.row = row.val ∧ run.columnStart ≤ column.val ∧
      column.val < run.columnStart + run.length then
    run.initial * run.ratio ^ (column.val - run.columnStart)
  else
    0

/-- Original, pre-bar terms contributed by one seeded block at a raw numeric
row. Rows outside the block decode to the empty list. -/
def SeededBlock.originalTermsAt (block : SeededBlock) (row : Nat) :
    List (Nat × Nat) :=
  if block.rowStart ≤ row ∧
      row < block.rowStart + ringDegree * block.kappa then
    let localRow := row - block.rowStart
    block.samplerBlock.terms
      (localRow / ringDegree) (localRow % ringDegree)
  else
    []

/-- Sum all duplicate input-word occurrences at one original matrix entry. -/
def SeededBlock.originalValueAtNat
    (block : SeededBlock) (row column : Nat) : F :=
  contributionSum (SeededBlock.originalTermsAt block row) fun term =>
    if term.1 = column then fieldOfNat term.2 else 0

/-- Apply the exact independent Phi81 bar matrix within the target column's
54-coordinate block. This follows the same output-row/input-column ordering
as `superneo_bar_matrix`; equality with that Rust matrix is a separate
existing refinement boundary. -/
def SeededBlock.transformedValueAtNat
    (block : SeededBlock) (row column : Nat) : F :=
  let columnBlock := column / ringDegree
  let output : Fin ringDegree :=
    ⟨column % ringDegree, Nat.mod_lt _ (by decide)⟩
  contributionSum (List.range ringDegree) fun input =>
    if inputLt : input < ringDegree then
      SeededBlock.originalValueAtNat block row
          (columnBlock * ringDegree + input) *
        nativeBarEntry output ⟨input, inputLt⟩
    else
      0

def SeededBlock.valueAt
    {rows columns : Nat} (block : SeededBlock)
    (row : Fin rows) (column : Fin columns) : F :=
  if block.transformedColumns then
    SeededBlock.transformedValueAtNat block row.val column.val
  else
    SeededBlock.originalValueAtNat block row.val column.val

/-- Add all three compact sources. Overlap is intentional and therefore never
resolved by list priority. -/
def CompactMatrix.valueAt
    {rows columns : Nat} (matrix : CompactMatrix)
    (row : Fin rows) (column : Fin columns) : F :=
  CscPayload.valueAt matrix.csc row column +
    contributionSum matrix.seededBlocks
      (fun block => SeededBlock.valueAt block row column) +
    contributionSum matrix.geometricRuns
      (fun run => GeometricRowRun.valueAt run row column)

/-- Select one of the exactly thirteen validated compact matrices. -/
def ValidatedBundle.matrixAt
    (artifact : ValidatedBundle) (port : Fin 13) : CompactMatrix :=
  artifact.raw.matrices.get ⟨port.val, by
    rw [artifact.valid.matrixCount]
    exact port.isLt⟩

/-- Internal typed matrix family consumed by the independent arity-13
polynomial. Artifact positions are converted through the exhaustive semantic
role/index map. Public consumers use `interpretRelation`, which does not permit
a caller-selected polynomial. -/
private def ValidatedBundle.interpretMatrices (artifact : ValidatedBundle) :
    Role → RowPadding.NumericMatrix F artifact.raw.rows artifact.raw.columns :=
  fun role row column =>
    CompactMatrix.valueAt
      (ValidatedBundle.matrixAt artifact role.index) row column

/-- Artifact-facing name for the independent finite selective relation. -/
abbrev InterpretedRelation :=
  Nightstream.Implementation.R1CS.SelectiveCcs.RelationProfile.FiniteRelation

/-- The only polynomial available for an interpreted selective relation. This
reduces definitionally to the independently specified gate polynomial. -/
def InterpretedRelation.constraintPolynomial
    {rows columns : Nat} (_relation : InterpretedRelation rows columns) :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.ConstraintPolynomial
      F Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports.portCount :=
  gatePolynomial

/-- Decode the validated raw arrays and compact generators into the typed
selective relation. This remains model-level until a generated Rust artifact
and byte decoder are separately refined to `Schema.Bundle`. -/
def ValidatedBundle.interpretRelation (artifact : ValidatedBundle) :
    InterpretedRelation artifact.raw.rows artifact.raw.columns where
  matrices := ValidatedBundle.interpretMatrices artifact

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
