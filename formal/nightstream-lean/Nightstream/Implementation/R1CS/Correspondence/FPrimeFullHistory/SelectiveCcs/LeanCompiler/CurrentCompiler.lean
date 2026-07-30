import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.Ownership
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.Profile

/-!
Contract: assemble the current Lean-owned selective-CCS compiler gate for one
source-aligned, receipt-conserved physical program.

Assurance tier: model-level.

Owns: exact compiled/source acceptance in both directions, transport from
equal physical columns and rows to equal compiled systems, the current receipt
obligation tree, exact manifest round trip and cost, canonical manifest
coefficients, and the least Boolean row-domain profile derived from the emitted
lists.

Does not own: application-specific public-value canonicality, a selected
F-prime application, construction of the full NIFS verifier encoding, a
closed deployment manifest, Rust equality, or protocol security events.
Those items must be supplied by the concrete deployment before the complete
M4 milestone is closed.

Emits constraints: no new rows.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentCompiler

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks

universe u

/-- The current selective relation uses only dimensions derived from the
Lean-emitted physical encoding. -/
def Accepts
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (publicWidth : 270 ≤ encoding.columnIds.length)
    (assignment : Fin encoding.columnIds.length → F) :
    Prop :=
  EncodingRows.IndexedAccepts encoding
    (Profile.ofEncoding encoding publicWidth) assignment

/-- The exact thirteen-matrix selective system compiled from this encoding.

This value is the physical relation that `Accepts` checks.  It is separate
from any recursive NIFS setup system until a deployment proves that both
systems are the same. -/
def compiledSystem
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (publicWidth : 270 ≤ encoding.columnIds.length) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.Structure
      (RelationProfile.Profile.shape
        (Profile.ofEncoding encoding publicWidth)) :=
  (DirectRows.relation
      (EncodingRows.columnIndex encoding encoding.one)
      (EncodingRows.program encoding)).toStructure
    (Profile.ofEncoding encoding publicWidth)

@[simp] theorem compiledSystem_constraintPolynomial
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (publicWidth : 270 ≤ encoding.columnIds.length) :
    (compiledSystem encoding publicWidth).constraintPolynomial =
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.polynomial :=
  rfl

/-- Compilation respects equality of complete encodings.

This is the transport rule used by the recursive fixed-point constructor:
an application compiler proves that changing only the seed matrices does not
change its physical Step encoding, and this lemma transports that fact to the
compiled relation. -/
theorem compiledSystem_congr
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    {leftEncoding rightEncoding : Encoding source}
    (leftPublic : 270 ≤ leftEncoding.columnIds.length)
    (rightPublic : 270 ≤ rightEncoding.columnIds.length)
    (same : leftEncoding = rightEncoding) :
    HEq (compiledSystem leftEncoding leftPublic)
      (compiledSystem rightEncoding rightPublic) := by
  cases same
  rfl

private def physicalColumnIndex
    (columns : List ColumnId)
    (one : ColumnId)
    (oneAllocated : one ∈ columns) :
    ColumnId → Fin columns.length :=
  fun column =>
    if member : column ∈ columns then
      (EncodingRows.locate columns column member).index
    else
      (EncodingRows.locate columns one oneAllocated).index

private theorem physicalColumnIndex_heq
    {leftColumns rightColumns : List ColumnId}
    {leftOne rightOne : ColumnId}
    (leftAllocated : leftOne ∈ leftColumns)
    (rightAllocated : rightOne ∈ rightColumns)
    (columnsEqual : leftColumns = rightColumns)
    (oneEqual : leftOne = rightOne) :
    HEq
      (physicalColumnIndex leftColumns leftOne leftAllocated)
      (physicalColumnIndex rightColumns rightOne rightAllocated) := by
  cases columnsEqual
  cases oneEqual
  rfl

private theorem physicalColumnIndex_at_heq
    {leftColumns rightColumns : List ColumnId}
    {leftOne rightOne leftColumn rightColumn : ColumnId}
    (leftAllocated : leftOne ∈ leftColumns)
    (rightAllocated : rightOne ∈ rightColumns)
    (columnsEqual : leftColumns = rightColumns)
    (oneEqual : leftOne = rightOne)
    (columnEqual : leftColumn = rightColumn) :
    HEq
      (physicalColumnIndex leftColumns leftOne leftAllocated leftColumn)
      (physicalColumnIndex rightColumns rightOne rightAllocated rightColumn) := by
  cases columnsEqual
  cases oneEqual
  cases columnEqual
  rfl

private theorem columnIndex_heq_of_physical_eq
    {leftSignature rightSignature : Signature.{u}}
    {leftInput leftOutput : Schema leftSignature.types}
    {rightInput rightOutput : Schema rightSignature.types}
    {leftSource : Program leftSignature leftInput leftOutput}
    {rightSource : Program rightSignature rightInput rightOutput}
    (leftEncoding : Encoding leftSource)
    (rightEncoding : Encoding rightSource)
    (columnsEqual :
      leftEncoding.columnIds = rightEncoding.columnIds)
    (oneEqual :
      leftEncoding.one = rightEncoding.one) :
    HEq
      (EncodingRows.columnIndex leftEncoding)
      (EncodingRows.columnIndex rightEncoding) := by
  change
    HEq
      (physicalColumnIndex leftEncoding.columnIds leftEncoding.one
        leftEncoding.oneAllocated)
      (physicalColumnIndex rightEncoding.columnIds rightEncoding.one
        rightEncoding.oneAllocated)
  exact
    physicalColumnIndex_heq
      leftEncoding.oneAllocated rightEncoding.oneAllocated
      columnsEqual oneEqual

private theorem oneIndex_heq_of_physical_eq
    {leftSignature rightSignature : Signature.{u}}
    {leftInput leftOutput : Schema leftSignature.types}
    {rightInput rightOutput : Schema rightSignature.types}
    {leftSource : Program leftSignature leftInput leftOutput}
    {rightSource : Program rightSignature rightInput rightOutput}
    (leftEncoding : Encoding leftSource)
    (rightEncoding : Encoding rightSource)
    (columnsEqual :
      leftEncoding.columnIds = rightEncoding.columnIds)
    (oneEqual :
      leftEncoding.one = rightEncoding.one) :
    HEq
      (EncodingRows.columnIndex leftEncoding leftEncoding.one)
      (EncodingRows.columnIndex rightEncoding rightEncoding.one) := by
  change
    HEq
      (physicalColumnIndex leftEncoding.columnIds leftEncoding.one
        leftEncoding.oneAllocated leftEncoding.one)
      (physicalColumnIndex rightEncoding.columnIds rightEncoding.one
        rightEncoding.oneAllocated rightEncoding.one)
  exact
    physicalColumnIndex_at_heq
      leftEncoding.oneAllocated rightEncoding.oneAllocated
      columnsEqual oneEqual oneEqual

private theorem physicalProgram_heq
    {leftCount rightCount : Nat}
    (leftIndex : ColumnId → Fin leftCount)
    (rightIndex : ColumnId → Fin rightCount)
    (leftRows rightRows : List OwnedRow)
    (countEqual : leftCount = rightCount)
    (indexEqual : HEq leftIndex rightIndex)
    (rowsEqual : leftRows = rightRows) :
    HEq
      (StableRows.program leftIndex leftRows)
      (StableRows.program rightIndex rightRows) := by
  cases countEqual
  have indexEq : leftIndex = rightIndex :=
    eq_of_heq indexEqual
  cases indexEq
  cases rowsEqual
  rfl

private theorem program_heq_of_physical_eq
    {leftSignature rightSignature : Signature.{u}}
    {leftInput leftOutput : Schema leftSignature.types}
    {rightInput rightOutput : Schema rightSignature.types}
    {leftSource : Program leftSignature leftInput leftOutput}
    {rightSource : Program rightSignature rightInput rightOutput}
    (leftEncoding : Encoding leftSource)
    (rightEncoding : Encoding rightSource)
    (columnsEqual :
      leftEncoding.columnIds = rightEncoding.columnIds)
    (rowsEqual :
      leftEncoding.rows = rightEncoding.rows)
    (oneEqual :
      leftEncoding.one = rightEncoding.one) :
    HEq
      (EncodingRows.program leftEncoding)
      (EncodingRows.program rightEncoding) := by
  unfold EncodingRows.program
  exact
    physicalProgram_heq
      (EncodingRows.columnIndex leftEncoding)
      (EncodingRows.columnIndex rightEncoding)
      leftEncoding.rows rightEncoding.rows
      (congrArg List.length columnsEqual)
      (columnIndex_heq_of_physical_eq leftEncoding rightEncoding
        columnsEqual oneEqual)
      rowsEqual

private def physicalProfile
    (rows columns : Nat)
    (publicWidth : 270 ≤ columns) :
    RelationProfile.Profile rows columns where
  rowVariables := Profile.rowVariables rows
  rowDomain := Profile.exactRowDomain rows
  publicFits := Profile.publicFits_of_alignedWidth publicWidth

private def physicalCompiledSystem
    {columns : Nat}
    (one : Fin columns)
    (program : List (DirectRows.SourceRow columns))
    (publicWidth : 270 ≤ columns) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.Structure
      (RelationProfile.Profile.shape
        (physicalProfile program.length columns publicWidth)) :=
  (DirectRows.relation one program).toStructure
    (physicalProfile program.length columns publicWidth)

private theorem physicalCompiledSystem_heq
    {leftCount rightCount : Nat}
    (leftOne : Fin leftCount)
    (rightOne : Fin rightCount)
    (leftProgram : List (DirectRows.SourceRow leftCount))
    (rightProgram : List (DirectRows.SourceRow rightCount))
    (leftPublic : 270 ≤ leftCount)
    (rightPublic : 270 ≤ rightCount)
    (countEqual : leftCount = rightCount)
    (oneEqual : HEq leftOne rightOne)
    (programEqual : HEq leftProgram rightProgram) :
    HEq
      (physicalCompiledSystem leftOne leftProgram leftPublic)
      (physicalCompiledSystem rightOne rightProgram rightPublic) := by
  cases countEqual
  have oneEq : leftOne = rightOne :=
    eq_of_heq oneEqual
  cases oneEq
  have programEq : leftProgram = rightProgram :=
    eq_of_heq programEqual
  cases programEq
  rfl

/-- Equal physical column identities, rows, and constant-one columns compile
to the same selective system. Semantic proof fields and matrix payloads are
not part of this transport boundary. -/
theorem compiledSystem_heq_of_physical_eq
    {leftSignature rightSignature : Signature.{u}}
    {leftInput leftOutput : Schema leftSignature.types}
    {rightInput rightOutput : Schema rightSignature.types}
    {leftSource : Program leftSignature leftInput leftOutput}
    {rightSource : Program rightSignature rightInput rightOutput}
    (leftEncoding : Encoding leftSource)
    (rightEncoding : Encoding rightSource)
    (leftPublic : 270 ≤ leftEncoding.columnIds.length)
    (rightPublic : 270 ≤ rightEncoding.columnIds.length)
    (columnsEqual :
      leftEncoding.columnIds = rightEncoding.columnIds)
    (rowsEqual :
      leftEncoding.rows = rightEncoding.rows)
    (oneEqual :
      leftEncoding.one = rightEncoding.one) :
    HEq
      (compiledSystem leftEncoding leftPublic)
      (compiledSystem rightEncoding rightPublic) := by
  change
    HEq
      (physicalCompiledSystem
        (EncodingRows.columnIndex leftEncoding leftEncoding.one)
        (EncodingRows.program leftEncoding) leftPublic)
      (physicalCompiledSystem
        (EncodingRows.columnIndex rightEncoding rightEncoding.one)
        (EncodingRows.program rightEncoding) rightPublic)
  exact
    physicalCompiledSystem_heq
      (EncodingRows.columnIndex leftEncoding leftEncoding.one)
      (EncodingRows.columnIndex rightEncoding rightEncoding.one)
      (EncodingRows.program leftEncoding)
      (EncodingRows.program rightEncoding)
      leftPublic rightPublic
      (congrArg List.length columnsEqual)
      (oneIndex_heq_of_physical_eq leftEncoding rightEncoding
        columnsEqual oneEqual)
      (program_heq_of_physical_eq leftEncoding rightEncoding
        columnsEqual rowsEqual oneEqual)

/-- Nondependent physical compiler input. Proof records and semantic source
indices are excluded. -/
structure PhysicalEncoding where
  columnIds : List ColumnId
  rows : List OwnedRow
  one : ColumnId

namespace PhysicalEncoding

/-- Project one typed encoding to the exact data consumed by the physical
compiler. -/
def ofEncoding
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    PhysicalEncoding where
  columnIds := encoding.columnIds
  rows := encoding.rows
  one := encoding.one

@[ext] theorem ext
    {left right : PhysicalEncoding}
    (columnIds : left.columnIds = right.columnIds)
    (rows : left.rows = right.rows)
    (one : left.one = right.one) :
    left = right := by
  cases left
  cases right
  simp_all

end PhysicalEncoding

/-- Equality of the nondependent physical projection is sufficient for
compiled-system equality. -/
theorem compiledSystem_heq_of_physicalEncoding_eq
    {leftSignature rightSignature : Signature.{u}}
    {leftInput leftOutput : Schema leftSignature.types}
    {rightInput rightOutput : Schema rightSignature.types}
    {leftSource : Program leftSignature leftInput leftOutput}
    {rightSource : Program rightSignature rightInput rightOutput}
    (leftEncoding : Encoding leftSource)
    (rightEncoding : Encoding rightSource)
    (leftPublic : 270 ≤ leftEncoding.columnIds.length)
    (rightPublic : 270 ≤ rightEncoding.columnIds.length)
    (same :
      PhysicalEncoding.ofEncoding leftEncoding =
        PhysicalEncoding.ofEncoding rightEncoding) :
    HEq
      (compiledSystem leftEncoding leftPublic)
      (compiledSystem rightEncoding rightPublic) :=
  compiledSystem_heq_of_physical_eq
    leftEncoding rightEncoding leftPublic rightPublic
    (congrArg PhysicalEncoding.columnIds same)
    (congrArg PhysicalEncoding.rows same)
    (congrArg PhysicalEncoding.one same)

/-- Current CIR-SOUND and same-assignment completeness. -/
theorem accepts_iff_physicalSatisfies
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (publicWidth : 270 ≤ encoding.columnIds.length)
    (assignment : Fin encoding.columnIds.length → F) :
    Accepts encoding publicWidth assignment ↔
      encoding.PhysicalSatisfies
        (StableRows.pulledAssignment
          (EncodingRows.columnIndex encoding) assignment) :=
  EncodingRows.indexedAccepts_iff_physicalSatisfies
    encoding (Profile.ofEncoding encoding publicWidth) assignment

/-- Current CIR-COMPLETE: rebuild the exact finite assignment from any honest
stable assignment, with the reverse implication proved as well. -/
theorem indexedAssignment_accepts_iff
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (publicWidth : 270 ≤ encoding.columnIds.length)
    (assignment : ColumnId → F) :
    Accepts encoding publicWidth
        (EncodingRows.indexedAssignment encoding assignment) ↔
      encoding.PhysicalSatisfies assignment :=
  EncodingRows.indexedAssignment_accepts_iff
    encoding (Profile.ofEncoding encoding publicWidth) assignment

/-- Exact physical meaning of the proof-free canonical manifest. -/
def ManifestPhysicalSatisfies
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (assignment : ColumnId → F) :
    Prop :=
  assignment encoding.one = 1 ∧
    Satisfies
      (CanonicalManifest.Program.ofEncoding encoding).decode.rows
      assignment

theorem manifestPhysicalSatisfies_iff
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (assignment : ColumnId → F) :
    ManifestPhysicalSatisfies encoding assignment ↔
      encoding.PhysicalSatisfies assignment := by
  unfold ManifestPhysicalSatisfies Encoding.PhysicalSatisfies
  rw [CanonicalManifest.Program.decoded_satisfies_iff]

/-- The exact current obligation tree. Instruction receipts are its deepest
leaves; row identity, not row-value equality, selects the leaf. -/
def ObligationTree
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (aligned : SourceAlignment.AlignedReceiptProgram source) :
    Prop :=
  (∀ position : Fin
      (EncodingRows.program aligned.toEncoding).length,
    ∃ receipt,
      receipt ∈ aligned.physical.receipts ∧
        receipt.owner ∈ SourceAlignment.programOwners source ∧
        (Ownership.sourceAt aligned.toEncoding position).id ∈
          receipt.rowIds ∧
        ∀ candidate,
          candidate ∈ aligned.physical.receipts →
          (Ownership.sourceAt aligned.toEncoding position).id ∈
            candidate.rowIds →
          candidate = receipt) ∧
  (∀ receipt,
    receipt ∈ aligned.physical.receipts →
    ∀ owned,
      owned ∈ receipt.rows →
      ∃ position : Fin
        (EncodingRows.program aligned.toEncoding).length,
        Ownership.sourceAt aligned.toEncoding position = owned) ∧
  (EncodingRows.program aligned.toEncoding).length =
    (aligned.physical.receipts.map
      fun receipt => receipt.rows.length).sum ∧
  (EncodingRows.program aligned.toEncoding).length =
    aligned.toEncoding.cost.recurringRows

theorem obligationTree
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (aligned : SourceAlignment.AlignedReceiptProgram source) :
    ObligationTree aligned := by
  refine ⟨
    Ownership.compiledRow_has_exactly_one_source_owner aligned,
    ?_,
    Ownership.compiledRowCount_eq_receiptRows aligned.toEncoding,
    Ownership.compiledRowCount_eq_cost aligned.toEncoding
  ⟩
  intro receipt receiptMember owned ownedMember
  exact Ownership.receiptRow_has_compiled_position aligned.toEncoding
    receipt receiptMember owned ownedMember

/-- Canonical, proof-free representation of the same current program. This
is field-coefficient and ownership canonicality; application public-value
canonicality remains a separate concrete obligation. -/
def ManifestCanonical
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    Prop :=
  (∀ assignment,
    ManifestPhysicalSatisfies encoding assignment ↔
      encoding.PhysicalSatisfies assignment) ∧
  (CanonicalManifest.Program.ofEncoding encoding).cost =
    encoding.cost ∧
  (CanonicalManifest.Program.ofEncoding encoding).columns =
    encoding.columns ∧
  (CanonicalManifest.Program.ofEncoding encoding).rows.length =
    encoding.rows.length ∧
  encoding.columnIds.Nodup ∧
  (∀ row ∈ (CanonicalManifest.Program.ofEncoding encoding).rows,
    (∀ term ∈ row.a, term.coefficient ≠ 0) ∧
    (∀ term ∈ row.b, term.coefficient ≠ 0) ∧
    (∀ term ∈ row.c, term.coefficient ≠ 0)) ∧
  (∀ row ∈ (CanonicalManifest.Program.ofEncoding encoding).rows,
    (∀ term ∈ row.a, term.coefficient < goldilocksModulus) ∧
    (∀ term ∈ row.b, term.coefficient < goldilocksModulus) ∧
    (∀ term ∈ row.c, term.coefficient < goldilocksModulus)) ∧
  (∀ row ∈ (CanonicalManifest.Program.ofEncoding encoding).rows,
    (row.a.map CanonicalManifest.ManifestTerm.column).Nodup ∧
    (row.b.map CanonicalManifest.ManifestTerm.column).Nodup ∧
    (row.c.map CanonicalManifest.ManifestTerm.column).Nodup)

theorem manifestCanonical
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    ManifestCanonical encoding := by
  exact ⟨
    manifestPhysicalSatisfies_iff encoding,
    CanonicalManifest.Program.cost_ofEncoding encoding,
    CanonicalManifest.Program.columns_ofEncoding encoding,
    CanonicalManifest.Program.rows_length_ofEncoding encoding,
    encoding.column_identities_nodup,
    CanonicalManifest.Program.all_coefficients_nonzero encoding,
    CanonicalManifest.Program.all_coefficients_canonical encoding,
    CanonicalManifest.Program.all_combination_columns_nodup encoding
  ⟩

/-- Assembled current compiler evidence. This is intentionally one theorem
past the individual ingredients. It excludes only concrete application
semantics and public-value canonicality. -/
structure Evidence
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (aligned : SourceAlignment.AlignedReceiptProgram source)
    (publicWidth : 270 ≤ aligned.toEncoding.columnIds.length) : Prop where
  soundAndComplete :
    ∀ assignment,
      Accepts aligned.toEncoding publicWidth assignment ↔
        aligned.toEncoding.PhysicalSatisfies
          (StableRows.pulledAssignment
            (EncodingRows.columnIndex aligned.toEncoding) assignment)
  honestReassembly :
    ∀ assignment,
      Accepts aligned.toEncoding publicWidth
          (EncodingRows.indexedAssignment aligned.toEncoding assignment) ↔
        aligned.toEncoding.PhysicalSatisfies assignment
  obligationOwnership : ObligationTree aligned
  canonicalManifest : ManifestCanonical aligned.toEncoding

theorem evidence
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (aligned : SourceAlignment.AlignedReceiptProgram source)
    (publicWidth : 270 ≤ aligned.toEncoding.columnIds.length) :
    Evidence aligned publicWidth where
  soundAndComplete :=
    accepts_iff_physicalSatisfies aligned.toEncoding publicWidth
  honestReassembly :=
    indexedAssignment_accepts_iff aligned.toEncoding publicWidth
  obligationOwnership := obligationTree aligned
  canonicalManifest := manifestCanonical aligned.toEncoding

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentCompiler
