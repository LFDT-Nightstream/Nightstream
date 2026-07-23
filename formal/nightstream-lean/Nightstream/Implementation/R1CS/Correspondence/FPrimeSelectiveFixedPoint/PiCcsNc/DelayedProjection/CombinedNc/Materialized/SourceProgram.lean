import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SourceRows
import Nightstream.Implementation.R1CS.Core.CheckedProgram
import Nightstream.Implementation.R1CS.Correspondence.Projection.IndexedRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDecodeBridge

/-!
Concrete source-program reconstruction contract for the production
combined-NC selective artifact.

Owns: the exact kernel bridge from an ordered mixed SSA/check program to a
canonical assignment satisfying the literal generated source rows, modulo
only per-row sparse-term permutation, and the lossless lift of that result
through the fail-closed source-row decoder.

Does not own: the missing generated instruction orientation, the generated
retained-owner records, selected-row satisfaction, selector truth, transcript
scheduling, raw-child authority, commitment binding, costs, or row removal.

Emits constraints: none.

This file deliberately does not infer an SSA program from sparse A/B/C rows.
Choosing a solvable column is semantic data: treating an assertion as a
definition can make an invalid witness satisfiable.  The Rust product-sum
validator already computes exact product/linear definitions and distinguishes
retained boundaries, but the current generated Lean artifact exports only the
rows and aggregate rewrite ranges.  The missing exporter payload is:

* one ordered `RawInstructionRecord` for every generated source row;
* the exact input-column boundary for that instruction stream;
* for every `.check`, the exact retained `(sourceRow, emittedRow)` pair.

The ordered relation below is intentionally shardable.  A generated owner can
check bounded record blocks and compose them with
`instructionRecordsMatch_append`; it need not normalize one proof-carrying
8,021-record object.  No record carries a proof of row satisfaction.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_program` | Define the exact ordered source equations refined by compiler rewrites and retained checks. | computed semantics |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-- Literal executable rows selected from the production source relation. -/
def generatedRows : List Row :=
  SourceRows.values.map SourceDecodeBridge.rawRow

/-! ## Missing proof-free exporter surface -/

/-- Stable retained owner for one physical assertion row.  These two indices
are coordinates, not semantic authority: the artifact owner must separately
check that this exact pair occurs in the retained compiler schedule. -/
structure RawCheckOwner where
  sourceRow : Nat
  emittedRow : Nat
deriving DecidableEq, Repr

/-- Required ordered instruction payload.  A definition has no owner; a check
has exactly one retained `(sourceRow, emittedRow)` owner. -/
structure RawInstructionRecord where
  sourceRow : Nat
  instruction : CheckedProgram.Instruction
  owner : Option RawCheckOwner
deriving DecidableEq, Repr

def RawInstructionRecord.WellClassified
    (record : RawInstructionRecord) : Prop :=
  match record.instruction, record.owner with
  | .define _, none => True
  | .check _, some owner => owner.sourceRow = record.sourceRow
  | _, _ => False

instance (record : RawInstructionRecord) :
    Decidable record.WellClassified := by
  cases record with
  | mk sourceRow instruction owner =>
      cases instruction <;> cases owner <;>
        simp only [RawInstructionRecord.WellClassified] <;> infer_instance

def recordInstructions (records : List RawInstructionRecord) :
    List CheckedProgram.Instruction :=
  records.map RawInstructionRecord.instruction

/-- Lockstep, fail-closed comparison of a source-row block with an oriented
instruction-record block.  Mismatched lengths are false.  Each source index
is preserved, every check names its exact retained owner pair, and each A/B/C
linear combination is preserved modulo sparse-term order. -/
def InstructionRecordsMatch :
    List RawSourceRow → List RawInstructionRecord → Prop
  | [], [] => True
  | source :: sources, record :: records =>
      record.sourceRow = source.sourceRow ∧
        record.WellClassified ∧
        RowsPermutationEquivalent
          (SourceDecodeBridge.rawRow source) record.instruction.row ∧
        InstructionRecordsMatch sources records
  | _, _ => False

private def instructionRecordsMatchDecidable :
    (sources : List RawSourceRow) →
      (records : List RawInstructionRecord) →
        Decidable (InstructionRecordsMatch sources records)
  | [], [] => isTrue True.intro
  | [], _ :: _ => isFalse id
  | _ :: _, [] => isFalse id
  | source :: sources, record :: records => by
      letI : Decidable (InstructionRecordsMatch sources records) :=
        instructionRecordsMatchDecidable sources records
      change Decidable
        (record.sourceRow = source.sourceRow ∧
          record.WellClassified ∧
          RowsPermutationEquivalent
            (SourceDecodeBridge.rawRow source) record.instruction.row ∧
          InstructionRecordsMatch sources records)
      infer_instance

instance (sources : List RawSourceRow)
    (records : List RawInstructionRecord) :
    Decidable (InstructionRecordsMatch sources records) :=
  instructionRecordsMatchDecidable sources records

/-- Exact ordered production contract.  Its lockstep definition fixes the
complete 8,021-record coverage without relying on a global `Nodup` proof or
literal sparse-term order.  Artifact owners should establish it from bounded
blocks using `instructionRecordsMatch_append`. -/
def InstructionRecordsExact (records : List RawInstructionRecord) : Prop :=
  InstructionRecordsMatch SourceRows.values records

/-- Bounded exact-record certificates compose without weakening length,
ordering, owner classification, or per-row A/B/C equivalence. -/
theorem instructionRecordsMatch_append
    {leftSources rightSources : List RawSourceRow}
    {leftRecords rightRecords : List RawInstructionRecord}
    (left : InstructionRecordsMatch leftSources leftRecords)
    (right : InstructionRecordsMatch rightSources rightRecords) :
    InstructionRecordsMatch (leftSources ++ rightSources)
      (leftRecords ++ rightRecords) := by
  induction leftSources generalizing leftRecords with
  | nil =>
      cases leftRecords with
      | nil => simpa using right
      | cons _ _ => simp [InstructionRecordsMatch] at left
  | cons source sources inductionHypothesis =>
      cases leftRecords with
      | nil => simp [InstructionRecordsMatch] at left
      | cons record records =>
          change record.sourceRow = source.sourceRow ∧
            record.WellClassified ∧
            RowsPermutationEquivalent
              (SourceDecodeBridge.rawRow source) record.instruction.row ∧
            InstructionRecordsMatch sources records at left
          change record.sourceRow = source.sourceRow ∧
            record.WellClassified ∧
            RowsPermutationEquivalent
              (SourceDecodeBridge.rawRow source) record.instruction.row ∧
            InstructionRecordsMatch
              (sources ++ rightSources) (records ++ rightRecords)
          exact ⟨left.1, left.2.1, left.2.2.1,
            inductionHypothesis left.2.2.2⟩

/-- Forgetting indices and owner metadata from an exact record block leaves
the existing lockstep row-permutation relation. -/
theorem instructionRecordsMatch_implies_rowsEquivalent
    {sources : List RawSourceRow}
    {records : List RawInstructionRecord}
    (certificate : InstructionRecordsMatch sources records) :
    RowsPermutationEquivalentList
      (sources.map SourceDecodeBridge.rawRow)
      (CheckedProgram.rows (recordInstructions records)) := by
  induction sources generalizing records with
  | nil =>
      cases records with
      | nil => trivial
      | cons _ _ => simp [InstructionRecordsMatch] at certificate
  | cons source sources inductionHypothesis =>
      cases records with
      | nil => simp [InstructionRecordsMatch] at certificate
      | cons record records =>
          change record.sourceRow = source.sourceRow ∧
            record.WellClassified ∧
            RowsPermutationEquivalent
              (SourceDecodeBridge.rawRow source) record.instruction.row ∧
            InstructionRecordsMatch sources records at certificate
          change RowsPermutationEquivalent
              (SourceDecodeBridge.rawRow source) record.instruction.row ∧
            RowsPermutationEquivalentList
              (sources.map SourceDecodeBridge.rawRow)
              (CheckedProgram.rows (recordInstructions records))
          exact ⟨certificate.2.2.1,
            inductionHypothesis certificate.2.2.2⟩

/-- Exact non-semantic orientation required from the owning Rust exporter.
"Exact" permits only sparse-term permutation inside each A/B/C vector; row
order and every coefficient/column pair remain fixed. -/
def ExactInstructionRows (instructions : List CheckedProgram.Instruction) : Prop :=
  RowsPermutationEquivalentList generatedRows
    (CheckedProgram.rows instructions)

theorem instructionRecordsExact_implies_exactRows
    {records : List RawInstructionRecord}
    (certificate : InstructionRecordsExact records) :
    ExactInstructionRows (recordInstructions records) := by
  exact instructionRecordsMatch_implies_rowsEquivalent certificate

private theorem instructionRecordsMatch_classified
    {sources : List RawSourceRow}
    {records : List RawInstructionRecord}
    (certificate : InstructionRecordsMatch sources records) :
    ∀ record ∈ records, record.WellClassified := by
  induction sources generalizing records with
  | nil =>
      cases records with
      | nil => simp
      | cons _ _ => simp [InstructionRecordsMatch] at certificate
  | cons source sources inductionHypothesis =>
      cases records with
      | nil => simp [InstructionRecordsMatch] at certificate
      | cons record records =>
          change record.sourceRow = source.sourceRow ∧
            record.WellClassified ∧
            RowsPermutationEquivalent
              (SourceDecodeBridge.rawRow source) record.instruction.row ∧
            InstructionRecordsMatch sources records at certificate
          intro candidate member
          simp only [List.mem_cons] at member
          rcases member with rfl | member
          · exact certificate.2.1
          · exact inductionHypothesis certificate.2.2.2 candidate member

/-- Exact owner evidence discharges every assertion extracted from a matched
instruction stream.  Definitions cannot enter this proof: classification
forces their owner to be absent, while every check carries one concrete
`(sourceRow, emittedRow)` owner. -/
theorem ownedChecksSatisfy_of_instructionRecordsMatch
    {sources : List RawSourceRow}
    {records : List RawInstructionRecord}
    {assignment : Nat → Nat}
    (certificate : InstructionRecordsMatch sources records)
    (ownerHolds : ∀ record ∈ records, ∀ owner,
      record.owner = some owner →
        RowHolds assignment record.instruction.row) :
    Satisfies (CheckedProgram.checks (recordInstructions records))
      assignment := by
  have classified := instructionRecordsMatch_classified certificate
  intro row member
  rcases List.mem_filterMap.mp member with
    ⟨instruction, instructionMember, mapped⟩
  rcases List.mem_map.mp instructionMember with
    ⟨record, recordMember, instructionEq⟩
  subst instruction
  cases instructionKind : record.instruction with
  | define definition =>
      simp [instructionKind] at mapped
  | check checkRow =>
      simp only [instructionKind] at mapped
      have rowEq : checkRow = row := Option.some.inj mapped
      subst row
      have wellClassified := classified record recordMember
      cases ownerEq : record.owner with
      | none =>
          simp [RawInstructionRecord.WellClassified, instructionKind,
            ownerEq] at wellClassified
      | some owner =>
          simpa [instructionKind] using
            ownerHolds record recordMember owner ownerEq

/-- Execute the oriented source definitions while leaving verifier assertions
as checks.  This is the only source-assignment constructor in this leaf. -/
def reconstruct (seed : Nat → Nat)
    (instructions : List CheckedProgram.Instruction) : Nat → Nat :=
  CheckedProgram.interpret seed instructions

/-- The reconstructed assignment remains canonically represented. -/
theorem reconstruct_canonical
    (seed : Nat → Nat) (instructions : List CheckedProgram.Instruction)
    (seedCanonical : ∀ column, seed column < goldilocksP) :
    ∀ column, reconstruct seed instructions column < goldilocksP := by
  exact Program.run_canonical seedCanonical

/-- A checked constant-one input is preserved by the reconstruction. -/
theorem reconstruct_constantOne
    {seed : Nat → Nat} {inputColumns : List Nat}
    {instructions : List CheckedProgram.Instruction}
    (wellFormed : Program.WellFormed inputColumns
      (CheckedProgram.definitions instructions))
    (constantOneColumn : 0 ∈ inputColumns)
    (constantOne : seed 0 = 1) :
    reconstruct seed instructions 0 = 1 := by
  exact (Program.run_preserves_known wellFormed seed 0 constantOneColumn).trans
    constantOne

/-- Retained emitted-row evidence can be joined to the exact generated owner
records without accepting `ChecksHold` as an opaque premise.  The later
artifact leaf supplies `ownerHolds` from the 52 retained compiler obligations
on this exact reconstructed assignment. -/
theorem reconstruct_checksHold_of_instructionRecordsMatch
    {sources : List RawSourceRow}
    {records : List RawInstructionRecord}
    {seed : Nat → Nat}
    (certificate : InstructionRecordsMatch sources records)
    (ownerHolds : ∀ record ∈ records, ∀ owner,
      record.owner = some owner →
        RowHolds (reconstruct seed (recordInstructions records))
          record.instruction.row) :
    CheckedProgram.ChecksHold seed (recordInstructions records) := by
  exact ownedChecksSatisfy_of_instructionRecordsMatch certificate ownerHolds

/-- Generic semantic transport from the oriented instruction schedule to the
literal generated source rows.  The only tolerated representation difference
is sparse-term order inside the corresponding A/B/C vectors. -/
theorem generatedRows_satisfied_of_instructionRows
    {instructions : List CheckedProgram.Instruction}
    {assignment : Nat → Nat}
    (rowsExact : ExactInstructionRows instructions)
    (instructionRowsSatisfy :
      Satisfies (CheckedProgram.rows instructions) assignment) :
    Satisfies generatedRows assignment := by
  exact sourceRows_satisfied_of_permutationEquivalent rowsExact
    instructionRowsSatisfy

/-- Kernel completion for the literal source rows.  `checksHold` is not an
artifact fact supplied by this file: the generated owner bridge must derive it
from the exact retained emitted-row obligations.

No source-row satisfaction appears in the premises. -/
theorem reconstruct_satisfies_generatedRows
    {seed : Nat → Nat} {inputColumns : List Nat}
    {instructions : List CheckedProgram.Instruction}
    (rowsExact : ExactInstructionRows instructions)
    (wellFormed : Program.WellFormed inputColumns
      (CheckedProgram.definitions instructions))
    (canonicalDefinitions :
      ∀ definition ∈ CheckedProgram.definitions instructions,
        definition.Canonical)
    (seedCanonical : ∀ column, seed column < goldilocksP)
    (constantOneColumn : 0 ∈ inputColumns)
    (constantOne : seed 0 = 1)
    (checksHold : CheckedProgram.ChecksHold seed instructions) :
    Satisfies generatedRows (reconstruct seed instructions) := by
  apply generatedRows_satisfied_of_instructionRows rowsExact
  exact CheckedProgram.complete wellFormed canonicalDefinitions
    seedCanonical constantOneColumn constantOne checksHold

/-! ## Lossless decoded-row lift -/

/-- Ordered fail-closed decoding of a proof-free source-row stream.  This
local relation avoids importing a list-relation compatibility layer merely
for the two constructors needed by the artifact bridge. -/
inductive RowsDecodeExactly :
    List RawSourceRow → List Decoder.DecodedSourceRow → Prop where
  | nil : RowsDecodeExactly [] []
  | cons {source : RawSourceRow} {row : Decoder.DecodedSourceRow}
      {raw : List RawSourceRow} {decoded : List Decoder.DecodedSourceRow}
      (headDecodes : Decoder.decodeSourceRow source = some row)
      (tailDecodes : RowsDecodeExactly raw decoded) :
      RowsDecodeExactly (source :: raw) (row :: decoded)

theorem sourceRows_eq_rawRows_of_decode
    {raw : List RawSourceRow} {decoded : List Decoder.DecodedSourceRow}
    (decodes : RowsDecodeExactly raw decoded) :
    Semantics.sourceRows decoded = raw.map SourceDecodeBridge.rawRow := by
  induction decodes with
  | nil => rfl
  | cons headDecodes tailDecodes inductionHypothesis =>
      simp only [Semantics.sourceRows, List.map_cons]
      rw [SourceDecodeBridge.sourceRowToRow_eq_rawRow_of_decode headDecodes]
      congr 1

/-- The same reconstructed assignment satisfies the typed decoded stream.
This theorem adds no trust: successful decoding is lossless and the raw rows
remain the exact generated coefficient authority. -/
theorem reconstruct_satisfies_decodedRows
    {seed : Nat → Nat} {inputColumns : List Nat}
    {instructions : List CheckedProgram.Instruction}
    {decoded : List Decoder.DecodedSourceRow}
    (decodes : RowsDecodeExactly SourceRows.values decoded)
    (rowsExact : ExactInstructionRows instructions)
    (wellFormed : Program.WellFormed inputColumns
      (CheckedProgram.definitions instructions))
    (canonicalDefinitions :
      ∀ definition ∈ CheckedProgram.definitions instructions,
        definition.Canonical)
    (seedCanonical : ∀ column, seed column < goldilocksP)
    (constantOneColumn : 0 ∈ inputColumns)
    (constantOne : seed 0 = 1)
    (checksHold : CheckedProgram.ChecksHold seed instructions) :
    Semantics.SourceRowsSatisfy decoded (reconstruct seed instructions) := by
  unfold Semantics.SourceRowsSatisfy
  rw [sourceRows_eq_rawRows_of_decode decodes]
  exact reconstruct_satisfies_generatedRows rowsExact wellFormed
    canonicalDefinitions seedCanonical constantOneColumn constantOne checksHold

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceProgram
