import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreArtifact
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedules
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes
import Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDecArtifact
import Nightstream.Implementation.R1CS.FPrimeFullHistoryParentCeSerialization
import Nightstream.Implementation.R1CS.ShiftedTernaryComplete

/-!
Contract: semantic soundness and compiler completeness for the exact terminal
post-fold accumulator owner.

The production owner is a hybrid schedule: three ordinary checked-program
segments are interleaved with two compact seeded-Phi81 linear blocks.  This
module reasons over that exact schedule without expanding either seeded
matrix.  Soundness reconstructs program equations, canonical shifted-ternary
openings, both seeded linear maps, and both Poseidon2 digests.  Completeness
starts from executable compiler states and semantic seeded-map equations; it
never assumes owner-row satisfaction or a verifier acceptance flag.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCore

set_option maxRecDepth 1048576
set_option maxHeartbeats 8000000

/-- Source-to-owner column map for one production shifted-ternary opening.
Columns 2--57 belong to the standalone commitment metadata and are unused by
the 124 canonical rows, so they are deliberately mapped to constant one. -/
def shiftedColumnMap (mapping : ShiftedTernaryMap) : List Nat :=
  FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.columnMap mapping

def Pulled (mapping : ShiftedTernaryMap) (assignment : Nat → Nat) : Nat → Nat :=
  Relabel.assignment (shiftedColumnMap mapping) assignment

def shiftedRows (mapping : ShiftedTernaryMap) : List Row :=
  canonicalRows.map (Relabel.row (shiftedColumnMap mapping))

/-- Every generated map has one exact schedule shard. -/
private theorem shiftedTernaryMaps_length :
    shiftedTernaryMaps.length = 192 := by native_decide

private theorem canonicalRows_length : canonicalRows.length = 124 := by
  native_decide


private theorem satisfies_slice
    {ownerRows : List Row} {assignment : Nat → Nat}
    (satisfies : Satisfies ownerRows assignment) (start count : Nat) :
    Satisfies ((ownerRows.drop start).take count) assignment := by
  intro row member
  exact satisfies row
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

private theorem shiftedMap_one (mapping : ShiftedTernaryMap) :
    Relabel.column (shiftedColumnMap mapping) 0 = 0 := by
  simp [shiftedColumnMap,
    FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.columnMap,
    Relabel.column]

/-- Verifier-authoritative part of the decoded strict-PiDEC accumulator. -/
structure ParentAuthority where
  childCount : Nat
  parentCeDigest : List Nat
deriving DecidableEq, Repr

/-- Packed `parent_authority/v2` domain tag.  The child count comes from the
generated strict-PiDEC layout rather than from a caller-provided value. -/
def parentAuthorityTagValues : List Nat :=
  [46, 30521782141150574, 31069335676202596, 33052923221205295,
    31577365934268780, 27408026413920865, 32767037514740853, 846606201]

def decodedPiDecAuthority (assignment : Nat → Nat) : ParentAuthority where
  childCount := FPrimeFullHistoryPiDec.layout.children.length
  parentCeDigest := parentCeDigestColumns.map assignment

def ParentAuthority.preimage (authority : ParentAuthority) : List Nat :=
  parentAuthorityTagValues ++ [authority.childCount, 1] ++
    authority.parentCeDigest

def parentAuthorityPrefixValues : List Nat :=
  parentAuthorityTagValues ++ [FPrimeFullHistoryPiDec.layout.children.length, 1]

def accumulatorPrefixColumns : List Nat :=
  (FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.accumulatorDigestTrace
    ).inputColumns.take parentAuthorityPrefixValues.length

def constantDefinition (column value : Nat) : Definition :=
  ⟨column, .linear [(0, value)]⟩

def parentAuthorityPrefixDefinitions : List Definition :=
  (accumulatorPrefixColumns.zip parentAuthorityPrefixValues).map fun pair =>
    constantDefinition pair.1 pair.2

private theorem parentAuthorityPrefixDefinitions_member :
    ∀ definition ∈ parentAuthorityPrefixDefinitions,
      definition ∈ definitions segment2Instructions := by
  native_decide

private theorem parentAuthorityPrefixValues_canonical :
    ∀ value ∈ parentAuthorityPrefixValues, value < goldilocksP := by
  native_decide

private theorem accumulatorPrefixColumns_length :
    accumulatorPrefixColumns.length = parentAuthorityPrefixValues.length := by
  native_decide

private theorem constantDefinition_value
    {assignment : Nat → Nat} {column value : Nat}
    (one : assignment 0 = 1) (valueLt : value < goldilocksP)
    (holds : (constantDefinition column value).Holds assignment) :
    assignment column = value := by
  simpa [constantDefinition, Definition.Holds, Rhs.eval, lcEval, one,
    Nat.mod_eq_of_lt valueLt] using holds

private theorem constantDefinitions_values
    {assignment : Nat → Nat} (one : assignment 0 = 1) :
    ∀ {columns values : List Nat},
      columns.length = values.length →
      (∀ value ∈ values, value < goldilocksP) →
      (∀ pair ∈ columns.zip values,
        (constantDefinition pair.1 pair.2).Holds assignment) →
      columns.map assignment = values := by
  intro columns
  induction columns with
  | nil =>
      intro values sameLength _ _
      cases values with
      | nil => rfl
      | cons _ _ => simp at sameLength
  | cons column columns inductionHypothesis =>
      intro values sameLength canonical holds
      cases values with
      | nil => simp at sameLength
      | cons value values =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          have headEq : assignment column = value :=
            constantDefinition_value one (canonical value (by simp))
              (holds (column, value) (by simp))
          have tailEq := inductionHypothesis sameLength
            (fun candidate member => canonical candidate (by simp [member]))
            (fun pair member => holds pair (by simp [member]))
          simp only [List.map_cons, List.cons.injEq]
          exact ⟨headEq, tailEq⟩

private theorem parentAuthorityPrefixValues_sound
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (program : AssignmentHolds segment2Instructions assignment) :
    accumulatorPrefixColumns.map assignment = parentAuthorityPrefixValues := by
  apply constantDefinitions_values one accumulatorPrefixColumns_length
    parentAuthorityPrefixValues_canonical
  intro pair member
  apply program.definitions (constantDefinition pair.1 pair.2)
  apply parentAuthorityPrefixDefinitions_member
  exact List.mem_map.mpr ⟨pair, member, rfl⟩

def parentCeClaimConstantColumns : List Nat :=
  FPrimeFullHistoryParentCeSerialization.constantColumnsFrom
    parentCeClaimSourceColumns

def parentCeClaimConstantDefinitions : List Definition :=
  (parentCeClaimConstantColumns.zip
    FPrimeFullHistoryParentCeSerialization.constantValues).map fun pair =>
      constantDefinition pair.1 pair.2

private theorem parentCeClaimSourceColumns_schema :
    parentCeClaimSourceColumns =
      FPrimeFullHistoryParentCeSerialization.expectedSourceColumns
        FPrimeFullHistoryPiDec.recursiveColumnMap
        parentCeClaimConstantColumns := by
  native_decide

private theorem parentCeClaimConstantColumns_length :
    parentCeClaimConstantColumns.length =
      FPrimeFullHistoryParentCeSerialization.constantValues.length := by
  native_decide

private theorem parentCeClaimConstantValues_canonical :
    ∀ value ∈ FPrimeFullHistoryParentCeSerialization.constantValues,
      value < goldilocksP := by
  native_decide

private theorem parentCeClaimConstantDefinitions_member :
    ∀ definition ∈ parentCeClaimConstantDefinitions,
      definition ∈ definitions segment0Instructions := by
  native_decide

private theorem parentCeClaimConstantValues_sound
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (program : AssignmentHolds segment0Instructions assignment) :
    parentCeClaimConstantColumns.map assignment =
      FPrimeFullHistoryParentCeSerialization.constantValues := by
  apply constantDefinitions_values one parentCeClaimConstantColumns_length
    parentCeClaimConstantValues_canonical
  intro pair member
  apply program.definitions (constantDefinition pair.1 pair.2)
  apply parentCeClaimConstantDefinitions_member
  exact List.mem_map.mpr ⟨pair, member, rfl⟩

private theorem accumulatorDigestTrace_inputs :
    (FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.accumulatorDigestTrace
      ).inputColumns = accumulatorPrefixColumns ++ parentCeDigestColumns := by
  native_decide

private theorem canonicalOpening_of_schedule
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {mapping : ShiftedTernaryMap}
    {ownerRows : List Row} {start : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies ownerRows assignment)
    (schedule :
      (ownerRows.drop start).take canonicalRows.length =
        shiftedRows mapping) :
    CanonicalOpening (Pulled mapping assignment) := by
  apply canonicalOpening_of_canonicalRows goldilocksPrime
    (Relabel.canonical canonical)
    (Relabel.constantOne (shiftedMap_one mapping) one)
  apply (Relabel.satisfies_mapped_iff canonicalRows
    (shiftedColumnMap mapping) assignment).mp
  change Satisfies (shiftedRows mapping) assignment
  rw [← schedule]
  exact satisfies_slice satisfies start canonicalRows.length

private theorem canonicalOpeningAt
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (segment0Satisfies : Satisfies segment0Rows assignment)
    (segment1Satisfies : Satisfies segment1Rows assignment) :
    ∀ index, index < shiftedTernaryMaps.length →
      CanonicalOpening
        (Pulled (shiftedTernaryMaps.getD index default) assignment) := by
  intro index indexLt
  have indexLt192 : index < 192 := by
    simpa [shiftedTernaryMaps_length] using indexLt
  let mapping := shiftedTernaryMaps.getD index default
  change CanonicalOpening (Pulled mapping assignment)
  have schedule := FPrimeFullHistoryRecursiveAccumulatorCoreSchedule.rows_schedule index indexLt192
  change
    ((FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.shiftedOwnerRows
      mapping).drop
      (FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.shiftedLocalRowStart
        mapping)).take 124 =
      canonicalRows.map
        (Relabel.row
          (FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.columnMap
            mapping)) at schedule
  by_cases owner : mapping.rowStart < segment1RowStart
  · have scheduled :
        (segment0Rows.drop mapping.rowStart).take canonicalRows.length =
          shiftedRows mapping := by
      simpa [shiftedRows, shiftedColumnMap, canonicalRows_length,
      FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.shiftedOwnerRows,
      FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.shiftedLocalRowStart, owner] using schedule
    exact canonicalOpening_of_schedule goldilocksPrime canonical one
      segment0Satisfies scheduled
  · have scheduled :
        (segment1Rows.drop
          (mapping.rowStart - segment1RowStart)).take canonicalRows.length =
            shiftedRows mapping := by
      simpa [shiftedRows, shiftedColumnMap, canonicalRows_length,
      FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.shiftedOwnerRows,
      FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.shiftedLocalRowStart, owner] using schedule
    exact canonicalOpening_of_schedule goldilocksPrime canonical one
      segment1Satisfies scheduled


/-- Same-assignment semantics of the three ordinary compiler segments. -/
structure ProgramFacts (assignment : Nat → Nat) : Prop where
  segment0 : AssignmentHolds segment0Instructions assignment
  segment1 : AssignmentHolds segment1Instructions assignment
  segment2 : AssignmentHolds segment2Instructions assignment

/-- Exact ordinary definitions bind the raw first-SIS source to the
verifier-normalized strict-PiDEC parent serialization. -/
theorem parentCeClaimSourceValues_sound
    {assignment : Nat → Nat} (one : assignment 0 = 1)
    (program : ProgramFacts assignment) :
    parentCeClaimSourceColumns.map assignment =
      FPrimeFullHistoryParentCeSerialization.parentPreimageWith
        FPrimeFullHistoryPiDec.recursiveColumnMap assignment := by
  rw [parentCeClaimSourceColumns_schema]
  exact FPrimeFullHistoryParentCeSerialization.expectedSourceColumns_values
    (parentCeClaimConstantValues_sound one program.segment0)

/-- Independent semantic conclusions reconstructed from all exact owner
rows.  The digest equations are pure Poseidon2 evaluations of the ordered
input values, not self-authenticating carried digests. -/
structure Facts (assignment : Nat → Nat) : Prop where
  program : ProgramFacts assignment
  canonicalOpenings : ∀ mapping ∈ shiftedTernaryMaps,
    CanonicalOpening (Pulled mapping assignment)
  seeded6 : FPrimeFullHistorySeededPhi81.block6.Holds assignment
  seeded7 : FPrimeFullHistorySeededPhi81.block7.Holds assignment
  parentClaimSource :
    parentCeClaimSourceColumns.map assignment =
      FPrimeFullHistoryParentCeSerialization.parentPreimageWith
        FPrimeFullHistoryPiDec.recursiveColumnMap assignment
  parentAuthorityPreimage :
    (FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.accumulatorDigestTrace
      ).inputColumns.map assignment =
        (decodedPiDecAuthority assignment).preimage
  parentCeDigest : ∀ lane, lane < 4 →
    assignment
        ((FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.parentCeDigestTrace
          ).outputColumns.getD lane 0) =
      Poseidon2Sponge.runValueRounds
        (FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.parentCeDigestTrace
          ).rounds
        ((FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.parentCeDigestTrace
          ).inputColumns.map assignment)
        (fun _ => 0) lane
  accumulatorDigest : ∀ lane, lane < 4 →
    assignment
        ((FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.accumulatorDigestTrace
          ).outputColumns.getD lane 0) =
      Poseidon2Sponge.runValueRounds
        (FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.accumulatorDigestTrace
          ).rounds
        ((FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.accumulatorDigestTrace
          ).inputColumns.map assignment)
        (fun _ => 0) lane

/-- `CIR-SOUND` for every one of the exact 37,295 recursive-accumulator-core rows. -/
theorem sound (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Facts assignment := by
  have pieces := (satisfies_flatten_iff rowPieces assignment).mp satisfies
  have segment0Satisfies : Satisfies segment0Rows assignment :=
    pieces segment0Rows (by simp [rowPieces])
  have block6Satisfies :
      Satisfies FPrimeFullHistorySeededPhi81.block6.rows assignment :=
    pieces FPrimeFullHistorySeededPhi81.block6.rows (by simp [rowPieces])
  have segment1Satisfies : Satisfies segment1Rows assignment :=
    pieces segment1Rows (by simp [rowPieces])
  have block7Satisfies :
      Satisfies FPrimeFullHistorySeededPhi81.block7.rows assignment :=
    pieces FPrimeFullHistorySeededPhi81.block7.rows (by simp [rowPieces])
  have segment2Satisfies : Satisfies segment2Rows assignment :=
    pieces segment2Rows (by simp [rowPieces])
  let programFacts : ProgramFacts assignment := {
    segment0 := assignmentHolds_sound segment0_definitions_canonical
      canonical one segment0Satisfies
    segment1 := assignmentHolds_sound segment1_definitions_canonical
      canonical one segment1Satisfies
    segment2 := assignmentHolds_sound segment2_definitions_canonical
      canonical one segment2Satisfies }
  refine {
    program := programFacts
    canonicalOpenings := ?_
    seeded6 := SeededPhi81.sound canonical one block6Satisfies
    seeded7 := SeededPhi81.sound canonical one block7Satisfies
    parentClaimSource := parentCeClaimSourceValues_sound one programFacts
    parentAuthorityPreimage := by
      rw [accumulatorDigestTrace_inputs, List.map_append,
        parentAuthorityPrefixValues_sound one
          (assignmentHolds_sound segment2_definitions_canonical
            canonical one segment2Satisfies)]
      rfl
    parentCeDigest := Poseidon2Sponge.trace_values_sound
      FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.parentCeDigestTrace_valid
      canonical one segment2Satisfies
    accumulatorDigest := Poseidon2Sponge.trace_values_sound
      FPrimeFullHistoryRecursiveAccumulatorCorePoseidonHashes.accumulatorDigestTrace_valid
      canonical one segment2Satisfies }
  intro mapping member
  rcases List.mem_iff_getElem.mp member with ⟨index, indexLt, mapEq⟩
  have getEq := List.getElem_eq_getD
    (l := shiftedTernaryMaps) (i := index) (h := indexLt) default
  rw [getEq] at mapEq
  subst mapping
  exact canonicalOpeningAt goldilocksPrime canonical one
    segment0Satisfies segment1Satisfies index indexLt

/-- An independent executable compiler witness for one ordinary segment.
`source` is the native pre-state, `checks` is evaluated after deterministic
execution, and `output` identifies the generated assignment with that
execution.  No R1CS satisfaction proposition occurs in this structure. -/
structure SegmentExecution
    (inputColumns : List Nat) (instructions : List Instruction)
    (assignment : Nat → Nat) where
  source : Nat → Nat
  sourceCanonical : ∀ column, source column < goldilocksP
  sourceOne : source 0 = 1
  output : interpret source instructions = assignment

theorem SegmentExecution.compiles
    {inputColumns : List Nat} {instructions : List Instruction}
    {assignment : Nat → Nat}
    (wellFormed : WellFormed inputColumns (definitions instructions))
    (canonicalDefinitions : ∀ definition ∈ definitions instructions,
      definition.Canonical)
    (oneOwned : 0 ∈ inputColumns)
    (execution : SegmentExecution inputColumns instructions assignment)
    (checks : ChecksHold execution.source instructions) :
    Satisfies (CheckedProgram.rows instructions) assignment := by
  have compiled := CheckedProgram.complete wellFormed canonicalDefinitions
    execution.sourceCanonical oneOwned execution.sourceOne checks
  rw [execution.output] at compiled
  exact compiled

/-- Native/compiler witness for the full hybrid owner.  The ordinary rows
come from executable SSA traces; the two dense linear maps are stated by
their independently executable seeded-Phi81 semantics. -/
structure CompilerWitness (assignment : Nat → Nat) where
  segment0 : SegmentExecution segment0InputColumns segment0Instructions assignment
  segment1 : SegmentExecution segment1InputColumns segment1Instructions assignment
  segment2 : SegmentExecution segment2InputColumns segment2Instructions assignment
  canonicalMaps : ∀ mapIndex,
    mapIndex < shiftedTernaryMaps.length →
      ShiftedTernaryComplete.CanonicalWitness
        (Pulled (shiftedTernaryMaps.getD mapIndex default) assignment)
  seeded6 : FPrimeFullHistorySeededPhi81.block6.Holds assignment
  seeded7 : FPrimeFullHistorySeededPhi81.block7.Holds assignment
  parentClaimSource :
    parentCeClaimSourceColumns.map assignment =
      FPrimeFullHistoryParentCeSerialization.parentPreimageWith
        FPrimeFullHistoryPiDec.recursiveColumnMap assignment

private theorem checkPatternTags_bounded : ∀ mapIndex,
    mapIndex <
        FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatternTags.length →
      (FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatternTags
        ).getD mapIndex 0 <
        FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatterns.length := by
  native_decide

private theorem selectedCanonicalRow_mem
    {mapIndex rowIndex : Nat}
    (mapIndexLt : mapIndex < shiftedTernaryMaps.length)
    (rowIndexMember : rowIndex ∈
      FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatterns.getD
        ((FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatternTags
          ).getD mapIndex 0) []) :
    canonicalRows.getD rowIndex
        FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.defaultRow ∈
      canonicalRows := by
  have tagIndexLt : mapIndex <
      FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatternTags.length := by
    simpa [FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatternTags_length]
      using mapIndexLt
  have tagLt := checkPatternTags_bounded mapIndex tagIndexLt
  have patternMember :
      FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatterns.getD
          ((FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatternTags
            ).getD mapIndex 0) [] ∈
        FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatterns := by
    have member := List.getElem_mem
      (l := FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatterns)
      tagLt
    rwa [List.getElem_eq_getD []] at member
  have rowIndexLt :=
    FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checkPatterns_bounded
      _ patternMember rowIndex rowIndexMember
  have member := List.getElem_mem (l := canonicalRows) rowIndexLt
  rwa [List.getElem_eq_getD
    FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.defaultRow] at member

private theorem checksForMapIndex_complete
    {assignment : Nat → Nat} {mapIndex : Nat}
    (mapIndexLt : mapIndex < shiftedTernaryMaps.length)
    (witness : ShiftedTernaryComplete.CanonicalWitness
      (Pulled (shiftedTernaryMaps.getD mapIndex default) assignment)) :
    Satisfies
      (FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checksForMapIndex
        mapIndex) assignment := by
  have canonicalSatisfies :=
    ShiftedTernaryComplete.canonicalRows_complete witness
  have mappedSatisfies : Satisfies
      (canonicalRows.map (Relabel.row
        (FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.columnMap
          (shiftedTernaryMaps.getD mapIndex default)))) assignment :=
    (Relabel.satisfies_mapped_iff canonicalRows
      (FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.columnMap
        (shiftedTernaryMaps.getD mapIndex default)) assignment).mpr
      canonicalSatisfies
  intro row member
  unfold FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checksForMapIndex at member
  rcases List.mem_map.mp member with ⟨rowIndex, rowIndexMember, rfl⟩
  apply mappedSatisfies
  exact List.mem_map.mpr ⟨_,
    selectedCanonicalRow_mem mapIndexLt rowIndexMember, rfl⟩

private theorem expectedChecks_complete
    {assignment : Nat → Nat} {mapIndices : List Nat}
    (indicesBounded : ∀ mapIndex ∈ mapIndices,
      mapIndex < shiftedTernaryMaps.length)
    (witness : CompilerWitness assignment) :
    Satisfies
      (mapIndices.flatMap
        FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.checksForMapIndex)
      assignment := by
  intro row member
  rcases List.mem_flatMap.mp member with
    ⟨mapIndex, mapIndexMember, rowMember⟩
  exact checksForMapIndex_complete
    (indicesBounded mapIndex mapIndexMember)
    (witness.canonicalMaps mapIndex
      (indicesBounded mapIndex mapIndexMember)) row rowMember

private theorem segment0MapIndices_bounded : ∀ mapIndex ∈
    FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.segment0MapIndices,
      mapIndex < shiftedTernaryMaps.length := by
  native_decide

private theorem segment1MapIndices_bounded : ∀ mapIndex ∈
    FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.segment1MapIndices,
      mapIndex < shiftedTernaryMaps.length := by
  native_decide

private theorem segment2MapIndices_bounded : ∀ mapIndex ∈
    FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.segment2MapIndices,
      mapIndex < shiftedTernaryMaps.length := by
  native_decide

private theorem segment0ChecksHold {assignment : Nat → Nat}
    (witness : CompilerWitness assignment) :
    ChecksHold witness.segment0.source segment0Instructions := by
  unfold ChecksHold
  rw [witness.segment0.output]
  intro row member
  apply expectedChecks_complete segment0MapIndices_bounded witness row
  exact (congrArg (fun ownerRows => row ∈ ownerRows)
    FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.segment0_checks_covered).mp
      member

private theorem segment1ChecksHold {assignment : Nat → Nat}
    (witness : CompilerWitness assignment) :
    ChecksHold witness.segment1.source segment1Instructions := by
  unfold ChecksHold
  rw [witness.segment1.output]
  intro row member
  apply expectedChecks_complete segment1MapIndices_bounded witness row
  exact (congrArg (fun ownerRows => row ∈ ownerRows)
    FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.segment1_checks_covered).mp
      member

private theorem segment2ChecksHold {assignment : Nat → Nat}
    (witness : CompilerWitness assignment) :
    ChecksHold witness.segment2.source segment2Instructions := by
  unfold ChecksHold
  rw [witness.segment2.output]
  intro row member
  apply expectedChecks_complete segment2MapIndices_bounded witness row
  exact (congrArg (fun ownerRows => row ∈ ownerRows)
    FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.segment2_checks_covered).mp
      member

/-- `CIR-COMPLETE` for the exact hybrid owner.  Its premise is a native
compiler execution plus the two seeded linear-map equations, never row
satisfaction and never `AssignmentHolds`. -/
theorem complete {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : CompilerWitness assignment) :
    Satisfies rows assignment := by
  have segment0Satisfies : Satisfies segment0Rows assignment :=
    witness.segment0.compiles segment0_definitions_wellFormed
      segment0_definitions_canonical (by native_decide)
      (segment0ChecksHold witness)
  have segment1Satisfies : Satisfies segment1Rows assignment :=
    witness.segment1.compiles segment1_definitions_wellFormed
      segment1_definitions_canonical (by native_decide)
      (segment1ChecksHold witness)
  have segment2Satisfies : Satisfies segment2Rows assignment :=
    witness.segment2.compiles segment2_definitions_wellFormed
      segment2_definitions_canonical (by native_decide)
      (segment2ChecksHold witness)
  have block6Satisfies := SeededPhi81.complete canonical one witness.seeded6
  have block7Satisfies := SeededPhi81.complete canonical one witness.seeded7
  apply (satisfies_flatten_iff rowPieces assignment).mpr
  intro piece member
  change piece ∈
    [segment0Rows, FPrimeFullHistorySeededPhi81.block6.rows,
      segment1Rows, FPrimeFullHistorySeededPhi81.block7.rows,
      segment2Rows] at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl
  · exact segment0Satisfies
  · exact block6Satisfies
  · exact segment1Satisfies
  · exact block7Satisfies
  · exact segment2Satisfies

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSound
