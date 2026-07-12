import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryCounterLocalSound
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPriorLinkArtifact
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursiveOutputArtifact
import Nightstream.Implementation.R1CS.Core.Relabel

/-!
Contract: transport the exact 660-row recursive counter certificate back to
the production full-history column space.

The generated artifact owns the local-to-global column map.  This module does
not assume counter correctness: it pulls any satisfying production assignment
through that map and applies the artifact-level arithmetic theorem.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound

open Nightstream.Implementation.R1CS

set_option maxRecDepth 65536
set_option maxHeartbeats 4000000

abbrev Pulled (assignment : Nat → Nat) : Nat → Nat :=
  Relabel.assignment FPrimeFullHistoryCounter.globalColumnMap assignment

def globalRows : List Row :=
  FPrimeFullHistoryCounter.rows.map
    (Relabel.row FPrimeFullHistoryCounter.globalColumnMap)

def chunkInputCol : Nat :=
  Relabel.column FPrimeFullHistoryCounter.globalColumnMap
    FPrimeFullHistoryCounter.chunkInputVarCol

def stepInputCol : Nat :=
  Relabel.column FPrimeFullHistoryCounter.globalColumnMap
    FPrimeFullHistoryCounter.stepInputVarCol

def chunkOutputCol : Nat :=
  Relabel.column FPrimeFullHistoryCounter.globalColumnMap
    FPrimeFullHistoryCounterLocalSound.chunkOutputVarCol

def stepOutputCol : Nat :=
  Relabel.column FPrimeFullHistoryCounter.globalColumnMap
    FPrimeFullHistoryCounterLocalSound.stepOutputVarCol

/-- Compose one of the generated local counter maps with the generated
local-to-production map. -/
def globalizeMap (localMap : List Nat) : List Nat :=
  localMap.map (Relabel.column FPrimeFullHistoryCounter.globalColumnMap)

def chunkOutputCanonicalMap : List Nat :=
  globalizeMap FPrimeFullHistoryCounter.chunkOutputCanonicalMap

def stepOutputCanonicalMap : List Nat :=
  globalizeMap FPrimeFullHistoryCounter.stepOutputCanonicalMap

theorem mapsOne :
    Relabel.column FPrimeFullHistoryCounter.globalColumnMap 0 = 0 := by
  native_decide

theorem concreteColumns :
    chunkInputCol = FPrimeFullHistoryPriorLink.stateInColumns.getD 8 0 ∧
    stepInputCol = FPrimeFullHistoryPriorLink.stateInColumns.getD 9 0 ∧
    chunkOutputCol = FPrimeFullHistoryRecursiveOutput.stateOutColumns.getD 8 0 ∧
    stepOutputCol = FPrimeFullHistoryRecursiveOutput.stateOutColumns.getD 9 0 := by
  native_decide

theorem outputMapsOne :
    Relabel.column chunkOutputCanonicalMap 0 = 0 ∧
    Relabel.column stepOutputCanonicalMap 0 = 0 := by
  native_decide

theorem chunkOutputRowsIncluded :
    rowsIncluded
      (CanonicalU64.rows.map (Relabel.row chunkOutputCanonicalMap))
      globalRows = true := by
  native_decide

theorem stepOutputRowsIncluded :
    rowsIncluded
      (CanonicalU64.rows.map (Relabel.row stepOutputCanonicalMap))
      globalRows = true := by
  native_decide

private theorem satisfiesPulled {assignment : Nat → Nat}
    (satisfies : Satisfies globalRows assignment) :
    Satisfies FPrimeFullHistoryCounter.rows (Pulled assignment) := by
  intro source sourceMember
  apply (Relabel.rowHolds_iff
    FPrimeFullHistoryCounter.globalColumnMap assignment source).mp
  apply satisfies
  exact List.mem_map.mpr ⟨source, sourceMember, rfl⟩

/-- Every satisfying assignment of the exact production counter rows advances
both recursive counters by one as integers, hence cannot hide field wraparound. -/
theorem sound (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies globalRows assignment) :
    assignment chunkOutputCol = assignment chunkInputCol + 1 ∧
    assignment stepOutputCol = assignment stepInputCol + 1 ∧
    assignment chunkInputCol < goldilocksP ∧
    assignment stepInputCol < goldilocksP ∧
    assignment chunkOutputCol < goldilocksP ∧
    assignment stepOutputCol < goldilocksP := by
  have hLocal := FPrimeFullHistoryCounterLocalSound.local_sound
    goldilocksPrime
    (Relabel.canonical canonical)
    (Relabel.constantOne mapsOne one)
    (satisfiesPulled satisfies)
  rcases hLocal with
    ⟨chunkInput, stepInput, chunkOutput, stepOutput,
      chunkAdvance, stepAdvance, chunkInputLt, stepInputLt,
      chunkOutputLt, stepOutputLt⟩
  have rowsInChunk : FPrimeFullHistoryCounter.rowsInChunk = 1 := by
    native_decide
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · simpa [chunkInputCol, chunkOutputCol, Pulled, Relabel.assignment] using
      chunkOutput.trans (chunkAdvance.trans (congrArg (fun value => value + 1)
        chunkInput.symm))
  · simpa [stepInputCol, stepOutputCol, Pulled, Relabel.assignment,
      rowsInChunk] using
      stepOutput.trans (stepAdvance.trans (congrArg (fun value => value + 1)
        stepInput.symm))
  · change Pulled assignment FPrimeFullHistoryCounter.chunkInputVarCol < goldilocksP
    calc
      Pulled assignment FPrimeFullHistoryCounter.chunkInputVarCol =
          FPrimeFullHistoryCounterLocalSound.chunkInputValue
            (Pulled assignment) := chunkInput
      _ < goldilocksP := chunkInputLt
  · change Pulled assignment FPrimeFullHistoryCounter.stepInputVarCol < goldilocksP
    calc
      Pulled assignment FPrimeFullHistoryCounter.stepInputVarCol =
          FPrimeFullHistoryCounterLocalSound.stepInputValue
            (Pulled assignment) := stepInput
      _ < goldilocksP := stepInputLt
  · change Pulled assignment
      FPrimeFullHistoryCounterLocalSound.chunkOutputVarCol < goldilocksP
    calc
      Pulled assignment FPrimeFullHistoryCounterLocalSound.chunkOutputVarCol =
          FPrimeFullHistoryCounterLocalSound.chunkOutputValue
            (Pulled assignment) := chunkOutput
      _ < goldilocksP := chunkOutputLt
  · change Pulled assignment
      FPrimeFullHistoryCounterLocalSound.stepOutputVarCol < goldilocksP
    calc
      Pulled assignment FPrimeFullHistoryCounterLocalSound.stepOutputVarCol =
          FPrimeFullHistoryCounterLocalSound.stepOutputValue
            (Pulled assignment) := stepOutput
      _ < goldilocksP := stepOutputLt

/-- The two exact production output counter gadgets expose their canonical
integer bit decompositions as well as the transition equations. -/
theorem outputCanonical_sound (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies globalRows assignment) :
    (Relabel.assignment chunkOutputCanonicalMap assignment
          CanonicalU64.varCol =
        bitsValue (Relabel.assignment chunkOutputCanonicalMap assignment) ∧
      bitsValue (Relabel.assignment chunkOutputCanonicalMap assignment) <
        goldilocksP) ∧
    (Relabel.assignment stepOutputCanonicalMap assignment
          CanonicalU64.varCol =
        bitsValue (Relabel.assignment stepOutputCanonicalMap assignment) ∧
      bitsValue (Relabel.assignment stepOutputCanonicalMap assignment) <
        goldilocksP) := by
  exact ⟨
    canonicalU64_sound goldilocksPrime (Relabel.canonical canonical)
      (Relabel.constantOne outputMapsOne.1 one)
      (Relabel.satisfies_of_included chunkOutputRowsIncluded satisfies),
    canonicalU64_sound goldilocksPrime (Relabel.canonical canonical)
      (Relabel.constantOne outputMapsOne.2 one)
      (Relabel.satisfies_of_included stepOutputRowsIncluded satisfies)⟩

namespace Compiler

abbrev Source := FPrimeFullHistoryCounterLocalSound.Compiler.Source

/-- One production-column execution of the exact counter compiler.  The
local witness contains only source words and component-interpreter equality. -/
structure ExecutionWitness
    (field : CanonicalU64Complete.FieldInverse)
    (assignment : Nat → Nat) where
  execution : FPrimeFullHistoryCounterLocalSound.Compiler.ExecutionWitness
    field (Pulled assignment)

/-- Honest compiler completeness transported through the generated
local-to-production column map. -/
theorem complete (prime : EuclidPrime goldilocksP)
    {field : CanonicalU64Complete.FieldInverse} {assignment : Nat → Nat}
    (witness : ExecutionWitness field assignment) :
    Satisfies globalRows assignment := by
  apply (Relabel.satisfies_mapped_iff FPrimeFullHistoryCounter.rows
    FPrimeFullHistoryCounter.globalColumnMap assignment).mpr
  exact FPrimeFullHistoryCounterLocalSound.Compiler.complete prime witness.execution

end Compiler

end Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound
