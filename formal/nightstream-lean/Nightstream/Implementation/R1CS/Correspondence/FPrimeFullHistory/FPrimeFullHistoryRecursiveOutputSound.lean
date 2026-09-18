import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Poseidon
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryOutputEncodingSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryCounterSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryPriorLinkSound
import Nightstream.Implementation.R1CS.Ownership.CanonicalU64.CanonicalU64Halves
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: soundness and completeness of the exact recursive output owner.

The 4,768 generated rows remain authoritative.  The theorem exposes their
mixed-program semantics, the functional Poseidon2 state-output digest, and the
canonical public encoding.  A separate theorem composes those producer rows
with the exact terminal delayed-link rows.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutput

set_option maxRecDepth 524288
set_option maxHeartbeats 5000000

def trace : Poseidon2Sponge.Trace :=
  FPrimeFullHistoryRecursiveOutputPoseidonHashes.xOutTrace

def traceOutputValues (assignment : Nat → Nat) : List Nat :=
  (List.range 4).map fun lane =>
    Poseidon2Sponge.runValueRounds trace.rounds
      (trace.inputColumns.map assignment) (fun _ => 0) lane

structure Facts (assignment : Nat → Nat) : Prop where
  program : SoundResult inputColumns instructions assignment assignment
  definitions : ∀ definition ∈ definitions instructions,
    definition.Holds assignment
  sponge : ∀ lane, lane < 4 →
    assignment (trace.outputColumns.getD lane 0) =
      Poseidon2Sponge.runValueRounds trace.rounds
        (trace.inputColumns.map assignment) (fun _ => 0) lane
  encoding : FPrimeEncodingSound.Holds
    (FPrimeFullHistoryOutputEncodingSound.Pulled assignment)

theorem outputEncodingRowsIncluded :
    rowsIncluded FPrimeFullHistoryOutputEncoding.rows rows = true := by
  native_decide

/-- Exact-row satisfaction derives all three independently useful output
facts; none is accepted as a certificate field. -/
theorem sound (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Facts assignment := by
  refine {
    program := CheckedProgram.sound definitions_wellFormed
      definitions_canonical checks_reference (by
        intro column member
        rfl) canonical one satisfies
    definitions := definitionsHold_of_satisfies definitions_canonical
      canonical one satisfies
    sponge := Poseidon2Sponge.trace_values_sound
      FPrimeFullHistoryRecursiveOutputPoseidonHashes.xOutTrace_valid
      canonical one satisfies
    encoding := FPrimeFullHistoryOutputEncodingSound.sound goldilocksPrime
      canonical one ?_
  }
  intro row member
  exact satisfies row (rowsIncluded_sound outputEncodingRowsIncluded row member)

private theorem rangeFour : List.range 4 = [0, 1, 2, 3] := by decide

/-- The four raw output wires are exactly the pure Poseidon2 evaluation of
the ordered state-output preimage wires. -/
theorem xOutValues_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    xOutColumns.map assignment = traceOutputValues assignment := by
  have lane0 := facts.sponge 0 (by decide)
  have lane1 := facts.sponge 1 (by decide)
  have lane2 := facts.sponge 2 (by decide)
  have lane3 := facts.sponge 3 (by decide)
  have traceOutput : trace.outputColumns = xOutColumns := by
    unfold trace
    exact FPrimeFullHistoryRecursiveOutputPoseidonHashes.xOutTrace_output
  rw [traceOutput] at lane0 lane1 lane2 lane3
  simpa [traceOutputValues, trace, rangeFour, xOutColumns] using
    And.intro lane0 (And.intro lane1 (And.intro lane2 lane3))

theorem outputDigestColumns :
    (List.range 4).map (fun lane =>
      Relabel.column FPrimeFullHistoryOutputEncoding.columnMap
        (FPrimeEncoding.digestCol lane)) = xOutColumns := by
  native_decide

theorem outputDigest_eq_xOutColumns (assignment : Nat → Nat) :
    FPrimeFullHistoryOutputEncodingSound.outputDigest assignment =
      xOutColumns.map assignment := by
  change List.map assignment
    ((List.range 4).map (fun lane =>
      Relabel.column FPrimeFullHistoryOutputEncoding.columnMap
        (FPrimeEncoding.digestCol lane))) = xOutColumns.map assignment
  rw [outputDigestColumns]

/-- The exact recursive producer and terminal consumer agree on the actual
four-lane output digest, not merely on an untyped bit range. -/
theorem terminalFreshDigest_eq_xOut
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputSatisfies : Satisfies rows assignment)
    (terminalSatisfies :
      Satisfies FPrimeFullHistoryTerminalLink.rows assignment) :
    FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest assignment =
      xOutColumns.map assignment := by
  have outputFacts := sound goldilocksPrime canonical one outputSatisfies
  have terminalFacts := FPrimeFullHistoryTerminalLinkSound.sound
    canonical one terminalSatisfies
  exact (FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest_eq_outputDigest
    outputFacts.encoding terminalFacts).trans
      (outputDigest_eq_xOutColumns assignment)

/-- `CIR-COMPLETE` compiler rule for this exact owner.  Subsequent semantic
completeness work must prove `ChecksHold`; it cannot assume row satisfaction. -/
theorem complete {state : Nat → Nat}
    (canonical : ∀ column, state column < goldilocksP)
    (one : state 0 = 1)
    (checksHold : ChecksHold state instructions) :
    Satisfies rows (interpret state instructions) := by
  exact CheckedProgram.complete definitions_wellFormed definitions_canonical
    canonical (by native_decide) one checksHold

def semanticStateColumns : List Nat :=
  (stateOutColumns.drop 19).take 4

def accumulatorStateColumns : List Nat :=
  (stateOutColumns.drop 23).take 4

def semanticAccumulatorPairs : List (Nat × Nat) :=
  semanticStateColumns.zip accumulatorStateColumns

theorem semanticAccumulatorRowsIncluded :
    rowsIncluded (EqualityPins.rows semanticAccumulatorPairs) rows = true := by
  native_decide

/-- Stateless output rows force the semantic-state digest to equal the newly
computed accumulator digest lane by lane. -/
theorem semanticAccumulator_sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    semanticStateColumns.map assignment =
      accumulatorStateColumns.map assignment := by
  have equal := EqualityPins.sound semanticAccumulatorRowsIncluded
    canonical one satisfies
  have lane0 := equal
    (semanticStateColumns.getD 0 0, accumulatorStateColumns.getD 0 0)
    (by native_decide)
  have lane1 := equal
    (semanticStateColumns.getD 1 0, accumulatorStateColumns.getD 1 0)
    (by native_decide)
  have lane2 := equal
    (semanticStateColumns.getD 2 0, accumulatorStateColumns.getD 2 0)
    (by native_decide)
  have lane3 := equal
    (semanticStateColumns.getD 3 0, accumulatorStateColumns.getD 3 0)
    (by native_decide)
  simpa [semanticStateColumns, accumulatorStateColumns, stateOutColumns] using
    And.intro lane0 (And.intro lane1 (And.intro lane2 lane3))

/-- Output chosen by the generated program for a deterministic RHS. -/
def definitionOutputFor (rhs : Rhs) : Nat :=
  match (definitions instructions).find? fun definition =>
      decide (definition.rhs = rhs) with
  | some definition => definition.output
  | none => 0

def halfOutputColumn (columnMap : List Nat) (offset : Nat) : Nat :=
  definitionOutputFor (.linear
    (Relabel.terms columnMap (CanonicalU64Halves.sourceTerms offset)))

def chunkLowHalfCol : Nat :=
  halfOutputColumn FPrimeFullHistoryCounterSound.chunkOutputCanonicalMap 0
def chunkHighHalfCol : Nat :=
  halfOutputColumn FPrimeFullHistoryCounterSound.chunkOutputCanonicalMap 32
def stepLowHalfCol : Nat :=
  halfOutputColumn FPrimeFullHistoryCounterSound.stepOutputCanonicalMap 0
def stepHighHalfCol : Nat :=
  halfOutputColumn FPrimeFullHistoryCounterSound.stepOutputCanonicalMap 32

def counterHalfDefinitions : List Definition :=
  [ CanonicalU64Halves.definition
      FPrimeFullHistoryCounterSound.chunkOutputCanonicalMap chunkLowHalfCol 0
  , CanonicalU64Halves.definition
      FPrimeFullHistoryCounterSound.chunkOutputCanonicalMap chunkHighHalfCol 32
  , CanonicalU64Halves.definition
      FPrimeFullHistoryCounterSound.stepOutputCanonicalMap stepLowHalfCol 0
  , CanonicalU64Halves.definition
      FPrimeFullHistoryCounterSound.stepOutputCanonicalMap stepHighHalfCol 32 ]

theorem counterHalfDefinitionsMember :
    ∀ definition ∈ counterHalfDefinitions,
      definition ∈ definitions instructions := by
  native_decide

structure CounterHalves (assignment : Nat → Nat) : Prop where
  chunk : assignment chunkLowHalfCol = 2 ∧ assignment chunkHighHalfCol = 0
  step : assignment stepLowHalfCol = 2 ∧ assignment stepHighHalfCol = 0

/-- For the exact `[1,1]` profile, adjacent-state pins plus the no-wrap
counter owner force both recursive output counters to two, and the output
owner forces their absorbed half-words to `(2,0)`. -/
theorem counterHalves_sound
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (counterSatisfies :
      Satisfies FPrimeFullHistoryCounterSound.globalRows assignment)
    (outputSatisfies : Satisfies rows assignment)
    (chunkInputOne :
      assignment FPrimeFullHistoryCounterSound.chunkInputCol = 1)
    (stepInputOne :
      assignment FPrimeFullHistoryCounterSound.stepInputCol = 1) :
    CounterHalves assignment := by
  have transition := FPrimeFullHistoryCounterSound.sound goldilocksPrime
    canonical one counterSatisfies
  have canonicalOutputs :=
    FPrimeFullHistoryCounterSound.outputCanonical_sound goldilocksPrime
      canonical one counterSatisfies
  have outputFacts := sound goldilocksPrime canonical one outputSatisfies
  have chunkTwo : assignment FPrimeFullHistoryCounterSound.chunkOutputCol = 2 := by
    rw [transition.1, chunkInputOne]
  have stepTwo : assignment FPrimeFullHistoryCounterSound.stepOutputCol = 2 := by
    rw [transition.2.1, stepInputOne]
  have chunkFieldTwo :
      Relabel.assignment FPrimeFullHistoryCounterSound.chunkOutputCanonicalMap
          assignment CanonicalU64.varCol = 2 := by
    simpa [Relabel.assignment, Relabel.column,
      FPrimeFullHistoryCounterSound.chunkOutputCanonicalMap,
      FPrimeFullHistoryCounterSound.globalizeMap,
      FPrimeFullHistoryCounterSound.chunkOutputCol,
      FPrimeFullHistoryCounterLocalSound.chunkOutputVarCol,
      FPrimeFullHistoryCounterLocalSound.canonicalChunkOutputMap,
      FPrimeFullHistoryCounterLocalSound.columnMap] using chunkTwo
  have stepFieldTwo :
      Relabel.assignment FPrimeFullHistoryCounterSound.stepOutputCanonicalMap
          assignment CanonicalU64.varCol = 2 := by
    simpa [Relabel.assignment, Relabel.column,
      FPrimeFullHistoryCounterSound.stepOutputCanonicalMap,
      FPrimeFullHistoryCounterSound.globalizeMap,
      FPrimeFullHistoryCounterSound.stepOutputCol,
      FPrimeFullHistoryCounterLocalSound.stepOutputVarCol,
      FPrimeFullHistoryCounterLocalSound.canonicalStepOutputMap,
      FPrimeFullHistoryCounterLocalSound.columnMap] using stepTwo
  refine ⟨?_, ?_⟩
  · exact CanonicalU64Halves.small_halves_sound chunkFieldTwo (by decide)
      canonicalOutputs.1
      (outputFacts.definitions _
        (counterHalfDefinitionsMember _ (by simp [counterHalfDefinitions])))
      (outputFacts.definitions _
        (counterHalfDefinitionsMember _ (by simp [counterHalfDefinitions])))
  · exact CanonicalU64Halves.small_halves_sound stepFieldTwo (by decide)
      canonicalOutputs.2
      (outputFacts.definitions _
        (counterHalfDefinitionsMember _ (by simp [counterHalfDefinitions])))
      (outputFacts.definitions _
        (counterHalfDefinitionsMember _ (by simp [counterHalfDefinitions])))

def programCounterLowHalfCol : Nat :=
  halfOutputColumn FPrimeFullHistoryPriorLinkSound.priorProgramCounterMap 0

def programCounterHighHalfCol : Nat :=
  halfOutputColumn FPrimeFullHistoryPriorLinkSound.priorProgramCounterMap 32

def programCounterHalfDefinitions : List Definition :=
  [ CanonicalU64Halves.definition
      FPrimeFullHistoryPriorLinkSound.priorProgramCounterMap
        programCounterLowHalfCol 0
  , CanonicalU64Halves.definition
      FPrimeFullHistoryPriorLinkSound.priorProgramCounterMap
        programCounterHighHalfCol 32 ]

theorem programCounterHalfDefinitionsMember :
    ∀ definition ∈ programCounterHalfDefinitions,
      definition ∈ definitions instructions := by
  native_decide

/-- The shared program counter is one, so the recursive output hash absorbs
the exact half-word pair `(1,0)`. -/
theorem programCounterHalves_sound {assignment : Nat → Nat}
    (outputFacts : Facts assignment)
    (priorFacts : FPrimeFullHistoryPriorLinkSound.Facts assignment)
    (programCounterOne :
      assignment (FPrimeFullHistoryPriorLink.stateInColumns.getD 18 0) = 1) :
    assignment programCounterLowHalfCol = 1 ∧
      assignment programCounterHighHalfCol = 0 := by
  apply CanonicalU64Halves.one_halves_sound
    (columnMap := FPrimeFullHistoryPriorLinkSound.priorProgramCounterMap)
  · simpa [Relabel.assignment, Relabel.column,
      FPrimeFullHistoryPriorLinkSound.priorProgramCounterMap] using
      programCounterOne
  · exact priorFacts.canonicalU64
      FPrimeFullHistoryPriorLinkSound.priorProgramCounterMap
      (by native_decide)
  · exact outputFacts.definitions _
      (programCounterHalfDefinitionsMember _
        (by simp [programCounterHalfDefinitions]))
  · exact outputFacts.definitions _
      (programCounterHalfDefinitionsMember _
        (by simp [programCounterHalfDefinitions]))

def tagColumn : Nat := definitionOutputFor (.linear [(0, 1313210370)])

def tagDefinition : Definition :=
  ⟨tagColumn, .linear [(0, 1313210370)]⟩

theorem tagDefinitionMember :
    tagDefinition ∈ definitions instructions := by
  native_decide

theorem tag_sound {assignment : Nat → Nat}
    (facts : Facts assignment) (one : assignment 0 = 1) :
    assignment tagColumn = 1313210370 := by
  have holds := facts.definitions tagDefinition tagDefinitionMember
  simpa [tagDefinition, Definition.Holds, Rhs.eval, lcEval, one,
    goldilocksP] using holds

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound
