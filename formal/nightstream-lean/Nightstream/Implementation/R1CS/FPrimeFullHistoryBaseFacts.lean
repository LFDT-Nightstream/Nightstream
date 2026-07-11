import Nightstream.Implementation.R1CS.CanonicalU64Sound
import Nightstream.Implementation.R1CS.CanonicalU64Halves
import Nightstream.Implementation.R1CS.ConstantPins
import Nightstream.Implementation.R1CS.EqualityPins
import Nightstream.Implementation.R1CS.FPrimeFullHistoryBasePoseidonHashes
import Nightstream.Implementation.R1CS.Relabel

/-!
Contract: semantic facts extracted from every satisfying assignment of the
exact full-history base owner.

Each certificate below is structural: literal row inclusion, a column
renaming, or a sponge wire schedule.  The exported artifact carries no
acceptance or high-level F' conclusion.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseFacts

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryBase

set_option maxRecDepth 524288
set_option maxHeartbeats 5000000

theorem stateInValuesCanonical :
    ConstantPins.ValuesCanonical stateInPins := by
  native_decide

theorem stateInRowsIncluded :
    rowsIncluded (ConstantPins.rows stateInPins) rows = true := by
  native_decide

theorem constantValuesCanonical :
    ConstantPins.ValuesCanonical constantPins := by
  native_decide

theorem constantRowsIncluded :
    rowsIncluded (ConstantPins.rows constantPins) rows = true := by
  native_decide

theorem chunkDigestRowsIncluded :
    rowsIncluded (EqualityPins.rows chunkDigestPairs) rows = true := by
  native_decide

theorem semanticAccumulatorRowsIncluded :
    rowsIncluded (EqualityPins.rows semanticAccumulatorPairs) rows = true := by
  native_decide

theorem canonicalU64RowsIncluded :
    ∀ columnMap ∈ canonicalU64Maps,
      rowsIncluded
        (CanonicalU64.rows.map (Relabel.row columnMap)) rows = true := by
  native_decide

theorem canonicalU64MapsOne :
    ∀ columnMap ∈ canonicalU64Maps,
      Relabel.column columnMap 0 = 0 := by
  native_decide

theorem halfDefinitionsMember :
    ∀ definition ∈ canonicalU64HalfDefinitions,
      definition ∈ definitions instructions := by
  native_decide

structure Facts (assignment : Nat → Nat) : Prop where
  stateIn :
    ∀ pin ∈ stateInPins, assignment pin.1 = pin.2
  constants :
    ∀ pin ∈ constantPins, assignment pin.1 = pin.2
  chunkDigest :
    ∀ pair ∈ chunkDigestPairs, assignment pair.1 = assignment pair.2
  semanticAccumulator :
    ∀ pair ∈ semanticAccumulatorPairs,
      assignment pair.1 = assignment pair.2
  canonicalU64 :
    ∀ columnMap ∈ canonicalU64Maps,
      Relabel.assignment columnMap assignment CanonicalU64.varCol =
          bitsValue (Relabel.assignment columnMap assignment) ∧
        bitsValue (Relabel.assignment columnMap assignment) < goldilocksP
  halfDefinitions :
    ∀ definition ∈ canonicalU64HalfDefinitions,
      definition.Holds assignment
  sponge :
    ∀ trace ∈ FPrimeFullHistoryBasePoseidonHashes.traces,
      ∀ lane, lane < 4 →
        assignment (trace.outputColumns.getD lane 0) =
          Poseidon2Sponge.runValueRounds trace.rounds
            (trace.inputColumns.map assignment) (fun _ => 0) lane

/-- Universal extraction theorem for the exact base-owner rows. -/
theorem sound (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Facts assignment := by
  have definitionFacts :=
    definitionsHold_of_satisfies definitions_canonical canonical one satisfies
  refine {
    stateIn := ConstantPins.sound stateInValuesCanonical stateInRowsIncluded
      canonical one satisfies
    constants := ConstantPins.sound constantValuesCanonical constantRowsIncluded
      canonical one satisfies
    chunkDigest := EqualityPins.sound chunkDigestRowsIncluded canonical one
      satisfies
    semanticAccumulator := EqualityPins.sound
      semanticAccumulatorRowsIncluded canonical one satisfies
    canonicalU64 := ?_
    halfDefinitions := ?_
    sponge := ?_
  }
  · intro columnMap member
    apply canonicalU64_sound goldilocksPrime
    · exact Relabel.canonical canonical
    · exact Relabel.constantOne
        (canonicalU64MapsOne columnMap member) one
    · exact Relabel.satisfies_of_included
        (canonicalU64RowsIncluded columnMap member) satisfies
  · intro definition member
    exact definitionFacts definition (halfDefinitionsMember definition member)
  · intro trace member lane laneLt
    exact Poseidon2Sponge.trace_values_sound
      (FPrimeFullHistoryBasePoseidonHashes.traces_valid trace member)
      canonical one satisfies lane laneLt

def chunkCountMap : List Nat := canonicalU64Maps[0]!
def stepCountMap : List Nat := canonicalU64Maps[1]!
def programCounterMap : List Nat := canonicalU64Maps[2]!

structure CounterHalves (assignment : Nat → Nat) : Prop where
  chunk : assignment 6272 = 1 ∧ assignment 6273 = 0
  step : assignment 6274 = 1 ∧ assignment 6275 = 0
  programCounter : assignment 6342 = 1 ∧ assignment 6343 = 0

theorem counterHalves_sound {assignment : Nat → Nat}
    (facts : Facts assignment) : CounterHalves assignment := by
  refine ⟨?_, ?_, ?_⟩
  · apply CanonicalU64Halves.one_halves_sound
      (columnMap := chunkCountMap)
    · change assignment 5947 = 1
      exact facts.constants (5947, 1) (by native_decide)
    · exact facts.canonicalU64 chunkCountMap (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition chunkCountMap 6272 0)
        (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition chunkCountMap 6273 32)
        (by native_decide)
  · apply CanonicalU64Halves.one_halves_sound
      (columnMap := stepCountMap)
    · change assignment 5948 = 1
      exact facts.constants (5948, 1) (by native_decide)
    · exact facts.canonicalU64 stepCountMap (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition stepCountMap 6274 0)
        (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition stepCountMap 6275 32)
        (by native_decide)
  · apply CanonicalU64Halves.one_halves_sound
      (columnMap := programCounterMap)
    · change assignment 1 = 1
      exact facts.stateIn (1, 1) (by native_decide)
    · exact facts.canonicalU64 programCounterMap (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition programCounterMap 6342 0)
        (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition programCounterMap 6343 32)
        (by native_decide)

def claimTrace : Poseidon2Sponge.Trace :=
  FPrimeFullHistoryBasePoseidonHashes.traces[0]!

def chunkTrace : Poseidon2Sponge.Trace :=
  FPrimeFullHistoryBasePoseidonHashes.traces[1]!

def xOutTrace : Poseidon2Sponge.Trace :=
  FPrimeFullHistoryBasePoseidonHashes.traces[2]!

theorem claimTrace_outputColumns :
    claimTrace.outputColumns = [2452, 2453, 2454, 2455] := by
  native_decide

theorem chunkTrace_outputColumns :
    chunkTrace.outputColumns = [5477, 5478, 5479, 5480] := by
  native_decide

theorem xOutTrace_inputColumns :
    xOutTrace.inputColumns =
      [6271, 2, 3, 4, 5, 6, 7, 8, 9, 6272, 6273, 6274, 6275,
       6342, 6343, 32, 33, 34, 35, 5939, 5940, 5941, 5942] := by
  native_decide

theorem xOutTrace_outputColumns :
    xOutTrace.outputColumns = [10561, 10562, 10563, 10564] := by
  native_decide

def outputValue (trace : Poseidon2Sponge.Trace) (lane : Nat) : Nat :=
  Poseidon2Sponge.runValueRounds trace.rounds
    (trace.inputColumns.map (ConstantPins.lookup constantPins))
    (fun _ => 0) lane

def traceOutputValues (trace : Poseidon2Sponge.Trace)
    (inputValues : List Nat) : List Nat :=
  (List.range 4).map fun lane =>
    Poseidon2Sponge.runValueRounds trace.rounds inputValues (fun _ => 0) lane

def traceOutputKeys (trace : Poseidon2Sponge.Trace) : List Nat :=
  (List.range 4).map fun lane => trace.outputColumns.getD lane 0

private theorem rangeFour : List.range 4 = [0, 1, 2, 3] := by
  decide

/-- Output pins parameterized by the value list actually used for a trace. -/
def traceOutputPins (trace : Poseidon2Sponge.Trace)
    (inputValues : List Nat) : List (Nat × Nat) :=
  (List.range 4).map fun lane =>
    (trace.outputColumns.getD lane 0,
      Poseidon2Sponge.runValueRounds trace.rounds inputValues (fun _ => 0) lane)

theorem traceOutputPins_keys (trace : Poseidon2Sponge.Trace)
    (inputValues : List Nat) :
    ConstantPins.keys (traceOutputPins trace inputValues) =
      traceOutputKeys trace := by
  simp [ConstantPins.keys, traceOutputPins, traceOutputKeys, List.map_map,
    Function.comp_def]

theorem traceOutputPins_sound {assignment : Nat → Nat}
    (facts : Facts assignment)
    (trace : Poseidon2Sponge.Trace)
    (traceMember : trace ∈ FPrimeFullHistoryBasePoseidonHashes.traces)
    (inputValues : List Nat)
    (inputs : trace.inputColumns.map assignment = inputValues) :
    ∀ pin ∈ traceOutputPins trace inputValues,
      assignment pin.1 = pin.2 := by
  intro pin member
  rcases List.mem_map.mp member with ⟨lane, laneMember, pinEq⟩
  subst pin
  have laneLt : lane < 4 := List.mem_range.mp laneMember
  simpa [inputs] using facts.sponge trace traceMember lane laneLt

def claimInputValues : List Nat :=
  claimTrace.inputColumns.map (ConstantPins.lookup constantPins)

def claimOutputPins : List (Nat × Nat) :=
  traceOutputPins claimTrace claimInputValues

theorem claimInputsCovered :
    ConstantPins.Covers claimTrace.inputColumns constantPins := by
  native_decide

theorem claimInputValues_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    claimTrace.inputColumns.map assignment = claimInputValues := by
  exact ConstantPins.map_assignment_eq_lookup facts.constants claimInputsCovered

theorem claimOutputPins_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    ∀ pin ∈ claimOutputPins, assignment pin.1 = pin.2 := by
  apply traceOutputPins_sound facts claimTrace
  · native_decide
  · exact claimInputValues_sound facts

def chunkKnownPins : List (Nat × Nat) :=
  constantPins ++ stateInPins ++ claimOutputPins

def chunkInputValues : List Nat :=
  chunkTrace.inputColumns.map (ConstantPins.lookup chunkKnownPins)

def chunkOutputPins : List (Nat × Nat) :=
  traceOutputPins chunkTrace chunkInputValues

theorem chunkOutput_lookup_0 :
    ConstantPins.lookup chunkOutputPins 5477 =
      Poseidon2Sponge.runValueRounds chunkTrace.rounds chunkInputValues
        (fun _ => 0) 0 := by
  simp [chunkOutputPins, traceOutputPins, chunkTrace_outputColumns,
    rangeFour, ConstantPins.lookup]

theorem chunkOutput_lookup_1 :
    ConstantPins.lookup chunkOutputPins 5478 =
      Poseidon2Sponge.runValueRounds chunkTrace.rounds chunkInputValues
        (fun _ => 0) 1 := by
  simp [chunkOutputPins, traceOutputPins, chunkTrace_outputColumns,
    rangeFour, ConstantPins.lookup]

theorem chunkOutput_lookup_2 :
    ConstantPins.lookup chunkOutputPins 5479 =
      Poseidon2Sponge.runValueRounds chunkTrace.rounds chunkInputValues
        (fun _ => 0) 2 := by
  simp [chunkOutputPins, traceOutputPins, chunkTrace_outputColumns,
    rangeFour, ConstantPins.lookup]

theorem chunkOutput_lookup_3 :
    ConstantPins.lookup chunkOutputPins 5480 =
      Poseidon2Sponge.runValueRounds chunkTrace.rounds chunkInputValues
        (fun _ => 0) 3 := by
  simp [chunkOutputPins, traceOutputPins, chunkTrace_outputColumns,
    rangeFour, ConstantPins.lookup]

def chunkKnownKeys : List Nat :=
  ConstantPins.keys constantPins ++ ConstantPins.keys stateInPins ++
    traceOutputKeys claimTrace

theorem chunkInputKeysCovered :
    ConstantPins.KeysCover chunkTrace.inputColumns chunkKnownKeys := by
  native_decide

theorem chunkInputsCovered :
    ConstantPins.Covers chunkTrace.inputColumns chunkKnownPins := by
  rw [ConstantPins.covers_iff_keys]
  have keysEq : ConstantPins.keys chunkKnownPins = chunkKnownKeys := by
    rw [chunkKnownPins, chunkKnownKeys]
    simp only [ConstantPins.keys, List.map_append, claimOutputPins]
    have claimKeys := traceOutputPins_keys claimTrace claimInputValues
    change List.map Prod.fst (traceOutputPins claimTrace claimInputValues) =
      traceOutputKeys claimTrace at claimKeys
    rw [claimKeys]
  rw [keysEq]
  exact chunkInputKeysCovered

theorem chunkKnownPins_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    ∀ pin ∈ chunkKnownPins, assignment pin.1 = pin.2 := by
  intro pin member
  simp only [chunkKnownPins, List.mem_append] at member
  rcases member with (constant | state) | claim
  · exact facts.constants pin constant
  · exact facts.stateIn pin state
  · exact claimOutputPins_sound facts pin claim

theorem chunkInputValues_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    chunkTrace.inputColumns.map assignment = chunkInputValues := by
  exact ConstantPins.map_assignment_eq_lookup
    (chunkKnownPins_sound facts) chunkInputsCovered

theorem chunkOutputPins_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    ∀ pin ∈ chunkOutputPins, assignment pin.1 = pin.2 := by
  apply traceOutputPins_sound facts chunkTrace
  · native_decide
  · exact chunkInputValues_sound facts

def chunkStatePins : List (Nat × Nat) :=
  EqualityPins.transferPins chunkDigestPairs chunkOutputPins

def semanticStatePins : List (Nat × Nat) :=
  EqualityPins.transferPins semanticAccumulatorPairs constantPins

def counterHalfPins : List (Nat × Nat) :=
  [(6272, 1), (6273, 0), (6274, 1), (6275, 0), (6342, 1), (6343, 0)]

theorem chunkOutputSourcesCovered :
    EqualityPins.SourcesCovered chunkDigestPairs chunkOutputPins := by
  rw [EqualityPins.sourcesCovered_iff_keys]
  rw [chunkOutputPins, traceOutputPins_keys]
  native_decide

theorem semanticSourcesCovered :
    EqualityPins.SourcesCovered semanticAccumulatorPairs constantPins := by
  native_decide

theorem chunkStatePins_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    ∀ pin ∈ chunkStatePins, assignment pin.1 = pin.2 :=
  EqualityPins.transfer_sound facts.chunkDigest
    (chunkOutputPins_sound facts) chunkOutputSourcesCovered

theorem semanticStatePins_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    ∀ pin ∈ semanticStatePins, assignment pin.1 = pin.2 :=
  EqualityPins.transfer_sound facts.semanticAccumulator facts.constants
    semanticSourcesCovered

theorem counterHalfPins_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    ∀ pin ∈ counterHalfPins, assignment pin.1 = pin.2 := by
  have halves := counterHalves_sound facts
  intro pin member
  simp [counterHalfPins] at member
  rcases member with h | h | h | h | h | h <;> subst pin
  · exact halves.chunk.1
  · exact halves.chunk.2
  · exact halves.step.1
  · exact halves.step.2
  · exact halves.programCounter.1
  · exact halves.programCounter.2

def xOutKnownPins : List (Nat × Nat) :=
  constantPins ++ stateInPins ++ chunkStatePins ++ semanticStatePins ++
    counterHalfPins

def xOutInputValues : List Nat :=
  xOutTrace.inputColumns.map (ConstantPins.lookup xOutKnownPins)

def xOutOutputPins : List (Nat × Nat) :=
  traceOutputPins xOutTrace xOutInputValues

def xOutKnownKeys : List Nat :=
  ConstantPins.keys constantPins ++ ConstantPins.keys stateInPins ++
    chunkDigestPairs.map Prod.fst ++ semanticAccumulatorPairs.map Prod.fst ++
    ConstantPins.keys counterHalfPins

theorem xOutKnownPins_keys :
    ConstantPins.keys xOutKnownPins = xOutKnownKeys := by
  rw [xOutKnownPins, xOutKnownKeys]
  simp only [ConstantPins.keys, List.map_append, chunkStatePins,
    semanticStatePins]
  have chunkKeys := EqualityPins.transferPins_keys
    chunkDigestPairs chunkOutputPins
  have semanticKeys := EqualityPins.transferPins_keys
    semanticAccumulatorPairs constantPins
  change List.map Prod.fst
    (EqualityPins.transferPins chunkDigestPairs chunkOutputPins) =
      chunkDigestPairs.map Prod.fst at chunkKeys
  change List.map Prod.fst
    (EqualityPins.transferPins semanticAccumulatorPairs constantPins) =
      semanticAccumulatorPairs.map Prod.fst at semanticKeys
  rw [chunkKeys, semanticKeys]

theorem xOutInputKeysCovered :
    ConstantPins.KeysCover xOutTrace.inputColumns xOutKnownKeys := by
  native_decide

theorem xOutInputsCovered :
    ConstantPins.Covers xOutTrace.inputColumns xOutKnownPins := by
  rw [ConstantPins.covers_iff_keys]
  rw [xOutKnownPins_keys]
  exact xOutInputKeysCovered

theorem xOutKnownPins_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    ∀ pin ∈ xOutKnownPins, assignment pin.1 = pin.2 := by
  intro pin member
  simp only [xOutKnownPins, List.mem_append] at member
  rcases member with (((constant | state) | chunk) | semantic) | half
  · exact facts.constants pin constant
  · exact facts.stateIn pin state
  · exact chunkStatePins_sound facts pin chunk
  · exact semanticStatePins_sound facts pin semantic
  · exact counterHalfPins_sound facts pin half

theorem xOutInputValues_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    xOutTrace.inputColumns.map assignment = xOutInputValues := by
  exact ConstantPins.map_assignment_eq_lookup
    (xOutKnownPins_sound facts) xOutInputsCovered

theorem xOutOutputPins_sound {assignment : Nat → Nat}
    (facts : Facts assignment) :
    ∀ pin ∈ xOutOutputPins, assignment pin.1 = pin.2 := by
  apply traceOutputPins_sound facts xOutTrace
  · native_decide
  · exact xOutInputValues_sound facts

end Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseFacts
