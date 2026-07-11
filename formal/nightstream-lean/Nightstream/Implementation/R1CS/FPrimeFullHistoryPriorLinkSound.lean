import Nightstream.Implementation.R1CS.FPrimeEncodingSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLinkPoseidonHashes
import Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound
import Nightstream.Implementation.R1CS.CanonicalU64Halves
import Nightstream.Implementation.R1CS.Relabel

/-!
Contract: artifact-level soundness of the generated recursive consumer for
the base step's delayed public link.

The proof derives the recomputed prior `x_out`, its canonical 256-bit
encoding, the affine-one public slot, and all 256 fresh-public equalities from
the exact 5,232 owner rows. No digest or link conclusion is carried by the
artifact.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLinkSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLink
open Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound

set_option maxRecDepth 524288
set_option maxHeartbeats 5000000

abbrev Pulled (assignment : Nat → Nat) : Nat → Nat :=
  Relabel.assignment encodingColumnMap assignment

theorem encodingRowsIncluded :
    rowsIncluded
      (FPrimeEncoding.rows.map (Relabel.row encodingColumnMap)) rows = true := by
  native_decide

theorem encodingMapsOne :
    Relabel.column encodingColumnMap 0 = 0 := by
  native_decide

theorem freshBitRowsIncluded :
    rowsIncluded (EqualityPins.rows freshBitPairs) rows = true := by
  native_decide

def onePins : List (Nat × Nat) := [freshOnePin]

theorem onePinsCanonical : ConstantPins.ValuesCanonical onePins := by
  native_decide

theorem freshOneRowsIncluded :
    rowsIncluded (ConstantPins.rows onePins) rows = true := by
  native_decide

theorem constantValuesCanonical :
    ConstantPins.ValuesCanonical constantPins := by
  native_decide

theorem constantRowsIncluded :
    rowsIncluded (ConstantPins.rows constantPins) rows = true := by
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
  constants :
    ∀ pin ∈ constantPins, assignment pin.1 = pin.2
  canonicalU64 :
    ∀ columnMap ∈ canonicalU64Maps,
      Relabel.assignment columnMap assignment CanonicalU64.varCol =
          bitsValue (Relabel.assignment columnMap assignment) ∧
        bitsValue (Relabel.assignment columnMap assignment) < goldilocksP
  halfDefinitions :
    ∀ definition ∈ canonicalU64HalfDefinitions,
      definition.Holds assignment
  encoding : FPrimeEncodingSound.Holds (Pulled assignment)
  freshBits :
    ∀ pair ∈ freshBitPairs, assignment pair.1 = assignment pair.2
  freshOne : assignment freshOnePin.1 = 1
  sponge :
    ∀ lane, lane < 4 →
      assignment
          (FPrimeFullHistoryPriorLinkPoseidonHashes.priorXOutTrace.outputColumns.getD
            lane 0) =
        Poseidon2Sponge.runValueRounds
          FPrimeFullHistoryPriorLinkPoseidonHashes.priorXOutTrace.rounds
          (FPrimeFullHistoryPriorLinkPoseidonHashes.priorXOutTrace.inputColumns.map
            assignment)
          (fun _ => 0) lane

theorem sound (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Facts assignment := by
  have definitionFacts :=
    definitionsHold_of_satisfies definitions_canonical canonical one satisfies
  refine {
    constants := ConstantPins.sound constantValuesCanonical
      constantRowsIncluded canonical one satisfies
    canonicalU64 := ?_
    halfDefinitions := ?_
    encoding := ?_
    freshBits := EqualityPins.sound freshBitRowsIncluded canonical one satisfies
    freshOne := ?_
    sponge := ?_
  }
  · intro columnMap member
    exact canonicalU64_sound goldilocksPrime
      (Relabel.canonical canonical)
      (Relabel.constantOne (canonicalU64MapsOne columnMap member) one)
      (Relabel.satisfies_of_included
        (canonicalU64RowsIncluded columnMap member) satisfies)
  · intro definition member
    exact definitionFacts definition (halfDefinitionsMember definition member)
  · exact FPrimeEncodingSound.fPrimeEncoding_sound goldilocksPrime
      (Relabel.canonical canonical)
      (Relabel.constantOne encodingMapsOne one)
      (Relabel.satisfies_of_included encodingRowsIncluded satisfies)
  · have pins := ConstantPins.sound onePinsCanonical freshOneRowsIncluded
      canonical one satisfies
    exact pins freshOnePin (by simp [onePins])
  · intro lane laneLt
    exact Poseidon2Sponge.trace_values_sound
      (FPrimeFullHistoryPriorLinkPoseidonHashes.traces_valid
        FPrimeFullHistoryPriorLinkPoseidonHashes.priorXOutTrace
        (by native_decide))
      canonical one satisfies lane laneLt

def priorChunkCountMap : List Nat := canonicalU64Maps[0]!
def priorStepCountMap : List Nat := canonicalU64Maps[1]!
def priorProgramCounterMap : List Nat := canonicalU64Maps[2]!

def defaultHalfDefinition : Definition := ⟨0, .linear []⟩

def halfOutputColumn (index : Nat) : Nat :=
  (canonicalU64HalfDefinitions.getD index defaultHalfDefinition).output

def priorChunkLowHalfCol : Nat := halfOutputColumn 0
def priorChunkHighHalfCol : Nat := halfOutputColumn 1
def priorStepLowHalfCol : Nat := halfOutputColumn 2
def priorStepHighHalfCol : Nat := halfOutputColumn 3
def priorProgramCounterLowHalfCol : Nat := halfOutputColumn 4
def priorProgramCounterHighHalfCol : Nat := halfOutputColumn 5

structure CounterHalves (assignment : Nat → Nat) : Prop where
  chunk : assignment priorChunkLowHalfCol = 1 ∧
    assignment priorChunkHighHalfCol = 0
  step : assignment priorStepLowHalfCol = 1 ∧
    assignment priorStepHighHalfCol = 0
  programCounter : assignment priorProgramCounterLowHalfCol = 1 ∧
    assignment priorProgramCounterHighHalfCol = 0

theorem counterHalves_sound {assignment : Nat → Nat}
    (facts : Facts assignment)
    (chunkOne : assignment (stateInColumns.getD 8 0) = 1)
    (stepOne : assignment (stateInColumns.getD 9 0) = 1)
    (programCounterOne : assignment (stateInColumns.getD 18 0) = 1) :
    CounterHalves assignment := by
  refine ⟨?_, ?_, ?_⟩
  · apply CanonicalU64Halves.one_halves_sound
      (columnMap := priorChunkCountMap)
    · exact chunkOne
    · exact facts.canonicalU64 priorChunkCountMap (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition priorChunkCountMap priorChunkLowHalfCol 0)
        (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition priorChunkCountMap priorChunkHighHalfCol 32)
        (by native_decide)
  · apply CanonicalU64Halves.one_halves_sound
      (columnMap := priorStepCountMap)
    · exact stepOne
    · exact facts.canonicalU64 priorStepCountMap (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition priorStepCountMap priorStepLowHalfCol 0)
        (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition priorStepCountMap priorStepHighHalfCol 32)
        (by native_decide)
  · apply CanonicalU64Halves.one_halves_sound
      (columnMap := priorProgramCounterMap)
    · exact programCounterOne
    · exact facts.canonicalU64 priorProgramCounterMap (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition priorProgramCounterMap
          priorProgramCounterLowHalfCol 0)
        (by native_decide)
    · exact facts.halfDefinitions
        (CanonicalU64Halves.definition priorProgramCounterMap
          priorProgramCounterHighHalfCol 32)
        (by native_decide)

def encodedBitColumn (lane : Fin 4) (bit : Fin 64) : Nat :=
  Relabel.column encodingColumnMap
    (FPrimeEncoding.publicBitCol lane.val bit.val)

def freshBitColumn (lane : Fin 4) (bit : Fin 64) : Nat :=
  freshPublicColumns.getD (1 + lane.val * 64 + bit.val) 0

theorem bitPairCensus :
    ∀ lane : Fin 4, ∀ bit : Fin 64,
      (freshBitColumn lane bit, encodedBitColumn lane bit) ∈ freshBitPairs := by
  native_decide

theorem digestColumnMap :
    ∀ lane : Fin 4,
      Relabel.column encodingColumnMap (FPrimeEncoding.digestCol lane.val) =
        digestColumns.getD lane.val 0 := by
  native_decide

private theorem foldl_bits_congr (xs : List Nat) (left right : Nat → Nat)
    (equal : ∀ bit ∈ xs, left bit = right bit) (initial : Nat) :
    xs.foldl (fun total bit => total + 2 ^ bit * left bit) initial =
      xs.foldl (fun total bit => total + 2 ^ bit * right bit) initial := by
  induction xs generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [equal head (by simp)]
      apply inductionHypothesis
      intro bit member
      exact equal bit (by simp [member])

def freshLaneBitsValue (assignment : Nat → Nat) (lane : Nat) : Nat :=
  (List.range 64).foldl
    (fun total bit => total + 2 ^ bit *
      assignment (freshPublicColumns.getD (1 + lane * 64 + bit) 0)) 0

def freshDigest (assignment : Nat → Nat) : List Nat :=
  (List.range 4).map fun lane =>
    freshLaneBitsValue assignment lane

def decodedFresh (assignment : Nat → Nat) : Fresh :=
  { publicXOut := freshDigest assignment }

theorem freshLane_eq_digest {assignment : Nat → Nat}
    (facts : Facts assignment) (lane : Fin 4) :
    freshLaneBitsValue assignment lane.val =
      assignment (digestColumns.getD lane.val 0) := by
  have canonical := (facts.encoding.laneCanonical lane.val lane.isLt).1
  change assignment
      (Relabel.column encodingColumnMap
        (FPrimeEncoding.digestCol lane.val)) = _ at canonical
  rw [digestColumnMap lane] at canonical
  rw [canonical]
  unfold FPrimeEncodingSound.laneBitsValue freshLaneBitsValue
  apply foldl_bits_congr
  intro bit member
  let bitFin : Fin 64 := ⟨bit, List.mem_range.mp member⟩
  change assignment (freshBitColumn lane bitFin) =
    assignment (encodedBitColumn lane bitFin)
  exact facts.freshBits _ (bitPairCensus lane bitFin)

private theorem rangeFour : List.range 4 = [0, 1, 2, 3] := by decide

theorem decodedFresh_digest {assignment : Nat → Nat}
    (facts : Facts assignment) :
    (decodedFresh assignment).publicXOut = digestColumns.map assignment := by
  have lane0 := freshLane_eq_digest facts (0 : Fin 4)
  have lane1 := freshLane_eq_digest facts (1 : Fin 4)
  have lane2 := freshLane_eq_digest facts (2 : Fin 4)
  have lane3 := freshLane_eq_digest facts (3 : Fin 4)
  simpa [decodedFresh, freshDigest, freshLaneBitsValue, digestColumns,
    rangeFour] using And.intro lane0 (And.intro lane1 (And.intro lane2 lane3))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLinkSound
