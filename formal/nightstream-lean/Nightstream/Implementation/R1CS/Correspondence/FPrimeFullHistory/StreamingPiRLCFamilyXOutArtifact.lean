import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyDigest
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.TranscriptMachineDuplex

/-!
Contract: exact full-`XOut` Poseidon2 execution for one PiRLC family arm.

Owns the zero row, 32 additive absorb rows, padding row, nine connected
production Poseidon2 calls, exact four-field chunking, and the four physical
digest outputs for both state sides and parity arms.

Does not own interpretation of the 32 preimage fields, collision resistance,
local family-state digest semantics, public bit decomposition, adjacent-arm
continuity, selective lowering, or recursive lifecycle integration.

Assurance tier: artifact-checked for property
`FPRIME-STREAMING-PIRLC-FAMILY-XOUT-SPONGE-V1`.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.CallRefinement
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachineDuplex
open Nightstream.Implementation.R1CS.Program

def xOutHashFor (kind : ArmKind) : StateSide -> RawHash
  | .after => (armFor kind).afterXOutHash
  | .before => (armFor kind).beforeXOutHash

def roundAt (kind : ArmKind) (side : StateSide) (round : Nat) : RawHashRound :=
  (xOutHashFor kind side).rounds.getD round default

def callAt (kind : ArmKind) (side : StateSide) (round : Nat) :
    Poseidon2Call.Call :=
  (armFor kind).poseidon2Calls.getD
    ((xOutHashFor kind side).permutationCallStart + round) default

def stateBeforeColumn
    (kind : ArmKind) (side : StateSide) (round lane : Nat) : Nat :=
  (roundAt kind side round).stateBeforeColumns.getD lane 0

def permutationInputColumn
    (kind : ArmKind) (side : StateSide) (round lane : Nat) : Nat :=
  (roundAt kind side round).permutationInputColumns.getD lane 0

def permutationOutputColumn
    (kind : ArmKind) (side : StateSide) (round lane : Nat) : Nat :=
  (roundAt kind side round).permutationOutputColumns.getD lane 0

def chunkColumn
    (kind : ArmKind) (side : StateSide) (round lane : Nat) : Nat :=
  (roundAt kind side round).chunkColumns.getD lane 0

def lowLane (lane : Fin 4) : Fin 8 :=
  ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩

def highLane (lane : Fin 4) : Fin 8 :=
  ⟨4 + lane.val, by omega⟩

def padTailLane (lane : Fin 7) : Fin 8 :=
  ⟨1 + lane.val, by omega⟩

structure ExactTrace (kind : ArmKind) (side : StateSide) : Prop where
  roundCount : (xOutHashFor kind side).rounds.length = 9
  callBound : forall round : Fin 9,
    (xOutHashFor kind side).permutationCallStart + round.val <
      (armFor kind).poseidon2Calls.length
  callInput : forall round : Fin 9, forall lane : Fin 8,
    (callAt kind side round.val).columnMap (lane.val + 1) =
      permutationInputColumn kind side round.val lane.val
  callOutput : forall round : Fin 9, forall lane : Fin 8,
    (callAt kind side round.val).columnMap (601 + lane.val) =
      permutationOutputColumn kind side round.val lane.val
  startState : forall lane : Fin 8,
    stateBeforeColumn kind side 0 lane.val = (xOutHashFor kind side).zeroColumn
  nextState : forall round : Fin 8, forall lane : Fin 8,
    stateBeforeColumn kind side (round.val + 1) lane.val =
      permutationOutputColumn kind side round.val lane.val
  dataChunk : forall round : Fin 8, forall lane : Fin 4,
    chunkColumn kind side round.val lane.val =
      (xOutHashFor kind side).inputColumns.getD
        (4 * round.val + lane.val) 0
  dataCapacity : forall round : Fin 8, forall lane : Fin 4,
    permutationInputColumn kind side round.val (highLane lane).val =
      stateBeforeColumn kind side round.val (highLane lane).val
  padTail : forall lane : Fin 7,
    permutationInputColumn kind side 8 (padTailLane lane).val =
      stateBeforeColumn kind side 8 (padTailLane lane).val
  digestOutput : forall lane : Fin 4,
    xOutDigestColumn kind side lane =
      permutationOutputColumn kind side 8 lane.val

theorem exact_trace (kind : ArmKind) (side : StateSide) :
    ExactTrace kind side := by
  cases kind <;> cases side <;>
    constructor <;> native_decide

def zeroDefinitionRow (kind : ArmKind) (side : StateSide) : Row :=
  builderLinearRow (xOutHashFor kind side).zeroColumn []

def absorbDefinitionRow
    (kind : ArmKind) (side : StateSide) (round lane : Nat) : Row :=
  builderLinearRow (permutationInputColumn kind side round lane)
    [(stateBeforeColumn kind side round lane, 1),
      (chunkColumn kind side round lane, 1)]

def padDefinitionRow (kind : ArmKind) (side : StateSide) : Row :=
  builderLinearRow (permutationInputColumn kind side 8 0)
    [(stateBeforeColumn kind side 8 0, 1), (0, 1)]

def absorbDefinitionRows (kind : ArmKind) (side : StateSide) : List Row :=
  (List.range 8).flatMap fun round =>
    (List.range 4).map fun lane =>
      absorbDefinitionRow kind side round lane

def rawHashDefinitionRows (kind : ArmKind) (side : StateSide) : List Row :=
  zeroDefinitionRow kind side ::
    (absorbDefinitionRows kind side ++ [padDefinitionRow kind side])

def hashDefinitionRows (kind : ArmKind) (side : StateSide) : List Row :=
  Poseidon2Normalized.normalizeProgram (rawHashDefinitionRows kind side)

private theorem definition_rows_in_glue
    (kind : ArmKind) (side : StateSide) :
    rowsIncluded (hashDefinitionRows kind side) (glueProgram kind) = true := by
  cases kind <;> cases side <;> native_decide

private theorem glue_satisfies
    (kind : ArmKind) (assignment : Nat -> Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (glueProgram kind) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds (armFor kind) assignment satisfied indexed indexedMember

private theorem definition_rows_satisfy
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (hashDefinitionRows kind side) assignment := by
  intro row member
  exact glue_satisfies kind assignment satisfied row
    (rowsIncluded_sound (definition_rows_in_glue kind side) row member)

private theorem raw_definition_rows_satisfy
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (rawHashDefinitionRows kind side) assignment :=
  (Poseidon2Normalized.satisfies_normalizeProgram
    (rawHashDefinitionRows kind side) assignment).mp
      (definition_rows_satisfy kind side assignment satisfied)

private theorem zero_definition_member
    (kind : ArmKind) (side : StateSide) :
    zeroDefinitionRow kind side ∈ rawHashDefinitionRows kind side := by
  simp [rawHashDefinitionRows]

private theorem absorb_definition_member
    (kind : ArmKind) (side : StateSide)
    (round : Fin 8) (lane : Fin 4) :
    absorbDefinitionRow kind side round.val lane.val ∈
      rawHashDefinitionRows kind side := by
  unfold rawHashDefinitionRows
  apply List.mem_cons_of_mem
  apply List.mem_append_left
  unfold absorbDefinitionRows
  apply List.mem_flatMap.mpr
  refine ⟨round.val, List.mem_range.mpr round.isLt, ?_⟩
  apply List.mem_map.mpr
  exact ⟨lane.val, List.mem_range.mpr lane.isLt, rfl⟩

private theorem pad_definition_member
    (kind : ArmKind) (side : StateSide) :
    padDefinitionRow kind side ∈ rawHashDefinitionRows kind side := by
  simp [rawHashDefinitionRows]

theorem zero_column_eq_zero
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    assignment (xOutHashFor kind side).zeroColumn = 0 := by
  have holds := raw_definition_rows_satisfy kind side assignment satisfied
    (zeroDefinitionRow kind side) (zero_definition_member kind side)
  have exact := builderLinearRow_sound canonical one
    (xOutHashFor kind side).zeroColumn [] (by simp [CanonicalTerms]) holds
  simpa [lcEval] using exact

theorem absorb_input_eq
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (round : Fin 8) (lane : Fin 4) :
    assignment (permutationInputColumn kind side round.val lane.val) =
      (assignment (stateBeforeColumn kind side round.val lane.val) +
        assignment (chunkColumn kind side round.val lane.val)) %
          goldilocksP := by
  have holds := raw_definition_rows_satisfy kind side assignment satisfied
    (absorbDefinitionRow kind side round.val lane.val)
    (absorb_definition_member kind side round lane)
  have exact := builderLinearRow_sound canonical one
    (permutationInputColumn kind side round.val lane.val)
    [(stateBeforeColumn kind side round.val lane.val, 1),
      (chunkColumn kind side round.val lane.val, 1)]
    (by simp [CanonicalTerms, goldilocksP]) holds
  simpa [lcEval] using exact

theorem pad_input_eq
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    assignment (permutationInputColumn kind side 8 0) =
      (assignment (stateBeforeColumn kind side 8 0) + 1) %
        goldilocksP := by
  have holds := raw_definition_rows_satisfy kind side assignment satisfied
    (padDefinitionRow kind side) (pad_definition_member kind side)
  have exact := builderLinearRow_sound canonical one
    (permutationInputColumn kind side 8 0)
    [(stateBeforeColumn kind side 8 0, 1), (0, 1)]
    (by simp [CanonicalTerms, goldilocksP]) holds
  simpa [lcEval, one] using exact

def xOutChunkAt
    (assignment : Nat -> Nat) (kind : ArmKind) (side : StateSide)
    (round : Nat) : List Nat :=
  if round < 8 then
    [assignment ((xOutHashFor kind side).inputColumns.getD (4 * round) 0),
      assignment ((xOutHashFor kind side).inputColumns.getD (4 * round + 1) 0),
      assignment ((xOutHashFor kind side).inputColumns.getD (4 * round + 2) 0),
      assignment ((xOutHashFor kind side).inputColumns.getD (4 * round + 3) 0)]
  else
    [1]

theorem xOutChunkAt_bounded
    (assignment : Nat -> Nat) (kind : ArmKind) (side : StateSide)
    (round : Nat) :
    (xOutChunkAt assignment kind side round).length <=
      Poseidon2Sponge.rate := by
  by_cases dataRound : round < 8 <;>
    simp [xOutChunkAt, dataRound, Poseidon2Sponge.rate]

def xOutChunks
    (assignment : Nat -> Nat) (kind : ArmKind) (side : StateSide) :
    List Poseidon2Sponge.RateChunk :=
  Poseidon2Sponge.chunkList (xOutChunkAt assignment kind side)
    (xOutChunkAt_bounded assignment kind side) 8

private theorem data_input_low_refines
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (round : Fin 8) (lane : Fin 4)
    (state : Poseidon2Reference.Values)
    (before : forall stateLane : Fin 8,
      assignment (stateBeforeColumn kind side round.val stateLane.val) =
        state stateLane) :
    assignment
        (permutationInputColumn kind side round.val (lowLane lane).val) =
      Poseidon2Sponge.absorbChunk
        (xOutChunkAt assignment kind side round.val) state (lowLane lane) := by
  have addition := absorb_input_eq kind side assignment canonical one satisfied
    round lane
  have chunkExact := (exact_trace kind side).dataChunk round lane
  have beforeLow :
      assignment (stateBeforeColumn kind side round.val lane.val) =
        state (lowLane lane) := by
    simpa [lowLane] using before (lowLane lane)
  rw [chunkExact, beforeLow] at addition
  fin_cases lane <;>
    simpa [lowLane, xOutChunkAt, Poseidon2Sponge.absorbChunk] using addition

private theorem data_input_high_refines
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (round : Fin 8) (lane : Fin 4)
    (state : Poseidon2Reference.Values)
    (before : forall stateLane : Fin 8,
      assignment (stateBeforeColumn kind side round.val stateLane.val) =
        state stateLane) :
    assignment
        (permutationInputColumn kind side round.val (highLane lane).val) =
      Poseidon2Sponge.absorbChunk
        (xOutChunkAt assignment kind side round.val) state (highLane lane) := by
  calc
    assignment
        (permutationInputColumn kind side round.val (highLane lane).val) =
        assignment
          (stateBeforeColumn kind side round.val (highLane lane).val) := by
      rw [(exact_trace kind side).dataCapacity round lane]
    _ = state (highLane lane) := before (highLane lane)
    _ = Poseidon2Sponge.absorbChunk
          (xOutChunkAt assignment kind side round.val) state
          (highLane lane) := by
      exact (Poseidon2Sponge.absorbChunk_beyond_chunk
        (xOutChunkAt assignment kind side round.val) state (highLane lane)
        (by simp [xOutChunkAt, round.isLt, highLane])).symm

private theorem pad_input_head_refines
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (state : Poseidon2Reference.Values)
    (before : forall stateLane : Fin 8,
      assignment (stateBeforeColumn kind side 8 stateLane.val) =
        state stateLane) :
    assignment (permutationInputColumn kind side 8 0) =
      Poseidon2Sponge.absorbChunk
        (xOutChunkAt assignment kind side 8) state ⟨0, by decide⟩ := by
  have padding := pad_input_eq kind side assignment canonical one satisfied
  rw [before ⟨0, by decide⟩] at padding
  simpa [xOutChunkAt, Poseidon2Sponge.absorbChunk] using padding

private theorem pad_input_tail_refines
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (lane : Fin 7)
    (state : Poseidon2Reference.Values)
    (before : forall stateLane : Fin 8,
      assignment (stateBeforeColumn kind side 8 stateLane.val) =
        state stateLane) :
    assignment
        (permutationInputColumn kind side 8 (padTailLane lane).val) =
      Poseidon2Sponge.absorbChunk
        (xOutChunkAt assignment kind side 8) state (padTailLane lane) := by
  calc
    assignment
        (permutationInputColumn kind side 8 (padTailLane lane).val) =
        assignment (stateBeforeColumn kind side 8 (padTailLane lane).val) := by
      rw [(exact_trace kind side).padTail lane]
    _ = state (padTailLane lane) := before (padTailLane lane)
    _ = Poseidon2Sponge.absorbChunk
          (xOutChunkAt assignment kind side 8) state (padTailLane lane) := by
      exact (Poseidon2Sponge.absorbChunk_beyond_chunk
        (xOutChunkAt assignment kind side 8) state (padTailLane lane)
        (by simp [xOutChunkAt, padTailLane])).symm

theorem permutation_input_refines
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (round : Fin 9) (state : Poseidon2Reference.Values)
    (before : forall lane : Fin 8,
      assignment (stateBeforeColumn kind side round.val lane.val) = state lane)
    (lane : Fin 8) :
    assignment (permutationInputColumn kind side round.val lane.val) =
      Poseidon2Sponge.absorbChunk
        (xOutChunkAt assignment kind side round.val) state lane := by
  by_cases dataRound : round.val < 8
  · let data : Fin 8 := ⟨round.val, dataRound⟩
    by_cases rateLane : lane.val < 4
    · let rate : Fin 4 := ⟨lane.val, rateLane⟩
      have laneEq : lane = lowLane rate := by apply Fin.ext; rfl
      rw [laneEq]
      apply data_input_low_refines kind side assignment canonical one satisfied
        data rate state
      intro stateLane
      simpa [data] using before stateLane
    · have laneAtLeastRate : 4 <= lane.val := by omega
      let capacity : Fin 4 := ⟨lane.val - 4, by omega⟩
      have laneEq : lane = highLane capacity := by
        apply Fin.ext
        simp [highLane, capacity]
        omega
      rw [laneEq]
      apply data_input_high_refines kind side assignment data capacity state
      intro stateLane
      simpa [data] using before stateLane
  · have padRound : round.val = 8 := by omega
    have roundEq : round = ⟨8, by decide⟩ := Fin.ext padRound
    subst round
    by_cases head : lane.val = 0
    · have laneEq : lane = ⟨0, by decide⟩ := Fin.ext head
      rw [laneEq]
      exact pad_input_head_refines kind side assignment canonical one satisfied
        state before
    · have lanePositive : 0 < lane.val := Nat.pos_of_ne_zero head
      let tail : Fin 7 := ⟨lane.val - 1, by omega⟩
      have laneEq : lane = padTailLane tail := by
        apply Fin.ext
        simp [padTailLane, tail]
        omega
      rw [laneEq]
      exact pad_input_tail_refines kind side assignment tail state before

private theorem getD_mem_of_lt {alpha : Type} [Inhabited alpha]
    {entries : List alpha} {index : Nat} (bounded : index < entries.length) :
    entries.getD index default ∈ entries := by
  have member := List.getElem_mem (l := entries) bounded
  rwa [List.getElem_eq_getD default] at member

theorem call_output_refines
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (round : Fin 9) (lane : Fin 8) :
    assignment (permutationOutputColumn kind side round.val lane.val) =
      Poseidon2Reference.referencePermutation
        Poseidon2CanonicalConstants.selected
        (fun inputLane =>
          assignment
            (permutationInputColumn kind side round.val inputLane.val)) lane := by
  let call := callAt kind side round.val
  have callMember : call ∈ (armFor kind).poseidon2Calls := by
    unfold call callAt
    exact getD_mem_of_lt ((exact_trace kind side).callBound round)
  have accepted := poseidon2_call_refines (armFor kind) assignment canonical
    one satisfied call callMember
  have transition := callAccepted_permute canonical one call
    ⟨0, by decide⟩ accepted
  calc
    assignment (permutationOutputColumn kind side round.val lane.val) =
        assignment (call.columnMap (601 + lane.val)) := by
      rw [(exact_trace kind side).callOutput round lane]
    _ = (toDuplex (callOutputState assignment canonical call)).lanes lane := rfl
    _ = (toDuplex
          (Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine.permute
            (callInputState assignment canonical call ⟨0, by decide⟩))).lanes
          lane := by rw [transition]
    _ = (Poseidon2Duplex.permute Poseidon2CanonicalConstants.selected
          (toDuplex
            (callInputState assignment canonical call ⟨0, by decide⟩))).lanes
          lane := by rw [permute_toDuplex]
    _ = Poseidon2Reference.referencePermutation
          Poseidon2CanonicalConstants.selected
          (fun inputLane =>
            assignment
              (permutationInputColumn kind side round.val inputLane.val))
          lane := by
      unfold Poseidon2Duplex.permute
      have inputsEqual :
          (fun inputLane : Fin 8 =>
            assignment (call.columnMap (inputLane.val + 1))) =
          (fun inputLane : Fin 8 =>
            assignment
              (permutationInputColumn kind side round.val inputLane.val)) := by
        funext inputLane
        rw [(exact_trace kind side).callInput round inputLane]
      change Poseidon2Reference.referencePermutation
          Poseidon2CanonicalConstants.selected
          (fun inputLane : Fin 8 =>
            assignment (call.columnMap (inputLane.val + 1))) lane = _
      rw [inputsEqual]
      rfl

theorem round_output_refines
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (round : Nat) (roundLt : round < 9) (lane : Fin 8) :
    assignment (permutationOutputColumn kind side round lane.val) =
      Poseidon2Sponge.absorbAt Poseidon2CanonicalConstants.selected
        (xOutChunkAt assignment kind side) round lane := by
  induction round generalizing lane with
  | zero =>
      let current : Fin 9 := ⟨0, by decide⟩
      have before : forall stateLane : Fin 8,
          assignment (stateBeforeColumn kind side 0 stateLane.val) =
            Poseidon2Sponge.initialSpongeState stateLane := by
        intro stateLane
        rw [(exact_trace kind side).startState stateLane]
        exact zero_column_eq_zero kind side assignment canonical one satisfied
      rw [call_output_refines kind side assignment canonical one satisfied
        current lane]
      apply congrArg (fun values =>
        Poseidon2Reference.referencePermutation
          Poseidon2CanonicalConstants.selected values lane)
      funext inputLane
      exact permutation_input_refines kind side assignment canonical one
        satisfied current Poseidon2Sponge.initialSpongeState before inputLane
  | succ previous inductionHypothesis =>
      have previousLt : previous < 8 := by omega
      let previousRound : Fin 8 := ⟨previous, previousLt⟩
      let current : Fin 9 := ⟨previous + 1, roundLt⟩
      have before : forall stateLane : Fin 8,
          assignment
              (stateBeforeColumn kind side (previous + 1) stateLane.val) =
            Poseidon2Sponge.absorbAt Poseidon2CanonicalConstants.selected
              (xOutChunkAt assignment kind side) previous stateLane := by
        intro stateLane
        rw [(exact_trace kind side).nextState previousRound stateLane]
        exact inductionHypothesis (by omega) stateLane
      rw [call_output_refines kind side assignment canonical one satisfied
        current lane]
      apply congrArg (fun values =>
        Poseidon2Reference.referencePermutation
          Poseidon2CanonicalConstants.selected values lane)
      funext inputLane
      exact permutation_input_refines kind side assignment canonical one
        satisfied current
        (Poseidon2Sponge.absorbAt Poseidon2CanonicalConstants.selected
          (xOutChunkAt assignment kind side) previous)
        before inputLane

private theorem chunk_chain_exact
    (assignment : Nat -> Nat) (kind : ArmKind) (side : StateSide) :
    Poseidon2Sponge.chunkList (xOutChunkAt assignment kind side)
        (xOutChunkAt_bounded assignment kind side) 9 =
      xOutChunks assignment kind side ++ [Poseidon2Sponge.paddingChunk] := by
  rw [show 9 = 8 + 1 by omega,
    Poseidon2Sponge.chunkList_succ]
  unfold xOutChunks
  congr 2

/-- Accepted generated rows compute the production additive Poseidon2 sponge
over all 32 exact full-`XOut` preimage columns. -/
theorem x_out_hash_refines
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (lane : Fin 4) :
    assignment (xOutDigestColumn kind side lane) =
      Poseidon2Sponge.digest Poseidon2CanonicalConstants.selected
        (xOutChunks assignment kind side) lane := by
  let outputLane : Fin 8 := ⟨lane.val, by omega⟩
  calc
    assignment (xOutDigestColumn kind side lane) =
        assignment (permutationOutputColumn kind side 8 outputLane.val) := by
      rw [(exact_trace kind side).digestOutput lane]
    _ = Poseidon2Sponge.absorbAt Poseidon2CanonicalConstants.selected
          (xOutChunkAt assignment kind side) 8 outputLane :=
      round_output_refines kind side assignment canonical one satisfied 8
        (by decide) outputLane
    _ = Poseidon2Sponge.absorb Poseidon2CanonicalConstants.selected
          (Poseidon2Sponge.chunkList (xOutChunkAt assignment kind side)
            (xOutChunkAt_bounded assignment kind side) 9)
          Poseidon2Sponge.initialSpongeState outputLane := by
      exact congrFun
        (Poseidon2Sponge.absorbAt_eq_absorb
          Poseidon2CanonicalConstants.selected
          (xOutChunkAt assignment kind side)
          (xOutChunkAt_bounded assignment kind side) 8) outputLane
    _ = Poseidon2Sponge.absorb Poseidon2CanonicalConstants.selected
          (xOutChunks assignment kind side ++ [Poseidon2Sponge.paddingChunk])
          Poseidon2Sponge.initialSpongeState outputLane := by
      rw [chunk_chain_exact assignment kind side]
    _ = Poseidon2Sponge.digest Poseidon2CanonicalConstants.selected
          (xOutChunks assignment kind side) lane :=
      (Poseidon2Sponge.digest_eq_absorb_padding
        Poseidon2CanonicalConstants.selected
        (xOutChunks assignment kind side) lane).symm

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutArtifact
