import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonChainedCapacity

/-!
Contract: retained-row replay transition for one exact production PiRLC
Poseidon2 run.

Assurance tier: artifact-checked same-assignment call-chain semantics for the
Nightstream b2/k16 profile.

Owns: exact fresh-word authority, every call's independent Poseidon2 result,
and all adjacent-call capacity transitions for the four Rust-emitted runs.

Does not own: first-state placement, final-state placement, selector
exclusivity, final matrix-slice identity, complete PiRLC semantics, or
lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayTransition

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedCapacity
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompactTrace
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonRunValues
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonValuePlacement
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

/-- Totalized authoritative input word. Out-of-range ordinals fail closed. -/
def inputWordAt
    (assignment : Fin productionFinalColumns → F) (ordinal : Nat) : F :=
  if bounded : ordinal < 918 then
    wordValue assignment (38340 + ordinal * 41) (by
      change 38340 + ordinal * 41 + 41 ≤ productionFinalColumns
      simp only [productionFinalColumns]
      omega)
  else
    0

/-- Totalized authoritative output word. Out-of-range ordinals fail closed. -/
def outputWordAt
    (assignment : Fin productionFinalColumns → F) (ordinal : Nat) : F :=
  if bounded : ordinal < 54 then
    wordValue assignment (75978 + ordinal * 41) (by
      change 75978 + ordinal * 41 + 41 ≤ productionFinalColumns
      simp only [productionFinalColumns]
      omega)
  else
    0

/-- Exact source assignment reconstructed for one call. -/
def sourceAt (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F) : SourceAssignment :=
  sourceFor (run.leafClassAt index)
    (projectFinalAssignment (run.callSiteAt index) assignment)

/-- Eight exact input words consumed by one call. -/
def callInputs (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F) : Values :=
  fun lane => (sourceInput (sourceAt run index assignment) lane).val

/-- Independent eight-lane reference result for one call. -/
def callReference (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F) : Values :=
  referencePermutation Poseidon2CanonicalConstants.selected
    (callInputs run index assignment)

/-- Complete internal transition of one retained call run. Initial and final
state placement are separate trust-boundary obligations. -/
structure RunReplayTransition
    (run : Run) (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F) : Prop where
  fresh : ∀ (index : Fin run.raw.callCount) (lane : Fin 4) (ordinal : Nat),
    (run.callSiteAt index.val).freshOrdinal lane = some ordinal →
      sourceInput (sourceAt run index.val assignment) (rateLane lane) =
        freshWord ordinal
  output : ∀ (index : Fin run.raw.callCount) (lane : Fin width),
    lcEval (sourcePhysical (sourceAt run index.val assignment))
        (traceFinalForm lane) =
      callReference run index.val assignment lane
  chained : ∀ (index : Nat), index.succ < run.raw.callCount →
    ∀ lane : Fin 4,
      (sourceInput (sourceAt run index.succ assignment)
        (capacityLane lane)).val =
      callReference run index assignment (capacityLane lane)

/-- Every fresh input-run lane reads the exact normalized input word. -/
theorem inputRun_freshValue_authoritative
    (run : Run) (selected : run = evenInputRun ∨ run = oddInputRun)
    (index : Fin run.raw.callCount) (lane : Fin 4) (ordinal : Nat)
    (fresh : (run.callSiteAt index.val).freshOrdinal lane = some ordinal)
    (assignment : Fin productionFinalColumns → F) :
    sourceInput (sourceAt run index.val assignment) (rateLane lane) =
      inputWordAt assignment ordinal := by
  rcases inputRun_freshSlot_exact run selected index lane ordinal fresh with
    ⟨bounded, boundedValue, _⟩
  subst ordinal
  unfold sourceAt
  rw [inputRun_freshValue_exact run selected index lane bounded.val fresh
    assignment]
  simp [inputWordAt, bounded.isLt]

/-- Every fresh output-run lane reads the exact normalized output word. -/
theorem outputRun_freshValue_authoritative
    (run : Run) (selected : run = evenOutputRun ∨ run = oddOutputRun)
    (index : Fin run.raw.callCount) (lane : Fin 4) (ordinal : Nat)
    (fresh : (run.callSiteAt index.val).freshOrdinal lane = some ordinal)
    (assignment : Fin productionFinalColumns → F) :
    sourceInput (sourceAt run index.val assignment) (rateLane lane) =
      outputWordAt assignment ordinal := by
  rcases outputRun_freshSlot_exact run selected index lane ordinal fresh with
    ⟨bounded, boundedValue, _⟩
  subst ordinal
  unfold sourceAt
  rw [outputRun_freshValue_exact run selected index lane bounded.val fresh
    assignment]
  simp [outputWordAt, bounded.isLt]

/-- Retained rows plus exact fresh-word placement imply the complete internal
transition for one frozen production run. -/
theorem rows_imply_run_replay_transition
    (run : Run) (productionRun : run ∈ runs)
    (assignment : Fin productionFinalColumns → F)
    (freshWord : Nat → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment run.raw.selectorColumn = 1)
    (satisfied : ∀ index : Fin run.raw.callCount,
      (run.emittedBlockAt index).Satisfied assignment)
    (freshExact :
      ∀ (index : Fin run.raw.callCount) (lane : Fin 4) (ordinal : Nat),
        (run.callSiteAt index.val).freshOrdinal lane = some ordinal →
          sourceInput (sourceAt run index.val assignment) (rateLane lane) =
            freshWord ordinal) :
    RunReplayTransition run assignment freshWord := by
  have valid := runs_valid run productionRun
  have selectorAt : ∀ index : Nat,
      absoluteValue assignment
          (selectorColumn (run.leafClassAt index)) = 1 := by
    intro index
    rw [valid.selectorColumn_owned]
    exact selectorOne
  refine { fresh := freshExact, output := ?_, chained := ?_ }
  · intro index lane
    exact production_emitted_block_computes_reference
      run productionRun index assignment one (selectorAt index.val)
        (satisfied index) lane
  · intro index currentInRange lane
    let prior : Fin run.raw.callCount :=
      ⟨index, Nat.lt_trans (Nat.lt_succ_self index) currentInRange⟩
    have currentKind :
        run.leafClassAt index.succ = .chained run.raw.selectorColumn := by
      simp [Run.leafClassAt]
    calc
      (sourceInput (sourceAt run index.succ assignment)
          (capacityLane lane)).val =
          (portAction (priorImagePort lane)
            (projectFinalAssignment (run.callSiteAt index.succ)
              assignment)).val := by
        unfold sourceAt
        rw [currentKind]
        exact congrArg Fin.val
          (chained_sourceInput_capacity run.raw.selectorColumn
            (projectFinalAssignment (run.callSiteAt index.succ) assignment)
            lane)
      _ = (sourceAction (expectedCapacitySource lane)
            (sourceAt run index assignment)).val := by
        unfold sourceAt
        exact congrArg Fin.val
          (portRealized_action
            (capacityPort_realized run index assignment lane))
      _ = lcEval (sourcePhysical (sourceAt run index assignment))
            (traceFinalForm (capacityLane lane)) := by
        exact (traceFinalForm_eval_eq_capacitySource
          (run.leafClassAt index)
          (projectFinalAssignment (run.callSiteAt index) assignment)
          lane).symm
      _ = callReference run index assignment (capacityLane lane) := by
        unfold callReference callInputs sourceAt
        exact production_emitted_block_computes_reference
          run productionRun prior assignment one (selectorAt index)
            (satisfied prior) (capacityLane lane)

/-- Both exact input runs satisfy the same authoritative 918-word replay
transition. -/
theorem input_rows_imply_run_replay_transition
    (run : Run) (selected : run = evenInputRun ∨ run = oddInputRun)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment run.raw.selectorColumn = 1)
    (satisfied : ∀ index : Fin run.raw.callCount,
      (run.emittedBlockAt index).Satisfied assignment) :
    RunReplayTransition run assignment (inputWordAt assignment) := by
  have productionRun : run ∈ runs := by
    rcases selected with rfl | rfl <;> simp [runs]
  apply rows_imply_run_replay_transition run productionRun assignment
    (inputWordAt assignment) one selectorOne satisfied
  intro index lane ordinal fresh
  exact inputRun_freshValue_authoritative
    run selected index lane ordinal fresh assignment

/-- Both exact output runs satisfy the same authoritative 54-word replay
transition. -/
theorem output_rows_imply_run_replay_transition
    (run : Run) (selected : run = evenOutputRun ∨ run = oddOutputRun)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment run.raw.selectorColumn = 1)
    (satisfied : ∀ index : Fin run.raw.callCount,
      (run.emittedBlockAt index).Satisfied assignment) :
    RunReplayTransition run assignment (outputWordAt assignment) := by
  have productionRun : run ∈ runs := by
    rcases selected with rfl | rfl <;> simp [runs]
  apply rows_imply_run_replay_transition run productionRun assignment
    (outputWordAt assignment) one selectorOne satisfied
  intro index lane ordinal fresh
  exact outputRun_freshValue_authoritative
    run selected index lane ordinal fresh assignment

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonReplayTransition
