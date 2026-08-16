import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilySequence
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyContinuity

/-!
Contract: exact 110-arm physical PiRLC family sequence.

Owns one accepted physical arm for every verifier-owned family ordinal, the
explicit adjacent local semantic-digest contract, construction of the
model-level `AcceptedRun`, and the reduction from physical start-to-finish
acceptance to exact PiRLC outputs, a concrete binding failure, or one
Poseidon2 collision.

Does not own generated shared-wire enforcement, start or finish circuits,
collision resistance, Module-SIS hardness, selective lowering, or the outer
recursive lifecycle.

Emits constraints: no.

Assurance tier: security-reduced prototype. Accepted physical family arms
with explicit adjacent private semantic-digest links refine the exact
110-step family sequence unless the exact Poseidon2 family-state digest has a
collision. The full-XOut sequence is the production continuity target.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalSequence

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlc
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySequence
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalState

abbrev PhysicalFamily :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.Family
abbrev PhysicalSource :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.Source
abbrev PhysicalInputRings :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.InputRings
abbrev PhysicalRing := Nightstream.SuperNeo.Concrete.RingF

/-- The last verifier-owned family ordinal. -/
def lastOrdinal : Fin exactFamilyCount := ⟨109, by decide⟩

/-- One accepted generated family arm at every exact ordinal, with an
explicit local semantic-digest link and the public cursor link between
successors. This is a prototype adapter; production continuity uses complete
XOut equality. -/
structure AcceptedPhysicalRun (setup : InputBindingSetup) where
  arm : ∀ ordinal : Fin exactFamilyCount,
    AcceptedArm setup (familyAtOrdinal ordinal)
  continuous : ∀ (ordinal : Fin exactFamilyCount)
      (hasNext : ordinal.val + 1 < exactFamilyCount),
    SemanticStateContinuous (arm ordinal)
      (arm ⟨ordinal.val + 1, hasNext⟩)

namespace AcceptedPhysicalRun

/-- The exact family-major input array decoded from the 110 accepted bodies. -/
def inputRings {setup : InputBindingSetup}
    (run : AcceptedPhysicalRun setup) : PhysicalInputRings :=
  fun source family =>
    (run.arm (familyIndex family)).phaseInputs source

/-- The exact output array decoded from the 110 accepted bodies. -/
def outputs {setup : InputBindingSetup}
    (run : AcceptedPhysicalRun setup) : PhysicalFamily → PhysicalRing :=
  fun family =>
    (run.arm (familyIndex family)).phaseOutput

/-- Total model state used by `AcceptedRun`. Indices 0 through 109 use the
before state of that ordinal. Index 110 and unused larger indices use the
after state of the last arm. -/
def boundaryState {setup : InputBindingSetup}
    (run : AcceptedPhysicalRun setup) (index : Nat) : FamilyState :=
  if bound : index < exactFamilyCount then
    (run.arm ⟨index, bound⟩).beforeState
  else
    (run.arm lastOrdinal).afterState

@[simp] theorem boundaryState_before
    {setup : InputBindingSetup} (run : AcceptedPhysicalRun setup)
    (ordinal : Fin exactFamilyCount) :
    run.boundaryState ordinal.val = (run.arm ordinal).beforeState := by
  simp [boundaryState, ordinal.isLt]

/-- Adjacent accepted public words give exact cursor continuity without a
cryptographic assumption. -/
theorem adjacent_cursor_eq
    {setup : InputBindingSetup} (run : AcceptedPhysicalRun setup)
    (ordinal : Fin exactFamilyCount)
    (hasNext : ordinal.val + 1 < exactFamilyCount) :
    (run.arm ordinal).afterState.familyCursor =
      (run.arm ⟨ordinal.val + 1, hasNext⟩).beforeState.familyCursor := by
  exact (accepted_semantic_continuity (run.arm ordinal)
    (run.arm ⟨ordinal.val + 1, hasNext⟩)
    (run.continuous ordinal hasNext)).1

/-- Adjacent accepted public words give equal complete states or the named
Poseidon2 collision. -/
theorem adjacent_state_or_collision
    {setup : InputBindingSetup} (run : AcceptedPhysicalRun setup)
    (ordinal : Fin exactFamilyCount)
    (hasNext : ordinal.val + 1 < exactFamilyCount) :
    (run.arm ordinal).afterState =
        (run.arm ⟨ordinal.val + 1, hasNext⟩).beforeState ∨
      Poseidon2FamilyStateCollision := by
  exact (accepted_semantic_continuity (run.arm ordinal)
    (run.arm ⟨ordinal.val + 1, hasNext⟩)
    (run.continuous ordinal hasNext)).2

private theorem adjacent_state_eq
    {setup : InputBindingSetup} (run : AcceptedPhysicalRun setup)
    (noCollision : ¬ Poseidon2FamilyStateCollision)
    (ordinal : Fin exactFamilyCount)
    (hasNext : ordinal.val + 1 < exactFamilyCount) :
    (run.arm ordinal).afterState =
      (run.arm ⟨ordinal.val + 1, hasNext⟩).beforeState :=
  (run.adjacent_state_or_collision ordinal hasNext).resolve_right noCollision

theorem ordinal_eq_last_of_no_next
    (ordinal : Fin exactFamilyCount)
    (noNext : ¬ ordinal.val + 1 < exactFamilyCount) :
    ordinal = lastOrdinal := by
  apply Fin.ext
  have bound : ordinal.val < 110 := by
    exact ordinal.isLt
  have noNext' : ¬ ordinal.val + 1 < 110 := by
    simpa [exactFamilyCount,
      ProductionStreamingPiRlcInputBinding.familyCount] using noNext
  have value : ordinal.val = 109 := by omega
  simpa [lastOrdinal] using value

/-- With no collision, the total boundary state after an arm is its exact
decoded after state. -/
theorem boundaryState_after
    {setup : InputBindingSetup} (run : AcceptedPhysicalRun setup)
    (noCollision : ¬ Poseidon2FamilyStateCollision)
    (ordinal : Fin exactFamilyCount) :
    run.boundaryState (ordinal.val + 1) =
      (run.arm ordinal).afterState := by
  by_cases hasNext : ordinal.val + 1 < exactFamilyCount
  · rw [boundaryState, dif_pos hasNext]
    exact (run.adjacent_state_eq noCollision ordinal hasNext).symm
  · rw [boundaryState, dif_neg hasNext]
    rw [ordinal_eq_last_of_no_next ordinal hasNext]

/-- In the no-collision branch, the 110 accepted physical arms construct the
exact model-level family sequence. -/
def semanticRun
    {setup : InputBindingSetup} (run : AcceptedPhysicalRun setup)
    (noCollision : ¬ Poseidon2FamilyStateCollision) :
    AcceptedRun setup run.inputRings where
  state := run.boundaryState
  output := run.outputs
  phase := by
    intro ordinal
    rw [run.boundaryState_before ordinal,
      run.boundaryState_after noCollision ordinal]
    change FamilyPhaseRelation setup (run.arm ordinal).beforeState
      (run.arm ordinal).afterState (familyAtOrdinal ordinal)
      (fun source =>
        (run.arm (familyIndex (familyAtOrdinal ordinal))).phaseInputs source)
      (run.arm (familyIndex (familyAtOrdinal ordinal))).phaseOutput
    rw [familyIndex_familyAtOrdinal]
    exact (run.arm ordinal).phase

/-- The complete physical family sequence refines `AcceptedRun`, or one
adjacent public equality exposes the named Poseidon2 collision. -/
theorem semanticRun_or_collision
    {setup : InputBindingSetup} (run : AcceptedPhysicalRun setup) :
    Nonempty (AcceptedRun setup run.inputRings) ∨
      Poseidon2FamilyStateCollision := by
  classical
  by_cases collision : Poseidon2FamilyStateCollision
  · exact Or.inr collision
  · exact Or.inl ⟨run.semanticRun collision⟩

/-- Start and finish authority recover every body-supplied input, or expose
the concrete Module-SIS binding failure or a Poseidon2 continuity collision. -/
theorem start_finish_recovers_inputs_or_failure_or_collision
    {setup : InputBindingSetup}
    {authoritative : PhysicalInputRings}
    {authoritativeChallenges : PhysicalSource → PhysicalRing}
    (run : AcceptedPhysicalRun setup)
    (start : FamilyStartRelation (run.boundaryState 0)
      authoritativeChallenges (concreteBinding setup authoritative))
    (finish : FamilyFinishRelation
      (run.boundaryState exactFamilyCount)) :
    run.inputRings = authoritative ∨
      ConcreteBindingFailure setup ∨ Poseidon2FamilyStateCollision := by
  classical
  by_cases collision : Poseidon2FamilyStateCollision
  · exact Or.inr (Or.inr collision)
  · let semantic := run.semanticRun collision
    have result := semantic.start_finish_recovers_inputs_or_failure
      (by simpa [semantic, semanticRun] using start)
      (by simpa [semantic, semanticRun] using finish)
    rcases result with exact | failure
    · exact Or.inl (by simpa [semantic, semanticRun] using exact)
    · exact Or.inr (Or.inl failure)

/-- In the non-failure branch, all 110 physical outputs are the exact PiRLC
combinations of the authoritative PiCCS inputs and challenge array. -/
theorem outputs_exact_or_failure_or_collision
    {setup : InputBindingSetup}
    {authoritative : PhysicalInputRings}
    {authoritativeChallenges : PhysicalSource → PhysicalRing}
    (run : AcceptedPhysicalRun setup)
    (start : FamilyStartRelation (run.boundaryState 0)
      authoritativeChallenges (concreteBinding setup authoritative))
    (finish : FamilyFinishRelation
      (run.boundaryState exactFamilyCount)) :
    (∀ family,
        run.outputs family =
          familyOutput authoritativeChallenges authoritative family) ∨
      ConcreteBindingFailure setup ∨ Poseidon2FamilyStateCollision := by
  classical
  by_cases collision : Poseidon2FamilyStateCollision
  · exact Or.inr (Or.inr collision)
  · let semantic := run.semanticRun collision
    have result := semantic.outputs_exact_or_failure
      (by simpa [semantic, semanticRun] using start)
      (by simpa [semantic, semanticRun] using finish)
    rcases result with exact | failure
    · exact Or.inl (by simpa [semantic, semanticRun] using exact)
    · exact Or.inr (Or.inl failure)

end AcceptedPhysicalRun

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhysicalSequence
