import NightstreamFPrime.Export.Stage1.PiRLCSamplerDirectSemantics

/-!
Owns the exact deterministic sampler-state sequence read from retained
Poseidon values. The initial state is the retained PiCCS endpoint. Every
scalar entry and every complete digest block use the verifier's schedule.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiRLCStates

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCSamplerPoseidonPlan
open PiRLCSamplerPoseidonPreservation
open PiRLCSamplerDirectSemantics (priorStep windowStep)

open Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule
open Spec.Folding.Nifs.NonInteractive.PiRlcSampler (stateAt)

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

def initialState (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Poseidon2.State :=
  List.ofFn (piCcsFinalValue geometry assignment)

def incomingState (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin sourceCount) : Poseidon2.State :=
  List.ofFn (previousValue geometry assignment (invocation source ⟨0, by decide⟩))

def state (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Fin sourceCount)
    (step : Fin invocationsPerSource) : Poseidon2.State :=
  List.ofFn (outputValue geometry assignment (invocation source step))

theorem incoming_zero (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    incomingState geometry assignment ⟨0, by decide⟩ = initialState geometry assignment := by
  simp [incomingState, initialState, previousValue, invocation, Fin.encodeProd]

theorem incoming_succ (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (previous : Nat)
    (bounded : previous + 1 < sourceCount) :
    incomingState geometry assignment ⟨previous + 1, bounded⟩ =
      state geometry assignment ⟨previous, by omega⟩ ⟨8, by decide⟩ := by
  exact congrArg List.ofFn
    (PiRLCSamplerDirectSemantics.previousValue_entrySucc geometry assignment previous bounded)

theorem entry_eq (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment)
    (source : Fin sourceCount) :
    state geometry assignment source ⟨0, by decide⟩ =
      Lifecycle.Transcript.PiRlcSampler.enterScalar (incomingState geometry assignment source)
        source.val := by
  rw [incomingState, PiRLCSamplerDirectSemantics.enterScalar_ofFn _ source]
  simpa [state, canonicalInput, descriptor_invocation] using
    semantics.invocation (invocation source ⟨0, by decide⟩)

theorem window_eq (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment)
    (source : Fin sourceCount) (round : Fin PiRLCSamplerOrdinaryRetainedBlocks.roundCount) :
    state geometry assignment source (windowStep round) =
      Poseidon2.permute (state geometry assignment source (priorStep round)) := by
  have equation := semantics.invocation (invocation source (windowStep round))
  simp only [canonicalInput, descriptor_invocation] at equation
  rw [if_neg (by simp [windowStep]),
    PiRLCSamplerDirectSemantics.previousValue_window geometry assignment source round] at equation
  exact equation

/-- Every retained step follows the verifier's digest-block recurrence. -/
theorem state_eq_blockState (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment)
    (source : Fin sourceCount) (step : Fin invocationsPerSource) :
    state geometry assignment source step =
      stateBeforeBlock Lifecycle.Transcript.PiRlcSampler.machine
        (state geometry assignment source ⟨0, by decide⟩) source.val step.val := by
  have atStep : ∀ current (bounded : current < invocationsPerSource),
      state geometry assignment source ⟨current, bounded⟩ =
        stateBeforeBlock Lifecycle.Transcript.PiRlcSampler.machine
          (state geometry assignment source ⟨0, by decide⟩) source.val current := by
    intro current
    induction current with
    | zero => intro bounded; rfl
    | succ previous ih =>
        intro bounded
        have roundLt : previous < PiRLCSamplerOrdinaryRetainedBlocks.roundCount := by
          change previous + 1 < 9 at bounded
          change previous < 8
          omega
        have recurrence : state geometry assignment source ⟨previous + 1, bounded⟩ =
            Poseidon2.permute (state geometry assignment source ⟨previous, by omega⟩) := by
          simpa only [windowStep, priorStep] using
            window_eq geometry assignment semantics source ⟨previous, roundLt⟩
        rw [recurrence, ih (by omega), stateBeforeBlock_succ]
        rfl
  exact atStep step.val step.isLt

/-- Each scalar starts from the state left by the preceding complete sampler. -/
theorem incoming_eq_stateAt (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment)
    (source : Fin sourceCount) :
    incomingState geometry assignment source =
      stateAt Lifecycle.Transcript.PiRlcSampler.specification
        (initialState geometry assignment) source.val := by
  have atSource : ∀ current (bounded : current < sourceCount),
      incomingState geometry assignment ⟨current, bounded⟩ =
        stateAt Lifecycle.Transcript.PiRlcSampler.specification
          (initialState geometry assignment) current := by
    intro current
    induction current with
    | zero => intro bounded; exact incoming_zero geometry assignment
    | succ previous ih =>
        intro bounded
        rw [incoming_succ, state_eq_blockState geometry assignment semantics,
          entry_eq geometry assignment semantics, ih (by omega)]
        exact (stateAt_succ_eq_fixedBlockState Lifecycle.Transcript.PiRlcSampler.machine
          Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.assembleCoefficients
          (initialState geometry assignment) (⟨previous, by omega⟩ : Fin sourceCount)).symm
  exact atSource source.val source.isLt

/-- All retained states are the exact verifier states from the retained PiCCS endpoint. -/
theorem state_eq_verifier (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment)
    (source : Fin sourceCount) (step : Fin invocationsPerSource) :
    state geometry assignment source step =
      stateBeforeBlock Lifecycle.Transcript.PiRlcSampler.machine
        (Lifecycle.Transcript.PiRlcSampler.enterScalar
          (stateAt Lifecycle.Transcript.PiRlcSampler.specification
            (initialState geometry assignment) source.val) source.val) source.val step.val := by
  rw [state_eq_blockState geometry assignment semantics, entry_eq geometry assignment semantics,
    incoming_eq_stateAt geometry assignment semantics]

theorem end_eq_nextState (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment)
    (source : Fin sourceCount) :
    state geometry assignment source ⟨8, by decide⟩ =
      stateAt Lifecycle.Transcript.PiRlcSampler.specification
        (initialState geometry assignment) (source.val + 1) := by
  rw [state_eq_verifier geometry assignment semantics]
  exact (stateAt_succ_eq_fixedBlockState Lifecycle.Transcript.PiRlcSampler.machine
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.assembleCoefficients
    (initialState geometry assignment) source).symm

theorem final_eq_stateAt (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment) :
    state geometry assignment ⟨16, by decide⟩ ⟨8, by decide⟩ =
      stateAt Lifecycle.Transcript.PiRlcSampler.specification
        (initialState geometry assignment) sourceCount :=
  end_eq_nextState geometry assignment semantics ⟨16, by decide⟩

end NightstreamFPrime.Export.Stage1.ActualPiRLCStates
