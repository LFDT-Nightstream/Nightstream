import NightstreamFPrime.Export.Stage1.PiRLCSamplerDirectSemantics

/-!
Owns the exact deterministic sampler-state sequence read from retained
Poseidon values. The initial state is the retained PiCCS endpoint. Every
scalar entry and advance use the verifier's four-field schedule.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiRLCStates

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiRLCSamplerPoseidonPlan
open PiRLCSamplerPoseidonPreservation
open Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript (stateAt)

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
      state geometry assignment ⟨previous, by omega⟩ ⟨1, by decide⟩ := by
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

theorem advance_eq (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment)
    (source : Fin sourceCount) :
    state geometry assignment source ⟨1, by decide⟩ =
      Poseidon2.permute (state geometry assignment source ⟨0, by decide⟩) := by
  have equation := semantics.invocation (invocation source ⟨1, by decide⟩)
  simp only [canonicalInput, descriptor_invocation] at equation
  rw [if_neg (by decide),
    PiRLCSamplerDirectSemantics.previousValue_advance geometry assignment source] at equation
  exact equation

/-- Each scalar starts from the state left by the preceding complete sampler. -/
theorem incoming_eq_stateAt (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment)
    (source : Fin sourceCount) :
    incomingState geometry assignment source =
      stateAt (initialState geometry assignment) source.val := by
  have atSource : ∀ current (bounded : current < sourceCount),
      incomingState geometry assignment ⟨current, bounded⟩ =
        stateAt (initialState geometry assignment) current := by
    intro current
    induction current with
    | zero => intro bounded; exact incoming_zero geometry assignment
    | succ previous ih =>
        intro bounded
        rw [incoming_succ, advance_eq geometry assignment semantics,
          entry_eq geometry assignment semantics, ih (by omega)]
        rfl
  exact atSource source.val source.isLt

/-- The retained draw state is the verifier's domain-separated entry. -/
theorem entry_eq_verifier (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment)
    (source : Fin sourceCount) :
    state geometry assignment source ⟨0, by decide⟩ =
      Lifecycle.Transcript.PiRlcSampler.enterScalar
        (stateAt (initialState geometry assignment) source.val) source.val := by
  rw [entry_eq geometry assignment semantics, incoming_eq_stateAt geometry assignment semantics]

theorem end_eq_nextState (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment)
    (source : Fin sourceCount) :
    state geometry assignment source ⟨1, by decide⟩ =
      stateAt (initialState geometry assignment) (source.val + 1) := by
  rw [advance_eq geometry assignment semantics, entry_eq_verifier geometry assignment semantics]
  rfl

theorem final_eq_stateAt (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) (semantics : CanonicalSemantics geometry assignment) :
    state geometry assignment ⟨16, by decide⟩ ⟨1, by decide⟩ =
      stateAt (initialState geometry assignment) sourceCount :=
  end_eq_nextState geometry assignment semantics ⟨16, by decide⟩

end NightstreamFPrime.Export.Stage1.ActualPiRLCStates
