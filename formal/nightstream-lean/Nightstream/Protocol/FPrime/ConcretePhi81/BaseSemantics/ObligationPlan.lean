import Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics
import Nightstream.SuperNeo.CheckPlan

/-!
Named obligation plan for the ConcretePhi81 base branch.

Assurance tier: model-level.

Owns: the three paper base checks, one typed case language, and exact
equivalence between plan acceptance and `BaseSemantics.Obligations`.

Does not own: the computed base output, the one-slot specialization, removal
witnesses, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: the case contains only the public outer input. Output
fields are computed by `BaseSemantics.outputOf` and therefore are not modeled
as caller-supplied checks.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.base.iteration` | `i = 0` | checked | `Family.iterationZero` |
| `fprime.base.initial_state` | `z_0 = z_i` | checked | `Family.initialState` |
| `fprime.base.dispatch` | `phi(z_i, w) = j` | checked in the general profile | `Family.dispatch` |
| `fprime.base.obligation_plan.exact` | the three leaves equal the independent base obligations | exact model theorem | `exact` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Protocol.FPrime.ConcretePhi81.Outer

universe uOuterKey uAppState uWitness uDigest

/-- Complete general base-check family. -/
inductive Family where
  | iterationZero
  | initialState
  | dispatch
  deriving DecidableEq

/-- Stable mathematical review order, not a physical row order. -/
def checks : List Family :=
  [.iterationZero, .initialState, .dispatch]

/-- One complete verifier-language base case. -/
structure Case
    (OuterKey : Type uOuterKey)
    (AppState : Type uAppState)
    (Witness : Type uWitness)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows slotCount : Nat) where
  input :
    Input OuterKey AppState Witness shape publicRingColumns publicFits
      verifierRows slotCount

section

variable {OuterKey : Type uOuterKey}
variable {Digest : Type uDigest}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows slotCount : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Interpret one named leaf directly as its independent base equation. -/
def semantics
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount) :
    Family ->
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount -> Prop
  | .iterationZero, case => case.input.iteration = 0
  | .initialState, case => case.input.z0 = case.input.zi
  | .dispatch, case =>
      machine.control case.input.zi case.input.witness =
        Paper.ProgramCounter.ofIndex functionIndex

/-- Independent target for the named plan. -/
def target
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (case :
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) : Prop :=
  Obligations machine functionIndex case.input

/-- Complete plan acceptance is field-for-field equivalent to the existing
independent base obligations. -/
theorem accepts_iff_obligations
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (case :
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount) :
    CheckPlan.Accepts (semantics machine functionIndex) checks case <->
      target machine functionIndex case := by
  constructor
  · intro accepted
    exact {
      iterationZero := accepted .iterationZero (by simp [checks])
      initialState := accepted .initialState (by simp [checks])
      dispatch := accepted .dispatch (by simp [checks])
    }
  · intro obligations family _member
    cases family with
    | iterationZero => exact obligations.iterationZero
    | initialState => exact obligations.initialState
    | dispatch => exact obligations.dispatch

/-- Exactness of the complete general base plan. -/
theorem exact
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount) :
    CheckPlan.Exact (semantics machine functionIndex)
      (target machine functionIndex) checks := by
  intro case
  exact accepts_iff_obligations machine functionIndex case

end

end Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan
