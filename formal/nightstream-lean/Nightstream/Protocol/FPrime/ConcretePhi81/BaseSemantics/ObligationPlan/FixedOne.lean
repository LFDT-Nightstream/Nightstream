import Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan

/-!
Exact base obligation plan for the one-slot F-prime profile.

Assurance tier: model-level.

Owns: derivation of dispatch from the one-element program-counter codomain,
the two retained base checks, the eliminated-family ledger, and exactness of
the reduced plan.

Does not own: selection of this profile by Rust, removal witnesses, output
computation, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: dispatch is eliminated only because the profile type is
`slotCount = 1`. This theorem does not authorize omitting dispatch from a
multi-slot or untyped production carrier.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.base.fixed_one.iteration` | `i = 0` | retained check | `checks` |
| `fprime.base.fixed_one.initial_state` | `z_0 = z_i` | retained check | `checks` |
| `fprime.base.fixed_one.dispatch` | every checked one-slot counter is the sole counter | derived/eliminated | `dispatch_derived` |
| `fprime.base.fixed_one.exact` | the two retained leaves equal all base obligations | exact model theorem | `exact` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Protocol.FPrime.ConcretePhi81.Outer

universe uOuterKey uAppState uWitness uDigest

/-- Retained checks after specializing the program counter to one slot. -/
def checks : List Family := [.iterationZero, .initialState]

/-- The sole eliminated general-family check. -/
def eliminated : List Family := [.dispatch]

@[simp] theorem dispatch_not_mem : Family.dispatch ∉ checks := by
  simp [checks]

/-- Every general base family is recorded exactly once. -/
theorem classified (family : Family) :
    family ∈ checks ∨ family ∈ eliminated := by
  cases family <;> simp [checks, eliminated]

/-- No base family is both retained and eliminated. -/
theorem classification_disjoint (family : Family) :
    ¬(family ∈ checks ∧ family ∈ eliminated) := by
  cases family <;> simp [checks, eliminated]

section

variable {OuterKey : Type uOuterKey}
variable {Digest : Type uDigest}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Dispatch is determined by the one-element checked-counter codomain. -/
theorem dispatch_derived
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (case :
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows 1) :
    ObligationPlan.semantics machine functionIndex .dispatch case := by
  change machine.control case.input.zi case.input.witness =
    Paper.ProgramCounter.ofIndex functionIndex
  calc
    machine.control case.input.zi case.input.witness =
        Paper.ProgramCounter.ofIndex
          (machine.control case.input.zi case.input.witness).index :=
      (Paper.ProgramCounter.ofIndex_index _).symm
    _ = Paper.ProgramCounter.ofIndex functionIndex := by
      congr
      exact Subsingleton.elim _ _

/-- The two retained leaves are exactly all base obligations in the fixed-one
profile. -/
theorem accepts_iff_obligations
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (case :
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows 1) :
    CheckPlan.Accepts
        (ObligationPlan.semantics machine functionIndex) checks case <->
      ObligationPlan.target machine functionIndex case := by
  constructor
  · intro accepted
    exact {
      iterationZero := accepted .iterationZero (by simp [checks])
      initialState := accepted .initialState (by simp [checks])
      dispatch := dispatch_derived machine functionIndex case
    }
  · intro obligations family member
    cases family with
    | iterationZero => exact obligations.iterationZero
    | initialState => exact obligations.initialState
    | dispatch => exact (dispatch_not_mem member).elim

/-- Exactness of the reduced fixed-one base plan. -/
theorem exact
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1) :
    CheckPlan.Exact
      (ObligationPlan.semantics machine functionIndex)
      (ObligationPlan.target machine functionIndex) checks := by
  intro case
  exact accepts_iff_obligations machine functionIndex case

end

end Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne
