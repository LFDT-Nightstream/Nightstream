import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Context

/-!
Minimal incoming-accumulator authority for the canonical fixed-active NIFS.

Protocol: fixed SuperNeo NIFS `CE^k x CCS -> CE^k`.
Phase: validate the complete incoming parent against fourteen running children.
Constraint family: shared point and strict `Pi_DEC` recomposition; this file
emits no rows.

Owns: the four retained incoming equations; their exact equivalence to the
generic checked-parent predicate on the canonical carrier; and an executable
Boolean checker exact to those equations.

Does not own: active-mode selection, parent presence, stages, relation
structure, transcript hashing, child openings, outgoing `Pi_RLC`/`Pi_DEC`,
Rust/R1CS lowering, physical rows, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: the complete parent is part of the canonical typed input.
Mode, parent presence, stages, and relation structure are installed by
`Canonical.Context.materialize`; they are not prover-controlled values checked
again here. The retained fields are compared coordinate-by-coordinate without
digest authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_active.running.point` | every running child uses the complete parent's evaluation point | checked | `Equations.points`, `check` |
| `nifs.fixed_active.running.commitment` | child commitments recompose to the complete parent commitment | checked | `Equations.commitment`, `check` |
| `nifs.fixed_active.running.public_input` | child public inputs recompose to the complete parent public input | checked | `Equations.publicInput`, `check` |
| `nifs.fixed_active.running.evaluations` | child evaluation arrays recompose to the complete parent evaluations | checked | `Equations.evaluations`, `check` |
| `nifs.fixed_active.running.mode` | active mode is fixed by the carrier | derived/eliminated | `accepted_iff_equations` |
| `nifs.fixed_active.running.parent_presence` | the complete parent is always present | derived/eliminated | `accepted_iff_equations` |
| `nifs.fixed_active.running.stages_structure` | stages and the shared relation structure are fixed by materialization | derived/eliminated | `accepted_iff_equations` |
| `nifs.fixed_active.running.exact_checker` | Boolean result iff generic checked-parent acceptance holds | exact model theorem | `check_eq_true_iff_accepted` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.RunningAuthority

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open CarrierEquality

universe uState

/-- All and only the non-canonical equations for incoming checked-parent
authority. Function equality names the complete fixed child family. -/
structure Equations
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Canonical.Context shape domain State publicRingColumns publicFits
        verifierRows) : Prop where
  points :
    (fun child => (context.input.running child).point) =
      (fun _ => context.input.parent.point)
  commitment :
    context.input.parent.commitment =
      (decAlgebra context.key).recomposeCommitment
        (fun child => (context.input.running child).commitment)
  publicInput :
    context.input.parent.publicInput =
      (decAlgebra context.key).recomposePublicInput
        (fun child => (context.input.running child).publicInput)
  evaluations :
    context.input.parent.evaluations =
      (decAlgebra context.key).recomposeEvaluations
        (fun child => (context.input.running child).evaluations)

/-- The reduced equations construct complete generic incoming-parent
acceptance because all omitted fields are canonical carrier data. -/
theorem accepted_of_equations
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Canonical.Context shape domain State publicRingColumns publicFits
        verifierRows}
    (equations : Equations context) :
    ConcretePhi81.RunningAuthority.Accepted context.materialize := by
  refine .active {
    active := arity_mode
    parent := context.input.parent.materialize context.input.system
    parentBound := rfl
    piDec := ?_
  }
  refine {
    parentCombined := rfl
    childFresh := ?_
    sameStructure := ?_
    samePoint := ?_
    commitmentEquation := ?_
    publicInputEquation := ?_
    evaluationEquation := ?_
  }
  · intro child
    rfl
  · intro child
    rfl
  · intro child
    simpa [ConcretePhi81.RunningAuthority.attempt,
      ConcretePhi81.RunningAuthority.children,
      ConcretePhi81.RunningAuthority.activeIndex, Canonical.Context.materialize,
      Canonical.Input.materialize, Canonical.RunningPayload.materialize,
      Canonical.ParentPayload.materialize] using congrFun equations.points child
  · simpa [ConcretePhi81.RunningAuthority.attempt,
      ConcretePhi81.RunningAuthority.children,
      ConcretePhi81.RunningAuthority.activeIndex, Canonical.Context.materialize,
      Canonical.Input.materialize, Canonical.RunningPayload.materialize,
      Canonical.ParentPayload.materialize] using equations.commitment
  · simpa [ConcretePhi81.RunningAuthority.attempt,
      ConcretePhi81.RunningAuthority.children,
      ConcretePhi81.RunningAuthority.activeIndex, Canonical.Context.materialize,
      Canonical.Input.materialize, Canonical.RunningPayload.materialize,
      Canonical.ParentPayload.materialize] using equations.publicInput
  · simpa [ConcretePhi81.RunningAuthority.attempt,
      ConcretePhi81.RunningAuthority.children,
      ConcretePhi81.RunningAuthority.activeIndex, Canonical.Context.materialize,
      Canonical.Input.materialize, Canonical.RunningPayload.materialize,
      Canonical.ParentPayload.materialize] using equations.evaluations

/-- Generic checked-parent acceptance on the canonical carrier implies only
the four retained equations. -/
theorem equations_of_accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Canonical.Context shape domain State publicRingColumns publicFits
        verifierRows}
    (accepted :
      ConcretePhi81.RunningAuthority.Accepted context.materialize) :
    Equations context := by
  cases accepted with
  | bootstrap mode _ =>
      have impossible : RunningMode.active = RunningMode.bootstrap :=
        arity_mode.symm.trans mode
      cases impossible
  | active bound =>
    rcases bound with ⟨active, parent, parentBound, piDec⟩
    have parentEq :
        parent =
          context.input.parent.materialize context.input.system := by
      exact Option.some.inj (by
        simpa [Canonical.Context.materialize] using parentBound.symm)
    subst parent
    refine {
      points := ?_
      commitment := ?_
      publicInput := ?_
      evaluations := ?_
    }
    · funext child
      simpa [ConcretePhi81.RunningAuthority.attempt,
        ConcretePhi81.RunningAuthority.children,
        ConcretePhi81.RunningAuthority.activeIndex, Canonical.Context.materialize,
        Canonical.Input.materialize, Canonical.RunningPayload.materialize,
        Canonical.ParentPayload.materialize] using piDec.samePoint child
    · simpa [ConcretePhi81.RunningAuthority.attempt,
        ConcretePhi81.RunningAuthority.children,
        ConcretePhi81.RunningAuthority.activeIndex, Canonical.Context.materialize,
        Canonical.Input.materialize, Canonical.RunningPayload.materialize,
        Canonical.ParentPayload.materialize] using piDec.commitmentEquation
    · simpa [ConcretePhi81.RunningAuthority.attempt,
        ConcretePhi81.RunningAuthority.children,
        ConcretePhi81.RunningAuthority.activeIndex, Canonical.Context.materialize,
        Canonical.Input.materialize, Canonical.RunningPayload.materialize,
        Canonical.ParentPayload.materialize] using piDec.publicInputEquation
    · simpa [ConcretePhi81.RunningAuthority.attempt,
        ConcretePhi81.RunningAuthority.children,
        ConcretePhi81.RunningAuthority.activeIndex, Canonical.Context.materialize,
        Canonical.Input.materialize, Canonical.RunningPayload.materialize,
        Canonical.ParentPayload.materialize] using piDec.evaluationEquation

/-- Exact retained/eliminated ledger for incoming checked-parent authority. -/
theorem accepted_iff_equations
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      Canonical.Context shape domain State publicRingColumns publicFits
        verifierRows} :
    ConcretePhi81.RunningAuthority.Accepted context.materialize <->
      Equations context := by
  exact ⟨equations_of_accepted, accepted_of_equations⟩

/-- Execute one complete finite comparison for each retained equation family. -/
def check
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Canonical.Context shape domain State publicRingColumns publicFits
        verifierRows) : Bool :=
  functionEqual pointEqual
      (fun child => (context.input.running child).point)
      (fun _ => context.input.parent.point) &&
    (commitmentEqual context.input.parent.commitment
        ((decAlgebra context.key).recomposeCommitment fun child =>
          (context.input.running child).commitment) &&
      (publicInputEqual context.input.parent.publicInput
          ((decAlgebra context.key).recomposePublicInput fun child =>
            (context.input.running child).publicInput) &&
        evaluationsEqual context.input.parent.evaluations
          ((decAlgebra context.key).recomposeEvaluations fun child =>
            (context.input.running child).evaluations)))

/-- The executable checker accepts exactly the four retained equations. -/
theorem check_eq_true_iff_equations
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Canonical.Context shape domain State publicRingColumns publicFits
        verifierRows) :
    check context = true <-> Equations context := by
  simp only [check, Bool.and_eq_true,
    functionEqual_eq_true_iff pointEqual pointEqual_eq_true_iff,
    commitmentEqual_eq_true_iff, publicInputEqual_eq_true_iff,
    evaluationsEqual_eq_true_iff]
  constructor
  · rintro ⟨points, commitment, publicInput, evaluations⟩
    exact ⟨points, commitment, publicInput, evaluations⟩
  · intro equations
    exact ⟨equations.points, equations.commitment, equations.publicInput,
      equations.evaluations⟩

/-- The Boolean checker is exact to the pre-existing generic checked-parent
acceptance predicate on the canonical carrier. -/
theorem check_eq_true_iff_accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      Canonical.Context shape domain State publicRingColumns publicFits
        verifierRows) :
    check context = true <->
      ConcretePhi81.RunningAuthority.Accepted context.materialize := by
  rw [check_eq_true_iff_equations, accepted_iff_equations]

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.RunningAuthority
