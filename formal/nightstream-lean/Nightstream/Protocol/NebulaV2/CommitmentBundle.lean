import Mathlib.LinearAlgebra.Pi
import Mathlib.Tactic.DeriveFintype
import Nightstream.Protocol.NebulaV2.Lifecycle

/-!
Contract: atomic product-commitment semantics for V2.

Assurance tier: model-level.

Owns the mandatory four-component product type, its construction as one
linear map from one assignment, component-complete forwarding, and common
linear-combination behavior.

Does not own Ajtai arithmetic, bounded-opening extraction, seeded setup,
PiCCS/PiRLC/PiDEC circuit rows, or a Module-SIS reduction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.CommitmentBundle

inductive Component where
  | full
  | operations
  | initialSnapshot
  | finalSnapshot
deriving DecidableEq, Fintype, Repr

/-- A total function makes all four components mandatory. -/
abbrev Bundle (Commitment : Type) := Component → Commitment

theorem component_count : Fintype.card Component = 4 := by
  decide

section Linear

variable {Scalar Assignment Commitment Lane : Type}
variable [Semiring Scalar]
variable [AddCommMonoid Assignment] [Module Scalar Assignment]
variable [AddCommMonoid Commitment] [Module Scalar Commitment]
variable [AddCommMonoid Lane] [Module Scalar Lane]

/-- One authority-bearing product map from one assignment. -/
def productMap
    (components : Component → Assignment →ₗ[Scalar] Commitment) :
    Assignment →ₗ[Scalar] Bundle Commitment :=
  LinearMap.pi components

theorem productMap_component
    (components : Component → Assignment →ₗ[Scalar] Commitment)
    (assignment : Assignment) (component : Component) :
    productMap components assignment component =
      components component assignment :=
  rfl

/-- A projected lane commitment remains a linear map on the full assignment. -/
def projectedMap
    (projection : Assignment →ₗ[Scalar] Lane)
    (commit : Lane →ₗ[Scalar] Commitment) :
    Assignment →ₗ[Scalar] Commitment :=
  commit.comp projection

/-- The same public linear combination acts on every bundle component. This
is the common algebraic obligation used by PiRLC and PiDEC. -/
theorem productMap_linear_combination
    {Index : Type} [Fintype Index]
    (components : Component → Assignment →ₗ[Scalar] Commitment)
    (coefficient : Index → Scalar)
    (assignments : Index → Assignment) :
    productMap components (∑ index, coefficient index • assignments index) =
      ∑ index, coefficient index • productMap components (assignments index) := by
  simp

end Linear

/-- Native and recursive PiCCS must establish this predicate, not only equality
of the full component. -/
def ForwardsExactly
    {Commitment : Type} (input output : Bundle Commitment) : Prop :=
  ∀ component, output component = input component

theorem eq_of_forwardsExactly
    {Commitment : Type}
    {input output : Bundle Commitment}
    (forwards : ForwardsExactly input output) :
    output = input := by
  funext component
  exact forwards component

/-- Terminal opening uses one assignment for all components. -/
def OpensAll
    {Scalar Assignment Commitment : Type}
    [Semiring Scalar]
    [AddCommMonoid Assignment] [Module Scalar Assignment]
    [AddCommMonoid Commitment] [Module Scalar Commitment]
    (components : Component → Assignment →ₗ[Scalar] Commitment)
    (assignment : Assignment)
    (bundle : Bundle Commitment) : Prop :=
  productMap components assignment = bundle

theorem opensAll_component
    {Scalar Assignment Commitment : Type}
    [Semiring Scalar]
    [AddCommMonoid Assignment] [Module Scalar Assignment]
    [AddCommMonoid Commitment] [Module Scalar Commitment]
    {components : Component → Assignment →ₗ[Scalar] Commitment}
    {assignment : Assignment}
    {bundle : Bundle Commitment}
    (opens : OpensAll components assignment bundle)
    (component : Component) :
    components component assignment = bundle component := by
  exact congrFun opens component

/-- Named computational boundary for two different bounded inputs that have
the same output under one map. Linearity alone does not exclude this event. -/
def MapBindingFailure
    {Assignment Output : Type}
    (bounded : Assignment → Prop)
    (commit : Assignment → Output) : Prop :=
  ∃ left right,
    bounded left ∧ bounded right ∧ left ≠ right ∧
      commit left = commit right

/-- Specialization to the complete four-component map. -/
abbrev BindingFailure
    {Assignment Commitment : Type}
    (bounded : Assignment → Prop)
    (commit : Assignment → Bundle Commitment) : Prop :=
  MapBindingFailure bounded commit

/-- A collision in the complete bundle gives a collision in each component,
including the full-assignment component. Thus a zero product map cannot pass
the binding interface only because it is linear. -/
theorem bindingFailure_implies_component_failure
    {Assignment Commitment : Type}
    {bounded : Assignment → Prop}
    {commit : Assignment → Bundle Commitment}
    (failure : BindingFailure bounded commit)
    (component : Component) :
    MapBindingFailure bounded (fun assignment => commit assignment component) := by
  rcases failure with
    ⟨left, right, leftBounded, rightBounded, different, equalBundle⟩
  exact
    ⟨left, right, leftBounded, rightBounded, different,
      congrFun equalBundle component⟩

theorem no_full_component_failure_implies_bundle_binding
    {Assignment Commitment : Type}
    {bounded : Assignment → Prop}
    {commit : Assignment → Bundle Commitment}
    (fullSecure :
      ¬ MapBindingFailure bounded (fun assignment => commit assignment .full)) :
    ¬ BindingFailure bounded commit := by
  intro failure
  exact fullSecure
    (bindingFailure_implies_component_failure failure .full)

theorem assignment_eq_or_bindingFailure
    {Assignment Commitment : Type}
    (bounded : Assignment → Prop)
    (commit : Assignment → Bundle Commitment)
    {left right : Assignment}
    (leftBounded : bounded left)
    (rightBounded : bounded right)
    (equalCommitment : commit left = commit right) :
    left = right ∨ BindingFailure bounded commit := by
  by_cases equal : left = right
  · exact Or.inl equal
  · exact Or.inr
      ⟨left, right, leftBounded, rightBounded, equal, equalCommitment⟩

end Nightstream.Protocol.NebulaV2.CommitmentBundle
