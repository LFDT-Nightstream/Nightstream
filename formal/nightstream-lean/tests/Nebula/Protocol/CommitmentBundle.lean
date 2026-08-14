import Nightstream.Protocol.Nebula

set_option autoImplicit false

namespace tests.NebulaCommitmentBundle

open Nightstream.Protocol.Nebula.CommitmentBundle

def identityComponents (_component : Component) : ℚ →ₗ[ℚ] ℚ :=
  LinearMap.id

def coefficients : Fin 2 → ℚ
  | 0 => 2
  | _ => 3

def assignments : Fin 2 → ℚ
  | 0 => 5
  | _ => 7

theorem all_components_use_the_same_linear_combination :
    productMap identityComponents
        (∑ index, coefficients index • assignments index) =
      ∑ index,
        coefficients index • productMap identityComponents (assignments index) :=
  productMap_linear_combination identityComponents coefficients assignments

theorem one_assignment_opens_every_component
    (component : Component) :
    identityComponents component 9 = productMap identityComponents 9 component := by
  rfl

/- Equality of only the full component does not forward an atomic bundle. -/
namespace MissingComponentForwarding

def input : Bundle Nat
  | .full => 10
  | .operations => 20
  | .initialSnapshot => 30
  | .finalSnapshot => 40

def output : Bundle Nat
  | .full => 10
  | .operations => 21
  | .initialSnapshot => 30
  | .finalSnapshot => 40

theorem full_component_is_equal : output .full = input .full := rfl

theorem bundles_are_not_equal : output ≠ input := by
  intro equal
  have operationsEqual := congrFun equal Component.operations
  change 21 = 20 at operationsEqual
  omega

theorem exact_forwarding_fails : ¬ ForwardsExactly input output := by
  intro forwards
  exact bundles_are_not_equal (eq_of_forwardsExactly forwards)

end MissingComponentForwarding

namespace NamedBindingBoundary

def bounded (value : Nat) : Prop := value < 4

def parityBundle (value : Nat) : Bundle Nat :=
  fun _component => value % 2

theorem parity_has_a_binding_failure :
    BindingFailure bounded parityBundle := by
  exact ⟨0, 2,
    by change 0 < 4; decide,
    by change 2 < 4; decide,
    by decide,
    by decide⟩

end NamedBindingBoundary

namespace ComponentBindingReduction

variable {Assignment Commitment : Type}
variable (bounded : Assignment → Prop)
variable (commit : Assignment → Bundle Commitment)

example (failure : BindingFailure bounded commit) (component : Component) :
    MapBindingFailure bounded
      (fun assignment => commit assignment component) :=
  bindingFailure_implies_component_failure failure component

example
    (fullSecure :
      ¬ MapBindingFailure bounded
        (fun assignment => commit assignment .full)) :
    ¬ BindingFailure bounded commit :=
  no_full_component_failure_implies_bundle_binding fullSecure

end ComponentBindingReduction

end tests.NebulaCommitmentBundle
