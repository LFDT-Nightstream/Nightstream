import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongProbability

/-! Finite indices for the actual PiCCS coins in the B.1 probability coupling. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.VerifierCoinSpace

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction

abbrev Request (shape : Shape) :=
  (Fin shape.cubeVariables → K) × K × (Fin shape.cubeVariables → K)

noncomputable instance requestFintype (shape : Shape) : Fintype (Request shape) := by
  letI : Fintype K := Fintype.ofEquiv (F × F) {
    toFun := fun value => ⟨value.1, value.2⟩
    invFun := fun value => (value.c0, value.c1)
    left_inv := fun _ => rfl
    right_inv := fun _ => rfl }
  exact inferInstanceAs (Fintype ((Fin shape.cubeVariables → K) × K ×
    (Fin shape.cubeVariables → K)))

def values {dimensionCount : Nat} (point : CubePoint K dimensionCount) : Fin dimensionCount → K :=
  fun index => point.coordinates[index.val]'(by rw [point.dimension]; exact index.isLt)

def point {dimensionCount : Nat} (coordinates : Fin dimensionCount → K) : CubePoint K dimensionCount :=
  ⟨List.ofFn coordinates, List.length_ofFn⟩

private theorem point_values {dimensionCount : Nat} (input : CubePoint K dimensionCount) :
    point (values input) = input := by
  cases input with
  | mk coordinates dimension =>
      subst dimensionCount
      unfold point values
      congr 1
      exact List.ofFn_get coordinates

def request {shape : Shape} (coins : PublicCoins K shape) : Request shape :=
  (values coins.alpha, coins.gamma, values coins.roundPoint)

def coins {shape : Shape} (input : Request shape) : PublicCoins K shape :=
  ⟨point input.1, input.2.1, point input.2.2⟩

/-- Looking up a coupled suffix never substitutes another verifier coin. -/
theorem coins_request {shape : Shape} (input : PublicCoins K shape) :
    coins (request input) = input := by
  cases input
  simp only [coins, request, point_values]

end NightstreamFPrime.Spec.Folding.Nifs.VerifierCoinSpace
