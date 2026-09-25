import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim

/-!
Owns variable-support propagation for the PiCCS initial-claim Horner leaf.
It changes no coefficient order, circuit, or row.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Circuit.SupportRange
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem coefficientExprs_supported (interface : Interface)
    (offset : Nat) (allowed : Nat → Prop)
    (evalKSupport : ∀ coordinate,
      Horner.KSupported (interface.eval_K offset coordinate) allowed)
    (evalASupport : ∀ coordinate,
      Horner.KSupported (interface.eval_A offset coordinate) allowed) :
    ∀ coefficient ∈ coefficientExprs interface offset,
      Horner.KSupported coefficient allowed := by
  intro coefficient member
  rw [coefficientExprs, List.mem_append] at member
  rcases member with member | member
  · rw [List.mem_map] at member
    rcases member with ⟨coordinate, _coordinateMember, rfl⟩
    exact evalKSupport coordinate
  · rw [List.mem_map] at member
    rcases member with ⟨coordinate, _coordinateMember, rfl⟩
    exact evalASupport coordinate

/-- Exact support propagation through the child-owned 12,960-coefficient
Horner program. -/
theorem flatConstraints_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (gammaSupport : Horner.KSupported (interface.gamma offset) allowed)
    (evalKSupport : ∀ coordinate,
      Horner.KSupported (interface.eval_K offset coordinate) allowed)
    (evalASupport : ∀ coordinate,
      Horner.KSupported (interface.eval_A offset coordinate) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + (program interface offset).recipes.length →
      allowed index) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (circuit interface).main offset),
      expression.VarsSatisfy allowed := by
  have supported := Horner.compile_recipeConstraints_varsSatisfy offset
    (interface.gamma offset) (coefficientExprs interface offset) allowed
    gammaSupport
    (coefficientExprs_supported interface offset allowed evalKSupport
      evalASupport)
  rw [circuit, Horner.Owned.circuit_ops,
    Horner.Owned.flatConstraints_opsAt]
  intro expression member
  apply Expr.VarsSatisfy.mono expression (supported expression member)
  intro index support
  rcases support with support | ⟨lower, upper⟩
  · exact support
  · exact localSupport index lower (by
      simpa [program, ownedInterface, Horner.Owned.program] using upper)

/-- The child-owned initial claim has the same exact support as its rows. -/
theorem output_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (gammaSupport : Horner.KSupported (interface.gamma offset) allowed)
    (evalKSupport : ∀ coordinate,
      Horner.KSupported (interface.eval_K offset coordinate) allowed)
    (evalASupport : ∀ coordinate,
      Horner.KSupported (interface.eval_A offset coordinate) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + (program interface offset).recipes.length →
      allowed index) :
    Horner.KSupported (output interface offset) allowed := by
  exact Horner.Owned.output_varsSatisfy (ownedInterface interface) offset
    allowed gammaSupport
    (coefficientExprs_supported interface offset allowed evalKSupport
      evalASupport) (by
        intro index lower upper
        exact localSupport index lower (by
          simpa [program, ownedInterface, Horner.Owned.program] using upper))

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim
