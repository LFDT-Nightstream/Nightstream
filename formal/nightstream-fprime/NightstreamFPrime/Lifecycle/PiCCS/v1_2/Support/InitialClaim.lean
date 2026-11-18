import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim

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

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

section Exporter

variable {shape : Shape}

theorem coefficientExprs_live (interface : ShapeInterface shape) (offset : Nat)
    {live : Nat → Prop}
    (evalK : ∀ coordinate, Horner.KSupported (interface.eval_K offset coordinate) live)
    (evalA : ∀ coordinate, Horner.KSupported (interface.eval_A offset coordinate) live) :
    ∀ coefficient ∈ coefficientExprs interface offset, Horner.KSupported coefficient live := by
  intro coefficient member
  rw [coefficientExprs, List.mem_append] at member
  rcases member with member | member
  · obtain ⟨coordinate, _, rfl⟩ := List.mem_map.mp member
    exact evalK coordinate
  · obtain ⟨coordinate, _, rfl⟩ := List.mem_map.mp member
    exact evalA coordinate

theorem marks_nil (interface : ShapeInterface shape) (offset : Nat) :
    marks (Circuit.ops (circuit interface).main offset) = [] :=
  Horner.Owned.marks_nil (ownedInterface interface) offset

/-- The rows read γ, the evaluations and the gadget's own locals. -/
theorem free (interface : ShapeInterface shape) (offset : Nat) {live : Nat → Prop}
    (gamma : Horner.KSupported (interface.gamma offset) live)
    (evalK : ∀ coordinate, Horner.KSupported (interface.eval_K offset coordinate) live)
    (evalA : ∀ coordinate, Horner.KSupported (interface.eval_A offset coordinate) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit interface).main offset) → live column) :
    ∀ row ∈ flatConstraints (Circuit.ops (circuit interface).main offset), row.VarsSatisfy live :=
  Horner.Owned.free (ownedInterface interface) offset gamma
    (coefficientExprs_live interface offset evalK evalA) locals

theorem output_live (interface : ShapeInterface shape) (offset : Nat) {live : Nat → Prop}
    (gamma : Horner.KSupported (interface.gamma offset) live)
    (evalK : ∀ coordinate, Horner.KSupported (interface.eval_K offset coordinate) live)
    (evalA : ∀ coordinate, Horner.KSupported (interface.eval_A offset coordinate) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit interface).main offset) → live column) :
    Horner.KSupported (output interface offset) live :=
  Horner.Owned.output_live (ownedInterface interface) offset gamma
    (coefficientExprs_live interface offset evalK evalA) locals

end Exporter

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim
