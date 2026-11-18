import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial

private theorem residualExpr_supported (value : KExpr)
    (allowed : Nat → Prop) (support : Horner.KSupported value allowed) :
    Horner.KSupported (residualExpr value) allowed := by
  unfold residualExpr
  exact Horner.KSupported.mul
    (Horner.KSupported.mul
      (Horner.KSupported.add support (Horner.KSupported.one allowed)) support)
    (Horner.KSupported.sub support (Horner.KSupported.one allowed))

private theorem coefficientExprs_supported (interface : Interface)
    (offset : Nat) (allowed : Nat → Prop)
    (sourceSupport : ∀ source,
      Horner.KSupported (interface.sourceAssignment offset source) allowed) :
    ∀ coefficient ∈ coefficientExprs interface offset,
      Horner.KSupported coefficient allowed := by
  intro coefficient member
  rw [coefficientExprs, List.mem_map] at member
  rcases member with ⟨source, _sourceMember, rfl⟩
  exact residualExpr_supported _ allowed (sourceSupport source)

theorem circuit_ops_eq_ownedOps (interface : Interface) (offset : Nat) :
    Circuit.ops (circuit interface).main offset =
      Horner.Owned.opsAt (ownedInterface interface) offset := by
  rfl

theorem flatConstraints_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (gammaSupport : Horner.KSupported (interface.gamma offset) allowed)
    (sourceSupport : ∀ source,
      Horner.KSupported (interface.sourceAssignment offset source) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset +
        (Horner.Owned.program (ownedInterface interface) offset).recipes.length →
      allowed index) :
    ∀ expression ∈ flatConstraints
        (Horner.Owned.opsAt (ownedInterface interface) offset),
      expression.VarsSatisfy allowed := by
  have supported := Horner.Owned.flatConstraints_varsSatisfy
    (ownedInterface interface) offset allowed
    gammaSupport
    (coefficientExprs_supported interface offset allowed sourceSupport)
    (by
      intro index lower upper
      exact localSupport index lower upper)
  rw [Horner.Owned.circuit_ops] at supported
  exact supported

theorem output_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (gammaSupport : Horner.KSupported (interface.gamma offset) allowed)
    (sourceSupport : ∀ source,
      Horner.KSupported (interface.sourceAssignment offset source) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset +
        (Horner.Owned.program (ownedInterface interface) offset).recipes.length →
      allowed index) :
    Horner.KSupported (output interface offset) allowed := by
  exact Horner.Owned.output_varsSatisfy (ownedInterface interface) offset
    allowed gammaSupport
    (coefficientExprs_supported interface offset allowed sourceSupport)
    localSupport

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

section Exporter

variable {shape : Spec.Folding.PiCCS.PaperJoint.Shape}

theorem coefficientExprs_live (interface : ShapeInterface shape) (offset : Nat)
    {live : Nat → Prop}
    (source : ∀ source, Horner.KSupported (interface.sourceAssignment offset source) live) :
    ∀ coefficient ∈ coefficientExprs interface offset, Horner.KSupported coefficient live := by
  intro coefficient member
  obtain ⟨index, _, rfl⟩ := List.mem_map.mp member
  exact residualExpr_supported _ live (source index)

theorem marks_nil (interface : ShapeInterface shape) (offset : Nat) :
    marks (Circuit.ops (circuit interface).main offset) = [] :=
  Horner.Owned.marks_nil (ownedInterface interface) offset

/-- The rows read γ, the source assignments and the gadget's own locals. -/
theorem free (interface : ShapeInterface shape) (offset : Nat) {live : Nat → Prop}
    (gamma : Horner.KSupported (interface.gamma offset) live)
    (source : ∀ source, Horner.KSupported (interface.sourceAssignment offset source) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit interface).main offset) → live column) :
    ∀ row ∈ flatConstraints (Circuit.ops (circuit interface).main offset), row.VarsSatisfy live :=
  Horner.Owned.free (ownedInterface interface) offset gamma
    (coefficientExprs_live interface offset source) locals

theorem output_live (interface : ShapeInterface shape) (offset : Nat) {live : Nat → Prop}
    (gamma : Horner.KSupported (interface.gamma offset) live)
    (source : ∀ source, Horner.KSupported (interface.sourceAssignment offset source) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit interface).main offset) → live column) :
    Horner.KSupported (output interface offset) live :=
  Horner.Owned.output_live (ownedInterface interface) offset gamma
    (coefficientExprs_live interface offset source) locals

end Exporter

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal
