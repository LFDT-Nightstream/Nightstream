import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal

/-!
Owns variable-support propagation for the production PiCCS norm terminal.
It changes no source order, cubic residual, circuit, or row.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal

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

/-- The support-facing operation list is definitionally the canonical norm
circuit operation list. Keeping this equation separate avoids expanding the
fixed 17-term Horner program while elaborating support theorem types. -/
theorem circuit_ops_eq_ownedOps (interface : Interface) (offset : Nat) :
    Circuit.ops (circuit interface).main offset =
      Horner.Owned.opsAt (ownedInterface interface) offset := by
  rfl

/-- Exact support propagation through the production norm terminal. -/
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

/-- The child-owned strict norm output preserves the exact row support. -/
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

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal
