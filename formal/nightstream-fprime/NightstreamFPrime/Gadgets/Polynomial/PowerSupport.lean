import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Gadgets.Polynomial.Power

/-!
Owns variable-support propagation for the fixed-exponent power circuit.
The coefficient list and Horner evaluation remain unchanged.
-/

namespace NightstreamFPrime.Gadgets.Polynomial.Power

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial

private theorem coefficientExprs_supported (exponent : Nat)
    (allowed : Nat → Prop) :
    ∀ coefficient ∈ coefficientExprs exponent,
      Horner.KSupported coefficient allowed := by
  intro coefficient member
  simp only [coefficientExprs, List.mem_append, List.mem_replicate,
    List.mem_singleton] at member
  rcases member with ⟨_, rfl⟩ | rfl
  · exact Horner.KSupported.zero allowed
  · exact Horner.KSupported.one allowed

/-- Exact support propagation through one fixed-exponent power circuit. -/
theorem flatConstraints_varsSatisfy (exponent : Nat)
    (interface : Interface) (offset : Nat) (allowed : Nat → Prop)
    (pointSupport : Horner.KSupported (interface.point offset) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (circuit exponent interface).main offset) →
      allowed index) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (circuit exponent interface).main offset),
      expression.VarsSatisfy allowed := by
  have supported := Horner.Owned.flatConstraints_varsSatisfy
    (hornerInterface exponent interface) offset allowed
    (by simpa [hornerInterface] using pointSupport)
    (coefficientExprs_supported exponent allowed)
    (by
      intro index lower upper
      apply localSupport index lower
      simpa [circuit] using upper)
  simpa [circuit] using supported

/-- The fixed-power output uses the same point support and exact Horner
interval as its rows. -/
theorem output_varsSatisfy (exponent : Nat) (interface : Interface)
    (offset : Nat) (allowed : Nat → Prop)
    (pointSupport : Horner.KSupported (interface.point offset) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (circuit exponent interface).main offset) →
      allowed index) :
    Horner.KSupported (output exponent interface offset) allowed := by
  have supported := Horner.Owned.output_varsSatisfy
    (hornerInterface exponent interface) offset allowed
    (by simpa [hornerInterface] using pointSupport)
    (coefficientExprs_supported exponent allowed)
    (by
      intro index lower upper
      apply localSupport index lower
      simpa [circuit] using upper)
  simpa [output] using supported

end NightstreamFPrime.Gadgets.Polynomial.Power
