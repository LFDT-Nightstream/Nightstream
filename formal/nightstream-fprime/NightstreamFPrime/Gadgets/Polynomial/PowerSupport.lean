import NightstreamFPrime.Gadgets.Polynomial.HornerSupport
import NightstreamFPrime.Gadgets.Polynomial.Power

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
      simpa [circuit] using! upper)
  simpa [circuit] using supported

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
      simpa [circuit] using! upper)
  simpa [output] using supported

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

theorem marks_nil (exponent : Nat) (interface : Interface) (offset : Nat) :
    marks (Circuit.ops (circuit exponent interface).main offset) = [] :=
  Horner.Owned.marks_nil (hornerInterface exponent interface) offset

/-- The rows read the point and the gadget's own locals. -/
theorem free (exponent : Nat) (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (point : Horner.KSupported (interface.point offset) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit exponent interface).main offset) →
        live column) :
    ∀ row ∈ flatConstraints (Circuit.ops (circuit exponent interface).main offset),
      row.VarsSatisfy live :=
  flatConstraints_varsSatisfy exponent interface offset live point locals

theorem output_live (exponent : Nat) (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (point : Horner.KSupported (interface.point offset) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit exponent interface).main offset) →
        live column) :
    Horner.KSupported (output exponent interface offset) live :=
  output_varsSatisfy exponent interface offset live point locals

end NightstreamFPrime.Gadgets.Polynomial.Power
