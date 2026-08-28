import Mathlib.Data.List.GetD
import NightstreamFPrime.Layout.BalancedTernary

/-!
Owns the three assignment-slot encodings used by the production low-norm
compiler. Public bits and already-centered units remain one coordinate.
General Goldilocks values use the exact 41-trit balanced encoding.

This module does not select retained source fields or assign final columns.
-/

namespace NightstreamFPrime.Layout.LowNormSlot

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix

/-- The complete production slot vocabulary. -/
inductive Kind where
  | bit
  | centered
  | field
deriving Repr, DecidableEq

/-- Exact committed width of one slot. -/
def Kind.width : Kind → Nat
  | .bit | .centered => 1
  | .field => BalancedTernary.width

/-- Canonical witness coordinates for one retained value. -/
def encode : Kind → F → List F
  | .bit, value | .centered, value => [value]
  | .field, value => BalancedTernary.digits value

/-- One indexed coordinate of the canonical slot encoding. -/
def coordinate (kind : Kind) (value : F) (index : Fin kind.width) : F :=
  (encode kind value).getD index.val 0

/-- Source condition required before a one-coordinate slot is admissible.
General fields are canonical by their exact balanced encoding. -/
def Valid : Kind → F → Prop
  | .bit, value => value = 0 ∨ value = 1
  | .centered, value => centeredMagnitude value < 2
  | .field, _ => True

@[simp] theorem encode_length (kind : Kind) (value : F) :
    (encode kind value).length = kind.width := by
  cases kind <;> simp [encode, Kind.width, BalancedTernary.digits_length]

/-- Enumerating the exact indexed-coordinate view recovers the reference
encoding list. -/
theorem coordinateList_eq_encode (kind : Kind) (value : F) :
    List.ofFn (coordinate kind value) = encode kind value := by
  apply List.ext_get
  · simp [encode_length]
  · intro index leftBound rightBound
    simp only [List.get_eq_getElem, List.getElem_ofFn, coordinate]
    exact List.getD_eq_getElem (l := encode kind value) (d := 0) rightBound

/-- Every valid production slot satisfies the exact fresh-opening norm. -/
theorem encode_norm (kind : Kind) (value : F) (valid : Valid kind value) :
    normBounded 2 (encode kind value) := by
  intro coordinate member
  cases kind with
  | bit =>
      rcases valid with rfl | rfl
      · simp [encode] at member
        subst coordinate
        decide
      · simp [encode] at member
        subst coordinate
        decide
  | centered =>
      simp [encode] at member
      subst coordinate
      exact valid
  | field =>
      exact BalancedTernary.digits_norm value coordinate member

/-- A general-field slot reconstructs its exact source value. -/
@[simp] theorem encode_field_recompose (value : F) :
    BalancedTernary.recompose (encode .field value) = value :=
  BalancedTernary.recompose_digits value

/-- Every production slot reconstructs its exact source value. -/
@[simp] theorem recompose_encode (kind : Kind) (value : F) :
    BalancedTernary.recompose (encode kind value) = value := by
  cases kind with
  | bit =>
      change value + fieldOfNat 3 * 0 = value
      rw [Fin.mul_zero, Fin.add_zero]
  | centered =>
      change value + fieldOfNat 3 * 0 = value
      rw [Fin.mul_zero, Fin.add_zero]
  | field => exact encode_field_recompose value

end NightstreamFPrime.Layout.LowNormSlot
