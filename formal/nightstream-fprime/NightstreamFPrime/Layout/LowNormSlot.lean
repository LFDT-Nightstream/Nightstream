import Mathlib.Data.List.GetD
import NightstreamFPrime.Layout.BalancedTernary

namespace NightstreamFPrime.Layout.LowNormSlot

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix

inductive Kind where
  | bit
  | centered
  | field
deriving Repr, DecidableEq

def Kind.width : Kind → Nat
  | .bit | .centered => 1
  | .field => BalancedTernary.width

def encode : Kind → F → List F
  | .bit, value | .centered, value => [value]
  | .field, value => BalancedTernary.digits value

def coordinate (kind : Kind) (value : F) (index : Fin kind.width) : F :=
  (encode kind value).getD index.val 0

theorem coordinate_cast {left right : Kind} (equal : left = right)
    (value : F) (index : Fin left.width) :
    coordinate right value
        (Fin.cast (congrArg Kind.width equal) index) =
      coordinate left value index := by
  cases equal
  rfl

def Valid : Kind → F → Prop
  | .bit, value => value = 0 ∨ value = 1
  | .centered, value => centeredMagnitude value < 2
  | .field, _ => True

@[simp] theorem encode_length (kind : Kind) (value : F) :
    (encode kind value).length = kind.width := by
  cases kind <;> simp [encode, Kind.width, BalancedTernary.digits_length]

theorem coordinateList_eq_encode (kind : Kind) (value : F) :
    List.ofFn (coordinate kind value) = encode kind value := by
  apply List.ext_get
  · simp [encode_length]
  · intro index leftBound rightBound
    simp only [List.get_eq_getElem, List.getElem_ofFn, coordinate]
    exact List.getD_eq_getElem (l := encode kind value) (d := 0) rightBound

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

@[simp] theorem encode_field_recompose (value : F) :
    BalancedTernary.recompose (encode .field value) = value :=
  BalancedTernary.recompose_digits value

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
