import NightstreamFPrime.Gadgets.Range.CanonicalPublicU64
import NightstreamFPrime.Layout.Range.CanonicalU64

/-!
Owns physical lowering for one canonical word bound to 64 caller-owned public
bits. The canonical child contributes 197 fresh columns and 264 physical rows;
the 64 equality rows are affine and add no fresh column.
-/

namespace NightstreamFPrime.Layout.Range.CanonicalPublicU64

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Range
open NightstreamFPrime.Layout

structure InputsAffine (interface : CanonicalPublicU64.Interface)
    (offset : Nat) : Prop where
  source : R1CS.IsAffine (interface.source offset)
  bit : ∀ index, index < CanonicalPublicU64.bitCount →
    R1CS.IsAffine (interface.bit offset index)

def logicalConstraints (interface : CanonicalPublicU64.Interface)
    (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops (CanonicalPublicU64.main interface) offset)

private theorem childInputs
    (interface : CanonicalPublicU64.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    NightstreamFPrime.Layout.Range.CanonicalU64.InputsAffine
      (CanonicalPublicU64.childInterface interface offset) offset :=
  ⟨inputs.source⟩

private theorem bindingAffine
    (interface : CanonicalPublicU64.Interface) (offset index : Nat)
    (inputs : InputsAffine interface offset)
    (bounded : index < CanonicalPublicU64.bitCount) :
    R1CS.IsAffine (CanonicalPublicU64.bindingConstraint interface offset index) := by
  unfold CanonicalPublicU64.bindingConstraint
  exact R1CS.IsAffine.add (inputs.bit index bounded)
    (R1CS.IsAffine.const_mul (-1) (R1CS.isAffine_var _))

private theorem bindingFreshCount_eq
    (interface : CanonicalPublicU64.Interface) (offset index : Nat)
    (inputs : InputsAffine interface offset)
    (bounded : index < CanonicalPublicU64.bitCount) :
    R1CS.constraintFreshCount
      (CanonicalPublicU64.bindingConstraint interface offset index) = 0 :=
  R1CS.constraintFreshCount_eq_zero_of_affine _
    (bindingAffine interface offset index inputs bounded)

private theorem bindingRowCount_eq
    (interface : CanonicalPublicU64.Interface) (offset index : Nat)
    (inputs : InputsAffine interface offset)
    (bounded : index < CanonicalPublicU64.bitCount) :
    R1CS.constraintRowCount
      (CanonicalPublicU64.bindingConstraint interface offset index) = 1 :=
  R1CS.constraintRowCount_eq_one_of_affine _
    (bindingAffine interface offset index inputs bounded)

private theorem bindingFreshTotal_eq
    (interface : CanonicalPublicU64.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalFreshCount
      ((List.range CanonicalPublicU64.bitCount).map
        (CanonicalPublicU64.bindingConstraint interface offset)) = 0 := by
  unfold R1CS.totalFreshCount
  rw [List.map_map]
  apply List.sum_eq_zero
  intro value member
  rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
  exact bindingFreshCount_eq interface offset index inputs
    (List.mem_range.mp indexMember)

private theorem bindingRowTotal_eq
    (interface : CanonicalPublicU64.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalRowCount
      ((List.range CanonicalPublicU64.bitCount).map
        (CanonicalPublicU64.bindingConstraint interface offset)) = 64 := by
  unfold R1CS.totalRowCount
  rw [List.map_map]
  have pointwise : List.map
      (R1CS.constraintRowCount ∘
        CanonicalPublicU64.bindingConstraint interface offset)
      (List.range CanonicalPublicU64.bitCount) =
      List.map (fun _ => 1) (List.range CanonicalPublicU64.bitCount) := by
    apply List.map_congr_left
    intro index member
    exact bindingRowCount_eq interface offset index inputs
      (List.mem_range.mp member)
  rw [pointwise]
  simp [CanonicalPublicU64.bitCount]
  rfl

theorem totalFreshCount_eq
    (interface : CanonicalPublicU64.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalFreshCount (logicalConstraints interface offset) = 197 := by
  unfold logicalConstraints
  change R1CS.totalFreshCount
    (flatConstraints (CanonicalPublicU64.opsAt interface offset)) = 197
  rw [CanonicalPublicU64.flatConstraints_opsAt,
    R1CS.totalFreshCount_append]
  change R1CS.totalFreshCount
      (NightstreamFPrime.Layout.Range.CanonicalU64.logicalConstraints
        (CanonicalPublicU64.childInterface interface offset) offset) + _ = _
  rw [NightstreamFPrime.Layout.Range.CanonicalU64.totalFreshCount_eq
      _ _ (childInputs interface offset inputs),
    bindingFreshTotal_eq interface offset inputs]

theorem totalRowCount_eq
    (interface : CanonicalPublicU64.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalRowCount (logicalConstraints interface offset) = 328 := by
  unfold logicalConstraints
  change R1CS.totalRowCount
    (flatConstraints (CanonicalPublicU64.opsAt interface offset)) = 328
  rw [CanonicalPublicU64.flatConstraints_opsAt,
    R1CS.totalRowCount_append]
  change R1CS.totalRowCount
      (NightstreamFPrime.Layout.Range.CanonicalU64.logicalConstraints
        (CanonicalPublicU64.childInterface interface offset) offset) + _ = _
  rw [NightstreamFPrime.Layout.Range.CanonicalU64.totalRowCount_eq
      _ _ (childInputs interface offset inputs),
    bindingRowTotal_eq interface offset inputs]

end NightstreamFPrime.Layout.Range.CanonicalPublicU64
