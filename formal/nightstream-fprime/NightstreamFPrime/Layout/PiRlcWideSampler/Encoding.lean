import NightstreamFPrime.Layout.PiRlcWideSampler.Witness

/-! The constructive assignment places every retained block at its proved
coordinates. This closes the source-to-coordinate witness mapping. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.Encoding

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open ProductionRelation BatchPlan Witness
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

theorem poseidon_coordinate (initial : FieldState) (slot : Fin (34 * 86)) (digit : Fin 41) :
    coordinate initial ⟨slot.val * 41 + digit.val, by omega⟩ =
      LowNormSlot.coordinate .field (sboxValues initial slot) digit := by
  unfold coordinate
  rw [dif_pos (by dsimp only; omega)]
  have quotient : (slot.val * 41 + digit.val) / 41 = slot.val := by omega
  have remainder : (slot.val * 41 + digit.val) % 41 = digit.val := by omega
  simp only [quotient, remainder, Fin.eta]

theorem range_coordinate (initial : FieldState) (source : Fin 17) (position : Fin 937) :
    coordinate initial ⟨119884 + source.val * 937 + position.val, by omega⟩ =
      rangeCoordinate (rangeSources initial source) position := by
  unfold coordinate
  rw [dif_neg (by dsimp only; omega)]
  have quotient : (119884 + source.val * 937 + position.val - 119884) / 937 = source.val := by omega
  have remainder : (119884 + source.val * 937 + position.val - 119884) % 937 = position.val := by omega
  simp only [quotient, remainder]

theorem bits_coordinate (env : Env) (slot : Fin 256) :
    rangeCoordinate env ⟨slot.val, by omega⟩ =
      LowNormSlot.coordinate .bit (env (Retained.canonicalBits.source slot).val) ⟨0, by decide⟩ := by
  unfold rangeCoordinate
  rw [dif_pos slot.isLt]
  simp only [Fin.eta]

theorem fields_coordinate (env : Env) (slot : Fin 8) (digit : Fin 41) :
    rangeCoordinate env ⟨256 + slot.val * 41 + digit.val, by omega⟩ =
      LowNormSlot.coordinate .field (env (Retained.canonicalFields.source slot).val) digit := by
  unfold rangeCoordinate
  rw [dif_neg (by dsimp only; omega), dif_pos (by dsimp only; omega)]
  have quotient : (256 + slot.val * 41 + digit.val - 256) / 41 = slot.val := by omega
  have remainder : (256 + slot.val * 41 + digit.val - 256) % 41 = digit.val := by omega
  simp only [quotient, remainder]
  exact congrArg₂ (LowNormSlot.coordinate .field)
    (congrArg (fun slot : Fin 8 => env (Retained.canonicalFields.source slot).val) (Fin.eta _ _))
    (Fin.eta _ _)

theorem results_coordinate (env : Env) (slot : Fin 353) :
    rangeCoordinate env ⟨584 + slot.val, by omega⟩ =
      LowNormSlot.coordinate .bit (env (Retained.resultBits.source slot).val) ⟨0, by decide⟩ := by
  unfold rangeCoordinate
  rw [dif_neg (by dsimp only; omega), dif_neg (by dsimp only; omega)]
  simp only [Nat.add_sub_cancel_left, Fin.eta]

private theorem assignment_owned {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (initial : FieldState) (column : Fin columns) (index : Fin 135813)
    (position : column.val = interface.start + index.val) :
    assignment interface base initial column = coordinate initial index := by
  have same : column = ⟨interface.start + index.val, by
      have fits := interface.fits; rw [BatchPlan.coordinateCount_eq] at fits; omega⟩ := Fin.ext position
  rw [same]
  exact assignment_at interface base initial index

theorem poseidon_encodes {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (initial : FieldState) (fits : interface.start + poseidonBlock.coordinateCount ≤ columns) :
    poseidonBlock.EncodesAt interface.start fits (assignment interface base initial) (sboxValues initial) := by
  intro slot digit
  have slotBound : slot.val < 2924 := slot.isLt
  have digitBound : digit.val < 41 := digit.isLt
  rw [assignment_owned interface base initial _ ⟨slot.val * 41 + digit.val, by omega⟩ (by rfl)]
  exact poseidon_coordinate initial slot digit

theorem range_encodes {columns : Nat} (interface : Interface columns) (base : Assignment F columns)
    (initial : FieldState) (source : Fin 17) :
    Retained.Encodes (rangeStart interface source) (rangeFits interface source)
      (assignment interface base initial) (rangeSources initial source) := by
  constructor
  · intro slot digit
    have slotBound : slot.val < 256 := slot.isLt
    have zero : digit = ⟨0, by decide⟩ := Fin.ext (by change digit.val = 0; have h : digit.val < 1 := digit.isLt; omega)
    subst digit
    rw [assignment_owned interface base initial _ ⟨119884 + source.val * 937 + slot.val, by omega⟩ (by
      change interface.start + 119884 + source.val * 937 + (slot.val * 1 + 0) =
        interface.start + (119884 + source.val * 937 + slot.val); omega),
      range_coordinate initial source ⟨slot.val, by omega⟩, bits_coordinate]
    rfl
  · intro slot digit
    have slotBound : slot.val < 8 := slot.isLt
    have digitBound : digit.val < 41 := digit.isLt
    rw [assignment_owned interface base initial _
      ⟨119884 + source.val * 937 + (256 + slot.val * 41 + digit.val), by omega⟩ (by
      change interface.start + 119884 + source.val * 937 + 256 + (slot.val * 41 + digit.val) =
        interface.start + (119884 + source.val * 937 + (256 + slot.val * 41 + digit.val)); omega),
      range_coordinate initial source ⟨256 + slot.val * 41 + digit.val, by omega⟩, fields_coordinate]
    rfl
  · intro slot digit
    have slotBound : slot.val < 353 := slot.isLt
    have zero : digit = ⟨0, by decide⟩ := Fin.ext (by change digit.val = 0; have h : digit.val < 1 := digit.isLt; omega)
    subst digit
    rw [assignment_owned interface base initial _ ⟨119884 + source.val * 937 + (584 + slot.val), by omega⟩ (by
      change interface.start + 119884 + source.val * 937 + 584 + (slot.val * 1 + 0) =
        interface.start + (119884 + source.val * 937 + (584 + slot.val)); omega),
      range_coordinate initial source ⟨584 + slot.val, by omega⟩, results_coordinate]
    rfl

end NightstreamFPrime.Layout.PiRlcWideSampler.Encoding
