import NightstreamFPrime.Layout.PiRlcWideSampler.Completeness

/-! The compact witness uses the unchanged strict norm bound b=2. Every
retained bit is constrained Boolean; field auxiliaries and S-box values use
the existing 41-coordinate balanced encoding. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.Norm

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling
open ProductionRelation BatchPlan Witness

private theorem slot_norm (kind : LowNormSlot.Kind) (value : F) (valid : LowNormSlot.Valid kind value)
    (index : Fin kind.width) : centeredMagnitude (LowNormSlot.coordinate kind value index) < 2 := by
  apply LowNormSlot.encode_norm kind value valid
  rw [← LowNormSlot.coordinateList_eq_encode]
  exact List.mem_ofFn.mpr ⟨index, rfl⟩

private theorem valid_slot (draw : Fin 4 → F) (slot : Slot) :
    LowNormSlot.Valid (kind slot) (rangeValues draw (1408 + PiRlcWideSampler.source slot)) := by
  have core := rangeValues_rows draw
  change ConstraintsHold (rangeValues draw) (flatConstraints (WideReduction.Program.operations RangePlan.interface 4)) at core
  rw [WideReduction.Program.constraints_eq] at core
  exact kind_valid (WideReduction.Program.coreInterface RangePlan.interface 4)
    (WideReduction.HintProgram.resultHints 4) (rangeValues draw) 1408 (fun lane => by have bound : lane.val < 4 := lane.isLt; change lane.val < 1408; omega)
    (holdsFlat_implies_holds _ _ core) slot

theorem canonical_bits_valid (draw : Fin 4 → F) (slot : Fin 256) :
    LowNormSlot.Valid .bit (rangeValues draw (Retained.canonicalBits.source slot).val) := by
  let location : Slot := .inl (⟨slot.val / 64, by omega⟩, ⟨slot.val % 64, by omega⟩)
  have valid := valid_slot draw location
  change LowNormSlot.Valid (if slot.val % 64 < 64 then .bit else .field)
    (rangeValues draw (1408 + (66 * (slot.val / 64) + slot.val % 64))) at valid
  rw [if_pos (Nat.mod_lt _ (by decide))] at valid
  change LowNormSlot.Valid .bit (rangeValues draw (1408 + 66 * (slot.val / 64) + slot.val % 64))
  simpa only [Nat.add_assoc] using valid

theorem result_bits_valid (draw : Fin 4 → F) (slot : Fin 353) :
    LowNormSlot.Valid .bit (rangeValues draw (Retained.resultBits.source slot).val) := by
  have valid := valid_slot draw (.inr slot)
  change LowNormSlot.Valid .bit (rangeValues draw (1408 + (264 + slot.val))) at valid
  rw [← Nat.add_assoc] at valid
  exact valid

private theorem range_norm (draw : Fin 4 → F) (position : Fin 937) :
    centeredMagnitude (rangeCoordinate (rangeValues draw) position) < 2 := by
  unfold rangeCoordinate
  split_ifs with bit field
  · exact slot_norm .bit _ (canonical_bits_valid draw _) _
  · exact slot_norm .field _ trivial _
  · exact slot_norm .bit _ (result_bits_valid draw _) _

theorem coordinate_norm (initial : FieldState) (index : Fin 135813) :
    centeredMagnitude (coordinate initial index) < 2 := by
  unfold coordinate
  split_ifs with poseidon
  · exact slot_norm .field _ trivial _
  · exact range_norm _ _

end NightstreamFPrime.Layout.PiRlcWideSampler.Norm
