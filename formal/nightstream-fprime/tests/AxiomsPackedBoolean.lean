import tests.AxiomAudit
import NightstreamFPrime.Export.Generic.PackedBoolean
import NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness
import NightstreamFPrime.Export.BoundedDepth.PackedBudget

#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.target_mono
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.packNat_bound
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.target_capacity_lt_modulus
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.packNat_below_modulus
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.packNat_eq_target
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.unit_cases
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.unitDigit_le
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.unitDigit_field
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.digit_le
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.digit_field
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.pack_field
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.digit_eq_three
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.packed_implies_pairs
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.pairs_imply_packed
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.packed_iff_pairs
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.honest_pairs_norm
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.honest_pairs_packed
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.linear_eval
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.row_holds
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.source_slot_norm
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.source_slot_norm_wide
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.forced_holds
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.restore_rows
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.restore_logical_rows
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.restore_logical_rows_wide
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.complete_rows
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.extendSource_before
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.extendSource_companion
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.companion_value
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.extended_rows
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.sourceForced_valid
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.complement_valid
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.extended_slots_valid
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.honest_extended_layout
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.rowsHold_move_middle
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.extended_ordered_rows
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.honest_extended_ordered_layout
#audit_axioms NightstreamFPrime.Export.Generic.PackedBoolean.freshHolds_implies_original_rows
#audit_axioms NightstreamFPrime.Export.Generic.WideLayout.widenForm_eval_restrict
#audit_axioms NightstreamFPrime.Export.Generic.WideLayout.sourceEnv_widen_restrict
#audit_axioms NightstreamFPrime.Layout.ProductionRelation.Plan.freshHolds_implies_rowsPublicAndNorm
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness.completeEnv_output
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness.completeEnv_refill
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness.originalRows_eq
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness.reads_before
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness.native_rows_hold
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness.native_derived_valid
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness.original_rows_hold
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness.extended_slots_valid
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness.native_complete
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.choose_lengths
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.choose_fits
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.choose_end
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.choose_prefix
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.choose_of_fits
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.pairs_columns
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.chosen_inside
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.row_vars_below
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.groups_inside
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.rowsMax_lt
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.keep_sublist
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.mem_keep
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.keep_nil
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.covers_keep
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.prefix_products
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.restore_chosen_rows
#audit_axioms NightstreamFPrime.Export.BoundedDepth.PackedBudget.native_premises

namespace NightstreamFPrime.Tests.PackedBoolean

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Export.Generic.PackedBoolean

private def wrappingPairs : List (F × F) :=
  List.ofFn fun index : Fin 32 =>
    if index.val = 0 then (0, 0)
    else if 16 ≤ index.val then (-1, -1)
    else (0, 1)

example : pack wrappingPairs = 0 := by decide

example : Norm wrappingPairs := by
  intro pair member
  obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
  split
  · decide
  · split <;> decide

example : ¬∀ pair ∈ wrappingPairs,
    LowNormSlot.Valid .bit pair.1 ∧ pair.2 = 1 - pair.1 := by
  intro valid
  have bad := (valid (0, 0) (List.mem_ofFn.mpr ⟨⟨0, by decide⟩, rfl⟩)).2
  have different : (0 : F) ≠ 1 := by decide
  exact different (by simpa using bad)

example : pack [(2, -1)] = 0 := by decide

example : ¬LowNormSlot.Valid .bit (2 : F) := by
  change ¬((2 : F) = 0 ∨ (2 : F) = 1)
  decide

namespace NativeOutputCopy

open NightstreamFPrime.Circuit NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness

private def pieces : List Pieces.Piece :=
  [.permutation 1000 (List.ofFn fun lane : Fin 16 => .var (600 + lane.val))]

private def outputRow : R1CS.Row := BooleanInference.booleanRow 2080

private def copyRow : R1CS.Row := ⟨.ofVar 2080, .one, .ofVar 2100⟩

private def nativeRows : List R1CS.Row :=
  [outputRow, copyRow, BooleanInference.booleanRow 500]

private def copyGate : BooleanInference.Gate where
  left := 2080
  right := 0
  output := 2100
  a := ⟨0, 1, 0⟩
  b := ⟨1, 0, 0⟩
  rest := ⟨0, 0, 0⟩
  coefficient := 1

private theorem copied_derived : BooleanInference.Derived nativeRows 2100 := by
  refine .gate (row := copyRow) copyGate (by simp [nativeRows]) ?_ (by decide) ?_ .one
  · intro env _
    simp [copyRow, copyGate, R1CS.Row.Holds, BooleanInference.Gate.Holds,
      BooleanInference.Form.eval]
  · exact .forced (row := outputRow) (by simp [nativeRows]) (by decide)

example : PackedBoolean.SourceForced nativeRows 500 := by
  exact ⟨BooleanInference.booleanRow 500, by simp [nativeRows], by decide⟩

example : CircuitRelation.outputIn (CircuitRelation.blocksOf pieces) 2080 = true := by decide

example : CircuitRelation.outputIn (CircuitRelation.blocksOf pieces) 2100 = false := by decide

example : NativeSlotKind nativeRows pieces 2101 [500]
    ({ source := ⟨2100, by decide⟩, kind := .bit } : LowNormAssignment.Slot 4096) := by
  exact Or.inr ⟨rfl, Or.inl ⟨by decide, Or.inl copied_derived⟩⟩

private def outputForm : R1CS.LinearCombination :=
  CircuitRelation.toCombination
    (CircuitRelation.outputStateAt (width := 4096) 2101 0 ⟨0, by decide⟩)

example : SlotLayout.forcedColumn?
    (Substitute.row (fun column => if column = 2080 then some outputForm else none) outputRow) = none := by
  decide

example (env : Env) (holds : R1CS.RowsHold env nativeRows) (one : env 0 = 1) :
    LowNormSlot.Valid .bit (RelationCompleteness.completeEnv pieces 2101 env 2100) := by
  apply native_derived_valid nativeRows (by decide) (by decide) _ _ env holds one copied_derived
  · intro r member
    simp only [nativeRows, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl <;>
      simp [outputRow, copyRow, BooleanInference.booleanRow, R1CS.Row.VarsBelow,
        R1CS.LinearCombination.VarsBelow, R1CS.LinearCombination.ofVar, R1CS.LinearCombination.one]
  · intro r member
    simp only [nativeRows, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl <;> decide

end NativeOutputCopy

namespace Budget

open NightstreamFPrime.Export.BoundedDepth.PackedBudget

/-- The rows of `packing_stops_at_the_package_width`: ten sources, two zero products and one
product, for each room. -/
private def counts (room : Nat) : Nat × Nat × Nat :=
  let choice := choose room (List.range 10) (List.replicate 2 ()) [()]
  (choice.sources.length, choice.zeros.length, choice.products.length)

example : counts 14 = (10, 2, 1) := by decide
example : counts 13 = (10, 2, 0) := by decide
example : counts 11 = (10, 1, 0) := by decide
example : counts 10 = (10, 0, 0) := by decide
example : counts 4 = (4, 0, 0) := by decide
example : counts 0 = (0, 0, 0) := by decide

end Budget

end NightstreamFPrime.Tests.PackedBoolean
