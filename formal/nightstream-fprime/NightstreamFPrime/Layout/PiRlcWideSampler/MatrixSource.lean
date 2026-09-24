import NightstreamFPrime.Layout.PiRlcWideSampler.Retained
import NightstreamFPrime.Layout.MatrixProgram

/-! Compact matrix operands for one wide reduction. Checked fields and bits
have retained views; the helper interval has no source mapping. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.MatrixSource

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open ProductionRelation MatrixProgram

def inputGrid (state : RetainedBlock) (slot : Nat) : SourceGrid where
  sourceStart := 0
  majorCount := 1
  majorSourceStride := 4
  minorCount := 1
  minorSourceStride := 4
  runCount := 4
  retained := state
  mode := .external8
  slotStart := slot
  majorSlotStride := 0
  minorSlotStride := 0

def bitGrid (start : Nat) : SourceGrid :=
  SourceGrid.ofSemantic Retained.canonicalBits start 1408 4 66 1 66 64 0 64 0

def fieldGrid (start : Nat) : SourceGrid :=
  SourceGrid.ofSemantic Retained.canonicalFields (start + 256) 1472 4 66 1 66 2 0 2 0

def resultRange (start : Nat) : SourceRange :=
  SourceRange.ofSemantic Retained.resultBits (start + 584) 1672 353 0

def substitution (state : RetainedBlock) (slot start : Nat) : SourceSubstitution where
  ranges := [resultRange start]
  grids := [inputGrid state slot, bitGrid start, fieldGrid start]

private theorem input_absent (state : RetainedBlock) (slot columns column : Nat)
    (after : 4 ≤ column) : (inputGrid state slot).form? columns column = none := by
  simp only [inputGrid, SourceGrid.form?, Nat.zero_le, Nat.sub_zero, Nat.ofNat_pos, if_pos]
  rw [if_neg (show ¬column / 4 < 1 by omega)]

private theorem bits_absent (start columns column : Nat)
    (absent : ¬(1408 ≤ column ∧ column < 1672 ∧ (column - 1408) % 66 < 64)) :
    (bitGrid start).form? columns column = none := by
  simp only [bitGrid, SourceGrid.ofSemantic, SourceGrid.form?]
  split_ifs <;> first | (exfalso; apply absent; omega) | rfl

private theorem fields_absent (start columns column : Nat)
    (absent : ¬(1408 ≤ column ∧ column < 1672 ∧ 64 ≤ (column - 1408) % 66)) :
    (fieldGrid start).form? columns column = none := by
  simp only [fieldGrid, SourceGrid.ofSemantic, SourceGrid.form?]
  split_ifs <;> first | (exfalso; apply absent; omega) | rfl

private theorem result_absent (start columns column : Nat)
    (absent : ¬(1672 ≤ column ∧ column < 2025)) :
    (resultRange start).form? columns column = none := by
  simp only [resultRange, SourceRange.ofSemantic, SourceRange.form?]
  split_ifs <;> first | (exfalso; apply absent; omega) | rfl

theorem temporary_unmapped (state : RetainedBlock) (slot start columns column : Nat)
    (lower : 4 ≤ column) (upper : column < 1408) :
    (substitution state slot start).form? columns column = none := by
  have a := input_absent state slot columns column lower
  have b := bits_absent start columns column (by omega)
  have c := fields_absent start columns column (by omega)
  have d := result_absent start columns column (by omega)
  simp only [substitution, SourceSubstitution.form?, List.filterMap_cons, List.filterMap_nil,
    a, b, c, d, List.nil_append]

private theorem bits_loaded {columns start : Nat} (fits : Retained.Fits columns start)
    (child : Fin 4) (bit : Fin 64) :
    (bitGrid start).form? columns (1408 + child.val * 66 + bit.val) =
      some (Retained.canonicalBits.form start (by change start + 256 ≤ columns; exact le_trans (by omega) fits)
        ⟨64 * child.val + bit.val, by change _ < 256; omega⟩) := by
  have bounded : 0 + child.val * 64 + (0 : Fin 1).val * 0 + bit.val < Retained.canonicalBits.slotCount := by
    change 0 + child.val * 64 + 0 * 0 + bit.val < 256
    omega
  simpa only [Fin.val_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.mul_comm] using
    SourceGrid.form?_ofSemantic Retained.canonicalBits start 1408 4 66 1 66 64 0 64 0
      (by change start + 256 ≤ columns; exact le_trans (by omega) fits)
      (by decide) (by decide) child (0 : Fin 1) bit (by have h := bit.isLt; change 0 * 66 + bit.val < 66; omega)
      (by have h := bit.isLt; omega) bounded

private theorem fields_loaded {columns start : Nat} (fits : Retained.Fits columns start)
    (child : Fin 4) (cell : Fin 2) :
    (fieldGrid start).form? columns (1472 + child.val * 66 + cell.val) =
      some (Retained.canonicalFields.form (start + 256)
        (by change (start + 256) + 328 ≤ columns; exact le_trans (by omega) fits)
        ⟨2 * child.val + cell.val, by change _ < 8; omega⟩) := by
  have bounded : 0 + child.val * 2 + (0 : Fin 1).val * 0 + cell.val < Retained.canonicalFields.slotCount := by
    change 0 + child.val * 2 + 0 * 0 + cell.val < 8
    omega
  simpa only [Fin.val_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.mul_comm] using
    SourceGrid.form?_ofSemantic Retained.canonicalFields (start + 256) 1472 4 66 1 66 2 0 2 0
      (by change (start + 256) + 328 ≤ columns; exact le_trans (by omega) fits)
      (by decide) (by decide) child (0 : Fin 1) cell (by have h := cell.isLt; change 0 * 66 + cell.val < 66; omega)
      (by have h := cell.isLt; omega) bounded

private theorem results_loaded {columns start : Nat} (fits : Retained.Fits columns start)
    (index : Fin 353) :
    (resultRange start).form? columns (1672 + index.val) =
      some (Retained.resultBits.form (start + 584)
        (by change (start + 584) + 353 ≤ columns; exact fits) index) := by
  simpa only [Nat.zero_add, Fin.eta] using SourceRange.form?_ofSemantic Retained.resultBits (start + 584) 1672 353 0
    (by change (start + 584) + 353 ≤ columns; exact fits) (by decide) index

/-- Each retained source resolves to the same form as the certified range
compiler. The caller supplies the four actual Poseidon output forms. -/
theorem form_eq {columns start : Nat} (fits : Retained.Fits columns start)
    (state : RetainedBlock) (slot : Nat) (input : Fin 4 → SparseForm columns)
    (inputs : ∀ lane : Fin 4, (inputGrid state slot).form? columns lane.val = some (input lane))
    (column : Fin 2025) (active : column.val < 4 ∨ 1408 ≤ column.val) :
    (substitution state slot start).form? columns column.val =
      some ((Retained.sourceMap start fits input).form column) := by
  by_cases caller : column.val < 4
  · have b := bits_absent start columns column.val (by omega)
    have c := fields_absent start columns column.val (by omega)
    have d := result_absent start columns column.val (by omega)
    simp only [substitution, SourceSubstitution.form?, List.filterMap_cons, List.filterMap_nil,
      inputs ⟨column.val, caller⟩, b, c, d, List.nil_append,
      Retained.sourceMap, dif_pos caller]
  · have retained : 1408 ≤ column.val := active.resolve_left caller
    have a := input_absent state slot columns column.val (by omega)
    by_cases canonical : column.val < 1672
    · have d := result_absent start columns column.val (by omega)
      let child : Fin 4 := ⟨(column.val - 1408) / 66, by omega⟩
      by_cases bit : (column.val - 1408) % 66 < 64
      · let digit : Fin 64 := ⟨(column.val - 1408) % 66, bit⟩
        have source : 1408 + child.val * 66 + digit.val = column.val := by dsimp only [child, digit]; omega
        have b := bits_loaded fits child digit
        rw [source] at b
        have c := fields_absent start columns column.val (by omega)
        simp only [substitution, SourceSubstitution.form?, List.filterMap_cons, List.filterMap_nil,
          a, b, c, d, List.nil_append, Retained.sourceMap, dif_neg caller,
          dif_neg (show ¬column.val < 1408 by omega), dif_pos canonical, dif_pos bit, child, digit]
      · let cell : Fin 2 := ⟨(column.val - 1408) % 66 - 64, by omega⟩
        have source : 1472 + child.val * 66 + cell.val = column.val := by dsimp only [child, cell]; omega
        have c := fields_loaded fits child cell
        rw [source] at c
        have b := bits_absent start columns column.val (by omega)
        simp only [substitution, SourceSubstitution.form?, List.filterMap_cons, List.filterMap_nil,
          a, b, c, d, List.nil_append, Retained.sourceMap, dif_neg caller,
          dif_neg (show ¬column.val < 1408 by omega), dif_pos canonical, dif_neg bit]
        apply congrArg some
        apply congrArg (Retained.canonicalFields.form (start + 256) _)
        apply Fin.ext
        dsimp only [child, cell]
        omega
    · let index : Fin 353 := ⟨column.val - 1672, by omega⟩
      have source : 1672 + index.val = column.val := by dsimp only [index]; omega
      have d := results_loaded fits index
      rw [source] at d
      have b := bits_absent start columns column.val (by omega)
      have c := fields_absent start columns column.val (by omega)
      simp only [substitution, SourceSubstitution.form?, List.filterMap_cons, List.filterMap_nil,
        a, b, c, d, List.append_nil, Retained.sourceMap, dif_neg caller,
        dif_neg (show ¬column.val < 1408 by omega), dif_neg canonical, index]
      apply congrArg some
      apply congrArg (Retained.resultBits.form (start + 584) _)
      apply Fin.ext
      rfl

end NightstreamFPrime.Layout.PiRlcWideSampler.MatrixSource
