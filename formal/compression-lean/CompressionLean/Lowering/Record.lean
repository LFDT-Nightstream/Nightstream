import CompressionLean.Ports.Circuit.Record

/-!
Owns the facts about the recorder's rows (`Ports/Circuit/Record.lean`) that
the lowering proofs read: `Call.computedRows` is the model's `Call.rows`
with decidable zero tests (`computedRows_eq`), and an extension product
row has at most three products (`extRow_length`).
-/

namespace CompressionLean.Lowering

open CompressionLean.Shrink CompressionLean.Circuit

variable {F : Type} [Field F] [DecidableEq F]

theorem expandArray_eq (side : List (ℕ × F)) (columns : List (Form F)) :
    expandArray side columns.toArray = expand side columns := by
  unfold expandArray expand
  congr 1
  funext t
  have read : columns.toArray.getD t.1 ⟨[], 0⟩ = columns.getD t.1 ⟨[], 0⟩ := by
    simp [Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD]
  rw [read]
  congr 1
  split_ifs <;> rfl

theorem _root_.CompressionLean.Shrink.Call.computedRows_eq (c : Call F) : c.computedRows = c.rows := by
  cases c <;> simp [Call.computedRows, Call.rows, expandArray_eq]

theorem extRow_length (a b : Fin 3 → Form F) (k : Fin 3) : (extRow a b k).length ≤ 3 := by
  fin_cases k <;> simp [extRow]

end CompressionLean.Lowering
