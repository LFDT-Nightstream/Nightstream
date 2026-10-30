import CompressionLean.Ports.Circuit.Record

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

theorem lower_counts (L : Array Instr) (stmt stmt' : List F) : (lower L stmt).counts = (lower L stmt').counts := by
  simp [lower, Run.counts, Run.rows]

end CompressionLean.Lowering
