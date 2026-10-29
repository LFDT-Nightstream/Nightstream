import Mathlib.Data.List.GetD
import Mathlib.Tactic.SplitIfs
import Mathlib.Order.MinMax

namespace CompressionLean.Lib

theorem getElem!_set! {α : Type} [Inhabited α] (xs : Array α) (j i : ℕ) (x : α) (h : i < xs.size) :
    (xs.set! j x)[i]! = if j = i then x else xs[i]! := by
  rw [Array.set!_eq_setIfInBounds, getElem!_pos _ _ (by simpa using h), Array.getElem_setIfInBounds h,
    getElem!_pos xs i h]

theorem getElem!_append {α : Type} [Inhabited α] (a b : Array α) (i : ℕ) :
    (a ++ b)[i]! = if i < a.size then a[i]! else b[i - a.size]! := by
  simp only [getElem!_def, Array.getElem?_append]
  split_ifs <;> rfl

theorem getElem!_extract_zero {α : Type} [Inhabited α] (a : Array α) {i k : ℕ} (hi : i < k) :
    (a.extract 0 k)[i]! = a[i]! := by
  simp only [getElem!_def, Array.getElem?_extract]
  split_ifs <;> simp_all

theorem getElem!_map_range {α : Type} [Inhabited α] (n : ℕ) (f : ℕ → α) {t : ℕ} (inside : t < n) :
    ((Array.range n).map f)[t]! = f t := by
  rw [getElem!_pos _ t (by simpa using inside)]
  simp

theorem getElem!_map_range₂ {α : Type} [Inhabited α] (n m : ℕ) (f : ℕ → ℕ → α) {i j : ℕ} (hi : i < n) (hj : j < m) :
    (((Array.range n).map fun i => (Array.range m).map (f i))[i]!)[j]! = f i j := by
  rw [getElem!_map_range n _ hi, getElem!_map_range m _ hj]

theorem array_getD_append {α : Type} (a : Array α) (l : List α) (n : ℕ) (d : α) :
    (a ++ l.toArray).getD n d = if n < a.size then a.getD n d else l.getD (n - a.size) d := by
  simp only [Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD]
  split_ifs with h
  · rw [Array.getElem?_append_left h]
  · rw [Array.getElem?_append_right (by omega)]
    simp

end CompressionLean.Lib
