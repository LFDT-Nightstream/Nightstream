import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Data.List.GetD
import Mathlib.Tactic.Ring

/-!
Owns generic facts about lists: entries read with `getD` after `++`, `map`
over `List.range`, `flatten` and `ofFn`, a fold that adds, and the lists of
at most `L` entries (a WHIR commitment's candidates) as a nonempty type.
-/

namespace CompressionLean.Lib

theorem list_getD_append {α : Type} (a l : List α) (n : ℕ) (d : α) :
    (a ++ l).getD n d = if n < a.length then a.getD n d else l.getD (n - a.length) d := by
  split_ifs with h
  · exact List.getD_append _ _ _ _ h
  · exact List.getD_append_right _ _ _ _ (by omega)

theorem getD_append_length {α : Type} (l l' : List α) (d : α) (k : ℕ) :
    (l ++ l').getD (l.length + k) d = l'.getD k d := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_append_right]

theorem getD_range_map {α : Type} (f : ℕ → α) (d : α) {n k : ℕ} (h : k < n) :
    ((List.range n).map f).getD k d = f k := by
  simp [List.getD_eq_getElem?_getD, h]

theorem getD_flatten {α : Type} (n : ℕ) (d : α) :
    ∀ (ls : List (List α)), (∀ x ∈ ls, x.length = n) → ∀ j i, i < n →
      ls.flatten.getD (n * j + i) d = if h : j < ls.length then ls[j].getD i d else d
  | [], _, j, i, _ => by simp
  | x :: rest, len, j, i, hi => by
      have head : x.length = n := len x (by simp)
      rw [List.flatten_cons]
      cases j with
      | zero =>
          rw [Nat.mul_zero, Nat.zero_add, List.getD_append _ _ _ _ (by rw [head]; exact hi)]
          simp
      | succ j =>
          rw [List.getD_append_right _ _ _ _ (by rw [head, Nat.mul_succ]; omega),
            show n * (j + 1) + i - x.length = n * j + i by rw [head, Nat.mul_succ]; omega,
            getD_flatten n d rest (fun y member => len y (by simp [member])) j i hi]
          simp

theorem length_flatten_uniform {α : Type} (n : ℕ) :
    ∀ (ls : List (List α)), (∀ x ∈ ls, x.length = n) → ls.flatten.length = n * ls.length
  | [], _ => by simp
  | x :: rest, len => by
      rw [List.flatten_cons, List.length_append, len x (by simp),
        length_flatten_uniform n rest (fun y member => len y (by simp [member])), List.length_cons, Nat.mul_succ]
      ring

theorem getD_ofFn {α : Type} {n : ℕ} (f : Fin n → α) (d : α) (k : ℕ) :
    (List.ofFn f).getD k d = if h : k < n then f ⟨k, h⟩ else d := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
  split <;> rfl

theorem getD_mem_cons (l : List ℕ) (k : ℕ) : l.getD k 0 ∈ 0 :: l := by
  rcases Nat.lt_or_ge k l.length with inside | past
  · exact List.mem_cons_of_mem _ (List.getD_eq_getElem _ _ inside ▸ List.getElem_mem inside)
  · rw [List.getD_eq_default _ _ past]
    exact List.mem_cons_self

theorem foldl_add_eq_sum {α M : Type} [AddCommMonoid M] (g : α → M) (l : List α) (init : M) :
    l.foldl (fun acc x => acc + g x) init = init + (l.map g).sum := by
  induction l generalizing init with
  | nil => simp
  | cons x rest ih => rw [List.foldl_cons, ih, List.map_cons, List.sum_cons, add_assoc]

/-- A commitment is a list of at most `L` candidates (WHIR's list decoding). -/
instance candidates_nonempty {α : Type} (L : ℕ) : Nonempty {l : List α // l.length ≤ L} :=
  ⟨⟨[], by simp⟩⟩

end CompressionLean.Lib
