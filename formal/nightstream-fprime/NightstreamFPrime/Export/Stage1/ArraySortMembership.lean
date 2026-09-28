module

import Mathlib.Tactic
import all Init.Data.Array.QSort.Basic
public import Init.Data.Array.QSort.Basic
public import Init.Data.Vector.Perm

/-!
The existing array quicksort only returns members of its input. This supports
execution-event provenance without changing the sort or proving its ordering.
-/

public section

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.ArraySortMembership

universe u
variable {α : Type u} {n : Nat}

private theorem optionalSwap_perm (values : Vector α n) (i j : Nat)
    (hi : i < n) (hj : j < n) (choose : Bool) :
    Vector.Perm (if choose then values.swap i j hi hj else values) values := by
  split
  · exact Vector.swap_perm hi hj
  · exact .rfl

private theorem partitionLoop_perm (lt : α → α → Bool) (lo hi : Nat)
    (hhi : hi < n) (pivot : α) (values : Vector α n) (i k : Nat)
    (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi) :
    Vector.Perm (Array.qpartition.loop lt lo hi hhi pivot values i k ilo ik w).2 values := by
  induction values, i, k, ilo, ik, w using Array.qpartition.loop.induct lt lo hi hhi pivot with
  | case1 values i k ilo ik w step less ih =>
      rw [Array.qpartition.loop, dif_pos step, if_pos less]
      exact ih.trans (Vector.swap_perm (by omega) (by omega))
  | case2 values i k ilo ik w step less ih =>
      rw [Array.qpartition.loop, dif_pos step, if_neg less]
      exact ih
  | case3 values i k ilo ik w step =>
      rw [Array.qpartition.loop, dif_neg step]
      exact Vector.swap_perm (by omega) hhi

private theorem partition_perm (lt : α → α → Bool) (values : Vector α n)
    (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n) :
    Vector.Perm (Array.qpartition values lt lo hi w hlo hhi).2 values := by
  dsimp only [Array.qpartition]
  refine (partitionLoop_perm lt lo hi hhi _ _ lo lo _ _ _).trans ?_
  refine (optionalSwap_perm _ _ _ _ _ _).trans ?_
  refine (optionalSwap_perm _ _ _ _ _ _).trans ?_
  exact optionalSwap_perm _ _ _ _ _ _

private theorem sort_perm (lt : α → α → Bool) (values : Vector α n)
    (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n) :
    Vector.Perm (Array.qsort.sort lt values lo hi w hlo hhi) values := by
  induction values, lo, hi, w, hlo, hhi using Array.qsort.sort.induct lt with
  | case1 values lo hi w hlo hhi step mid midBound partitioned partitionEquation last =>
      rw [Array.qsort.sort, dif_pos step, partitionEquation]
      dsimp only
      rw [dif_pos last]
      simpa only [partitionEquation] using partition_perm lt values lo hi w hlo hhi
  | case2 values lo hi w hlo hhi step mid midBound partitioned partitionEquation last ih₁ _ih₂ ih₃ =>
      rw [Array.qsort.sort, dif_pos step, partitionEquation]
      dsimp only
      rw [dif_neg last]
      exact ih₃.trans (ih₁.trans (by
        simpa only [partitionEquation] using partition_perm lt values lo hi w hlo hhi))
  | case3 values lo hi w hlo hhi step =>
      rw [Array.qsort.sort, dif_neg step]

/-- Sorting cannot introduce an event that was absent from the input array. -/
theorem mem_qsort (values : Array α) (lt : α → α → Bool) {value : α} :
    value ∈ values.qsort lt → value ∈ values := by
  unfold Array.qsort
  split
  · exact fun present => present
  · intro present
    exact ((sort_perm lt values.toVector _ _ _ _ _).toArray.mem_iff).mp present

end NightstreamFPrime.Export.Stage1.ArraySortMembership
