import CompressionLean.Ports.Shrink.Ccs
import CompressionLean.Lib.Cube
import Mathlib.Algebra.Algebra.Defs
import Mathlib.Algebra.BigOperators.Fin

/-!
Owns the shrink relation over the CCS of `Ports/Shrink/Ccs.lean`
(`Ccs.Relation`: every row's constraint is zero), the entry sum
`matrixValue = Σ_j ρ^j·M̃_j(rx, ry)`, summed entry by entry as the verifier
sums it (`mod.rs:275`), and the extensions of the tables (`mle_weight`,
`mle_z`, `publicValue_eq`). The extension lemmas need every entry inside the
cubes (`Ccs.Fits`).
-/

namespace CompressionLean.Shrink

open CompressionLean.Lib

theorem map_constraint {R S : Type} [CommRing R] [CommRing S] (f : R →+* S) (v : Fin 8 → R) :
    f (constraint v) = constraint fun j => f (v j) := by
  simp [constraint, map_add, map_sub, map_mul, map_pow]

section Base

variable {F : Type} [Field F]

namespace Ccs

variable (C : Ccs F)

/-- The shrink relation: every row of the CCS holds on `z = (w, 1, x)`. -/
def Relation (w : ℕ → F) : Prop :=
  ∀ r, r < 2 ^ C.rowVars → constraint (fun j => rowValue C.entries (C.z w) j r) = 0

/-- Every entry sits inside the cubes, and the constant one and the
statement fit the upper half of `z`. -/
structure Fits : Prop where
  row_lt : ∀ e ∈ C.entries, e.row < 2 ^ C.rowVars
  col_lt : ∀ e ∈ C.entries, e.col < 2 ^ (C.cellVars + 1)
  statement_le : C.statement.length + 1 ≤ 2 ^ C.cellVars

end Ccs

end Base

section Extension

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

/-- `Σ_j ρ^j·M̃_j(rx, ry)`, entry by entry (`mod.rs:275`). -/
def matrixValue (es : List (Entry F)) (rx ry : List K) (ρ : K) : K :=
  (es.map fun e => ρ ^ (e.matrix : ℕ) * eqAt rx e.row * eqAt ry e.col * ι e.coef).sum

omit [Field F] [Algebra F K] in
/-- The extension of a table with one value, at `r`. -/
theorem mle_single (p : List K) (r : ℕ) (c : K) (inside : r < 2 ^ p.length) :
    mle p (fun i => if r = i then c else 0) = eqAt p r * c := by
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range]
  simp only [mul_ite, mul_zero]
  rw [Finset.sum_ite_eq]
  simp [inside]

/-- The extension of row values: each entry of matrix `j` gives
`eq(rx, row)·coefficient·z(column)`. -/
theorem mle_rowValue (es : List (Entry F)) (z : ℕ → F) (j : Fin 8) (rx : List K)
    (fits : ∀ e ∈ es, e.row < 2 ^ rx.length) :
    mle rx (fun r => ι (rowValue es z j r)) =
      (es.map fun e => if e.matrix = j then eqAt rx e.row * ι (e.coef * z e.col) else 0).sum := by
  induction es with
  | nil => simpa [rowValue] using mle_const rx (0 : K)
  | cons e es ih =>
      have rest := ih fun e' member => fits e' (List.mem_cons_of_mem _ member)
      simp only [rowValue, List.map_cons, List.sum_cons, map_add] at rest ⊢
      rw [mle_add, rest]
      congr 1
      by_cases same : e.matrix = j
      · simp only [same, true_and, if_true]
        rw [← mle_single rx e.row _ (fits e List.mem_cons_self)]
        congr 1
        funext r
        split_ifs <;> simp
      · simp only [same, false_and, if_false, map_zero]
        exact mle_const rx 0

/-- The inner sum-check's total over the column cube, entry by entry. -/
theorem cubeSum_weight (es : List (Entry F)) (rx : List K) (ρ : K) (z : ℕ → K) (n : ℕ)
    (fits : ∀ e ∈ es, e.col < 2 ^ n) :
    cubeSum n (fun y => weight es rx ρ y * z y) =
      (es.map fun e => ρ ^ (e.matrix : ℕ) * eqAt rx e.row * ι e.coef * z e.col).sum := by
  induction es with
  | nil => simp [weight, cubeSum_zero]
  | cons e es ih =>
      have rest := ih fun e' member => fits e' (List.mem_cons_of_mem _ member)
      simp only [weight, List.map_cons, List.sum_cons, add_mul] at rest ⊢
      rw [cubeSum_add, rest]
      congr 1
      rw [cubeSum_eq_sum_range]
      simp only [ite_mul, zero_mul]
      rw [Finset.sum_ite_eq]
      simp [fits e List.mem_cons_self]

omit [Field F] [Algebra F K] in
/-- `Σ_j ρ^j` against a per-entry value of matrix `j` gives `ρ^matrix` per entry. -/
theorem sum_powers_entries (es : List (Entry F)) (ρ : K) (A : Entry F → K) :
    ∑ j : Fin 8, ρ ^ (j : ℕ) * (es.map fun e => if e.matrix = j then A e else 0).sum =
      (es.map fun e => ρ ^ (e.matrix : ℕ) * A e).sum := by
  induction es with
  | nil => simp
  | cons e es ih =>
      simp only [List.map_cons, List.sum_cons, mul_add, Finset.sum_add_distrib, ih]
      congr 1
      simp only [mul_ite, mul_zero]
      rw [Finset.sum_ite_eq]
      simp

/-- The inner sum-check's total is the batch of the true matrix values. -/
theorem cubeSum_weight_rows (es : List (Entry F)) (z : ℕ → F) (rx : List K) (ρ : K) (n : ℕ)
    (rows : ∀ e ∈ es, e.row < 2 ^ rx.length) (cols : ∀ e ∈ es, e.col < 2 ^ n) :
    cubeSum n (fun y => weight es rx ρ y * ι (z y)) =
      ∑ j : Fin 8, ρ ^ (j : ℕ) * mle rx (fun r => ι (rowValue es z j r)) := by
  rw [cubeSum_weight es rx ρ (fun y => ι (z y)) n cols]
  simp only [mle_rowValue es z _ rx rows]
  rw [sum_powers_entries]
  congr 1
  apply List.map_congr_left
  intro e _
  rw [map_mul]
  ring

/-- The weights' extension at `ry` is the verifier's entry sum. -/
theorem mle_weight (es : List (Entry F)) (rx ry : List K) (ρ : K)
    (fits : ∀ e ∈ es, e.col < 2 ^ ry.length) :
    mle ry (weight es rx ρ) = matrixValue es rx ry ρ := by
  show mle ry (fun y => weight es rx ρ y) = _
  induction es with
  | nil => simpa [weight, matrixValue] using mle_const ry (0 : K)
  | cons e es ih =>
      have rest := ih fun e' member => fits e' (List.mem_cons_of_mem _ member)
      simp only [weight, matrixValue, List.map_cons, List.sum_cons] at rest ⊢
      rw [mle_add, rest]
      congr 1
      rw [mle_single ry e.col _ (fits e List.mem_cons_self)]
      ring

/-- The verifier's `x̃` is the extension of the upper half of `z`. -/
theorem publicValue_eq (C : Ccs F) (q : List K) (fits : C.statement.length + 1 ≤ 2 ^ q.length) :
    C.publicValue q = mle q (fun i => ι (C.publicTable i)) := by
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range]
  apply Finset.sum_subset (Finset.range_mono fits)
  intro i _ outside
  simp only [Finset.mem_range, not_lt] at outside
  have zero : C.publicTable i = 0 := by
    unfold Ccs.publicTable
    rw [if_neg (by omega), List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega)]
    rfl
  rw [zero, map_zero, mul_zero]

/-- `z̃(ry) = (1 - ry_n)·w̃(ry[..n]) + ry_n·x̃(ry[..n])` (`mod.rs:283`). -/
theorem mle_z (C : Ccs F) (w : ℕ → F) (ry : List K) (length : ry.length = C.cellVars + 1)
    (fits : C.statement.length + 1 ≤ 2 ^ C.cellVars) :
    mle ry (fun y => ι (C.z w y)) =
      (1 - ry.getD C.cellVars 0) * mle (ry.take C.cellVars) (fun i => ι (w i)) +
        ry.getD C.cellVars 0 * C.publicValue (ry.take C.cellVars) := by
  obtain ⟨q, t, rfl, qLength⟩ : ∃ q t, ry = q ++ [t] ∧ q.length = C.cellVars := by
    refine ⟨ry.take C.cellVars, ry.getD C.cellVars 0, ?_, by simp [length]⟩
    have drop : ry.drop C.cellVars = [ry.getD C.cellVars 0] := by
      apply List.ext_getElem
      · simp [length]
      · intro i h _
        simp only [List.length_drop, length] at h
        simp [List.getD_eq_getElem?_getD, show i = 0 by omega, length]
    conv_lhs => rw [← List.take_append_drop C.cellVars ry, drop]
  have take : (q ++ [t]).take C.cellVars = q := List.take_left' qLength
  have last : (q ++ [t]).getD C.cellVars 0 = t := by
    simp [List.getD_eq_getElem?_getD, ← qLength]
  rw [take, last, mle_append_top, qLength, publicValue_eq C q (by rw [qLength]; exact fits)]
  congr 2
  · apply mle_congr
    intro i lt
    rw [qLength] at lt
    simp [Ccs.z, lt]
  · congr 1
    funext i
    simp [Ccs.z]

end Extension

end CompressionLean.Shrink
