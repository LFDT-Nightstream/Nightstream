import CompressionLean.Layer1.HonestFold
import Mathlib.Data.Fintype.Pi
import Mathlib.Basic.Real.Basic

namespace CompressionLean.Layer1

open CompressionLean.Lib
open Polynomial

variable {K : Type} [Field K]

theorem agreement_card [DecidableEq K] (m : ℕ) (g h : ℕ → K) (different : ∃ i, i < 2 ^ m ∧ g i ≠ h i)
    (q : ℕ → K) (leaves : ℕ) (distinct : ∀ i j, i < leaves → j < leaves → q i = q j → i = j) :
    ((Finset.range leaves).filter fun j => (univ m g).eval (q j) = (univ m h).eval (q j)).card ≤
      2 ^ m - 1 := by
  classical
  have nonzero : univ m (fun i => g i - h i) ≠ 0 := by
    intro zero
    obtain ⟨i, inside, unequal⟩ := different
    exact unequal (sub_eq_zero.mp (univ_eq_zero m _ zero i inside))
  have degree : (univ m (fun i => g i - h i)).natDegree ≤ 2 ^ m - 1 := by
    have := univ_natDegree m (fun i => g i - h i)
    omega
  set S := (Finset.range leaves).filter fun j => (univ m g).eval (q j) = (univ m h).eval (q j)
  have injective : Set.InjOn q S := by
    intro i hi j hj equal
    simp only [S, Finset.coe_filter, Finset.mem_range, Set.mem_ofPred_eq] at hi hj
    exact distinct i j hi.1 hj.1 equal
  rw [← Finset.card_image_of_injOn injective]
  refine (Polynomial.card_le_degree_of_subset_roots ?_).trans degree
  intro x member
  obtain ⟨j, inside, rfl⟩ := Finset.mem_image.mp (Finset.mem_val.mp member)
  simp only [S, Finset.mem_filter] at inside
  rw [Polynomial.mem_roots nonzero, IsRoot, univ_sub, eval_sub, inside.2, sub_self]

open Classical in
noncomputable def queryBad {L : ℕ} (t : ℕ) (agree : Fin L → Prop) : Finset (Fin t → Fin L) :=
  Fintype.piFinset fun _ => Finset.univ.filter agree

def lowBit {P : ℕ} (L : ℕ) [NeZero L] (w : Fin P) : Fin L :=
  ⟨(w : ℕ) % L, Nat.mod_lt _ (Nat.pos_of_neZero L)⟩

def lowBits {t P : ℕ} (L : ℕ) [NeZero L] (x : Fin t → Fin P) : Fin t → Fin L := fun j => lowBit L (x j)

theorem residue_card (P L j : ℕ) :
    ((Finset.univ : Finset (Fin P)).filter fun w : Fin P => (w : ℕ) % L = j).card ≤ P / L + 1 := by
  calc _ ≤ (Finset.range (P / L + 1)).card := by
        apply Finset.card_le_card_of_injOn (fun w : Fin P => (w : ℕ) / L)
        · intro w _
          simp only [Finset.coe_range, Set.mem_Iio]
          exact Nat.lt_succ_of_le (Nat.div_le_div_right w.isLt.le)
        · intro w hw w' hw' same
          simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hw hw'
          apply Fin.ext
          rw [← Nat.div_add_mod (w : ℕ) L, ← Nat.div_add_mod (w' : ℕ) L, hw, hw']
          simp only at same
          rw [same]
    _ = P / L + 1 := Finset.card_range _

theorem lowBits_fiber_card {L : ℕ} [NeZero L] (P A : ℕ) (agree : Fin L → Prop) [DecidablePred agree]
    (small : (Finset.univ.filter agree).card ≤ A) :
    ((Finset.univ : Finset (Fin P)).filter fun w => agree (lowBit L w)).card ≤ A * (P / L + 1) := by
  classical
  set residues := fun j : Fin L => (Finset.univ : Finset (Fin P)).filter fun w : Fin P => (w : ℕ) % L = j
  have cover : ((Finset.univ : Finset (Fin P)).filter fun w => agree (lowBit L w)) ⊆
      (Finset.univ.filter agree).biUnion residues := by
    intro w member
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at member
    exact Finset.mem_biUnion.mpr ⟨_, by simpa using member, by simp [residues, lowBit]⟩
  calc _ ≤ _ := Finset.card_le_card cover
    _ ≤ ∑ j ∈ Finset.univ.filter agree, (residues j).card := Finset.card_biUnion_le
    _ ≤ ∑ _j ∈ Finset.univ.filter agree, (P / L + 1) := Finset.sum_le_sum fun j _ => residue_card P L j
    _ ≤ A * (P / L + 1) := by
        rw [Finset.sum_const, smul_eq_mul]
        exact Nat.mul_le_mul_right _ small

open Classical in
theorem lowBitsBad_density {L : ℕ} [NeZero L] (t P A : ℕ) (agree : Fin L → Prop)
    (small : (Finset.univ.filter agree).card ≤ A) :
    (((Finset.univ.filter fun x : Fin t → Fin P => lowBits L x ∈ queryBad t agree).card : ℝ) /
      Fintype.card (Fin t → Fin P)) ≤ (((A * (P / L + 1) : ℕ) : ℝ) / P) ^ t := by
  classical
  have pi : (Finset.univ.filter fun x : Fin t → Fin P => lowBits L x ∈ queryBad t agree) =
      Fintype.piFinset fun _ => (Finset.univ : Finset (Fin P)).filter fun w => agree (lowBit L w) := by
    ext x
    simp [queryBad, lowBits, Fintype.mem_piFinset]
  rw [pi, Fintype.card_piFinset, Finset.prod_const, Finset.card_univ, Fintype.card_fin, Fintype.card_fun,
    Fintype.card_fin, Fintype.card_fin, div_pow]
  push_cast
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact pow_le_pow_left₀ (Nat.cast_nonneg _) (by exact_mod_cast lowBits_fiber_card P A agree small) t

end CompressionLean.Layer1
