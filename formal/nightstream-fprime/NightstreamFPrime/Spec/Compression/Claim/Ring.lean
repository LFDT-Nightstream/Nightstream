import NightstreamFPrime.Spec.Compression.Production
import NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Commitment

/-! The Ajtai rows of `Spec.CE.Holds` as ring equations of the compression argument (SPEC.md,
Algorithm 9, step 7). `ringFMul` is the product `mul` of `F[X]/(X^64 + 1)` (`ringFMul_apply`), and
row `i` of the Ajtai commitment of an assignment is `Σ_j mul(A_ij, z_j)` over its blocks `j`, where
`z_j` is cells `64j` to `64j + 63` (`ajtaiRow_apply`). -/

namespace NightstreamFPrime.Spec.Compression.Claim

open Finset NightstreamFPrime.Spec NightstreamFPrime.Spec.Compression.Algebra
open NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.RingRelation NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Cell `x` of a vector of `n` cells, and 0 past its end. -/
def cell {R : Type} [Zero R] {n : ℕ} (z : Fin n → R) (x : ℕ) : R :=
  if h : x < n then z ⟨x, h⟩ else 0

section Product

variable {R : Type} [CommRing R]

theorem mul_congr_left {D l : ℕ} (lane : l < D) {a a' : ℕ → R} (same : ∀ i < D, a i = a' i)
    (b : ℕ → R) : mul D a b l = mul D a' b l := by
  refine sum_congr rfl fun h member => ?_
  have := mem_range.mp member
  simp only [mulX]
  split_ifs <;> rw [same _ (by omega)]

theorem mul_congr_right (D l : ℕ) (a : ℕ → R) {b b' : ℕ → R} (same : ∀ h < D, b h = b' h) :
    mul D a b l = mul D a b' l :=
  sum_congr rfl fun h member => by rw [same h (mem_range.mp member)]

/-- A ring map commutes with the product of `R[X]/(X^D + 1)`. -/
theorem map_mul {S : Type} [CommRing S] (f : R →+* S) (D : ℕ) (a b : ℕ → R) (l : ℕ) :
    f (mul D a b l) = mul D (fun h => f (a h)) (fun h => f (b h)) l := by
  simp only [mul, mulX, map_sum, _root_.map_mul]
  refine sum_congr rfl fun h _ => ?_
  split_ifs <;> simp

end Product

theorem foldl_ite_range {R : Type} [AddCommMonoid R] (p : ℕ → Prop) [DecidablePred p]
    (f : ℕ → R) : ∀ (n : ℕ) (init : R),
    (List.range n).foldl (fun acc i => if p i then acc + f i else acc) init =
      init + ∑ i ∈ range n, if p i then f i else 0
  | 0, init => by simp
  | n + 1, init => by
      rw [List.range_succ, List.foldl_append, foldl_ite_range p f n init, sum_range_succ]
      simp only [List.foldl_cons, List.foldl_nil]
      split_ifs <;> simp [add_assoc]

theorem rawMulCoeffF_eq (a b : RingF) (degree : ℕ) :
    rawMulCoeffF a b degree = ∑ i ∈ range ringDegree,
      if i ≤ degree ∧ degree - i < ringDegree then ringFCoeff a i * ringFCoeff b (degree - i)
      else 0 :=
  (foldl_ite_range (fun i => i ≤ degree ∧ degree - i < ringDegree)
    (fun i => ringFCoeff a i * ringFCoeff b (degree - i)) ringDegree 0).trans (zero_add _)

/-- `ringFMul` is the product of `F[X]/(X^64 + 1)`. -/
theorem ringFMul_apply (a b : RingF) (out : Fin ringDegree) :
    ringFMul a b out = mul ringDegree (ringFCoeff a) (ringFCoeff b) out := by
  have D : ringDegree = 64 := rfl
  have o_lt : (out : ℕ) < ringDegree := out.isLt
  have split (g : ℕ → F) : ∑ i ∈ range ringDegree, g i =
      ∑ i ∈ range (out + 1), g i + ∑ i ∈ Ico (out + 1 : ℕ) ringDegree, g i :=
    (sum_range_add_sum_Ico g (by omega)).symm
  have low : rawMulCoeffF a b out =
      ∑ h ∈ range (out + 1), ringFCoeff b h * ringFCoeff a (out - h) := by
    rw [rawMulCoeffF_eq, split, sum_eq_zero (s := Ico (out + 1 : ℕ) ringDegree)
      (fun i member => by rw [ite_eq_right (by rw [mem_Ico] at member; omega)]), add_zero,
      ← sum_range_reflect]
    refine sum_congr rfl fun h member => ?_
    have := mem_range.mp member
    rw [ite_eq_left (by omega), show (out : ℕ) + 1 - 1 - h = out - h by omega,
      show (out : ℕ) - (out - h) = h by omega, mul_comm]
  have high : rawMulCoeffF a b (out + ringDegree) =
      ∑ h ∈ Ico (out + 1 : ℕ) ringDegree, ringFCoeff b h * ringFCoeff a (out + ringDegree - h) := by
    rw [rawMulCoeffF_eq, split, sum_eq_zero (s := range (out + 1))
      (fun i member => by rw [ite_eq_right (by rw [mem_range] at member; omega)]), zero_add]
    refine sum_nbij' (fun i => out + ringDegree - i) (fun h => out + ringDegree - h)
      (fun i member => ?_) (fun h member => ?_) (fun i member => ?_) (fun h member => ?_)
      (fun i member => ?_)
    · rw [mem_Ico] at member ⊢
      omega
    · rw [mem_Ico] at member ⊢
      omega
    · rw [mem_Ico] at member
      omega
    · rw [mem_Ico] at member
      omega
    · rw [mem_Ico] at member
      rw [ite_eq_left (by omega), show (out : ℕ) + ringDegree - (out + ringDegree - i) = i by omega,
        mul_comm]
  show rawMulCoeffF a b out - rawMulCoeffF a b (out + ringDegree) = _
  rw [low, high, mul, split, sub_eq_add_neg, ← sum_neg_distrib]
  congr 1
  · refine sum_congr rfl fun h member => ?_
    rw [mulX, ite_eq_left (by rw [mem_range] at member; omega)]
  · refine sum_congr rfl fun h member => ?_
    rw [mem_Ico] at member
    rw [mulX, ite_eq_right (by omega), mul_neg]

theorem ringFSum_apply : ∀ {n : ℕ} (terms : Fin n → RingF) (lane : Fin ringDegree),
    Commitment.ringFSum terms lane = ∑ i, terms i lane
  | 0, _, _ => rfl
  | n + 1, terms, lane => by
      rw [Fin.sum_univ_succ, ← ringFSum_apply (fun i => terms i.succ) lane]
      rfl

/-- Row `row` of the Ajtai commitment is `Σ_j mul(A_{row, j}, z_j)` over the blocks `j`. -/
theorem ajtaiRow_apply {shape : RingRelation.Shape} {rows : ℕ} (key : Commitment.Key shape rows)
    (z : Assignment shape) (row : Fin rows) (lane : Fin ringDegree) :
    Commitment.ajtaiRow key z row lane =
      ∑ j ∈ range (ConcreteRingColumnLayout.blockCount shape.carrierWidth),
        mul ringDegree (keyOf key row j) (fun h => cell z (ringDegree * j + h)) lane := by
  rw [Commitment.ajtaiRow, Commitment.blockSum, ringFSum_apply,
    ← Fin.sum_univ_eq_sum_range (fun j => mul ringDegree (keyOf key row j)
      (fun h => cell z (ringDegree * j + h)) lane)]
  refine sum_congr rfl fun j _ => ?_
  rw [ringFMul_apply]
  refine (mul_congr_left lane.isLt (fun i lt => ?_) _).trans (mul_congr_right _ _ _ fun h lt => ?_)
  · simp only [keyOf, row.isLt, j.isLt, lt, dite_true, ringFCoeff]
  · have inside : ringDegree * j + h < shape.carrierWidth := by
      have := (RingCarrierLayout.carrierColumn j ⟨h, lt⟩).isLt
      change (j : ℕ) * ringDegree + h < _ at this
      rw [Nat.mul_comm]
      exact this
    simp only [ringFCoeff, lt, dite_true, cell, inside]
    show z (RingCarrierLayout.carrierColumn j ⟨h, lt⟩) = _
    exact congrArg z (Fin.ext (by change (j : ℕ) * ringDegree + h = _; rw [Nat.mul_comm]))

end NightstreamFPrime.Spec.Compression.Claim
