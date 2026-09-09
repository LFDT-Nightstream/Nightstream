import NightstreamFPrime.Spec.SumCheck.GoldilocksRoots
import NightstreamFPrime.Spec.SumCheck.FixedPhase.Sequential
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Data.Real.Basic

/-!
Interactive SumCheck over the actual K field. The semantic polynomial and
strategy are fixed before the verifier samples any round challenge. A strategy
sees only the prior challenge prefix and may abort before its next message.
Private prover coins can be fixed in that strategy before this experiment.

Uniform averaging includes every verifier challenge stream. It never
conditions on successful or non-aborting runs. The event records a collision
at a visited round, including a collision followed by a later prover abort.
Binding of the source witness, mixing, and Fiat-Shamir are separate claims.
-/

namespace NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal

open scoped BigOperators
open GoldilocksRoots (ops fullChallengeSet)
attribute [local instance] Classical.propDecidable

/-- Each message is chosen from past challenges only. None is prover abort. -/
abbrev Strategy (degree : Nat) := List K → Option (FixedPolynomial K degree)

/-- The semantic round is fixed by the prefix and remaining dimension. -/
def expected (q : List K → K) (fixed : List K) (remaining : Nat) (point : K) : K :=
  HypercubeTruth.sumCompletions ops q (fixed ++ [point]) remaining

def Hit {degree : Nat} (q : List K → K) (fixed : List K) (remaining : Nat)
    (message : FixedPolynomial K degree) (challenge : K) : Prop :=
  (∃ point, message.evaluate ops point ≠ expected q fixed remaining point) ∧
    message.evaluate ops challenge = expected q fixed remaining challenge

/-- A visited adaptive round has distinct claimed and semantic polynomials
that agree at that round's fresh verifier challenge. -/
def Collision {degree : Nat} (q : List K → K) (strategy : Strategy degree)
    (fixed : List K) : List K → Prop
  | [] => False
  | challenge :: rest =>
      match strategy fixed with
      | none => False
      | some message => Hit q fixed rest.length message challenge ∨
          Collision q strategy (fixed ++ [challenge]) rest

/-- Independent uniform samples from the stated set of K elements. -/
noncomputable def uniformAverage (samples : Finset K) : Nat → (List K → ℝ) → ℝ
  | 0, value => value []
  | remaining + 1, value =>
      𝔼 challenge ∈ samples,
        uniformAverage samples remaining (fun rest => value (challenge :: rest))

noncomputable def collisionProbability {degree : Nat} (samples : Finset K) (q : List K → K)
    (strategy : Strategy degree) (fixed : List K) (remaining : Nat) : ℝ := by
  classical
  exact uniformAverage samples remaining (fun challenges =>
    if Collision q strategy fixed challenges then 1 else 0)

/-- The ideal full-field sampling set has positive cardinality. -/
theorem sampleSpace_nonempty : fullChallengeSet.Nonempty := by
  apply Finset.card_pos.mp
  rw [GoldilocksRoots.fullChallengeSet_card]
  exact pow_pos (by decide : 0 < goldilocksModulus) 2

/-- Averaging a constant over the independent sample stream preserves it. -/
theorem uniformAverage_const (samples : Finset K) (nonempty : samples.Nonempty)
    (remaining : Nat) (value : ℝ) :
    uniformAverage samples remaining (fun _ => value) = value := by
  induction remaining with
  | zero => rfl
  | succ remaining ih =>
      simp only [uniformAverage, ih, Finset.expect_const nonempty]

/-- Only streams of the selected length contribute to this average. -/
theorem uniformAverage_congr (samples : Finset K) (remaining : Nat) (left right : List K → ℝ)
    (equal : ∀ challenges, challenges.length = remaining → left challenges = right challenges) :
    uniformAverage samples remaining left = uniformAverage samples remaining right := by
  induction remaining generalizing left right with
  | zero => exact equal [] rfl
  | succ remaining ih =>
      apply Finset.expect_congr rfl
      intro challenge _
      apply ih
      intro rest length
      exact equal (challenge :: rest) (by simp [length])

/-- Compare events on every stream of the selected exact length. -/
theorem uniformAverage_mono (samples : Finset K) (remaining : Nat) (left right : List K → ℝ)
    (ordered : ∀ challenges, challenges.length = remaining → left challenges ≤ right challenges) :
    uniformAverage samples remaining left ≤ uniformAverage samples remaining right := by
  induction remaining generalizing left right with
  | zero => exact ordered [] rfl
  | succ remaining ih =>
      apply Finset.expect_le_expect
      intro challenge _
      apply ih
      intro rest length
      exact ordered (challenge :: rest) (by simp [length])

/-- Addition distributes through the independent challenge average. -/
theorem uniformAverage_add (samples : Finset K) (remaining : Nat)
    (left right : List K → ℝ) :
    uniformAverage samples remaining (fun challenges => left challenges + right challenges) =
      uniformAverage samples remaining left + uniformAverage samples remaining right := by
  induction remaining generalizing left right with
  | zero => rfl
  | succ remaining ih =>
      simp only [uniformAverage]
      simp_rw [ih]
      exact Finset.expect_add_distrib _ _ _

/-- A separate finite average commutes with the independent challenge stream. -/
theorem uniformAverage_expect {Index : Type*} (samples : Finset K) (remaining : Nat)
    (indices : Finset Index) (value : List K → Index → ℝ) :
    uniformAverage samples remaining (fun challenges => 𝔼 index ∈ indices, value challenges index) =
      (𝔼 index ∈ indices, uniformAverage samples remaining (fun challenges => value challenges index)) := by
  induction remaining generalizing value with
  | zero => rfl
  | succ remaining ih =>
      simp only [uniformAverage]
      simp_rw [ih]
      exact Finset.expect_comm _ _ _

private theorem hit_count_le {degree : Nat} (samples : Finset K) (q : List K → K)
    (fixed : List K) (remaining : Nat) (message semantic : FixedPolynomial K degree)
    (represents : FixedPhase.Represents ops semantic (expected q fixed remaining)) :
    (samples.filter (Hit q fixed remaining message)).card ≤ degree := by
  classical
  by_cases different : ∃ point, message.evaluate ops point ≠ expected q fixed remaining point
  · have distinct : ∃ point, message.evaluate ops point ≠ semantic.evaluate ops point := by
      obtain ⟨point, unequal⟩ := different
      exact ⟨point, by simpa only [represents point] using unequal⟩
    have subset : samples.filter (Hit q fixed remaining message) ⊆
        samples.filter (fun point => message.evaluate ops point = semantic.evaluate ops point) := by
      intro point member
      obtain ⟨inside, hit⟩ := Finset.mem_filter.mp member
      exact Finset.mem_filter.mpr ⟨inside, hit.2.trans (represents point).symm⟩
    exact (Finset.card_le_card subset).trans
      (GoldilocksRoots.agreement_count_le message semantic samples distinct)
  · have empty : samples.filter (Hit q fixed remaining message) = ∅ := by
      apply Finset.eq_empty_iff_forall_notMem.mpr
      intro point member
      exact different (Finset.mem_filter.mp member).2.1
    simp only [empty, Finset.card_empty]
    exact Nat.zero_le degree

private theorem hit_probability_le {degree : Nat} (samples : Finset K) (q : List K → K)
    (fixed : List K) (remaining : Nat) (message semantic : FixedPolynomial K degree)
    (represents : FixedPhase.Represents ops semantic (expected q fixed remaining)) :
    (𝔼 challenge ∈ samples,
      if Hit q fixed remaining message challenge then (1 : ℝ) else 0) ≤
        (degree : ℝ) / samples.card := by
  classical
  rw [Finset.expect_eq_sum_div_card, ← Finset.sum_filter]
  simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg _)
  exact_mod_cast hit_count_le samples q fixed remaining message semantic represents

private theorem collisionProbability_le_card {degree totalRounds : Nat}
    (samples : Finset K) (nonempty : samples.Nonempty)
    (q : List K → K) (strategy : Strategy degree)
    (representable : FixedPhase.Sequential.RoundRepresentable ops q degree totalRounds)
    (fixed : List K) (remaining : Nat) (length : fixed.length + remaining = totalRounds) :
    collisionProbability samples q strategy fixed remaining ≤
      (remaining : ℝ) * degree / samples.card := by
  classical
  induction remaining generalizing fixed with
  | zero => simp [collisionProbability, uniformAverage, Collision]
  | succ remaining ih =>
      cases chosen : strategy fixed with
      | none =>
          have zero : collisionProbability samples q strategy fixed (remaining + 1) = 0 := by
            simp only [collisionProbability, uniformAverage, Collision, chosen,
              if_false, uniformAverage_const samples nonempty, Finset.expect_const nonempty]
          rw [zero]
          positivity
      | some message =>
          obtain ⟨semantic, represents⟩ := representable fixed remaining (by omega)
          have step (challenge : K) :
              uniformAverage samples remaining (fun rest =>
                if Collision q strategy fixed (challenge :: rest) then (1 : ℝ) else 0) ≤
              (if Hit q fixed remaining message challenge then 1 else 0) +
                (remaining : ℝ) * degree / samples.card := by
            by_cases hit : Hit q fixed remaining message challenge
            · have averageOne : uniformAverage samples remaining (fun rest =>
                  if Collision q strategy fixed (challenge :: rest) then (1 : ℝ) else 0) = 1 := by
                refine (uniformAverage_congr samples remaining _ (fun _ => 1) ?_).trans
                  (uniformAverage_const samples nonempty remaining 1)
                intro rest restLength
                simp only [Collision, chosen, restLength, hit, true_or, if_true]
              rw [averageOne, if_pos hit]
              have nonnegative : 0 ≤ (remaining : ℝ) * degree / samples.card := by positivity
              linarith
            · have averageTail : uniformAverage samples remaining (fun rest =>
                  if Collision q strategy fixed (challenge :: rest) then (1 : ℝ) else 0) =
                  collisionProbability samples q strategy (fixed ++ [challenge]) remaining := by
                apply uniformAverage_congr
                intro rest restLength
                simp only [Collision, chosen, restLength, hit, false_or]
              rw [averageTail, if_neg hit, zero_add]
              apply ih (fixed ++ [challenge])
              simp only [List.length_append, List.length_singleton]
              omega
          change (𝔼 challenge ∈ samples,
            uniformAverage samples remaining (fun rest =>
              if Collision q strategy fixed (challenge :: rest) then (1 : ℝ) else 0)) ≤ _
          calc
            _ ≤ (𝔼 challenge ∈ samples,
                ((if Hit q fixed remaining message challenge then (1 : ℝ) else 0) +
                  (remaining : ℝ) * degree / samples.card)) :=
              Finset.expect_le_expect (fun challenge _ => step challenge)
            _ = (𝔼 challenge ∈ samples,
                if Hit q fixed remaining message challenge then (1 : ℝ) else 0) +
                  (remaining : ℝ) * degree / samples.card := by
              rw [Finset.expect_add_distrib, Finset.expect_const nonempty]
            _ ≤ (degree : ℝ) / samples.card +
                  (remaining : ℝ) * degree / samples.card :=
              _root_.add_le_add
                (hit_probability_le samples q fixed remaining message semantic represents) le_rfl
            _ = ((remaining + 1 : Nat) : ℝ) * degree / samples.card := by
              push_cast
              ring

/-- Causal round strategies have the interactive degree-times-rounds bound.
Both q and the strategy are arguments fixed before these uniform K samples.
The expected round bound comes from the existing prefix-only representation
contract, not from a supplied degree annotation on a sampled round. -/
theorem collisionProbability_le {degree totalRounds : Nat}
    (q : List K → K) (strategy : Strategy degree)
    (representable : FixedPhase.Sequential.RoundRepresentable ops q degree totalRounds)
    (fixed : List K) (remaining : Nat) (length : fixed.length + remaining = totalRounds) :
    collisionProbability fullChallengeSet q strategy fixed remaining ≤
      (remaining : ℝ) * degree / (goldilocksModulus ^ 2 : Nat) := by
  have bound := collisionProbability_le_card fullChallengeSet sampleSpace_nonempty
    q strategy representable fixed remaining length
  simpa only [GoldilocksRoots.fullChallengeSet_card] using bound

end NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal
