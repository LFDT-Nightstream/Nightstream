import CompressionLean.Ports.Layer1.Budget
import CompressionLean.Layer1.Cost
import NightstreamFPrime.Spec.Algebra

/-!
Owns the layer-1 budget at Rust's split (`layer1_budget`): layer 1 gets
`2^-t`, with `t` the layer bits of `finish.rs`, and each of its two WHIR
commitments gets `2^-(t+1)` (`Relation::new` targets `t + 1` bits).

Plonky3 composes each commitment's error as its WHIR terms plus every term
the caller charges, times the commitment's candidate count. P0 is charged
the draws of `outer_terms` (`crates/neo-spartan/src/lib.rs`), each
`degree · L1 / 2^191` (`rustDegrees`, `Ports/Layer1/Budget.lean`); P1 is
charged the setup queries, `L1 · 2^(1-t_q)`. `Pcs::new` accepts a
configuration only when each composed error is at most `2^-(t+1)`, which is
what `p0` and `p1` state. `layer1Count_le_rust` shows that the Lean count is
at most what Rust charges, so the two checks bound the whole layer.

The query leaves are low bits of Goldilocks words. With at most 35 setup
variables a leaf lands in a wrong fold's agreement set with probability at
most `1/2` (`query_half`), so the charge holds under that law.
-/

namespace CompressionLean.Final

open Polynomial CompressionLean.Layer1
open NightstreamFPrime.Spec (goldilocksModulus)


theorem gkrLayerCost_le_rust (s : GkrShape) (small : ∀ t, s.factors t ≤ 2) (n : ℕ) (E : ℕ → ℕ) (k : ℕ)
    (positive : 1 ≤ k) :
    gkrLayerCost s n E k ≤ n * (4 * k + (s.active k).length) + k * E k := by
  have degree : s.degree k ≤ 4 := s.degree_le k 4 (by norm_num) fun t _ => by
    unfold GkrShape.partFactors
    split
    · have := small t; omega
    · omega
  obtain ⟨i, rfl⟩ : ∃ i, k = i + 1 := ⟨k - 1, by omega⟩
  have h1 : n * ((s.active (i + 1)).length - 1) ≤ n * (s.active (i + 1)).length :=
    Nat.mul_le_mul_left _ (Nat.sub_le _ _)
  have h2 : i * (n * s.degree (i + 1) + E (i + 1)) ≤ i * (n * 4 + E (i + 1)) :=
    Nat.mul_le_mul_left _ (by have := Nat.mul_le_mul_left n degree; omega)
  unfold gkrLayerCost
  rw [Nat.add_sub_cancel]
  nlinarith

theorem gkrCost_le_rust (s : GkrShape) (small : ∀ t, s.factors t ≤ 2) (n : ℕ) (E : ℕ → ℕ) :
    ∀ remaining k, 1 ≤ k →
      gkrCost s n E remaining k ≤
        ∑ j ∈ Finset.Ico k (k + remaining), (n * (4 * j + (s.active j).length) + j * E j) := by
  intro remaining
  induction remaining with
  | zero => intro k _; simp [gkrCost]
  | succ r ih =>
      intro k positive
      rw [gkrCost, Finset.sum_eq_sum_Ico_succ_bot (by omega : k < k + (r + 1)),
        show k + (r + 1) = k + 1 + r by omega]
      exact Nat.add_le_add (gkrLayerCost_le_rust s small n E k positive) (ih (k + 1) (by omega))

/-- A whole GKR costs at most `n` times its `outer_terms` degree, plus its
riding values. -/
theorem gkrCost_le_rustGkr (s : GkrShape) (small : ∀ t, s.factors t ≤ 2) (n : ℕ) (E : ℕ → ℕ) :
    gkrCost s n E s.layers 1 ≤ n * rustGkr s + ∑ j ∈ Finset.Ico 1 (s.layers + 1), j * E j := by
  have bound := gkrCost_le_rust s small n E s.layers 1 le_rfl
  rw [show 1 + s.layers = s.layers + 1 by omega, Finset.sum_add_distrib, ← Finset.mul_sum] at bound
  exact bound

/-- The Lean count of layer 1 is at most what Rust charges P0. -/
theorem layer1Count_le_rust (s : Shape) (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) :
    layer1Count s L0 L1 ≤ L0 * L1 * rustDegrees s := by
  have earlySmall : ∀ t, (earlyShape s).factors t ≤ 2 := by
    intro t; simp only [earlyShape]; split_ifs <;> omega
  have lateSmall : ∀ t, (lateShape s).factors t ≤ 2 := by
    intro t; simp only [lateShape]; split_ifs <;> omega
  -- The slot-block test rides on the run layer only.
  have ride : ∑ j ∈ Finset.Ico 1 ((earlyShape s).layers + 1), j * earlyRide s L0 j ≤ s.runVars * L0 := by
    have each : ∀ j ∈ Finset.Ico 1 ((earlyShape s).layers + 1),
        j * earlyRide s L0 j = if s.runVars = j then s.runVars * L0 else 0 := by
      intro j _
      unfold earlyRide
      by_cases same : j = s.runVars
      · rw [if_pos same, if_pos same.symm, same]
      · rw [if_neg same, if_neg (Ne.symm same), mul_zero]
    rw [Finset.sum_congr rfl each, Finset.sum_ite_eq]
    split_ifs <;> omega
  have earlyGkr := gkrCost_le_rustGkr (earlyShape s) earlySmall L0 (earlyRide s L0)
  have lateGkr := gkrCost_le_rustGkr (lateShape s) lateSmall (L0 * L1) (fun _ => 0)
  simp only [mul_zero, Finset.sum_const_zero, add_zero] at lateGkr
  have early : earlyCount s L0 ≤ L0 * rustEarly s := by
    unfold earlyCount rustEarly
    rw [Nat.mul_comm s.runVars L0] at ride
    nlinarith [Nat.zero_le (L0 * 2 ^ s.runVars)]
  have linear : linearCount s L0 ≤ L0 * rustLinear s := by
    unfold linearCount rustLinear Shape.cubeVars
    nlinarith
  have late : lateCount s L0 L1 ≤ L0 * L1 * rustLate s := by
    unfold lateCount rustLate
    nlinarith
  have product : L0 ≤ L0 * L1 := Nat.le_mul_of_pos_right _ h1
  have one : 1 ≤ L0 * L1 := Nat.mul_le_mul h0 h1
  unfold layer1Count rustDegrees
  nlinarith [Nat.mul_le_mul_right (rustEarly s) product, Nat.mul_le_mul_right (rustLinear s) product]

/-- `2·(2^(v-3) - 1)·(⌊p / 2^(v-2)⌋ + 1) ≤ p` up to `v = 35`; it fails at `36`. -/
theorem goldilocks_half : ∀ v < 36, 2 * ((2 ^ (v - 3) - 1) * (goldilocksModulus / 2 ^ (v - 2) + 1)) ≤ goldilocksModulus := by
  decide

/-- A query leaf drawn as the low `setupVars - 2` bits of a Goldilocks word
agrees with a wrong fold at most half the time, for at most 35 setup
variables. -/
theorem query_half (s : Shape) (index : s.setupVars ≤ 35) :
    ((((2 ^ (s.setupVars - 3) - 1) * (goldilocksModulus / 2 ^ (s.setupVars - 2) + 1) : ℕ) : ℝ) / goldilocksModulus) ≤
      1 / 2 := by
  rw [div_le_div_iff₀ (by exact_mod_cast Nat.pos_of_neZero goldilocksModulus) (by norm_num)]
  exact_mod_cast (by have := goldilocks_half s.setupVars (by omega); omega :
    (2 ^ (s.setupVars - 3) - 1) * (goldilocksModulus / 2 ^ (s.setupVars - 2) + 1) * 2 ≤ 1 * goldilocksModulus)

/-- **The layer-1 budget at Rust's split.** If each commitment's composed
error, as Plonky3 charges it, is at most `2^-(t+1)`, the whole layer errs
with probability at most `2^-t`. `whirP0`, `whirP1` are the WHIR errors of
the idealized openings (`Opens`). The query leaves are the low bits of
Goldilocks words; `index` is the setup bound of `query_half`. -/
theorem layer1_budget (s : Shape) (L0 L1 size t : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (big : 2 ^ 191 ≤ size)
    (index : s.setupVars ≤ 35) (whirP0 whirP1 : ℝ)
    (p0 : whirP0 + L0 * L1 * rustDegrees s / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : whirP1 + L1 * (1 / 2) ^ (s.queries - 1) ≤ 1 / 2 ^ (t + 1)) :
    whirP0 + whirP1 + layer1Error s L0 L1 (2 ^ (s.setupVars - 2)) goldilocksModulus size ≤ 1 / 2 ^ t := by
  unfold layer1Error queryError
  have positive : (0 : ℝ) < 2 ^ 191 := by positivity
  have sizeBig : (2 : ℝ) ^ 191 ≤ size := by exact_mod_cast big
  have count : (layer1Count s L0 L1 : ℝ) ≤ L0 * L1 * rustDegrees s := by
    exact_mod_cast layer1Count_le_rust s L0 L1 h0 h1
  have field : (layer1Count s L0 L1 : ℝ) / size ≤ L0 * L1 * rustDegrees s / 2 ^ 191 :=
    div_le_div₀ (by positivity) count positive sizeBig
  have half := query_half s index
  have shrink : ((((2 ^ (s.setupVars - 3) - 1) * (goldilocksModulus / 2 ^ (s.setupVars - 2) + 1) : ℕ) : ℝ) /
      goldilocksModulus) ^ s.queries ≤ (1 / 2) ^ (s.queries - 1) :=
    (pow_le_pow_left₀ (by positivity) half _).trans
      (pow_le_pow_of_le_one (by norm_num) (by norm_num) (Nat.sub_le _ _))
  have query : (L1 : ℝ) * ((((2 ^ (s.setupVars - 3) - 1) * (goldilocksModulus / 2 ^ (s.setupVars - 2) + 1) : ℕ) : ℝ) /
      goldilocksModulus) ^ s.queries ≤ L1 * (1 / 2) ^ (s.queries - 1) :=
    mul_le_mul_of_nonneg_left shrink (by positivity)
  have split : (1 : ℝ) / 2 ^ (t + 1) + 1 / 2 ^ (t + 1) = 1 / 2 ^ t := by
    rw [pow_succ]; field_simp; ring
  linarith

end CompressionLean.Final
