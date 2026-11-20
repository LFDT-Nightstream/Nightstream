import NightstreamFPrime.Spec.Compression.Akita.FoldCheck
import NightstreamFPrime.Spec.Compression.Game.Complete

/-! The completeness of the checks `foldCheck` of one fold of Akita (ePrint 2026/1983, revision
of 2026-10-15, Theorem 10.9: "All honest algebraic checks are exact identities", with
Theorem 3.3 for the sum-checks and Theorem 7.4, item 1, for the ring reduction).

- `sumcheckWith_winsTo` is `Game.sumcheck_winsTo` for `sumcheckWith`.
- `foldCheck_winsTo` states that for a witness with the relation, the digit range statement and
  the binary statement, the prover has a message for each draw sequence, and `foldCheck` ends in
  the claim `(r_2, w̃(r_2))` after `2·μ' + 5` draws.
- `foldCheck_complete` is this statement for the probability of a strategy.

The probability that the response of a fold is not admissible is not in this file: it is the
term `ξ_h` of Akita (155). -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] (fits : Type → Prop)

omit [Algebra Zq E] in
/-- `Game.sumcheck_winsTo` for the sum-check with a second bad set in each round. -/
theorem sumcheckWith_winsTo {d bound : ℕ} {truth : List E → E} {round : List E → E[X]}
    (true_rounds : RoundTruth d bound truth round) (message : fits (Fin (d + 1) → E))
    (extra : List E → Finset E) (rest : ℕ) (tail : Option (List E × E) → ℕ → Bool → Prop) :
    ∀ (rounds : ℕ) (point : List E) (claim : E), point.length + rounds ≤ bound →
      claim = truth point →
      (∀ final : List E, final.length = point.length + rounds →
        tail (some (final, truth final)) rest true) →
      WinsTo fits (sumcheckWith d round extra rounds point claim) (rest + rounds) true tail := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim _ true_claim ended
      rw [true_claim]
      exact ended point rfl
  | succ rounds step =>
      intro point claim inside true_claim ended
      have within : point.length < bound := by omega
      refine ⟨rfl, message, fun index => (round point).coeff index, ?_⟩
      dsimp only
      rw [coeffPoly_coeff (round point) (true_rounds.degree point within),
        ite_eq_left (true_claim ▸ true_rounds.halves point within)]
      intro challenge
      show WinsTo fits (sumcheckWith d round extra rounds (point ++ [challenge])
        ((round point).eval challenge)) (rest + rounds) true tail
      rw [true_rounds.binds point challenge within]
      exact step (point ++ [challenge]) _ (by simp; omega) rfl fun final length =>
        ended final (by simp at length; omega)

/-- The completeness of the fused sum-check (125) with its final check (Akita Section 7.4): for
a true claim, the prover sends the true round polynomials and the value `w̃(r_2)`. -/
theorem fusedCheck_winsTo (scalar : fits E) (cubic : fits (Fin (3 + 1) → E)) (μ : ℕ)
    (support : ℕ → Prop) [DecidablePred support] (w : Table) (weight : ℕ → E) (target : E)
    (r : List E) (sClaim ζ γ : E) (rest : ℕ)
    (true_claim : target + γ * sClaim =
      combTruth μ (fusedTables w weight (virtWeight support r γ ζ)) (fusedComb E) []) :
    WinsTo fits (fusedCheck μ support w weight target r sClaim ζ γ) (rest + μ) true
      fun result _ _ => ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E w)) := by
  refine (winsTo_bind fits _ _ _ _ _).mpr (sumcheck_winsTo fits
    (fused_roundTruth μ w weight (virtWeight support r γ ζ)) cubic rest _ μ [] _ (by simp)
    true_claim fun final length => ?_)
  simp only [List.length_nil, zero_add] at length
  have last := fusedTruth_final final w weight (virtWeight support r γ ζ)
  rw [length] at last
  refine ⟨rfl, scalar, mle final (lifted E w), ?_⟩
  dsimp only
  rw [ite_eq_left last]
  exact ⟨final, length, rfl⟩

/-- The completeness of the checks of one fold (Akita Theorem 10.9 for the checks of Figure 6):
for a witness with the relation of the ring reduction, the digit range statement of Akita
Section 6.1 and the binary statement of Akita Remark 6.2, the prover has a message for each draw
sequence, and `foldCheck` ends in the claim `(r_2, w̃(r_2))` with `μ'` coordinates. The draws are
`α`, `τ_0`, `τ_1`, the `μ'` rounds of the range sum-check, `ζ_bin`, `γ` and the `μ'` rounds of
the fused sum-check. A message is one element of `E` or the coefficients of one round
polynomial. -/
theorem foldCheck_winsTo (R : Reduction E) (b : ℕ) (scalar : fits E)
    (range : fits (Fin (b / 2 + 1 + 1) → E)) (cubic : fits (Fin (3 + 1) → E))
    (support : ℕ → Prop) [DecidablePred support] {w : Table} (holds : R.Holds w)
    (ranged : RangeHolds b R.vars w) (binary : BinaryOn support R.vars w) :
    WinsTo fits (foldCheck R b support w) (0 + R.vars + 1 + 1 + R.vars + 1 + 1 + 1) true
      fun result _ _ =>
        ∃ r2 : List E, r2.length = R.vars ∧ result = some (r2, mle r2 (lifted E w)) := by
  intro α τ0 τ1
  have sum := rangeTruth_nil b (List.ofFn τ0) w
  rw [List.length_ofFn] at sum
  refine (winsTo_bind fits _ _ _ _ _).mpr (sumcheckWith_winsTo fits
    (range_roundTruth b R.vars (List.ofFn τ0) w) range _ (0 + R.vars + 1 + 1) _ R.vars [] 0
    (by simp) (sum.trans (mle_rangeResidual_eq_zero ranged _ List.length_ofFn)).symm
    fun final length => ?_)
  simp only [List.length_nil, zero_add] at length
  have last := rangeTruth_final b (List.ofFn τ0) final w
  rw [length] at last
  refine ⟨rfl, scalar, mle final (derived E w), ?_⟩
  dsimp only
  rw [ite_eq_left last]
  intro ζ γ
  refine fusedCheck_winsTo fits scalar cubic R.vars support w _ _ final _ ζ γ 0 ?_
  have virt := fusedTruth_virt support w ((R.rows α).batchWeight (List.ofFn τ1)) final γ ζ
  rw [length] at virt
  have batched : (R.rows α).Batched R.vars w (List.ofFn τ1) := R.batched_of_holds holds α _
  have zero : mle final (supportTable support w) = (0 : E) := by
    rw [mle_congr final (u := fun _ => 0) fun x small => by
      unfold supportTable
      split
      · rename_i member
        rw [derived, binary x (length ▸ small) member, map_zero]
      · rfl]
    exact mle_const final 0
  rw [virt, zero, mul_zero, add_zero]
  exact congrArg (· + γ * mle final (derived E w)) batched.symm

/-- Perfect completeness of the checks of one fold: for a witness with the relation, the digit
range statement and the binary statement, a strategy of the prover reaches the claim
`(r_2, w̃(r_2))` with probability 1. -/
theorem foldCheck_complete (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] {w : Table} (holds : R.Holds w) (ranged : RangeHolds b R.vars w)
    (binary : BinaryOn support R.vars w) :
    ∃ strategy : Strategy (Option (List E × E)),
      prob strategy (foldCheck R b support w) (fun result =>
        ∃ r2 : List E, r2.length = R.vars ∧ result = some (r2, mle r2 (lifted E w))) = 1 :=
  ⟨_, prob_honest (fun _ => True)
    (foldCheck_winsTo (fun _ => True) R b trivial trivial trivial support holds ranged binary)
    fun _ _ _ ended => ended⟩

end NightstreamFPrime.Spec.Compression.Akita
