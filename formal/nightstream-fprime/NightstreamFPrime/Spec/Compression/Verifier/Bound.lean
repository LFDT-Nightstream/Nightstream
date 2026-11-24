import NightstreamFPrime.Spec.Compression.Verifier.Sound

/-! The soundness bound of the verifier game (`soundness`): a prover with `q` oracle queries in the
state-restoration experiment names an accepted proof whose committed `W` does not satisfy
`Relation` with probability at most `q · error`, where `error` bounds the bad-set density of each
draw (`game_densityLe`):

`error = Σ_tables ((N + T) / |F|)^L + (s − 1 + E + max(D − 1, m − 1)) / |K| + (2 + span) / |K|
  + (L + 2) / |K| + 2 / |K|`.

The Akita openings are ideal in this bound; Akita's knowledge error is a separate term. At the
production shapes, `error < 2^−119` and the sum of Rust `error_bits` (585 / |K| and the lookups) is
below `2^−118` (`production_error_lt`, `production_rust_error_lt`). -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra
open NightstreamFPrime.Spec.Compression.Game Polynomial

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K]

variable {P : Params F K}

/-- The largest bad-set density of a draw of the game. -/
noncomputable def error (_vf : VerifierData P) : ℝ :=
  tauError P + challengeError P + (P.lookups + 2 : ℕ) / Fintype.card K +
    2 / Fintype.card K

omit [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] [Algebra F K] in
theorem tauError_nonneg : 0 ≤ tauError P :=
  sum_nonneg fun c _ => by split <;> positivity

omit [Field F] [Fintype F] [DecidableEq F] [DecidableEq K] [Algebra F K] in
theorem challengeError_nonneg : 0 ≤ challengeError P := by
  unfold challengeError
  positivity

omit [DecidableEq F] [DecidableEq K] [Algebra F K] in
theorem error_nonneg (vf : VerifierData P) : 0 ≤ error vf := by
  unfold error
  have := tauError_nonneg (P := P)
  have := challengeError_nonneg (P := P)
  positivity

/-- Each draw of `game` has density at most `error`. -/
theorem game_densityLe (vf : VerifierData P) (wf : WellFormed P) : (game vf).DensityLe (error vf) := by
  have tau := tauError_nonneg (P := P)
  have challenge := challengeError_nonneg (P := P)
  have degree : (0 : ℝ) ≤ (P.lookups + 2 : ℕ) / Fintype.card K := by positivity
  have two : (0 : ℝ) ≤ 2 / Fintype.card K := by positivity
  intro st W
  refine ⟨(density_tauBad_le P W).trans (by unfold error; linarith), fun τ V => ?_⟩
  refine ⟨(density_challengeBad_le st W V τ wf).trans (by unfold error; linarith), fun ch => ?_⟩
  have main_truth := combRound_truth P.witnessVars (P.lookups + 2) (mainTables P st W V τ ch) (mainComb P τ ch)
    (mainComb_degree P τ ch)
  refine densityLe_bind ((sumcheck_densityLe (P.lookups + 2) P.witnessVars _ main_truth.degree P.witnessVars []
    _ (by simp)).mono (by unfold error; push_cast at degree ⊢; linarith))
    (Ends.of_forall (P := fun _ => True) (fun _ => trivial) _) fun first _ => ?_
  rcases first with _ | ⟨ρ, value⟩
  · trivial
  intro vals
  have key_truth := combRound_truth (P.laneVars + P.keyBlockVars) 2 (keyTables ch ρ) productComb
    productComb_degree
  refine densityLe_bind ((sumcheck_densityLe 2 _ _ key_truth.degree _ [] _ (by simp)).mono
    (by unfold error; push_cast at degree ⊢; linarith))
    (Ends.of_forall (P := fun _ => True) (fun _ => trivial) _) fun second _ => ?_
  rcases second with _ | ⟨kp, kfinal⟩
  · trivial
  intro kv
  trivial

/-- Soundness in the state-restoration experiment with `q` oracle queries. -/
theorem soundness (vf : VerifierData P) (wf : WellFormed P) (counts : CountsNonzero P)
    (matrix : MatrixSpec vf)
    (rows : P.rows ≤ 2 ^ P.rowVars)
    (key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars)
    (s : Strategy (Option (Option (Statement P × (ℕ → F))))) (q : ℕ) :
    prob s (restore (game vf) q [])
        (fun r => ¬ ∀ a, r = some a → ∀ o, a = some o → Relation P o.1 o.2) ≤ q * error vf :=
  restore_sound (game vf) s _ (error vf)
    (game_avoids vf wf counts matrix rows key_fits) (game_densityLe vf wf)
    (error_nonneg vf) q

/-- `q = 2^64 − 59`. -/
def fieldOrder : ℕ := 18446744073709551557

/-- `error` at the production shapes, with each table bounded by the larger one: three claims,
`E = 18 + 1 + 4`, `D = 64`, `m ≤ 320`, `span ≤ 27`, `L = 4`, `N + T ≤ 2^25 + 20737`. -/
theorem production_error_lt :
    3 * (((2 ^ 25 + 20737 : ℕ) : ℝ) / fieldOrder) ^ 4 +
      (((3 - 1 + (23 + max (64 - 1) (320 - 1)) : ℕ) : ℝ) / (fieldOrder ^ 2 : ℕ) +
        ((2 + 27 : ℕ) : ℝ) / (fieldOrder ^ 2 : ℕ)) +
      6 / (fieldOrder ^ 2 : ℕ) + 2 / (fieldOrder ^ 2 : ℕ) <
      1 / 2 ^ 119 := by
  norm_num [fieldOrder]

/-- The sum of Rust `error_bits` at the production shapes: the lookups of the two tables
(`N = 2^25` and `2^23`, `T = 20737`) and `(344 + 29 + 162 + 50) / q²`. -/
theorem production_rust_error_lt :
    (((2 ^ 25 + 20737 : ℕ) : ℝ) / fieldOrder) ^ 4 + (((2 ^ 23 + 20737 : ℕ) : ℝ) / fieldOrder) ^ 4 +
      (344 + 29 + 162 + 50 : ℝ) / (fieldOrder ^ 2 : ℕ) < 1 / 2 ^ 118 := by
  norm_num [fieldOrder]

omit [Field F] [DecidableEq F] [Field K] [DecidableEq K] [Algebra F K] in
/-- `error` at the production shapes, with at most three claims, is at most the expression of
`production_error_lt`. -/
theorem error_le_production (vf : VerifierData P) (card_F : Fintype.card F = fieldOrder)
    (card_K : Fintype.card K = fieldOrder ^ 2) (claims : P.claims ≤ 3)
    (equations : P.equations = 23) (lanes : P.lanes = 64) (public_le : maxPublic P ≤ 320)
    (span_le : P.spanVars ≤ 27) (lookups : P.lookups = 4)
    (tables : ∀ c, (P.shape c).lookup = true →
      2 ^ (P.shape c).witness.vars + (2 * (P.shape c).bound + 1) ≤ 2 ^ 25 + 20737) :
    error vf ≤ 3 * (((2 ^ 25 + 20737 : ℕ) : ℝ) / fieldOrder) ^ 4 +
      (((3 - 1 + (23 + max (64 - 1) (320 - 1)) : ℕ) : ℝ) / (fieldOrder ^ 2 : ℕ) +
        ((2 + 27 : ℕ) : ℝ) / (fieldOrder ^ 2 : ℕ)) +
      6 / (fieldOrder ^ 2 : ℕ) + 2 / (fieldOrder ^ 2 : ℕ) := by
  have q_pos : (0 : ℝ) < fieldOrder := by norm_num [fieldOrder]
  have tau : tauError P ≤ 3 * (((2 ^ 25 + 20737 : ℕ) : ℝ) / fieldOrder) ^ 4 := by
    unfold tauError
    rw [card_F, lookups]
    have each : ∀ c : Fin P.claims, (if (P.shape c).lookup then
        (((2 ^ (P.shape c).witness.vars + (2 * (P.shape c).bound + 1) : ℕ) : ℝ) /
          fieldOrder) ^ 4 else 0) ≤ (((2 ^ 25 + 20737 : ℕ) : ℝ) / fieldOrder) ^ 4 := fun c => by
      split
      · rename_i table
        refine pow_le_pow_left₀ (by positivity) (div_le_div_of_nonneg_right ?_ q_pos.le) 4
        exact_mod_cast tables c table
      · positivity
    refine (sum_le_sum fun c _ => each c).trans ?_
    rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast claims) (by positivity)
  have challenge : challengeError P ≤
      (((3 - 1 + (23 + max (64 - 1) (320 - 1)) : ℕ) : ℝ) / (fieldOrder ^ 2 : ℕ) +
        ((2 + 27 : ℕ) : ℝ) / (fieldOrder ^ 2 : ℕ)) := by
    unfold challengeError
    rw [card_K, equations, lanes]
    refine add_le_add (div_le_div_of_nonneg_right ?_ (by positivity))
      (div_le_div_of_nonneg_right ?_ (by positivity))
    · exact_mod_cast (by omega : P.claims - 1 + (23 + max (64 - 1) (maxPublic P - 1)) ≤
        3 - 1 + (23 + max (64 - 1) (320 - 1)))
    · exact_mod_cast (by omega : 2 + P.spanVars ≤ 2 + 27)
  unfold error
  rw [lookups, card_K]
  push_cast at tau challenge ⊢
  linarith

end NightstreamFPrime.Spec.Compression.Verifier
