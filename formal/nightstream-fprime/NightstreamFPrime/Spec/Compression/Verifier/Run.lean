import NightstreamFPrime.Spec.Compression.Verifier.Bound

/-! The bound of one interactive run of the verifier game (`run_soundness`): a prover makes the
verifier accept claims whose committed `W` does not satisfy `Relation` with probability at most
`runCost`, the sum of the bad-set densities of the draws on one path (`game_costLe`):

`runCost = Σ_tables ((N + T) / |F|)^L + (s − 1 + E + max(D − 1, m − 1)) / |K| + (2 + span) / |K|
  + n · 3 / |K| + cv / |K| + (lv + n_A) · 2 / |K|`.

This is the quantity that Rust `error_bits` sums. `soundness` bounds the non-interactive argument
with `q` queries by `q · error` instead. The Akita openings are ideal in both bounds. -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra
open NightstreamFPrime.Spec.Compression.Game

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K]

variable {P : Params F K}

/-- The sum of the bad-set densities of the draws on one path of `game`. -/
noncomputable def runCost (vf : VerifierData P) : ℝ :=
  tauError P + challengeError P + P.witnessVars * (3 / Fintype.card K) +
    vf.columnVars / Fintype.card K + (P.laneVars + P.keyBlockVars) * (2 / Fintype.card K)

/-- One path of `game` has bad-set densities that sum to at most `runCost`. -/
theorem game_costLe (vf : VerifierData P) (wf : WellFormed P) : (game vf).CostLe (runCost vf) := by
  have challenge := challengeError_nonneg (P := P)
  have three : (0 : ℝ) ≤ P.witnessVars * (3 / Fintype.card K) := by positivity
  have column : (0 : ℝ) ≤ vf.columnVars / Fintype.card K := by positivity
  have two : (0 : ℝ) ≤ (P.laneVars + P.keyBlockVars) * (2 / Fintype.card K) := by positivity
  intro st W
  refine ⟨challengeError P + P.witnessVars * (3 / Fintype.card K) +
    vf.columnVars / Fintype.card K + (P.laneVars + P.keyBlockVars) * (2 / Fintype.card K),
    by have := density_tauBad_le P W; unfold density at this; unfold runCost; linarith,
    fun τ V => ?_⟩
  refine ⟨P.witnessVars * (3 / Fintype.card K) +
    (vf.columnVars / Fintype.card K + (P.laneVars + P.keyBlockVars) * (2 / Fintype.card K)),
    by have := density_challengeBad_le st W V τ wf; unfold density at this; linarith,
    fun ch => ?_⟩
  have main_truth := combRound_truth P.witnessVars 3 (mainTables P st W V τ ch) mainComb
    mainComb_degree
  refine (costLe_bind (δ := vf.columnVars / Fintype.card K +
    (P.laneVars + P.keyBlockVars) * (2 / Fintype.card K))
    (sumcheck_costLe 3 P.witnessVars _ main_truth.degree P.witnessVars [] _ (by simp))
    fun first => ?_).mono (by push_cast; linarith)
  rcases first with _ | ⟨ρ, value⟩
  · show (0 : ℝ) ≤ _
    positivity
  intro vals
  refine ⟨(P.laneVars + P.keyBlockVars) * (2 / Fintype.card K),
    by have := density_columnBad_le vf.columnVars vals.2.1 (columnAt P V ρ); unfold density at this
       linarith, fun col => ?_⟩
  have key_truth := combRound_truth (P.laneVars + P.keyBlockVars) 2 (keyTables ch ρ) productComb
    productComb_degree
  refine (costLe_bind (δ := 0) (sumcheck_costLe 2 _ _ key_truth.degree _ [] _ (by simp))
    fun second => ?_).mono (by push_cast; linarith)
  rcases second with _ | ⟨kp, kfinal⟩
  · show (0 : ℝ) ≤ 0
    exact le_rfl
  intro kv
  show (0 : ℝ) ≤ 0
  exact le_rfl

/-- Soundness of one interactive run. -/
theorem run_soundness (vf : VerifierData P) (wf : WellFormed P) (counts : CountsNonzero P)
    (matrix : MatrixSpec vf) (columns : P.lookups ≤ 2 ^ vf.columnVars)
    (rows : P.rows ≤ 2 ^ P.rowVars)
    (key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars)
    (s : Strategy (Option (Statement P × (ℕ → F)))) :
    prob s (game vf) (fun r => ¬ ∀ o, r = some o → Relation P o.1 o.2) ≤ runCost vf :=
  prob_not_le s (game vf) _ (runCost vf)
    (game_avoids vf wf counts matrix columns rows key_fits) (game_costLe vf wf)

end NightstreamFPrime.Spec.Compression.Verifier
