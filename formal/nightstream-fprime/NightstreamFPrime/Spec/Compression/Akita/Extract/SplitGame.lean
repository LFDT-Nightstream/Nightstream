import NightstreamFPrime.Spec.Compression.Verifier.Run
import NightstreamFPrime.Spec.Compression.Verifier.Split
import NightstreamFPrime.Spec.Compression.Akita.Extract.PackingFold
import NightstreamFPrime.Spec.Compression.Akita.Extract.Uniform

/-! The two parts `front` and `back` of the verifier game as the stages of a Fiat–Shamir
transcript (Akita, ePrint 2026/1983, revision of 2026-11-23, Definition 10.16 (vii): the bad sets
of the stages of a checkpoint segment).

- `front_staged` and `back_staged`: each draw has its bound of `[tauError P]` and of `backCosts`.
  `backCosts_sum`: the bounds add to `runCost`, the bound of one run of `game`.
- `front_bare` and `back_bare`: the verifier does not read the tables `W` and `V`. They are in
  the bad sets only.
- `led` is a game `lead` before `front`: the stages of a checkpoint of `W` that has draws before
  the lookup challenges. `led_avoids`, `led_staged` and `led_bare` are the three properties for
  it. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Verifier

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra
open NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression.Akita.Extract

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K] {P : Params F K} (vf : VerifierData P)

/-- The bounds of the draws of `back` in their order: the challenges, the rounds of the first
sum-check, the column point and the rounds of the key sum-check. -/
noncomputable def backCosts : List ℝ :=
  challengeError P :: (List.replicate P.witnessVars (((3 : ℕ) : ℝ) / Fintype.card K) ++
    ((vf.columnVars : ℝ) / Fintype.card K ::
      List.replicate (P.laneVars + P.keyBlockVars) (((2 : ℕ) : ℝ) / Fintype.card K)))

omit [Field F] [DecidableEq F] [Field K] [DecidableEq K] [Algebra F K] in
/-- The bounds of the draws of `front` and `back` add to the bound of one run of `game`. -/
theorem backCosts_sum : tauError P + (backCosts vf).sum = runCost vf := by
  simp only [backCosts, runCost, List.sum_cons, List.sum_append, List.sum_replicate]
  push_cast
  ring

omit [Field F] [Fintype F] [DecidableEq F] [DecidableEq K] [Algebra F K] in
theorem backCosts_nonnegative : ∀ cost ∈ backCosts vf, 0 ≤ cost := by
  intro cost member
  simp only [backCosts, List.mem_cons, List.mem_append, List.mem_replicate] at member
  rcases member with rfl | ⟨_, rfl⟩ | rfl | ⟨_, rfl⟩
  · exact challengeError_nonneg
  all_goals positivity

variable {Reply : Type} [Fintype Reply]

omit [Field K] [Fintype K] [DecidableEq K] [Algebra F K] in
/-- The draw of `front` has the bound `tauError`. -/
theorem front_staged (fair : ∃ decode : Reply → Fin P.lookups → F, Fair decode) (W : ℕ → F) :
    Staged Reply (front P W) [tauError P] :=
  stagedTo_draw fair (density_tauBad_le P W) fun _ => trivial

omit [Fintype F] [DecidableEq F] in
/-- Each draw of `back` has its bound of `backCosts`. `challenges`, `scalar` and `column`: an
answer of the oracle gives uniform challenges, a uniform element of `K` and a uniform column
point. -/
theorem back_staged (challenges : ∃ decode : Reply → Challenges P, Fair decode)
    (scalar : ∃ decode : Reply → K, Fair decode)
    (column : ∃ decode : Reply → Fin vf.columnVars → K, Fair decode) (wf : WellFormed P)
    (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F) :
    Staged Reply (back vf st W V τ) (backCosts vf) := by
  refine stagedTo_draw challenges (density_challengeBad_le st W V τ wf) fun ch => ?_
  have main_truth := combRound_truth P.witnessVars 3 (mainTables P st W V τ ch) mainComb
    mainComb_degree
  refine (stagedTo_bind _ _ _ _).mpr (sumcheck_staged scalar 3 P.witnessVars _ main_truth.degree
    _ _ (fun _ => trivial) ?_ P.witnessVars [] _ (by simp))
  rintro ⟨ρ, value⟩ vals
  refine stagedTo_draw column (density_columnBad_le _ _ _) fun col => ?_
  have key_truth := combRound_truth (P.laneVars + P.keyBlockVars) 2 (keyTables ch ρ) productComb
    productComb_degree
  have rounds := sumcheck_staged (Reply := Reply) scalar 2 (P.laneVars + P.keyBlockVars) _
    key_truth.degree [] (fun result rest =>
      StagedTo Reply (backKey vf st τ ch ρ value vals col result) rest fun _ _ => True)
    (fun _ => trivial) (fun final _ => trivial) (P.laneVars + P.keyBlockVars) [] vals.2.2
    (by simp)
  rw [List.append_nil] at rounds
  exact (stagedTo_bind _ _ _ _).mpr rounds

omit [Field K] [Fintype K] [DecidableEq K] [Algebra F K] in
theorem front_bare (W other : ℕ → F) : bare (front P W) = bare (front P other) :=
  bare_draw_congr fun _ => rfl

omit [Fintype F] [DecidableEq F] in
/-- The verifier of `back` is the same for each pair of tables. -/
theorem back_bare (st : Statement P) (W V W' V' : ℕ → F) (τ : Fin P.lookups → F) :
    bare (back vf st W V τ) = bare (back vf st W' V' τ) := by
  refine bare_draw_congr fun ch => bare_bind_congr (sumcheck_bare 3 _ _ _ _ _) fun first => ?_
  rcases first with _ | ⟨ρ, value⟩
  · rfl
  · exact bare_recv_congr fun vals => bare_draw_congr fun col => rfl

/-- A game `lead` and then, when it ends in a value, the draw of the lookup challenges. -/
noncomputable def led {Lead : Type} (P : Params F K) (lead : Game (Option Lead)) (W : ℕ → F) :
    Game (Option (Lead × (Fin P.lookups → F))) :=
  lead.bind fun result =>
    match result with
    | none => .pure none
    | some value => (front P W).bind fun τ => .pure (some (value, τ))

variable {Lead : Type}

omit [Fintype K] [DecidableEq K] [Fintype Reply] in
/-- `sound`: on each path of `lead` that avoids the bad sets, `W` does not satisfy `Relation` for
the statement of the value. Then on each path of `led` that avoids the bad sets no table `V` is
without error. -/
theorem led_avoids (counts : CountsNonzero P) (st : Lead → Statement P) (W : ℕ → F)
    {lead : Game (Option Lead)}
    (sound : lead.Avoids fun result => ∀ value, result = some value → ¬ Relation P (st value) W) :
    (led P lead W).Avoids fun result => ∀ value τ, result = some (value, τ) →
      ¬ (Apart P τ ∧ ∃ V, ¬ Fails P (st value) W V τ) := by
  refine (avoids_bind _ _ _).mpr (sound.mono fun result refuted => ?_)
  rcases result with _ | value
  · exact fun _ _ equal => nomatch equal
  · refine (avoids_bind _ _ _).mpr ((front_avoids counts (st value) W (refuted value rfl)).mono
      fun τ none other τ' equal => ?_)
    cases equal
    exact none

omit [Field K] [Fintype K] [DecidableEq K] [Algebra F K] in
/-- `staged`: each draw of `lead` has its bound of `costs`, and a path that ends in a value has
all the draws. Then each draw of `led` has its bound of `costs` and `tauError`. -/
theorem led_staged (fair : ∃ decode : Reply → Fin P.lookups → F, Fair decode) (W : ℕ → F)
    {lead : Game (Option Lead)} {costs : List ℝ}
    (staged : StagedTo Reply lead (costs ++ [tauError P]) fun result rest =>
      ∀ value, result = some value → rest = [tauError P]) :
    Staged Reply (led P lead W) (costs ++ [tauError P]) := by
  refine (stagedTo_bind _ _ _ _).mpr (staged.mono fun result rest all => ?_)
  rcases result with _ | value
  · trivial
  · rw [all value rfl]
    exact (stagedTo_bind _ _ _ _).mpr ((front_staged fair W).mono fun _ _ _ => trivial)

omit [Field K] [Fintype K] [DecidableEq K] [Algebra F K] [Fintype Reply] in
theorem led_bare {lead other : Game (Option Lead)} (same : bare lead = bare other)
    (W W' : ℕ → F) : bare (led P lead W) = bare (led P other W') :=
  bare_bind_congr same fun result => by
    rcases result with _ | value
    · rfl
    · exact bare_bind_congr (front_bare W W') fun _ => rfl

end NightstreamFPrime.Spec.Compression.Verifier
