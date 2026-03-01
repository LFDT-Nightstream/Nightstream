import SuperNeo.MLE
import SuperNeo.EqPoly

/-!
SumCheck protocol scaffold.

This module provides:
- protocol objects (`SumCheckInstance`, `SumCheckTranscript`),
- nontrivial acceptance semantics (round shape + fold consistency + final claim),
- explicit soundness/completeness assumption boundaries.
-/

namespace SuperNeo

structure SumCheckInstance where
  rounds : Nat
  maxDegree : Nat
  domainSize : Nat
  claimedValue : F

structure SumCheckTranscript where
  challenges : Array F
  roundPolys : Array (Array F)

/-- Evaluate a univariate polynomial (coefficient form, low degree first). -/
def sumcheckEvalPoly (poly : Array F) (x : F) : F :=
  poly.foldr (fun c acc => c + x * acc) 0

/-- Basic transcript well-formedness against an instance. -/
def sumcheckRoundConsistent
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  tr.challenges.size = inst.rounds ∧
  tr.roundPolys.size = inst.rounds

/-- Each round polynomial has the expected coefficient length. -/
def sumcheckRoundPolyShape
  (inst : SumCheckInstance)
  (poly : Array F) : Prop :=
  poly.size = inst.maxDegree + 1

/-- Every round polynomial satisfies the expected shape. -/
def sumcheckRoundShapes
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  ∀ i : Fin tr.roundPolys.size,
    sumcheckRoundPolyShape inst tr.roundPolys[i.1]

/--
Round-transition consistency:
- each next-round polynomial satisfies the `0/1` sum equation against the
  previous round challenge evaluation,
- and each round challenge evaluation is linked to the next-round constant term.
-/
def sumcheckFoldConsistent
  (tr : SumCheckTranscript) : Prop :=
  tr.challenges.size = tr.roundPolys.size ∧
  (∀ i : Nat,
    i + 1 < tr.roundPolys.size →
      sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
          sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
        sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!)) ∧
  ∀ i : Nat,
    i + 1 < tr.roundPolys.size →
      sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) =
        tr.roundPolys[i + 1]!.getD 0 0

/--
Initial round-sum consistency: the first round polynomial opened at `{0, 1}`
must sum to the claimed value.
-/
def sumcheckInitialRoundConsistent
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  if _hZero : tr.roundPolys.size = 0 then
    inst.claimedValue = 0
  else
    sumcheckEvalPoly (tr.roundPolys[0]!) 0 + sumcheckEvalPoly (tr.roundPolys[0]!) 1 =
      inst.claimedValue

/-- Final claim check against the first round polynomial's constant term. -/
def sumcheckFinalClaimConsistent
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  if _hZero : tr.roundPolys.size = 0 then
    inst.claimedValue = 0
  else
    tr.roundPolys[0]!.getD 0 0 = inst.claimedValue

/-- Verifier acceptance predicate. -/
def sumcheckAccepted
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  sumcheckRoundConsistent inst tr ∧
  sumcheckRoundShapes inst tr ∧
  sumcheckInitialRoundConsistent inst tr ∧
  sumcheckFoldConsistent tr ∧
  sumcheckFinalClaimConsistent inst tr

/-- Truth-side claim predicate (compact scaffold). -/
def sumcheckClaimTrue (inst : SumCheckInstance) : Prop :=
  inst.maxDegree ≤ inst.domainSize

/-- Soundness boundary: acceptance implies claim truth. -/
def SumcheckSoundnessAssumption : Prop :=
  ∀ inst tr,
    sumcheckAccepted inst tr →
    sumcheckClaimTrue inst

/-- Completeness boundary: true claims have an accepting transcript. -/
def SumcheckCompletenessAssumption : Prop :=
  ∀ inst,
    sumcheckClaimTrue inst →
    ∃ tr, sumcheckAccepted inst tr

/-- Structured SumCheck assumption bundle used by protocol composition. -/
structure SumCheckAssumptions where
  soundness : SumcheckSoundnessAssumption
  completeness : SumcheckCompletenessAssumption

theorem sumcheckAccepted_rounds_eq
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr) :
  tr.roundPolys.size = inst.rounds :=
  hAcc.1.2

theorem sumcheckAccepted_challenges_eq
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr) :
  tr.challenges.size = tr.roundPolys.size := by
  rcases hAcc with ⟨hRound, _hShape, _hInit, hFold, _hFinal⟩
  calc
    tr.challenges.size = inst.rounds := hRound.1
    _ = tr.roundPolys.size := hRound.2.symm

theorem sumcheckAccepted_fold_step
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) =
    tr.roundPolys[i + 1]!.getD 0 0 := by
  rcases hAcc with ⟨_hRound, _hShape, _hInit, hFold, _hFinal⟩
  exact hFold.2.2 i hi

theorem sumcheckAccepted_initial_round
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr) :
  sumcheckInitialRoundConsistent inst tr := by
  rcases hAcc with ⟨_hRound, _hShape, hInit, _hFold, _hFinal⟩
  exact hInit

theorem sumcheckAccepted_round_sum_step
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
      sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
    sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) := by
  rcases hAcc with ⟨_hRound, _hShape, _hInit, hFold, _hFinal⟩
  exact hFold.2.1 i hi

theorem sumcheckAccepted_not_of_challenge_size_ne
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hNe : tr.challenges.size ≠ inst.rounds) :
  ¬ sumcheckAccepted inst tr := by
  intro hAcc
  exact hNe hAcc.1.1

theorem sumcheckAccepted_not_of_roundpoly_size_ne
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hNe : tr.roundPolys.size ≠ inst.rounds) :
  ¬ sumcheckAccepted inst tr := by
  intro hAcc
  exact hNe hAcc.1.2

theorem sumcheckAccepted_not_of_bad_round_shape
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hBad : ∃ i : Fin tr.roundPolys.size,
    ¬ sumcheckRoundPolyShape inst tr.roundPolys[i.1]) :
  ¬ sumcheckAccepted inst tr := by
  intro hAcc
  rcases hAcc with ⟨_hRound, hShapes, _hInit, _hFold, _hFinal⟩
  rcases hBad with ⟨i, hiBad⟩
  exact hiBad (hShapes i)

theorem sumcheckAccepted_not_of_bad_final_claim
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hBad : ¬ sumcheckFinalClaimConsistent inst tr) :
  ¬ sumcheckAccepted inst tr := by
  intro hAcc
  rcases hAcc with ⟨_hRound, _hShapes, _hInit, _hFold, hFinal⟩
  exact hBad hFinal

theorem sumcheckAccepted_not_of_bad_initial_round
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hBad : ¬ sumcheckInitialRoundConsistent inst tr) :
  ¬ sumcheckAccepted inst tr := by
  intro hAcc
  rcases hAcc with ⟨_hRound, _hShapes, hInit, _hFold, _hFinal⟩
  exact hBad hInit

theorem sumcheckClaimTrue_of_soundness
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hSound : SumcheckSoundnessAssumption)
  (hAcc : sumcheckAccepted inst tr) :
  sumcheckClaimTrue inst := by
  exact hSound inst tr hAcc

/-- Re-export under paper-facing names. -/
abbrev SumCheckAccepted := sumcheckAccepted
abbrev SumCheckClaimTrue := sumcheckClaimTrue
abbrev SumCheckRoundConsistent := sumcheckRoundConsistent

end SuperNeo
