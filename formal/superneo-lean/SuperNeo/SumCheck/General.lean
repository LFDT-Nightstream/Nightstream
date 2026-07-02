import SuperNeo.SumCheck.RootCountingAligned

/-!
Paper-facing SumCheck endpoint: round-by-round soundness boundaries,
error boundaries, and the constructive theorem package. Importing this
module exposes the full structured SumCheck API.
-/

namespace SuperNeo.ProofSystem

namespace Sumcheck

/--
Round-by-round soundness boundary:
for each round, a failure event with an explicit probability bound, and
coverage from global soundness failure to the union of round failures.
-/
structure RoundByRoundSoundnessBoundary
  (prob : ProbModel)
  (inst : Instance)
  (tr : Transcript) where
  roundFailure : Nat → Prop
  epsRound : Nat → Rat
  roundFailureBound :
    ∀ i : Nat, i < inst.rounds → prob.Pr (roundFailure i) ≤ epsRound i
  soundnessFailureCovered :
    SoundnessFailureEvent inst tr → roundFailureUnion roundFailure inst.rounds

/-- Aggregate round-by-round soundness error bound up to `inst.rounds`. -/
def RoundByRoundSoundnessBoundary.totalRoundError
  {prob : ProbModel}
  {inst : Instance}
  {tr : Transcript}
  (hRbr : RoundByRoundSoundnessBoundary prob inst tr) : Rat :=
  roundErrorSum hRbr.epsRound inst.rounds

theorem soundnessFailureEvent_not
  (hSound : SoundnessAssumption)
  {inst : Instance}
  {tr : Transcript} :
  ¬ SoundnessFailureEvent inst tr := by
  intro hFail
  exact hFail.2 (hSound inst tr hFail.1)

theorem soundnessFailureAdvantage_eq_zero_of_soundness
  (prob : ProbModel)
  (hSound : SoundnessAssumption)
  {inst : Instance}
  {tr : Transcript} :
  SoundnessFailureAdvantage prob inst tr = 0 := by
  unfold SoundnessFailureAdvantage
  have hEventFalse : SoundnessFailureEvent inst tr → False := by
    exact soundnessFailureEvent_not hSound
  have hLeZero : prob.Pr (SoundnessFailureEvent inst tr) ≤ 0 := by
    calc
      prob.Pr (SoundnessFailureEvent inst tr) ≤ prob.Pr False := prob.prMonotone hEventFalse
      _ = 0 := prob.prFalse
  exact Rat.le_antisymm hLeZero (prob.prNonneg _)

/--
If soundness holds and `eps` is pointwise nonnegative, the soundness-failure
advantage is bounded by `eps`.
-/
theorem soundnessFailureAdvantageBound_of_soundness
  (hSound : SoundnessAssumption)
  {inst : Instance}
  {tr : Transcript}
  {eps : ErrorFn}
  (hEpsNonneg : ∀ n : Nat, 0 ≤ eps n) :
  SoundnessFailureAdvantageBound inst tr eps := by
  intro prob n
  have hZero :
      SoundnessFailureAdvantage prob inst tr = 0 :=
    soundnessFailureAdvantage_eq_zero_of_soundness prob hSound
  have hLeZero : SoundnessFailureAdvantage prob inst tr ≤ 0 := by
    simpa [hZero] using (show (0 : Rat) ≤ 0 by decide)
  exact Rat.le_trans hLeZero (hEpsNonneg n)

theorem pr_roundFailureUnion_le_roundErrorSum
  (prob : ProbModel)
  (E : Nat → Prop)
  (eps : Nat → Rat)
  (n : Nat)
  (hBound : ∀ i : Nat, i < n → prob.Pr (E i) ≤ eps i) :
  prob.Pr (roundFailureUnion E n) ≤ roundErrorSum eps n := by
  induction n with
  | zero =>
      simpa [roundFailureUnion, roundErrorSum, prob.prFalse] using (Rat.le_refl : (0 : Rat) ≤ 0)
  | succ n ih =>
      have hBoundPrev : ∀ i : Nat, i < n → prob.Pr (E i) ≤ eps i := by
        intro i hi
        exact hBound i (Nat.lt_trans hi (Nat.lt_succ_self n))
      have hBoundN : prob.Pr (E n) ≤ eps n := hBound n (Nat.lt_succ_self n)
      have hAddPrev :
          prob.Pr (roundFailureUnion E n) + prob.Pr (E n) ≤
            roundErrorSum eps n + prob.Pr (E n) := by
        exact (Rat.add_le_add_right (c := prob.Pr (E n))).2 (ih hBoundPrev)
      have hAddLast :
          roundErrorSum eps n + prob.Pr (E n) ≤
            roundErrorSum eps n + eps n := by
        exact (Rat.add_le_add_left (c := roundErrorSum eps n)).2 hBoundN
      calc
        prob.Pr (roundFailureUnion E (n + 1))
            = prob.Pr (roundFailureUnion E n ∨ E n) := by
                simp [roundFailureUnion]
        _ ≤ prob.Pr (roundFailureUnion E n) + prob.Pr (E n) := prob.prUnionLeAdd _ _
        _ ≤ roundErrorSum eps n + prob.Pr (E n) := hAddPrev
        _ ≤ roundErrorSum eps n + eps n := hAddLast
        _ = roundErrorSum eps (n + 1) := by
              simp [roundErrorSum]

theorem RoundByRoundSoundnessBoundary.soundnessFailureAdvantage_le_totalRoundError
  {prob : ProbModel}
  {inst : Instance}
  {tr : Transcript}
  (hRbr : RoundByRoundSoundnessBoundary prob inst tr) :
  SoundnessFailureAdvantage prob inst tr ≤ hRbr.totalRoundError := by
  unfold SoundnessFailureAdvantage RoundByRoundSoundnessBoundary.totalRoundError
  have hCover :
      prob.Pr (SoundnessFailureEvent inst tr) ≤
        prob.Pr (roundFailureUnion hRbr.roundFailure inst.rounds) := by
    exact prob.prMonotone hRbr.soundnessFailureCovered
  exact Rat.le_trans hCover
    (pr_roundFailureUnion_le_roundErrorSum
      prob hRbr.roundFailure hRbr.epsRound inst.rounds hRbr.roundFailureBound)

/--
Convert a concrete round-by-round bound into the theorem-facing advantage-bound
contract, for a fixed probability model.
-/
theorem RoundByRoundSoundnessBoundary.soundnessFailureAdvantageBound
  {prob : ProbModel}
  {inst : Instance}
  {tr : Transcript}
  {eps : ErrorFn}
  (hRbr : RoundByRoundSoundnessBoundary prob inst tr)
  (hTotalLe : ∀ n : Nat, hRbr.totalRoundError ≤ eps n) :
  ∀ n : Nat, SoundnessFailureAdvantage prob inst tr ≤ eps n := by
  intro n
  exact Rat.le_trans
    (hRbr.soundnessFailureAdvantage_le_totalRoundError)
    (hTotalLe n)

/-- Explicit soundness-error boundary surface for SumCheck. -/
structure SoundnessErrorBoundary where
  epsSoundness : ErrorFn
  nonnegEpsSoundness : ∀ n : Nat, 0 ≤ epsSoundness n
  negligibleEpsSoundness : IsNegligible epsSoundness

/--
Boundary-complete SumCheck theorem package:
- soundness/completeness are carried as typed parameters,
- soundness error surface is carried explicitly as a theorem-facing boundary.
-/
structure TheoremPackage
  (soundness : SoundnessAssumption)
  (completeness : CompletenessAssumption) where
  soundnessError : SoundnessErrorBoundary

/-- Project SumCheck soundness error function from theorem package. -/
def TheoremPackage.eps
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (hPkg : TheoremPackage soundness completeness) : ErrorFn :=
  hPkg.soundnessError.epsSoundness

/-- Project nonnegativity of the soundness-error function from theorem package. -/
theorem TheoremPackage.nonneg
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (hPkg : TheoremPackage soundness completeness) :
  ∀ n : Nat, 0 ≤ hPkg.eps n := by
  exact hPkg.soundnessError.nonnegEpsSoundness

/-- Project negligible soundness-error boundary from theorem package. -/
theorem TheoremPackage.negligible
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (hPkg : TheoremPackage soundness completeness) :
  IsNegligible hPkg.eps := by
  exact hPkg.soundnessError.negligibleEpsSoundness

/-- Soundness projection from theorem package plus acceptance witness. -/
theorem TheoremPackage.soundness
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (_hPkg : TheoremPackage soundness completeness)
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  ClaimTrue inst := by
  exact soundness inst tr hAccepted

/-- Completeness projection from theorem package plus claim-truth witness. -/
theorem TheoremPackage.completeness
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (_hPkg : TheoremPackage soundness completeness)
  {inst : Instance}
  (hClaim : ClaimTrue inst) :
  ∃ tr, Accepted inst tr := by
  exact completeness inst hClaim

theorem TheoremPackage.soundnessFailureAdvantage_eq_zero
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (hPkg : TheoremPackage soundness completeness)
  (prob : ProbModel)
  {inst : Instance}
  {tr : Transcript} :
  SoundnessFailureAdvantage prob inst tr = 0 := by
  apply soundnessFailureAdvantage_eq_zero_of_soundness (prob := prob)
  intro inst tr hAccepted
  exact hPkg.soundness hAccepted

/--
Theorem-package soundness implies a full theorem-facing soundness-failure
advantage bound against the package error function.
-/
theorem TheoremPackage.soundnessFailureAdvantageBound
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (hPkg : TheoremPackage soundness completeness)
  {inst : Instance}
  {tr : Transcript} :
  SoundnessFailureAdvantageBound inst tr hPkg.eps := by
  intro prob n
  have hZero :
      SoundnessFailureAdvantage prob inst tr = 0 :=
    hPkg.soundnessFailureAdvantage_eq_zero prob
  have hLeZero : SoundnessFailureAdvantage prob inst tr ≤ 0 := by
    simpa [hZero] using (show (0 : Rat) ≤ 0 by decide)
  exact Rat.le_trans hLeZero (hPkg.nonneg n)

theorem accepted_rounds_eq
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  tr.roundPolys.size = inst.rounds := by
  exact SingleRound.accepted_rounds_eq hAccepted

theorem accepted_challenges_eq
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  tr.challenges.size = tr.roundPolys.size := by
  exact SingleRound.accepted_challenges_eq hAccepted

theorem accepted_fold_step
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
      SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
    SuperNeo.sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) := by
  exact SingleRound.accepted_fold_step hAccepted hi

theorem accepted_initial_round
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  InitialRoundConsistent inst tr := by
  exact SingleRound.accepted_initial_round hAccepted

theorem accepted_round_sum_step
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
      SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
    SuperNeo.sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) := by
  exact SingleRound.accepted_round_sum_step hAccepted hi

/-- Soundness theorem surface (assumption-instantiated). -/
theorem soundness
  (h : SoundnessAssumption)
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  ClaimTrue inst := by
  exact h inst tr hAccepted

/-- Completeness theorem surface (assumption-instantiated). -/
theorem completeness
  (h : CompletenessAssumption)
  {inst : Instance}
  (hClaim : ClaimTrue inst) :
  ∃ tr, Accepted inst tr := by
  exact h inst hClaim

/--
Canonical constructor for the constructive SumCheck closure path from
`SuperNeo.SumCheck`.
-/
def theoremPackage_constructive
  (soundnessError : SoundnessErrorBoundary) :
  TheoremPackage
    SuperNeo.sumcheckSoundness_constructive
    SuperNeo.sumcheckCompleteness_constructive where
  soundnessError := soundnessError

/--
Canonical zero-error soundness boundary for the constructive SumCheck closure
path.
-/
def soundnessErrorBoundary_zero : SoundnessErrorBoundary where
  epsSoundness := fun _ => 0
  nonnegEpsSoundness := by
    intro n
    exact (show (0 : Rat) ≤ 0 by decide)
  negligibleEpsSoundness := by
    simpa using (isNegligible_zero : IsNegligible (fun _ => (0 : Rat)))

/-- Canonical constructive theorem package with zero soundness error. -/
def theoremPackage_constructive_zeroError :
  TheoremPackage
    SuperNeo.sumcheckSoundness_constructive
    SuperNeo.sumcheckCompleteness_constructive :=
  theoremPackage_constructive soundnessErrorBoundary_zero

end Sumcheck

end SuperNeo.ProofSystem
