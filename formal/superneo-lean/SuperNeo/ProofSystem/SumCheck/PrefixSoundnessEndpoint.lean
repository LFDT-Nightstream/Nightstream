import SuperNeo.ProofSystem.SumCheck.PrefixSoundness

namespace SuperNeo.ProofSystem

namespace Sumcheck

private theorem exists_true_false_transition
  (P : Nat → Prop)
  {n : Nat}
  (h0 : P 0)
  (hn : ¬ P n) :
  ∃ i, i < n ∧ P i ∧ ¬ P (i + 1) := by
  induction n generalizing P with
  | zero =>
      exact False.elim (hn h0)
  | succ n ih =>
      by_cases h1 : P 1
      · rcases ih (P := fun k => P (k + 1)) h1 hn with ⟨i, hi, hPi, hNext⟩
        exact ⟨i + 1, by omega, hPi, hNext⟩
      · exact ⟨0, Nat.succ_pos _, h0, h1⟩

private theorem extract_succ_eq_push
  (coins : Array F)
  {i : Nat}
  (hi : i < coins.size) :
  coins.extract 0 (i + 1) = (coins.extract 0 i).push (coins[i]!) := by
  apply Array.ext
  · have hLeft : (coins.extract 0 (i + 1)).size = i + 1 := by
      simp [Array.size_extract, hi]
    have hRight : ((coins.extract 0 i).push (coins[i]!)).size = i + 1 := by
      simp [Array.size_extract, hi.le]
    exact hLeft.trans hRight.symm
  · intro j hjL hjR
    have hjInfo : j ≤ i ∧ j < coins.size := by
      simpa [Array.size_extract] using hjL
    have hjLt : j < i + 1 := by omega
    by_cases hj : j < i
    · simp [Array.getElem_extract, Array.getElem_push_lt, hi.le, hj]
    · have hEq : j = i := by omega
      subst j
      have hSizeExtract : (coins.extract 0 i).size = i := by
        simp [Array.size_extract, hi.le]
      have hRight : ((coins.extract 0 i).push (coins[i]!))[i] = coins[i]! := by
        simpa [hSizeExtract] using Array.getElem_push_eq (xs := coins.extract 0 i) (x := coins[i]!)
      have hCoin : coins[i] = coins[i]! := by
        symm
        exact getElem!_pos (c := coins) (i := i) hi
      rw [Array.getElem_extract]
      · calc
          coins[0 + i] = coins[i] := by simp
          _ = coins[i]! := hCoin
          _ = ((coins.extract 0 i).push (coins[i]!))[i] := hRight.symm

noncomputable def SoundnessGame.prefixGapRootSet
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F) : Finset F := by
  classical
  exact
    if hReady : g.prefixRoundConsistent i pre ∧ g.falseClaimAtPrefix i pre then
      Finset.univ.filter (fun r : F =>
        (sumcheckPolynomialZMod (g.prefixGapPoly i pre)).eval (fToZMod r) = 0)
    else
      ∅

noncomputable def SoundnessGame.prefixGapEvent
  (g : SoundnessGame)
  (i : Nat) : Array F → Prop :=
  fun coins => coins[i]! ∈ g.prefixGapRootSet i (coins.extract 0 i)

theorem SoundnessGame.prefixGapRootSet_card_le
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F)
  (hi : i < g.inst.rounds)
  (hPre : pre ∈ fullFieldCoinSpace i)
  (hDegPos : 0 < g.inst.maxDegree) :
  (g.prefixGapRootSet i pre).card ≤ g.inst.maxDegree := by
  classical
  by_cases hReady : g.prefixRoundConsistent i pre ∧ g.falseClaimAtPrefix i pre
  · have hSize : pre.size = i := mem_fullFieldCoinSpace_size hPre
    have hNZ : sumcheckPolynomialZMod (g.prefixGapPoly i pre) ≠ 0 :=
      g.prefixGapPoly_nonzero_of_falseClaim i pre hSize hi hReady.1 hReady.2 hDegPos
    simpa [SoundnessGame.prefixGapRootSet, hReady, fullFieldPolyRootCount] using
      (fullFieldPolyRootCount_le_maxDegree_of_shape_nonzero
        (poly := g.prefixGapPoly i pre)
        (maxDegree := g.inst.maxDegree)
        (hShape := g.prefixGapPoly_size i pre)
        hNZ)
  · simp [SoundnessGame.prefixGapRootSet, hReady]

theorem SoundnessGame.mem_prefixGapRootSet_of_transition
  (g : SoundnessGame)
  (i : Nat)
  (pre : Array F)
  (r : F)
  (hSize : pre.size = i)
  (hi : i < g.inst.rounds)
  (hRound : g.prefixRoundConsistent i pre)
  (hFalse : g.falseClaimAtPrefix i pre)
  (hRepair : ¬ g.falseClaimAtPrefix (i + 1) (pre.push r))
  (hDegPos : 0 < g.inst.maxDegree) :
  r ∈ g.prefixGapRootSet i pre := by
  classical
  have hReady : g.prefixRoundConsistent i pre ∧ g.falseClaimAtPrefix i pre :=
    ⟨hRound, hFalse⟩
  have hEvalZero :
      SuperNeo.sumcheckEvalPoly (g.prefixGapPoly i pre) r = 0 :=
    g.prefixGapPoly_eval_zero_of_repair i pre r hSize hi hRound hRepair hDegPos
  have hRoot :
      (sumcheckPolynomialZMod (g.prefixGapPoly i pre)).eval (fToZMod r) = 0 := by
    calc
      (sumcheckPolynomialZMod (g.prefixGapPoly i pre)).eval (fToZMod r)
          = fToZMod (SuperNeo.sumcheckEvalPoly (g.prefixGapPoly i pre) r) := by
              symm
              exact sumcheckEvalPoly_fToZMod (g.prefixGapPoly i pre) r
      _ = (0 : Fq) := by
            simp [hEvalZero, fToZMod]
  unfold SoundnessGame.prefixGapRootSet
  simp [hReady, hRoot]

theorem SoundnessGame.failureEvent_covered_by_prefixGapEvent
  (g : SoundnessGame)
  (coins : Array F)
  (hFail : g.failureEvent coins)
  (hRoundsPos : 0 < g.inst.rounds)
  (hDegPos : 0 < g.inst.maxDegree) :
  roundFailureUnionCoins g.prefixGapEvent g.inst.rounds coins := by
  have hAcc : g.acceptsOn coins := hFail
  have hStart : g.falseClaimAtPrefix 0 (coins.extract 0 0) := by
    exact g.falseClaimAtPrefix_zero (coins.extract 0 0)
  have hEnd : ¬ g.falseClaimAtPrefix g.inst.rounds (coins.extract 0 g.inst.rounds) :=
    g.not_falseClaimAtPrefix_rounds_of_acceptsOn coins hAcc hRoundsPos
  rcases exists_true_false_transition
      (P := fun k => g.falseClaimAtPrefix k (coins.extract 0 k)) hStart hEnd with
    ⟨i, hi, hFalse, hRepairExtract⟩
  have hCoinsSize : coins.size = g.inst.rounds := by
    have hCore : SuperNeo.sumcheckAcceptedCore g.inst (g.transcript coins) := hAcc.1
    have hRoundCons : SuperNeo.sumcheckRoundConsistent g.inst (g.transcript coins) := hCore.2.2.1
    simpa [SoundnessGame.transcript] using hRoundCons.1
  have hPreSize : (coins.extract 0 i).size = i := by
    simp [Array.size_extract, hi.le, hCoinsSize]
  have hRound : g.prefixRoundConsistent i (coins.extract 0 i) :=
    g.prefixRoundConsistent_of_acceptsOn coins hAcc hi hRoundsPos
  have hPushEq :
      coins.extract 0 (i + 1) = (coins.extract 0 i).push (coins[i]!) := by
    exact extract_succ_eq_push coins (by simpa [hCoinsSize] using hi)
  have hRepair :
      ¬ g.falseClaimAtPrefix (i + 1) ((coins.extract 0 i).push (coins[i]!)) := by
    simpa [hPushEq] using hRepairExtract
  have hMem : coins[i]! ∈ g.prefixGapRootSet i (coins.extract 0 i) :=
    g.mem_prefixGapRootSet_of_transition i (coins.extract 0 i) (coins[i]!)
      hPreSize hi hRound hFalse hRepair hDegPos
  exact roundFailureUnionCoins_of_mem hi hMem

theorem SoundnessGame.prefixGapEvent_probBoundScaled
  (g : SoundnessGame)
  (i : Nat)
  (hi : i < g.inst.rounds)
  (hAligned : SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hDegPos : 0 < g.inst.maxDegree) :
  (fullFieldUniformCoinProbModel g.inst.rounds).Pr (g.prefixGapEvent i) *
      (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
    (g.inst.maxDegree : Rat) := by
  have hCountPow :
      fullFieldCoinEventCount g.inst.rounds (g.prefixGapEvent i) ≤
        g.inst.maxDegree * Goldilocks.q ^ (g.inst.rounds - 1) := by
    simpa [SoundnessGame.prefixGapEvent] using
      (fullFieldCoinEventCount_prefixRootSet_le
        (m := g.inst.rounds)
        (i := i)
        (d := g.inst.maxDegree)
        hi
        (rootSet := fun pre => g.prefixGapRootSet i pre)
        (by
          intro a ha
          exact g.prefixGapRootSet_card_le i a hi ha hDegPos))
  have hScaled :
      fullFieldCoinEventCount g.inst.rounds (g.prefixGapEvent i) *
          (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
        g.inst.maxDegree * (fullFieldCoinSpace g.inst.rounds).length := by
    have hMulQ :
        fullFieldCoinEventCount g.inst.rounds (g.prefixGapEvent i) * Goldilocks.q ≤
          (g.inst.maxDegree * Goldilocks.q ^ (g.inst.rounds - 1)) * Goldilocks.q := by
      exact Nat.mul_le_mul_right Goldilocks.q hCountPow
    have hPowStep :
        Goldilocks.q ^ g.inst.rounds = Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q := by
      have hIdx : (g.inst.rounds - 1) + 1 = g.inst.rounds := by omega
      calc
        Goldilocks.q ^ g.inst.rounds = Goldilocks.q ^ ((g.inst.rounds - 1) + 1) := by simp [hIdx]
        _ = Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q := by
              simp [Nat.pow_succ, Nat.mul_comm]
    calc
      fullFieldCoinEventCount g.inst.rounds (g.prefixGapEvent i) *
          (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
          = fullFieldCoinEventCount g.inst.rounds (g.prefixGapEvent i) * Goldilocks.q := by
              simp [hAligned]
      _ ≤ (g.inst.maxDegree * Goldilocks.q ^ (g.inst.rounds - 1)) * Goldilocks.q := hMulQ
      _ = g.inst.maxDegree * Goldilocks.q ^ g.inst.rounds := by
            rw [hPowStep]
            simp [Nat.mul_assoc]
      _ = g.inst.maxDegree * (fullFieldCoinSpace g.inst.rounds).length := by
            simp [fullFieldCoinSpace_length]
  simpa [fullFieldUniformCoinProbModel] using
    (fullFieldCoinPr_mul_nat_le_of_countScaled
      g.inst.rounds (g.prefixGapEvent i)
      (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
      g.inst.maxDegree hScaled)

noncomputable def SoundnessGame.prefixGapSchwartzZippelLemmas
  (g : SoundnessGame)
  (hAligned : SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hRoundsPos : 0 < g.inst.rounds)
  (hDegPos : 0 < g.inst.maxDegree) :
  SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g := by
  refine
    { roundFailure := g.prefixGapEvent
      covered := ?_
      roundRootBudget := fun _ => g.inst.maxDegree
      roundRootBudgetBound := ?_
      roundProbBoundScaled := ?_ }
  · intro coins hFail
    exact g.failureEvent_covered_by_prefixGapEvent coins hFail hRoundsPos hDegPos
  · intro i hi
    exact le_rfl
  · intro i hi
    exact g.prefixGapEvent_probBoundScaled i hi hAligned hDegPos

theorem SoundnessGame.lundBoundHolds_of_prefixGapSchwartzZippel
  (g : SoundnessGame)
  (hAligned : SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hRoundsPos : 0 < g.inst.rounds)
  (hDegPos : 0 < g.inst.maxDegree) :
  g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds) := by
  let prob := fullFieldUniformCoinProbModel g.inst.rounds
  let hSz := g.prefixGapSchwartzZippelLemmas hAligned hRoundsPos hDegPos
  let hKernel : LundRoundKernel prob g :=
    LundRoundKernel.of_schwartzZippelRoundEventLemmas prob g hSz
  let hScaled : LundRoundBoundaryScaled prob g :=
    LundRoundBoundaryScaled.of_kernel prob g hKernel
  exact SoundnessGame.lundBoundHolds_of_scaledRoundBoundary prob g hScaled

theorem lundSoundnessAssumptionFullFieldAlignedPosRoundsPosDegree_prefix
  (g : SoundnessGame)
  (hAligned : SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hRoundsPos : 0 < g.inst.rounds)
  (hDegPos : 0 < g.inst.maxDegree) :
  g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds) := by
  exact g.lundBoundHolds_of_prefixGapSchwartzZippel hAligned hRoundsPos hDegPos

private theorem SoundnessGame.no_failureEvent_of_roundsPos_degreeZero
  (g : SoundnessGame)
  (hRoundsPos : 0 < g.inst.rounds)
  (hDegZero : g.inst.maxDegree = 0) :
  ∀ coins, ¬ g.failureEvent coins := by
  intro coins hFail
  have hCompat :
      SuperNeo.sumcheckDegreeCompatible g.inst :=
    SuperNeo.sumcheckAccepted_degree_compatible hFail.1
  have hDegPos :
      0 < g.inst.maxDegree :=
    SuperNeo.sumcheckDegreeCompatible.maxDegree_pos_of_rounds_pos hCompat hRoundsPos
  omega

private theorem SoundnessGame.lundBoundHolds_of_roundsPos_degreeZero
  (g : SoundnessGame)
  (hRoundsPos : 0 < g.inst.rounds)
  (hDegZero : g.inst.maxDegree = 0) :
  g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds) := by
  unfold SoundnessGame.lundBoundHolds SoundnessGame.advantage
  have hPrLeZero :
      (fullFieldUniformCoinProbModel g.inst.rounds).Pr g.failureEvent ≤ 0 := by
    calc
      (fullFieldUniformCoinProbModel g.inst.rounds).Pr g.failureEvent
          ≤ (fullFieldUniformCoinProbModel g.inst.rounds).Pr (fun _ => False) :=
            (fullFieldUniformCoinProbModel g.inst.rounds).prMonotone
              (g.no_failureEvent_of_roundsPos_degreeZero hRoundsPos hDegZero)
      _ = 0 := (fullFieldUniformCoinProbModel g.inst.rounds).prFalse
  have hPrZero :
      (fullFieldUniformCoinProbModel g.inst.rounds).Pr g.failureEvent = 0 :=
    le_antisymm hPrLeZero
      ((fullFieldUniformCoinProbModel g.inst.rounds).prNonneg g.failureEvent)
  calc
    (fullFieldUniformCoinProbModel g.inst.rounds).Pr g.failureEvent *
        SuperNeo.sumcheckLundSoundnessDenominator g.inst
        = 0 := by simp [hPrZero]
    _ ≤ SuperNeo.sumcheckLundSoundnessNumerator g.inst := by
          simp [SuperNeo.sumcheckLundSoundnessNumerator, hDegZero]

private theorem SoundnessGame.no_failureEvent_of_roundsZero
  (g : SoundnessGame)
  (hRoundsZero : g.inst.rounds = 0) :
  ∀ coins, ¬ g.failureEvent coins := by
  intro coins hFail
  have hAcc :
      SuperNeo.sumcheckAcceptedForTable g.inst g.table (g.transcript coins) := by
    simpa [SoundnessGame.failureEvent, SoundnessGame.acceptsOn] using hFail
  have hTableSize : g.table.size = 1 := by
    simpa [hRoundsZero] using g.tableSize
  have hTableNe : g.table.size ≠ 0 := by
    simp [hTableSize]
  have hFinalClaim : mleByFolding g.table #[] = g.inst.claimedValue := by
    simpa [hRoundsZero] using hAcc.2.2.2
  have hTableSumEqClaimed : SuperNeo.sumcheckTableSum g.table = g.inst.claimedValue := by
    calc
      SuperNeo.sumcheckTableSum g.table = g.table[0]! := by
        rw [sumcheckTableSum_eq_arraySum]
        simp [arraySum, hTableSize]
      _ = mleByFolding g.table #[] := by
          symm
          exact mleByFolding_empty g.table hTableNe
      _ = g.inst.claimedValue := hFinalClaim
  exact g.falseClaim hTableSumEqClaimed

private theorem SoundnessGame.lundBoundHolds_of_roundsZero
  (g : SoundnessGame)
  (hRoundsZero : g.inst.rounds = 0) :
  g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds) := by
  unfold SoundnessGame.lundBoundHolds SoundnessGame.advantage
  have hPrLeZero :
      (fullFieldUniformCoinProbModel g.inst.rounds).Pr g.failureEvent ≤ 0 := by
    calc
      (fullFieldUniformCoinProbModel g.inst.rounds).Pr g.failureEvent
          ≤ (fullFieldUniformCoinProbModel g.inst.rounds).Pr (fun _ => False) :=
            (fullFieldUniformCoinProbModel g.inst.rounds).prMonotone
              (g.no_failureEvent_of_roundsZero hRoundsZero)
      _ = 0 := (fullFieldUniformCoinProbModel g.inst.rounds).prFalse
  have hPrZero :
      (fullFieldUniformCoinProbModel g.inst.rounds).Pr g.failureEvent = 0 :=
    le_antisymm hPrLeZero
      ((fullFieldUniformCoinProbModel g.inst.rounds).prNonneg g.failureEvent)
  calc
    (fullFieldUniformCoinProbModel g.inst.rounds).Pr g.failureEvent *
        SuperNeo.sumcheckLundSoundnessDenominator g.inst
        = 0 := by simp [hPrZero]
    _ ≤ SuperNeo.sumcheckLundSoundnessNumerator g.inst := by
          simp [SuperNeo.sumcheckLundSoundnessNumerator, hRoundsZero]

theorem lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix
  (g : SoundnessGame)
  (hAligned : SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hRoundsPos : 0 < g.inst.rounds) :
  g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds) := by
  by_cases hDegPos : 0 < g.inst.maxDegree
  · exact lundSoundnessAssumptionFullFieldAlignedPosRoundsPosDegree_prefix
      g hAligned hRoundsPos hDegPos
  · have hDegZero : g.inst.maxDegree = 0 := by
      omega
    exact g.lundBoundHolds_of_roundsPos_degreeZero hRoundsPos hDegZero

theorem lundSoundnessAssumptionFullFieldAligned_prefix
  (g : SoundnessGame)
  (hAligned : SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q) :
  g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds) := by
  by_cases hRoundsZero : g.inst.rounds = 0
  · exact g.lundBoundHolds_of_roundsZero hRoundsZero
  · exact lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix
      g hAligned (Nat.pos_iff_ne_zero.mpr hRoundsZero)

end Sumcheck

end SuperNeo.ProofSystem
