import SuperNeo.ProofSystem.SumCheck.PrefixSoundnessEndpoint

open SuperNeo
open SuperNeo.ProofSystem
open SuperNeo.ProofSystem.Sumcheck

namespace tests

private def f (n : Nat) : F :=
  F.ofNat n

/-
Concrete one-round false-claim game:
- table is identically zero
- claimed sum is `1`
- prover sends `p(X) = X`

On verifier coin `0`, the run is accepted against a false claim.
On verifier coin `1`, the endpoint check fails.
-/
def smokeInst : SumCheckInstance :=
  { rounds := 1
    maxDegree := 1
    domainSize := Goldilocks.q
    claimedValue := f 1 }

def smokeTable : Array F := #[0, 0]

def smokeProver : OnlineProverStrategy smokeInst :=
  { roundPoly := fun _ _ => #[0, f 1]
    roundPolyShape := by
      intro i hi _coins
      have hi' : i < 1 := by simpa [smokeInst] using hi
      have hi0 : i = 0 := by omega
      subst hi0
      native_decide
    nonanticipatory := by
      intro i hi _coins1 _coins2 _hPrefix
      have hi' : i < 1 := by simpa [smokeInst] using hi
      have hi0 : i = 0 := by omega
      subst hi0
      rfl
  }

def smokeGame : SoundnessGame :=
  { inst := smokeInst
    table := smokeTable
    tableSize := by rfl
    falseClaim := by
      native_decide
    prover := smokeProver }

def smokeCoins0 : Array F := #[0]

def smokeCoins1 : Array F := #[f 1]

def smokePoly : Array F := #[0, f 1]

def smokeTranscript0 : SumCheckTranscript :=
  { challenges := smokeCoins0
    roundPolys := #[smokePoly] }

def smokeTranscript1 : SumCheckTranscript :=
  { challenges := smokeCoins1
    roundPolys := #[smokePoly] }

theorem smokeAccepted0 : sumcheckAccepted smokeInst smokeTranscript0 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · change smokeInst.maxDegree ≤ smokeInst.domainSize
    native_decide
  · change smokeInst.rounds = 0 ∨ 0 < smokeInst.maxDegree
    native_decide
  · change smokeTranscript0.challenges.size = smokeInst.rounds ∧
      smokeTranscript0.roundPolys.size = smokeInst.rounds
    simp [smokeTranscript0, smokeInst, smokeCoins0]
  · intro i
    fin_cases i
    simp [sumcheckRoundPolyShape, smokeTranscript0, smokePoly, smokeInst]
  · simp [sumcheckInitialRoundConsistent, smokeTranscript0, smokeInst, smokePoly, sumcheckEvalPoly]
  · constructor
    · native_decide
    · intro i hi
      have hi' : i + 1 < 1 := by simp [smokeTranscript0] at hi
      omega

theorem smokeAcceptsOn0 : smokeGame.acceptsOn smokeCoins0 := by
  change sumcheckAcceptedForTable smokeInst smokeTable (smokeGame.transcript smokeCoins0)
  refine ⟨?_, ?_⟩
  · simpa [smokeTranscript0, smokeGame, SoundnessGame.transcript, smokeCoins0, smokePoly] using
      smokeAccepted0
  constructor
  · rfl
  · refine ⟨smokeAccepted0.2.2.1, ?_⟩
    have hEval :
        sumcheckEvalPoly (smokeGame.prover.roundPoly 0 smokeCoins0) 0 =
          mleByFolding smokeTable smokeCoins0 := by
        native_decide
    simpa [sumcheckFinalOracleConsistentWithTable, smokeGame, SoundnessGame.transcript,
      smokeInst, smokeTable, smokeCoins0, smokePoly] using hEval

theorem smokeFailureEvent0 : smokeGame.failureEvent smokeCoins0 := by
  exact smokeAcceptsOn0

example : sumcheckAccepted smokeInst smokeTranscript0 := smokeAccepted0

example : smokeGame.acceptsOn smokeCoins0 := smokeAcceptsOn0

example : smokeGame.failureEvent smokeCoins0 := smokeFailureEvent0

example : ¬ smokeGame.acceptsOn smokeCoins1 := by
  intro hAcc
  have hFinal :
      sumcheckFinalOracleConsistentWithTable smokeInst smokeTable
        (smokeGame.transcript smokeCoins1) := hAcc.2
  have hNotFinal :
      ¬ sumcheckFinalOracleConsistentWithTable smokeInst smokeTable
        (smokeGame.transcript smokeCoins1) := by
    intro h
    have hEval :
        sumcheckEvalPoly (smokeProver.roundPoly 0 smokeCoins1) (f 1) =
          mleByFolding smokeTable smokeCoins1 := by
      simpa [sumcheckFinalOracleConsistentWithTable, smokeGame, SoundnessGame.transcript,
        smokeInst, smokeTable, smokeCoins1] using h.2.2
    have hEvalNe :
        ¬ sumcheckEvalPoly (smokeProver.roundPoly 0 smokeCoins1) (f 1) =
          mleByFolding smokeTable smokeCoins1 := by
      native_decide
    exact hEvalNe hEval
  exact hNotFinal hFinal

example :
    smokeGame.prefixRoundConsistent 0 (smokeCoins0.extract 0 0) := by
  exact smokeGame.prefixRoundConsistent_of_acceptsOn
    smokeCoins0
    smokeAcceptsOn0
    (i := 0)
    (by decide)
    (by decide)

example :
    roundFailureUnionCoins smokeGame.prefixGapEvent smokeGame.inst.rounds smokeCoins0 := by
  exact smokeGame.failureEvent_covered_by_prefixGapEvent
    smokeCoins0
    smokeFailureEvent0
    (by decide)
    (by decide)

example :
    smokeGame.lundBoundHolds (fullFieldUniformCoinProbModel smokeGame.inst.rounds) := by
  exact lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix smokeGame rfl (by decide)

/-!
Two-round smoke case:
- round 0 sends `1 + X`
- round 1 depends on the first verifier challenge
- the accepting false-claim run is on coins `#[0, 1]`

This exercises a nontrivial prefix of length `1`.
-/

def smoke2Inst : SumCheckInstance :=
  { rounds := 2
    maxDegree := 1
    domainSize := Goldilocks.q
    claimedValue := f 3 }

def smoke2Table : Array F := #[0, 0, 0, 0]

def smoke2Round0 : Array F := #[f 1, f 1]

def smoke2Round1Accept : Array F := #[f 1, f (Goldilocks.q - 1)]

def smoke2Round1Other : Array F := #[0, 0]

def smoke2Prover : OnlineProverStrategy smoke2Inst :=
  { roundPoly := fun i coins =>
      if h0 : i = 0 then
        smoke2Round0
      else if coins[0]! = 0 then
        smoke2Round1Accept
      else
        smoke2Round1Other
    roundPolyShape := by
      intro i hi _coins
      have hi' : i < 2 := by simpa [smoke2Inst] using hi
      have : i = 0 ∨ i = 1 := by omega
      rcases this with rfl | rfl
      · simp [smoke2Round0, smoke2Inst]
      · by_cases hCoin : _coins[0]! = 0
        · simp [hCoin, smoke2Round1Accept, smoke2Inst]
        · simp [hCoin, smoke2Round1Other, smoke2Inst]
    nonanticipatory := by
      intro i hi coins1 coins2 hPrefix
      have hi' : i < 2 := by simpa [smoke2Inst] using hi
      have : i = 0 ∨ i = 1 := by omega
      rcases this with rfl | rfl
      · simp
      · have h0 : coins1[0]! = coins2[0]! := hPrefix 0 (by omega)
        simp [h0]
  }

def smoke2Game : SoundnessGame :=
  { inst := smoke2Inst
    table := smoke2Table
    tableSize := by rfl
    falseClaim := by
      native_decide
    prover := smoke2Prover }

def smoke2Coins01 : Array F := #[0, f 1]

def smoke2Transcript01 : SumCheckTranscript :=
  { challenges := smoke2Coins01
    roundPolys := #[smoke2Round0, smoke2Round1Accept] }

theorem smoke2Accepted01 : sumcheckAccepted smoke2Inst smoke2Transcript01 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · change smoke2Inst.maxDegree ≤ smoke2Inst.domainSize
    native_decide
  · change smoke2Inst.rounds = 0 ∨ 0 < smoke2Inst.maxDegree
    native_decide
  · change smoke2Transcript01.challenges.size = smoke2Inst.rounds ∧
      smoke2Transcript01.roundPolys.size = smoke2Inst.rounds
    native_decide
  · intro i
    fin_cases i <;> simp [sumcheckRoundPolyShape, smoke2Transcript01, smoke2Inst,
      smoke2Round0, smoke2Round1Accept]
  · change sumcheckEvalPoly smoke2Round0 0 + sumcheckEvalPoly smoke2Round0 1 =
      smoke2Inst.claimedValue
    native_decide
  · constructor
    · native_decide
    · intro i hi
      have hi' : i + 1 < 2 := by
        simpa [smoke2Transcript01] using hi
      have hi0 : i = 0 := by omega
      subst hi0
      change sumcheckEvalPoly smoke2Round1Accept 0 + sumcheckEvalPoly smoke2Round1Accept 1 =
        sumcheckEvalPoly smoke2Round0 (smoke2Coins01[0]!)
      native_decide

theorem smoke2AcceptsOn01 : smoke2Game.acceptsOn smoke2Coins01 := by
  change sumcheckAcceptedForTable smoke2Inst smoke2Table (smoke2Game.transcript smoke2Coins01)
  refine ⟨?_, ?_⟩
  · simpa [smoke2Transcript01, smoke2Game, SoundnessGame.transcript, smoke2Coins01,
      smoke2Round0, smoke2Round1Accept] using smoke2Accepted01
  · constructor
    · rfl
    · refine ⟨smoke2Accepted01.2.2.1, ?_⟩
      have hEval :
          sumcheckEvalPoly (smoke2Round1Accept) (f 1) =
            mleByFolding smoke2Table smoke2Coins01 := by
        native_decide
      simpa [sumcheckFinalOracleConsistentWithTable, smoke2Game, SoundnessGame.transcript,
        smoke2Inst, smoke2Table, smoke2Coins01, smoke2Round0, smoke2Round1Accept] using hEval

theorem smoke2FailureEvent01 : smoke2Game.failureEvent smoke2Coins01 := by
  exact smoke2AcceptsOn01

example : smoke2Game.acceptsOn smoke2Coins01 := smoke2AcceptsOn01

example :
    smoke2Game.prefixRoundConsistent 1 (smoke2Coins01.extract 0 1) := by
  exact smoke2Game.prefixRoundConsistent_of_acceptsOn
    smoke2Coins01
    smoke2AcceptsOn01
    (i := 1)
    (by decide)
    (by decide)

example :
    roundFailureUnionCoins smoke2Game.prefixGapEvent smoke2Game.inst.rounds smoke2Coins01 := by
  exact smoke2Game.failureEvent_covered_by_prefixGapEvent
    smoke2Coins01
    smoke2FailureEvent01
    (by decide)
    (by decide)

example :
    smoke2Game.lundBoundHolds (fullFieldUniformCoinProbModel smoke2Game.inst.rounds) := by
  exact lundSoundnessAssumptionFullFieldAlignedPosRounds_prefix smoke2Game rfl (by decide)

def main : IO Unit := pure ()

end tests
