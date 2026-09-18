import SuperNeo.SumCheck.FullFieldCoins

/-!
`SoundnessGame`: per-round failure events, round target polynomials,
and the Lund round-boundary/kernel assumption layer.
-/

namespace SuperNeo.ProofSystem

namespace Sumcheck

/--
Paper-style SumCheck soundness game:
- an externally fixed table witness,
- a false-claim condition against `inst.claimedValue`,
- an online (non-anticipatory) prover strategy.
-/
structure SoundnessGame where
  inst : Instance
  table : Array F
  tableSize : table.size = 2 ^ inst.rounds
  falseClaim : SuperNeo.sumcheckTableSum table ≠ inst.claimedValue
  prover : OnlineProverStrategy inst

/-- Build a transcript by running an online prover strategy on verifier coins. -/
def SoundnessGame.transcript (g : SoundnessGame) (coins : Array F) : Transcript :=
  { challenges := coins
    roundPolys := Array.ofFn (fun i : Fin g.inst.rounds =>
      g.prover.roundPoly i.1 coins) }

/-- Game acceptance event on a specific verifier-coin sample. -/
def SoundnessGame.acceptsOn (g : SoundnessGame) (coins : Array F) : Prop :=
  let tr := g.transcript coins
  SuperNeo.sumcheckAcceptedForTable g.inst g.table tr

/-- Soundness-failure event family over verifier coins. -/
def SoundnessGame.failureEvent (g : SoundnessGame) : Array F → Prop :=
  fun coins => g.acceptsOn coins

/--
Challenge-vector replacement helper: substitute coordinate `i` with `x` while
keeping all other challenge coordinates unchanged.
-/
def SoundnessGame.challengeWith
  (g : SoundnessGame)
  (coins : Array F)
  (i : Nat)
  (x : F) : Array F :=
  Array.ofFn (fun k : Fin coins.size =>
    if k.1 = i then x else coins[k.1]!)

@[simp] theorem SoundnessGame.challengeWith_size
  (g : SoundnessGame)
  (coins : Array F)
  (i : Nat)
  (x : F) :
  (g.challengeWith coins i x).size = coins.size := by
  simp [SoundnessGame.challengeWith]

theorem SoundnessGame.challengeWith_eq_self_of_lt
  (g : SoundnessGame)
  (coins : Array F)
  (i : Nat)
  (hi : i < coins.size) :
  g.challengeWith coins i (coins[i]!) = coins := by
  apply Array.ext
  · simp [SoundnessGame.challengeWith]
  · intro j hj1 hj2
    by_cases hji : j = i
    · subst hji
      simp [SoundnessGame.challengeWith, hi]
    · simp [SoundnessGame.challengeWith, hji]

theorem SoundnessGame.challengeWith_getElem!_eq
  (g : SoundnessGame)
  (coins : Array F)
  (i : Nat)
  (x : F)
  (hi : i < coins.size) :
  (g.challengeWith coins i x)[i]! = x := by
  rw [getElem!_pos (c := g.challengeWith coins i x) (i := i)]
  · simp [SoundnessGame.challengeWith]
  · simpa [SoundnessGame.challengeWith]

theorem SoundnessGame.challengeWith_getElem!_of_ne
  (g : SoundnessGame)
  (coins : Array F)
  (i j : Nat)
  (x : F)
  (hji : j ≠ i) :
  (g.challengeWith coins i x)[j]! = coins[j]! := by
  by_cases hj : j < coins.size
  · rw [getElem!_pos (c := g.challengeWith coins i x) (i := j)]
    · rw [getElem!_pos (c := coins) (i := j) hj]
      simp [SoundnessGame.challengeWith, hji]
    · simpa [SoundnessGame.challengeWith] using hj
  · have hOut : ¬ j < (g.challengeWith coins i x).size := by
      simpa [SoundnessGame.challengeWith] using hj
    rw [getElem!_neg (c := g.challengeWith coins i x) (i := j) (h := hOut)]
    rw [getElem!_neg (c := coins) (i := j) (h := hj)]

theorem SoundnessGame.challengeWith_overwrite
  (g : SoundnessGame)
  (coins : Array F)
  (i : Nat)
  (x y : F) :
  g.challengeWith (g.challengeWith coins i x) i y = g.challengeWith coins i y := by
  apply Array.ext
  · simp [SoundnessGame.challengeWith]
  · intro j hj1 hj2
    by_cases hji : j = i
    · subst hji
      simp [SoundnessGame.challengeWith]
    · have hLeftPos :
          (g.challengeWith (g.challengeWith coins i x) i y)[j]! =
            (g.challengeWith (g.challengeWith coins i x) i y)[j] :=
        getElem!_pos (c := g.challengeWith (g.challengeWith coins i x) i y) (i := j) hj1
      have hRightPos :
          (g.challengeWith coins i y)[j]! = (g.challengeWith coins i y)[j] :=
        getElem!_pos (c := g.challengeWith coins i y) (i := j) hj2
      calc
        (g.challengeWith (g.challengeWith coins i x) i y)[j]
            = (g.challengeWith (g.challengeWith coins i x) i y)[j]! := by
                simpa using hLeftPos.symm
        _ = (g.challengeWith coins i x)[j]! := by
              exact g.challengeWith_getElem!_of_ne (g.challengeWith coins i x) i j y hji
        _ = coins[j]! := by
              exact g.challengeWith_getElem!_of_ne coins i j x hji
        _ = (g.challengeWith coins i y)[j]! := by
              symm
              exact g.challengeWith_getElem!_of_ne coins i j y hji
        _ = (g.challengeWith coins i y)[j] := by
              simpa using hRightPos

/--
Canonical prefix weight for table index `j` at round `i` under verifier coins.

This keeps only the first `i` challenge coordinates and multiplies the Boolean
selector factors against the corresponding index bits.
-/
def SoundnessGame.prefixWeight
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (j : Nat) : F :=
  (List.range i).foldl
    (fun acc k =>
      acc * eqTerm ((bitsToFieldArray g.inst.rounds j)[k]!) (coins[k]!))
    1

/--
Table-induced canonical round target evaluator at round `i` and point `x`.

This is the pointwise sum of index contributions weighted by:
- prefix constraints on coordinates `< i`,
- the current coordinate selector at `i`.
-/
def SoundnessGame.roundTargetEval
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (x : F) : F :=
  mleByInnerProduct g.table (g.challengeWith coins i x)

theorem SoundnessGame.roundTargetEval_invariant_at_index
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (x y : F) :
  g.roundTargetEval i (g.challengeWith coins i y) x = g.roundTargetEval i coins x := by
  unfold SoundnessGame.roundTargetEval
  simp [SoundnessGame.challengeWith_overwrite]

/--
Canonical linear target polynomial coefficients for round `i`, truncated to the
configured degree shape `maxDegree + 1`.

For degrees `>= 2`, coefficients are zero.
-/
def SoundnessGame.roundTargetPoly
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) : Array F :=
  let v0 := g.roundTargetEval i coins 0
  let v1 := g.roundTargetEval i coins 1
  Array.ofFn (fun k : Fin (g.inst.maxDegree + 1) =>
    if h0 : k.1 = 0 then
      v0
    else if h1 : k.1 = 1 then
      v1 - v0
    else
      0)

theorem SoundnessGame.roundTargetPoly_invariant_at_index
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (x : F) :
  g.roundTargetPoly i (g.challengeWith coins i x) = g.roundTargetPoly i coins := by
  unfold SoundnessGame.roundTargetPoly
  have h0 :
      g.roundTargetEval i (g.challengeWith coins i x) 0 =
        g.roundTargetEval i coins 0 :=
    g.roundTargetEval_invariant_at_index i coins 0 x
  have h1 :
      g.roundTargetEval i (g.challengeWith coins i x) 1 =
        g.roundTargetEval i coins 1 :=
    g.roundTargetEval_invariant_at_index i coins 1 x
  simp [h0, h1]

/--
Canonical round witness polynomial:
`proverRoundPoly_i - canonicalTargetPoly_i`.

This is the algebraic object whose sampled root-event is used in the
Schwartz-Zippel/Lund soundness path.
-/
def SoundnessGame.roundPolyWitness
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) : Array F :=
  let p := g.prover.roundPoly i coins
  let q := g.roundTargetPoly i coins
  Array.ofFn (fun k : Fin (g.inst.maxDegree + 1) => p[k.1]! - q[k.1]!)

theorem SoundnessGame.roundPolyWitness_invariant_at_index
  (g : SoundnessGame)
  (i : Nat)
  (hi : i < g.inst.rounds)
  (coins : Array F)
  (x : F) :
  g.roundPolyWitness i (g.challengeWith coins i x) = g.roundPolyWitness i coins := by
  have hProver :
      g.prover.roundPoly i (g.challengeWith coins i x) = g.prover.roundPoly i coins := by
    apply g.prover.nonanticipatory i hi
    intro j hj
    exact g.challengeWith_getElem!_of_ne coins i j x (by omega)
  have hTarget :
      g.roundTargetPoly i (g.challengeWith coins i x) = g.roundTargetPoly i coins :=
    g.roundTargetPoly_invariant_at_index i coins x
  unfold SoundnessGame.roundPolyWitness
  simp [hProver, hTarget]

/--
Canonical per-round event: the round witness polynomial vanishes at the sampled
challenge coordinate.
-/
def SoundnessGame.roundFailureCanonical
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) : Prop :=
  i < g.inst.rounds ∧
    sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) =
      g.roundTargetEval i coins (coins[i]!)

@[simp] theorem SoundnessGame.roundTargetPoly_size
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) :
  (g.roundTargetPoly i coins).size = g.inst.maxDegree + 1 := by
  simp [SoundnessGame.roundTargetPoly]

@[simp] theorem SoundnessGame.roundPolyWitness_size
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) :
  (g.roundPolyWitness i coins).size = g.inst.maxDegree + 1 := by
  simp [SoundnessGame.roundPolyWitness]

@[simp] theorem SoundnessGame.roundTargetPoly_get_zero
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) :
  (g.roundTargetPoly i coins)[0]! = g.roundTargetEval i coins 0 := by
  have hZero : 0 < (g.roundTargetPoly i coins).size := by
    simpa using Nat.succ_pos g.inst.maxDegree
  simp [SoundnessGame.roundTargetPoly, hZero]

theorem SoundnessGame.roundTargetPoly_get_one
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (hDegPos : 0 < g.inst.maxDegree) :
  (g.roundTargetPoly i coins)[1]! =
    g.roundTargetEval i coins 1 - g.roundTargetEval i coins 0 := by
  have hOne' : 1 < g.inst.maxDegree + 1 := Nat.succ_lt_succ hDegPos
  have hOne : 1 < (g.roundTargetPoly i coins).size := by
    simpa [SoundnessGame.roundTargetPoly_size] using hOne'
  rw [getElem!_pos (c := g.roundTargetPoly i coins) (i := 1) hOne]
  simp [SoundnessGame.roundTargetPoly]

theorem SoundnessGame.roundTargetPoly_get_fin
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (k : Fin (g.inst.maxDegree + 1)) :
  (g.roundTargetPoly i coins)[k] =
    if k.1 = 0 then g.roundTargetEval i coins 0
    else if k.1 = 1 then g.roundTargetEval i coins 1 - g.roundTargetEval i coins 0
    else 0 := by
  simp [SoundnessGame.roundTargetPoly]

theorem SoundnessGame.roundTargetPoly_get_ge_two
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  {k : Nat}
  (hk : k < g.inst.maxDegree + 1)
  (hk2 : 2 ≤ k) :
  (g.roundTargetPoly i coins)[k]! = 0 := by
  have hPos : k < (g.roundTargetPoly i coins).size := by
    simpa [SoundnessGame.roundTargetPoly_size] using hk
  rw [getElem!_pos (c := g.roundTargetPoly i coins) (i := k) hPos]
  have hk0 : k ≠ 0 := by omega
  have hk1 : k ≠ 1 := by omega
  simp [SoundnessGame.roundTargetPoly, hk0, hk1]

theorem SoundnessGame.roundPolyWitness_get
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  {k : Nat}
  (hk : k < g.inst.maxDegree + 1) :
  (g.roundPolyWitness i coins)[k]! =
    (g.prover.roundPoly i coins)[k]! - (g.roundTargetPoly i coins)[k]! := by
  simp [SoundnessGame.roundPolyWitness, hk]

theorem SoundnessGame.roundPolyWitness_eval_zero_iff
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) :
  g.roundFailureCanonical i coins ↔
    i < g.inst.rounds ∧
      sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) =
        g.roundTargetEval i coins (coins[i]!) := by
  rfl

theorem SoundnessGame.roundFailureCanonical_implies_lt
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (h : g.roundFailureCanonical i coins) :
  i < g.inst.rounds := h.1

theorem SoundnessGame.roundFailureCanonical_implies_root
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (h : g.roundFailureCanonical i coins) :
  sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) =
    g.roundTargetEval i coins (coins[i]!) := h.2

theorem SoundnessGame.roundFailureCanonical_implies_sub_eq_zero
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (h : g.roundFailureCanonical i coins) :
  sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) -
      g.roundTargetEval i coins (coins[i]!) = 0 := by
  exact sub_eq_zero.mpr h.2

theorem SoundnessGame.roundFailureCanonical_of_sub_eq_zero
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (hi : i < g.inst.rounds)
  (hSub :
    sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) -
      g.roundTargetEval i coins (coins[i]!) = 0) :
  g.roundFailureCanonical i coins := by
  exact ⟨hi, sub_eq_zero.mp hSub⟩

theorem SoundnessGame.roundFailureCanonical_iff_sub_eq_zero
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) :
  g.roundFailureCanonical i coins ↔
    i < g.inst.rounds ∧
      sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) -
        g.roundTargetEval i coins (coins[i]!) = 0 := by
  constructor
  · intro h
    exact ⟨h.1, sub_eq_zero.mpr h.2⟩
  · intro h
    exact ⟨h.1, sub_eq_zero.mp h.2⟩

theorem SoundnessGame.roundTargetEval_at_challenge_eq_mleByInnerProduct
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (hi : i < coins.size) :
  g.roundTargetEval i coins (coins[i]!) = mleByInnerProduct g.table coins := by
  unfold SoundnessGame.roundTargetEval
  simpa [SoundnessGame.challengeWith_eq_self_of_lt (g := g) (coins := coins) (i := i) hi]

theorem SoundnessGame.roundTargetEval_at_challenge_eq_mleByFolding
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (hi : i < coins.size)
  (hSize : coins.size = g.inst.rounds) :
  g.roundTargetEval i coins (coins[i]!) = mleByFolding g.table coins := by
  have hInner :
      g.roundTargetEval i coins (coins[i]!) = mleByInnerProduct g.table coins :=
    g.roundTargetEval_at_challenge_eq_mleByInnerProduct i coins hi
  have hTableSize :
      g.table.size = 2 ^ coins.size := by
    simpa [hSize] using g.tableSize
  have hBridge :
      mleByInnerProduct g.table coins = mleByFolding g.table coins :=
    mleByInnerProduct_eq_mleByFolding_of_size (v := g.table) (r := coins) hTableSize
  exact hInner.trans hBridge

theorem SoundnessGame.roundFailureCanonical_last_of_failureEvent
  (g : SoundnessGame)
  (coins : Array F)
  (hFail : g.failureEvent coins)
  (hRoundsPos : 0 < g.inst.rounds) :
  g.roundFailureCanonical (g.inst.rounds - 1) coins := by
  have hAccepted :
      SuperNeo.sumcheckAcceptedForTable g.inst g.table (g.transcript coins) := by
    simpa [SoundnessGame.failureEvent, SoundnessGame.acceptsOn] using hFail
  have hCore : SuperNeo.sumcheckAcceptedCore g.inst (g.transcript coins) := hAccepted.1
  have hRoundCons : SuperNeo.sumcheckRoundConsistent g.inst (g.transcript coins) := hCore.2.2.1
  have hChSize : (g.transcript coins).challenges.size = g.inst.rounds := hRoundCons.1
  have hRpSize : (g.transcript coins).roundPolys.size = g.inst.rounds := hRoundCons.2
  have hCoinsSize : coins.size = g.inst.rounds := by
    simpa [SoundnessGame.transcript] using hChSize
  have hRoundsNe : g.inst.rounds ≠ 0 := Nat.ne_of_gt hRoundsPos
  have hLastLtRounds : g.inst.rounds - 1 < g.inst.rounds := by omega
  have hLastLtRp : g.inst.rounds - 1 < (g.transcript coins).roundPolys.size := by
    omega
  have hLastLtCoins : g.inst.rounds - 1 < coins.size := by
    omega
  have hFinalEval :
      sumcheckEvalPoly (g.transcript coins).roundPolys[g.inst.rounds - 1]!
        (g.transcript coins).challenges[g.inst.rounds - 1]! =
      mleByFolding g.table (g.transcript coins).challenges := by
    simpa [SuperNeo.sumcheckFinalOracleConsistentWithTable, hRoundsNe] using hAccepted.2.2.2
  have hPolyLast :
      (g.transcript coins).roundPolys[g.inst.rounds - 1]! =
        g.prover.roundPoly (g.inst.rounds - 1) coins := by
    simpa [SoundnessGame.transcript, hLastLtRounds]
  have hChallengeLast :
      (g.transcript coins).challenges[g.inst.rounds - 1]! =
        coins[g.inst.rounds - 1]! := by
    simp [SoundnessGame.transcript, hLastLtCoins]
  have hProverEval :
      sumcheckEvalPoly (g.prover.roundPoly (g.inst.rounds - 1) coins)
          (coins[g.inst.rounds - 1]!) =
        mleByFolding g.table coins := by
    calc
      sumcheckEvalPoly (g.prover.roundPoly (g.inst.rounds - 1) coins)
          (coins[g.inst.rounds - 1]!)
          = sumcheckEvalPoly (g.transcript coins).roundPolys[g.inst.rounds - 1]!
              (g.transcript coins).challenges[g.inst.rounds - 1]! := by
                simp [hPolyLast, hChallengeLast]
      _ = mleByFolding g.table (g.transcript coins).challenges := hFinalEval
      _ = mleByFolding g.table coins := by simp [SoundnessGame.transcript]
  have hTargetEval :
      g.roundTargetEval (g.inst.rounds - 1) coins (coins[g.inst.rounds - 1]!) =
        mleByFolding g.table coins :=
    g.roundTargetEval_at_challenge_eq_mleByFolding
      (i := g.inst.rounds - 1) (coins := coins) hLastLtCoins hCoinsSize
  refine ⟨hLastLtRounds, ?_⟩
  exact hProverEval.trans hTargetEval.symm

/-- Soundness-failure advantage for a fixed game under a coin-probability model. -/
def SoundnessGame.advantage (prob : CoinProbModel) (g : SoundnessGame) : Rat :=
  prob.Pr g.failureEvent

theorem SoundnessGame.advantage_nonneg
  (prob : CoinProbModel) (g : SoundnessGame) :
  0 ≤ g.advantage prob := by
  exact prob.prNonneg g.failureEvent

theorem SoundnessGame.advantage_le_one
  (prob : CoinProbModel) (g : SoundnessGame) :
  g.advantage prob ≤ 1 := by
  exact prob.prLeOne g.failureEvent

/--
Cross-multiplied Lund/Schwartz-Zippel soundness bound shape:
`advantage * |K| ≤ ℓ·d`.

This avoids explicit division and remains well-defined for all `Nat` parameters.
-/
def SoundnessGame.lundBoundHolds (prob : CoinProbModel) (g : SoundnessGame) : Prop :=
  g.advantage prob * (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
    SuperNeo.sumcheckLundSoundnessNumerator g.inst

/--
Paper-facing boundary assumption for SumCheck soundness over the explicit game.

This is the non-scaffolded endpoint: probability is taken over verifier coins,
with fixed false claim and an adversarial prover strategy.
-/
def LundSoundnessAssumption : Prop :=
  ∀ (prob : CoinProbModel) (g : SoundnessGame), g.lundBoundHolds prob

/-- Finite union of round failure events over the first `n` rounds. -/
def roundFailureUnion (E : Nat → Prop) : Nat → Prop
  | 0 => False
  | n + 1 => roundFailureUnion E n ∨ E n

/-- Finite sum of per-round error bounds over the first `n` rounds. -/
def roundErrorSum (eps : Nat → Rat) : Nat → Rat
  | 0 => 0
  | n + 1 => roundErrorSum eps n + eps n

/-- Finite union of round-failure events over verifier coins. -/
def roundFailureUnionCoins (E : Nat → Array F → Prop) : Nat → (Array F → Prop)
  | 0 => fun _ => False
  | n + 1 => fun coins => roundFailureUnionCoins E n coins ∨ E n coins

theorem roundFailureUnionCoins_of_mem
  {E : Nat → Array F → Prop}
  {n i : Nat}
  {coins : Array F}
  (hi : i < n)
  (hEi : E i coins) :
  roundFailureUnionCoins E n coins := by
  induction n generalizing i with
  | zero =>
      exact (Nat.not_lt_zero _ hi).elim
  | succ n ih =>
      by_cases hEq : i = n
      · subst hEq
        simpa [roundFailureUnionCoins] using Or.inr hEi
      · have hi' : i < n := by omega
        exact Or.inl (ih hi' hEi)

theorem SoundnessGame.failureEvent_covered_by_roundFailureCanonical_of_rounds_pos
  (g : SoundnessGame)
  (coins : Array F)
  (hFail : g.failureEvent coins)
  (hRoundsPos : 0 < g.inst.rounds) :
  roundFailureUnionCoins g.roundFailureCanonical g.inst.rounds coins := by
  have hLast : g.roundFailureCanonical (g.inst.rounds - 1) coins :=
    g.roundFailureCanonical_last_of_failureEvent coins hFail hRoundsPos
  have hLastLt : g.inst.rounds - 1 < g.inst.rounds := by omega
  exact roundFailureUnionCoins_of_mem hLastLt hLast

theorem pr_roundFailureUnionCoins_le_roundErrorSum
  (prob : CoinProbModel)
  (E : Nat → Array F → Prop)
  (eps : Nat → Rat)
  (n : Nat)
  (hBound : ∀ i : Nat, i < n → prob.Pr (E i) ≤ eps i) :
  prob.Pr (roundFailureUnionCoins E n) ≤ roundErrorSum eps n := by
  induction n with
  | zero =>
      simpa [roundFailureUnionCoins, roundErrorSum, prob.prFalse] using (Rat.le_refl : (0 : Rat) ≤ 0)
  | succ n ih =>
      have hBoundPrev : ∀ i : Nat, i < n → prob.Pr (E i) ≤ eps i := by
        intro i hi
        exact hBound i (Nat.lt_trans hi (Nat.lt_succ_self n))
      have hBoundN : prob.Pr (E n) ≤ eps n := hBound n (Nat.lt_succ_self n)
      have hAddPrev :
          prob.Pr (roundFailureUnionCoins E n) + prob.Pr (E n) ≤
            roundErrorSum eps n + prob.Pr (E n) := by
        exact (Rat.add_le_add_right (c := prob.Pr (E n))).2 (ih hBoundPrev)
      have hAddLast :
          roundErrorSum eps n + prob.Pr (E n) ≤
            roundErrorSum eps n + eps n := by
        exact (Rat.add_le_add_left (c := roundErrorSum eps n)).2 hBoundN
      calc
        prob.Pr (roundFailureUnionCoins E (n + 1))
            = prob.Pr (fun coins => roundFailureUnionCoins E n coins ∨ E n coins) := by
                simp [roundFailureUnionCoins]
        _ ≤ prob.Pr (roundFailureUnionCoins E n) + prob.Pr (E n) := prob.prUnionLeAdd _ _
        _ ≤ roundErrorSum eps n + prob.Pr (E n) := hAddPrev
        _ ≤ roundErrorSum eps n + eps n := hAddLast
        _ = roundErrorSum eps (n + 1) := by
              simp [roundErrorSum]

/--
Round-by-round boundary sufficient to derive the Lund-style soundness bound.

This isolates the remaining closure work to:
1) constructing round-failure events that cover global failure,
2) proving per-round probability bounds (typically via Schwartz-Zippel),
3) proving the final accumulated-error inequality.
-/
structure LundRoundBoundary
  (prob : CoinProbModel)
  (g : SoundnessGame) where
  roundFailure : Nat → Array F → Prop
  epsRound : Nat → Rat
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundBound :
    ∀ i : Nat, i < g.inst.rounds → prob.Pr (roundFailure i) ≤ epsRound i
  totalBound :
    roundErrorSum epsRound g.inst.rounds *
      (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
        SuperNeo.sumcheckLundSoundnessNumerator g.inst

theorem SoundnessGame.lundBoundHolds_of_roundBoundary
  (prob : CoinProbModel)
  (g : SoundnessGame)
  (hRbr : LundRoundBoundary prob g) :
  g.lundBoundHolds prob := by
  unfold SoundnessGame.lundBoundHolds SoundnessGame.advantage
  let dRat : Rat := (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat)
  have hdNonnegCast :
      0 ≤ (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat) := by
    exact_mod_cast (Nat.zero_le (SuperNeo.sumcheckLundSoundnessDenominator g.inst))
  have hdNonneg : 0 ≤ dRat := by
    simpa [dRat] using hdNonnegCast
  have hCover :
      prob.Pr g.failureEvent ≤
        prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) := by
    exact prob.prMonotone hRbr.covered
  have hUnion :
      prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) ≤
        roundErrorSum hRbr.epsRound g.inst.rounds := by
    exact pr_roundFailureUnionCoins_le_roundErrorSum
      prob hRbr.roundFailure hRbr.epsRound g.inst.rounds hRbr.roundBound
  have hMul1 : prob.Pr g.failureEvent * dRat ≤
      prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat := by
    exact Rat.mul_le_mul_of_nonneg_right hCover hdNonneg
  have hMul2 :
      prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat ≤
        roundErrorSum hRbr.epsRound g.inst.rounds * dRat := by
    exact Rat.mul_le_mul_of_nonneg_right hUnion hdNonneg
  calc
    prob.Pr g.failureEvent * (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
        = prob.Pr g.failureEvent * dRat := by simp [dRat]
    _ ≤ prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat := hMul1
    _ ≤ roundErrorSum hRbr.epsRound g.inst.rounds * dRat := hMul2
    _ = roundErrorSum hRbr.epsRound g.inst.rounds *
          (SuperNeo.sumcheckLundSoundnessDenominator g.inst) := by simp [dRat]
    _ ≤ SuperNeo.sumcheckLundSoundnessNumerator g.inst := hRbr.totalBound

/--
Theorem-native closure surface for Lund soundness:
constructing round-by-round boundaries for every game suffices to prove the
global `LundSoundnessAssumption`.
-/
def LundRoundBoundaryAssumption : Prop :=
  ∀ (prob : CoinProbModel) (g : SoundnessGame), Nonempty (LundRoundBoundary prob g)

theorem lundSoundnessAssumption_of_roundBoundary
  (hRound : LundRoundBoundaryAssumption) :
  LundSoundnessAssumption := by
  intro prob g
  rcases hRound prob g with ⟨hRbr⟩
  exact SoundnessGame.lundBoundHolds_of_roundBoundary prob g hRbr

/--
Cross-multiplied union-bound helper:
if every per-round event satisfies `Pr(E_i) * d ≤ k`, then the finite union up to
`n` rounds satisfies `Pr(⋃_{i<n} E_i) * d ≤ n * k`.
-/
theorem pr_roundFailureUnionCoins_mul_le_const
  (prob : CoinProbModel)
  (E : Nat → Array F → Prop)
  (n : Nat)
  (d k : Rat)
  (hdNonneg : 0 ≤ d)
  (hBound : ∀ i : Nat, i < n → prob.Pr (E i) * d ≤ k) :
  prob.Pr (roundFailureUnionCoins E n) * d ≤ (n : Rat) * k := by
  induction n with
  | zero =>
      simpa [roundFailureUnionCoins, prob.prFalse]
        using (Rat.le_refl : (0 : Rat) ≤ 0)
  | succ n ih =>
      have hBoundPrev : ∀ i : Nat, i < n → prob.Pr (E i) * d ≤ k := by
        intro i hi
        exact hBound i (Nat.lt_trans hi (Nat.lt_succ_self n))
      have hBoundN : prob.Pr (E n) * d ≤ k := hBound n (Nat.lt_succ_self n)
      have hUnion :
          prob.Pr (roundFailureUnionCoins E (n + 1)) ≤
            prob.Pr (roundFailureUnionCoins E n) + prob.Pr (E n) := by
        simpa [roundFailureUnionCoins] using
          (prob.prUnionLeAdd (roundFailureUnionCoins E n) (E n))
      have hMulUnion :
          prob.Pr (roundFailureUnionCoins E (n + 1)) * d ≤
            (prob.Pr (roundFailureUnionCoins E n) + prob.Pr (E n)) * d := by
        exact Rat.mul_le_mul_of_nonneg_right hUnion hdNonneg
      have hAddPrev :
          prob.Pr (roundFailureUnionCoins E n) * d + prob.Pr (E n) * d ≤
            (n : Rat) * k + prob.Pr (E n) * d := by
        exact (Rat.add_le_add_right (c := prob.Pr (E n) * d)).2 (ih hBoundPrev)
      have hAddLast :
          (n : Rat) * k + prob.Pr (E n) * d ≤
            (n : Rat) * k + k := by
        exact (Rat.add_le_add_left (c := (n : Rat) * k)).2 hBoundN
      calc
        prob.Pr (roundFailureUnionCoins E (n + 1)) * d
            ≤ (prob.Pr (roundFailureUnionCoins E n) + prob.Pr (E n)) * d := hMulUnion
        _ = prob.Pr (roundFailureUnionCoins E n) * d + prob.Pr (E n) * d := by
              simpa using
                (Rat.add_mul (prob.Pr (roundFailureUnionCoins E n)) (prob.Pr (E n)) d)
        _ ≤ (n : Rat) * k + prob.Pr (E n) * d := hAddPrev
        _ ≤ (n : Rat) * k + k := hAddLast
        _ = ((n : Rat) + 1) * k := by
              calc
                (n : Rat) * k + k = (n : Rat) * k + 1 * k := by simp [Rat.one_mul]
                _ = ((n : Rat) + 1) * k := by
                      simpa [Rat.one_mul, Rat.add_comm, Rat.add_left_comm, Rat.add_assoc] using
                        (Rat.add_mul (n : Rat) 1 k).symm
        _ = ((n + 1 : Nat) : Rat) * k := by simp

/--
Cross-multiplied round-bound package (Schwartz-Zippel style):
per-round event bounds are stated directly as `Pr(E_i) * |K| ≤ d`.
-/
structure LundRoundBoundaryScaled
  (prob : CoinProbModel)
  (g : SoundnessGame) where
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundBoundScaled :
    ∀ i : Nat, i < g.inst.rounds →
      prob.Pr (roundFailure i) * (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
        (g.inst.maxDegree : Rat)

/--
Lower-level round-event kernel (Schwartz-Zippel style):
each round carries an explicit root-budget witness `d_i`, with
`d_i ≤ maxDegree` and cross-multiplied bound
`Pr(E_i) * |K| ≤ d_i`.
-/
structure LundRoundKernel
  (prob : CoinProbModel)
  (g : SoundnessGame) where
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundRootBudget : Nat → Nat
  roundRootBudgetBound :
    ∀ i : Nat, i < g.inst.rounds →
      roundRootBudget i ≤ g.inst.maxDegree
  roundProbBound :
    ∀ i : Nat, i < g.inst.rounds →
      prob.Pr (roundFailure i) * (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
        (roundRootBudget i : Rat)

/--
Kernel-to-boundary lift:
a Schwartz-Zippel round kernel induces the scaled Lund round boundary.
-/
def LundRoundBoundaryScaled.of_kernel
  (prob : CoinProbModel)
  (g : SoundnessGame)
  (hK : LundRoundKernel prob g) :
  LundRoundBoundaryScaled prob g := by
  refine
    { roundFailure := hK.roundFailure
      covered := hK.covered
      roundBoundScaled := ?_ }
  intro i hi
  have hProb : prob.Pr (hK.roundFailure i) * (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
      (hK.roundRootBudget i : Rat) := hK.roundProbBound i hi
  have hBudgetNat : hK.roundRootBudget i ≤ g.inst.maxDegree := hK.roundRootBudgetBound i hi
  have hBudget : (hK.roundRootBudget i : Rat) ≤ (g.inst.maxDegree : Rat) := by
    exact_mod_cast hBudgetNat
  exact Rat.le_trans hProb hBudget

theorem SoundnessGame.lundBoundHolds_of_scaledRoundBoundary
  (prob : CoinProbModel)
  (g : SoundnessGame)
  (hRbr : LundRoundBoundaryScaled prob g) :
  g.lundBoundHolds prob := by
  unfold SoundnessGame.lundBoundHolds SoundnessGame.advantage
  let dRat : Rat := (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat)
  have hdNonnegCast :
      0 ≤ (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat) := by
    exact_mod_cast (Nat.zero_le (SuperNeo.sumcheckLundSoundnessDenominator g.inst))
  have hdNonneg : 0 ≤ dRat := by
    simpa [dRat] using hdNonnegCast
  have hCover :
      prob.Pr g.failureEvent ≤
        prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) := by
    exact prob.prMonotone hRbr.covered
  have hCoverMul :
      prob.Pr g.failureEvent * dRat ≤
        prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat := by
    exact Rat.mul_le_mul_of_nonneg_right hCover hdNonneg
  have hRoundBound :
      ∀ i : Nat, i < g.inst.rounds →
        prob.Pr (hRbr.roundFailure i) * dRat ≤ (g.inst.maxDegree : Rat) := by
    intro i hi
    simpa [dRat] using hRbr.roundBoundScaled i hi
  have hUnionMul :
      prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat ≤
        (g.inst.rounds : Rat) * (g.inst.maxDegree : Rat) := by
    simpa [dRat] using
      (pr_roundFailureUnionCoins_mul_le_const
        prob hRbr.roundFailure g.inst.rounds dRat (g.inst.maxDegree : Rat) hdNonneg hRoundBound)
  calc
    prob.Pr g.failureEvent * (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
        = prob.Pr g.failureEvent * dRat := by simp [dRat]
    _ ≤ prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat := hCoverMul
    _ ≤ (g.inst.rounds : Rat) * (g.inst.maxDegree : Rat) := hUnionMul
    _ = SuperNeo.sumcheckLundSoundnessNumerator g.inst := by
          simp [SuperNeo.sumcheckLundSoundnessNumerator]

/--
Theorem-native closure surface using cross-multiplied per-round bounds.
-/
def LundRoundScaledBoundaryAssumption : Prop :=
  ∀ (prob : CoinProbModel) (g : SoundnessGame), Nonempty (LundRoundBoundaryScaled prob g)

theorem lundSoundnessAssumption_of_scaledRoundBoundary
  (hRound : LundRoundScaledBoundaryAssumption) :
  LundSoundnessAssumption := by
  intro prob g
  rcases hRound prob g with ⟨hRbr⟩
  exact SoundnessGame.lundBoundHolds_of_scaledRoundBoundary prob g hRbr

/--
Kernel-level assumption surface:
for every game, lower-level round-event lemmas produce a Schwartz-Zippel kernel.
-/
def LundRoundKernelAssumption : Prop :=
  ∀ (prob : CoinProbModel) (g : SoundnessGame), Nonempty (LundRoundKernel prob g)

theorem lundRoundScaledBoundaryAssumption_of_kernel
  (hKernel : LundRoundKernelAssumption) :
  LundRoundScaledBoundaryAssumption := by
  intro prob g
  rcases hKernel prob g with ⟨hK⟩
  exact ⟨LundRoundBoundaryScaled.of_kernel prob g hK⟩

theorem lundSoundnessAssumption_of_kernel
  (hKernel : LundRoundKernelAssumption) :
  LundSoundnessAssumption := by
  exact lundSoundnessAssumption_of_scaledRoundBoundary
    (lundRoundScaledBoundaryAssumption_of_kernel hKernel)

/--
Lower-level Schwartz-Zippel/round-event lemma package for a fixed game.

This is the intended theorem-native input surface from lower algebraic/probabilistic
proofs: a concrete round-event family, event coverage of global failure, and
cross-multiplied per-round bounds with explicit root budgets.
-/
structure SchwartzZippelRoundEventLemmas
  (prob : CoinProbModel)
  (g : SoundnessGame) where
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundRootBudget : Nat → Nat
  roundRootBudgetBound :
    ∀ i : Nat, i < g.inst.rounds →
      roundRootBudget i ≤ g.inst.maxDegree
  roundProbBoundScaled :
    ∀ i : Nat, i < g.inst.rounds →
      prob.Pr (roundFailure i) * (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
        (roundRootBudget i : Rat)

/--
Canonical cross-multiplied per-round bound surface for `roundFailureCanonical`.
-/
def CanonicalRoundBoundScaled
  (prob : CoinProbModel)
  (g : SoundnessGame) : Prop :=
  ∀ i : Nat, i < g.inst.rounds →
    prob.Pr (g.roundFailureCanonical i) *
      (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
        (g.inst.maxDegree : Rat)

/--
Constructive instantiation of `SchwartzZippelRoundEventLemmas` from the
canonical event family plus canonical per-round scaled bounds.

This theorem closes the event-packaging step once per-round bounds are available.
-/
def SchwartzZippelRoundEventLemmas.of_canonicalRoundBoundScaled
  (prob : CoinProbModel)
  (g : SoundnessGame)
  (hRoundsPos : 0 < g.inst.rounds)
  (hBound : CanonicalRoundBoundScaled prob g) :
  SchwartzZippelRoundEventLemmas prob g := by
  refine
    { roundFailure := g.roundFailureCanonical
      covered := ?_
      roundRootBudget := fun _ => g.inst.maxDegree
      roundRootBudgetBound := ?_
      roundProbBoundScaled := ?_ }
  · intro coins hFail
    exact g.failureEvent_covered_by_roundFailureCanonical_of_rounds_pos coins hFail hRoundsPos
  · intro i hi
    exact Nat.le_refl _
  · intro i hi
    simpa using hBound i hi

/--
Canonical lower-level SZ round-event package for the full-field model.

This is a named wrapper around
`SchwartzZippelRoundEventLemmas.of_canonicalRoundBoundScaled` used by the
aligned/positive-round closure chain below.
-/
noncomputable def canonicalRoundSzLemmas
  (g : SoundnessGame)
  (hRoundsPos : 0 < g.inst.rounds)
  (hBound : CanonicalRoundBoundScaled (fullFieldUniformCoinProbModel g.inst.rounds) g) :
  SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g :=
  SchwartzZippelRoundEventLemmas.of_canonicalRoundBoundScaled
    (prob := fullFieldUniformCoinProbModel g.inst.rounds)
    (g := g)
    hRoundsPos
    hBound

/-- Build a `LundRoundKernel` directly from lower-level Schwartz-Zippel lemmas. -/
def LundRoundKernel.of_schwartzZippelRoundEventLemmas
  (prob : CoinProbModel)
  (g : SoundnessGame)
  (hSz : SchwartzZippelRoundEventLemmas prob g) :
  LundRoundKernel prob g :=
  { roundFailure := hSz.roundFailure
    covered := hSz.covered
    roundRootBudget := hSz.roundRootBudget
    roundRootBudgetBound := hSz.roundRootBudgetBound
    roundProbBound := hSz.roundProbBoundScaled }

/--
Global closure surface: for every game, lower-level Schwartz-Zippel round-event
lemmas are available.
-/
def SchwartzZippelRoundEventAssumption : Prop :=
  ∀ (prob : CoinProbModel) (g : SoundnessGame),
    Nonempty (SchwartzZippelRoundEventLemmas prob g)

theorem lundRoundKernelAssumption_of_schwartzZippelRoundEvent
  (hSz : SchwartzZippelRoundEventAssumption) :
  LundRoundKernelAssumption := by
  intro prob g
  rcases hSz prob g with ⟨hSzGame⟩
  exact ⟨LundRoundKernel.of_schwartzZippelRoundEventLemmas prob g hSzGame⟩

theorem lundRoundScaledBoundaryAssumption_of_schwartzZippelRoundEvent
  (hSz : SchwartzZippelRoundEventAssumption) :
  LundRoundScaledBoundaryAssumption := by
  exact lundRoundScaledBoundaryAssumption_of_kernel
    (lundRoundKernelAssumption_of_schwartzZippelRoundEvent hSz)

theorem lundSoundnessAssumption_of_schwartzZippelRoundEvent
  (hSz : SchwartzZippelRoundEventAssumption) :
  LundSoundnessAssumption := by
  exact lundSoundnessAssumption_of_kernel
    (lundRoundKernelAssumption_of_schwartzZippelRoundEvent hSz)

end Sumcheck

end SuperNeo.ProofSystem
