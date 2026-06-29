import SuperNeo.ProofSystem.SumCheck.RootCounting

/-!
Domain-aligned witness packages for the full-field root-counting
layer and their conversions into Lund soundness assumptions.
-/

namespace SuperNeo.ProofSystem

namespace Sumcheck

/--
Lower-level executable root-set witness package for one game.

This is strictly weaker/more primitive than directly providing witness
polynomials: for each round we only require a finite root set that covers every
failing challenge coordinate and is budgeted by `maxDegree`.

From this, witness polynomials are constructed internally as finite products
`∏ (X - r)` and bridged back to coefficient arrays.
-/
structure FullFieldRoundPolynomialRootSetWitness
  (g : SoundnessGame)
  (hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g) where
  domainAligned :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q
  roundRootSet : Nat → Finset F
  roundRootSetBound :
    ∀ i : Nat, i < g.inst.rounds →
      (roundRootSet i).card ≤ g.inst.maxDegree
  roundFailureInRootSet :
    ∀ i : Nat, i < g.inst.rounds →
      ∀ coins : Array F,
        hSz.roundFailure i coins →
          coins[i]! ∈ roundRootSet i

/--
Constructive lift:
root-set witnesses induce full polynomial witnesses by using vanishing products
`∏_{r∈Sᵢ} (X-r)` and the coefficient-array bridge.
-/
noncomputable def FullFieldRoundPolynomialRootMathlibWitness.of_rootSetWitness
  (g : SoundnessGame)
  (hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g)
  (hSet : FullFieldRoundPolynomialRootSetWitness g hSz) :
  FullFieldRoundPolynomialRootMathlibWitness g hSz := by
  classical
  refine
    { domainAligned := hSet.domainAligned
      roundPoly := fun i =>
        zmodPolyToCoeffArray (g.inst.maxDegree + 1)
          (rootVanishingPoly (hSet.roundRootSet i))
      roundPolyShape := ?_
      roundPolyNonzero := ?_
      roundFailureImpliesPolyRoot := ?_ }
  · intro i hi
    simp [zmodPolyToCoeffArray]
  · intro i hi
    let pRoot : Polynomial Fq := rootVanishingPoly (hSet.roundRootSet i)
    have hDegLe : pRoot.natDegree ≤ g.inst.maxDegree := by
      simpa [pRoot, rootVanishingPoly_natDegree_eq_card] using
        hSet.roundRootSetBound i hi
    have hDegLt : pRoot.natDegree < g.inst.maxDegree + 1 :=
      Nat.lt_succ_of_le hDegLe
    have hEq :
        sumcheckPolynomialZMod
          (zmodPolyToCoeffArray (g.inst.maxDegree + 1) pRoot) = pRoot :=
      sumcheckPolynomialZMod_zmodPolyToCoeffArray (g.inst.maxDegree + 1) pRoot hDegLt
    have hNe : pRoot ≠ 0 := by
      simpa [pRoot] using rootVanishingPoly_ne_zero (hSet.roundRootSet i)
    intro hZero
    exact hNe (hEq.symm.trans hZero)
  · intro i hi coins hFail
    let pRoot : Polynomial Fq := rootVanishingPoly (hSet.roundRootSet i)
    have hDegLe : pRoot.natDegree ≤ g.inst.maxDegree := by
      simpa [pRoot, rootVanishingPoly_natDegree_eq_card] using
        hSet.roundRootSetBound i hi
    have hDegLt : pRoot.natDegree < g.inst.maxDegree + 1 :=
      Nat.lt_succ_of_le hDegLe
    have hEq :
        sumcheckPolynomialZMod
          (zmodPolyToCoeffArray (g.inst.maxDegree + 1) pRoot) = pRoot :=
      sumcheckPolynomialZMod_zmodPolyToCoeffArray (g.inst.maxDegree + 1) pRoot hDegLt
    have hMem : coins[i]! ∈ hSet.roundRootSet i :=
      hSet.roundFailureInRootSet i hi coins hFail
    have hEval :
        pRoot.eval (fToZMod (coins[i]!)) = 0 := by
      simpa [pRoot] using rootVanishingPoly_eval_eq_zero_of_mem hMem
    have hEval' :
        (sumcheckPolynomialZMod
          (zmodPolyToCoeffArray (g.inst.maxDegree + 1) pRoot)).eval
            (fToZMod (coins[i]!)) = 0 := by
      simpa [hEq] using hEval
    simpa [pRoot] using hEval'

/-- Global all-games root-set witness assumption (lower than polynomial witness layer). -/
def FullFieldRoundPolynomialRootSetWitnessAssumption : Prop :=
  ∀ g : SoundnessGame,
    ∃ hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g,
      Nonempty (FullFieldRoundPolynomialRootSetWitness g hSz)

/--
Domain-aligned variant of the root-set witness-layer assumption:
for aligned games (`|K| = |F| = q`), per-game SZ+root-set packages exist.
-/
def FullFieldRoundPolynomialRootSetWitnessAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      ∃ hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g,
        Nonempty (FullFieldRoundPolynomialRootSetWitness g hSz)

theorem fullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned_of_full
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumption) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned := by
  intro g _hAligned
  exact hWit g

theorem fullFieldRoundPolynomialRootSetWitnessAssumptionAligned_of_full
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumption) :
  FullFieldRoundPolynomialRootSetWitnessAssumptionAligned := by
  intro g _hAligned
  exact hSet g

theorem fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_aligned
  (hAligned :
    ∀ g : SoundnessGame,
      SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumption := by
  intro g
  exact hWit g (hAligned g)

theorem fullFieldRoundPolynomialRootSetWitnessAssumption_of_aligned
  (hAligned :
    ∀ g : SoundnessGame,
      SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumptionAligned) :
  FullFieldRoundPolynomialRootSetWitnessAssumption := by
  intro g
  exact hSet g (hAligned g)

theorem fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_rootSetWitness
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumption) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumption := by
  intro g
  rcases hSet g with ⟨hSz, hSetGameNonempty⟩
  rcases hSetGameNonempty with ⟨hSetGame⟩
  refine ⟨hSz, ?_⟩
  exact ⟨FullFieldRoundPolynomialRootMathlibWitness.of_rootSetWitness g hSz hSetGame⟩

theorem fullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned_of_rootSetWitnessAligned
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumptionAligned) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned := by
  intro g hAligned
  rcases hSet g hAligned with ⟨hSz, hSetGameNonempty⟩
  rcases hSetGameNonempty with ⟨hSetGame⟩
  refine ⟨hSz, ?_⟩
  exact ⟨FullFieldRoundPolynomialRootMathlibWitness.of_rootSetWitness g hSz hSetGame⟩

/--
Combined lower-level package for one game:
- full-field Schwartz-Zippel round-event lemmas, and
- polynomial witness/root lemmas for the same round events.

This is the theorem-native "single-source" surface that can instantiate both
`SchwartzZippelRoundEventAssumptionFullField` and
`FullFieldRoundPolynomialRootMathlibAssumption` without an extra witness layer.
-/
structure FullFieldRoundMathlibLemmas
  (g : SoundnessGame) where
  domainAligned :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q
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
      (fullFieldUniformCoinProbModel g.inst.rounds).Pr (roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
          (roundRootBudget i : Rat)
  roundPoly : Nat → Array F
  roundPolyShape :
    ∀ i : Nat, i < g.inst.rounds →
      (roundPoly i).size = g.inst.maxDegree + 1
  roundPolyNonzero :
    ∀ i : Nat, i < g.inst.rounds →
      sumcheckPolynomialZMod (roundPoly i) ≠ 0
  roundFailureImpliesPolyRoot :
    ∀ i : Nat, i < g.inst.rounds →
      ∀ coins : Array F,
        roundFailure i coins →
          (sumcheckPolynomialZMod (roundPoly i)).eval (fToZMod (coins[i]!)) = 0

/-- Global all-games combined lower-level package. -/
def FullFieldRoundMathlibAssumption : Prop :=
  ∀ g : SoundnessGame, Nonempty (FullFieldRoundMathlibLemmas g)

/--
Domain-aligned variant of the combined full-field package:
for aligned games (`|K| = |F| = q`), per-game Mathlib round lemmas exist.
-/
def FullFieldRoundMathlibAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      Nonempty (FullFieldRoundMathlibLemmas g)

/--
Aligned + positive-round full-field Lund endpoint.
-/
def LundSoundnessAssumptionFullFieldAlignedPosRounds : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      0 < g.inst.rounds →
        g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds)

/--
Domain-mismatch blocker: full-field round Mathlib lemmas are impossible for a game
whose challenge-domain denominator is not `Goldilocks.q`.
-/
theorem no_fullFieldRoundMathlibLemmas_of_domain_mismatch
  (g : SoundnessGame)
  (hMismatch :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q) :
  ¬ Nonempty (FullFieldRoundMathlibLemmas g) := by
  intro h
  rcases h with ⟨hGame⟩
  exact hMismatch hGame.domainAligned

theorem fullFieldDomainAligned_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q := by
  intro g
  rcases hMath g with ⟨hGame⟩
  exact hGame.domainAligned

theorem fullFieldRoundMathlibAssumptionAligned_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldRoundMathlibAssumptionAligned := by
  intro g _hAligned
  exact hMath g

theorem fullFieldRoundMathlibAssumption_of_aligned
  (hAligned :
    ∀ g : SoundnessGame,
      SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hMath : FullFieldRoundMathlibAssumptionAligned) :
  FullFieldRoundMathlibAssumption := by
  intro g
  exact hMath g (hAligned g)

private def fullFieldRoundMathlibMismatchGame : SoundnessGame where
  inst := { rounds := 0, maxDegree := 0, domainSize := 0, claimedValue := 0 }
  table := #[1]
  tableSize := by simp
  falseClaim := by
    simp [SuperNeo.sumcheckTableSum]
  prover :=
    { roundPoly := fun _ _ => #[0]
      roundPolyShape := by
        intro i hi
        exact (False.elim (Nat.not_lt_zero i hi))
      nonanticipatory := by
        intro i hi
        exact (False.elim (Nat.not_lt_zero i hi)) }

theorem not_fullFieldRoundMathlibAssumption :
  ¬ FullFieldRoundMathlibAssumption := by
  intro hAll
  let g : SoundnessGame := fullFieldRoundMathlibMismatchGame
  have hMismatch :
      SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q := by
    have hqNe : (0 : Nat) ≠ Goldilocks.q := Nat.ne_of_lt Goldilocks.q_pos
    simpa [g, fullFieldRoundMathlibMismatchGame,
      SuperNeo.sumcheckLundSoundnessDenominator] using hqNe
  exact (no_fullFieldRoundMathlibLemmas_of_domain_mismatch g hMismatch) (hAll g)

theorem no_fullFieldRoundEventRootCountLemmas_of_domain_mismatch
  (g : SoundnessGame)
  (hMismatch :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q) :
  ¬ Nonempty (FullFieldRoundEventRootCountLemmas g) := by
  intro h
  rcases h with ⟨hGame⟩
  exact hMismatch hGame.domainAligned

theorem no_fullFieldRoundPolynomialRootLemmas_of_domain_mismatch
  (g : SoundnessGame)
  (hMismatch :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q) :
  ¬ Nonempty (FullFieldRoundPolynomialRootLemmas g) := by
  intro h
  rcases h with ⟨hGame⟩
  exact hMismatch hGame.domainAligned

theorem no_fullFieldRoundPolynomialRootMathlibLemmas_of_domain_mismatch
  (g : SoundnessGame)
  (hMismatch :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q) :
  ¬ Nonempty (FullFieldRoundPolynomialRootMathlibLemmas g) := by
  intro h
  rcases h with ⟨hGame⟩
  exact hMismatch hGame.domainAligned

theorem not_fullFieldRoundEventRootCountAssumption :
  ¬ FullFieldRoundEventRootCountAssumption := by
  intro hAll
  let g : SoundnessGame := fullFieldRoundMathlibMismatchGame
  have hMismatch :
      SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q := by
    have hqNe : (0 : Nat) ≠ Goldilocks.q := Nat.ne_of_lt Goldilocks.q_pos
    simpa [g, fullFieldRoundMathlibMismatchGame,
      SuperNeo.sumcheckLundSoundnessDenominator] using hqNe
  exact (no_fullFieldRoundEventRootCountLemmas_of_domain_mismatch g hMismatch) (hAll g)

theorem not_fullFieldRoundPolynomialRootAssumption :
  ¬ FullFieldRoundPolynomialRootAssumption := by
  intro hAll
  let g : SoundnessGame := fullFieldRoundMathlibMismatchGame
  have hMismatch :
      SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q := by
    have hqNe : (0 : Nat) ≠ Goldilocks.q := Nat.ne_of_lt Goldilocks.q_pos
    simpa [g, fullFieldRoundMathlibMismatchGame,
      SuperNeo.sumcheckLundSoundnessDenominator] using hqNe
  exact (no_fullFieldRoundPolynomialRootLemmas_of_domain_mismatch g hMismatch) (hAll g)

theorem not_fullFieldRoundPolynomialRootMathlibAssumption :
  ¬ FullFieldRoundPolynomialRootMathlibAssumption := by
  intro hAll
  let g : SoundnessGame := fullFieldRoundMathlibMismatchGame
  have hMismatch :
      SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q := by
    have hqNe : (0 : Nat) ≠ Goldilocks.q := Nat.ne_of_lt Goldilocks.q_pos
    simpa [g, fullFieldRoundMathlibMismatchGame,
      SuperNeo.sumcheckLundSoundnessDenominator] using hqNe
  exact (no_fullFieldRoundPolynomialRootMathlibLemmas_of_domain_mismatch g hMismatch) (hAll g)

/-- Forgetful projection: combined package -> full-field SZ round-event lemmas. -/
def FullFieldRoundMathlibLemmas.to_schwartzZippel
  (g : SoundnessGame)
  (h : FullFieldRoundMathlibLemmas g) :
  SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g :=
  { roundFailure := h.roundFailure
    covered := h.covered
    roundRootBudget := h.roundRootBudget
    roundRootBudgetBound := h.roundRootBudgetBound
    roundProbBoundScaled := h.roundProbBoundScaled }

/-- Forgetful projection: combined package -> Mathlib witness package. -/
def FullFieldRoundMathlibLemmas.to_witness
  (g : SoundnessGame)
  (h : FullFieldRoundMathlibLemmas g) :
  FullFieldRoundPolynomialRootMathlibWitness g
    (FullFieldRoundMathlibLemmas.to_schwartzZippel g h) :=
  { domainAligned := h.domainAligned
    roundPoly := h.roundPoly
    roundPolyShape := h.roundPolyShape
    roundPolyNonzero := h.roundPolyNonzero
    roundFailureImpliesPolyRoot := h.roundFailureImpliesPolyRoot }

theorem fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumption := by
  intro g
  rcases hMath g with ⟨hGame⟩
  refine ⟨FullFieldRoundMathlibLemmas.to_schwartzZippel g hGame, ?_⟩
  exact ⟨FullFieldRoundMathlibLemmas.to_witness g hGame⟩

theorem fullFieldRoundPolynomialRootSetWitnessAssumption_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldRoundPolynomialRootSetWitnessAssumption := by
  intro g
  rcases hMath g with ⟨hGame⟩
  let hSzGame := FullFieldRoundMathlibLemmas.to_schwartzZippel g hGame
  refine ⟨hSzGame, ?_⟩
  refine ⟨{
    domainAligned := hGame.domainAligned
    roundRootSet := fun i =>
      Finset.univ.filter (fun r : F =>
        (sumcheckPolynomialZMod (hGame.roundPoly i)).eval (fToZMod r) = 0)
    roundRootSetBound := ?_
    roundFailureInRootSet := ?_
  }⟩
  · intro i hi
    have hShape : (hGame.roundPoly i).size = g.inst.maxDegree + 1 :=
      hGame.roundPolyShape i hi
    have hNz : sumcheckPolynomialZMod (hGame.roundPoly i) ≠ 0 :=
      hGame.roundPolyNonzero i hi
    simpa [fullFieldPolyRootCount] using
      (fullFieldPolyRootCount_le_maxDegree_of_shape_nonzero
        (poly := hGame.roundPoly i)
        (maxDegree := g.inst.maxDegree)
        hShape hNz)
  · intro i hi coins hFail
    have hEval :
        (sumcheckPolynomialZMod (hGame.roundPoly i)).eval (fToZMod (coins[i]!)) = 0 := by
      simpa [hSzGame, FullFieldRoundMathlibLemmas.to_schwartzZippel] using
        (hGame.roundFailureImpliesPolyRoot i hi coins hFail)
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, hEval⟩

theorem schwartzZippelRoundEventAssumptionFullField_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  ∀ g : SoundnessGame,
    Nonempty (SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g) := by
  intro g
  rcases hMath g with ⟨hGame⟩
  exact ⟨FullFieldRoundMathlibLemmas.to_schwartzZippel g hGame⟩

theorem fullFieldRoundPolynomialRootMathlibAssumption_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldRoundPolynomialRootMathlibAssumption := by
  intro g
  rcases hMath g with ⟨hGame⟩
  let hSzGame := FullFieldRoundMathlibLemmas.to_schwartzZippel g hGame
  let hWitGame := FullFieldRoundMathlibLemmas.to_witness g hGame
  exact ⟨{
    domainAligned := hWitGame.domainAligned
    roundFailure := hSzGame.roundFailure
    covered := hSzGame.covered
    roundPoly := hWitGame.roundPoly
    roundPolyShape := hWitGame.roundPolyShape
    roundPolyNonzero := hWitGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := hWitGame.roundFailureImpliesPolyRoot
  }⟩

theorem fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_fullFieldRoundMathlibAligned
  (hMath : FullFieldRoundMathlibAssumptionAligned) :
  FullFieldRoundPolynomialRootMathlibAssumptionAligned := by
  intro g hAligned
  rcases hMath g hAligned with ⟨hGame⟩
  let hSzGame := FullFieldRoundMathlibLemmas.to_schwartzZippel g hGame
  let hWitGame := FullFieldRoundMathlibLemmas.to_witness g hGame
  exact ⟨{
    domainAligned := hWitGame.domainAligned
    roundFailure := hSzGame.roundFailure
    covered := hSzGame.covered
    roundPoly := hWitGame.roundPoly
    roundPolyShape := hWitGame.roundPolyShape
    roundPolyNonzero := hWitGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := hWitGame.roundFailureImpliesPolyRoot
  }⟩

/--
Constructive instantiation of the global Mathlib-root package from:
1) full-field Schwartz-Zippel round-event lemmas for each game, and
2) polynomial witness/root lemmas for those round events.
-/
theorem fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelWitness
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumption) :
  FullFieldRoundPolynomialRootMathlibAssumption := by
  intro g
  rcases hWit g with ⟨hSzGame, hWitGameNonempty⟩
  rcases hWitGameNonempty with ⟨hWitGame⟩
  refine ⟨{
    domainAligned := hWitGame.domainAligned
    roundFailure := hSzGame.roundFailure
    covered := hSzGame.covered
    roundPoly := hWitGame.roundPoly
    roundPolyShape := hWitGame.roundPolyShape
    roundPolyNonzero := hWitGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := ?_
  }⟩
  intro i hi coins hFail
  exact hWitGame.roundFailureImpliesPolyRoot i hi coins hFail

theorem fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_schwartzZippelWitnessAligned
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned) :
  FullFieldRoundPolynomialRootMathlibAssumptionAligned := by
  intro g hAligned
  rcases hWit g hAligned with ⟨hSzGame, hWitGameNonempty⟩
  rcases hWitGameNonempty with ⟨hWitGame⟩
  refine ⟨{
    domainAligned := hWitGame.domainAligned
    roundFailure := hSzGame.roundFailure
    covered := hSzGame.covered
    roundPoly := hWitGame.roundPoly
    roundPolyShape := hWitGame.roundPolyShape
    roundPolyNonzero := hWitGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := ?_
  }⟩
  intro i hi coins hFail
  exact hWitGame.roundFailureImpliesPolyRoot i hi coins hFail

theorem fullFieldRoundPolynomialRootAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundPolynomialRootAssumption := by
  intro g
  rcases hMathlib g with ⟨hMathlibGame⟩
  exact ⟨FullFieldRoundPolynomialRootLemmas.of_mathlib g hMathlibGame⟩

/--
Constructive closure:
polynomial-root lemmas imply paper-style round-event root-count lemmas.
-/
theorem fullFieldRoundEventRootCountAssumption_of_polynomialRoot
  (hPoly : FullFieldRoundPolynomialRootAssumption) :
  FullFieldRoundEventRootCountAssumption := by
  intro g
  rcases hPoly g with ⟨hPolyGame⟩
  exact ⟨FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas g hPolyGame⟩

/--
Direct constructive closure:
Mathlib-root-count package implies full-field root-count assumption.
-/
theorem fullFieldRoundEventRootCountAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundEventRootCountAssumption := by
  exact fullFieldRoundEventRootCountAssumption_of_polynomialRoot
    (fullFieldRoundPolynomialRootAssumption_of_mathlib hMathlib)

/--
Global denominator-alignment surface for full-field soundness games:
`|K| = |F| = Goldilocks.q`.
-/
def FullFieldDomainAlignedAssumption : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q

theorem fullFieldDomainAlignedAssumption_of_fullFieldRoundEventRootCount
  (hRoot : FullFieldRoundEventRootCountAssumption) :
  FullFieldDomainAlignedAssumption := by
  intro g
  rcases hRoot g with ⟨hGame⟩
  exact hGame.domainAligned

theorem fullFieldDomainAlignedAssumption_of_fullFieldRoundPolynomialRoot
  (hPoly : FullFieldRoundPolynomialRootAssumption) :
  FullFieldDomainAlignedAssumption := by
  intro g
  rcases hPoly g with ⟨hGame⟩
  exact hGame.domainAligned

theorem fullFieldDomainAlignedAssumption_of_fullFieldRoundPolynomialRootMathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldDomainAlignedAssumption := by
  intro g
  rcases hMathlib g with ⟨hGame⟩
  exact hGame.domainAligned

theorem fullFieldRoundEventRootCountAssumptionAligned_of_fullFieldRoundEventRootCount
  (hRoot : FullFieldRoundEventRootCountAssumption) :
  FullFieldRoundEventRootCountAssumptionAligned := by
  intro g _hAligned
  exact hRoot g

theorem fullFieldRoundEventRootCountAssumption_of_aligned
  (hAligned : FullFieldDomainAlignedAssumption)
  (hRoot : FullFieldRoundEventRootCountAssumptionAligned) :
  FullFieldRoundEventRootCountAssumption := by
  intro g
  exact hRoot g (hAligned g)

theorem fullFieldRoundPolynomialRootAssumptionAligned_of_fullFieldRoundPolynomialRoot
  (hPoly : FullFieldRoundPolynomialRootAssumption) :
  FullFieldRoundPolynomialRootAssumptionAligned := by
  intro g _hAligned
  exact hPoly g

theorem fullFieldRoundPolynomialRootAssumption_of_aligned
  (hAligned : FullFieldDomainAlignedAssumption)
  (hPoly : FullFieldRoundPolynomialRootAssumptionAligned) :
  FullFieldRoundPolynomialRootAssumption := by
  intro g
  exact hPoly g (hAligned g)

theorem fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_fullFieldRoundPolynomialRootMathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundPolynomialRootMathlibAssumptionAligned := by
  intro g _hAligned
  exact hMathlib g

theorem fullFieldRoundPolynomialRootMathlibAssumption_of_aligned
  (hAligned : FullFieldDomainAlignedAssumption)
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumptionAligned) :
  FullFieldRoundPolynomialRootMathlibAssumption := by
  intro g
  exact hMathlib g (hAligned g)

theorem fullFieldRoundEventRootCountAssumptionAligned_of_fullFieldRoundMathlibAligned
  (hMath : FullFieldRoundMathlibAssumptionAligned) :
  FullFieldRoundEventRootCountAssumptionAligned := by
  intro g hAligned
  have hMathlibAligned :
      FullFieldRoundPolynomialRootMathlibAssumptionAligned :=
    fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_fullFieldRoundMathlibAligned hMath
  rcases hMathlibAligned g hAligned with ⟨hMathGame⟩
  let hPoly : FullFieldRoundPolynomialRootLemmas g :=
    FullFieldRoundPolynomialRootLemmas.of_mathlib g hMathGame
  exact ⟨FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas g hPoly⟩

theorem fullFieldDomainAlignedAssumption_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldDomainAlignedAssumption := by
  exact fullFieldDomainAligned_of_fullFieldRoundMathlib hMath

theorem fullFieldRoundMathlibAssumption_of_domainAlignedAssumption
  (hAligned : FullFieldDomainAlignedAssumption)
  (hMath : FullFieldRoundMathlibAssumptionAligned) :
  FullFieldRoundMathlibAssumption := by
  exact fullFieldRoundMathlibAssumption_of_aligned hAligned hMath

/--
Constructive lift from full-field round-event cardinality lemmas to the
Schwartz-Zippel round-event theorem surface.
-/
def SchwartzZippelRoundEventLemmas.of_fullFieldCardinality
  (g : SoundnessGame)
  (hCard : FullFieldRoundEventCardinalityLemmas g) :
  SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g := by
  refine
    { roundFailure := hCard.roundFailure
      covered := hCard.covered
      roundRootBudget := hCard.roundRootBudget
      roundRootBudgetBound := hCard.roundRootBudgetBound
      roundProbBoundScaled := ?_ }
  intro i hi
  have hScaledNat :
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
          hCard.roundRootBudget i * (fullFieldCoinSpace g.inst.rounds).length :=
    hCard.roundCountBoundScaled i hi
  have hProb :
      fullFieldCoinPr g.inst.rounds (hCard.roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat) ≤
          (hCard.roundRootBudget i : Rat) := by
    exact fullFieldCoinPr_mul_nat_le_of_countScaled
      g.inst.rounds
      (hCard.roundFailure i)
      (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
      (hCard.roundRootBudget i)
      hScaledNat
  simpa [fullFieldUniformCoinProbModel] using hProb

/-- Global full-field closure surface for round-event cardinality lemmas. -/
def FullFieldRoundEventCardinalityAssumption : Prop :=
  ∀ g : SoundnessGame, Nonempty (FullFieldRoundEventCardinalityLemmas g)

theorem fullFieldRoundEventCardinalityAssumption_of_rootCount
  (hRoot : FullFieldRoundEventRootCountAssumption) :
  FullFieldRoundEventCardinalityAssumption := by
  intro g
  rcases hRoot g with ⟨hRootGame⟩
  exact ⟨FullFieldRoundEventCardinalityLemmas.of_rootCount g hRootGame⟩

/--
Direct constructive closure:
Mathlib-root-count package implies full-field cardinality assumption.
-/
theorem fullFieldRoundEventCardinalityAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundEventCardinalityAssumption := by
  exact fullFieldRoundEventCardinalityAssumption_of_rootCount
    (fullFieldRoundEventRootCountAssumption_of_mathlib hMathlib)

/--
Constructive combined-package instantiation from Mathlib-root packages.

Given per-game Mathlib-root polynomial witnesses, we build the full-field SZ
round-event lemmas through the canonical root-count/cardinality conversions and
package both layers together as `FullFieldRoundMathlibLemmas`.
-/
theorem fullFieldRoundMathlibAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundMathlibAssumption := by
  intro g
  rcases hMathlib g with ⟨hMathGame⟩
  let hPoly : FullFieldRoundPolynomialRootLemmas g :=
    FullFieldRoundPolynomialRootLemmas.of_mathlib g hMathGame
  let hRoot : FullFieldRoundEventRootCountLemmas g :=
    FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas g hPoly
  let hCard : FullFieldRoundEventCardinalityLemmas g :=
    FullFieldRoundEventCardinalityLemmas.of_rootCount g hRoot
  let hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g :=
    SchwartzZippelRoundEventLemmas.of_fullFieldCardinality g hCard
  refine ⟨{
    domainAligned := hMathGame.domainAligned
    roundFailure := hSz.roundFailure
    covered := hSz.covered
    roundRootBudget := hSz.roundRootBudget
    roundRootBudgetBound := hSz.roundRootBudgetBound
    roundProbBoundScaled := hSz.roundProbBoundScaled
    roundPoly := hMathGame.roundPoly
    roundPolyShape := hMathGame.roundPolyShape
    roundPolyNonzero := hMathGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := ?_
  }⟩
  intro i hi coins hFail
  have hFailMath : hMathGame.roundFailure i coins := by
    simpa [hSz, hCard, hRoot, hPoly] using hFail
  exact hMathGame.roundFailureImpliesPolyRoot i hi coins hFailMath

theorem fullFieldRoundMathlibAssumptionAligned_of_mathlibAligned
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumptionAligned) :
  FullFieldRoundMathlibAssumptionAligned := by
  intro g hAligned
  rcases hMathlib g hAligned with ⟨hMathGame⟩
  let hPoly : FullFieldRoundPolynomialRootLemmas g :=
    FullFieldRoundPolynomialRootLemmas.of_mathlib g hMathGame
  let hRoot : FullFieldRoundEventRootCountLemmas g :=
    FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas g hPoly
  let hCard : FullFieldRoundEventCardinalityLemmas g :=
    FullFieldRoundEventCardinalityLemmas.of_rootCount g hRoot
  let hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g :=
    SchwartzZippelRoundEventLemmas.of_fullFieldCardinality g hCard
  refine ⟨{
    domainAligned := hMathGame.domainAligned
    roundFailure := hSz.roundFailure
    covered := hSz.covered
    roundRootBudget := hSz.roundRootBudget
    roundRootBudgetBound := hSz.roundRootBudgetBound
    roundProbBoundScaled := hSz.roundProbBoundScaled
    roundPoly := hMathGame.roundPoly
    roundPolyShape := hMathGame.roundPolyShape
    roundPolyNonzero := hMathGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := ?_
  }⟩
  intro i hi coins hFail
  have hFailMath : hMathGame.roundFailure i coins := by
    simpa [hSz, hCard, hRoot, hPoly] using hFail
  exact hMathGame.roundFailureImpliesPolyRoot i hi coins hFailMath

theorem fullFieldRoundPolynomialRootSetWitnessAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundPolynomialRootSetWitnessAssumption := by
  exact fullFieldRoundPolynomialRootSetWitnessAssumption_of_fullFieldRoundMathlib
    (fullFieldRoundMathlibAssumption_of_mathlib hMathlib)

/--
Direct constructive closure:
root-set witness packages imply global Mathlib-root packages.
-/
theorem fullFieldRoundPolynomialRootMathlibAssumption_of_rootSetWitness
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumption) :
  FullFieldRoundPolynomialRootMathlibAssumption := by
  exact fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelWitness
    (fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_rootSetWitness hSet)

/--
Direct constructive closure:
root-set witness packages imply the combined full-field round package.
-/
theorem fullFieldRoundMathlibAssumption_of_rootSetWitness
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumption) :
  FullFieldRoundMathlibAssumption := by
  exact fullFieldRoundMathlibAssumption_of_mathlib
    (fullFieldRoundPolynomialRootMathlibAssumption_of_rootSetWitness hSet)

theorem fullFieldRoundMathlibAssumptionAligned_of_rootSetWitnessAligned
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumptionAligned) :
  FullFieldRoundMathlibAssumptionAligned := by
  exact fullFieldRoundMathlibAssumptionAligned_of_mathlibAligned
    (fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_schwartzZippelWitnessAligned
      (fullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned_of_rootSetWitnessAligned hSet))

/--
Direct constructive closure:
combined full-field round package implies full-field root-count assumption.
-/
theorem fullFieldRoundEventRootCountAssumption_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldRoundEventRootCountAssumption := by
  exact fullFieldRoundEventRootCountAssumption_of_mathlib
    (fullFieldRoundPolynomialRootMathlibAssumption_of_fullFieldRoundMathlib hMath)

/--
Direct constructive closure:
root-set witness packages imply full-field root-count assumption.
-/
theorem fullFieldRoundEventRootCountAssumption_of_rootSetWitness
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumption) :
  FullFieldRoundEventRootCountAssumption := by
  exact fullFieldRoundEventRootCountAssumption_of_mathlib
    (fullFieldRoundPolynomialRootMathlibAssumption_of_rootSetWitness hSet)

/--
Global full-field Schwartz-Zippel round-event theorem surface.
-/
def SchwartzZippelRoundEventAssumptionFullField : Prop :=
  ∀ g : SoundnessGame,
    Nonempty (SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g)

/--
Constructive witness-layer instantiation from the Mathlib-root package itself.

This removes the need to provide a separate witness assumption when an all-games
`FullFieldRoundPolynomialRootMathlibAssumption` package is already available:
the required full-field SZ event package is constructed canonically from the
Mathlib-root chain and paired with the original polynomial witnesses.
-/
theorem fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumption := by
  exact fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_rootSetWitness
    (fullFieldRoundPolynomialRootSetWitnessAssumption_of_mathlib hMathlib)

theorem schwartzZippelRoundEventAssumptionFullField_of_witness
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumption) :
  SchwartzZippelRoundEventAssumptionFullField := by
  intro g
  rcases hWit g with ⟨hSzGame, _hWitGame⟩
  exact ⟨hSzGame⟩

theorem fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelFullFieldWitness
  (_hSz : SchwartzZippelRoundEventAssumptionFullField)
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumption) :
  FullFieldRoundPolynomialRootMathlibAssumption := by
  exact fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelWitness hWit

theorem fullFieldRoundEventCardinalityAssumption_of_schwartzZippelFullField
  (hSz : SchwartzZippelRoundEventAssumptionFullField) :
  FullFieldRoundEventCardinalityAssumption := by
  intro g
  rcases hSz g with ⟨hSzGame⟩
  exact ⟨FullFieldRoundEventCardinalityLemmas.of_schwartzZippel g hSzGame⟩

theorem fullFieldRoundEventRootCountAssumption_of_cardinality
  (hDomain : FullFieldDomainAlignedAssumption)
  (hCard : FullFieldRoundEventCardinalityAssumption) :
  FullFieldRoundEventRootCountAssumption := by
  intro g
  rcases hCard g with ⟨hCardGame⟩
  exact ⟨FullFieldRoundEventRootCountLemmas.of_cardinality g (hDomain g) hCardGame⟩

theorem fullFieldRoundEventRootCountAssumption_of_schwartzZippelFullField
  (hDomain : FullFieldDomainAlignedAssumption)
  (hSz : SchwartzZippelRoundEventAssumptionFullField) :
  FullFieldRoundEventRootCountAssumption := by
  exact fullFieldRoundEventRootCountAssumption_of_cardinality
    hDomain
    (fullFieldRoundEventCardinalityAssumption_of_schwartzZippelFullField hSz)

theorem schwartzZippelRoundEventAssumptionFullField_of_cardinality
  (hCard : FullFieldRoundEventCardinalityAssumption) :
  SchwartzZippelRoundEventAssumptionFullField := by
  intro g
  rcases hCard g with ⟨hCardGame⟩
  exact ⟨SchwartzZippelRoundEventLemmas.of_fullFieldCardinality g hCardGame⟩

theorem schwartzZippelRoundEventAssumptionFullField_of_rootCount
  (hRoot : FullFieldRoundEventRootCountAssumption) :
  SchwartzZippelRoundEventAssumptionFullField := by
  exact schwartzZippelRoundEventAssumptionFullField_of_cardinality
    (fullFieldRoundEventCardinalityAssumption_of_rootCount hRoot)

/--
Direct constructive closure:
Mathlib-root-count package implies full-field Schwartz-Zippel round-event assumption.
-/
theorem schwartzZippelRoundEventAssumptionFullField_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  SchwartzZippelRoundEventAssumptionFullField := by
  exact schwartzZippelRoundEventAssumptionFullField_of_cardinality
    (fullFieldRoundEventCardinalityAssumption_of_mathlib hMathlib)

/--
Full-field Lund soundness endpoint:
for every game, the canonical full-field coin model satisfies the Lund bound.
-/
def LundSoundnessAssumptionFullField : Prop :=
  ∀ g : SoundnessGame, g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds)

/--
Aligned full-field Lund soundness endpoint:
for aligned games (`|K| = |F| = q`), the canonical full-field coin model
satisfies the Lund bound.
-/
def LundSoundnessAssumptionFullFieldAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds)

theorem lundSoundnessAssumptionFullField_of_schwartzZippelRoundEvent
  (hSz : SchwartzZippelRoundEventAssumptionFullField) :
  LundSoundnessAssumptionFullField := by
  intro g
  rcases hSz g with ⟨hSzGame⟩
  let prob := fullFieldUniformCoinProbModel g.inst.rounds
  have hKernel : LundRoundKernel prob g :=
    LundRoundKernel.of_schwartzZippelRoundEventLemmas prob g hSzGame
  have hScaled : LundRoundBoundaryScaled prob g :=
    LundRoundBoundaryScaled.of_kernel prob g hKernel
  exact SoundnessGame.lundBoundHolds_of_scaledRoundBoundary prob g hScaled

theorem lundSoundnessAssumptionFullField_of_rootCount
  (hRoot : FullFieldRoundEventRootCountAssumption) :
  LundSoundnessAssumptionFullField := by
  exact lundSoundnessAssumptionFullField_of_schwartzZippelRoundEvent
    (schwartzZippelRoundEventAssumptionFullField_of_rootCount hRoot)

/--
Direct constructive closure:
Mathlib-root-count package implies full-field Lund soundness endpoint.
-/
theorem lundSoundnessAssumptionFullField_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  LundSoundnessAssumptionFullField := by
  exact lundSoundnessAssumptionFullField_of_schwartzZippelRoundEvent
    (schwartzZippelRoundEventAssumptionFullField_of_mathlib hMathlib)

theorem lundSoundnessAssumptionFullFieldAligned_of_mathlibAligned
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumptionAligned) :
  LundSoundnessAssumptionFullFieldAligned := by
  intro g hAligned
  rcases hMathlib g hAligned with ⟨hMathGame⟩
  let hPoly : FullFieldRoundPolynomialRootLemmas g :=
    FullFieldRoundPolynomialRootLemmas.of_mathlib g hMathGame
  let hRoot : FullFieldRoundEventRootCountLemmas g :=
    FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas g hPoly
  let hCard : FullFieldRoundEventCardinalityLemmas g :=
    FullFieldRoundEventCardinalityLemmas.of_rootCount g hRoot
  let hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g :=
    SchwartzZippelRoundEventLemmas.of_fullFieldCardinality g hCard
  let prob := fullFieldUniformCoinProbModel g.inst.rounds
  have hKernel : LundRoundKernel prob g :=
    LundRoundKernel.of_schwartzZippelRoundEventLemmas prob g hSz
  have hScaled : LundRoundBoundaryScaled prob g :=
    LundRoundBoundaryScaled.of_kernel prob g hKernel
  exact SoundnessGame.lundBoundHolds_of_scaledRoundBoundary prob g hScaled


theorem lundSoundnessAssumptionFullField_of_mathlibAligned
  (hAligned : FullFieldDomainAlignedAssumption)
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumptionAligned) :
  LundSoundnessAssumptionFullField := by
  exact lundSoundnessAssumptionFullField_of_mathlib
    (fullFieldRoundPolynomialRootMathlibAssumption_of_aligned hAligned hMathlib)

end Sumcheck

end SuperNeo.ProofSystem
