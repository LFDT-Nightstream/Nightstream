import SuperNeo.ProofSystem.SumCheck.Game
import Mathlib.Algebra.Polynomial.OfFn

/-!
Full-field round-event cardinality via polynomial root counting:
the ZMod bridge and mathlib-backed root-count lemmas.
-/

namespace SuperNeo.ProofSystem

namespace Sumcheck

/--
Full-field lower-level round-event package:
for the canonical coin model `fullFieldUniformCoinProbModel rounds`, each round
provides a root-budget witness and a count-scaled inequality.
-/
structure FullFieldRoundEventCardinalityLemmas
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
  roundCountBoundScaled :
    ∀ i : Nat, i < g.inst.rounds →
      fullFieldCoinEventCount g.inst.rounds (roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
          roundRootBudget i * (fullFieldCoinSpace g.inst.rounds).length

/--
Lower-level full-field root-count package (paper-style shape):
for each round event, the event count over `F^ℓ` is bounded by
`dᵢ * |F|^(ℓ-1)`.

This is the direct finite-field root-count form typically produced by
Schwartz-Zippel style arguments before converting to probability-scaled bounds.
-/
structure FullFieldRoundEventRootCountLemmas
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
  roundCountBoundPow :
    ∀ i : Nat, i < g.inst.rounds →
      fullFieldCoinEventCount g.inst.rounds (roundFailure i) ≤
        roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1)

/--
Full-field root count of a univariate coefficient-array polynomial over `F`.

Mathlib bridge:
- map `F = Fin q` values into `ZMod q`,
- build the corresponding univariate polynomial in `ZMod q[X]`,
- count roots by filtering all `F`-challenges through polynomial evaluation.
-/
abbrev Fq : Type := ZMod Goldilocks.q

/-- Canonical coercion from `F = Fin q` into `ZMod q`. -/
def fToZMod (a : F) : Fq :=
  (a.val : Fq)

theorem fToZMod_injective : Function.Injective fToZMod := by
  intro a b hEq
  apply Fin.ext
  have hMod :
      a.val % Goldilocks.q = b.val % Goldilocks.q :=
    (ZMod.natCast_eq_natCast_iff' a.val b.val Goldilocks.q).1 hEq
  simpa [Nat.mod_eq_of_lt a.isLt, Nat.mod_eq_of_lt b.isLt] using hMod

/-- Mathlib polynomial corresponding to a coefficient array (low degree first). -/
def sumcheckPolynomialZMod (poly : Array F) : Polynomial Fq :=
  Polynomial.ofFn poly.size (fun i => fToZMod (poly[i.1]!))

/--
Polynomial that vanishes exactly on a chosen finite set of field points
(up to root multiplicity one): `∏_{r∈S} (X - r)`.
-/
noncomputable def rootVanishingPoly (S : Finset F) : Polynomial Fq :=
  S.prod (fun r => (Polynomial.X - Polynomial.C (fToZMod r)))

theorem rootVanishingPoly_eval_eq_zero_of_mem
    {S : Finset F} {r : F}
    (hr : r ∈ S) :
    (rootVanishingPoly S).eval (fToZMod r) = 0 := by
  classical
  induction S using Finset.induction_on with
  | empty =>
      cases hr
  | @insert a S ha ih =>
      simp [rootVanishingPoly, Finset.prod_insert, ha] at hr ⊢
      rcases hr with rfl | hr'
      · simp
      · right
        simpa [rootVanishingPoly] using ih hr'

theorem rootVanishingPoly_natDegree_eq_card
    (S : Finset F) :
    (rootVanishingPoly S).natDegree = S.card := by
  classical
  unfold rootVanishingPoly
  simpa using
    (Polynomial.natDegree_finset_prod_X_sub_C_eq_card
      (s := S) (f := fun r : F => fToZMod r))

theorem rootVanishingPoly_ne_zero
    (S : Finset F) :
    rootVanishingPoly S ≠ 0 := by
  classical
  have hMonic : (rootVanishingPoly S).Monic := by
    unfold rootVanishingPoly
    simpa using
      (Polynomial.monic_prod_X_sub_C (s := S) (b := fun r : F => fToZMod r))
  exact hMonic.ne_zero

/-- Canonical conversion from `ZMod q` into `F = Fin q`. -/
noncomputable def zmodToF (z : Fq) : F :=
  ⟨z.val, z.val_lt⟩

theorem fToZMod_zmodToF (z : Fq) : fToZMod (zmodToF z) = z := by
  unfold fToZMod zmodToF
  simp

/--
Truncate/pad a `ZMod q[X]` polynomial into exactly `n` coefficients in `F`.
This is the executable coefficient-array surface used by the SumCheck bridge.
-/
noncomputable def zmodPolyToCoeffArray (n : Nat) (p : Polynomial Fq) : Array F :=
  Array.ofFn (fun i : Fin n => zmodToF ((Polynomial.toFn n p) i))

@[simp] theorem zmodPolyToCoeffArray_size (n : Nat) (p : Polynomial Fq) :
    (zmodPolyToCoeffArray n p).size = n := by
  simp [zmodPolyToCoeffArray]

/--
If a `ZMod q[X]` polynomial has degree `< n`, converting it to `n` coefficients
and back through `sumcheckPolynomialZMod` is exact.
-/
theorem sumcheckPolynomialZMod_zmodPolyToCoeffArray
    (n : Nat)
    (p : Polynomial Fq)
    (hdeg : p.natDegree < n) :
    sumcheckPolynomialZMod (zmodPolyToCoeffArray n p) = p := by
  let arr : Array F := zmodPolyToCoeffArray n p
  have hSize : arr.size = n := by
    simp [arr, zmodPolyToCoeffArray]
  unfold sumcheckPolynomialZMod
  rw [hSize]
  have hfun :
      (fun i : Fin n => fToZMod (arr[i.1]!)) = Polynomial.toFn n p := by
    funext i
    simp [arr, zmodPolyToCoeffArray, fToZMod_zmodToF]
  calc
    Polynomial.ofFn n (fun i : Fin n => fToZMod (arr[i.1]!))
      = Polynomial.ofFn n (Polynomial.toFn n p) := by
          simp [hfun]
    _ = p := Polynomial.ofFn_comp_toFn_eq_id_of_natDegree_lt hdeg

/-- Root count over the full finite challenge domain using the Mathlib polynomial bridge. -/
noncomputable def fullFieldPolyRootCount (poly : Array F) : Nat :=
  (Finset.univ.filter (fun r : F =>
      (sumcheckPolynomialZMod poly).eval (fToZMod r) = 0)).card

theorem sumcheckPolynomialZMod_natDegree_lt_size
    {poly : Array F}
    (hSizePos : 0 < poly.size) :
    (sumcheckPolynomialZMod poly).natDegree < poly.size := by
  unfold sumcheckPolynomialZMod
  have hOneLe : 1 ≤ poly.size := Nat.succ_le_of_lt hSizePos
  simpa using
    (Polynomial.ofFn_natDegree_lt (R := Fq) hOneLe
      (fun i => fToZMod (poly[i.1]!)))

/--
Mathlib root-count bridge:
for nonzero bridged polynomials, counted full-field roots are bounded by
the multiset-cardinality of roots.
-/
theorem fullFieldPolyRootCount_le_card_roots
    [Fact (Nat.Prime Goldilocks.q)]
    {poly : Array F}
    (hPolyNeZero : sumcheckPolynomialZMod poly ≠ 0) :
    fullFieldPolyRootCount poly ≤ (sumcheckPolynomialZMod poly).roots.card := by
  classical
  let rootsF : Finset F :=
    Finset.univ.filter (fun r : F =>
      (sumcheckPolynomialZMod poly).eval (fToZMod r) = 0)
  have hDef : fullFieldPolyRootCount poly = rootsF.card := by
    simp [fullFieldPolyRootCount, rootsF]
  have hImageSub :
      rootsF.image fToZMod ⊆ (sumcheckPolynomialZMod poly).roots.toFinset := by
    intro z hz
    rcases Finset.mem_image.mp hz with ⟨r, hrMem, rfl⟩
    have hrEval : (sumcheckPolynomialZMod poly).eval (fToZMod r) = 0 :=
      (Finset.mem_filter.mp hrMem).2
    have hrRoot : (sumcheckPolynomialZMod poly).IsRoot (fToZMod r) := by
      simpa [Polynomial.IsRoot] using hrEval
    exact Multiset.mem_toFinset.mpr ((Polynomial.mem_roots hPolyNeZero).2 hrRoot)
  have hCardImage :
      (rootsF.image fToZMod).card = rootsF.card :=
    Finset.card_image_of_injective rootsF fToZMod_injective
  calc
    fullFieldPolyRootCount poly = rootsF.card := hDef
    _ = (rootsF.image fToZMod).card := hCardImage.symm
    _ ≤ (sumcheckPolynomialZMod poly).roots.toFinset.card := Finset.card_le_card hImageSub
    _ ≤ (sumcheckPolynomialZMod poly).roots.card :=
      Multiset.toFinset_card_le (sumcheckPolynomialZMod poly).roots

theorem fullFieldPolyRootCount_le_natDegree_of_nonzero
    [Fact (Nat.Prime Goldilocks.q)]
    {poly : Array F}
    (hPolyNeZero : sumcheckPolynomialZMod poly ≠ 0) :
    fullFieldPolyRootCount poly ≤ (sumcheckPolynomialZMod poly).natDegree := by
  calc
    fullFieldPolyRootCount poly ≤ (sumcheckPolynomialZMod poly).roots.card :=
      fullFieldPolyRootCount_le_card_roots hPolyNeZero
    _ ≤ (sumcheckPolynomialZMod poly).natDegree :=
      Polynomial.card_roots' (sumcheckPolynomialZMod poly)

theorem fullFieldPolyRootCount_le_pred_size_of_nonzero
    [Fact (Nat.Prime Goldilocks.q)]
    {poly : Array F}
    (hPolyNeZero : sumcheckPolynomialZMod poly ≠ 0)
    (hSizePos : 0 < poly.size) :
    fullFieldPolyRootCount poly ≤ poly.size - 1 := by
  have hDegLt : (sumcheckPolynomialZMod poly).natDegree < poly.size :=
    sumcheckPolynomialZMod_natDegree_lt_size (poly := poly) hSizePos
  have hDegLe : (sumcheckPolynomialZMod poly).natDegree ≤ poly.size - 1 :=
    Nat.le_pred_of_lt hDegLt
  exact Nat.le_trans
    (fullFieldPolyRootCount_le_natDegree_of_nonzero (poly := poly) hPolyNeZero)
    hDegLe

theorem fullFieldPolyRootCount_le_maxDegree_of_shape_nonzero
    [Fact (Nat.Prime Goldilocks.q)]
    {poly : Array F}
    {maxDegree : Nat}
    (hShape : poly.size = maxDegree + 1)
    (hPolyNeZero : sumcheckPolynomialZMod poly ≠ 0) :
    fullFieldPolyRootCount poly ≤ maxDegree := by
  have hSizePos : 0 < poly.size := by
    simpa [hShape]
  have hPred :
      fullFieldPolyRootCount poly ≤ poly.size - 1 :=
    fullFieldPolyRootCount_le_pred_size_of_nonzero
      (poly := poly) hPolyNeZero hSizePos
  simpa [hShape] using hPred

/--
Lower-level polynomial-root package for one game.

This captures the intended "real math" input layer:
1) each round event is controlled by a concrete polynomial witness,
2) each polynomial has a full-field root-count budget,
3) round-event counting over `F^ℓ` is bounded by the corresponding root set
   times `|F|^(ℓ-1)` (coordinate-lift counting).

From this package we can construct `FullFieldRoundEventRootCountLemmas`.
-/
structure FullFieldRoundPolynomialRootLemmas
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
  roundPoly : Nat → Array F
  /-- Event-to-root relation for each round witness polynomial (Mathlib bridge form). -/
  roundFailureImpliesPolyRoot :
    ∀ i : Nat, i < g.inst.rounds →
      ∀ coins : Array F,
        roundFailure i coins →
          (sumcheckPolynomialZMod (roundPoly i)).eval (fToZMod (coins[i]!)) = 0
  /-- Root-budget bound in the full field for each round witness polynomial. -/
  roundPolyRootCountBound :
    ∀ i : Nat, i < g.inst.rounds →
      fullFieldPolyRootCount (roundPoly i) ≤ roundRootBudget i
  /--
  Coordinate-lift counting bridge:
  event count over `F^ℓ` is bounded by root count times `|F|^(ℓ-1)`.
  -/
  roundFailureCountLePolyRoots :
    ∀ i : Nat, i < g.inst.rounds →
      fullFieldCoinEventCount g.inst.rounds (roundFailure i) ≤
        fullFieldPolyRootCount (roundPoly i) * Goldilocks.q ^ (g.inst.rounds - 1)

/--
Mathlib-root-count flavored lower-level polynomial package for one game.

This variant fixes round root budgets to `inst.maxDegree` and derives the
per-round polynomial root-count bounds from Mathlib theorems using:
- bridged polynomial nonzero proofs,
- bridged polynomial shape (`size = maxDegree + 1`).
-/
structure FullFieldRoundPolynomialRootMathlibLemmas
  (g : SoundnessGame) where
  domainAligned :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
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

/--
Constructive conversion from Mathlib-root-count flavored polynomial lemmas to
the existing `FullFieldRoundPolynomialRootLemmas` package.
-/
def FullFieldRoundPolynomialRootLemmas.of_mathlib
  (g : SoundnessGame)
  (hMathlib : FullFieldRoundPolynomialRootMathlibLemmas g) :
  FullFieldRoundPolynomialRootLemmas g := by
  refine
    { domainAligned := hMathlib.domainAligned
      roundFailure := hMathlib.roundFailure
      covered := hMathlib.covered
      roundRootBudget := fun _ => g.inst.maxDegree
      roundRootBudgetBound := ?_
      roundPoly := hMathlib.roundPoly
      roundFailureImpliesPolyRoot := hMathlib.roundFailureImpliesPolyRoot
      roundPolyRootCountBound := ?_
      roundFailureCountLePolyRoots := ?_ }
  · intro i _hi
    exact le_rfl
  · intro i hi
    have hShape : (hMathlib.roundPoly i).size = g.inst.maxDegree + 1 :=
      hMathlib.roundPolyShape i hi
    have hNz : sumcheckPolynomialZMod (hMathlib.roundPoly i) ≠ 0 :=
      hMathlib.roundPolyNonzero i hi
    simpa using
      (fullFieldPolyRootCount_le_maxDegree_of_shape_nonzero
        (poly := hMathlib.roundPoly i)
        (maxDegree := g.inst.maxDegree)
        hShape hNz)
  · intro i hi
    classical
    have hImp :
        ∀ coins, hMathlib.roundFailure i coins →
          (sumcheckPolynomialZMod (hMathlib.roundPoly i)).eval (fToZMod (coins[i]!)) = 0 := by
      intro coins hFail
      exact hMathlib.roundFailureImpliesPolyRoot i hi coins hFail
    simpa [fullFieldPolyRootCount] using
      (fullFieldCoinEventCount_le_coordPredicate
        (m := g.inst.rounds)
        (i := i)
        (hi := hi)
        (E := hMathlib.roundFailure i)
        (P := fun r : F =>
          (sumcheckPolynomialZMod (hMathlib.roundPoly i)).eval (fToZMod r) = 0)
        hImp)

/--
Constructive lift from polynomial-root lemmas to paper-style full-field
round-event root-count lemmas.
-/
def FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas
  (g : SoundnessGame)
  (hPoly : FullFieldRoundPolynomialRootLemmas g) :
  FullFieldRoundEventRootCountLemmas g := by
  refine
    { domainAligned := hPoly.domainAligned
      roundFailure := hPoly.roundFailure
      covered := hPoly.covered
      roundRootBudget := hPoly.roundRootBudget
      roundRootBudgetBound := hPoly.roundRootBudgetBound
      roundCountBoundPow := ?_ }
  intro i hi
  have hCountRoots :
      fullFieldCoinEventCount g.inst.rounds (hPoly.roundFailure i) ≤
        fullFieldPolyRootCount (hPoly.roundPoly i) * Goldilocks.q ^ (g.inst.rounds - 1) :=
    hPoly.roundFailureCountLePolyRoots i hi
  have hRootBound :
      fullFieldPolyRootCount (hPoly.roundPoly i) ≤ hPoly.roundRootBudget i :=
    hPoly.roundPolyRootCountBound i hi
  have hMul :
      fullFieldPolyRootCount (hPoly.roundPoly i) * Goldilocks.q ^ (g.inst.rounds - 1) ≤
        hPoly.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1) := by
    exact Nat.mul_le_mul_right (Goldilocks.q ^ (g.inst.rounds - 1)) hRootBound
  exact Nat.le_trans hCountRoots hMul

/--
Constructive conversion from full-field Schwartz-Zippel round-event lemmas
to count-scaled cardinality lemmas.
-/
def FullFieldRoundEventCardinalityLemmas.of_schwartzZippel
  (g : SoundnessGame)
  (hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g) :
  FullFieldRoundEventCardinalityLemmas g := by
  refine
    { roundFailure := hSz.roundFailure
      covered := hSz.covered
      roundRootBudget := hSz.roundRootBudget
      roundRootBudgetBound := hSz.roundRootBudgetBound
      roundCountBoundScaled := ?_ }
  intro i hi
  have hProb :
      fullFieldCoinPr g.inst.rounds (hSz.roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat) ≤
          (hSz.roundRootBudget i : Rat) := by
    simpa [fullFieldUniformCoinProbModel] using hSz.roundProbBoundScaled i hi
  exact fullFieldCoinEventCount_scaled_of_pr_mul_nat_le
    g.inst.rounds
    (hSz.roundFailure i)
    (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
    (hSz.roundRootBudget i)
    hProb

/--
Constructive conversion from count-scaled cardinality lemmas to paper-style
root-count lemmas for full-field games.

Requires denominator alignment `|K| = |F| = Goldilocks.q`.
-/
def FullFieldRoundEventRootCountLemmas.of_cardinality
  (g : SoundnessGame)
  (hDomain :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hCard : FullFieldRoundEventCardinalityLemmas g) :
  FullFieldRoundEventRootCountLemmas g := by
  refine
    { domainAligned := hDomain
      roundFailure := hCard.roundFailure
      covered := hCard.covered
      roundRootBudget := hCard.roundRootBudget
      roundRootBudgetBound := hCard.roundRootBudgetBound
      roundCountBoundPow := ?_ }
  intro i hi
  have hScaled :
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
          hCard.roundRootBudget i * (fullFieldCoinSpace g.inst.rounds).length :=
    hCard.roundCountBoundScaled i hi
  have hScaledQ :
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) * Goldilocks.q ≤
        hCard.roundRootBudget i * Goldilocks.q ^ g.inst.rounds := by
    calc
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) * Goldilocks.q
          = fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) *
              (SuperNeo.sumcheckLundSoundnessDenominator g.inst) := by
                simp [hDomain]
      _ ≤ hCard.roundRootBudget i * (fullFieldCoinSpace g.inst.rounds).length := hScaled
      _ = hCard.roundRootBudget i * Goldilocks.q ^ g.inst.rounds := by
            simp [fullFieldCoinSpace_length]
  have hRoundsPos : 0 < g.inst.rounds := by
    exact Nat.pos_of_ne_zero (by
      intro hZero
      simpa [hZero] using hi)
  have hPowStep :
      Goldilocks.q ^ g.inst.rounds =
        Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q := by
    have hIdx : (g.inst.rounds - 1) + 1 = g.inst.rounds := by
      omega
    calc
      Goldilocks.q ^ g.inst.rounds
          = Goldilocks.q ^ ((g.inst.rounds - 1) + 1) := by
              simp [hIdx]
      _ = Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q := by
            simp [Nat.pow_succ, Nat.mul_comm]
  have hScaledQ' :
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) * Goldilocks.q ≤
        (hCard.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1)) *
          Goldilocks.q := by
    calc
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) * Goldilocks.q
          ≤ hCard.roundRootBudget i * Goldilocks.q ^ g.inst.rounds := hScaledQ
      _ = hCard.roundRootBudget i * (Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q) := by
            rw [hPowStep]
      _ = (hCard.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1)) * Goldilocks.q := by
            simp [Nat.mul_assoc]
  exact Nat.le_of_mul_le_mul_right hScaledQ' Goldilocks.q_pos

/--
Constructive conversion from root-count bounds
`count(Eᵢ) ≤ dᵢ * |F|^(ℓ-1)` into the cross-multiplied cardinality surface
`count(Eᵢ) * |K| ≤ dᵢ * |F|^ℓ`, using `|K| = |F|`.
-/
def FullFieldRoundEventCardinalityLemmas.of_rootCount
  (g : SoundnessGame)
  (hRoot : FullFieldRoundEventRootCountLemmas g) :
  FullFieldRoundEventCardinalityLemmas g := by
  refine
    { roundFailure := hRoot.roundFailure
      covered := hRoot.covered
      roundRootBudget := hRoot.roundRootBudget
      roundRootBudgetBound := hRoot.roundRootBudgetBound
      roundCountBoundScaled := ?_ }
  intro i hi
  have hRoundsPos : 0 < g.inst.rounds := by
    exact Nat.pos_of_ne_zero (by
      intro hZero
      simpa [hZero] using hi)
  have hPowStep :
      Goldilocks.q ^ g.inst.rounds =
        Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q := by
    have hIdx : (g.inst.rounds - 1) + 1 = g.inst.rounds := by
      omega
    calc
      Goldilocks.q ^ g.inst.rounds
          = Goldilocks.q ^ ((g.inst.rounds - 1) + 1) := by
              simp [hIdx]
      _ = Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q := by
            simp [Nat.pow_succ, Nat.mul_comm]
  have hCountPow :
      fullFieldCoinEventCount g.inst.rounds (hRoot.roundFailure i) ≤
        hRoot.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1) :=
    hRoot.roundCountBoundPow i hi
  have hMul :
      fullFieldCoinEventCount g.inst.rounds (hRoot.roundFailure i) * Goldilocks.q ≤
        (hRoot.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1)) * Goldilocks.q := by
    exact Nat.mul_le_mul_right Goldilocks.q hCountPow
  calc
    fullFieldCoinEventCount g.inst.rounds (hRoot.roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
        = fullFieldCoinEventCount g.inst.rounds (hRoot.roundFailure i) * Goldilocks.q := by
            simp [hRoot.domainAligned]
    _ ≤ (hRoot.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1)) * Goldilocks.q := hMul
    _ = hRoot.roundRootBudget i * (Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q) := by
          simp [Nat.mul_assoc]
    _ = hRoot.roundRootBudget i * Goldilocks.q ^ g.inst.rounds := by
          rw [← hPowStep]
    _ = hRoot.roundRootBudget i * (fullFieldCoinSpace g.inst.rounds).length := by
          simp [fullFieldCoinSpace_length]

/-- Global lower-level closure surface in root-count form for full-field games. -/
def FullFieldRoundEventRootCountAssumption : Prop :=
  ∀ g : SoundnessGame, Nonempty (FullFieldRoundEventRootCountLemmas g)

/--
Domain-aligned variant of the full-field root-count closure surface:
for aligned games (`|K| = |F| = q`), per-game root-count lemmas exist.
-/
def FullFieldRoundEventRootCountAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      Nonempty (FullFieldRoundEventRootCountLemmas g)

/--
Global lower-level closure surface in polynomial-root form for full-field games.

This is the stronger theorem-native input that can be converted constructively
to `FullFieldRoundEventRootCountAssumption`.
-/
def FullFieldRoundPolynomialRootAssumption : Prop :=
  ∀ g : SoundnessGame, Nonempty (FullFieldRoundPolynomialRootLemmas g)

/--
Domain-aligned variant of the full-field polynomial-root closure surface:
for aligned games (`|K| = |F| = q`), per-game polynomial-root lemmas exist.
-/
def FullFieldRoundPolynomialRootAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      Nonempty (FullFieldRoundPolynomialRootLemmas g)

/--
Global lower-level closure surface in Mathlib-root-count form for full-field games.

This is a theorem-native strengthening: it carries nonzero/shape witnesses for
bridged polynomials and derives the original polynomial-root package
constructively (`FullFieldRoundPolynomialRootLemmas.of_mathlib`).
-/
def FullFieldRoundPolynomialRootMathlibAssumption : Prop :=
  ∀ g : SoundnessGame, Nonempty (FullFieldRoundPolynomialRootMathlibLemmas g)

/--
Domain-aligned variant of the full-field Mathlib-root closure surface:
for aligned games (`|K| = |F| = q`), per-game Mathlib-root lemmas exist.
-/
def FullFieldRoundPolynomialRootMathlibAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      Nonempty (FullFieldRoundPolynomialRootMathlibLemmas g)

/--
Lower-level algebraic witness package for constructing
`FullFieldRoundPolynomialRootMathlibAssumption` from full-field
Schwartz-Zippel round-event lemmas.

This intentionally separates:
- probabilistic/event coverage lemmas (`SchwartzZippelRoundEventLemmas`), and
- polynomial witness/root lemmas (this structure).
-/
structure FullFieldRoundPolynomialRootMathlibWitness
  (g : SoundnessGame)
  (hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g) where
  domainAligned :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q
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
        hSz.roundFailure i coins →
          (sumcheckPolynomialZMod (roundPoly i)).eval (fToZMod (coins[i]!)) = 0

/--
Global all-games witness assumption for constructing the Mathlib-root package
from internal probabilistic + algebraic lemmas.
-/
def FullFieldRoundPolynomialRootMathlibWitnessAssumption : Prop :=
  ∀ g : SoundnessGame,
    ∃ hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g,
      Nonempty (FullFieldRoundPolynomialRootMathlibWitness g hSz)

/--
Domain-aligned variant of the witness-layer assumption:
for aligned games (`|K| = |F| = q`), per-game SZ+witness packages exist.
-/
def FullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      ∃ hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g,
        Nonempty (FullFieldRoundPolynomialRootMathlibWitness g hSz)

end Sumcheck

end SuperNeo.ProofSystem
