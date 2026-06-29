import OpeningConvergence.Basic
import Mathlib.Algebra.Polynomial.OfFn
import Mathlib.Algebra.Polynomial.Roots

/-!
# Module 2: BatchEvalReduction — Interface

Owns the proof that Phase 1 (point unification via eta/gamma-linearization +
rho-batched sumcheck + u_i* scalar outputs) is a sound batch evaluation
reduction.

## Theorems
- Theorem 3: ClaimedSumCorrectness
- Theorem 4: CoefficientLinearization
- Theorem 5: GammaLinearization
- Theorem 5b: RhoLinearization
- Theorem 6: Phase1Soundness (split: Core + FailureBound)
- Bridge Lemma: SameObjectPayloadUniqueness

## Spec
See `specs/BatchEvalReduction.spec.md`
-/

namespace OpeningConvergence.BatchEvalReduction

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K]

/-! ## Boolean-cube bit decomposition

Convert a hypercube index (Fin 2^ell) into its bit-vector representation
(Fin ell → K). Needed for evaluating eq polynomials at Boolean points. -/

/-- Decompose a hypercube index into its bit representation over K. -/
abbrev boolCubeBits {K : Type*} [Field K]
    {ell : Nat} (x : Fin (2 ^ ell)) : Fin ell → K :=
  OpeningConvergence.boolCubeBits x

/-! ## Theorem 3: ClaimedSumCorrectness

For the batched polynomial P(x) = Σ_i ρ^i · eq(r_i, x) · g_i(x):

    Σ_{x ∈ {0,1}^ell} P(x) = Σ_i ρ^i · g_i(r_i) = Σ_i ρ^i · u_i = T

This is verifier-computable before the sumcheck begins.
-/

/-- ClaimedSumCorrectness: the verifier's claimed sum T equals the
    Boolean-hypercube sum of the batched polynomial. -/
theorem claimedSumCorrectness
    {ell N : Nat}
    (points : Fin N → (Fin ell → K))
    (scalarValues : Fin N → K)
    (gPolys : Fin N → Fin (2 ^ ell) → K)
    (rho : K)
    (hEval : ∀ i : Fin N, mleEval (gPolys i) (points i) = scalarValues i)
    :
    let T := Finset.sum Finset.univ fun i : Fin N =>
      rho ^ ((i : Nat) + 1) * scalarValues i
    let P : Fin (2 ^ ell) → K := fun x =>
      Finset.sum Finset.univ fun i : Fin N =>
        rho ^ ((i : Nat) + 1) * eqPoly (points i) (boolCubeBits x) * gPolys i x
    Finset.sum Finset.univ P = T := by
  have hEqSymm : ∀ (x y : Fin ell → K), eqPoly x y = eqPoly y x := by
    intro x y
    unfold eqPoly
    refine Finset.prod_congr rfl ?_
    intro i _
    ring
  dsimp
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl ?_
  intro i _
  calc
    ∑ x : Fin (2 ^ ell), rho ^ ((i : Nat) + 1) * eqPoly (points i) (boolCubeBits x) * gPolys i x
        = ∑ x : Fin (2 ^ ell),
            rho ^ ((i : Nat) + 1) * (gPolys i x * eqPoly (boolCubeBits x) (points i)) := by
            refine Finset.sum_congr rfl ?_
            intro x _
            rw [hEqSymm]
            ring
    _ = rho ^ ((i : Nat) + 1) *
          ∑ x : Fin (2 ^ ell), gPolys i x * eqPoly (boolCubeBits x) (points i) := by
          rw [Finset.mul_sum]
    _ = rho ^ ((i : Nat) + 1) * scalarValues i := by
          rw [show (∑ x : Fin (2 ^ ell), gPolys i x * eqPoly (boolCubeBits x) (points i)) =
              mleEval (gPolys i) (points i) by
                rfl]
          rw [← hEval i]

/-! ## Theorem 4: CoefficientLinearization

Given eta ∈ K sampled uniformly and packed column j with AJTAI_D coefficient
polynomials, the eta-consistency check

    w_i*^{(j)} = Σ_t η^t · v_i*[j].coeffs[t]

is sound with error probability at most AJTAI_D / |K|.
-/

/-- CoefficientLinearization: eta-based coefficient collapse is sound.
    If claimed coefficients differ from true evaluations, then
    wClaimed = wTrue with probability at most AJTAI_D / |K|
    (Schwartz-Zippel on degree-(D-1) poly in eta). -/
noncomputable def coefficientBadSet
    {ell : Nat}
    (coeffPolys : Fin AJTAI_D → Fin (2 ^ ell) → K)
    (rStar : Fin ell → K)
    (claimedEval : PackedColumnEval K)
    : Finset K :=
  Finset.univ.filter fun eta =>
    coeffLinearize eta claimedEval =
      Finset.sum Finset.univ fun t : Fin AJTAI_D =>
        eta ^ (t : Nat) * mleEval (coeffPolys t) rStar

theorem coefficientLinearization
    {ell : Nat}
    (coeffPolys : Fin AJTAI_D → Fin (2 ^ ell) → K)
    (rStar : Fin ell → K)
    (claimedEval : PackedColumnEval K)
    (hMismatch : ∃ t : Fin AJTAI_D,
      claimedEval.coeffs t ≠ mleEval (coeffPolys t) rStar)
    :
    ((coefficientBadSet coeffPolys rStar claimedEval).card : ℚ) /
        Fintype.card K
      ≤ (AJTAI_D : ℚ) / Fintype.card K := by
  classical
  let trueCoeffs : Fin AJTAI_D → K := fun t => mleEval (coeffPolys t) rStar
  let diff : Fin AJTAI_D → K := fun t => claimedEval.coeffs t - trueCoeffs t
  let q : Polynomial K := Polynomial.ofFn AJTAI_D diff
  have hDiffNe : diff ≠ 0 := by
    intro hZero
    rcases hMismatch with ⟨t, ht⟩
    have hAtT := congrFun hZero t
    simp [diff, trueCoeffs] at hAtT
    exact ht (sub_eq_zero.mp hAtT)
  have hqNe : q ≠ 0 := by
    intro hq
    have hq' : Polynomial.ofFn AJTAI_D diff = Polynomial.ofFn AJTAI_D 0 := by
      simpa [q]
    exact hDiffNe ((Polynomial.injective_ofFn AJTAI_D) hq')
  have hOfFnEval :
      ∀ (f : Fin AJTAI_D → K) (eta : K),
        (Polynomial.ofFn AJTAI_D f).eval eta =
          Finset.sum Finset.univ fun t : Fin AJTAI_D => eta ^ (t : Nat) * f t := by
    intro f eta
    rw [Polynomial.ofFn_eq_sum_monomial]
    rw [Polynomial.eval_finset_sum]
    simp [Polynomial.eval_monomial, mul_comm]
  have hqEval :
      ∀ eta : K,
        q.eval eta =
          coeffLinearize eta claimedEval -
            Finset.sum Finset.univ fun t : Fin AJTAI_D =>
              eta ^ (t : Nat) * mleEval (coeffPolys t) rStar := by
    intro eta
    rw [show q = Polynomial.ofFn AJTAI_D diff by rfl, hOfFnEval diff eta]
    simp [diff, trueCoeffs, coeffLinearize, Finset.sum_sub_distrib, mul_sub]
  have hSubset : coefficientBadSet coeffPolys rStar claimedEval ⊆ q.roots.toFinset := by
    intro eta hEta
    simp only [coefficientBadSet, Finset.mem_filter, Finset.mem_univ, true_and] at hEta
    have hRoot : q.eval eta = 0 := by
      rw [hqEval eta]
      exact sub_eq_zero.mpr hEta
    exact Multiset.mem_toFinset.mpr ((Polynomial.mem_roots hqNe).2 hRoot)
  have hCardLt : (coefficientBadSet coeffPolys rStar claimedEval).card < AJTAI_D := by
    calc
      (coefficientBadSet coeffPolys rStar claimedEval).card
          ≤ q.roots.toFinset.card := Finset.card_le_card hSubset
      _ ≤ q.roots.card := Multiset.toFinset_card_le _
      _ ≤ q.natDegree := Polynomial.card_roots' q
      _ < AJTAI_D := by
        have hDOne : 1 ≤ AJTAI_D := by native_decide
        exact Polynomial.ofFn_natDegree_lt hDOne diff
  have hCardNat : (coefficientBadSet coeffPolys rStar claimedEval).card ≤ AJTAI_D := hCardLt.le
  have hCardRat : ((coefficientBadSet coeffPolys rStar claimedEval).card : ℚ) ≤ (AJTAI_D : ℚ) := by
    exact_mod_cast hCardNat
  have hCardPos : (0 : ℚ) < (↑(Fintype.card K) : ℚ) := by
    exact_mod_cast Fintype.card_pos_iff.mpr inferInstance
  have hInvNonneg : 0 ≤ (↑(Fintype.card K) : ℚ)⁻¹ := inv_nonneg.mpr hCardPos.le
  rw [div_eq_mul_inv, div_eq_mul_inv]
  exact mul_le_mul_of_nonneg_right hCardRat hInvNonneg

/-! ## Theorem 5: GammaLinearization

For m > 1, the gamma-consistency check

    u_i* = Σ_j γ^j · w_i*^{(j)}

is sound with error probability at most m / |K|.
-/

/-- GammaLinearization: gamma-based column collapse is sound.
    If any column disagrees, then uClaimed = uTrue with probability
    at most m / |K| (Schwartz-Zippel on degree-(m-1) poly in gamma). -/
noncomputable def gammaBadSet
    {m : Nat}
    (wsClaimed wsTrue : Fin m → K)
    : Finset K :=
  Finset.univ.filter fun gamma =>
    gammaLinearize gamma wsClaimed = gammaLinearize gamma wsTrue

theorem gammaLinearization
    {m : Nat} (hm : m > 1)
    (wsClaimed : Fin m → K)
    (wsTrue : Fin m → K)
    (hMismatch : ∃ j : Fin m, wsClaimed j ≠ wsTrue j)
    :
    ((gammaBadSet wsClaimed wsTrue).card : ℚ) / Fintype.card K
      ≤ (m : ℚ) / Fintype.card K := by
  classical
  let diff : Fin m → K := fun j => wsClaimed j - wsTrue j
  let q : Polynomial K := Polynomial.ofFn m diff
  have hDiffNe : diff ≠ 0 := by
    intro hZero
    rcases hMismatch with ⟨j, hj⟩
    have hAtJ := congrFun hZero j
    simp [diff] at hAtJ
    exact hj (sub_eq_zero.mp hAtJ)
  have hqNe : q ≠ 0 := by
    intro hq
    have hq' : Polynomial.ofFn m diff = Polynomial.ofFn m 0 := by
      simpa [q]
    exact hDiffNe ((Polynomial.injective_ofFn m) hq')
  have hOfFnEval :
      ∀ (f : Fin m → K) (gamma : K),
        (Polynomial.ofFn m f).eval gamma = gammaLinearize gamma f := by
    intro f gamma
    rw [Polynomial.ofFn_eq_sum_monomial]
    rw [Polynomial.eval_finset_sum]
    simp [gammaLinearize, Polynomial.eval_monomial, mul_comm]
  have hqEval :
      ∀ gamma : K,
        q.eval gamma = gammaLinearize gamma wsClaimed - gammaLinearize gamma wsTrue := by
    intro gamma
    rw [show q = Polynomial.ofFn m diff by rfl, hOfFnEval diff gamma]
    simp [diff, gammaLinearize, Finset.sum_sub_distrib, mul_sub]
  have hSubset : gammaBadSet wsClaimed wsTrue ⊆ q.roots.toFinset := by
    intro gamma hGamma
    simp only [gammaBadSet, Finset.mem_filter, Finset.mem_univ, true_and] at hGamma
    have hRoot : q.eval gamma = 0 := by
      rw [hqEval gamma]
      exact sub_eq_zero.mpr hGamma
    exact Multiset.mem_toFinset.mpr ((Polynomial.mem_roots hqNe).2 hRoot)
  have hCardLt : (gammaBadSet wsClaimed wsTrue).card < m := by
    calc
      (gammaBadSet wsClaimed wsTrue).card
          ≤ q.roots.toFinset.card := Finset.card_le_card hSubset
      _ ≤ q.roots.card := Multiset.toFinset_card_le _
      _ ≤ q.natDegree := Polynomial.card_roots' q
      _ < m := by
        have hm1 : 1 ≤ m := Nat.le_of_lt hm
        exact Polynomial.ofFn_natDegree_lt hm1 diff
  have hCardNat : (gammaBadSet wsClaimed wsTrue).card ≤ m := hCardLt.le
  have hCardRat : ((gammaBadSet wsClaimed wsTrue).card : ℚ) ≤ (m : ℚ) := by
    exact_mod_cast hCardNat
  have hCardPos : (0 : ℚ) < (↑(Fintype.card K) : ℚ) := by
    exact_mod_cast Fintype.card_pos_iff.mpr inferInstance
  have hInvNonneg : 0 ≤ (↑(Fintype.card K) : ℚ)⁻¹ := inv_nonneg.mpr hCardPos.le
  rw [div_eq_mul_inv, div_eq_mul_inv]
  exact mul_le_mul_of_nonneg_right hCardRat hInvNonneg

/-! ## Theorem 5b: RhoLinearization

For the final Phase 1 batching across claims, the rho-consistency check

    Σ_i ρ^i · claimed_i = Σ_i ρ^i · true_i

is sound with error probability at most N / |K|.
-/

/-- RhoLinearization: rho-based claim collapse is sound.
    If any weighted claim disagrees, then the rho-linearized values agree
    only on a bounded bad challenge set of size at most N. -/
noncomputable def rhoBadSet
    {N : Nat}
    (claimedWeighted trueWeighted : Fin N → K)
    : Finset K :=
  Finset.univ.filter fun rho =>
    Finset.sum Finset.univ (fun i : Fin N => rho ^ ((i : Nat) + 1) * claimedWeighted i) =
      Finset.sum Finset.univ (fun i : Fin N => rho ^ ((i : Nat) + 1) * trueWeighted i)

theorem rhoLinearization
    {N : Nat}
    (claimedWeighted trueWeighted : Fin N → K)
    (hMismatch : ∃ i : Fin N, claimedWeighted i ≠ trueWeighted i)
    :
    ((rhoBadSet claimedWeighted trueWeighted).card : ℚ) / Fintype.card K
      ≤ (N : ℚ) / Fintype.card K := by
  classical
  let diff : Fin N → K := fun i => claimedWeighted i - trueWeighted i
  let q : Polynomial K := Polynomial.ofFn N diff
  have hDiffNe : diff ≠ 0 := by
    intro hZero
    rcases hMismatch with ⟨i, hi⟩
    have hAtI := congrFun hZero i
    simp [diff] at hAtI
    exact hi (sub_eq_zero.mp hAtI)
  have hNPos : 0 < N := by
    rcases hMismatch with ⟨i, _⟩
    exact lt_of_le_of_lt (Nat.zero_le _) i.isLt
  have hqNe : q ≠ 0 := by
    intro hq
    have hq' : Polynomial.ofFn N diff = Polynomial.ofFn N 0 := by
      simpa [q]
    exact hDiffNe ((Polynomial.injective_ofFn N) hq')
  have hOfFnEval :
      ∀ (f : Fin N → K) (rho : K),
        (Polynomial.ofFn N f).eval rho =
          Finset.sum Finset.univ fun i : Fin N => rho ^ (i : Nat) * f i := by
    intro f rho
    rw [Polynomial.ofFn_eq_sum_monomial]
    rw [Polynomial.eval_finset_sum]
    simp [Polynomial.eval_monomial, mul_comm]
  have hqEval :
      ∀ rho : K,
        rho * q.eval rho =
          Finset.sum Finset.univ (fun i : Fin N => rho ^ ((i : Nat) + 1) * claimedWeighted i) -
            Finset.sum Finset.univ (fun i : Fin N => rho ^ ((i : Nat) + 1) * trueWeighted i) := by
    intro rho
    rw [show q = Polynomial.ofFn N diff by rfl, hOfFnEval diff rho]
    rw [Finset.mul_sum]
    simp [diff, Finset.sum_sub_distrib, mul_sub, pow_succ, mul_left_comm, mul_comm]
  have hSubset : rhoBadSet claimedWeighted trueWeighted ⊆ insert 0 q.roots.toFinset := by
    intro rho hRho
    simp only [rhoBadSet, Finset.mem_filter, Finset.mem_univ, true_and] at hRho
    by_cases hZero : rho = 0
    · exact Finset.mem_insert.mpr (Or.inl hZero)
    · apply Finset.mem_insert.mpr <| Or.inr ?_
      have hMulZero : rho * q.eval rho = 0 := by
        rw [hqEval rho]
        exact sub_eq_zero.mpr hRho
      have hRoot : q.eval rho = 0 := by
        rcases mul_eq_zero.mp hMulZero with hRhoZero | hEvalZero
        · exact (hZero hRhoZero).elim
        · exact hEvalZero
      exact Multiset.mem_toFinset.mpr ((Polynomial.mem_roots hqNe).2 hRoot)
  have hCardLt : (rhoBadSet claimedWeighted trueWeighted).card < N + 1 := by
    calc
      (rhoBadSet claimedWeighted trueWeighted).card
          ≤ (insert 0 q.roots.toFinset).card := Finset.card_le_card hSubset
      _ ≤ q.roots.toFinset.card + 1 := Finset.card_insert_le 0 (q.roots.toFinset)
      _ ≤ q.roots.card + 1 := by gcongr; exact Multiset.toFinset_card_le _
      _ ≤ q.natDegree + 1 := by gcongr; exact Polynomial.card_roots' q
      _ < N + 1 := by
        exact Nat.succ_lt_succ (Polynomial.ofFn_natDegree_lt (Nat.succ_le_of_lt hNPos) diff)
  have hCardNat : (rhoBadSet claimedWeighted trueWeighted).card ≤ N := by omega
  have hCardRat : ((rhoBadSet claimedWeighted trueWeighted).card : ℚ) ≤ (N : ℚ) := by
    exact_mod_cast hCardNat
  have hCardPos : (0 : ℚ) < (↑(Fintype.card K) : ℚ) := by
    exact_mod_cast Fintype.card_pos_iff.mpr inferInstance
  have hInvNonneg : 0 ≤ (↑(Fintype.card K) : ℚ)⁻¹ := inv_nonneg.mpr hCardPos.le
  rw [div_eq_mul_inv, div_eq_mul_inv]
  exact mul_le_mul_of_nonneg_right hCardRat hInvNonneg

/-- Gamma linearization on a singleton vector is the unique entry. -/
theorem gammaLinearize_singleton
    (gamma : K) (ws : Fin 1 → K) :
    gammaLinearize gamma ws = ws ⟨0, by omega⟩ := by
  simp [gammaLinearize]

/-- Gamma linearization for any length known to be 1. -/
theorem gammaLinearize_eq_of_one
    {m : Nat} (h : m = 1)
    (gamma : K) (ws : Fin m → K) :
    gammaLinearize gamma ws = ws ⟨0, by omega⟩ := by
  subst h
  simpa using gammaLinearize_singleton (K := K) gamma ws

/-! ## Theorem 6: Phase1Soundness

If the verifier accepts all Phase 1 checks:
1. Sumcheck verification passes
2. Combined scalar check: terminal = Σ_i ρ^i · eq(r_i, r*) · u_i*
3. Per-claim eta/gamma-consistency

then with probability ≥ 1 - ε:
    ∀ i, j, t: f_i^{(j,t)}(r*) = v_i*[j].coeffs[t]

where ε = 2·ell/|K| + N·ell/|K| + AJTAI_D/|K| + m/|K| + N/|K|

r* is the verifier-derived point from the transcript, not prover-chosen.
-/

/-- The Phase 1 soundness error bound. -/
noncomputable def phase1ErrorBound (K : Type*) [Fintype K]
    (ell N m : Nat) : ℚ :=
  (2 * ell + N * ell + AJTAI_D + m + N : ℚ) / Fintype.card K

/-- Phase 1 carried data plus the explicit checked equalities needed for the
    paper-faithful deterministic core. -/
structure Phase1Accepted (K : Type*) [Field K] (ell N : Nat) where
  openedObjects : Fin N → OpenedObjectId
  points : Fin N → (Fin ell → K)
  payloads : Fin N → FamilyEvalPayload K
  eta : K
  gamma : K
  rho : K
  rStar : Fin ell → K
  terminalSumcheckValue : K
  unifiedPayloads : Fin N → FamilyEvalPayload K
  wStar : (i : Fin N) → Fin (packedColumnCount (unifiedPayloads i).schema) → K
  uStar : Fin N → K
  trueCoeffPolys :
    (i : Fin N) →
      (j : Fin (packedColumnCount (unifiedPayloads i).schema)) →
      Fin AJTAI_D → Fin (2 ^ ell) → K
  etaConsistency :
    ∀ i : Fin N,
      ∀ j : Fin (packedColumnCount (unifiedPayloads i).schema),
        wStar i j = coeffLinearize eta ((unifiedPayloads i).columnEvals j)
  gammaConsistency :
    ∀ i : Fin N,
      uStar i =
        let m := packedColumnCount (unifiedPayloads i).schema
        let ws : Fin m → K := wStar i
        if h : m = 1 then
          ws ⟨0, by omega⟩
        else
          gammaLinearize gamma ws
  rhoConsistency :
    terminalSumcheckValue =
      Finset.sum Finset.univ fun i : Fin N =>
        rho ^ ((i : Nat) + 1) * eqPoly (points i) rStar * uStar i

/-- True eta-linearized value for one claim/column at the verifier-derived point. -/
noncomputable def trueColumnLinearized
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (i : Fin N)
    (j : Fin (packedColumnCount (accepted.unifiedPayloads i).schema)) : K :=
  Finset.sum Finset.univ fun t : Fin AJTAI_D =>
    accepted.eta ^ (t : Nat) * mleEval (accepted.trueCoeffPolys i j t) accepted.rStar

/-- True gamma-linearized claim scalar at the verifier-derived point. -/
noncomputable def trueClaimLinearized
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (i : Fin N) : K :=
  let m := packedColumnCount (accepted.unifiedPayloads i).schema
  let ws : Fin m → K := fun j => trueColumnLinearized accepted i j
  if h : m = 1 then
    ws ⟨0, by omega⟩
  else
    gammaLinearize accepted.gamma ws

/-- The eq-polynomial weight tying claim `i`'s original point to the
    verifier-derived unified point `rStar`. -/
noncomputable def pointWeight
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (i : Fin N) : K :=
  eqPoly (accepted.points i) accepted.rStar

/-- Sumcheck/no-eq bad events excluded: the terminal value is the true
    rho-weighted claim scalar sum at `rStar`. -/
def sumcheckTerminalCorrect
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) : Prop :=
  accepted.terminalSumcheckValue =
    Finset.sum Finset.univ fun i : Fin N =>
      accepted.rho ^ ((i : Nat) + 1) * pointWeight accepted i * trueClaimLinearized accepted i

/-- Phase 1 semantic correctness: every coefficient in every unified payload
    equals the true coefficient polynomial evaluated at the verifier-derived
    unified point `rStar`. -/
def Phase1UnifiedPayloadCorrect
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) : Prop :=
  ∀ i : Fin N,
    ∀ j : Fin (packedColumnCount (accepted.unifiedPayloads i).schema),
      ∀ t : Fin AJTAI_D,
        ((accepted.unifiedPayloads i).columnEvals j).coeffs t =
          mleEval (accepted.trueCoeffPolys i j t) accepted.rStar

/-- The true packed payload for claim `i` at the verifier-derived point `rStar`,
    reconstructed from the coefficient polynomial family. -/
noncomputable def truePayloadAt
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (i : Fin N) : FamilyEvalPayload K where
  schema := (accepted.unifiedPayloads i).schema
  columnEvals := fun j =>
    { coeffs := fun t => mleEval (accepted.trueCoeffPolys i j t) accepted.rStar }

/-- Eta challenge avoids every mismatch bad set for this accepted instance. -/
def etaAvoidsMismatchBadSets
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) : Prop :=
  ∀ i : Fin N,
    ∀ j : Fin (packedColumnCount (accepted.unifiedPayloads i).schema),
      ∀ _hMismatch : ∃ t : Fin AJTAI_D,
        ((accepted.unifiedPayloads i).columnEvals j).coeffs t ≠
          mleEval (accepted.trueCoeffPolys i j t) accepted.rStar,
        accepted.eta ∉ coefficientBadSet (accepted.trueCoeffPolys i j) accepted.rStar
          ((accepted.unifiedPayloads i).columnEvals j)

/-- Gamma challenge avoids every mismatch bad set for this accepted instance. -/
def gammaAvoidsMismatchBadSets
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) : Prop :=
  ∀ i : Fin N,
    let m := packedColumnCount (accepted.unifiedPayloads i).schema
    let wsClaimed : Fin m → K := accepted.wStar i
    let wsTrue : Fin m → K := fun j => trueColumnLinearized accepted i j
    ∀ _hMismatch : ∃ j : Fin m, wsClaimed j ≠ wsTrue j,
      accepted.gamma ∉ gammaBadSet wsClaimed wsTrue

/-- Rho challenge avoids every mismatch bad set for this accepted instance. -/
def rhoAvoidsMismatchBadSets
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) : Prop :=
  let claimed : Fin N → K := fun i => pointWeight accepted i * accepted.uStar i
  let trueVals : Fin N → K := fun i => pointWeight accepted i * trueClaimLinearized accepted i
  ∀ _hMismatch : ∃ i : Fin N, claimed i ≠ trueVals i,
    accepted.rho ∉ rhoBadSet claimed trueVals

/-- The verifier-derived point does not zero out any claim weight. This is the
    deterministic side-condition corresponding to the frozen zero-weight escape
    term in the Phase 1 error bound. -/
def zeroWeightAvoidsEscape
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N) : Prop :=
  ∀ i : Fin N, pointWeight accepted i ≠ 0

/-- Bookkeeping upper bound for the sumcheck soundness-failure contribution. -/
noncomputable def phase1SumcheckFailureBound
    (K : Type*) [Fintype K] (ell : Nat) : ℚ :=
  (2 * ell : ℚ) / Fintype.card K

/-- Bookkeeping upper bound for the zero-weight escape contribution. -/
noncomputable def phase1ZeroWeightFailureBound
    (K : Type*) [Fintype K] (ell N : Nat) : ℚ :=
  (N * ell : ℚ) / Fintype.card K

/-- Bookkeeping upper bound for the eta-linearization contribution. -/
noncomputable def phase1EtaFailureBound
    (K : Type*) [Fintype K] : ℚ :=
  (AJTAI_D : ℚ) / Fintype.card K

/-- Bookkeeping upper bound for the gamma-linearization contribution. -/
noncomputable def phase1GammaFailureBound
    (K : Type*) [Fintype K] (m : Nat) : ℚ :=
  (m : ℚ) / Fintype.card K

/-- Bookkeeping upper bound for the rho-linearization contribution. -/
noncomputable def phase1RhoFailureBound
    (K : Type*) [Fintype K] (N : Nat) : ℚ :=
  (N : ℚ) / Fintype.card K

/-- The current Phase 1 bad-event bookkeeping quantity.
    This is the frozen union-bound expression over the five failure
    components carried by the design, not yet an exact probability space
    construction over transcript coins. -/
noncomputable def phase1BadEventProbability
    {ell N : Nat}
    (_accepted : Phase1Accepted K ell N) (m : Nat) : ℚ :=
  phase1SumcheckFailureBound K ell +
    phase1ZeroWeightFailureBound K ell N +
    phase1EtaFailureBound K +
    phase1GammaFailureBound K m +
    phase1RhoFailureBound K N

/-- Phase1SoundnessCore (deterministic): if the explicit verifier equalities hold
    and eta/gamma/rho avoid every mismatch bad set, then the unified payloads
    are coefficient-correct at `rStar`. -/
theorem phase1SoundnessCore
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (hSumcheck : sumcheckTerminalCorrect accepted)
    (hNoZero : zeroWeightAvoidsEscape accepted)
    (hEtaGood : etaAvoidsMismatchBadSets accepted)
    (hGammaGood : gammaAvoidsMismatchBadSets accepted)
    (hRhoGood : rhoAvoidsMismatchBadSets accepted)
    :
    Phase1UnifiedPayloadCorrect accepted := by
  have hClaimScalarEq : ∀ i : Fin N, accepted.uStar i = trueClaimLinearized accepted i := by
    intro i
    by_contra hNe
    have hWeightedNe :
        pointWeight accepted i * accepted.uStar i ≠
          pointWeight accepted i * trueClaimLinearized accepted i := by
      intro hEq
      exact hNe ((mul_left_cancel₀ (hNoZero i)) hEq)
    have hNotIn :
        accepted.rho ∉ rhoBadSet
          (fun i => pointWeight accepted i * accepted.uStar i)
          (fun i => pointWeight accepted i * trueClaimLinearized accepted i) := by
      simpa [rhoAvoidsMismatchBadSets] using hRhoGood ⟨i, hWeightedNe⟩
    have hIn :
        accepted.rho ∈ rhoBadSet
          (fun i => pointWeight accepted i * accepted.uStar i)
          (fun i => pointWeight accepted i * trueClaimLinearized accepted i) := by
      have hEqSums :
          (∑ i : Fin N, accepted.rho ^ ((i : Nat) + 1) *
            (pointWeight accepted i * accepted.uStar i)) =
            ∑ i : Fin N, accepted.rho ^ ((i : Nat) + 1) *
              (pointWeight accepted i * trueClaimLinearized accepted i) := by
        calc
          ∑ i : Fin N, accepted.rho ^ ((i : Nat) + 1) *
            (pointWeight accepted i * accepted.uStar i)
              = accepted.terminalSumcheckValue := by
                  simpa [pointWeight, mul_assoc] using accepted.rhoConsistency.symm
          _ = ∑ i : Fin N, accepted.rho ^ ((i : Nat) + 1) *
            (pointWeight accepted i * trueClaimLinearized accepted i) := by
              simpa [sumcheckTerminalCorrect, pointWeight, mul_assoc] using hSumcheck
      simpa [rhoBadSet] using hEqSums
    exact hNotIn hIn
  have hColumnScalarEq :
      ∀ i : Fin N,
        ∀ j : Fin (packedColumnCount (accepted.unifiedPayloads i).schema),
          accepted.wStar i j = trueColumnLinearized accepted i j := by
    intro i j
    by_contra hNe
    let wsClaimed : Fin (packedColumnCount (accepted.unifiedPayloads i).schema) → K := accepted.wStar i
    let wsTrue : Fin (packedColumnCount (accepted.unifiedPayloads i).schema) → K :=
      fun j => trueColumnLinearized accepted i j
    have hNotIn : accepted.gamma ∉ gammaBadSet wsClaimed wsTrue := by
      simpa [gammaAvoidsMismatchBadSets, wsClaimed, wsTrue] using hGammaGood i ⟨j, hNe⟩
    have hIn : accepted.gamma ∈ gammaBadSet wsClaimed wsTrue := by
      have hGammaEq :
          gammaLinearize accepted.gamma wsClaimed = gammaLinearize accepted.gamma wsTrue := by
        by_cases hOne : packedColumnCount (accepted.unifiedPayloads i).schema = 1
        · have hClaimed :
              accepted.uStar i = wsClaimed ⟨0, by omega⟩ := by
            simpa [wsClaimed, hOne] using accepted.gammaConsistency i
          have hTrue :
              trueClaimLinearized accepted i = wsTrue ⟨0, by omega⟩ := by
            simpa [trueClaimLinearized, wsTrue, hOne] using rfl
          have hHead :
              wsClaimed ⟨0, by omega⟩ = wsTrue ⟨0, by omega⟩ := by
            calc
              wsClaimed ⟨0, by omega⟩ = accepted.uStar i := hClaimed.symm
              _ = trueClaimLinearized accepted i := hClaimScalarEq i
              _ = wsTrue ⟨0, by omega⟩ := hTrue
          calc
            gammaLinearize accepted.gamma wsClaimed = wsClaimed ⟨0, by omega⟩ := by
              exact gammaLinearize_eq_of_one (K := K) hOne accepted.gamma wsClaimed
            _ = wsTrue ⟨0, by omega⟩ := hHead
            _ = gammaLinearize accepted.gamma wsTrue := by
              symm
              exact gammaLinearize_eq_of_one (K := K) hOne accepted.gamma wsTrue
        · have hClaimed :
              accepted.uStar i = gammaLinearize accepted.gamma wsClaimed := by
            simpa [wsClaimed, hOne] using accepted.gammaConsistency i
          have hTrue :
              trueClaimLinearized accepted i = gammaLinearize accepted.gamma wsTrue := by
            simp [trueClaimLinearized, wsTrue, hOne]
          calc
            gammaLinearize accepted.gamma wsClaimed = accepted.uStar i := hClaimed.symm
            _ = trueClaimLinearized accepted i := hClaimScalarEq i
            _ = gammaLinearize accepted.gamma wsTrue := hTrue
      simpa [gammaBadSet, wsClaimed, wsTrue] using hGammaEq
    exact hNotIn hIn
  intro i j t
  by_contra hNe
  have hNotIn :
      accepted.eta ∉ coefficientBadSet (accepted.trueCoeffPolys i j) accepted.rStar
        ((accepted.unifiedPayloads i).columnEvals j) := by
    simpa [etaAvoidsMismatchBadSets] using hEtaGood i j ⟨t, hNe⟩
  have hIn :
      accepted.eta ∈ coefficientBadSet (accepted.trueCoeffPolys i j) accepted.rStar
        ((accepted.unifiedPayloads i).columnEvals j) := by
    have hCoeffEq :
        coeffLinearize accepted.eta ((accepted.unifiedPayloads i).columnEvals j) =
          ∑ t : Fin AJTAI_D, accepted.eta ^ (t : Nat) *
            mleEval (accepted.trueCoeffPolys i j t) accepted.rStar := by
      calc
        coeffLinearize accepted.eta ((accepted.unifiedPayloads i).columnEvals j)
            = accepted.wStar i j := (accepted.etaConsistency i j).symm
        _ = trueColumnLinearized accepted i j := hColumnScalarEq i j
        _ = ∑ t : Fin AJTAI_D, accepted.eta ^ (t : Nat) *
              mleEval (accepted.trueCoeffPolys i j t) accepted.rStar := rfl
    simpa [coefficientBadSet] using hCoeffEq
  exact hNotIn hIn

/-- Phase1SoundnessFailureBound (probabilistic): the probability of any
    bad event is bounded by ε. -/
theorem phase1SoundnessFailureBound
    {ell N m : Nat}
    (accepted : Phase1Accepted K ell N)
    (hCard : Fintype.card K ≥ MIN_FIELD_CARD)
    :
    phase1BadEventProbability accepted m ≤ phase1ErrorBound K ell N m := by
  have _hCard := hCard
  unfold phase1BadEventProbability phase1ErrorBound
  unfold phase1SumcheckFailureBound phase1ZeroWeightFailureBound
  unfold phase1EtaFailureBound phase1GammaFailureBound phase1RhoFailureBound
  ring_nf
  exact le_rfl

/-- Phase1Soundness: the composition. If Phase 1 verifier accepts, then
    all unified claims are true evaluation claims at r* with probability
    ≥ 1 - ε. -/
theorem phase1Soundness
    {ell N m : Nat}
    (accepted : Phase1Accepted K ell N)
    (hCard : Fintype.card K ≥ MIN_FIELD_CARD)
    (hSumcheck : sumcheckTerminalCorrect accepted)
    (hNoZero : zeroWeightAvoidsEscape accepted)
    (hEtaGood : etaAvoidsMismatchBadSets accepted)
    (hGammaGood : gammaAvoidsMismatchBadSets accepted)
    (hRhoGood : rhoAvoidsMismatchBadSets accepted)
    :
    Phase1UnifiedPayloadCorrect accepted ∧
      phase1BadEventProbability accepted m ≤ phase1ErrorBound K ell N m := by
  exact ⟨
    phase1SoundnessCore accepted hSumcheck hNoZero hEtaGood hGammaGood hRhoGood,
    phase1SoundnessFailureBound accepted hCard
  ⟩

/-! ## Bridge Lemma: SameObjectPayloadUniqueness

If two accepted unified claims share the same opened object and point,
then their payloads are equal. -/

/-- SameObjectPayloadUniqueness: same object + unique-object functionality
    gives the same payload after Phase 1 soundness is applied. -/
theorem sameObjectPayloadUniqueness
    {ell N : Nat}
    (accepted : Phase1Accepted K ell N)
    (a b : Fin N)
    (hSound : Phase1UnifiedPayloadCorrect accepted)
    (hObj : accepted.openedObjects a = accepted.openedObjects b)
    (hFunctional :
      accepted.openedObjects a = accepted.openedObjects b →
        truePayloadAt accepted a = truePayloadAt accepted b)
    :
    accepted.unifiedPayloads a = accepted.unifiedPayloads b := by
  have hUnifiedIsTrue :
      ∀ i : Fin N, accepted.unifiedPayloads i = truePayloadAt accepted i := by
    intro i
    have hColsEq :
        ∀ j : Fin (packedColumnCount (accepted.unifiedPayloads i).schema),
          (accepted.unifiedPayloads i).columnEvals j =
            (truePayloadAt accepted i).columnEvals j := by
      intro j
      apply PackedColumnEval.ext
      funext t
      simpa [truePayloadAt] using hSound i j t
    refine FamilyEvalPayload.ext
      (x := accepted.unifiedPayloads i)
      (y := truePayloadAt accepted i)
      rfl
      ?_
    exact heq_of_eq (funext hColsEq)
  calc
    accepted.unifiedPayloads a = truePayloadAt accepted a := hUnifiedIsTrue a
    _ = truePayloadAt accepted b := hFunctional hObj
    _ = accepted.unifiedPayloads b := (hUnifiedIsTrue b).symm

end OpeningConvergence.BatchEvalReduction
