import NightstreamFPrime.Spec.Compression.Verifier.Sum

/-! SPEC.md section 6. -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K]

local notation "ι" => algebraMap F K

section General

theorem density_biUnion_le {I C : Type} [Fintype C] [DecidableEq C] (s : Finset I)
    (S : I → Finset C) : density (s.biUnion S) ≤ ∑ i ∈ s, density (S i) := by
  unfold density
  rw [← sum_div]
  exact div_le_div_of_nonneg_right (by exact_mod_cast card_biUnion_le) (by positivity)

theorem density_univ_one {C : Type} [Fintype C] [Nonempty C] : density (univ : Finset C) = 1 := by
  simp [density, Fintype.card_ne_zero]

/-- `(Fin v → C) × (Fin w → C) ≃ (Fin (v + w) → C)`. -/
def appendEquiv (v w : ℕ) (C : Type) : (Fin v → C) × (Fin w → C) ≃ (Fin (v + w) → C) where
  toFun p := Fin.append p.1 p.2
  invFun f := (fun i => f (Fin.castAdd w i), fun i => f (Fin.natAdd v i))
  left_inv p := by simp
  right_inv f := by simp [Fin.append_castAdd_natAdd]

/-- A set of points that depends only on the first `v` coordinates has the density of its
projection. -/
theorem density_take_le (n v : ℕ) (le : v ≤ n) (Q : List K → Prop) [DecidablePred Q] (ε : ℝ)
    (small : density (univ.filter fun β : Fin v → K => Q (List.ofFn β)) ≤ ε) :
    density (univ.filter fun β : Fin n → K => Q ((List.ofFn β).take v)) ≤ ε := by
  obtain ⟨w, rfl⟩ := Nat.exists_eq_add_of_le le
  rw [density_congr (appendEquiv v w K).symm _
    ((univ.filter fun β : Fin v → K => Q (List.ofFn β)) ×ˢ univ) (by
      intro β
      have : β = Fin.append (fun i => β (Fin.castAdd w i)) (fun i => β (Fin.natAdd v i)) := by
        simp [Fin.append_castAdd_natAdd]
      simp only [mem_filter, mem_univ, true_and, mem_product, and_true]
      conv_lhs => rw [this, List.ofFn_fin_append, List.take_left' (by simp)]
      rfl), density_product, density_univ_one, mul_one]
  exact small

end General

section Draws

variable (P : Params F K) (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)

/-- The bad lookup challenges: for a claim with a table, each challenge is a root of its lookup
polynomial. -/
noncomputable def tauBad : Finset (Fin P.lookups → F) :=
  univ.biUnion fun c => if (P.shape c).lookup then
    lookupBad P.lookups (2 ^ (P.shape c).witness.vars) (2 * (P.shape c).bound + 1)
      (fun y => W ((P.shape c).witness.offset + y)) (tableValue (P.shape c).bound)
      (fun v => W ((P.shape c).counts.offset + v))
  else ∅

/-- The term of `tauBad` in the error: `Σ_tables ((N + T) / |F|)^L`. -/
noncomputable def tauError : ℝ :=
  ∑ c : Fin P.claims, if (P.shape c).lookup then
    (((2 ^ (P.shape c).witness.vars + (2 * (P.shape c).bound + 1) : ℕ) : ℝ) /
      Fintype.card F) ^ P.lookups
  else 0

omit [Field K] [Fintype K] [DecidableEq K] [Algebra F K] in
theorem density_tauBad_le : density (tauBad P W) ≤ tauError P := by
  refine (density_biUnion_le _ _).trans (sum_le_sum fun c _ => ?_)
  split
  · exact density_lookupBad_le _ _ _ _ _ _
  · rw [density_empty]

/-- An error of a claim or of a lookup is not zero. -/
def Fails : Prop :=
  (∃ c : Fin P.claims, ClaimFails P.lanes P.equations (P.shape c).publicCells
    (ringLeft P.lanes (P.shape c).blocks (P.form c (st c).point) (sliceOf W (P.shape c).witness))
    (P.right (st c)) (sliceOf W (P.shape c).witness)
    (fun y => algebraMap F K ((st c).publicInput y))) ∨
  ∃ c : Fin P.claims, (P.shape c).lookup = true ∧
    ((∃ y < 2 ^ (P.shape c).witness.vars, inverseError P W V τ c y ≠ 0) ∨
      (∃ k, sumError P W V τ c k ≠ 0))

/-- The bad challenges after `V`: an error is not zero and the batch of the errors vanishes. -/
noncomputable def challengeBad : Finset (Challenges P) := by
  classical
  exact if Fails P st W V τ then
    univ.filter fun ch : Challenges P => ringBatch P st W ch.1 + lookupBatch P W V τ ch.2 = 0
  else ∅

/-- The largest number of public cells of a claim. -/
def maxPublic : ℕ := univ.sup fun c : Fin P.claims => (P.shape c).publicCells

/-- The term of `challengeBad` in the error. -/
noncomputable def challengeError : ℝ :=
  ((P.claims - 1 + (P.equations + max (P.lanes - 1) (maxPublic P - 1)) : ℕ) : ℝ) /
      Fintype.card K + ((2 + P.spanVars : ℕ) : ℝ) / Fintype.card K

end Draws

section Densities

variable {P : Params F K} (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)

def lookupCoefficient (lam : Fin P.claims → K) (k : Fin P.lookups) : K :=
  ∑ c, lam c * (lookupWeight P c * ι (sumError P W V τ c k))

def inverseBatch (xi : Fin P.claims → K) (β : Fin P.spanVars → K) : K :=
  ∑ c, xi c * (lookupWeight P c * inverseResidual P W V τ c β)

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem lookupBatch_split (look : LookupDraw K P.lookups P.claims P.spanVars) :
    lookupBatch P W V τ look = inverseBatch W V τ look.2.1.1 look.2.2 +
      ∑ k, look.1 k * lookupCoefficient W V τ look.2.1.2 k := by
  simp only [lookupBatch, inverseBatch, lookupCoefficient, mul_add, sum_add_distrib, mul_sum]
  congr 1
  · exact sum_congr rfl fun c _ => by ring
  · rw [sum_comm]
    exact sum_congr rfl fun k _ => sum_congr rfl fun c _ => by ring

omit [Fintype F] [DecidableEq F] in
theorem density_lookupBatch_le (wf : WellFormed P) (c₀ : Fin P.claims) (table : (P.shape c₀).lookup)
    (fails : (∃ y < 2 ^ (P.shape c₀).witness.vars, inverseError P W V τ c₀ y ≠ 0) ∨
      (∃ k, sumError P W V τ c₀ k ≠ 0)) (target : K) :
    density (univ.filter fun look : LookupDraw K P.lookups P.claims P.spanVars =>
      lookupBatch P W V τ look = target) ≤ ((2 + P.spanVars : ℕ) : ℝ) / Fintype.card K := by
  classical
  have one : (0 : ℝ) ≤ 1 / Fintype.card K := by positivity
  have weight : lookupWeight P c₀ = 1 := by simp [lookupWeight, table]
  rcases fails with ⟨y, cell, differs⟩ | ⟨k, differs⟩
  · let Sβ := univ.filter fun β : Fin P.spanVars → K => inverseResidual P W V τ c₀ β = 0
    have inside := (wf.span_fit c₀ table).2
    have eqMle (β : Fin P.spanVars → K) : inverseResidual P W V τ c₀ β =
        mle ((List.ofFn β).take (P.shape c₀).witness.vars) (fun i => ι (inverseError P W V τ c₀ i)) := by
      rw [mle_eq_cubeSum, cubeSum_eq_sum_range]
      simp [inverseResidual, inside]
    have bound : density Sβ ≤ (P.spanVars : ℝ) / Fintype.card K := by
      simp only [Sβ, eqMle]
      let values := fun i => ι (inverseError P W V τ c₀ i)
      have small := density_mle_zero_le (P.shape c₀).witness.vars values ⟨y, cell, by simpa [values] using differs⟩
      refine (density_take_le P.spanVars (P.shape c₀).witness.vars inside
        (fun point => mle point values = 0) _ small).trans ?_
      exact div_le_div_of_nonneg_right (by exact_mod_cast inside) (by positivity)
    have nonneg : 0 ≤ 1 / (Fintype.card K : ℝ) + density Sβ := add_nonneg one (by unfold density; positivity)
    refine (density_prod_le' _ ∅ _ nonneg fun psi _ => ?_).trans ?_
    · apply density_prod_le _ Sβ _ one
      intro β outside
      apply (density_prod_le _ ∅ _ one fun lam _ => ?_).trans (by rw [density_empty, add_zero])
      have nonzero : inverseResidual P W V τ c₀ β ≠ 0 := by simpa [Sβ] using outside
      have linear := density_linear_le P.claims (fun c => lookupWeight P c * inverseResidual P W V τ c β)
        (target - ∑ k, psi k * lookupCoefficient W V τ lam k) ⟨c₀, by rwa [weight, one_mul]⟩
      refine le_trans (le_of_eq (congrArg density ?_)) linear
      ext xi
      simp only [mem_filter, mem_univ, true_and, lookupBatch_split, inverseBatch, eq_sub_iff_add_eq]
    · rw [density_empty, add_zero]
      push_cast
      rw [add_div]
      linarith [div_nonneg (show (0 : ℝ) ≤ 1 by norm_num) (show (0 : ℝ) ≤ Fintype.card K by positivity),
        show (2 : ℝ) / Fintype.card K = 1 / Fintype.card K + 1 / Fintype.card K by ring]
  · let S := univ.filter fun rest : ((Fin P.claims → K) × (Fin P.claims → K)) × (Fin P.spanVars → K) =>
        lookupCoefficient W V τ rest.1.2 k = 0
    refine (density_prod_le _ S _ one fun rest outside => ?_).trans ?_
    · have nz : lookupCoefficient W V τ rest.1.2 k ≠ 0 := by simpa [S] using outside
      have linear := density_linear_le P.lookups (lookupCoefficient W V τ rest.1.2)
        (target - inverseBatch W V τ rest.1.1 rest.2) ⟨k, nz⟩
      refine le_trans (le_of_eq (congrArg density ?_)) linear
      ext psi
      simp only [mem_filter, mem_univ, true_and, lookupBatch_split, eq_sub_iff_add_eq]
      rw [add_comm]
    · have sb : density S ≤ 1 / Fintype.card K := by
        apply (density_prod_le _ ∅ _ one fun β _ => ?_).trans (by rw [density_empty, add_zero])
        apply (density_prod_le' _ ∅ _ one fun xi _ => ?_).trans (by rw [density_empty, add_zero])
        have linear := density_linear_le P.claims (fun c => lookupWeight P c * ι (sumError P W V τ c k)) 0
          ⟨c₀, by rw [weight, one_mul]; exact (map_ne_zero _).mpr differs⟩
        refine le_trans (le_of_eq (congrArg density ?_)) linear
        ext lam
        rw [mem_filter, mem_filter, mem_filter, mem_filter]
        simp [lookupCoefficient]
      push_cast
      rw [add_div]
      linarith [show (2 : ℝ) / Fintype.card K = 1 / Fintype.card K + 1 / Fintype.card K by ring,
        show (0 : ℝ) ≤ (P.spanVars : ℝ) / Fintype.card K by positivity]

omit [Fintype F] [DecidableEq F] in
theorem density_ringBatch_le (c : Fin P.claims)
    (fails : ClaimFails P.lanes P.equations (P.shape c).publicCells
      (ringLeft P.lanes (P.shape c).blocks (P.form c (st c).point) (sliceOf W (P.shape c).witness))
      (P.right (st c)) (sliceOf W (P.shape c).witness)
      (fun y => algebraMap F K ((st c).publicInput y))) :
    density (univ.filter fun r : RingDraw K => ringBatch P st W r = 0) ≤
      ((P.claims - 1 : ℕ) : ℝ) / Fintype.card K +
        (P.equations + max (P.lanes - 1) ((P.shape c).publicCells - 1) : ℕ) / Fintype.card K := by
  let m : ℕ → ℕ := fun k => if h : k < P.claims then (P.shape ⟨k, h⟩).publicCells else 0
  let L : ℕ → ℕ → ℕ → K := fun k => if h : k < P.claims then
    ringLeft P.lanes (P.shape ⟨k, h⟩).blocks (P.form ⟨k, h⟩ (st ⟨k, h⟩).point)
      (sliceOf W (P.shape ⟨k, h⟩).witness) else fun _ _ => 0
  let Y : ℕ → ℕ → ℕ → K := fun k => if h : k < P.claims then P.right (st ⟨k, h⟩)
    else fun _ _ => 0
  let z : ℕ → ℕ → K := fun k => if h : k < P.claims then sliceOf W (P.shape ⟨k, h⟩).witness
    else fun _ => 0
  let X : ℕ → ℕ → K := fun k => if h : k < P.claims then
    fun y => algebraMap F K ((st ⟨k, h⟩).publicInput y) else fun _ => 0
  have bound := density_claims_le P.claims P.lanes P.equations m L Y z X c c.isLt (by
    simpa [m, L, Y, z, X] using fails)
  have same : (univ.filter fun r : RingDraw K => ringBatch P st W r = 0) =
      univ.filter fun p : K × K × K × K => ∑ k ∈ range P.claims, p.1 ^ k *
        claimBatch P.lanes P.equations (m k) (L k) (Y k) (z k) (X k) p.2.1 p.2.2.1 p.2.2.2 = 0 := by
    ext r
    rw [mem_filter, mem_filter]
    simp only [mem_univ, true_and]
    unfold ringBatch
    rw [← Fin.sum_univ_eq_sum_range (fun k => r.1 ^ k *
      claimBatch P.lanes P.equations (m k) (L k) (Y k) (z k) (X k) r.2.1 r.2.2.1 r.2.2.2)]
    simp [m, L, Y, z, X]
  rw [same]
  refine bound.trans (le_of_eq ?_)
  simp [m]

omit [Fintype F] [DecidableEq F] in
theorem density_challengeBad_le (wf : WellFormed P) :
    density (challengeBad P st W V τ) ≤ challengeError P := by
  classical
  have ringTerm : ∀ c : Fin P.claims, ((P.claims - 1 : ℕ) : ℝ) / Fintype.card K +
      (P.equations + max (P.lanes - 1) ((P.shape c).publicCells - 1) : ℕ) / Fintype.card K ≤
      ((P.claims - 1 + (P.equations + max (P.lanes - 1) (maxPublic P - 1)) : ℕ) : ℝ) /
        Fintype.card K := fun c => by
    have pub : (P.shape c).publicCells ≤ maxPublic P :=
      le_sup (f := fun c : Fin P.claims => (P.shape c).publicCells) (mem_univ c)
    rw [← add_div]
    refine div_le_div_of_nonneg_right ?_ (by positivity)
    push_cast
    have : (max (P.lanes - 1) ((P.shape c).publicCells - 1) : ℕ) ≤
        max (P.lanes - 1) (maxPublic P - 1) := max_le_max le_rfl (by omega)
    have cast : ((max (P.lanes - 1) ((P.shape c).publicCells - 1) : ℕ) : ℝ) ≤
        ((max (P.lanes - 1) (maxPublic P - 1) : ℕ) : ℝ) := by exact_mod_cast this
    push_cast at cast
    linarith
  have nonneg_look : (0 : ℝ) ≤ ((2 + P.spanVars : ℕ) : ℝ) / Fintype.card K := by positivity
  have nonneg_ring : (0 : ℝ) ≤ ((P.claims - 1 + (P.equations + max (P.lanes - 1)
      (maxPublic P - 1)) : ℕ) : ℝ) / Fintype.card K := by positivity
  unfold challengeBad challengeError
  split
  · rename_i fails
    by_cases look : ∃ c : Fin P.claims, (P.shape c).lookup = true ∧
        ((∃ y < 2 ^ (P.shape c).witness.vars, inverseError P W V τ c y ≠ 0) ∨
          (∃ k, sumError P W V τ c k ≠ 0))
    · obtain ⟨c₀, table, differ⟩ := look
      refine (density_prod_le' _ ∅ _ nonneg_look fun r _ => ?_).trans ?_
      · refine le_trans (le_of_eq (congrArg density ?_))
          (density_lookupBatch_le W V τ wf c₀ table differ (-ringBatch P st W r))
        ext lk
        simp only [mem_filter, mem_univ, true_and]
        constructor <;> intro h <;> linear_combination h
      · rw [density_empty, add_zero]
        linarith
    · obtain ⟨c, claim⟩ := fails.resolve_right look
      push Not at look
      have zero : ∀ lk, lookupBatch P W V τ lk = 0 := fun lk => by
        refine sum_eq_zero fun c _ => ?_
        by_cases table : (P.shape c).lookup = true
        · obtain ⟨inverse, sums⟩ := look c table
          have res : inverseResidual P W V τ c lk.2.2 = 0 :=
            sum_eq_zero fun y member => by rw [inverse y (mem_range.mp member), map_zero, mul_zero]
          simp [res, sums]
        · simp [lookupWeight, table]
      have same : (univ.filter fun ch : Challenges P =>
          ringBatch P st W ch.1 + lookupBatch P W V τ ch.2 = 0) =
          (univ.filter fun r : RingDraw K => ringBatch P st W r = 0) ×ˢ univ := by
        ext ch
        simp [zero]
      rw [same, density_product, density_univ_one, mul_one]
      exact (density_ringBatch_le st W c claim).trans ((ringTerm c).trans (by linarith))
  · rw [density_empty]
    positivity

end Densities

end NightstreamFPrime.Spec.Compression.Verifier
