import NightstreamFPrime.Spec.Compression.Verifier.Weights

/-! The bad sets of the draws of the verifier and their densities.

- The lookup challenges `τ` (`tauBad`): for each claim with a table, each challenge is a root of its
  lookup polynomial. Density `Σ_tables ((N + T) / |F|)^L` (`density_tauBad_le`).
- The challenges after `V` (`challengeBad`): when an error of a claim or of a lookup is not zero, the
  batch of the errors vanishes. Density `(s − 1 + E + max(D − 1, m − 1)) / |K| + (2 + span) / |K|`
  (`density_challengeBad_le`).
- The column point (`columnBad`): the four values of `V` differ from the columns of `V`, and their
  extension at the column point agrees. Density `cv / |K|` (`density_columnBad_le`). -/

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
  ∃ c : Fin P.claims, (P.shape c).lookup = true ∧ ∃ k : Fin P.lookups,
    (∃ y < 2 ^ (P.shape c).witness.vars, inverseError P W V τ c k y ≠ 0) ∨
      sumError P W V τ c k ≠ 0

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

/-- The coefficient of `ψ_k` in the batch of the lookups. -/
def lookupCoefficient (rest : ((Fin P.claims → K) × (Fin P.claims → K)) × (Fin P.spanVars → K))
    (k : Fin P.lookups) : K :=
  ∑ c, lookupWeight P c * (rest.1.1 c * ∑ y ∈ range (2 ^ (P.shape c).witness.vars),
      eqAt ((List.ofFn rest.2).take (P.shape c).witness.vars) y * ι (inverseError P W V τ c k y) +
    rest.1.2 c * ι (sumError P W V τ c k))

omit [Fintype F] [DecidableEq F] in
theorem density_lookupBatch_le (wf : WellFormed P) (c₀ : Fin P.claims) (table : (P.shape c₀).lookup)
    (k₀ : Fin P.lookups) (fails : (∃ y < 2 ^ (P.shape c₀).witness.vars,
      inverseError P W V τ c₀ k₀ y ≠ 0) ∨ sumError P W V τ c₀ k₀ ≠ 0) (target : K) :
    density (univ.filter fun look : LookupDraw K P.lookups P.claims P.spanVars =>
      lookupBatch P W V τ look = target) ≤ ((2 + P.spanVars : ℕ) : ℝ) / Fintype.card K := by
  classical
  have one : (0 : ℝ) ≤ 1 / Fintype.card K := by positivity
  have weight : lookupWeight P c₀ = 1 := by simp [lookupWeight, table]
  set SB := univ.filter fun rest : ((Fin P.claims → K) × (Fin P.claims → K)) ×
    (Fin P.spanVars → K) => lookupCoefficient W V τ rest k₀ = 0
  have first := density_prod_le (univ.filter fun look : LookupDraw K P.lookups P.claims
      P.spanVars => lookupBatch P W V τ look = target) SB (1 / Fintype.card K) one
    fun rest outside => by
      have nonzero : lookupCoefficient W V τ rest k₀ ≠ 0 := by simpa [SB] using outside
      refine le_trans (le_of_eq (congrArg density ?_))
        (density_linear_le P.lookups (lookupCoefficient W V τ rest) target ⟨k₀, nonzero⟩)
      ext a
      simp only [mem_filter, mem_univ, true_and]
      rfl
  refine first.trans ?_
  have rest_bound : density SB ≤ ((1 + P.spanVars : ℕ) : ℝ) / Fintype.card K := by
    rcases fails with ⟨y₀, cell, differ⟩ | differ
    · set Sβ := univ.filter fun β : Fin P.spanVars → K =>
        ∑ y ∈ range (2 ^ (P.shape c₀).witness.vars),
          eqAt ((List.ofFn β).take (P.shape c₀).witness.vars) y *
            ι (inverseError P W V τ c₀ k₀ y) = 0
      refine (density_prod_le SB Sβ (1 / Fintype.card K) one fun β outside => ?_).trans ?_
      · have live : ∑ y ∈ range (2 ^ (P.shape c₀).witness.vars),
            eqAt ((List.ofFn β).take (P.shape c₀).witness.vars) y *
              ι (inverseError P W V τ c₀ k₀ y) ≠ 0 := by simpa [Sβ] using outside
        refine (density_prod_le _ ∅ (1 / Fintype.card K) one fun lam _ => ?_).trans
          (by rw [density_empty, add_zero])
        have := density_linear_le P.claims (fun c => lookupWeight P c *
          ∑ y ∈ range (2 ^ (P.shape c).witness.vars),
            eqAt ((List.ofFn β).take (P.shape c).witness.vars) y *
              ι (inverseError P W V τ c k₀ y))
          (-∑ c, lam c * (lookupWeight P c * ι (sumError P W V τ c k₀)))
          ⟨c₀, by rw [weight, one_mul]; exact live⟩
        refine le_trans (le_of_eq ?_) this
        congr 1
        ext ξ
        simp only [SB, mem_filter, mem_univ, true_and, lookupCoefficient, mul_add, sum_add_distrib]
        rw [eq_neg_iff_add_eq_zero]
        constructor <;> intro h <;> rw [← h] <;> congr 1 <;>
          exact sum_congr rfl fun c _ => by ring
      · have spanned := (wf.span_fit c₀ table).2
        have mle_form : ∀ β : Fin P.spanVars → K,
            ∑ y ∈ range (2 ^ (P.shape c₀).witness.vars),
              eqAt ((List.ofFn β).take (P.shape c₀).witness.vars) y *
                ι (inverseError P W V τ c₀ k₀ y) =
            mle ((List.ofFn β).take (P.shape c₀).witness.vars)
              (fun y => ι (inverseError P W V τ c₀ k₀ y)) := fun β => by
          rw [mle_eq_cubeSum, cubeSum_eq_sum_range]
          simp [spanned]
        have small := density_take_le P.spanVars (P.shape c₀).witness.vars spanned
          (fun point => mle point (fun y => ι (inverseError P W V τ c₀ k₀ y)) = 0) _
          (density_mle_zero_le _ _ ⟨y₀, cell, by simpa using differ⟩)
        have same : Sβ = univ.filter fun β : Fin P.spanVars → K =>
            mle ((List.ofFn β).take (P.shape c₀).witness.vars)
              (fun y => ι (inverseError P W V τ c₀ k₀ y)) = 0 := by
          ext β
          simp only [Sβ, mem_filter, mem_univ, true_and, mle_form]
        rw [same]
        have bound := small.trans (div_le_div_of_nonneg_right
          (show ((P.shape c₀).witness.vars : ℝ) ≤ P.spanVars by exact_mod_cast spanned)
          (show (0 : ℝ) ≤ Fintype.card K by positivity))
        have split : ((1 + P.spanVars : ℕ) : ℝ) / Fintype.card K =
            1 / Fintype.card K + (P.spanVars : ℝ) / Fintype.card K := by
          push_cast
          ring
        rw [split]
        linarith
    · refine (density_prod_le SB ∅ (1 / Fintype.card K) one fun β _ => ?_).trans ?_
      · refine (density_prod_le' _ ∅ (1 / Fintype.card K) one fun ξ _ => ?_).trans
          (by rw [density_empty, add_zero])
        have := density_linear_le P.claims (fun c => lookupWeight P c *
          ι (sumError P W V τ c k₀))
          (-∑ c, ξ c * (lookupWeight P c * ∑ y ∈ range (2 ^ (P.shape c).witness.vars),
            eqAt ((List.ofFn β).take (P.shape c).witness.vars) y *
              ι (inverseError P W V τ c k₀ y)))
          ⟨c₀, by
            rw [weight, one_mul]
            exact (map_ne_zero _).mpr differ⟩
        refine le_trans (le_of_eq ?_) this
        congr 1
        ext lam
        simp only [SB, mem_filter, mem_univ, true_and, lookupCoefficient, mul_add, sum_add_distrib]
        rw [eq_neg_iff_add_eq_zero, add_comm]
        constructor <;> intro h <;> rw [← h] <;> congr 1 <;>
          exact sum_congr rfl fun c _ => by ring
      · rw [density_empty, add_zero]
        exact div_le_div_of_nonneg_right (by exact_mod_cast (by omega : 1 ≤ 1 + P.spanVars))
          (by positivity)
  refine (add_le_add le_rfl rest_bound).trans (le_of_eq ?_)
  push_cast
  ring

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

/-- The bad column points: the values of `V` differ from its columns, and their extensions at the
column point agree. -/
noncomputable def columnBad (cv : ℕ) (values columns : ℕ → K) : Finset (Fin cv → K) := by
  classical
  exact if ∃ k < 2 ^ cv, values k ≠ columns k then
    univ.filter fun col : Fin cv → K => mle (List.ofFn col) (fun k => values k - columns k) = 0
  else ∅

omit [Field F] [Fintype F] [DecidableEq F] [Algebra F K] in
theorem density_columnBad_le (cv : ℕ) (values columns : ℕ → K) :
    density (columnBad cv values columns) ≤ cv / Fintype.card K := by
  classical
  unfold columnBad
  split
  · rename_i differ
    obtain ⟨k, small, value⟩ := differ
    exact density_mle_zero_le cv _ ⟨k, small, sub_ne_zero.mpr value⟩
  · rw [density_empty]
    positivity

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
    by_cases look : ∃ c : Fin P.claims, (P.shape c).lookup = true ∧ ∃ k : Fin P.lookups,
        (∃ y < 2 ^ (P.shape c).witness.vars, inverseError P W V τ c k y ≠ 0) ∨
          sumError P W V τ c k ≠ 0
    · obtain ⟨c₀, table, k₀, differ⟩ := look
      refine (density_prod_le' _ ∅ _ nonneg_look fun r _ => ?_).trans ?_
      · refine le_trans (le_of_eq (congrArg density ?_))
          (density_lookupBatch_le W V τ wf c₀ table k₀ differ (-ringBatch P st W r))
        ext lk
        simp only [mem_filter, mem_univ, true_and]
        constructor <;> intro h <;> linear_combination h
      · rw [density_empty, add_zero]
        linarith
    · obtain ⟨c, claim⟩ := fails.resolve_right look
      push Not at look
      have zero : ∀ lk, lookupBatch P W V τ lk = 0 := fun lk => by
        refine sum_eq_zero fun k _ => ?_
        refine mul_eq_zero_of_right _ (sum_eq_zero fun c _ => ?_)
        by_cases table : (P.shape c).lookup = true
        · obtain ⟨inverse, sums⟩ := look c table k
          rw [sum_eq_zero fun y member => by rw [inverse y (mem_range.mp member), map_zero,
            mul_zero], sums, map_zero]
          ring
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
