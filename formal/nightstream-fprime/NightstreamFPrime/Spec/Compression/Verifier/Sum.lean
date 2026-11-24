import NightstreamFPrime.Spec.Compression.Verifier.Model
import NightstreamFPrime.Spec.Compression.Lib.Combination

/-! SPEC.md section 6. -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra

variable {F K : Type} [Field F] [Field K] [Algebra F K]

/-- The layout facts that the verifier relies on. Rust `Layout::new` places each slice so; Lean
does not model that function. -/
structure WellFormed (P : Params F K) : Prop where
  witness_fits : ∀ c, (P.shape c).witness.Fits P.witnessVars
  counts_fits : ∀ c, (P.shape c).lookup = true → (P.shape c).counts.Fits P.witnessVars
  blocks_fit : ∀ c, P.lanes * (P.shape c).blocks ≤ 2 ^ (P.shape c).witness.vars
  public_fit : ∀ c, (P.shape c).publicCells ≤ P.lanes * (P.shape c).blocks
  table_fit : ∀ c, (P.shape c).lookup = true →
    2 * (P.shape c).bound + 1 ≤ 2 ^ (P.shape c).counts.vars
  span_fit : ∀ c, (P.shape c).lookup = true →
    (P.shape c).witness.offset + 2 ^ (P.shape c).witness.vars ≤ 2 ^ P.spanVars ∧
      (P.shape c).witness.vars ≤ P.spanVars
  span_le : P.spanVars ≤ P.witnessVars

section Errors

variable (P : Params F K) (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)

def inverseOf (c : Fin P.claims) (k : Fin P.lookups) (y : ℕ) : F :=
  let x := (P.shape c).witness.offset + y
  V x * cofactor τ k (W x)

def inverseError (c : Fin P.claims) (y : ℕ) : F :=
  let x := (P.shape c).witness.offset + y
  V x * denominator τ (W x) - 1

omit [Field K] [Algebra F K] in
theorem inverseOf_spec (c : Fin P.claims) (k : Fin P.lookups) (y : ℕ)
    (h : inverseError P W V τ c y = 0) :
    (τ k - W ((P.shape c).witness.offset + y)) * inverseOf P W V τ c k y = 1 := by
  unfold inverseOf
  dsimp only
  rw [mul_left_comm, factor_mul_cofactor]
  exact sub_eq_zero.mp h

/-- The lookup sum of challenge `k`: `Σ_y I_k(y) − Σ_v m(v) / (τ_k − value v)`. -/
def sumError (c : Fin P.claims) (k : Fin P.lookups) : F :=
  ∑ y ∈ range (2 ^ (P.shape c).witness.vars), inverseOf P W V τ c k y -
    ∑ v ∈ range (2 * (P.shape c).bound + 1),
      W ((P.shape c).counts.offset + v) * (1 / (τ k - tableValue (P.shape c).bound v))

/-- The errors of the claims, batched in `(μ, ω, δ, γ)`. -/
def ringBatch (ring : RingDraw K) : K :=
  ∑ c : Fin P.claims, ring.1 ^ (c : ℕ) *
    claimBatch P.lanes P.equations (P.shape c).publicCells
      (ringLeft P.lanes (P.shape c).blocks (P.form c (st c).point) (sliceOf W (P.shape c).witness))
      (P.right (st c)) (sliceOf W (P.shape c).witness)
      (fun y => algebraMap F K ((st c).publicInput y)) ring.2.1 ring.2.2.1 ring.2.2.2

def inverseResidual (c : Fin P.claims) (β : Fin P.spanVars → K) : K :=
  ∑ y ∈ range (2 ^ (P.shape c).witness.vars),
    eqAt ((List.ofFn β).take (P.shape c).witness.vars) y * algebraMap F K (inverseError P W V τ c y)

def lookupBatch (look : LookupDraw K P.lookups P.claims P.spanVars) : K :=
  ∑ c, lookupWeight P c * (look.2.1.1 c * inverseResidual P W V τ c look.2.2 +
    look.2.1.2 c * ∑ k, look.1 k * algebraMap F K (sumError P W V τ c k))

end Errors

section Identity

variable {P : Params F K} (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
  (ch : Challenges P)

local notation "ι" => algebraMap F K

theorem sum_range_ite_mul (n N : ℕ) (fits : n ≤ N) (f g : ℕ → K) :
    ∑ y ∈ range N, (if y < n then f y else 0) * g y = ∑ y ∈ range n, f y * g y := by
  rw [← sum_range_add_sum_Ico _ fits, sum_eq_zero (s := Ico n N) fun y member => by
    rw [ite_eq_right (by have := (mem_Ico.mp member).1; omega), zero_mul], add_zero]
  exact sum_congr rfl fun y member => by rw [ite_eq_left (mem_range.mp member)]

theorem sum_weightTable (wf : WellFormed P) :
    ∑ x ∈ range (2 ^ P.witnessVars), ι (W x) * weightTable P st τ ch x =
      ∑ c : Fin P.claims, (ch.mu ^ (c : ℕ) * (claimTarget P.lanes P.equations
          (P.shape c).publicCells (P.right (st c)) (fun y => ι ((st c).publicInput y))
          ch.omega ch.delta ch.gamma +
        claimBatch P.lanes P.equations (P.shape c).publicCells
          (ringLeft P.lanes (P.shape c).blocks (P.form c (st c).point)
            (sliceOf W (P.shape c).witness))
          (P.right (st c)) (sliceOf W (P.shape c).witness)
          (fun y => ι ((st c).publicInput y)) ch.omega ch.delta ch.gamma) +
        lookupWeight P c * -(ch.lam c * ∑ k, ch.psi k *
          ∑ v ∈ range (2 * (P.shape c).bound + 1), ι (W ((P.shape c).counts.offset + v)) *
            ι (1 / (τ k - tableValue (P.shape c).bound v)))) := by
  simp only [weightTable, mul_sum]
  rw [sum_comm]
  refine sum_congr rfl fun c _ => ?_
  simp only [mul_add, sum_add_distrib]
  congr 1
  · rw [sum_congr rfl fun x _ => mul_comm _ _,
      Slice.sum_place _ _ (wf.witness_fits c) _ (fun x => ι (W x))]
    simp only [claimWeights]
    rw [sum_range_ite_mul _ _ (wf.blocks_fit c)]
    have := claim_difference P.lanes (P.shape c).blocks P.equations (P.shape c).publicCells
      (P.form c (st c).point) (P.right (st c)) (fun y => ι ((st c).publicInput y))
      (sliceOf W (P.shape c).witness) ch.omega ch.delta ch.gamma (pow_pos two_pos _)
      (wf.public_fit c)
    rw [← claimBatch] at this
    rw [show (∑ y ∈ range (P.lanes * (P.shape c).blocks), ch.mu ^ (c : ℕ) *
        claimWeight P.lanes P.equations (P.shape c).publicCells (P.form c (st c).point) ch.omega
          ch.delta ch.gamma y * ι (W ((P.shape c).witness.offset + y))) = ch.mu ^ (c : ℕ) *
        ∑ y ∈ range (P.lanes * (P.shape c).blocks), sliceOf W (P.shape c).witness y *
          claimWeight P.lanes P.equations (P.shape c).publicCells (P.form c (st c).point) ch.omega
            ch.delta ch.gamma y by
      rw [mul_sum]
      exact sum_congr rfl fun y _ => by simp only [sliceOf]; ring]
    linear_combination (ch.mu ^ (c : ℕ)) * this
  · by_cases table : (P.shape c).lookup = true
    · rw [sum_congr rfl fun x _ => show ι (W x) * (lookupWeight P c *
          (P.shape c).counts.place (countWeights P τ ch c) x) = lookupWeight P c *
          ((P.shape c).counts.place (countWeights P τ ch c) x * ι (W x)) by ring, ← mul_sum,
        Slice.sum_place _ _ (wf.counts_fits c table) _ (fun x => ι (W x))]
      simp only [countWeights]
      rw [sum_range_ite_mul _ _ (wf.table_fit c table)]
      congr 1
      simp only [neg_mul, sum_neg_distrib, mul_sum, sum_mul]
      congr 1
      rw [sum_comm]
      exact sum_congr rfl fun k _ => sum_congr rfl fun v _ => by ring
    · simp [lookupWeight, table]

theorem sum_checkTable (wf : WellFormed P) :
    ∑ x ∈ range (2 ^ P.witnessVars), checkTable P ch x *
        (inverseTable P V x * denominator (fun k => ι (τ k)) (ι (W x))) =
      ∑ c, lookupWeight P c * (ch.xi c * ∑ y ∈ range (2 ^ (P.shape c).witness.vars),
          eqAt (ch.beta.take (P.shape c).witness.vars) y *
            ι (V ((P.shape c).witness.offset + y) * denominator τ (W ((P.shape c).witness.offset + y)))) := by
  simp only [checkTable, sum_mul]
  rw [sum_comm]
  refine sum_congr rfl fun c _ => ?_
  by_cases table : (P.shape c).lookup = true
  · simp only [mul_assoc, ← mul_sum]
    congr 1
    rw [Slice.sum_place _ _ (wf.witness_fits c)]
    rw [mul_sum]
    apply sum_congr rfl
    intro y member
    have small : (P.shape c).witness.offset + y < 2 ^ P.spanVars := by
      have := (wf.span_fit c table).1
      have := mem_range.mp member
      omega
    simp only [inverseTable, ite_eq_left small, map_mul, denominator, map_prod, map_sub, mul_assoc]
  · simp [lookupWeight, table]

theorem sum_sumTable (wf : WellFormed P) :
    ∑ x ∈ range (2 ^ P.witnessVars), sumTable P ch x *
        (inverseTable P V x * ∑ k, ch.psi k * cofactor (fun j => ι (τ j)) k (ι (W x))) =
      ∑ c, lookupWeight P c * (ch.lam c * ∑ k, ch.psi k *
        ∑ y ∈ range (2 ^ (P.shape c).witness.vars), ι (inverseOf P W V τ c k y)) := by
  simp only [sumTable, sum_mul]
  rw [sum_comm]
  refine sum_congr rfl fun c _ => ?_
  by_cases table : (P.shape c).lookup = true
  · simp only [mul_assoc, ← mul_sum]
    congr 1
    rw [Slice.sum_place _ _ (wf.witness_fits c), ← mul_sum]
    congr 1
    simp only [mul_sum]
    rw [sum_comm]
    apply sum_congr rfl
    intro k _
    apply sum_congr rfl
    intro y member
    have small : (P.shape c).witness.offset + y < 2 ^ P.spanVars := by
      have := (wf.span_fit c table).1
      have := mem_range.mp member
      omega
    simp only [inverseTable, ite_eq_left small, inverseOf, map_mul, cofactor, map_prod, map_sub]
    ring
  · simp [lookupWeight, table]

theorem sum_sub_claimed (wf : WellFormed P) :
    ∑ x ∈ range (2 ^ P.witnessVars), mainSummand (fun k => ι (τ k)) ch.psi (ι (W x))
        (weightTable P st τ ch x) (checkTable P ch x) (inverseTable P V x) (sumTable P ch x) -
        claimedSum P st ch = ringBatch P st W ch.1 + lookupBatch P W V τ ch.2 := by
  simp only [mainSummand, sum_add_distrib]
  rw [sum_weightTable st W τ ch wf, sum_checkTable W V τ ch wf, sum_sumTable W V τ ch wf,
    claimedSum, ringBatch, lookupBatch]
  simp only [inverseResidual, show ch.1.1 = ch.mu from rfl, show ch.1.2.1 = ch.omega from rfl,
    show ch.1.2.2.1 = ch.delta from rfl, show ch.1.2.2.2 = ch.gamma from rfl,
    show ch.2.1 = ch.psi from rfl, show ch.2.2.1.1 = ch.xi from rfl,
    show ch.2.2.1.2 = ch.lam from rfl, show List.ofFn ch.2.2.2 = ch.beta from rfl,
    ← sum_add_distrib, ← sum_sub_distrib]
  refine sum_congr rfl fun c _ => ?_
  by_cases table : (P.shape c).lookup = true
  · have length : (ch.beta.take (P.shape c).witness.vars).length = (P.shape c).witness.vars := by
      simp [Challenges.beta, (wf.span_fit c table).2]
    have inverse : ∑ y ∈ range (2 ^ (P.shape c).witness.vars),
        eqAt (ch.beta.take (P.shape c).witness.vars) y * ι (inverseError P W V τ c y) =
        ∑ y ∈ range (2 ^ (P.shape c).witness.vars),
          eqAt (ch.beta.take (P.shape c).witness.vars) y *
            ι (V ((P.shape c).witness.offset + y) * denominator τ (W ((P.shape c).witness.offset + y))) - 1 := by
      simp only [inverseError, map_sub, map_one, mul_sub, mul_one, sum_sub_distrib]
      congr 1
      have := sum_eqAt (ch.beta.take (P.shape c).witness.vars)
      rwa [length] at this
    have sums (k : Fin P.lookups) : ι (sumError P W V τ c k) =
        ∑ y ∈ range (2 ^ (P.shape c).witness.vars), ι (inverseOf P W V τ c k y) -
          ∑ v ∈ range (2 * (P.shape c).bound + 1), ι (W ((P.shape c).counts.offset + v)) *
            ι (1 / (τ k - tableValue (P.shape c).bound v)) := by
      simp only [sumError, map_sub, map_sum, map_mul]
    simp only [inverse, sums, mul_sub, sum_sub_distrib, mul_one]
    ring
  · simp only [lookupWeight, table, Bool.false_eq_true, ite_false, zero_mul]
    ring

end Identity

end NightstreamFPrime.Spec.Compression.Verifier
