import NightstreamFPrime.Spec.Compression.Verifier.Bound

/-! Completeness at the level of the claims. For a witness that satisfies `Relation`, honest
counts and honest inverses:

- the sum of the first sum-check is the claimed sum for every challenge (`honest_sum`), with the
  counting identity of the lookup (`lookup_complete`);
- the key part is the sum of the key sum-check (`key_sum`), and the honest key value passes the
  last check of the key sum-check (`key_check_honest`);
- the honest values at `ρ` pass the last check of the first sum-check (`main_check_honest`).

The lookup has no proof when a challenge `τ_k` is a table value, and the key has no row point when
`1 + ω^(2^k) = 0` (`rowPoint_degenerate_card_le`: at most 7 values of `ω` when `16 ∤ |K| − 1`). -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra
open NightstreamFPrime.Spec.Compression.Game Polynomial

/-- Each cell is one table value, so a sum over the cells is a sum over the values with counts. -/
theorem lookup_complete {K : Type} [Field K] [DecidableEq K] (N T : ℕ) (w value : ℕ → K)
    (distinct : ∀ s t, s < T → t < T → value s = value t → s = t)
    (inside : ∀ x < N, ∃ t < T, w x = value t) (f : K → K) :
    ∑ x ∈ range N, f (w x) =
      ∑ t ∈ range T, (((range N).filter fun x => w x = value t).card : K) * f (value t) := by
  have each (t : ℕ) : (((range N).filter fun x => w x = value t).card : K) * f (value t) =
      ∑ x ∈ range N, if w x = value t then f (w x) else 0 := by
    rw [← sum_filter, sum_congr rfl fun x member => by rw [(mem_filter.mp member).2], sum_const,
      nsmul_eq_mul]
  rw [sum_congr rfl fun t _ => each t, sum_comm]
  refine sum_congr rfl fun x member => ?_
  obtain ⟨t₀, lt, equal⟩ := inside x (mem_range.mp member)
  rw [sum_eq_single t₀ (fun t member' different => by
    rw [ite_eq_right fun same => different (distinct t t₀ (mem_range.mp member') lt
      (same.symm.trans equal))]) (fun outside => absurd (mem_range.mpr lt) outside),
    ite_eq_left equal]

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
  [Algebra F K]

local notation "ι" => algebraMap F K

variable {P : Params F K}

/-- The counts in `W` are the honest counts of the table values in each witness with a table. -/
def HonestCounts (P : Params F K) (W : ℕ → F) : Prop :=
  ∀ c, (P.shape c).lookup = true → ∀ v < 2 * (P.shape c).bound + 1,
    W ((P.shape c).counts.offset + v) = (((range (2 ^ (P.shape c).witness.vars)).filter fun y =>
      W ((P.shape c).witness.offset + y) = tableValue (P.shape c).bound v).card : F)

/-- `V` holds the honest inverses `1/(τ_k − w)`. -/
def HonestInverses (P : Params F K) (W V : ℕ → F) (τ : Fin P.lookups → F) : Prop :=
  ∀ c, (P.shape c).lookup = true → ∀ k : Fin P.lookups, ∀ y < 2 ^ (P.shape c).witness.vars,
    inverseOf P V c k y = 1 / (τ k - W ((P.shape c).witness.offset + y))

/-- The values of a table are distinct in `F`: `2 · bound + 1 ≤ char F`. -/
def TablesDistinct (P : Params F K) : Prop :=
  ∀ c, (P.shape c).lookup = true → ∀ s t, s < 2 * (P.shape c).bound + 1 →
    t < 2 * (P.shape c).bound + 1 → (tableValue (P.shape c).bound s : F) =
      tableValue (P.shape c).bound t → s = t

omit [Fintype F] [Fintype K] [DecidableEq K] in
/-- The sum of the first sum-check is the claimed sum for every challenge. -/
theorem honest_sum (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F) (ch : Challenges P)
    (wf : WellFormed P) (relation : Relation P st W) (counts : HonestCounts P W)
    (inverses : HonestInverses P W V τ) (distinct : TablesDistinct P)
    (τ_ok : ∀ c, (P.shape c).lookup = true → ∀ k, ∀ v < 2 * (P.shape c).bound + 1,
      τ k ≠ tableValue (P.shape c).bound v) :
    ∑ x ∈ range (2 ^ P.witnessVars), (ι (W x) * (weightTable P st τ ch x -
        checkTable P ch x * aTable P V ch x) + checkTable P ch x * bTable P V τ ch x +
        sumTable P ch x * aTable P V ch x) = claimedSum P st ch := by
  rw [← sub_eq_zero, sum_sub_claimed st W V τ ch wf]
  have ring_zero : ringBatch P st W ch.1 = 0 := by
    refine sum_eq_zero fun c _ => ?_
    obtain ⟨rings, publics, _⟩ := relation c
    have values : ∀ e ∈ range P.equations, ch.1.2.1 ^ e * (value P.lanes ch.1.2.2.1
        (ringLeft P.lanes (P.shape c).blocks (P.form c (st c).point)
          (sliceOf W (P.shape c).witness) e) - value P.lanes ch.1.2.2.1 (P.right (st c) e)) = 0 :=
      fun e member => by
        rw [show value P.lanes ch.1.2.2.1 (ringLeft P.lanes (P.shape c).blocks
          (P.form c (st c).point) (sliceOf W (P.shape c).witness) e) =
            value P.lanes ch.1.2.2.1 (P.right (st c) e) from sum_congr rfl fun l lane => by
              rw [rings e (mem_range.mp member) l (mem_range.mp lane)], sub_self, mul_zero]
    have cells : ∀ y ∈ range (P.shape c).publicCells, ch.1.2.2.2 ^ y *
        (sliceOf W (P.shape c).witness y - ι ((st c).publicInput y)) = 0 := fun y member => by
      rw [sliceOf, publics y (mem_range.mp member), sub_self, mul_zero]
    simp only [claimBatch, sum_eq_zero values, sum_eq_zero cells, mul_zero, add_zero]
  have look_zero : lookupBatch P W V τ ch.2 = 0 := by
    refine sum_eq_zero fun k _ => mul_eq_zero_of_right _ (sum_eq_zero fun c _ => ?_)
    by_cases table : (P.shape c).lookup = true
    · obtain ⟨_, _, bounded⟩ := relation c
      have cell (y : ℕ) (member : y < 2 ^ (P.shape c).witness.vars) :
          τ k - W ((P.shape c).witness.offset + y) ≠ 0 := by
        obtain ⟨v, lt, equal⟩ := bounded table y member
        rw [sub_ne_zero, equal]
        exact τ_ok c table k v lt
      have inverse_zero : ∀ y ∈ range (2 ^ (P.shape c).witness.vars),
          eqAt ((List.ofFn ch.2.2.2).take (P.shape c).witness.vars) y *
            ι (inverseError P W V τ c k y) = 0 := fun y member => by
        rw [inverseError, inverses c table k y (mem_range.mp member),
          mul_one_div_cancel (cell y (mem_range.mp member)), sub_self, map_zero, mul_zero]
      have sum_zero : sumError P W V τ c k = 0 := by
        rw [sumError, sub_eq_zero, sum_congr rfl fun y member =>
          inverses c table k y (mem_range.mp member)]
        rw [lookup_complete _ _ (fun y => W ((P.shape c).witness.offset + y))
          (tableValue (P.shape c).bound) (distinct c table) (bounded table)
          (fun u => 1 / (τ k - u))]
        exact sum_congr rfl fun v member => by
          rw [counts c table v (mem_range.mp member)]
      rw [sum_eq_zero inverse_zero, sum_zero, map_zero]
      ring
    · simp [lookupWeight, table]
  rw [ring_zero, look_zero, add_zero]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- The honest key value passes the last check of the key sum-check: the claim at its point is
`scale · value · weight_at`. -/
theorem key_check_honest (ch : Challenges P) (ρ : List K) (kp : List K) (wf : WellFormed P)
    (st : Statement P) (shaped : Shaped P st) (length : ρ.length = P.witnessVars)
    (kp_length : kp.length = P.laneVars + P.keyBlockVars) (rows : P.rows ≤ 2 ^ P.rowVars)
    (regular : ∀ i < P.rowVars, 1 + ch.omega ^ 2 ^ i ≠ 0)
    (key_fits : ∀ c, (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars) :
    mle kp (keyTables ch ρ 0) * mle kp (keyTables ch ρ 1) =
      rowScale P.rowVars ch.omega * mle (kp ++ rowPoint P.rowVars ch.omega)
        (keyTable P.lanes P.keyBlockVars P.rows fun i j h => ι (P.key i j h)) *
      weightAt P.lanes P.laneVars (laneForm P ch ρ) (uses P ch ρ) kp := by
  have point_length (c : Fin P.claims) :
      (uses P ch ρ c).point.length = (P.shape c).witness.vars - P.laneVars := by
    have := (wf.witness_fits c).2.2
    simp [uses, length, this]
  simp only [keyTables, Matrix.cons_val_zero, Matrix.cons_val_one, Params.lanes]
  rw [weightAt_spec P.laneVars P.keyBlockVars _ _ kp kp_length fun c =>
    ⟨by rw [point_length]; exact key_fits c, by rw [point_length]; exact shaped.blocks_le c⟩,
    folded_at (2 ^ P.laneVars) P.keyBlockVars P.rows P.rowVars _ ch.omega kp
      (by rw [kp_length, pow_add]) rows regular]
  ring

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- The honest values at `ρ` pass the last check of the first sum-check: with `W(ρ)`, the columns
of `V`, the key part and the matrix part, the verifier's value is the summand at `ρ`. -/
theorem main_check_honest (vf : VerifierData P) (st : Statement P) (W V : ℕ → F)
    (τ : Fin P.lookups → F) (ch : Challenges P) (ρ : List K) (wf : WellFormed P)
    (shaped : Shaped P st) (length : ρ.length = P.witnessVars) (matrix : MatrixSpec vf) :
    combTruth P.witnessVars (mainTables P st W V τ ch) mainComb ρ =
      mle ρ (fun x => ι (W x)) * (keyTrue P ch ρ + vf.matrixPart st ch ρ +
          closedPart P st τ ch ρ -
        checkAt P ch ρ * (lowAt P ρ * ∑ k, ch.psi k * columnAt P V ρ k)) +
      checkAt P ch ρ * (lowAt P ρ * ∑ k, ch.psi k * ι (τ k) * columnAt P V ρ k) +
      sumAt P ch ρ * (lowAt P ρ * ∑ k, ch.psi k * columnAt P V ρ k) := by
  rw [main_truth_final st W V τ ch ρ length, matrix st ch ρ length,
    mle_weightTable st τ ch ρ wf shaped length, mle_checkTable ch ρ wf length,
    mle_sumTable ch ρ wf length, mle_aTable V ch ρ wf length, mle_bTable V τ ch ρ wf length]

end NightstreamFPrime.Spec.Compression.Verifier
