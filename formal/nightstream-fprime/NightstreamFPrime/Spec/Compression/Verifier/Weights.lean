import NightstreamFPrime.Spec.Compression.Verifier.Sum

/-! The values of the public tables at the point `ρ` of the first sum-check, as the verifier
computes them (Rust `weights::Uses`, `key_part`, `matrix_part`, `closed_part` and `verifier::verify`).

`G(ρ)` is the key part (`keyPart`, which the proof carries and the key sum-check proves), plus the
matrix part (`matrixTrue`, which Rust `matrix_part` computes and this unit takes as a hypothesis),
plus the closed part (`closedPart`): `mle_weightTable`. The weights of the inverse checks, of the
sums and of the inverses have the closed forms of Rust `verify` (`mle_checkTable`, `mle_sumTable`,
`mle_aTable`, `mle_bTable`). -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

section Values

variable (P : Params F K) (st : Statement P) (τ : Fin P.lookups → F) (ch : Challenges P)
  (ρ : List K)

/-- Rust `Uses::form`: the lane weights of `E = Σ_h eq(ρ_lo, h) X^h` at `δ`. -/
def laneForm : ℕ → K := lanes P.lanes ch.delta (eqAt (ρ.take P.laneVars))

/-- Rust `Uses::claims`: the coefficient `μ^c · selector`, the point `β` and the blocks. -/
def uses (c : Fin P.claims) : Use K :=
  ⟨ch.mu ^ (c : ℕ) * (P.shape c).witness.selector ρ,
    (ρ.take (P.shape c).witness.vars).drop P.laneVars, (P.shape c).blocks⟩

/-- Rust `Evaluation::lanes`: `ω^κ · conjugate(eq(r_lo, ·))`. -/
def evalLanes (c : Fin P.claims) (h : ℕ) : K :=
  ch.omega ^ P.rows * conjugate P.lanes (eqAt ((st c).point.take P.laneVars)) h

/-- Rust `closed_part`: `Eval_K` and the public input of each claim, and the counts of each table. -/
def closedPart : K :=
  ∑ c, ((uses P ch ρ c).coefficient *
      ((∑ h ∈ range P.lanes, laneForm P ch ρ h * evalLanes P st ch c h) *
          prefixEquality (uses P ch ρ c).point (P.shape c).blocks
            ((st c).point.drop P.laneVars) +
        ch.omega ^ P.equations * ∑ y ∈ range (P.shape c).publicCells,
          eqAt (ρ.take (P.shape c).witness.vars) y * ch.gamma ^ y) +
    lookupWeight P c * ((P.shape c).counts.selector ρ *
      ∑ v ∈ range (2 * (P.shape c).bound + 1),
        eqAt (ρ.take (P.shape c).counts.vars) v * countWeights P τ ch c v))

/-- The matrix part of `G(ρ)` that the honest weights give: what Rust `matrix_part` must compute. -/
def matrixTrue : K :=
  ∑ c, (uses P ch ρ c).coefficient * ∑ m ∈ range P.matrices, ch.omega ^ (P.rows + 1 + m) *
    ∑ j ∈ range (P.shape c).blocks, eqAt (uses P ch ρ c).point j *
      ∑ h ∈ range P.lanes, laneForm P ch ρ h * P.matrixForm c (st c).point m j h

/-- The key part of `G(ρ)`, from `F = Σ_i ω^i A_i` (Rust `key_part`). -/
def keyTrue : K :=
  keyPart P.lanes (laneForm P ch ρ) (uses P ch ρ) (folded P.lanes P.rows
    (fun i j h => ι (P.key i j h)) ch.omega)

/-- Rust `verify`: the weight of the inverse checks at `ρ`. -/
def checkAt : K :=
  ∑ c, lookupWeight P c * (ch.xi c * eqEval (ch.beta.take (P.shape c).witness.vars)
    (ρ.take (P.shape c).witness.vars) * (P.shape c).witness.selector ρ)

/-- Rust `verify`: the weight of the lookup sums at `ρ`. -/
def sumAt : K := ∑ c, lookupWeight P c * (ch.lam c * (P.shape c).witness.selector ρ)

/-- Rust `verify`: `Π_{i ≥ span} (1 − ρ_i)`. -/
def lowAt : K := ((ρ.drop P.spanVars).map fun x => 1 - x).prod

end Values

section Proofs

variable {P : Params F K} (st : Statement P) (W V : ℕ → F) (τ : Fin P.lookups → F)
  (ch : Challenges P) (ρ : List K)

/-- The shape checks of Rust `layout` that the weights rely on: the lanes are inside each slice,
and the point of each claim has the variables of its blocks. -/
structure Shaped (P : Params F K) (st : Statement P) : Prop where
  lanes_le : ∀ c, P.laneVars ≤ (P.shape c).witness.vars
  point_long : ∀ c, (P.shape c).witness.vars ≤ (st c).point.length
  blocks_le : ∀ c, (P.shape c).blocks ≤ 2 ^ ((P.shape c).witness.vars - P.laneVars)

theorem mle_blocks (d v n : ℕ) (point : List K) (length : point.length = v) (lanes_le : d ≤ v)
    (fits : n ≤ 2 ^ (v - d)) (g : ℕ → ℕ → K) :
    mle point (fun y => if y < 2 ^ d * n then g (y / 2 ^ d) (y % 2 ^ d) else 0) =
      ∑ j ∈ range n, eqAt (point.drop d) j *
        ∑ h ∈ range (2 ^ d), eqAt (point.take d) h * g j h := by
  have split : point = point.take d ++ point.drop d := (List.take_append_drop _ _).symm
  have low_length : (point.take d).length = d := by simp [length, lanes_le]
  have high_length : (point.drop d).length = v - d := by simp [length]
  have positive : 0 < 2 ^ d := pow_pos two_pos d
  rw [split, mle_append_split, low_length, mle_eq_cubeSum, cubeSum_eq_sum_range, high_length]
  rw [List.take_append_drop]
  rw [← sum_range_add_sum_Ico _ fits, sum_eq_zero (s := Ico n (2 ^ (v - d))) fun j member => by
    have large := (mem_Ico.mp member).1
    have zero : ∀ y, y < 2 ^ (point.take d).length →
        (if y + 2 ^ d * j < 2 ^ d * n then g ((y + 2 ^ d * j) / 2 ^ d) ((y + 2 ^ d * j) % 2 ^ d)
          else 0) = (0 : K) := fun y _ => by
      rw [ite_eq_right]
      intro below
      have : 2 ^ d * n ≤ 2 ^ d * j := Nat.mul_le_mul_left _ large
      omega
    rw [mle_congr _ zero, mle_const, mul_zero], add_zero]
  refine sum_congr rfl fun j member => ?_
  have small_j := mem_range.mp member
  congr 1
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range, low_length]
  refine sum_congr rfl fun h lane => ?_
  have small_h := mem_range.mp lane
  have below : h + 2 ^ d * j < 2 ^ d * n := by
    have : 2 ^ d * (j + 1) ≤ 2 ^ d * n := Nat.mul_le_mul_left _ small_j
    rw [mul_add, mul_one] at this
    omega
  rw [ite_eq_left below, Nat.add_mul_div_left _ _ positive, Nat.div_eq_of_lt small_h, zero_add,
    Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt small_h]

/-- Equation `e` of claim `c` at `ρ`: `Σ_{j < blocks} eq(β, j) · Σ_h u(h) · P_{e,j}[h]`. -/
def equationAt (c : Fin P.claims) (e : ℕ) : K :=
  ∑ j ∈ range (P.shape c).blocks, eqAt (uses P ch ρ c).point j *
    ∑ h ∈ range P.lanes, laneForm P ch ρ h * P.form c (st c).point e j h

theorem mle_claimWeights (wf : WellFormed P) (shaped : Shaped P st)
    (length : ρ.length = P.witnessVars) (c : Fin P.claims) :
    mle (ρ.take (P.shape c).witness.vars) (claimWeights P st ch c) =
      ch.mu ^ (c : ℕ) * (∑ e ∈ range P.equations, ch.omega ^ e * equationAt st ch ρ c e +
        ch.omega ^ P.equations * ∑ y ∈ range (P.shape c).publicCells,
          eqAt (ρ.take (P.shape c).witness.vars) y * ch.gamma ^ y) := by
  have short : (P.shape c).witness.vars ≤ ρ.length := by
    rw [length]; exact (wf.witness_fits c).2.2
  have v_length : (ρ.take (P.shape c).witness.vars).length = (P.shape c).witness.vars := by
    simp [short]
  have lane_point : (ρ.take (P.shape c).witness.vars).take P.laneVars = ρ.take P.laneVars := by
    rw [List.take_take, min_eq_left (shaped.lanes_le c)]
  have fits := wf.blocks_fit c
  have pub := wf.public_fit c
  have decompose : claimWeights P st ch c = fun y => ch.mu ^ (c : ℕ) *
      (∑ e ∈ range P.equations, ch.omega ^ e *
          (if y < 2 ^ P.laneVars * (P.shape c).blocks then
            lanes P.lanes ch.delta (P.form c (st c).point e (y / 2 ^ P.laneVars)) (y % 2 ^ P.laneVars)
          else 0) +
        ch.omega ^ P.equations * (if y < (P.shape c).publicCells then ch.gamma ^ y else 0)) := by
    funext y
    simp only [claimWeights, claimWeight, Params.lanes]
    by_cases inside : y < 2 ^ P.laneVars * (P.shape c).blocks
    · simp only [inside, ite_true]
      congr 1
      split <;> simp
    · have outside : ¬ y < (P.shape c).publicCells := by
        simp only [Params.lanes] at pub
        omega
      simp [inside, outside]
  rw [decompose, mle_mul_left, mle_add, mle_finset_sum]
  congr 1
  congr 1
  · refine sum_congr rfl fun e _ => ?_
    rw [mle_mul_left]
    congr 1
    rw [mle_blocks P.laneVars (P.shape c).witness.vars (P.shape c).blocks _ v_length
      (shaped.lanes_le c) (shaped.blocks_le c)
      (fun j h => lanes P.lanes ch.delta (P.form c (st c).point e j) h), lane_point, equationAt]
    refine sum_congr rfl fun j _ => ?_
    congr 1
    have := value_mul_comm P.lanes ch.delta (P.form c (st c).point e j) (eqAt (ρ.take P.laneVars))
    simp only [Params.lanes] at this ⊢
    rw [this]
    exact sum_congr rfl fun h _ => by simp only [laneForm, Params.lanes]; ring
  · rw [mle_mul_left]
    congr 1
    rw [mle_eq_cubeSum, cubeSum_eq_sum_range, v_length]
    simp only [mul_ite, mul_zero]
    rw [← sum_range_add_sum_Ico _
      ((pub.trans (by simp only [Params.lanes]; exact fits)) :
        (P.shape c).publicCells ≤ 2 ^ (P.shape c).witness.vars)]
    rw [sum_eq_zero (s := Ico _ _) fun y member => by
      rw [ite_eq_right (by have := (mem_Ico.mp member).1; omega)], add_zero]
    exact sum_congr rfl fun y member => by rw [ite_eq_left (mem_range.mp member)]

theorem equations_split (wf : WellFormed P) (shaped : Shaped P st)
    (length : ρ.length = P.witnessVars) (c : Fin P.claims) :
    ∑ e ∈ range P.equations, ch.omega ^ e * equationAt st ch ρ c e =
      ∑ j ∈ range (P.shape c).blocks, eqAt (uses P ch ρ c).point j *
          ∑ h ∈ range P.lanes, laneForm P ch ρ h *
            folded P.lanes P.rows (fun i j h => ι (P.key i j h)) ch.omega (P.lanes * j + h) +
        (∑ h ∈ range P.lanes, laneForm P ch ρ h * evalLanes P st ch c h) *
          prefixEquality (uses P ch ρ c).point (P.shape c).blocks ((st c).point.drop P.laneVars) +
        ∑ m ∈ range P.matrices, ch.omega ^ (P.rows + 1 + m) *
          ∑ j ∈ range (P.shape c).blocks, eqAt (uses P ch ρ c).point j *
            ∑ h ∈ range P.lanes, laneForm P ch ρ h * P.matrixForm c (st c).point m j h := by
  have lanes_pos : 0 < P.lanes := pow_pos two_pos _
  have point_length : (uses P ch ρ c).point.length = (P.shape c).witness.vars - P.laneVars := by
    have := (wf.witness_fits c).2.2
    simp [uses, length, this]
  rw [Params.equations, sum_range_add, sum_range_succ]
  congr 1
  congr 1
  · simp only [equationAt, Params.form, mul_sum]
    rw [sum_comm]
    refine sum_congr rfl fun j _ => ?_
    rw [sum_comm]
    refine sum_congr rfl fun h member => ?_
    have lane := mem_range.mp member
    simp only [folded, show (P.lanes * j + h) / P.lanes = j by
      rw [Nat.add_comm, Nat.add_mul_div_left _ _ lanes_pos, Nat.div_eq_of_lt lane, zero_add],
      show (P.lanes * j + h) % P.lanes = h by
      rw [Nat.add_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt lane], mul_sum]
    refine sum_congr rfl fun e member' => ?_
    rw [ite_eq_left (mem_range.mp member')]
    ring
  · simp only [equationAt, Params.form, lt_irrefl, ite_false, ite_true]
    rw [prefixEquality_spec _ _ _ (by
      rw [point_length, List.length_drop]
      have := shaped.point_long c
      omega), min_eq_left (by rw [point_length]; exact shaped.blocks_le c), mul_sum, mul_sum]
    refine sum_congr rfl fun j _ => ?_
    rw [Finset.sum_mul, mul_sum, mul_sum]
    exact sum_congr rfl fun h _ => by simp only [evalLanes]; ring
  · refine sum_congr rfl fun m _ => ?_
    simp only [equationAt, Params.form, show ¬ P.rows + 1 + m < P.rows by omega,
      show ¬ P.rows + 1 + m = P.rows by omega, ite_false,
      show P.rows + 1 + m - P.rows - 1 = m by omega]

/-- `G(ρ)` is the key part plus the matrix part plus the closed part. -/
theorem mle_weightTable (wf : WellFormed P) (shaped : Shaped P st)
    (length : ρ.length = P.witnessVars) :
    mle ρ (weightTable P st τ ch) =
      keyTrue P ch ρ + matrixTrue P st ch ρ + closedPart P st τ ch ρ := by
  have each (c : Fin P.claims) : mle ρ (fun x => (P.shape c).witness.place
      (claimWeights P st ch c) x + lookupWeight P c * (P.shape c).counts.place
        (countWeights P τ ch c) x) =
      (uses P ch ρ c).coefficient * (∑ e ∈ range P.equations, ch.omega ^ e *
          equationAt st ch ρ c e + ch.omega ^ P.equations * ∑ y ∈ range (P.shape c).publicCells,
            eqAt (ρ.take (P.shape c).witness.vars) y * ch.gamma ^ y) +
        lookupWeight P c * ((P.shape c).counts.selector ρ *
          ∑ v ∈ range (2 * (P.shape c).bound + 1),
            eqAt (ρ.take (P.shape c).counts.vars) v * countWeights P τ ch c v) := by
    rw [mle_add, mle_mul_left, Slice.mle_place _ _ _ (by rw [length]; exact wf.witness_fits c),
      mle_claimWeights st ch ρ wf shaped length c]
    congr 1
    · simp only [uses]
      ring
    · by_cases table : (P.shape c).lookup = true
      · rw [Slice.mle_place _ _ _ (by rw [length]; exact wf.counts_fits c table), mle_eq_cubeSum,
          cubeSum_eq_sum_range]
        have cv : (ρ.take (P.shape c).counts.vars).length = (P.shape c).counts.vars := by
          have := (wf.counts_fits c table).2.2
          simp [length, this]
        rw [cv, ← sum_range_add_sum_Ico _ (wf.table_fit c table),
          sum_eq_zero (s := Ico _ _) fun v member => by
            simp only [countWeights]
            rw [ite_eq_right (by have := (mem_Ico.mp member).1; omega), mul_zero], add_zero]
      · simp [lookupWeight, table]
  unfold weightTable
  rw [mle_finset_sum, sum_congr rfl fun c _ => each c, keyTrue, matrixTrue, closedPart, keyPart]
  simp only [equations_split st ch ρ wf shaped length]
  rw [← sum_add_distrib, ← sum_add_distrib]
  refine sum_congr rfl fun c _ => ?_
  rw [show (uses P ch ρ c).blocks = (P.shape c).blocks from rfl]
  ring

/-- Column `k` of `V` on the span at `ρ`. -/
def columnAt (P : Params F K) (V : ℕ → F) (ρ : List K) (k : ℕ) : K :=
  mle (ρ.take P.spanVars) (fun y => ι (V (k * 2 ^ P.spanVars + y)))

omit [Field F] [Algebra F K] in
theorem mle_checkTable (wf : WellFormed P) (length : ρ.length = P.witnessVars) :
    mle ρ (checkTable P ch) = checkAt P ch ρ := by
  unfold checkTable checkAt
  rw [mle_finset_sum]
  refine sum_congr rfl fun c _ => ?_
  rw [mle_mul_left]
  by_cases table : (P.shape c).lookup = true
  · congr 1
    rw [Slice.mle_place _ _ _ (by rw [length]; exact wf.witness_fits c), mle_mul_left,
      mle_eqAt _ _ (by
        have := (wf.witness_fits c).2.2
        have := (wf.span_fit c table).2
        simp [Challenges.beta, *])]
    ring
  · simp [lookupWeight, table]

omit [Field F] [Algebra F K] in
theorem mle_sumTable (wf : WellFormed P) (length : ρ.length = P.witnessVars) :
    mle ρ (sumTable P ch) = sumAt P ch ρ := by
  unfold sumTable sumAt
  rw [mle_finset_sum]
  refine sum_congr rfl fun c _ => ?_
  rw [mle_mul_left, Slice.mle_place _ _ _ (by rw [length]; exact wf.witness_fits c), mle_const]
  ring

theorem mle_inverseTable (wf : WellFormed P) (length : ρ.length = P.witnessVars) (k : ℕ) :
    mle ρ (inverseTable P V k) = lowAt P ρ * columnAt P V ρ k := by
  have place : inverseTable P V k = (⟨0, P.spanVars⟩ : Slice).place
      (fun y => ι (V (k * 2 ^ P.spanVars + y))) := by
    funext x
    simp [inverseTable, Slice.place]
  rw [place, Slice.mle_place _ _ _ ⟨dvd_zero _, by
    rw [length, zero_add]; exact Nat.pow_le_pow_right two_pos wf.span_le, by
    rw [length]; exact wf.span_le⟩]
  simp [Slice.selector, lowAt, columnAt, eqAt_zero]

theorem mle_aTable (wf : WellFormed P) (length : ρ.length = P.witnessVars) :
    mle ρ (aTable P V ch) = lowAt P ρ * ∑ k, ch.psi k * columnAt P V ρ k := by
  unfold aTable
  rw [mle_finset_sum, mul_sum]
  refine sum_congr rfl fun k _ => ?_
  rw [mle_mul_left, mle_inverseTable V ρ wf length]
  ring

theorem mle_bTable (wf : WellFormed P) (length : ρ.length = P.witnessVars) :
    mle ρ (bTable P V τ ch) = lowAt P ρ * ∑ k, ch.psi k * ι (τ k) * columnAt P V ρ k := by
  unfold bTable
  rw [mle_finset_sum, mul_sum]
  refine sum_congr rfl fun k _ => ?_
  rw [mle_mul_left, mle_inverseTable V ρ wf length]
  ring

/-- Rust `inverse_point`: `V` at the span of `ρ`, the column point and zeros is the extension of
the columns at the column point. -/
theorem mle_V_opening (wf : WellFormed P) (length : ρ.length = P.witnessVars) (column : List K)
    (zeros : ℕ) :
    mle (ρ.take P.spanVars ++ column ++ List.replicate zeros 0) (fun x => ι (V x)) =
      mle column (columnAt P V ρ) := by
  have span_length : (ρ.take P.spanVars).length = P.spanVars := by
    simp [length, wf.span_le]
  rw [mle_inverse_point, span_length]
  congr 1
  funext k
  unfold columnAt
  congr 1
  funext y
  rw [Nat.add_comm, Nat.mul_comm]

end Proofs

end NightstreamFPrime.Spec.Compression.Verifier
