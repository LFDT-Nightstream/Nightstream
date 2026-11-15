import NightstreamFPrime.Spec.Compression.Claim.Evaluation

/-! The witness of a claim with `blocks` blocks as an assignment of a ring relation: the cells of
its slice of `W`, then zeros (`sliceAssignment`). On such an assignment, the Ajtai rows
(`ajtaiRow_slice`), the padded evaluation `Eval_K` (`evaluate_pad`) and the matrix evaluations
`Eval_A` (`evaluate_matrix`) of `Spec.CE.Holds` are the left sides of the ring equations of the
compression argument (SPEC.md, Algorithm 9, step 7).

The verifier holds the rows of a relation as the dense rows `M` of `Matrices` (Rust
`FactoredRows`). `Represents M mats` states that they are the matrices `mats` of the relation on
the cells of the claim's blocks, with zero rows of `mats` from `M.rows`. -/

namespace NightstreamFPrime.Spec.Compression.Claim

open Finset NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Algebra
open NightstreamFPrime.Spec.Compression.Verifier NightstreamFPrime.Spec.Compression.Production
open NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

local notation "ι" => algebraMap F K

/-- The witness of a claim as an assignment of `C` cells: cells `offset` to `offset + width − 1` of
`W`, then zeros. -/
def sliceAssignment {C : ℕ} (W : ℕ → F) (offset width : ℕ) : Fin C → F :=
  fun column => if (column : ℕ) < width then W (offset + column) else 0

theorem cell_sliceAssignment {C : ℕ} (W : ℕ → F) (offset width x : ℕ) :
    cell (sliceAssignment (C := C) W offset width) x =
      if x < C ∧ x < width then W (offset + x) else 0 := by
  unfold cell sliceAssignment
  by_cases inside : x < C <;> by_cases within : x < width <;> simp [inside, within]

theorem block_lt {D blocks j h : ℕ} (j_lt : j < blocks) (h_lt : h < D) : D * j + h < D * blocks :=
  calc D * j + h < D * j + D := by omega
    _ = D * (j + 1) := by ring
    _ ≤ D * blocks := Nat.mul_le_mul_left D j_lt

theorem conjugate_congr {R : Type} [CommRing R] {D l : ℕ} (lane : l < D) {w w' : ℕ → R}
    (same : ∀ i < D, w i = w' i) : conjugate D w l = conjugate D w' l := by
  simp only [Algebra.conjugate]
  split_ifs <;> rw [same _ (by omega)]

/-- The blocks past the slice are zero, and the blocks of the slice are the cells of `W`. -/
theorem sum_blocks_slice {C nb blocks : ℕ} (cover : ringDegree * blocks ≤ C) (le : blocks ≤ nb)
    (W : ℕ → F) (offset : ℕ) (a a' : ℕ → ℕ → K)
    (same : ∀ j < blocks, ∀ i < ringDegree, a j i = a' j i) {o : ℕ} (lane : o < ringDegree) :
    ∑ j ∈ range nb, mul ringDegree (a j)
        (fun h => ι (cell (sliceAssignment (C := C) W offset (ringDegree * blocks))
          (ringDegree * j + h))) o =
      ∑ j ∈ range blocks, mul ringDegree (a' j) (fun h => ι (W (offset + (ringDegree * j + h)))) o := by
  rw [← sum_range_add_sum_Ico _ le, sum_eq_zero (s := Ico blocks nb) fun j member => ?_, add_zero]
  · refine sum_congr rfl fun j member => ?_
    have j_lt := mem_range.mp member
    rw [mul_congr_left lane (same j j_lt)]
    refine mul_congr_right _ _ _ fun h h_lt => ?_
    have within := block_lt j_lt h_lt
    rw [cell_sliceAssignment, ite_eq_left ⟨by omega, within⟩]
  · have j_ge := (mem_Ico.mp member).1
    rw [mul_congr_right _ _ _ (b' := fun _ => 0) fun h _ => by
      rw [cell_sliceAssignment, ite_eq_right (fun inside => by
        have := Nat.mul_le_mul_left ringDegree j_ge
        omega), map_zero]]
    simp [mul]

/-- On a slice, row `row` of the Ajtai commitment is `Σ_{j < blocks} mul(A_{row, j}, z_j)`. -/
theorem ajtaiRow_slice {shape : RingRelation.Shape} {rows blocks : ℕ}
    (key : Commitment.Key shape rows) (W : ℕ → F) (offset : ℕ)
    (cover : ringDegree * blocks ≤ shape.carrierWidth) (row : Fin rows) (lane : Fin ringDegree) :
    ι (Commitment.ajtaiRow key (sliceAssignment W offset (ringDegree * blocks)) row lane) =
      ∑ j ∈ range blocks, mul ringDegree (fun h => ι (keyOf key row j h))
        (fun h => ι (W (offset + (ringDegree * j + h)))) lane := by
  have le : blocks ≤ ConcreteRingColumnLayout.blockCount shape.carrierWidth := by
    change blocks ≤ ConcreteRingColumnLayout.blockCount (RingCarrierLayout.carrierWidth _)
    rw [RingCarrierLayout.blockCount_carrierWidth]
    change ringDegree * blocks ≤ ConcreteRingColumnLayout.blockCount _ * ringDegree at cover
    rw [Nat.mul_comm] at cover
    exact Nat.le_of_mul_le_mul_right cover (by decide)
  rw [ajtaiRow_apply, map_sum]
  simp only [Claim.map_mul]
  exact sum_blocks_slice cover le W offset _ _ (fun _ _ _ _ => rfl) lane.isLt

theorem paddedIdentity_cell {v C : ℕ} (fits : C ≤ 2 ^ v) {i x : ℕ} (i_lt : i < 2 ^ v)
    (x_lt : x < C) :
    cell ((CanonicalRowLayout.layout v C fits).paddedIdentityEntry baseOps.zero baseOps.one
        (NumericBooleanDomain.vertex v ⟨i, i_lt⟩)) x = if i = x then 1 else 0 := by
  simp only [cell, x_lt, dite_true]
  by_cases live : i < C
  · have vertex : NumericBooleanDomain.vertex v ⟨i, i_lt⟩ =
        (CanonicalRowLayout.layout v C fits).toVertex ⟨i, live⟩ := rfl
    rw [vertex, UnifiedSources.ColumnLayout.paddedIdentityEntry_toVertex]
    by_cases same : i = x
    · subst same
      simp only [ite_true]
      rfl
    · rw [ite_eq_right (fun equal => same (congrArg Fin.val equal).symm), ite_eq_right same]
      rfl
  · have none := (CanonicalRowLayout.toColumn?_eq_none_iff v C fits
      (NumericBooleanDomain.vertex v ⟨i, i_lt⟩)).mpr (by
        rw [NumericBooleanDomain.index_vertex]
        exact Nat.le_of_not_lt live)
    have different : i ≠ x := by omega
    simp only [UnifiedSources.ColumnLayout.paddedIdentityEntry, none, different, ite_false]
    rfl

variable {v fc rc mc lw : ℕ} (mats : Fin mc → BooleanMatrix F v lw)
  (poly : CCSResidualTable.ConstraintPolynomial F mc)

/-- On a slice, lane `o` of the padded evaluation at `r` is lane `o` of
`Σ_{j < blocks} eq(r_hi, j) · conj(e) · z_j` with `e = Σ_h eq(r_lo, h) · X^h`, the left side of
`Eval_K`. -/
theorem evaluate_pad {blocks : ℕ} (fits : RingCarrierLayout.carrierWidth lw ≤ 2 ^ v)
    (cover : ringDegree * blocks ≤ RingCarrierLayout.carrierWidth lw) (W : ℕ → F) (offset : ℕ)
    (point : CubePoint K v) (o : Fin ringDegree) :
    (BooleanTable.tabulate fun vertex => K.embed (matrixVectorAt baseOps
        ((RingMatrixSource.source v fc rc mc lw mats poly).coefficientMatrixOf baseOps
          (fun row column => (CanonicalRowLayout.layout v (RingCarrierLayout.carrierWidth lw)
            fits).paddedIdentityEntry baseOps.zero baseOps.one row column) o)
        (sliceAssignment W offset (ringDegree * blocks)) vertex)).evaluate extensionOps point =
      ∑ j ∈ range blocks, mul ringDegree (fun h => eqAt (point.coordinates.drop 6) j *
          conjugate ringDegree (eqAt (point.coordinates.take 6)) h)
        (fun h => ι (W (offset + (ringDegree * j + h)))) o := by
  have le : blocks ≤ ConcreteRingColumnLayout.blockCount lw := by
    change ringDegree * blocks ≤ ConcreteRingColumnLayout.blockCount lw * ringDegree at cover
    rw [Nat.mul_comm] at cover
    exact Nat.le_of_mul_le_mul_right cover (by decide)
  rw [evaluate_coefficientMatrixOf]
  refine sum_blocks_slice cover le W offset _ _ (fun j j_lt i i_lt => ?_) o.isLt
  refine (conjugate_congr i_lt fun l l_lt => ?_).trans (congrFun (conjugate_smul _ _ _) i)
  have x_lt : ringDegree * j + l < RingCarrierLayout.carrierWidth lw :=
    lt_of_lt_of_le (block_lt j_lt l_lt) cover
  rw [sum_eq_single (ringDegree * j + l) (fun i' member other => ?_) (fun out =>
      absurd (mem_range.mpr (lt_of_lt_of_le x_lt fits)) out)]
  · simp only [atIndex, lt_of_lt_of_le x_lt fits, dite_true, paddedIdentity_cell fits _ x_lt,
      ite_true, map_one, mul_one]
    rw [← eqAt_split 6 point.coordinates (ringDegree * j + l)]
    have div : (ringDegree * j + l) / 2 ^ 6 = j := by
      rw [show (2 : ℕ) ^ 6 = ringDegree from rfl, Nat.mul_add_div (by decide),
        Nat.div_eq_of_lt l_lt, add_zero]
    have mod : (ringDegree * j + l) % 2 ^ 6 = l := by
      rw [show (2 : ℕ) ^ 6 = ringDegree from rfl, Nat.mul_add_mod, Nat.mod_eq_of_lt l_lt]
    rw [div, mod, mul_comm]
  · simp only [atIndex, mem_range.mp member, dite_true, paddedIdentity_cell fits _ x_lt,
      ite_eq_right other, map_zero, mul_zero]

/-- The dense rows `M` of a claim are the matrices `mats` of a relation on the cells of the claim's
blocks: each row below `M.rows` is the relation's row, and each row of the relation from `M.rows`
is zero on these cells. -/
structure Represents {t : ℕ} (M : Matrices F) (mats : Fin t → BooleanMatrix F v lw) : Prop where
  cover : ringDegree * (M.columns / ringDegree) ≤ RingCarrierLayout.carrierWidth lw
  rows_le : M.rows ≤ 2 ^ v
  entry : ∀ (m : Fin t) (row x : ℕ) (inside : row < 2 ^ v),
    x < ringDegree * (M.columns / ringDegree) →
      (if row < M.rows then M.entry m row x else 0) =
        if h : x < lw then mats m (NumericBooleanDomain.vertex v ⟨row, inside⟩) ⟨x, h⟩ else 0

/-- On a slice, lane `o` of the evaluation of matrix `m` at `r` is lane `o` of
`Σ_{j < blocks} conj(w_{m,j}) · z_j` with the rows `M`, the left side of `Eval_A`. -/
theorem evaluate_matrix {M : Matrices F} (represents : Represents M mats) (W : ℕ → F) (offset : ℕ) (point : CubePoint K v) (m : Fin mc) (o : Fin ringDegree) :
    (BooleanTable.tabulate fun vertex => K.embed (matrixVectorAt baseOps
        ((RingMatrixSource.source v fc rc mc lw mats poly).coefficientMatrix baseOps m o)
        (sliceAssignment W offset (ringDegree * (M.columns / ringDegree))) vertex)).evaluate
        extensionOps point =
      ∑ j ∈ range (M.columns / ringDegree), mul ringDegree (M.form ringDegree point.coordinates m j)
        (fun h => ι (W (offset + (ringDegree * j + h)))) o := by
  have cover := represents.cover
  have le : M.columns / ringDegree ≤ ConcreteRingColumnLayout.blockCount lw := by
    change ringDegree * _ ≤ ConcreteRingColumnLayout.blockCount lw * ringDegree at cover
    rw [Nat.mul_comm] at cover
    exact Nat.le_of_mul_le_mul_right cover (by decide)
  refine (evaluate_coefficientMatrixOf mats poly _ _ point o).trans
    (sum_blocks_slice cover le W offset _ _ (fun j j_lt i i_lt => ?_) o.isLt)
  refine conjugate_congr i_lt fun l l_lt => ?_
  have x_in := block_lt (D := ringDegree) j_lt l_lt
  have x_lt : ringDegree * j + l < RingCarrierLayout.carrierWidth lw := lt_of_lt_of_le x_in cover
  rw [← Algebra.sum_range_ite _ _ represents.rows_le]
  refine sum_congr rfl fun row member => ?_
  have row_lt := mem_range.mp member
  rw [← mul_ite_zero, ← map_zero (algebraMap F K), ← apply_ite, represents.entry m row _ row_lt x_in]
  simp only [atIndex, row_lt, dite_true, cell, x_lt]
  congr 2
  show RingCarrierLayout.extendMatrix 0 (mats m) _ _ = _
  by_cases logical : ringDegree * j + l < lw
  · simp only [RingCarrierLayout.extendMatrix, RingCarrierLayout.logicalColumn?, logical, dite_true]
  · simp only [RingCarrierLayout.extendMatrix, RingCarrierLayout.logicalColumn?, logical, dite_false]

end NightstreamFPrime.Spec.Compression.Claim
