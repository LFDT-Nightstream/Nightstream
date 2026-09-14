import NightstreamFPrime.Export.Stage1.PiDECEvaluationBlock

/-!
Regroup one block-filtered sparse evaluation into the canonical 54-lane
coefficient sum. Duplicate entries are combined by SparseForm.coefficient;
positions beyond the logical width contribute zero. No selected matrix or
assignment representation is introduced.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationSparseBlock

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)

private theorem sumRange_add (count : Nat) (left right : Nat → F) :
    sumRange baseOps count (fun index => left index + right index) =
      sumRange baseOps count left + sumRange baseOps count right := by
  induction count with
  | zero => exact (Fin.zero_add (0 : F)).symm
  | succ count inductionHypothesis =>
      change sumRange baseOps count (fun index => left index + right index) +
          (left count + right count) =
        (sumRange baseOps count left + left count) +
          (sumRange baseOps count right + right count)
      rw [inductionHypothesis]
      abel

private theorem selected_flat_iff {columns : Nat} (selected : Fin columns)
    (block index : Nat) (indexLt : index < ringDegree) :
    selected.val = block * ringDegree + index ↔
      selected.val / ringDegree = block ∧ index = selected.val % ringDegree := by
  simp only [ringDegree] at indexLt ⊢
  omega

private theorem singleton_block_sum {columns : Nat} (selected : Fin columns)
    (value : F) (block : Nat) (term : Fin ringDegree → F) :
    sumRange baseOps ringDegree (fun index =>
      if indexLt : index < ringDegree then
        if within : block * ringDegree + index < columns then
          (SparseForm.singleton selected value).coefficient
              ⟨block * ringDegree + index, within⟩ * term ⟨index, indexLt⟩
        else 0
      else 0) =
      value * (if selected.val / ringDegree = block then
        term ⟨selected.val % ringDegree, Nat.mod_lt _ (by decide)⟩
      else 0) := by
  let selectedLane : Fin ringDegree :=
    ⟨selected.val % ringDegree, Nat.mod_lt _ (by decide)⟩
  by_cases same : selected.val / ringDegree = block
  · rw [if_pos same]
    calc
      _ = sumRange baseOps ringDegree (fun index =>
          if index = selectedLane.val then value * term selectedLane else 0) := by
        apply sumRange_congr baseOps ringDegree
        intro index indexLt
        rw [dif_pos indexLt]
        by_cases within : block * ringDegree + index < columns
        · rw [dif_pos within, SparseForm.singleton_coefficient]
          by_cases equal : index = selectedLane.val
          · have flat : selected = ⟨block * ringDegree + index, within⟩ :=
              Fin.ext ((selected_flat_iff selected block index indexLt).mpr
                ⟨same, equal⟩)
            have laneEqual : (⟨index, indexLt⟩ : Fin ringDegree) = selectedLane :=
              Fin.ext equal
            rw [if_pos flat, if_pos equal, laneEqual]
          · have different : selected ≠ ⟨block * ringDegree + index, within⟩ := by
              intro flat
              exact equal ((selected_flat_iff selected block index indexLt).mp
                (congrArg Fin.val flat)).2
            rw [if_neg different, if_neg equal, Fin.zero_mul]
        · have different : index ≠ selectedLane.val := by
            intro equal
            have flat := (selected_flat_iff selected block index indexLt).mpr
              ⟨same, equal⟩
            apply within
            rw [← flat]
            exact selected.isLt
          rw [dif_neg within, if_neg different]
      _ = value * term selectedLane :=
        sumRange_select baseOps baseLaws ringDegree selectedLane.val
          (fun _ => value * term selectedLane) selectedLane.isLt
  · rw [if_neg same, Fin.mul_zero]
    apply sumRange_eq_zero baseOps baseLaws ringDegree
    intro index indexLt
    change _ = (0 : F)
    rw [dif_pos indexLt]
    by_cases within : block * ringDegree + index < columns
    · have different : selected ≠ ⟨block * ringDegree + index, within⟩ := by
        intro flat
        exact same ((selected_flat_iff selected block index indexLt).mp
          (congrArg Fin.val flat)).1
      rw [dif_pos within, SparseForm.singleton_coefficient,
        if_neg different, Fin.zero_mul]
    · rw [dif_neg within]

/-- The sparse block filter is exactly the 54-lane coefficient contraction.
The bounded-index guard is the public sumRange spelling of PiRLC's private
sumFinF. The inner guard includes every final partial-block zero position.
No distinctness, divisibility, selected-matrix or assignment premise is used. -/
theorem evalSparse_eq_blockSum {columns : Nat} (form : SparseForm columns)
    (block : Nat) (term : Fin ringDegree → F) :
    form.evalSparse (fun column =>
      if column.val / ringDegree = block then
        term ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
      else 0) =
    sumRange baseOps ringDegree (fun index =>
      if indexLt : index < ringDegree then
        if within : block * ringDegree + index < columns then
          form.coefficient ⟨block * ringDegree + index, within⟩ *
            term ⟨index, indexLt⟩
        else 0
      else 0) := by
  rw [SparseForm.evalSparse_eq_eval]
  rcases form with ⟨entries⟩
  induction entries with
  | nil =>
      rw [show (⟨[]⟩ : SparseForm columns) = SparseForm.empty from rfl,
        SparseForm.empty_eval]
      symm
      apply sumRange_eq_zero baseOps baseLaws ringDegree
      intro index indexLt
      change _ = (0 : F)
      rw [dif_pos indexLt]
      by_cases within : block * ringDegree + index < columns
      · rw [dif_pos within, SparseForm.empty_coefficient, Fin.zero_mul]
      · rw [dif_neg within]
  | cons entry entries inductionHypothesis =>
      have asAdd : (⟨entry :: entries⟩ : SparseForm columns) =
          SparseForm.add (SparseForm.singleton entry.column entry.coefficient)
            ⟨entries⟩ := rfl
      rw [asAdd, SparseForm.add_eval, SparseForm.singleton_eval,
        inductionHypothesis, ← singleton_block_sum entry.column entry.coefficient block term,
        ← sumRange_add]
      apply sumRange_congr baseOps ringDegree
      intro index indexLt
      simp only [dif_pos indexLt]
      by_cases within : block * ringDegree + index < columns
      · simp only [dif_pos within, SparseForm.add_coefficient, add_mul]
      · simp only [dif_neg within, Fin.add_zero]

/-- The computed child product is the canonical source-coefficient contraction
for one complete block, including zero positions beyond the logical width. -/
theorem rowBlock_coefficient_value {columns : Nat} (form : SparseForm columns)
    (block : Nat) (children : Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (output : Fin ringDegree) :
    ((PiDECEvaluationBlock.rowBlock form block children).get child).get output =
      sumRange baseOps ringDegree (fun index =>
        if indexLt : index < ringDegree then
          if within : block * ringDegree + index < columns then
            form.coefficient ⟨block * ringDegree + index, within⟩ *
              CarrierAction.kernelImage ⟨index, indexLt⟩ (children.get child).get output
          else 0
        else 0) := by
  rw [PiDECEvaluationBlock.rowBlock_value]
  exact evalSparse_eq_blockSum form block
    (fun lane => CarrierAction.kernelImage lane (children.get child).get output)

end NightstreamFPrime.Export.Stage1.PiDECEvaluationSparseBlock
