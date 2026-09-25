import NightstreamFPrime.Layout.ProductionRelation.SparseEvaluation
import NightstreamFPrime.Export.Stage1.PiDECCommitmentBlock
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFLaws

/-!
One sparse Phi81 row restricted to one complete carrier block. The stored
bar transform is shared by all sixteen child products. No point weighting,
row accumulation, expected evaluation, or IO is defined here.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationBlock

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Layout.ProductionRelation

/-- A stored entry updates this block only. Repeated columns are retained. -/
private def addEntry {columns : Nat} (block : Nat)
    (initial : StoredRing) (entry : SparseEntry columns) : StoredRing :=
  if entry.column.val / ringDegree = block then
    let lane : Fin ringDegree :=
      ⟨entry.column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
    Vector.ofFn fun output =>
      initial.get output +
        entry.coefficient * Phi81CoefficientKernel.nativeBarEntry output lane
  else initial

private theorem addEntry_get {columns : Nat} (block : Nat)
    (initial : StoredRing) (entry : SparseEntry columns)
    (output : Fin ringDegree) :
    (addEntry block initial entry).get output =
      initial.get output + entry.coefficient *
        (if entry.column.val / ringDegree = block then
          Phi81CoefficientKernel.nativeBarEntry output
            ⟨entry.column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
        else 0) := by
  by_cases same : entry.column.val / ringDegree = block
  · simp only [addEntry, if_pos same]
    change (Vector.ofFn (fun lane : Fin ringDegree =>
      initial.get lane + entry.coefficient *
        Phi81CoefficientKernel.nativeBarEntry lane
          ⟨entry.column.val % ringDegree, Nat.mod_lt _ (by decide)⟩))[output.val] = _
    rw [Vector.getElem_ofFn]
  · simp only [addEntry, if_neg same, Fin.mul_zero, Fin.add_zero]

private theorem foldEntries_get {columns : Nat}
    (entries : List (SparseEntry columns)) (block : Nat)
    (initial : StoredRing) (output : Fin ringDegree) :
    (entries.foldl (addEntry block) initial).get output =
      entries.foldl (fun total entry => total + entry.coefficient *
        (if entry.column.val / ringDegree = block then
          Phi81CoefficientKernel.nativeBarEntry output
            ⟨entry.column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
        else 0)) (initial.get output) := by
  induction entries generalizing initial with
  | nil => rfl
  | cons entry entries inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [inductionHypothesis, addEntry_get]

/-- Compute the bar transform of this sparse row's selected complete block.
Entries outside the block leave the stored accumulator unchanged. -/
def barBlock {columns : Nat} (form : SparseForm columns)
    (block : Nat) : StoredRing :=
  form.entries.foldl (addEntry block) (Vector.replicate ringDegree 0)

/-- Every stored lane is the exact sparse bar sum. This includes duplicate
columns, cancelling coefficients, empty forms and blocks with no entries. -/
theorem barBlock_value {columns : Nat} (form : SparseForm columns)
    (block : Nat) (output : Fin ringDegree) :
    (barBlock form block).get output =
      form.evalSparse (fun column =>
        if column.val / ringDegree = block then
          Phi81CoefficientKernel.nativeBarEntry output
            ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
        else 0) := by
  have zero : (Vector.replicate ringDegree (0 : F)).get output = 0 := by
    change (Vector.replicate ringDegree (0 : F))[output.val] = 0
    rw [Vector.getElem_replicate]
  simpa only [barBlock, SparseForm.evalSparse, zero] using
    foldEntries_get form.entries block (Vector.replicate ringDegree 0) output

private theorem addEntry_product {columns : Nat} (block : Nat)
    (initial : StoredRing) (entry : SparseEntry columns)
    (right : RingF) (output : Fin ringDegree) :
    ringFMul (addEntry block initial entry).get right output =
      ringFMul initial.get right output + entry.coefficient *
        (if entry.column.val / ringDegree = block then
          CarrierAction.kernelImage
            ⟨entry.column.val % ringDegree, Nat.mod_lt _ (by decide)⟩ right output
        else 0) := by
  by_cases same : entry.column.val / ringDegree = block
  · have transformed : (addEntry block initial entry).get =
        ringFAdd initial.get (CarrierAction.ringFScale entry.coefficient
          (Phi81CoefficientKernel.barBasis
            ⟨entry.column.val % ringDegree, Nat.mod_lt _ (by decide)⟩)) := by
      funext lane
      rw [addEntry_get, if_pos same]
      rfl
    rw [transformed, CarrierAction.ringFMul_add_left,
      CarrierAction.ringFMul_scale_left,
      ← CarrierAction.kernelImage_eq_ringFMul, if_pos same]
    rfl
  · simp only [addEntry, if_neg same, Fin.mul_zero, Fin.add_zero]

private theorem foldEntries_product {columns : Nat}
    (entries : List (SparseEntry columns)) (block : Nat)
    (initial : StoredRing) (right : RingF) (output : Fin ringDegree) :
    ringFMul (entries.foldl (addEntry block) initial).get right output =
      entries.foldl (fun total entry => total + entry.coefficient *
        (if entry.column.val / ringDegree = block then
          CarrierAction.kernelImage
            ⟨entry.column.val % ringDegree, Nat.mod_lt _ (by decide)⟩ right output
        else 0)) (ringFMul initial.get right output) := by
  induction entries generalizing initial with
  | nil => rfl
  | cons entry entries inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [inductionHypothesis, addEntry_product]

/-- Apply the existing ring product to the computed sparse bar transform.
Linearity is proved over the stored entry list, with no distinctness premise. -/
private theorem barBlock_product {columns : Nat} (form : SparseForm columns)
    (block : Nat) (right : RingF) (output : Fin ringDegree) :
    ringFMul (barBlock form block).get right output =
      form.evalSparse (fun column =>
        if column.val / ringDegree = block then
          CarrierAction.kernelImage
            ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩ right output
        else 0) := by
  have zero : (Vector.replicate ringDegree (0 : F)).get = ringFZero := by
    funext lane
    change (Vector.replicate ringDegree (0 : F))[lane.val] = 0
    rw [Vector.getElem_replicate]
  rw [barBlock, foldEntries_product, zero, RingFLaws.ringFMul_zero_left]
  rfl

/-- Share this row's computed bar block across the sixteen complete child
blocks. The product implementation owns exact zero-child omission. -/
def rowBlock {columns : Nat} (form : SparseForm columns) (block : Nat)
    (children : Vector StoredRing productionGlobalParams.k) :
    Vector StoredRing productionGlobalParams.k :=
  PiDECCommitmentBlock.products (barBlock form block) children

/-- Each returned coefficient is the existing sparse Phi81 kernel action
on the same complete child block. No norm, expected output, opening, point,
or row-generation premise is used. This is one block contribution only. -/
theorem rowBlock_value {columns : Nat} (form : SparseForm columns) (block : Nat)
    (children : Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (output : Fin ringDegree) :
    ((rowBlock form block children).get child).get output =
      form.evalSparse (fun column =>
        if column.val / ringDegree = block then
          CarrierAction.kernelImage
            ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
            (children.get child).get output
        else 0) := by
  rw [rowBlock, PiDECCommitmentBlock.products_value, barBlock_product]

end NightstreamFPrime.Export.Stage1.PiDECEvaluationBlock
