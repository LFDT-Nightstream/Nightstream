import NightstreamFPrime.Export.Stage1.PiDECEvaluationBlock
import NightstreamFPrime.Export.Stage1.PiDECCommitmentFold
import Mathlib.Algebra.Group.Fin.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.List.Dedup

/-!
Sum the existing row-block products only at blocks named by sparse entries.
Only block indices are deduplicated: every original entry remains in the
row form. The proof restores the complete block range from zero contributions
outside this derived support. No expected values or support premises are inputs.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationBlockSupport

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Layout.ProductionRelation

/-- Retain one visit per block, without changing or filtering row entries. -/
def blockIndices {columns : Nat} (form : SparseForm columns) : List Nat :=
  (form.entries.map fun entry => entry.column.val / ringDegree).dedup

private def step {columns : Nat} (form : SparseForm columns)
    (children : Nat → Vector StoredRing productionGlobalParams.k)
    (initial : Vector StoredRing productionGlobalParams.k) (block : Nat) :
    Vector StoredRing productionGlobalParams.k :=
  let products := PiDECEvaluationBlock.rowBlock form block (children block)
  Vector.ofFn fun child =>
    PiDECCommitmentFold.add (initial.get child) (products.get child)

private theorem step_value {columns : Nat} (form : SparseForm columns)
    (children : Nat → Vector StoredRing productionGlobalParams.k)
    (initial : Vector StoredRing productionGlobalParams.k) (block : Nat)
    (child : Fin productionGlobalParams.k) (lane : Fin ringDegree) :
    ((step form children initial block).get child).get lane =
      (initial.get child).get lane +
        ((PiDECEvaluationBlock.rowBlock form block (children block)).get child).get lane := by
  change ((Vector.ofFn (fun selected : Fin productionGlobalParams.k =>
    PiDECCommitmentFold.add (initial.get selected)
      ((PiDECEvaluationBlock.rowBlock form block (children block)).get selected)))[child.val]).get lane = _
  rw [Vector.getElem_ofFn]
  exact congrFun (PiDECCommitmentFold.add_value _ _) lane

/-- Share each computed block product across all child accumulators. -/
def kernel {columns : Nat} (form : SparseForm columns)
    (children : Nat → Vector StoredRing productionGlobalParams.k) :
    Vector StoredRing productionGlobalParams.k :=
  (blockIndices form).foldl (step form children)
    (Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero)

private theorem fold_value {columns : Nat} (form : SparseForm columns)
    (children : Nat → Vector StoredRing productionGlobalParams.k)
    (indices : List Nat) (initial : Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (lane : Fin ringDegree) :
    ((indices.foldl (step form children) initial).get child).get lane =
      (initial.get child).get lane +
        (indices.map fun block =>
          ((PiDECEvaluationBlock.rowBlock form block (children block)).get child).get lane).sum := by
  induction indices generalizing initial with
  | nil => simp only [List.foldl_nil, List.map_nil, List.sum_nil, add_zero]
  | cons block indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis, step_value,
        List.map_cons, List.sum_cons]
      exact add_assoc _ _ _

private theorem kernel_value {columns : Nat} (form : SparseForm columns)
    (children : Nat → Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (lane : Fin ringDegree) :
    ((kernel form children).get child).get lane =
      ((blockIndices form).map fun block =>
        ((PiDECEvaluationBlock.rowBlock form block (children block)).get child).get lane).sum := by
  rw [kernel, fold_value]
  have initialZero :
      (((Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero).get child).get lane) =
        (0 : F) := by
    change ((Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero)[child.val]).get lane = 0
    rw [Vector.getElem_replicate]
    exact congrFun PiDECCommitmentFold.zero_value lane
  rw [initialZero, zero_add]

private theorem sparseFold_zero {columns : Nat}
    (entries : List (SparseEntry columns)) (term : SparseEntry columns → F)
    (vanishes : ∀ entry ∈ entries, term entry = 0) (initial : F) :
    entries.foldl (fun total entry => total + term entry) initial = initial := by
  induction entries generalizing initial with
  | nil => rfl
  | cons entry entries inductionHypothesis =>
      rw [List.foldl_cons, vanishes entry (List.mem_cons_self), add_zero]
      exact inductionHypothesis
        (fun current member => vanishes current (List.mem_cons_of_mem _ member)) initial

/-- Every block absent from the entry-derived support has zero contribution,
for all child blocks, including nonzero carried tails. -/
theorem rowBlock_zero_of_not_mem {columns : Nat} (form : SparseForm columns)
    (block : Nat) (outside : block ∉ blockIndices form)
    (children : Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (lane : Fin ringDegree) :
    ((PiDECEvaluationBlock.rowBlock form block children).get child).get lane = 0 := by
  rw [PiDECEvaluationBlock.rowBlock_value]
  unfold SparseForm.evalSparse
  apply sparseFold_zero
  intro entry member
  have different : entry.column.val / ringDegree ≠ block := by
    intro equal
    apply outside
    apply List.mem_dedup.mpr
    exact List.mem_map.mpr ⟨entry, member, equal⟩
  simp only [if_neg different, Fin.mul_zero]

private theorem blockIndices_lt {columns blocks : Nat} (form : SparseForm columns)
    (fits : columns ≤ blocks * ringDegree) (block : Nat)
    (member : block ∈ blockIndices form) : block < blocks := by
  rcases List.mem_map.mp (List.mem_dedup.mp member) with ⟨entry, _, equal⟩
  have columnLt := entry.column.isLt
  subst block
  simp only [ringDegree] at fits ⊢
  omega

private theorem sumRange_eq_finset (blocks : Nat) (term : Nat → F) :
    sumRange baseOps blocks term = (Finset.range blocks).sum term := by
  induction blocks with
  | zero => simp only [sumRange, Finset.range_zero, Finset.sum_empty]; rfl
  | succ blocks inductionHypothesis =>
      change sumRange baseOps blocks term + term blocks = _
      rw [inductionHypothesis, Finset.sum_range_succ]

/-- The stored sparse traversal equals the complete block sum coefficientwise.
The only size premise ensures every logical column has a carrier block.
Block support and all omitted zero contributions are derived from the form. -/
theorem kernel_eq_fullBlockSum {columns blocks : Nat} (form : SparseForm columns)
    (fits : columns ≤ blocks * ringDegree)
    (children : Nat → Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (lane : Fin ringDegree) :
    ((kernel form children).get child).get lane =
      sumRange baseOps blocks (fun block =>
        ((PiDECEvaluationBlock.rowBlock form block (children block)).get child).get lane) := by
  let term : Nat → F := fun block =>
    ((PiDECEvaluationBlock.rowBlock form block (children block)).get child).get lane
  calc
    _ = ((blockIndices form).map term).sum := kernel_value form children child lane
    _ = (blockIndices form).toFinset.sum term :=
      (List.sum_toFinset term (List.nodup_dedup _)).symm
    _ = (Finset.range blocks).sum term := by
      apply Finset.sum_subset
      · intro block member
        exact Finset.mem_range.mpr
          (blockIndices_lt form fits block (List.mem_toFinset.mp member))
      · intro block _ outside
        exact rowBlock_zero_of_not_mem form block
          (fun member => outside (List.mem_toFinset.mpr member))
          (children block) child lane
    _ = _ := (sumRange_eq_finset blocks term).symm


open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism

private theorem entries_fold_eq_sum {Index : Type} (entries : List Index)
    (term : Index → F) (initial : F) :
    entries.foldl (fun total entry => total + term entry) initial =
      initial + (entries.map term).sum := by
  induction entries generalizing initial with
  | nil => simp only [List.foldl_nil, List.map_nil, List.sum_nil, add_zero]
  | cons entry entries inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis, List.map_cons, List.sum_cons]
      exact add_assoc _ _ _

private theorem evalSparse_eq_entries_sum {columns : Nat}
    (form : SparseForm columns) (read : Fin columns → F) :
    form.evalSparse read =
      (form.entries.map fun entry => entry.coefficient * read entry.column).sum := by
  simpa only [SparseForm.evalSparse, zero_add] using
    entries_fold_eq_sum form.entries
      (fun entry => entry.coefficient * read entry.column) 0

private theorem sum_entries_swap {columns : Nat} (indices : Finset Nat)
    (entries : List (SparseEntry columns)) (term : Nat → SparseEntry columns → F) :
    indices.sum (fun block => (entries.map (term block)).sum) =
      (entries.map fun entry => indices.sum fun block => term block entry).sum := by
  induction entries with
  | nil => simp only [List.map_nil, List.sum_nil, Finset.sum_const_zero]
  | cons entry entries inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, Finset.sum_add_distrib, inductionHypothesis]

/-- The complete stored row coefficient is the original sparse form evaluated
at its Phi81 kernel reads. Each entry selects exactly one support block;
repeated entries and cancelling coefficients remain unchanged. No width,
row-validity, support or expected-value premise is required. -/
theorem kernel_eq_evalSparse {columns : Nat} (form : SparseForm columns)
    (children : Nat → Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (output : Fin ringDegree) :
    ((kernel form children).get child).get output =
      form.evalSparse (fun column =>
        CarrierAction.kernelImage
          ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
          ((children (column.val / ringDegree)).get child).get output) := by
  let read : Fin columns → F := fun column =>
    CarrierAction.kernelImage
      ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
      ((children (column.val / ringDegree)).get child).get output
  let term : Nat → SparseEntry columns → F := fun block entry =>
    entry.coefficient * (if entry.column.val / ringDegree = block then
      CarrierAction.kernelImage
        ⟨entry.column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
        ((children block).get child).get output
      else 0)
  calc
    _ = ((blockIndices form).map fun block =>
        ((PiDECEvaluationBlock.rowBlock form block (children block)).get child).get output).sum :=
      kernel_value form children child output
    _ = (blockIndices form).toFinset.sum (fun block =>
        ((PiDECEvaluationBlock.rowBlock form block (children block)).get child).get output) :=
      (List.sum_toFinset _ (List.nodup_dedup _)).symm
    _ = (blockIndices form).toFinset.sum (fun block =>
        (form.entries.map (term block)).sum) := by
      apply Finset.sum_congr rfl
      intro block _
      exact (PiDECEvaluationBlock.rowBlock_value form block
        (children block) child output).trans (evalSparse_eq_entries_sum form _)
    _ = (form.entries.map fun entry =>
        (blockIndices form).toFinset.sum fun block => term block entry).sum :=
      sum_entries_swap (blockIndices form).toFinset form.entries term
    _ = (form.entries.map fun entry => entry.coefficient * read entry.column).sum := by
      apply congrArg List.sum
      apply List.map_congr_left
      intro entry member
      have selected : entry.column.val / ringDegree ∈ (blockIndices form).toFinset := by
        apply List.mem_toFinset.mpr
        apply List.mem_dedup.mpr
        exact List.mem_map.mpr ⟨entry, member, rfl⟩
      have others : ∀ block ∈ (blockIndices form).toFinset,
          block ≠ entry.column.val / ringDegree → term block entry = 0 := by
        intro block _ different
        dsimp only [term]
        rw [if_neg (Ne.symm different), Fin.mul_zero]
      calc
        _ = term (entry.column.val / ringDegree) entry :=
          Finset.sum_eq_single_of_mem (entry.column.val / ringDegree) selected others
        _ = entry.coefficient * read entry.column := by
          dsimp only [term, read]
          rw [if_pos rfl]
    _ = _ := (evalSparse_eq_entries_sum form read).symm

end NightstreamFPrime.Export.Stage1.PiDECEvaluationBlockSupport
