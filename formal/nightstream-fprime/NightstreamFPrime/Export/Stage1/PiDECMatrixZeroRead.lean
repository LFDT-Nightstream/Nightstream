import NightstreamFPrime.Export.Stage1.PiDECMatrixSparseRange
import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocationRange
import NightstreamFPrime.Export.Stage1.PiDECMatrixSelectedBatch
import NightstreamFPrime.Export.Stage1.PiDECParentMagnitude
import Mathlib.Data.List.Basic

/-!
Both existing matrix range sums return zero under a zero scalar read.
The rows, interfaces, point and offsets are arbitrary. These arithmetic
identities do not replace any runtime row or interface load guard.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixZeroRead

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Export.Stage1.PiDECParentMagnitude (ifActive)

private theorem evalSparse_zero {columns : Nat} (form : SparseForm columns) :
    form.evalSparse (fun _ => 0) = 0 := by
  simp only [SparseForm.evalSparse, mul_zero, add_zero, List.foldl_fixed]

private theorem weighted_zero (weight : K) :
    extensionOps.mul weight (K.embed (0 : F)) = extensionOps.zero := by
  change extensionOps.mul weight (K.embed baseOps.zero) = extensionOps.zero
  rw [embed_zero, extensionLaws.mul_zero]

private theorem numericSum_zero (count : Nat) :
    NumericCompletionSum.numericSum extensionOps count (fun _ => extensionOps.zero) =
      extensionOps.zero := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [NumericCompletionSum.numericSum, Nat.fold_succ]
      change extensionOps.add
          (NumericCompletionSum.numericSum extensionOps count (fun _ => extensionOps.zero))
          extensionOps.zero = extensionOps.zero
      rw [inductionHypothesis, extensionLaws.add_zero]

private theorem numericSum_of_zero (count : Nat) (term : Nat → K)
    (zero : ∀ index, term index = extensionOps.zero) :
    NumericCompletionSum.numericSum extensionOps count term = extensionOps.zero := by
  rw [funext zero]
  exact numericSum_zero count

/-- Every matrix port and both coordinates of all 54 K coefficients are zero.
No source-row, parent, interface, or point correctness premise is required. -/
theorem sparse_sum_zero {columns arity count : Nat} (firstRow : Nat)
    (point : CubePoint K arity) (forms : Vector (MatrixProgram.RowForms columns) count)
    (port : Fin matrixCount) :
    ((PiDECMatrixSparseRange.sum firstRow point (fun _ _ => 0) forms).get port).toRing =
      ringKZero := by
  funext output
  rw [PiDECMatrixSparseRange.sum_value]
  change NumericCompletionSum.numericSum extensionOps count _ = extensionOps.zero
  apply numericSum_of_zero
  intro index
  by_cases live : index < count
  · simp only [dif_pos live, evalSparse_zero, weighted_zero]
  · simp only [dif_neg live, weighted_zero]

private theorem invocation_zero {columns arity : Nat} (firstRow : Nat)
    (point : CubePoint K arity) (interface : PoseidonSboxPlan.Interface columns)
    (port : Fin matrixCount) :
    ((PiDECMatrixInvocation.sum firstRow point
      (PiDECMatrixInvocation.prepare (fun _ _ => 0) interface)).get port).toRing =
      ringKZero := by
  funext output
  rw [PiDECMatrixInvocation.sum_prepare_value]
  change NumericCompletionSum.numericSum extensionOps 94 _ = extensionOps.zero
  apply numericSum_of_zero
  intro index
  by_cases live : index < 94
  · simp only [dif_pos live, evalSparse_zero, weighted_zero]
  · simp only [dif_neg live, weighted_zero]

/-- Any vector of loaded invocations contributes zero under the zero read.
The result retains every matrix port and every K coefficient explicitly. -/
theorem invocation_sum_zero {columns arity count : Nat} (firstRow : Nat)
    (point : CubePoint K arity)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count)
    (port : Fin matrixCount) :
    ((PiDECMatrixInvocationRange.sum firstRow point (fun _ _ => 0) interfaces).get port).toRing =
      ringKZero := by
  funext output
  rw [PiDECMatrixInvocationRange.sum, PiDECEvaluationBatch.sum_value]
  change NumericCompletionSum.numericSum extensionOps count _ = extensionOps.zero
  apply numericSum_of_zero
  intro index
  by_cases live : index < count
  · rw [dif_pos live, invocation_zero]
    rfl
  · rw [dif_neg live, PiDECEvaluationBatch.zero_value]
    rfl

private theorem parentRead_zero {count columns : Nat}
    (parents : Vector (Vector Int ringDegree) count) (child : Radix.ChildIndex)
    (below : PiDECParentMagnitude.maximumMagnitude parents < 2 ^ child.val) :
    PiDECMatrixSelectedBatch.intParentRead (columns := columns) parents.get child =
      fun _ _ => 0 := by
  funext output column
  dsimp only [PiDECMatrixSelectedBatch.intParentRead]
  split_ifs with live
  · rw [PiDECParentIntRead.sparseRead_eq_evalSparse]
    have digits : (fun input => PiDECParentIntRead.cachedDigit
        ((parents.get ⟨column.val / ringDegree, live⟩).get input) child) =
        fun _ => (0 : F) := by
      funext input
      exact PiDECParentMagnitude.cachedDigit_eq_zero parents child below _ input
    rw [digits, evalSparse_zero]
  · rfl

/-- The input-derived guard preserves the existing complete sparse range.
No external bound or expected matrix value is a premise. -/
theorem ifActive_sparse_value {blocks columns arity count : Nat}
    (parents : Vector (Vector Int ringDegree) blocks) (child : Radix.ChildIndex)
    (firstRow : Nat) (point : CubePoint K arity)
    (forms : Vector (MatrixProgram.RowForms columns) count) (port : Fin matrixCount) :
    ((ifActive (PiDECParentMagnitude.maximumMagnitude parents) child (fun _ =>
      PiDECMatrixSparseRange.sum firstRow point
        (PiDECMatrixSelectedBatch.intParentRead parents.get child) forms)).get port).toRing =
      ((PiDECMatrixSparseRange.sum firstRow point
        (PiDECMatrixSelectedBatch.intParentRead parents.get child) forms).get port).toRing := by
  unfold ifActive
  split_ifs with below
  · rw [PiDECEvaluationBatch.zero_value, parentRead_zero parents child below,
      sparse_sum_zero]
  · rfl

/-- The same guard preserves every loaded Poseidon invocation and matrix
coefficient. It does not assume that selectors or retained values are valid. -/
theorem ifActive_invocation_value {blocks columns arity count : Nat}
    (parents : Vector (Vector Int ringDegree) blocks) (child : Radix.ChildIndex)
    (firstRow : Nat) (point : CubePoint K arity)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count)
    (port : Fin matrixCount) :
    ((ifActive (PiDECParentMagnitude.maximumMagnitude parents) child (fun _ =>
      PiDECMatrixInvocationRange.sum firstRow point
        (PiDECMatrixSelectedBatch.intParentRead parents.get child) interfaces)).get port).toRing =
      ((PiDECMatrixInvocationRange.sum firstRow point
        (PiDECMatrixSelectedBatch.intParentRead parents.get child) interfaces).get port).toRing := by
  unfold ifActive
  split_ifs with below
  · rw [PiDECEvaluationBatch.zero_value, parentRead_zero parents child below,
      invocation_sum_zero]
  · rfl

end NightstreamFPrime.Export.Stage1.PiDECMatrixZeroRead
