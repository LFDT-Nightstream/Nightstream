import NightstreamFPrime.Lifecycle.ProductionKey

namespace NightstreamFPrime.Layout.ProductionRelation

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

structure SparseEntry (columns : Nat) where
  column : Fin columns
  coefficient : F
deriving Repr, DecidableEq

structure SparseForm (columns : Nat) where
  entries : List (SparseEntry columns)
deriving Repr, DecidableEq

namespace SparseForm

def empty {columns : Nat} : SparseForm columns := ⟨[]⟩

def singleton {columns : Nat} (column : Fin columns)
    (coefficient : F) : SparseForm columns :=
  ⟨[⟨column, coefficient⟩]⟩

def add {columns : Nat} (left right : SparseForm columns) :
    SparseForm columns :=
  ⟨left.entries ++ right.entries⟩

def scale {columns : Nat} (scalar : F) (form : SparseForm columns) :
    SparseForm columns :=
  ⟨form.entries.map fun entry =>
    ⟨entry.column, scalar * entry.coefficient⟩⟩

def coefficient {columns : Nat}
    (form : SparseForm columns) (column : Fin columns) : F :=
  form.entries.foldl (fun total entry =>
    if entry.column = column then total + entry.coefficient else total) 0

def eval {columns : Nat}
    (form : SparseForm columns) (assignment : Assignment F columns) : F :=
  (canonicalFinIndices columns).foldl (fun total column =>
    baseOps.add total (baseOps.mul (form.coefficient column) (assignment column)))
    baseOps.zero

@[simp] theorem empty_coefficient {columns : Nat} (column : Fin columns) :
    (empty : SparseForm columns).coefficient column = 0 := by
  rfl

@[simp] theorem singleton_coefficient {columns : Nat}
    (selected column : Fin columns) (value : F) :
    (singleton selected value).coefficient column =
      if selected = column then value else 0 := by
  simp [singleton, coefficient]

private theorem foldl_coefficient_from {columns : Nat}
    (entries : List (SparseEntry columns)) (column : Fin columns)
    (initial : F) :
    entries.foldl (fun total entry =>
        if entry.column = column then total + entry.coefficient else total)
        initial =
      initial +
        entries.foldl (fun total entry =>
          if entry.column = column then total + entry.coefficient else total) 0 := by
  induction entries generalizing initial with
  | nil => simp
  | cons entry rest inductionHypothesis =>
      simp only [List.foldl_cons]
      by_cases equal : entry.column = column
      · rw [ite_eq_left equal, ite_eq_left equal,
          inductionHypothesis (initial + entry.coefficient)]
        simp only [zero_add]
        rw [inductionHypothesis entry.coefficient]
        abel
      · rw [ite_eq_right equal, ite_eq_right equal, inductionHypothesis initial]

@[simp] theorem add_coefficient {columns : Nat}
    (left right : SparseForm columns) (column : Fin columns) :
    (add left right).coefficient column =
      left.coefficient column + right.coefficient column := by
  unfold add coefficient
  rw [List.foldl_append, foldl_coefficient_from]

private theorem foldl_scaled_entries {columns : Nat}
    (entries : List (SparseEntry columns)) (scalar : F)
    (column : Fin columns) (initial : F) :
    (entries.map fun entry =>
        SparseEntry.mk entry.column (scalar * entry.coefficient)).foldl
        (fun total entry =>
          if entry.column = column then total + entry.coefficient else total)
        initial =
      initial + scalar *
        entries.foldl (fun total entry =>
          if entry.column = column then total + entry.coefficient else total) 0 := by
  induction entries generalizing initial with
  | nil => simp
  | cons entry rest inductionHypothesis =>
      simp only [List.map_cons, List.foldl_cons]
      by_cases equal : entry.column = column
      · rw [ite_eq_left equal, ite_eq_left equal,
          inductionHypothesis (initial + scalar * entry.coefficient)]
        simp only [zero_add]
        rw [foldl_coefficient_from (initial := entry.coefficient)]
        rw [mul_add]
        exact baseLaws.add_assoc _ _ _
      · rw [ite_eq_right equal, ite_eq_right equal, inductionHypothesis initial]

@[simp] theorem scale_coefficient {columns : Nat}
    (scalar : F) (form : SparseForm columns) (column : Fin columns) :
    (scale scalar form).coefficient column =
      scalar * form.coefficient column := by
  unfold scale coefficient
  rw [foldl_scaled_entries]
  simp

@[simp] theorem empty_eval {columns : Nat}
    (assignment : Assignment F columns) :
    (empty : SparseForm columns).eval assignment = 0 := by
  unfold eval
  generalize canonicalFinIndices columns = indices
  induction indices with
  | nil => rfl
  | cons column rest inductionHypothesis =>
      rw [List.foldl_cons]
      simp [baseOps]

private theorem foldl_add_terms {Index : Type}
    (indices : List Index) (left right : Index → F)
    (leftInitial rightInitial : F) :
    indices.foldl (fun total index => total + (left index + right index))
        (leftInitial + rightInitial) =
      indices.foldl (fun total index => total + left index) leftInitial +
        indices.foldl (fun total index => total + right index) rightInitial := by
  induction indices generalizing leftInitial rightInitial with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [← inductionHypothesis
        (leftInitial := leftInitial + left index)
        (rightInitial := rightInitial + right index)]
      congr 1
      abel

@[simp] theorem add_eval {columns : Nat}
    (left right : SparseForm columns) (assignment : Assignment F columns) :
    (add left right).eval assignment =
      left.eval assignment + right.eval assignment := by
  unfold eval
  have combined := foldl_add_terms (canonicalFinIndices columns)
    (fun column => left.coefficient column * assignment column)
    (fun column => right.coefficient column * assignment column) 0 0
  simpa [add_coefficient, add_mul, baseOps] using combined

private theorem foldl_scale_terms {Index : Type}
    (indices : List Index) (scalar : F) (term : Index → F)
    (initial : F) :
    indices.foldl (fun total index => total + scalar * term index)
        (scalar * initial) =
      scalar * indices.foldl (fun total index => total + term index) initial := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [← mul_add]
      exact inductionHypothesis (initial + term index)

@[simp] theorem scale_eval {columns : Nat} (scalar : F)
    (form : SparseForm columns) (assignment : Assignment F columns) :
    (scale scalar form).eval assignment = scalar * form.eval assignment := by
  unfold eval
  have scaled := foldl_scale_terms (canonicalFinIndices columns) scalar
    (fun column => form.coefficient column * assignment column) 0
  simpa [scale_coefficient, mul_assoc, baseOps] using scaled

@[simp] theorem singleton_eval {columns : Nat} (selected : Fin columns)
    (value : F) (assignment : Assignment F columns) :
    (singleton selected value).eval assignment =
      value * assignment selected := by
  have identity := matrixVectorAt_identityRow baseOps baseLaws
    (fun _ column => (singleton selected 1).coefficient column)
    assignment (NumericBooleanDomain.vertex 0 ⟨0, by decide⟩) selected
    (by
      intro column
      simp [baseOps, eq_comm])
  have unit : (singleton selected 1).eval assignment = assignment selected := by
    simpa [SparseForm.eval, matrixVectorAt] using identity
  have formEqual :
      singleton selected value = scale value (singleton selected 1) := by
    simp [singleton, scale]
  rw [formEqual, scale_eval, unit]

end SparseForm

def meaningfulPort?
    (port : Fin Spec.ProductionRelation.matrixCount) :
    Option (Fin Spec.ProductionRelation.meaningfulPortCount) :=
  if bounded : port.val < Spec.ProductionRelation.meaningfulPortCount then
    some ⟨port.val, bounded⟩
  else
    none

structure Plan (logicalWidth : Nat) where
  rowCount : Nat
  rowCount_le : rowCount ≤ 2 ^ cubeVariables
  forms : Fin rowCount →
    Fin Spec.ProductionRelation.meaningfulPortCount → SparseForm logicalWidth

namespace Plan

def rowLayout {logicalWidth : Nat} (plan : Plan logicalWidth) :=
  CanonicalRowLayout.layout cubeVariables plan.rowCount plan.rowCount_le

def portForm {logicalWidth : Nat} (plan : Plan logicalWidth)
    (row : Fin plan.rowCount)
    (port : Fin Spec.ProductionRelation.matrixCount) : SparseForm logicalWidth :=
  match meaningfulPort? port with
  | some meaningful => plan.forms row meaningful
  | none => .empty

def matrix {logicalWidth : Nat} (plan : Plan logicalWidth)
    (port : Fin Spec.ProductionRelation.matrixCount) :
    BooleanMatrix F cubeVariables logicalWidth :=
  fun vertex column =>
    match plan.rowLayout.toColumn? vertex with
    | some row => (plan.portForm row port).coefficient column
    | none => 0

def rowImage {logicalWidth : Nat} (plan : Plan logicalWidth)
    (assignment : Assignment F logicalWidth)
    (vertex : BooleanVertex cubeVariables)
    (port : Fin Spec.ProductionRelation.matrixCount) : F :=
  match plan.rowLayout.toColumn? vertex with
  | some row => (plan.portForm row port).eval assignment
  | none => 0

theorem matrixVectorAt_matrix {logicalWidth : Nat} (plan : Plan logicalWidth)
    (assignment : Assignment F logicalWidth)
    (vertex : BooleanVertex cubeVariables)
    (port : Fin Spec.ProductionRelation.matrixCount) :
    matrixVectorAt baseOps (plan.matrix port) assignment vertex =
      plan.rowImage assignment vertex port := by
  cases decoded : plan.rowLayout.toColumn? vertex with
  | none =>
      simp only [rowImage, decoded]
      apply matrixVectorAt_zeroRow baseOps baseLaws
      intro column
      simp [matrix, decoded, baseOps]
  | some row =>
      simp only [rowImage, decoded]
      unfold matrixVectorAt SparseForm.eval
      simp [matrix, decoded]

def logicalRelation {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤
      2 ^ cubeVariables) : ProductionKey.LogicalRelation logicalWidth publicFits where
  matrices := plan.matrix
  cubeFits := cubeFits

@[simp] theorem logicalRelation_matrices {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤
      2 ^ cubeVariables) :
    (plan.logicalRelation (publicFits := publicFits) cubeFits).matrices =
      plan.matrix := by
  rfl

@[simp] theorem logicalRelation_system_matrices {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (plan : Plan logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤
      2 ^ cubeVariables) :
    (plan.logicalRelation (publicFits := publicFits) cubeFits).system.matrices =
      plan.matrix := by
  rfl

end Plan

end NightstreamFPrime.Layout.ProductionRelation
