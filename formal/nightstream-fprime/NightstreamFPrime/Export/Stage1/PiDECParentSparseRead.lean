import NightstreamFPrime.Export.Stage1.PiDECParentScalarRead
import NightstreamFPrime.Layout.ProductionRelation.SparseEvaluation

/-!
Read child Phi81 coefficients through the nonzero entries of the existing
prepared basis action. Each basis/output pair owns one existing SparseForm.
Preparation preserves input order and removes only exact zero coefficients.
No parent value, expected evaluation, or matrix is an input to preparation.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECParentSparseRead

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment)
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

private theorem filterZero_evalSparse {columns : Nat} (form : SparseForm columns)
    (source : Fin columns → F) :
    ({ entries := form.entries.filter (fun entry => decide (entry.coefficient ≠ 0)) } :
      SparseForm columns).evalSparse source = form.evalSparse source := by
  unfold SparseForm.evalSparse
  rw [List.foldl_filter]
  apply congrArg (fun step : F → SparseEntry columns → F => form.entries.foldl step 0)
  funext initial entry
  by_cases zero : entry.coefficient = 0 <;> simp [zero]

/-- Retain each exact nonzero coefficient in its original input-lane order.
The complete coefficient table is inspected only during preparation. -/
def coefficientForm (table : FixedArray MaterializedRingF ringDegree)
    (output : Fin ringDegree) : SparseForm ringDegree where
  entries := ((List.finRange ringDegree).map fun input =>
    (⟨input, (table.get input).toRing output⟩ : SparseEntry ringDegree)).filter
      (fun entry => decide (entry.coefficient ≠ 0))

private theorem coefficientForm_eval (table : FixedArray MaterializedRingF ringDegree)
    (output : Fin ringDegree) (source : RingF) :
    (coefficientForm table output).evalSparse source =
      NumericCompletionSum.numericSum baseOps ringDegree (fun index =>
        if live : index < ringDegree then
          (table.get ⟨index, live⟩).toRing output * source ⟨index, live⟩
        else 0) := by
  let form : SparseForm ringDegree :=
    ⟨(List.finRange ringDegree).map fun input =>
      (⟨input, (table.get input).toRing output⟩ : SparseEntry ringDegree)⟩
  change ({ entries := form.entries.filter (fun entry => decide (entry.coefficient ≠ 0)) } :
    SparseForm ringDegree).evalSparse source = _
  rw [filterZero_evalSparse form source,
    NumericCompletionSum.numericSum, Nat.fold_eq_finRange_foldl]
  simp only [form, SparseForm.evalSparse, List.foldl_map]
  apply congrArg (fun step : F → Fin ringDegree → F =>
    (List.finRange ringDegree).foldl step 0)
  funext initial input
  rcases input with ⟨input, bounded⟩
  rw [dif_pos bounded]
  rfl

private theorem numericSum_eq_sumRange (count : Nat) (term : Nat → F) :
    NumericCompletionSum.numericSum baseOps count term =
      sumRange baseOps count term := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [NumericCompletionSum.numericSum, Nat.fold_succ]
      change baseOps.add (NumericCompletionSum.numericSum baseOps count term)
          (term count) =
        baseOps.add (sumRange baseOps count term) (term count)
      rw [inductionHypothesis]

/-- Sparse evaluation equals the same fixed ring action for every scalar
source. This includes cancellation and both signs, with no support premise. -/
theorem coefficientForm_value (challenge : MaterializedRingF)
    (source : RingF) (output : Fin ringDegree) :
    (coefficientForm (PiRLCWitnessAction.prepare challenge) output).evalSparse source =
      ringFMul challenge.toRing source output := by
  rw [coefficientForm_eval, numericSum_eq_sumRange,
    CarrierAction.ringFMul_apply_eq_rightLinear]
  apply sumRange_congr
  intro index live
  simp only [dif_pos live, PiRLCWitnessAction.prepare, FixedArray.get_ofFn,
    MaterializedRingF.toRing_ofRing, CarrierAction.rightCoefficient]

/-- On signed sources this is also the exact previously checked scalar fold. -/
theorem coefficientForm_eq_coefficient (challenge : MaterializedRingF)
    (source : RingF)
    (signed : ∀ input, source input = 0 ∨ source input = 1 ∨ source input = -1)
    (output : Fin ringDegree) :
    (coefficientForm (PiRLCWitnessAction.prepare challenge) output).evalSparse source =
      PiDECParentScalarRead.coefficient
        (PiRLCWitnessAction.prepare challenge) source output := by
  rw [coefficientForm_value, PiDECParentScalarRead.coefficient_value challenge source signed]

/-- Compute each basis/output sparse form once from the existing prepared
basis products. The nested containers are the existing FixedArray type. -/
def prepare (_unit : Unit := ()) :
    FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree :=
  let tables := PiDECParentScalarRead.prepare ()
  FixedArray.ofFn fun basis =>
    FixedArray.ofFn fun output => coefficientForm (tables.get basis) output

/-- Only retained nonzero entries request a scalar digit from the parent.
All 54 parent lanes, including carried tails, remain addressable. -/
def read (forms : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (parent : StoredAssignment ringDegree) (basis : Fin ringDegree)
    (child : Radix.ChildIndex) (output : Fin ringDegree) : F :=
  ((forms.get basis).get output).evalSparse
    (fun input => Radix.splitScalar (parent.get input) child)

/-- The sparse reader has exactly the existing total scalar-split meaning.
The caller's strict parent-bound check remains the PiDEC acceptance gate. -/
theorem read_eq_splitScalar (parent : StoredAssignment ringDegree)
    (basis : Fin ringDegree) (child : Radix.ChildIndex) (output : Fin ringDegree) :
    read (prepare ()) parent basis child output =
      CarrierAction.kernelImage basis
        (fun input => Radix.splitScalar (parent.get input) child) output := by
  rw [read, prepare, FixedArray.get_ofFn, FixedArray.get_ofFn,
    PiDECParentScalarRead.prepare, FixedArray.get_ofFn,
    coefficientForm_value, MaterializedRingF.toRing_ofRing]
  exact (congrFun (CarrierAction.kernelImage_eq_ringFMul basis _) output).symm

/-- The sparse optimization preserves every value of the previous scalar
reader on the same checked parent domain. -/
theorem read_eq_scalarRead (parent : StoredAssignment ringDegree)
    (bounded : ∀ input, centeredMagnitude (parent.get input) < Radix.combinedBound)
    (basis : Fin ringDegree) (child : Radix.ChildIndex) (output : Fin ringDegree) :
    read (prepare ()) parent basis child output =
      PiDECParentScalarRead.read (PiDECParentScalarRead.prepare ())
        parent basis child output := by
  rw [read_eq_splitScalar,
    PiDECParentScalarRead.read_eq_splitScalar parent bounded basis child output]

/-- The exact checked child coefficient follows from the original split;
children are theorem data and are not supplied to the executable reader. -/
theorem read_eq_checkedChild (parent : StoredAssignment ringDegree)
    (children : Vector (StoredAssignment ringDegree) productionGlobalParams.k)
    (success : StoredSplit.splitChecked parent = some children)
    (basis : Fin ringDegree) (child : Radix.ChildIndex) (output : Fin ringDegree) :
    read (prepare ()) parent basis child output =
      CarrierAction.kernelImage basis (children.get child).get output := by
  rw [read_eq_splitScalar]
  apply congrArg (fun source : RingF => CarrierAction.kernelImage basis source output)
  funext input
  exact (StoredSplit.splitChecked_value parent children success child input).symm

end NightstreamFPrime.Export.Stage1.PiDECParentSparseRead
