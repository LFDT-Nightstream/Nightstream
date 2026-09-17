import NightstreamFPrime.Export.Stage1.FreshCommitmentBlock
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment

/-! Complete the existing native block accumulator over an explicit finite
index order. Runtime partition coverage and file decoding are separate. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.FreshCommitmentFold

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic (StoredRing)
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment (ringFSum commit)

private theorem add_assoc (a b c : RingF) :
    ringFAdd (ringFAdd a b) c = ringFAdd a (ringFAdd b c) := by
  funext lane
  exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem add_zero (a : RingF) : ringFAdd a ringFZero = a := by
  funext lane
  exact ConcreteCarrier.baseLaws.add_zero _

private theorem zero_add (a : RingF) : ringFAdd ringFZero a = a := by
  funext lane
  exact ConcreteCarrier.baseLaws.zero_add _

/-- The existing native accumulator body, with the visited block indices
explicit. This is the pure arithmetic core of a range, not a file runner. -/
def fold {rows columns count : Nat} (setup : AjtaiSetupV1.Setup rows columns)
    (row : Fin rows) (indices : Fin count → Fin columns)
    (digits : Fin columns → StoredRing) (initial : PiDECNativeProduct.Accumulator) :
    PiDECNativeProduct.Accumulator :=
  Fin.foldl count (fun accumulated index =>
    FreshCommitmentBlock.accumulatePrepared setup row (indices index)
      (PiDECNativeProduct.prepareDigit (digits (indices index))) accumulated) initial

theorem fold_value {rows columns : Nat}
    (setup : AjtaiSetupV1.Setup rows columns) (row : Fin rows) :
    ∀ {count : Nat} (indices : Fin count → Fin columns)
      (digits : Fin columns → StoredRing) (initial : PiDECNativeProduct.Accumulator),
      (fold setup row indices digits initial).finish.get =
        ringFAdd initial.finish.get
          (ringFSum fun index => ringFMul (setup.verifierKey row (indices index))
            (digits (indices index)).get)
  | 0, _, _, initial => by
      simp only [fold, Fin.foldl_zero, ringFSum]
      exact (add_zero initial.finish.get).symm
  | count + 1, indices, digits, initial => by
      rw [fold, Fin.foldl_succ]
      have tail := fold_value setup row (fun index : Fin count => indices index.succ)
        digits (FreshCommitmentBlock.accumulatePrepared setup row (indices 0)
          (PiDECNativeProduct.prepareDigit (digits (indices 0))) initial)
      calc
        _ = ringFAdd
            (FreshCommitmentBlock.accumulatePrepared setup row (indices 0)
              (PiDECNativeProduct.prepareDigit (digits (indices 0))) initial).finish.get
            (ringFSum fun index : Fin count =>
              ringFMul (setup.verifierKey row (indices index.succ))
                (digits (indices index.succ)).get) := tail
        _ = _ := by
          rw [FreshCommitmentBlock.accumulatePrepared_value]
          exact add_assoc _ _ _

/-- Visit every complete carrier block. No logical-prefix truncation, unit
value, or nonzero premise enters the sum. -/
def completeRow {shape : Phi81Relation.Shape} {rows : Nat}
    (setup : AjtaiSetupV1.Setup rows (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (assignment : Phi81Relation.Assignment shape) (row : Fin rows) : StoredRing :=
  (fold setup row id (fun block => Vector.ofFn (CarrierAction.assignmentBlock assignment block))
    PiDECNativeProduct.Accumulator.zero).finish

theorem completeRow_value {shape : Phi81Relation.Shape} {rows : Nat}
    (setup : AjtaiSetupV1.Setup rows (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (assignment : Phi81Relation.Assignment shape) (row : Fin rows) :
    (completeRow setup assignment row).get = commit setup.verifierKey assignment row := by
  rw [completeRow, fold_value, PiDECNativeProduct.Accumulator.zero_value, zero_add]
  apply congrArg ringFSum
  funext block
  apply congrArg (ringFMul (setup.verifierKey row block))
  funext lane
  change (Vector.ofFn (CarrierAction.assignmentBlock assignment block))[lane.val] = _
  rw [Vector.getElem_ofFn]

end NightstreamFPrime.Export.Stage1.FreshCommitmentFold
