import NightstreamFPrime.Export.Stage1.PiDECMatrixRangeSum
import NightstreamFPrime.Export.Stage1.PiDECMatrixSelectedBatch

/-!
Two composition links for the complete matrix replay: an ordered fold of
adjacent computed ranges is the complete prefix, and scalar-split parent
blocks are the complete blocks of the same successful stored split.
No codec, IO, row producer, or new representation is introduced.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixMergeClosure

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment)
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open PiDECMatrixRangeSum (Values)

/-- The actual nested add fold preserves arbitrary adjacent canonical ranges.
Each part equality is supplied by the existing selected range theorem.
The merger's positive endpoints imply the weaker monotonicity used here.
At start zero, specializing cuts count to the selected row count gives the full accumulator. -/
theorem fold_adjacent_eq_range {arity : Nat} (start : Nat)
    (cuts : Nat → Nat) (parts : Nat → Values) (point : CubePoint K arity)
    (rows : Fin matrixCount → Nat → Vector StoredRing productionGlobalParams.k)
    (first : cuts 0 = 0) (count : Nat)
    (child : Fin productionGlobalParams.k) (port : Fin matrixCount) :
    (∀ index, index < count → cuts index ≤ cuts (index + 1)) →
    (∀ index, index < count →
      (((parts index).get child).get port).toRing =
        ((PiDECEvaluationBatch.range (start + cuts index) (cuts (index + 1) - cuts index)
          point (rows port)).get child).toRing) →
    (((Nat.fold count (fun index _ accumulated =>
        PiDECMatrixRangeSum.add accumulated (parts index)) PiDECMatrixRangeSum.zero).get
      child).get port).toRing =
      ((PiDECEvaluationBatch.range start (cuts count) point (rows port)).get child).toRing := by
  induction count with
  | zero =>
      intro _ _
      rw [first]
      change ((PiDECMatrixRangeSum.zero.get child).get port).toRing =
        ((PiDECEvaluationBatch.zero productionGlobalParams.k).get child).toRing
      rw [PiDECMatrixRangeSum.zero_value, PiDECEvaluationBatch.zero_value]
  | succ count inductionHypothesis =>
      intro ordered each
      have previous := inductionHypothesis
        (fun index live => ordered index (Nat.lt_trans live (Nat.lt_succ_self count)))
        (fun index live => each index (Nat.lt_trans live (Nat.lt_succ_self count)))
      have last := each count (Nat.lt_succ_self count)
      have completeCount : cuts count + (cuts (count + 1) - cuts count) = cuts (count + 1) :=
        Nat.add_sub_of_le (ordered count (Nat.lt_succ_self count))
      have joined := PiDECEvaluationBatch.range_append start (cuts count)
        (cuts (count + 1) - cuts count) point (rows port) child
      rw [PiDECEvaluationBatch.add_value, completeCount] at joined
      rw [Nat.fold_succ, PiDECMatrixRangeSum.add_value, previous, last]
      exact joined.symm

/-- Proportional worker cuts cover the full unit interval in order. The
worker count affects execution only; neither values nor rows choose a cut. -/
theorem proportionalCuts (units parts : Nat) (positive : 0 < parts) :
    units * 0 / parts = 0 ∧ units * parts / parts = units ∧
      ∀ index, index < parts →
        units * index / parts ≤ units * (index + 1) / parts ∧
          units * (index + 1) / parts ≤ units := by
  refine ⟨by simp, Nat.mul_div_cancel units positive, ?_⟩
  intro index bound
  constructor
  · exact Nat.div_le_div_right (Nat.mul_le_mul_left units (Nat.le_succ index))
  · calc
      units * (index + 1) / parts ≤ units * parts / parts :=
        Nat.div_le_div_right (Nat.mul_le_mul_left units (by omega))
      _ = units := Nat.mul_div_cancel units positive

private theorem get_ofFn {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = _
  rw [Vector.getElem_ofFn]

/-- The reference blocks used by the numeric range proofs are exactly those
used by familyFromBlocks_honestMessages for the same successful full split.
Parent and child vectors are theorem data; no full vector is built by a reader. -/
theorem splitBlocks_eq_childBlocks {shape : Phi81Relation.Shape}
    (parent : StoredAssignment shape.carrierWidth)
    (children : Vector (StoredAssignment shape.carrierWidth) productionGlobalParams.k)
    (success : StoredSplit.splitChecked parent = some children) :
    PiDECMatrixSelectedBatch.splitBlocks
        (fun block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
          Vector.ofFn (CarrierAction.assignmentBlock
            (logicalWidth := shape.logicalWidth) parent.get block)) =
      PiDECCommitmentFold.childBlocks (shape := shape) children := by
  funext block
  apply Vector.ext
  intro child childBound
  apply Vector.ext
  intro input inputBound
  change ((PiDECMatrixSelectedBatch.splitBlocks
    (fun selected : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) =>
      Vector.ofFn (CarrierAction.assignmentBlock
        (logicalWidth := shape.logicalWidth) parent.get selected)) block).get
      ⟨child, childBound⟩).get ⟨input, inputBound⟩ =
    ((PiDECCommitmentFold.childBlocks (shape := shape) children block).get
      ⟨child, childBound⟩).get ⟨input, inputBound⟩
  rw [PiDECMatrixSelectedBatch.splitBlocks, get_ofFn, get_ofFn, get_ofFn,
    PiDECCommitmentFold.childBlocks_value]
  exact (StoredSplit.splitChecked_value parent children success ⟨child, childBound⟩
    (CarrierAction.carrierColumn (logicalWidth := shape.logicalWidth)
      block ⟨input, inputBound⟩)).symm

end NightstreamFPrime.Export.Stage1.PiDECMatrixMergeClosure
