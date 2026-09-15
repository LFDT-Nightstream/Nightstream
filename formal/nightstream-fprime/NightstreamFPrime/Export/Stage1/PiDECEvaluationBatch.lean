import NightstreamFPrime.Export.Stage1.PiDECEvaluationWeights

/-!
Weighted row batches in the existing stored extension-ring vectors. Each
step computes its point weight and complete child-row batch once. Child
projections equal the scalar accumulator; range sums can be combined in Lean.
No expected values, work premises, IO, or new row representation is used.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

/-- One stored zero for every child. -/
def zero (k : Nat) : Vector MaterializedRingK k :=
  Vector.replicate k (MaterializedRingK.ofRing ringKZero)

theorem zero_value {k : Nat} (child : Fin k) :
    ((zero k).get child).toRing = ringKZero := by
  change ((Vector.replicate k (MaterializedRingK.ofRing ringKZero))[child.val]).toRing = _
  rw [Vector.getElem_replicate, MaterializedRingK.toRing_ofRing]

/-- Combine complete extension-field partial sums, without Rust arithmetic. -/
def add {k : Nat} (left right : Vector MaterializedRingK k) :
    Vector MaterializedRingK k :=
  Vector.ofFn fun child => MaterializedRingK.ofRing
    (ringKAdd (left.get child).toRing (right.get child).toRing)

theorem add_value {k : Nat} (left right : Vector MaterializedRingK k)
    (child : Fin k) :
    ((add left right).get child).toRing =
      ringKAdd (left.get child).toRing (right.get child).toRing := by
  change ((Vector.ofFn (fun selected : Fin k => MaterializedRingK.ofRing
    (ringKAdd (left.get selected).toRing (right.get selected).toRing)))[child.val]).toRing = _
  rw [Vector.getElem_ofFn, MaterializedRingK.toRing_ofRing]

/-- The row producer and point weight each occur once outside the child loop. -/
def step {arity k : Nat} (point : CubePoint K arity)
    (rows : Nat → Vector StoredRing k) (index : Nat)
    (initial : Vector MaterializedRingK k) : Vector MaterializedRingK k :=
  let rowWeight := PiDECEvaluationWeights.weight point index
  let values := rows index
  Vector.ofFn fun child =>
    PiDECEvaluationWeights.addWeighted rowWeight (initial.get child) (values.get child)

theorem step_child {arity k : Nat} (point : CubePoint K arity)
    (rows : Nat → Vector StoredRing k) (index : Nat)
    (initial : Vector MaterializedRingK k) (child : Fin k) :
    (step point rows index initial).get child =
      PiDECEvaluationWeights.addWeighted (PiDECEvaluationWeights.weight point index)
        (initial.get child) ((rows index).get child) := by
  change (Vector.ofFn (fun selected : Fin k => PiDECEvaluationWeights.addWeighted
    (PiDECEvaluationWeights.weight point index)
    (initial.get selected) ((rows index).get selected)))[child.val] = _
  rw [Vector.getElem_ofFn]

/-- Continue from a stored batch over [start, start + count). Weights always
use the global row index, including when the range starts after zero. -/
def foldFrom {arity k : Nat} (start count : Nat) (point : CubePoint K arity)
    (rows : Nat → Vector StoredRing k) (initial : Vector MaterializedRingK k) :
    Vector MaterializedRingK k :=
  Nat.fold count (fun index _ accumulated =>
    step point rows (start + index) accumulated) initial

private theorem foldFrom_succ {arity k : Nat} (start count : Nat)
    (point : CubePoint K arity) (rows : Nat → Vector StoredRing k)
    (initial : Vector MaterializedRingK k) :
    foldFrom start (count + 1) point rows initial =
      step point rows (start + count) (foldFrom start count point rows initial) := by
  simp only [foldFrom, Nat.fold_succ]

/-- Exact scalar projection of a continued weighted range. -/
theorem foldFrom_value {arity k : Nat} (start count : Nat)
    (point : CubePoint K arity) (rows : Nat → Vector StoredRing k)
    (initial : Vector MaterializedRingK k) (child : Fin k) (lane : Fin ringDegree) :
    ((foldFrom start count point rows initial).get child).toRing lane =
      extensionOps.add ((initial.get child).toRing lane)
        (NumericCompletionSum.numericSum extensionOps count (fun index =>
          extensionOps.mul (PiDECEvaluationWeights.weight point (start + index))
            (K.embed (((rows (start + index)).get child).get lane)))) := by
  induction count with
  | zero =>
      change (initial.get child).toRing lane =
        extensionOps.add ((initial.get child).toRing lane) extensionOps.zero
      exact (extensionLaws.add_zero _).symm
  | succ count inductionHypothesis =>
      rw [foldFrom_succ, step_child, PiDECEvaluationWeights.addWeighted_value]
      change extensionOps.add
          (((foldFrom start count point rows initial).get child).toRing lane)
          (extensionOps.mul (PiDECEvaluationWeights.weight point (start + count))
            (K.embed (((rows (start + count)).get child).get lane))) =
        extensionOps.add ((initial.get child).toRing lane)
          (extensionOps.add
            (NumericCompletionSum.numericSum extensionOps count (fun index =>
              extensionOps.mul (PiDECEvaluationWeights.weight point (start + index))
                (K.embed (((rows (start + index)).get child).get lane))))
            (extensionOps.mul (PiDECEvaluationWeights.weight point (start + count))
              (K.embed (((rows (start + count)).get child).get lane))))
      rw [inductionHypothesis]
      exact extensionLaws.add_assoc _ _ _

/-- Independently compute one range from zero. -/
def range {arity k : Nat} (start count : Nat) (point : CubePoint K arity)
    (rows : Nat → Vector StoredRing k) : Vector MaterializedRingK k :=
  foldFrom start count point rows (zero k)

theorem range_value {arity k : Nat} (start count : Nat)
    (point : CubePoint K arity) (rows : Nat → Vector StoredRing k)
    (child : Fin k) (lane : Fin ringDegree) :
    ((range start count point rows).get child).toRing lane =
      NumericCompletionSum.numericSum extensionOps count (fun index =>
        extensionOps.mul (PiDECEvaluationWeights.weight point (start + index))
          (K.embed (((rows (start + index)).get child).get lane))) := by
  rw [range, foldFrom_value, zero_value]
  exact extensionLaws.zero_add _

/-- The full prefix interface, with no range or zero-suffix assumption. -/
def accumulate {arity k : Nat} (count : Nat) (point : CubePoint K arity)
    (rows : Nat → Vector StoredRing k) : Vector MaterializedRingK k :=
  range 0 count point rows

/-- Every child has exactly the existing scalar accumulator's complete
54-lane extension-ring value. This includes both field coordinates. -/
theorem accumulate_child {arity k : Nat} (count : Nat)
    (point : CubePoint K arity) (rows : Nat → Vector StoredRing k) (child : Fin k) :
    ((accumulate count point rows).get child).toRing =
      (PiDECEvaluationWeights.accumulate count point
        (fun index => (rows index).get child)).toRing := by
  funext lane
  rw [accumulate, range_value, PiDECEvaluationWeights.accumulate_value]
  simp only [Nat.zero_add]

/-- Splitting a numeric range is exactly continuation of the same fold. -/
theorem foldFrom_append {arity k : Nat} (start leftCount rightCount : Nat)
    (point : CubePoint K arity) (rows : Nat → Vector StoredRing k)
    (initial : Vector MaterializedRingK k) :
    foldFrom start (leftCount + rightCount) point rows initial =
      foldFrom (start + leftCount) rightCount point rows
        (foldFrom start leftCount point rows initial) := by
  unfold foldFrom
  rw [Nat.fold_add]
  simp only [Nat.add_assoc]

/-- Continuing from an initial batch adds the independently computed range. -/
theorem foldFrom_eq_add_range {arity k : Nat} (start count : Nat)
    (point : CubePoint K arity) (rows : Nat → Vector StoredRing k)
    (initial : Vector MaterializedRingK k) (child : Fin k) :
    ((foldFrom start count point rows initial).get child).toRing =
      ringKAdd (initial.get child).toRing ((range start count point rows).get child).toRing := by
  funext lane
  change ((foldFrom start count point rows initial).get child).toRing lane =
    extensionOps.add ((initial.get child).toRing lane)
      (((range start count point rows).get child).toRing lane)
  rw [foldFrom_value, range_value]

/-- Adding independent adjacent ranges in Lean preserves the complete value. -/
theorem range_append {arity k : Nat} (start leftCount rightCount : Nat)
    (point : CubePoint K arity) (rows : Nat → Vector StoredRing k) (child : Fin k) :
    ((range start (leftCount + rightCount) point rows).get child).toRing =
      ((add (range start leftCount point rows)
        (range (start + leftCount) rightCount point rows)).get child).toRing := by
  rw [range, foldFrom_append, add_value]
  exact foldFrom_eq_add_range (start + leftCount) rightCount point rows
    (range start leftCount point rows) child

/-- Final partial-batch addition remains in Lean. The caller supplies the
validated finite range count and each computed partial batch. -/
def sum {k : Nat} (count : Nat) (parts : Nat → Vector MaterializedRingK k) :
    Vector MaterializedRingK k :=
  Nat.fold count (fun index _ initial => add initial (parts index)) (zero k)

private theorem sum_succ {k : Nat} (count : Nat)
    (parts : Nat → Vector MaterializedRingK k) :
    sum (count + 1) parts = add (sum count parts) (parts count) := by
  simp only [sum, Nat.fold_succ]

/-- Every final child/lane is the exact numeric sum of its partial values. -/
theorem sum_value {k : Nat} (count : Nat)
    (parts : Nat → Vector MaterializedRingK k) (child : Fin k) (lane : Fin ringDegree) :
    ((sum count parts).get child).toRing lane =
      NumericCompletionSum.numericSum extensionOps count
        (fun index => ((parts index).get child).toRing lane) := by
  induction count with
  | zero =>
      change ((zero k).get child).toRing lane = extensionOps.zero
      rw [zero_value]
      rfl
  | succ count inductionHypothesis =>
      rw [sum_succ, add_value]
      change extensionOps.add (((sum count parts).get child).toRing lane)
          (((parts count).get child).toRing lane) =
        extensionOps.add
          (NumericCompletionSum.numericSum extensionOps count
            (fun index => ((parts index).get child).toRing lane))
          (((parts count).get child).toRing lane)
      rw [inductionHypothesis]

end NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch
