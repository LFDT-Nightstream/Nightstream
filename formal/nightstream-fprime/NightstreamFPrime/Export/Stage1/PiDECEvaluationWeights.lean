import NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericCompletionSum
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanReproduction

/-!
Store a common-point weighted sum of computed base-ring rows. Each row's
point weight is shared by all 54 lane updates. The loop creates no Boolean
table or index list. No expected evaluation, matrix, work model or IO is used.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationWeights

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

/-- Use the existing little-endian tensor-weight recursion. The caller's
point fixes every factor; no table of weights is an input. -/
def weight {arity : Nat} (point : CubePoint K arity) (index : Nat) : K :=
  NumericBooleanDomain.tensorWeightCoordinates extensionOps point.coordinates index

/-- At every actual Boolean row, the numeric weight is the paper equality
weight at the same point and in the existing coordinate order. -/
theorem weight_at_vertex {arity : Nat} (point : CubePoint K arity)
    (vertex : BooleanVertex arity) :
    weight point (NumericBooleanDomain.index vertex) =
      vertex.equalityWeight extensionOps point := by
  simpa only [weight, NumericBooleanDomain.tensorWeight,
    NumericBooleanDomain.vertex_index] using
      NumericBooleanDomain.tensorWeight_eq_equalityWeight extensionOps
        ⟨NumericBooleanDomain.index vertex, NumericBooleanDomain.index_lt_twoPow vertex⟩
        point

/-- Update every stored extension coefficient with the same row weight.
The embedded row has zero imaginary part, so each lane needs two base-field
products. Materialization occurs before the next row reads the accumulator. -/
def addWeighted (rowWeight : K) (initial : MaterializedRingK)
    (row : StoredRing) : MaterializedRingK :=
  MaterializedRingK.ofRing fun lane =>
    let current := initial.toRing lane
    let scalar := row.get lane
    ⟨current.c0 + rowWeight.c0 * scalar,
     current.c1 + rowWeight.c1 * scalar⟩

theorem addWeighted_value (rowWeight : K) (initial : MaterializedRingK)
    (row : StoredRing) :
    (addWeighted rowWeight initial row).toRing =
      fun lane => extensionOps.add (initial.toRing lane)
        (extensionOps.mul rowWeight (K.embed (row.get lane))) := by
  rw [addWeighted, MaterializedRingK.toRing_ofRing]
  funext lane
  change _ = K.add (initial.toRing lane)
    (K.mul rowWeight (K.embed (row.get lane)))
  simp only [K.add, K.mul, K.embed,
    Fin.mul_zero, Fin.add_zero, Fin.zero_add]

/-- Visit the numeric prefix with one stored accumulator. The complete-domain
consumer below fixes count to exactly 2^arity. -/
def accumulate {arity : Nat} (count : Nat) (point : CubePoint K arity)
    (rows : Nat → StoredRing) : MaterializedRingK :=
  Nat.fold count (fun index _ initial =>
    let rowWeight := weight point index
    addWeighted rowWeight initial (rows index))
    (MaterializedRingK.ofRing ringKZero)

private theorem accumulate_succ {arity : Nat} (count : Nat)
    (point : CubePoint K arity) (rows : Nat → StoredRing) :
    accumulate (count + 1) point rows =
      addWeighted (weight point count) (accumulate count point rows) (rows count) := by
  simp only [accumulate, Nat.fold_succ]

/-- Each stored lane is the existing numeric sum of the embedded row values.
This holds for every prefix, without a zero-suffix or row-validity premise. -/
theorem accumulate_value {arity : Nat} (count : Nat)
    (point : CubePoint K arity) (rows : Nat → StoredRing)
    (lane : Fin ringDegree) :
    (accumulate count point rows).toRing lane =
      NumericCompletionSum.numericSum extensionOps count (fun index =>
        extensionOps.mul (weight point index) (K.embed ((rows index).get lane))) := by
  induction count with
  | zero =>
      change (MaterializedRingK.ofRing ringKZero).toRing lane = extensionOps.zero
      rw [MaterializedRingK.toRing_ofRing]
      rfl
  | succ count inductionHypothesis =>
      rw [accumulate_succ, addWeighted_value]
      change extensionOps.add ((accumulate count point rows).toRing lane)
          (extensionOps.mul (weight point count) (K.embed ((rows count).get lane))) =
        extensionOps.add
          (NumericCompletionSum.numericSum extensionOps count (fun index =>
            extensionOps.mul (weight point index) (K.embed ((rows index).get lane))))
          (extensionOps.mul (weight point count) (K.embed ((rows count).get lane)))
      rw [inductionHypothesis]

/-- The complete numeric accumulation is exactly Boolean-table evaluation at
the supplied point, for every coefficient of the supplied computed rows.
There is no expected result or independent correctness premise. -/
theorem accumulate_eq_evaluate {arity : Nat}
    (point : CubePoint K arity) (rows : Nat → StoredRing)
    (lane : Fin ringDegree) :
    (accumulate (2 ^ arity) point rows).toRing lane =
      (BooleanTable.tabulate (fun vertex : BooleanVertex arity =>
        K.embed ((rows (NumericBooleanDomain.index vertex)).get lane))).evaluate
        extensionOps point := by
  rw [accumulate_value,
    NumericCompletionSum.numericSum_eq_vertexSum extensionOps extensionLaws arity]
  calc
    _ = BooleanReproduction.equalityWeighted extensionOps point
        (fun vertex : BooleanVertex arity =>
          K.embed ((rows (NumericBooleanDomain.index vertex)).get lane)) := by
      apply FiniteSumAlgebra.sumMap_congr extensionOps (BooleanVertex.all arity)
      intro vertex _
      rw [weight_at_vertex]
    _ = _ := BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
      extensionOps extensionLaws point _

/-- A prefix accumulation equals the complete Boolean-table evaluation when
all omitted in-domain rows are zero. The selected row owner must prove both
the domain bound and zero suffix; nothing is assumed beyond the domain. -/
theorem accumulate_prefix_eq_evaluate {arity : Nat} (count : Nat)
    (point : CubePoint K arity) (rows : Nat → StoredRing)
    (fits : count ≤ 2 ^ arity)
    (outsideZero : ∀ index, count ≤ index → index < 2 ^ arity →
      (rows index).get = ringFZero)
    (lane : Fin ringDegree) :
    (accumulate count point rows).toRing lane =
      (BooleanTable.tabulate (fun vertex : BooleanVertex arity =>
        K.embed ((rows (NumericBooleanDomain.index vertex)).get lane))).evaluate
        extensionOps point := by
  rw [← accumulate_eq_evaluate point rows lane]
  simp only [accumulate_value]
  have omitted : ∀ index, count ≤ index → index < 2 ^ arity →
      extensionOps.mul (weight point index) (K.embed ((rows index).get lane)) =
        extensionOps.zero := by
    intro index lower upper
    rw [outsideZero index lower upper]
    change extensionOps.mul (weight point index) (K.embed baseOps.zero) =
      extensionOps.zero
    rw [embed_zero, extensionLaws.mul_zero]
  exact (NumericCompletionSum.numericSum_prefix_eq_vertexSum
      extensionOps extensionLaws arity count
      (fun index => extensionOps.mul (weight point index) (K.embed ((rows index).get lane)))
      fits omitted).trans
    (NumericCompletionSum.numericSum_eq_vertexSum
      extensionOps extensionLaws arity
      (fun index => extensionOps.mul (weight point index) (K.embed ((rows index).get lane)))).symm

end NightstreamFPrime.Export.Stage1.PiDECEvaluationWeights
