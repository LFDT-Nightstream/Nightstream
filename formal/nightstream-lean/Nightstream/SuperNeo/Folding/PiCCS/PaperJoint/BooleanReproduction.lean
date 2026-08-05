import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanHypercubeSum
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain

/-!
Boolean-selector reproduction for the canonical `Pi_CCS` hypercube.

Protocol: shared `Pi_CCS` polynomial infrastructure.
Phase: Boolean restriction and multilinear reproduction.
Constraint family: none; this file emits no rows.

Owns: the canonical Bool-to-field point embedding, partition of unity for the
canonical equality weights, reproduction of an explicit Boolean table, and
reproduction of the shared little-endian numeric tensor selector.

Does not own: construction of protocol tables, arbitrary-point equality
polynomials, off-cube polynomial degree, SumCheck, Fiat--Shamir, Rust, R1CS,
or constraint counts.

Emits constraints: no.

Authority boundary: every sum traverses exactly `BooleanVertex.all`; every
table is derived by `BooleanTable.tabulate`; and all rearrangement uses the
shared `FiniteSumAlgebra.sumMap`. No caller-supplied enumeration or evaluator
is accepted.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.boolean.vertex.field_point` | `false -> 0`, `true -> 1`, preserving prepended-coordinate order | computed | `BooleanVertex.toCubePoint` |
| `pi_ccs.boolean.selector.partition` | `sum_x eq(x,r) = 1` | derived | `equalityWeight_sum_eq_one` |
| `pi_ccs.boolean.selector.reproduce` | `sum_x eq(x,r) * f(x) = MLE(f)(r)` | derived | `equalityWeighted_tabulate_eq_evaluate` |
| `pi_ccs.boolean.selector.linear` | equality weighting commutes with an explicit weighted family sum | derived | `equalityWeighted_sumMap` |
| `pi_ccs.boolean.selector.tensor` | `sum_x eq(x,r) * chi_i(x) = chi_i(r)` | derived | `equalityWeighted_tensorWeight_eq_tensorWeight` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField uIndex

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

namespace BooleanVertex

/-- Canonical field-coordinate serialization of a typed Boolean vertex. The
head Boolean coordinate remains the head field coordinate. -/
@[simp] def fieldCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field) :
    {variables : Nat} -> BooleanVertex variables -> List Field
  | 0, .nil => []
  | _ + 1, .cons false tail => ops.zero :: fieldCoordinates ops tail
  | _ + 1, .cons true tail => ops.one :: fieldCoordinates ops tail

theorem fieldCoordinates_length
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (vertex : BooleanVertex variables) :
    (fieldCoordinates ops vertex).length = variables := by
  induction vertex with
  | nil => rfl
  | cons coordinate tail inductionHypothesis =>
      cases coordinate <;> simp [fieldCoordinates, inductionHypothesis]

/-- Canonical Boolean vertex embedded as a dimension-checked field point. -/
def toCubePoint
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (vertex : BooleanVertex variables) : CubePoint Field variables where
  coordinates := fieldCoordinates ops vertex
  dimension := fieldCoordinates_length ops vertex

@[simp] theorem toCubePoint_coordinates
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (vertex : BooleanVertex variables) :
    (toCubePoint ops vertex).coordinates = fieldCoordinates ops vertex := by
  rfl

end BooleanVertex

namespace BooleanReproduction

/-- Equality-weighted sum of an explicit value at every canonical Boolean
vertex. This is notation over the shared finite-sum owner, not a second sum
implementation. -/
def equalityWeighted
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (point : CubePoint Field variables)
    (values : BooleanVertex variables -> Field) : Field :=
  FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) fun vertex =>
    ops.mul (vertex.equalityWeight ops point) (values vertex)

private theorem evaluateCoordinates_tabulate_constant
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    {variables : Nat} ->
      (value : Field) ->
      (coordinates : List Field) ->
      coordinates.length = variables ->
      BooleanTable.evaluateCoordinates ops
          (BooleanTable.tabulate
            (fun _ : BooleanVertex variables => value)) coordinates = value
  | 0, value, coordinates, dimension => by
      have coordinatesEmpty : coordinates = [] :=
        List.eq_nil_of_length_eq_zero dimension
      subst coordinates
      rfl
  | variables + 1, value, coordinates, dimension => by
      cases coordinates with
      | nil => simp at dimension
      | cons coordinate coordinates =>
          have tailDimension : coordinates.length = variables :=
            Nat.succ.inj dimension
          change ops.add
              (BooleanTable.evaluateCoordinates ops
                (BooleanTable.tabulate
                  (fun _ : BooleanVertex variables => value)) coordinates)
              (ops.mul coordinate
                (ops.sub
                  (BooleanTable.evaluateCoordinates ops
                    (BooleanTable.tabulate
                      (fun _ : BooleanVertex variables => value)) coordinates)
                  (BooleanTable.evaluateCoordinates ops
                    (BooleanTable.tabulate
                      (fun _ : BooleanVertex variables => value)) coordinates))) =
            value
          rw [evaluateCoordinates_tabulate_constant ops laws value coordinates
            tailDimension]
          unfold InterpolationOps.sub
          rw [laws.add_neg, laws.mul_zero, laws.add_zero]

/-- A canonically tabulated constant evaluates to that constant. This helper
is stated publicly because partition of unity is its `value = 1` consequence
and later Boolean-domain proofs may need the same fact without redoing the
recursion. -/
theorem evaluate_tabulate_constant
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (value : Field)
    (point : CubePoint Field variables) :
    BooleanTable.evaluate ops
        (BooleanTable.tabulate
          (fun _ : BooleanVertex variables => value)) point = value := by
  exact evaluateCoordinates_tabulate_constant ops laws value
    point.coordinates point.dimension

/-- Equality-weighted values at every canonical Boolean vertex reproduce the
independently recursive MLE of their canonical table. -/
theorem equalityWeighted_tabulate_eq_evaluate
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (point : CubePoint Field variables)
    (values : BooleanVertex variables -> Field) :
    equalityWeighted ops point values =
      (BooleanTable.tabulate values).evaluate ops point := by
  simpa only [equalityWeighted, FiniteSumAlgebra.sumMap,
    BooleanTable.equalityWeightedSum, BooleanTable.valueAt_tabulate] using
    (BooleanTable.evaluate_eq_equalityWeightedSum ops laws
      (BooleanTable.tabulate values) point).symm

/-- Equality weighting commutes with a separately indexed finite weighted
family. This is the shared linearity step used by the joint residual
decomposition. -/
theorem equalityWeighted_sumMap
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (indices : List Index)
    (weights : Index -> Field)
    (values : Index -> BooleanVertex variables -> Field)
    (point : CubePoint Field variables) :
    equalityWeighted ops point (fun vertex =>
        FiniteSumAlgebra.sumMap ops indices fun index =>
          ops.mul (weights index) (values index vertex)) =
      FiniteSumAlgebra.sumMap ops indices fun index =>
        ops.mul (weights index)
          (equalityWeighted ops point (values index)) := by
  unfold equalityWeighted
  calc
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) (fun vertex =>
        ops.mul (vertex.equalityWeight ops point)
          (FiniteSumAlgebra.sumMap ops indices fun index =>
            ops.mul (weights index) (values index vertex))) =
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) (fun vertex =>
        FiniteSumAlgebra.sumMap ops indices fun index =>
          ops.mul (weights index)
            (ops.mul (vertex.equalityWeight ops point)
              (values index vertex))) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro vertex _
        rw [<- FiniteSumAlgebra.sumMap_mul_left ops laws
          (vertex.equalityWeight ops point) indices]
        apply FiniteSumAlgebra.sumMap_congr
        intro index _
        calc
          ops.mul (vertex.equalityWeight ops point)
              (ops.mul (weights index) (values index vertex)) =
            ops.mul
              (ops.mul (vertex.equalityWeight ops point) (weights index))
              (values index vertex) := (laws.mul_assoc _ _ _).symm
          _ = ops.mul
              (ops.mul (weights index) (vertex.equalityWeight ops point))
              (values index vertex) := by
                rw [laws.mul_comm
                  (vertex.equalityWeight ops point) (weights index)]
          _ = ops.mul (weights index)
              (ops.mul (vertex.equalityWeight ops point)
                (values index vertex)) := laws.mul_assoc _ _ _
    _ = FiniteSumAlgebra.sumMap ops indices (fun index =>
        FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) fun vertex =>
          ops.mul (weights index)
            (ops.mul (vertex.equalityWeight ops point)
              (values index vertex))) :=
      FiniteSumAlgebra.sumMap_swap ops laws
        (BooleanVertex.all variables) indices _
    _ = FiniteSumAlgebra.sumMap ops indices (fun index =>
        ops.mul (weights index)
          (FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) fun vertex =>
            ops.mul (vertex.equalityWeight ops point)
              (values index vertex))) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro index _
        exact FiniteSumAlgebra.sumMap_mul_left ops laws (weights index)
          (BooleanVertex.all variables) _

/-- The canonical Boolean equality weights form a partition of unity at every
dimension-checked point. -/
theorem equalityWeight_sum_eq_one
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (point : CubePoint Field variables) :
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables)
        (fun vertex => vertex.equalityWeight ops point) = ops.one := by
  calc
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables)
        (fun vertex => vertex.equalityWeight ops point) =
      equalityWeighted ops point (fun _ => ops.one) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro vertex _
        exact (laws.mul_one _).symm
    _ = (BooleanTable.tabulate
          (fun _ : BooleanVertex variables => ops.one)).evaluate ops point :=
      equalityWeighted_tabulate_eq_evaluate ops laws point _
    _ = ops.one := evaluate_tabulate_constant ops laws ops.one point

private theorem sub_one_zero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    ops.sub ops.one ops.zero = ops.one := by
  unfold InterpolationOps.sub
  rw [FiniteSumAlgebra.neg_zero ops laws, laws.add_zero]

private theorem sub_self
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (value : Field) :
    ops.sub value value = ops.zero := by
  exact laws.add_neg value

private theorem zero_mul
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (value : Field) :
    ops.mul ops.zero value = ops.zero := by
  rw [laws.mul_comm, laws.mul_zero]

/-- Equality weight at a canonically embedded Boolean point is the exact
Kronecker selector. This proof is structural in the typed cube and does not
pass through a numeric enumeration. -/
theorem equalityWeight_toCubePoint
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (selected sampled : BooleanVertex variables) :
    selected.equalityWeight ops (sampled.toCubePoint ops) =
      if selected = sampled then ops.one else ops.zero := by
  induction selected with
  | nil =>
      cases sampled
      rfl
  | @cons variables selectedBit selectedTail inductionHypothesis =>
      cases sampled with
      | cons sampledBit sampledTail =>
          have tailInduction :
              BooleanVertex.equalityWeightCoordinates ops selectedTail
                  (BooleanVertex.fieldCoordinates ops sampledTail) =
                if selectedTail = sampledTail then ops.one else ops.zero := by
            simpa only [BooleanVertex.equalityWeight,
              BooleanVertex.toCubePoint] using
              inductionHypothesis sampledTail
          cases selectedBit <;> cases sampledBit
          · simp [BooleanVertex.equalityWeight,
              BooleanVertex.equalityWeightCoordinates,
              BooleanVertex.toCubePoint, BooleanVertex.fieldCoordinates,
              sub_one_zero ops laws, laws.one_mul,
              tailInduction]
          · simp [BooleanVertex.equalityWeight,
              BooleanVertex.equalityWeightCoordinates,
              BooleanVertex.toCubePoint, BooleanVertex.fieldCoordinates,
              sub_self ops laws, zero_mul ops laws]
          · simp [BooleanVertex.equalityWeight,
              BooleanVertex.equalityWeightCoordinates,
              BooleanVertex.toCubePoint, BooleanVertex.fieldCoordinates,
              zero_mul ops laws]
          · simp [BooleanVertex.equalityWeight,
              BooleanVertex.equalityWeightCoordinates,
              BooleanVertex.toCubePoint, BooleanVertex.fieldCoordinates,
              laws.one_mul, tailInduction]

theorem sumMap_ite_eq_of_mem_nodup
    {Field : Type uField}
    {Index : Type}
    [DecidableEq Index]
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (indices : List Index)
    (selected : Index)
    (value : Index -> Field)
    (member : selected ∈ indices)
    (nodup : indices.Nodup) :
    FiniteSumAlgebra.sumMap ops indices
        (fun index => if index = selected then value index else ops.zero) =
      value selected := by
  induction indices with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      have parts := List.nodup_cons.mp nodup
      by_cases headSelected : head = selected
      · subst head
        rw [show FiniteSumAlgebra.sumMap ops (selected :: tail)
              (fun index =>
                if index = selected then value index else ops.zero) =
            ops.add (value selected)
              (FiniteSumAlgebra.sumMap ops tail
                (fun index =>
                  if index = selected then value index else ops.zero)) by
              simp only [FiniteSumAlgebra.sumMap, List.map_cons,
                BooleanTable.finiteSum, if_pos]]
        have tailZero :
            FiniteSumAlgebra.sumMap ops tail
                (fun index =>
                  if index = selected then value index else ops.zero) =
              FiniteSumAlgebra.sumMap ops tail (fun _ => ops.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro index indexMember
          have indexNe : index ≠ selected := by
            intro equal
            subst index
            exact parts.1 indexMember
          simp [indexNe]
        rw [tailZero, FiniteSumAlgebra.sumMap_zero ops laws tail,
          laws.add_zero]
      · have selectedInTail : selected ∈ tail := by
          rcases List.mem_cons.mp member with selectedHead | selectedTail
          · exact False.elim (headSelected selectedHead.symm)
          · exact selectedTail
        rw [show FiniteSumAlgebra.sumMap ops (head :: tail)
              (fun index =>
                if index = selected then value index else ops.zero) =
            ops.add ops.zero
              (FiniteSumAlgebra.sumMap ops tail
                (fun index =>
                  if index = selected then value index else ops.zero)) by
              simp only [FiniteSumAlgebra.sumMap, List.map_cons,
                BooleanTable.finiteSum, headSelected, if_false]]
        rw [inductionHypothesis selectedInTail parts.2, laws.zero_add]

/-- Equality weighting reproduces another equality selector whose selected
point is Boolean. This is the structural selector identity used by the
numeric tensor corollary. -/
theorem equalityWeighted_equalityWeight_eq_equalityWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (selected : BooleanVertex variables)
    (point : CubePoint Field variables) :
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) (fun sampled =>
        ops.mul (sampled.equalityWeight ops point)
          (selected.equalityWeight ops (sampled.toCubePoint ops))) =
      selected.equalityWeight ops point := by
  calc
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) (fun sampled =>
        ops.mul (sampled.equalityWeight ops point)
          (selected.equalityWeight ops (sampled.toCubePoint ops))) =
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) (fun sampled =>
        if sampled = selected then sampled.equalityWeight ops point
        else ops.zero) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro sampled _
          rw [equalityWeight_toCubePoint ops laws]
          by_cases equal : sampled = selected
          · subst sampled
            simp [laws.mul_one]
          · have reverse : selected ≠ sampled := by
              exact fun selectedSampled => equal selectedSampled.symm
            simp [equal, reverse, laws.mul_zero]
    _ = selected.equalityWeight ops point :=
      sumMap_ite_eq_of_mem_nodup ops laws
        (BooleanVertex.all variables) selected
        (fun sampled => sampled.equalityWeight ops point)
        (BooleanVertex.mem_all selected) (BooleanVertex.all_nodup variables)

/-- Equality weighting of the shared little-endian numeric tensor selector at
every canonical Boolean vertex reproduces that selector at the target point.
No equality between numeric enumeration order and `BooleanVertex.all` is used;
the bounded numeric index is decoded once to its typed selected vertex. -/
theorem equalityWeighted_tensorWeight_eq_tensorWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (value : Fin (2 ^ variables))
    (point : CubePoint Field variables) :
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) (fun sampled =>
        ops.mul (sampled.equalityWeight ops point)
          (NumericBooleanDomain.tensorWeight ops value
            (sampled.toCubePoint ops))) =
      NumericBooleanDomain.tensorWeight ops value point := by
  calc
    FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) (fun sampled =>
        ops.mul (sampled.equalityWeight ops point)
          (NumericBooleanDomain.tensorWeight ops value
            (sampled.toCubePoint ops))) =
      FiniteSumAlgebra.sumMap ops (BooleanVertex.all variables) (fun sampled =>
        ops.mul (sampled.equalityWeight ops point)
          ((NumericBooleanDomain.vertex variables value).equalityWeight ops
            (sampled.toCubePoint ops))) := by
          apply FiniteSumAlgebra.sumMap_congr
          intro sampled _
          rw [NumericBooleanDomain.tensorWeight_eq_equalityWeight]
    _ = (NumericBooleanDomain.vertex variables value).equalityWeight ops point :=
      equalityWeighted_equalityWeight_eq_equalityWeight ops laws
        (NumericBooleanDomain.vertex variables value) point
    _ = NumericBooleanDomain.tensorWeight ops value point :=
      (NumericBooleanDomain.tensorWeight_eq_equalityWeight
        ops value point).symm

end BooleanReproduction

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
