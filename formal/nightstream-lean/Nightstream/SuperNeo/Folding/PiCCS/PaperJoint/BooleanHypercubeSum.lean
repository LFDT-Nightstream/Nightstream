import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanDomain
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanEvaluation

/-!
Equality-weighted hypercube expansion of the canonical Boolean-table MLE.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: Boolean-table evaluation before alpha/gamma compression.
Constraint family: semantic equality-weight expansion; this file emits no
constraints.

Owns: the paper equality weight `eq(x, r)`, the explicit finite sum over the
sole canonical `BooleanVertex.all` enumeration, and its equality with the
independently recursive `BooleanTable.evaluate` MLE.

Does not own: construction of CCS, norm, or carried-evaluation tables; mapping
the semantic vertex order to an external paper integer, matrix-row, bitstring,
or Rust serialization; alpha/gamma mixing; SumCheck; Fiat--Shamir; R1CS; or
constraint counts.

Emits constraints: no.

Authority boundary: the outermost, newly introduced Boolean coordinate is the
head of both `BooleanVertex` and `CubePoint.coordinates`. `false` has factor
`1 - r_head`, `true` has factor `r_head`, and tails recurse in the same order.
The finite sum traverses `BooleanVertex.all`; no second cube enumeration or
caller-supplied evaluator is accepted.

| Protocol object | Phase | Family / leaf | Exact mathematical obligation |
|---|---|---|---|
| `Pi_CCS` Boolean cube | evaluation | `equalityWeight` | `eq(x,r) = product_i (x_i ? r_i : 1-r_i)` in prepended-coordinate order |
| `Pi_CCS` Boolean cube | evaluation | `equalityWeightedSum` | `sum_{x in BooleanVertex.all ell} eq(x,r) * table[x]` |
| `Pi_CCS` Boolean cube | tabulation lookup | `valueAt_tabulate` | a canonical table returns its defining value at every typed vertex |
| `Pi_CCS` Boolean cube | evaluation | `evaluate_eq_equalityWeightedSum` | recursive MLE equals the explicit hypercube sum at every typed point |
| `Pi_CCS` Boolean cube | evaluation | `toAlphaPolynomial_evaluate_eq_equalityWeightedSum` | canonical coefficient polynomial equals the same explicit hypercube sum |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField

namespace BooleanVertex

/-- Raw equality-weight recursion. Mismatch cases only make the function total;
`CubePoint` makes them unreachable in `equalityWeight`. The newly prepended
vertex coordinate is paired with the head point coordinate. -/
def equalityWeightCoordinates
    {Field : Type uField}
    (ops : InterpolationOps Field) :
    {variables : Nat} -> BooleanVertex variables -> List Field -> Field
  | 0, .nil, [] => ops.one
  | _ + 1, .cons false tail, coordinate :: coordinates =>
      ops.mul (ops.sub ops.one coordinate)
        (equalityWeightCoordinates ops tail coordinates)
  | _ + 1, .cons true tail, coordinate :: coordinates =>
      ops.mul coordinate (equalityWeightCoordinates ops tail coordinates)
  | _, _, _ => ops.zero

/-- The explicit paper equality weight at a dimension-checked cube point.
For each coordinate, bit `false` contributes `1-r_i` and bit `true`
contributes `r_i`. -/
def equalityWeight
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (vertex : BooleanVertex variables)
    (point : CubePoint Field variables) : Field :=
  equalityWeightCoordinates ops vertex point.coordinates

end BooleanVertex

namespace BooleanTable

/-- Value of an explicit table at a typed Boolean vertex. The indexed types
make every lookup branch exhaustive without a default value. -/
def valueAt
    {Field : Type uField} :
    {variables : Nat} -> BooleanTable Field variables ->
      BooleanVertex variables -> Field
  | 0, .leaf value, .nil => value
  | _ + 1, .branch low _, .cons false tail => valueAt low tail
  | _ + 1, .branch _ high, .cons true tail => valueAt high tail

/-- Canonical tabulation is pointwise exact at every typed Boolean vertex.
This is a leaf-level bridge only: it deliberately makes no claim that an
off-cube table MLE equals a nonlinear protocol polynomial. -/
@[simp] theorem valueAt_tabulate
    {Field : Type uField}
    {variables : Nat}
    (values : BooleanVertex variables -> Field)
    (vertex : BooleanVertex variables) :
    (tabulate values).valueAt vertex = values vertex := by
  induction vertex with
  | nil => rfl
  | cons coordinate tail inductionHypothesis =>
      cases coordinate <;>
        simp [tabulate, valueAt, inductionHypothesis]

/-- Explicit right-associated finite sum using the same addition and zero as
the interpolation model. -/
def finiteSum
    {Field : Type uField}
    (ops : InterpolationOps Field) : List Field -> Field
  | [] => ops.zero
  | value :: values => ops.add value (finiteSum ops values)

/-- The paper-shaped hypercube sum
`sum_x eq(x,r) * table[x]`, traversing exactly `BooleanVertex.all`. -/
def equalityWeightedSum
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (table : BooleanTable Field variables)
    (point : CubePoint Field variables) : Field :=
  finiteSum ops <|
    (BooleanVertex.all variables).map fun vertex =>
      ops.mul (vertex.equalityWeight ops point) (table.valueAt vertex)

private theorem finiteSum_append
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left right : List Field) :
    finiteSum ops (left ++ right) =
      ops.add (finiteSum ops left) (finiteSum ops right) := by
  induction left with
  | nil => simp [finiteSum, laws.zero_add]
  | cons value values inductionHypothesis =>
      simp only [List.cons_append, finiteSum, inductionHypothesis]
      exact (laws.add_assoc _ _ _).symm

private theorem finiteSum_map_mul_left
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (factor : Field)
    (values : List Field) :
    finiteSum ops (values.map (ops.mul factor)) =
      ops.mul factor (finiteSum ops values) := by
  induction values with
  | nil => simp [finiteSum, laws.mul_zero]
  | cons value values inductionHypothesis =>
      simp only [List.map_cons, finiteSum, inductionHypothesis]
      exact (laws.left_distrib factor value (finiteSum ops values)).symm

private theorem mul_neg
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left right : Field) :
    ops.mul left (ops.neg right) = ops.neg (ops.mul left right) := by
  rw [laws.mul_comm left (ops.neg right), laws.neg_mul,
    laws.mul_comm right left]

private theorem mul_sub
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left middle right : Field) :
    ops.mul left (ops.sub middle right) =
      ops.sub (ops.mul left middle) (ops.mul left right) := by
  unfold InterpolationOps.sub
  rw [laws.left_distrib, mul_neg ops laws]

private theorem one_sub_mul
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (coordinate value : Field) :
    ops.mul (ops.sub ops.one coordinate) value =
      ops.sub value (ops.mul coordinate value) := by
  unfold InterpolationOps.sub
  rw [laws.right_distrib, laws.one_mul, laws.neg_mul]

private theorem weightedBranches_eq_interpolation
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (coordinate low high : Field) :
    ops.add
        (ops.mul (ops.sub ops.one coordinate) low)
        (ops.mul coordinate high) =
      ops.add low (ops.mul coordinate (ops.sub high low)) := by
  rw [one_sub_mul ops laws, mul_sub ops laws]
  unfold InterpolationOps.sub
  calc
    ops.add
        (ops.add low (ops.neg (ops.mul coordinate low)))
        (ops.mul coordinate high) =
      ops.add low
        (ops.add (ops.neg (ops.mul coordinate low))
          (ops.mul coordinate high)) := laws.add_assoc _ _ _
    _ = ops.add low
        (ops.add (ops.mul coordinate high)
          (ops.neg (ops.mul coordinate low))) := by
      rw [laws.add_comm
        (ops.neg (ops.mul coordinate low))
        (ops.mul coordinate high)]

private theorem equalityWeightedSum_probe
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (table : BooleanTable Field variables)
    (point : CubePoint Field variables) :
    table.equalityWeightedSum ops point = table.evaluate ops point := by
  induction table with
  | leaf value =>
      rcases point with ⟨coordinates, dimension⟩
      have coordinatesEmpty : coordinates = [] :=
        List.eq_nil_of_length_eq_zero dimension
      subst coordinates
      simp [equalityWeightedSum, BooleanVertex.all,
        BooleanVertex.equalityWeight, BooleanVertex.equalityWeightCoordinates,
        valueAt, finiteSum, evaluate, evaluateCoordinates,
        laws.one_mul, laws.add_zero]
  | @branch tailVariables low high lowInduction highInduction =>
      rcases point with ⟨coordinates, dimension⟩
      cases coordinates with
      | nil => simp at dimension
      | cons coordinate coordinates =>
          have tailDimension : coordinates.length = tailVariables :=
            Nat.succ.inj dimension
          let tailPoint : CubePoint Field tailVariables :=
            ⟨coordinates, tailDimension⟩
          rw [equalityWeightedSum]
          simp only [BooleanVertex.all, List.map_append,
            finiteSum_append ops laws]
          simp only [List.map_map, BooleanVertex.equalityWeight]
          change ops.add
              (finiteSum ops
                ((BooleanVertex.all tailVariables).map fun vertex =>
                  ops.mul
                    (ops.mul (ops.sub ops.one coordinate)
                      (vertex.equalityWeight ops tailPoint))
                    (low.valueAt vertex)))
              (finiteSum ops
                ((BooleanVertex.all tailVariables).map fun vertex =>
                  ops.mul
                    (ops.mul coordinate
                      (vertex.equalityWeight ops tailPoint))
                    (high.valueAt vertex))) = _
          have lowFactored :
              finiteSum ops
                  ((BooleanVertex.all tailVariables).map fun vertex =>
                    ops.mul
                      (ops.mul (ops.sub ops.one coordinate)
                        (vertex.equalityWeight ops tailPoint))
                      (low.valueAt vertex)) =
                ops.mul (ops.sub ops.one coordinate)
                  (low.equalityWeightedSum ops tailPoint) := by
            rw [equalityWeightedSum]
            rw [← finiteSum_map_mul_left ops laws]
            simp only [List.map_map]
            congr 1
            apply List.map_congr_left
            intro vertex member
            exact laws.mul_assoc _ _ _
          have highFactored :
              finiteSum ops
                  ((BooleanVertex.all tailVariables).map fun vertex =>
                    ops.mul
                      (ops.mul coordinate
                        (vertex.equalityWeight ops tailPoint))
                      (high.valueAt vertex)) =
                ops.mul coordinate
                  (high.equalityWeightedSum ops tailPoint) := by
            rw [equalityWeightedSum]
            rw [← finiteSum_map_mul_left ops laws]
            simp only [List.map_map]
            congr 1
            apply List.map_congr_left
            intro vertex member
            exact laws.mul_assoc _ _ _
          rw [lowFactored, highFactored,
            lowInduction tailPoint, highInduction tailPoint]
          exact weightedBranches_eq_interpolation ops laws _ _ _

/-- At every dimension-checked point, the independently recursive canonical
Boolean-table MLE equals the explicit paper hypercube sum of equality weights.
This is a model-level algebraic theorem. It makes no paper-bit-order, Rust,
R1CS, or constraint claim. -/
theorem evaluate_eq_equalityWeightedSum
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (table : BooleanTable Field variables)
    (point : CubePoint Field variables) :
    table.evaluate ops point = table.equalityWeightedSum ops point := by
  exact (equalityWeightedSum_probe ops laws table point).symm

/-- The verifier-owned canonical alpha polynomial therefore evaluates to the
same explicit equality-weighted hypercube sum. This composes two independently
proved model-level identities; it does not add an external-order refinement. -/
theorem toAlphaPolynomial_evaluate_eq_equalityWeightedSum
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {shape : Shape}
    (table : BooleanTable Field shape.cubeVariables)
    (point : CubePoint Field shape.cubeVariables) :
    (table.toAlphaPolynomial ops).evaluate ops.toOps point =
      table.equalityWeightedSum ops point := by
  rw [toAlphaPolynomial_evaluate_eq_evaluate ops laws table point,
    evaluate_eq_equalityWeightedSum ops laws table point]

end BooleanTable

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
