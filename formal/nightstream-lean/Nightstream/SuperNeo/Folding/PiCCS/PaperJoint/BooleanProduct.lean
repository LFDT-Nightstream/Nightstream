import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanReproduction
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain

/-!
Boolean-prefix products for structured `Pi_CCS` row domains.

Protocol: shared finite-domain infrastructure for paper `Pi_CCS` and concrete
identity-matrix openings.
Phase: a Boolean prefix followed by an arbitrary suffix point.
Constraint family: semantic domain composition only; this file emits no rows.

Owns: exact concatenation of typed cube points and Boolean vertices; the
little-endian numeric index of that concatenation; the canonical all-zero
Boolean prefix; and specialization of a tabulated MLE at a Boolean prefix
without requiring the suffix point to be Boolean.

Does not own: any particular lane/block dimensions, matrix entries, Phi81
coefficient algebra, transcript ordering, Rust serialization, R1CS lowering,
row removal, or constraint counts.

Emits constraints: no.

Authority boundary: coordinates are concatenated in list order. Because the
numeric convention is little-endian, a prefix with `m` variables contributes
the low bits and the suffix begins at stride `2^m`. The specialization theorem
uses only an actual Boolean prefix; it does not claim a general arbitrary-point
product/Fubini decomposition.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.domain.product.point` | product-point coordinates are exactly prefix then suffix | computed | `CubePoint.withLowPrefix` |
| `pi_ccs.domain.product.vertex` | Boolean product vertices use the same coordinate order | computed | `BooleanVertex.withLowPrefix` |
| `pi_ccs.domain.product.index` | `index(prefix ++ suffix) = index(prefix) + 2^m * index(suffix)` | derived | `NumericBooleanDomain.index_withLowPrefix` |
| `pi_ccs.domain.product.zero_prefix` | the canonical zero prefix has numeric index zero | computed / derived | `BooleanVertex.zeros`, `NumericBooleanDomain.index_zeros` |
| `pi_ccs.domain.product.specialize_prefix` | evaluating a tabulated table at a Boolean prefix restricts exactly to the suffix table | derived | `BooleanTable.evaluate_tabulate_booleanPrefix` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField

namespace CubePoint

/-- Put a point in the low-coordinate prefix of a product point. The result
dimension is written `suffixVariables + prefixVariables` so recursion on the
prefix is definitionally transparent; its coordinates remain prefix first. -/
def withLowPrefix
    {Field : Type uField}
    {prefixVariables suffixVariables : Nat}
    (left : CubePoint Field prefixVariables)
    (right : CubePoint Field suffixVariables) :
    CubePoint Field (suffixVariables + prefixVariables) where
  coordinates := left.coordinates ++ right.coordinates
  dimension := by
    rw [List.length_append, left.dimension, right.dimension, Nat.add_comm]

@[simp] theorem withLowPrefix_coordinates
    {Field : Type uField}
    {prefixVariables suffixVariables : Nat}
    (left : CubePoint Field prefixVariables)
    (right : CubePoint Field suffixVariables) :
    (withLowPrefix left right).coordinates =
      left.coordinates ++ right.coordinates := by
  rfl

end CubePoint

namespace BooleanVertex

/-- Put a Boolean vertex in the low-coordinate prefix of a product vertex.
The coordinate order agrees with `CubePoint.withLowPrefix`. -/
def withLowPrefix :
    {prefixVariables suffixVariables : Nat} ->
      BooleanVertex prefixVariables -> BooleanVertex suffixVariables ->
        BooleanVertex (suffixVariables + prefixVariables)
  | 0, _, .nil, suffix => suffix
  | _ + 1, _, .cons coordinate tail, suffix =>
      .cons coordinate (withLowPrefix tail suffix)

/-- Canonical all-false Boolean vertex. -/
def zeros : (variables : Nat) -> BooleanVertex variables
  | 0 => .nil
  | variables + 1 => .cons false (zeros variables)

@[simp] theorem fieldCoordinates_withLowPrefix
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {prefixVariables suffixVariables : Nat}
    (left : BooleanVertex prefixVariables)
    (suffix : BooleanVertex suffixVariables) :
    fieldCoordinates ops (withLowPrefix left suffix) =
      fieldCoordinates ops left ++ fieldCoordinates ops suffix := by
  induction left with
  | nil => rfl
  | cons coordinate tail inductionHypothesis =>
      cases coordinate <;>
        simp only [withLowPrefix, fieldCoordinates,
          inductionHypothesis, List.cons_append]

end BooleanVertex

namespace NumericBooleanDomain

/-- Little-endian product law: the prefix occupies the low bits and the
suffix begins at stride `2^prefixVariables`. -/
theorem index_withLowPrefix
    {prefixVariables suffixVariables : Nat}
    (left : BooleanVertex prefixVariables)
    (suffix : BooleanVertex suffixVariables) :
    index (left.withLowPrefix suffix) =
      index left + 2 ^ prefixVariables * index suffix := by
  induction left with
  | nil => simp [BooleanVertex.withLowPrefix, index]
  | @cons variables coordinate tail inductionHypothesis =>
      cases coordinate <;>
        simp only [BooleanVertex.withLowPrefix, index, inductionHypothesis,
          Nat.pow_succ, Nat.mul_add, Nat.mul_assoc, Nat.mul_comm] <;>
        omega

@[simp] theorem index_zeros (variables : Nat) :
    index (BooleanVertex.zeros variables) = 0 := by
  induction variables with
  | zero => rfl
  | succ variables inductionHypothesis =>
      change 0 + 2 * index (BooleanVertex.zeros variables) = 0
      rw [inductionHypothesis]

end NumericBooleanDomain

namespace BooleanTable

private theorem zero_mul
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (value : Field) :
    ops.mul ops.zero value = ops.zero := by
  rw [laws.mul_comm, laws.mul_zero]

private theorem add_sub_self_right
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (low high : Field) :
    ops.add low (ops.sub high low) = high := by
  unfold InterpolationOps.sub
  calc
    ops.add low (ops.add high (ops.neg low)) =
        ops.add (ops.add low high) (ops.neg low) :=
      (laws.add_assoc _ _ _).symm
    _ = ops.add (ops.add high low) (ops.neg low) := by
      rw [laws.add_comm low high]
    _ = ops.add high (ops.add low (ops.neg low)) :=
      laws.add_assoc _ _ _
    _ = ops.add high ops.zero := by rw [laws.add_neg]
    _ = high := laws.add_zero high

/-- Raw-coordinate form of Boolean-prefix specialization. The suffix list is
kept explicit so the induction does not need a caller-supplied dimension
witness. -/
private theorem evaluateCoordinates_tabulate_booleanPrefix
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {prefixVariables suffixVariables : Nat}
    (left : BooleanVertex prefixVariables)
    (values : BooleanVertex (suffixVariables + prefixVariables) -> Field)
    (suffixCoordinates : List Field) :
    (tabulate values).evaluateCoordinates ops
        (left.fieldCoordinates ops ++ suffixCoordinates) =
      (tabulate (fun suffix => values (left.withLowPrefix suffix))).evaluateCoordinates
        ops suffixCoordinates := by
  induction left with
  | nil => rfl
  | @cons variables coordinate tail inductionHypothesis =>
      cases coordinate
      · simp only [BooleanVertex.fieldCoordinates, List.cons_append,
          tabulate, evaluateCoordinates]
        rw [zero_mul ops laws, laws.add_zero]
        simpa only [BooleanVertex.withLowPrefix] using
          inductionHypothesis
            (fun rest => values (BooleanVertex.cons false rest))
      · simp only [BooleanVertex.fieldCoordinates, List.cons_append,
          tabulate, evaluateCoordinates]
        rw [laws.one_mul, add_sub_self_right ops laws]
        simpa only [BooleanVertex.withLowPrefix] using
          inductionHypothesis
            (fun rest => values (BooleanVertex.cons true rest))

/-- Evaluating a canonically tabulated product table at a Boolean prefix and
an arbitrary suffix point is exactly evaluation of the restricted suffix
table. -/
theorem evaluate_tabulate_booleanPrefix
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {prefixVariables suffixVariables : Nat}
    (left : BooleanVertex prefixVariables)
    (values : BooleanVertex (suffixVariables + prefixVariables) -> Field)
    (suffixPoint : CubePoint Field suffixVariables) :
    (tabulate values).evaluate ops
        ((left.toCubePoint ops).withLowPrefix suffixPoint) =
      (tabulate (fun suffix => values (left.withLowPrefix suffix))).evaluate
        ops suffixPoint := by
  exact evaluateCoordinates_tabulate_booleanPrefix ops laws left values
    suffixPoint.coordinates

end BooleanTable

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
