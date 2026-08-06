import Std
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetConvention

/-!
Finite carried-target polynomials for the paper-level `Pi_CCS` exponent audit.

Owns: typed carried coefficients, a fixed finite coordinate enumeration, local
and selected `2K+k`-shifted target evaluation, the exact shift identity, and a
support witness that separates the local and shifted conventions.

Does not own: `Q`, carried-evaluation residual formulas, the signed joint
identity, SumCheck, transcript semantics, Rust, R1CS, constraint removal, or
production approval.

Emits constraints: no.

Authority boundary: coefficient values are explicit typed data indexed by all
carried coordinates. Exponents and the finite traversal are derived from
`Shape`. The shifted convention is the corrected paper selection. This file
proves its finite algebraic relation to the local helper target and does not
identify either target with `Q`.

| Mathematical object | Definition | Proven property |
|---|---|---|
| carried coefficients | `CarriedTargetCoefficients` | one value per typed `(running, matrix, coefficient)` coordinate |
| local helper target | `evaluateLocal` | uses exponent `I(i,j,l)` |
| selected absolute target | `evaluateShifted` | uses exponent `2K+k+I(i,j,l)` |
| shift relation | `evaluateShifted_eq_shift_mul_evaluateLocal` | `T_abs(gamma) = gamma^(2K+k) * T_local(gamma)` |
| layout support | `ExponentLayoutSupport` | exponent `0` witnesses literal/shifted support mismatch for positive paper dimensions |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetPolynomial

open Nightstream.SuperNeo.SumCheck

universe uField

/-- Exactly the algebraic laws used by the finite target-shift proof. They are
a strict fragment of the field laws available in the paper model. -/
structure ShiftLaws
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field) : Prop where
  one_mul : forall value, ops.mul ops.one value = value
  mul_assoc : forall left middle right,
    ops.mul (ops.mul left middle) right =
      ops.mul left (ops.mul middle right)
  mul_zero : forall value, ops.mul value ops.zero = ops.zero
  mul_add : forall left middle right,
    ops.mul left (ops.add middle right) =
      ops.add (ops.mul left middle) (ops.mul left right)

/-- Finite exponentiation using only the verifier-selected operations. -/
def power
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field)
    (value : Field) : Nat -> Field
  | 0 => ops.one
  | exponent + 1 => ops.mul value (power ops value exponent)

/-- Powers split exactly over addition under the stated minimal laws. -/
theorem power_add
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ShiftLaws ops)
    (value : Field)
    (left right : Nat) :
    power ops value (left + right) =
      ops.mul (power ops value left) (power ops value right) := by
  induction left with
  | zero =>
      simpa [power] using (laws.one_mul (power ops value right)).symm
  | succ left inductionHypothesis =>
      rw [Nat.succ_add]
      simp only [power]
      rw [inductionHypothesis]
      exact (laws.mul_assoc value (power ops value left)
        (power ops value right)).symm

/-- One target coefficient for every typed carried coordinate. No flat list or
caller-selected exponent accompanies the values. -/
structure CarriedTargetCoefficients
    (Field : Type uField)
    (shape : Shape) where
  coefficient : CarriedCoordinate shape -> Field

/-- One finite target term under an explicit exponent convention. Powers are
placed on the left so the shift theorem needs no commutativity assumption. -/
def term
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : CarriedTargetCoefficients Field shape)
    (convention : CarriedTargetConvention)
    (gamma : Field)
    (coordinate : CarriedCoordinate shape) : Field :=
  ops.mul
    (power ops gamma (convention.exponent coordinate))
    (coefficients.coefficient coordinate)

private def sumTerms
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field)
    (values : List Field) : Field :=
  values.foldr ops.add ops.zero

/-- Finite target evaluation over every typed carried coordinate. -/
def evaluate
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : CarriedTargetCoefficients Field shape)
    (convention : CarriedTargetConvention)
    (gamma : Field) : Field :=
  sumTerms ops <|
    (canonicalCarriedCoordinates shape).map fun coordinate =>
      term ops coefficients convention gamma coordinate

/-- Paper helper target evaluation using the local exponent `I`. -/
def evaluateLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : CarriedTargetCoefficients Field shape)
    (gamma : Field) : Field :=
  evaluate ops coefficients .literalLocal gamma

/-- Reviewable finite expansion of the local helper target. The traversal and
exponents are derived from the shared typed carried-coordinate owner. -/
theorem evaluateLocal_eq_foldr
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : CarriedTargetCoefficients Field shape)
    (gamma : Field) :
    evaluateLocal ops coefficients gamma =
      ((canonicalCarriedCoordinates shape).map fun coordinate =>
        term ops coefficients .literalLocal gamma coordinate).foldr
          ops.add ops.zero := by
  rfl

/-- Selected absolute target evaluation using exponent `2K+k+I`. -/
def evaluateShifted
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (coefficients : CarriedTargetCoefficients Field shape)
    (gamma : Field) : Field :=
  evaluate ops coefficients .coherentAbsolute gamma

private theorem term_shifted_eq_shift_mul_local
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ShiftLaws ops)
    (coefficients : CarriedTargetCoefficients Field shape)
    (gamma : Field)
    (coordinate : CarriedCoordinate shape) :
    term ops coefficients .coherentAbsolute gamma coordinate =
      ops.mul
        (power ops gamma shape.carriedEvaluationOffset)
        (term ops coefficients .literalLocal gamma coordinate) := by
  unfold term
  change ops.mul
      (power ops gamma
        (shape.carriedEvaluationOffset + coordinate.localGammaExponent))
      (coefficients.coefficient coordinate) = _
  rw [power_add ops laws gamma shape.carriedEvaluationOffset
    coordinate.localGammaExponent]
  exact laws.mul_assoc
    (power ops gamma shape.carriedEvaluationOffset)
    (power ops gamma coordinate.localGammaExponent)
    (coefficients.coefficient coordinate)

private theorem sumTerms_map_mul_left
    {Field : Type uField}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ShiftLaws ops)
    (factor : Field) : forall values : List Field,
    sumTerms ops (values.map (ops.mul factor)) =
      ops.mul factor (sumTerms ops values)
  | [] => by
      change ops.zero = ops.mul factor ops.zero
      exact (laws.mul_zero factor).symm
  | value :: values => by
      change ops.add (ops.mul factor value)
          (sumTerms ops (values.map (ops.mul factor))) =
        ops.mul factor (ops.add value (sumTerms ops values))
      rw [sumTerms_map_mul_left ops laws factor values]
      exact (laws.mul_add factor value (sumTerms ops values)).symm

private theorem shiftedTerms_eq_map_shiftedLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ShiftLaws ops)
    (coefficients : CarriedTargetCoefficients Field shape)
    (gamma : Field) :
    (canonicalCarriedCoordinates shape).map
        (term ops coefficients .coherentAbsolute gamma) =
      ((canonicalCarriedCoordinates shape).map
        (term ops coefficients .literalLocal gamma)).map
          (ops.mul (power ops gamma shape.carriedEvaluationOffset)) := by
  rw [List.map_map]
  apply List.map_congr_left
  intro coordinate _
  exact term_shifted_eq_shift_mul_local ops laws coefficients gamma coordinate

/-- Exact finite target-shift theorem:
`T_abs(gamma) = gamma^(2K+k) * T_local(gamma)`.

This theorem does not mention or identify the target with `Q`. -/
theorem evaluateShifted_eq_shift_mul_evaluateLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : SumCheck.Finite.Ops Field)
    (laws : ShiftLaws ops)
    (coefficients : CarriedTargetCoefficients Field shape)
    (gamma : Field) :
    evaluateShifted ops coefficients gamma =
      ops.mul
        (power ops gamma shape.carriedEvaluationOffset)
        (evaluateLocal ops coefficients gamma) := by
  unfold evaluateShifted evaluateLocal evaluate
  rw [shiftedTerms_eq_map_shiftedLocal ops laws coefficients gamma]
  exact sumTerms_map_mul_left ops laws
    (power ops gamma shape.carriedEvaluationOffset)
    ((canonicalCarriedCoordinates shape).map
      (term ops coefficients .literalLocal gamma))

/-- Explicit positivity assumptions inherited from the paper dimensions
`K`, `k`, `t`, and `d`. All four are needed to construct the zero coordinate
and separate it from the positive shifted range. -/
structure PaperPositiveDimensions (shape : Shape) : Prop where
  fresh : 0 < shape.freshCount
  running : 0 < shape.runningCount
  matrix : 0 < shape.matrixCount
  coefficient : 0 < shape.coefficientCount

/-- Exponent-layout support independent of coefficient values. It records the
positions owned by a target convention, not the nonzero support of one sampled
coefficient assignment. -/
def ExponentLayoutSupport
    (shape : Shape)
    (convention : CarriedTargetConvention)
    (exponent : Nat) : Prop :=
  exists coordinate : CarriedCoordinate shape,
    convention.exponent coordinate = exponent

private def zeroCoordinate
    {shape : Shape}
    (positive : PaperPositiveDimensions shape) : CarriedCoordinate shape where
  running := ⟨0, positive.running⟩
  matrix := ⟨0, positive.matrix⟩
  coefficient := ⟨0, positive.coefficient⟩

/-- The literal-local layout contains exponent zero for paper-valid positive
dimensions. -/
theorem zero_mem_literalLocalSupport
    {shape : Shape}
    (positive : PaperPositiveDimensions shape) :
    ExponentLayoutSupport shape .literalLocal 0 := by
  refine ⟨zeroCoordinate positive, ?_⟩
  simp [CarriedTargetConvention.exponent,
    CarriedCoordinate.localGammaExponent, zeroCoordinate]

/-- The shifted layout cannot contain exponent zero because its `2K+k` offset
is strictly positive for paper-valid dimensions. -/
theorem zero_not_mem_shiftedSupport
    {shape : Shape}
    (positive : PaperPositiveDimensions shape) :
    Not (ExponentLayoutSupport shape .coherentAbsolute 0) := by
  rintro ⟨coordinate, exponentZero⟩
  change coordinate.gammaExponent = 0 at exponentZero
  unfold CarriedCoordinate.gammaExponent at exponentZero
  rw [Shape.carriedEvaluationOffset_eq] at exponentZero
  have freshPositive := positive.fresh
  omega

/-- A real support-set mismatch witness: exponent zero belongs to the literal
layout and not to the shifted layout. This is stronger than comparing the two
exponents attached to one caller-supplied coordinate. -/
theorem literalLocal_shifted_support_mismatch_witness
    {shape : Shape}
    (positive : PaperPositiveDimensions shape) :
    exists exponent,
      ExponentLayoutSupport shape .literalLocal exponent ∧
      Not (ExponentLayoutSupport shape .coherentAbsolute exponent) := by
  exact ⟨0, zero_mem_literalLocalSupport positive,
    zero_not_mem_shiftedSupport positive⟩

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetPolynomial
