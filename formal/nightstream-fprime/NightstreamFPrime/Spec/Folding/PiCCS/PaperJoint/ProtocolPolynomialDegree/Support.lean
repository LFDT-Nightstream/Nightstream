import NightstreamFPrime.Spec.SumCheck.FixedPhase
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SumCheckTruthPath

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/ProtocolPolynomialDegree/Support.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Artifact-independent one-variable polynomial support for the paper `Pi_CCS`
degree proof.

Owns: fixed-width representations over an arbitrary paper field, closure under
widening/scaling/addition/multiplication, affine Boolean-MLE and equality
slices, the strict-`b = 2` cubic, finite sums, and Boolean suffix sums.

Does not own: a protocol polynomial, a degree ceiling, SumCheck acceptance,
probability, Fiat--Shamir, Rust, R1CS, artifacts, or costs.

Emits constraints: no.

Every representation contains explicit constant-first coefficients evaluated
by the verifier-visible Horner machine. No generated artifact or executable
emitter is consulted.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree.Support

open NightstreamFPrime.Spec.SumCheck
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uField uIndex

abbrev Polynomial (Field : Type uField) :=
  SumCheck.Finite.FixedPolynomial Field

/-- A scalar function has an explicit verifier-visible representation at the
declared degree. High coefficients may be zero. -/
def Represents
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (degree : Nat)
    (function : Field -> Field) : Prop :=
  exists polynomial : Polynomial Field degree, forall point,
    polynomial.evaluate ops.toOps point = function point

/-- The paper interpolation laws imply the generic fixed-polynomial laws. -/
def polynomialLaws
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops) :
    SumCheck.Finite.FixedPolynomial.Laws ops.toOps where
  add_assoc := laws.add_assoc
  add_comm := laws.add_comm
  zero_add := laws.zero_add
  add_zero := laws.add_zero
  mul_assoc := laws.mul_assoc
  mul_comm := laws.mul_comm
  mul_zero := laws.mul_zero
  left_distrib := laws.left_distrib
  right_distrib := laws.right_distrib

namespace Represents

/-- Append only verifier-visible high zero coefficients. -/
theorem widen
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {degree target : Nat}
    {function : Field -> Field}
    (degreeLe : degree <= target)
    (represented : Represents ops degree function) :
    Represents ops target function := by
  rcases represented with ⟨polynomial, represents⟩
  refine ⟨SumCheck.Finite.FixedPolynomial.widen
    ops.toOps degreeLe polynomial, ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_widen
    ops.toOps (polynomialLaws laws), represents]

/-- A constant function has a degree-zero representation. -/
theorem constant
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (value : Field) :
    Represents ops 0 fun _ => value := by
  refine ⟨SumCheck.Finite.FixedPolynomial.constant value, ?_⟩
  intro point
  exact SumCheck.Finite.FixedPolynomial.evaluate_constant
    ops.toOps (polynomialLaws laws) value point

/-- Scalar multiplication preserves the declared degree. -/
theorem scale
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {degree : Nat}
    {function : Field -> Field}
    (scalar : Field)
    (represented : Represents ops degree function) :
    Represents ops degree fun point => ops.mul scalar (function point) := by
  rcases represented with ⟨polynomial, represents⟩
  refine ⟨SumCheck.Finite.FixedPolynomial.scale ops.toOps scalar polynomial, ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_scale
    ops.toOps (polynomialLaws laws), represents]

/-- Pointwise addition preserves a shared declared degree. -/
theorem add
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {degree : Nat}
    {left right : Field -> Field}
    (leftRepresented : Represents ops degree left)
    (rightRepresented : Represents ops degree right) :
    Represents ops degree fun point => ops.add (left point) (right point) := by
  rcases leftRepresented with ⟨leftPolynomial, leftRepresents⟩
  rcases rightRepresented with ⟨rightPolynomial, rightRepresents⟩
  refine ⟨SumCheck.Finite.FixedPolynomial.add
    ops.toOps leftPolynomial rightPolynomial, ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_add
    ops.toOps (polynomialLaws laws), leftRepresents, rightRepresents]

/-- Pointwise multiplication adds declared degrees. -/
theorem mul
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {leftDegree rightDegree : Nat}
    {left right : Field -> Field}
    (leftRepresented : Represents ops leftDegree left)
    (rightRepresented : Represents ops rightDegree right) :
    Represents ops (leftDegree + rightDegree) fun point =>
      ops.mul (left point) (right point) := by
  rcases leftRepresented with ⟨leftPolynomial, leftRepresents⟩
  rcases rightRepresented with ⟨rightPolynomial, rightRepresents⟩
  refine ⟨SumCheck.Finite.FixedPolynomial.mul
    ops.toOps leftPolynomial rightPolynomial, ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_mul
    ops.toOps (polynomialLaws laws), leftRepresents, rightRepresents]

end Represents

private def negOne
    {Field : Type uField}
    (ops : InterpolationOps Field) : Field :=
  ops.neg ops.one

private def subtract
    {Field : Type uField}
    {ops : InterpolationOps Field}
    {degree : Nat}
    (left right : Polynomial Field degree) : Polynomial Field degree :=
  SumCheck.Finite.FixedPolynomial.add ops.toOps left
    (SumCheck.Finite.FixedPolynomial.scale ops.toOps (negOne ops) right)

private theorem evaluate_subtract
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {degree : Nat}
    (left right : Polynomial Field degree)
    (point : Field) :
    (subtract (ops := ops) left right).evaluate ops.toOps point =
      ops.sub (left.evaluate ops.toOps point)
        (right.evaluate ops.toOps point) := by
  rw [subtract,
    SumCheck.Finite.FixedPolynomial.evaluate_add
      ops.toOps (polynomialLaws laws),
    SumCheck.Finite.FixedPolynomial.evaluate_scale
      ops.toOps (polynomialLaws laws)]
  unfold negOne
  rw [laws.neg_mul, laws.one_mul]
  rfl

private def affineConstant
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (value : Field) : Polynomial Field 1 :=
  SumCheck.Finite.FixedPolynomial.affine value ops.zero

private theorem evaluate_affineConstant
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (value point : Field) :
    (affineConstant ops value).evaluate ops.toOps point = value := by
  rw [affineConstant,
    SumCheck.Finite.FixedPolynomial.evaluate_affine
      ops.toOps (polynomialLaws laws),
    laws.mul_zero, laws.add_zero]

/-- Apply the exact strict-`b = 2` cubic to an affine representation. -/
theorem strictNormOfAffine
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {value : Field -> Field}
    (represented : Represents ops 1 value) :
    Represents ops 3 fun point =>
      ProtocolPolynomial.strictNormResidual ops (value point) := by
  rcases represented with ⟨valuePolynomial, valueRepresents⟩
  let plusOne := SumCheck.Finite.FixedPolynomial.add ops.toOps valuePolynomial
    (affineConstant ops ops.one)
  let minusOne := subtract (ops := ops) valuePolynomial
    (affineConstant ops ops.one)
  refine ⟨SumCheck.Finite.FixedPolynomial.mul ops.toOps
    (SumCheck.Finite.FixedPolynomial.mul ops.toOps plusOne valuePolynomial)
    minusOne, ?_⟩
  intro point
  unfold ProtocolPolynomial.strictNormResidual
  rw [SumCheck.Finite.FixedPolynomial.evaluate_mul
      ops.toOps (polynomialLaws laws),
    SumCheck.Finite.FixedPolynomial.evaluate_mul
      ops.toOps (polynomialLaws laws),
    SumCheck.Finite.FixedPolynomial.evaluate_add
      ops.toOps (polynomialLaws laws),
    evaluate_subtract laws, evaluate_affineConstant laws,
    valueRepresents]

/-- Finite weighted sums preserve one shared degree. -/
theorem weightedSum
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {Index : Type uIndex}
    {degree : Nat}
    (indices : List Index)
    (weight : Index -> Field)
    (value : Index -> Field -> Field)
    (represented : forall index, index ∈ indices ->
      Represents ops degree (value index)) :
    Represents ops degree fun point =>
      FiniteSumAlgebra.sumMap ops indices fun index =>
        ops.mul (weight index) (value index point) := by
  induction indices with
  | nil =>
      refine ⟨SumCheck.Finite.FixedPolynomial.zero ops.toOps degree, ?_⟩
      intro point
      exact SumCheck.Finite.FixedPolynomial.evaluate_zero
        ops.toOps (polynomialLaws laws) degree point
  | cons index indices inductionHypothesis =>
      rcases represented index (by simp) with
        ⟨headPolynomial, headRepresents⟩
      rcases inductionHypothesis (by
        intro tail tailMember
        exact represented tail (by simp [tailMember])) with
        ⟨tailPolynomial, tailRepresents⟩
      refine ⟨SumCheck.Finite.FixedPolynomial.add ops.toOps
        (SumCheck.Finite.FixedPolynomial.scale ops.toOps
          (weight index) headPolynomial)
        tailPolynomial, ?_⟩
      intro point
      rw [SumCheck.Finite.FixedPolynomial.evaluate_add
          ops.toOps (polynomialLaws laws),
        SumCheck.Finite.FixedPolynomial.evaluate_scale
          ops.toOps (polynomialLaws laws),
        headRepresents, tailRepresents]
      rfl

/-- Replace exactly one coordinate between a fixed prefix and suffix. -/
def cubeSlice
    {Field : Type uField}
    {variables : Nat}
    (before after : List Field)
    (length : before.length + 1 + after.length = variables)
    (point : Field) : CubePoint Field variables where
  coordinates := before ++ point :: after
  dimension := by simp; omega

/-- Every coordinate slice of an explicit Boolean-table MLE is affine. -/
theorem evaluateCoordinates_affine
    {Field : Type uField}
    {variables : Nat}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (table : BooleanTable Field variables)
    (before after : List Field)
    (length : before.length + 1 + after.length = variables) :
    Represents ops 1 fun point =>
      table.evaluateCoordinates ops (before ++ point :: after) := by
  induction table generalizing before after with
  | leaf => simp at length
  | @branch variables low high lowInduction highInduction =>
      cases before with
      | nil =>
          refine ⟨SumCheck.Finite.FixedPolynomial.affine
            (low.evaluateCoordinates ops after)
            (ops.sub
              (high.evaluateCoordinates ops after)
              (low.evaluateCoordinates ops after)), ?_⟩
          intro point
          rw [SumCheck.Finite.FixedPolynomial.evaluate_affine
            ops.toOps (polynomialLaws laws)]
          rfl
      | cons head before =>
          have tailLength : before.length + 1 + after.length = variables := by
            simp only [List.length_cons] at length
            omega
          rcases lowInduction before after tailLength with
            ⟨lowPolynomial, lowRepresents⟩
          rcases highInduction before after tailLength with
            ⟨highPolynomial, highRepresents⟩
          refine ⟨SumCheck.Finite.FixedPolynomial.add ops.toOps lowPolynomial
            (SumCheck.Finite.FixedPolynomial.scale ops.toOps head
              (subtract (ops := ops) highPolynomial lowPolynomial)), ?_⟩
          intro point
          rw [SumCheck.Finite.FixedPolynomial.evaluate_add
              ops.toOps (polynomialLaws laws),
            SumCheck.Finite.FixedPolynomial.evaluate_scale
              ops.toOps (polynomialLaws laws),
            evaluate_subtract laws, lowRepresents, highRepresents]
          rfl

private theorem equalityFactor_eq_affine
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (point beta : Field) :
    SumCheckTruthPath.equalityFactor ops point beta =
      ops.add (ops.sub ops.one beta)
        (ops.mul point (ops.sub beta (ops.sub ops.one beta))) := by
  calc
    SumCheckTruthPath.equalityFactor ops point beta =
        ops.add
          (ops.mul (ops.sub ops.one point) (ops.sub ops.one beta))
          (ops.mul point beta) := by
      rfl
    _ = ops.add
        (ops.add (ops.sub ops.one beta)
          (ops.neg (ops.mul point (ops.sub ops.one beta))))
        (ops.mul point beta) := by
      unfold InterpolationOps.sub
      rw [laws.right_distrib ops.one (ops.neg point)
        (ops.add ops.one (ops.neg beta))]
      rw [laws.one_mul, laws.neg_mul]
    _ = ops.add (ops.sub ops.one beta)
        (ops.add (ops.mul point beta)
          (ops.neg (ops.mul point (ops.sub ops.one beta)))) := by
      letI : Std.Associative ops.add := ⟨laws.add_assoc⟩
      letI : Std.Commutative ops.add := ⟨laws.add_comm⟩
      ac_rfl
    _ = ops.add (ops.sub ops.one beta)
        (ops.mul point (ops.sub beta (ops.sub ops.one beta))) := by
      congr 1
      exact (FiniteSumAlgebra.mul_sub ops laws point beta
        (ops.sub ops.one beta)).symm

/-- Every coordinate slice of an equality polynomial is affine. -/
theorem pointEqualityCoordinates_affine
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (before after beta : List Field)
    (length : before.length + 1 + after.length = beta.length) :
    Represents ops 1 fun point =>
      SumCheckTruthPath.pointEqualityCoordinates ops
        (before ++ point :: after) beta := by
  induction beta generalizing before after with
  | nil => simp at length
  | cons betaHead betaTail inductionHypothesis =>
      cases before with
      | nil =>
          let tailEquality :=
            SumCheckTruthPath.pointEqualityCoordinates ops after betaTail
          let oneMinusBeta := ops.sub ops.one betaHead
          let factor := SumCheck.Finite.FixedPolynomial.affine oneMinusBeta
            (ops.sub betaHead oneMinusBeta)
          refine ⟨SumCheck.Finite.FixedPolynomial.scale
            ops.toOps tailEquality factor, ?_⟩
          intro point
          rw [SumCheck.Finite.FixedPolynomial.evaluate_scale
              ops.toOps (polynomialLaws laws),
            SumCheck.Finite.FixedPolynomial.evaluate_affine
              ops.toOps (polynomialLaws laws)]
          change ops.mul tailEquality
              (ops.add oneMinusBeta
                (ops.mul point (ops.sub betaHead oneMinusBeta))) =
            ops.mul (SumCheckTruthPath.equalityFactor ops point betaHead)
              tailEquality
          rw [laws.mul_comm tailEquality, equalityFactor_eq_affine laws]
      | cons head before =>
          have tailLength : before.length + 1 + after.length =
              betaTail.length := by
            simp only [List.length_cons] at length
            omega
          rcases inductionHypothesis before after tailLength with
            ⟨tailPolynomial, tailRepresents⟩
          refine ⟨SumCheck.Finite.FixedPolynomial.scale ops.toOps
            (SumCheckTruthPath.equalityFactor ops head betaHead)
            tailPolynomial, ?_⟩
          intro point
          rw [SumCheck.Finite.FixedPolynomial.evaluate_scale
              ops.toOps (polynomialLaws laws),
            tailRepresents]
          rfl

/-- Boolean suffix summation preserves a fixed per-variable degree. -/
theorem sumCompletions_represents
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {degree : Nat}
    (polynomial : List Field -> Field)
    (fixed : List Field)
    (remaining : Nat)
    (represented : forall vertex : BooleanVertex remaining,
      Represents ops degree fun point =>
        polynomial
          ((fixed ++ [point]) ++
            SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex)) :
    Represents ops degree fun point =>
      SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps polynomial
        (fixed ++ [point]) remaining := by
  have summed := weightedSum laws
    (BooleanVertex.all remaining)
    (fun _ => ops.one)
    (fun vertex point =>
      polynomial
        ((fixed ++ [point]) ++
          SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex))
    (by
      intro vertex _
      exact represented vertex)
  rcases summed with ⟨sumPolynomial, sumRepresents⟩
  refine ⟨sumPolynomial, ?_⟩
  intro point
  rw [sumRepresents]
  change _ = SumCheck.Finite.HypercubeTruth.sumCompletions ops.toOps
    polynomial (fixed ++ [point]) remaining
  rw [SumCheckTruthPath.sumCompletions_eq_vertexSum ops laws]
  unfold FiniteSumAlgebra.sumMap
  simp only [laws.one_mul]

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree.Support
