import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree.Sparse
import Nightstream.SuperNeo.SumCheck.FixedPhase.Sequential

/-!
Exact per-round degree bound for the paper-joint `Pi_CCS` polynomial.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: the single joint SumCheck.
Constraint family: semantic degree and honest-message representability only;
this file emits no rows.

Owns: a fixed-polynomial representation of every one-coordinate slice of the
actual nonlinear `ProtocolPolynomial.polynomial`; the syntax-derived CCS
ceiling, strict-`b = 2` quartic ceiling, and carried-evaluation quadratic; and
representability of every canonical expected SumCheck round at the exact
verifier-owned bound.

Does not own: SumCheck completeness, an honest execution, probability,
Fiat--Shamir, Rust, R1CS, artifacts, row removal, or costs.

Emits constraints: no.

Authority boundary: the CCS ceiling is computed from explicit sparse monomial
syntax. Every Boolean-table MLE and equality selector is independently proved
affine in the exposed coordinate. No declared degree metadata or
implementation artifact enters the proof.
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree

open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open ProtocolPolynomialDegree.Support
open ProtocolPolynomialDegree.Sparse

universe uField

private theorem tableSlice_affine
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (table : BooleanTable Field shape.cubeVariables)
    (before after : List Field)
    (length : before.length + 1 + after.length = shape.cubeVariables) :
    Represents ops 1 fun point =>
      table.evaluate ops (cubeSlice before after length point) := by
  simpa [BooleanTable.evaluate, cubeSlice] using
    (evaluateCoordinates_affine ops laws table before after length)

private theorem selectorSlice_affine
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (target : CubePoint Field shape.cubeVariables)
    (before after : List Field)
    (length : before.length + 1 + after.length = shape.cubeVariables) :
    Represents ops 1 fun point =>
      SumCheckTruthPath.pointEquality ops
        (cubeSlice before after length point) target := by
  have targetLength : target.coordinates.length = shape.cubeVariables :=
    target.dimension
  simpa [SumCheckTruthPath.pointEquality, cubeSlice] using
    (pointEqualityCoordinates_affine ops laws before after target.coordinates
      (by omega))

private theorem ccsGatedSlice_represents
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (before after : List Field)
    (length : before.length + 1 + after.length = shape.cubeVariables) :
    Represents ops
      data.constraintPolynomial.canonicalEqualityGatedDegreeBound
      fun point =>
        ops.mul
          (SumCheckTruthPath.pointEquality ops
            (cubeSlice before after length point) alpha)
          (ProtocolPolynomial.ccsAtMessage ops data.toVerifierInput gamma
            (ProtocolPolynomial.messageAt ops data
              (cubeSlice before after length point))) := by
  let selector : Field -> Field := fun point =>
    SumCheckTruthPath.pointEquality ops
      (cubeSlice before after length point) alpha
  have selectorAffine : Represents ops 1 selector :=
    selectorSlice_affine ops laws alpha before after length
  let sourceValue : Fin shape.freshCount -> Field -> Field :=
    fun source point =>
      CCSResidualTable.evaluatePolynomial ops data.constraintPolynomial
        (fun matrix =>
          (data.freshMatrixImages source matrix).evaluate ops
            (cubeSlice before after length point))
  have sourceGated : forall source,
      Represents ops
        data.constraintPolynomial.canonicalEqualityGatedDegreeBound
        fun point => ops.mul (selector point) (sourceValue source point) := by
    intro source
    apply equalityGated_represents laws data.constraintPolynomial selector
      selectorAffine
    intro matrix
    exact tableSlice_affine ops laws (data.freshMatrixImages source matrix)
      before after length
  have summed := weightedSum laws
    (canonicalFinIndices shape.freshCount)
    (fun source => TargetPolynomial.power ops.toOps gamma source.val)
    (fun source point => ops.mul (selector point) (sourceValue source point))
    (by
      intro source _
      exact sourceGated source)
  rcases summed with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  change
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.freshCount)
        (fun source =>
          ops.mul (TargetPolynomial.power ops.toOps gamma source.val)
            (ops.mul (selector point) (sourceValue source point))) =
      ops.mul (selector point)
        (FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.freshCount) fun source =>
            ops.mul (TargetPolynomial.power ops.toOps gamma source.val)
              (sourceValue source point))
  rw [← FiniteSumAlgebra.sumMap_mul_left ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro source _
  calc
    ops.mul
        (TargetPolynomial.power ops.toOps gamma source.val)
        (ops.mul
          (SumCheckTruthPath.pointEquality ops
            (cubeSlice before after length point) alpha)
          _) =
      ops.mul
        (SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) alpha)
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma source.val) _) := by
        rw [← laws.mul_assoc, laws.mul_comm
          (TargetPolynomial.power ops.toOps gamma source.val),
          laws.mul_assoc]
    _ = _ := rfl

private theorem normGatedSlice_represents
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (before after : List Field)
    (length : before.length + 1 + after.length = shape.cubeVariables) :
    Represents ops 4 fun point =>
      ops.mul
        (SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) alpha)
        (ProtocolPolynomial.normAtMessage ops gamma
          (ProtocolPolynomial.messageAt ops data
            (cubeSlice before after length point))) := by
  let selector : Field -> Field := fun point =>
    SumCheckTruthPath.pointEquality ops
      (cubeSlice before after length point) alpha
  have selectorAffine : Represents ops 1 selector :=
    selectorSlice_affine ops laws alpha before after length
  let sourceNorm : Fin shape.sourceCount -> Field -> Field :=
    fun source point =>
      ProtocolPolynomial.strictNormResidual ops
        ((data.sourceAssignments source).evaluate ops
          (cubeSlice before after length point))
  have sourceQuartic : forall source,
      Represents ops 4 fun point =>
        ops.mul (selector point) (sourceNorm source point) := by
    intro source
    have assignmentAffine := tableSlice_affine ops laws
      (data.sourceAssignments source) before after length
    have cubic := strictNormOfAffine laws assignmentAffine
    exact Represents.mul laws selectorAffine cubic
  have summed := weightedSum laws
    (canonicalFinIndices shape.sourceCount)
    (fun source => TargetPolynomial.power ops.toOps gamma source.val)
    (fun source point => ops.mul (selector point) (sourceNorm source point))
    (by
      intro source _
      exact sourceQuartic source)
  rcases summed with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  change
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices shape.sourceCount)
        (fun source =>
          ops.mul (TargetPolynomial.power ops.toOps gamma source.val)
            (ops.mul (selector point) (sourceNorm source point))) =
      ops.mul (selector point)
        (FiniteSumAlgebra.sumMap ops
          (canonicalFinIndices shape.sourceCount) fun source =>
            ops.mul (TargetPolynomial.power ops.toOps gamma source.val)
              (sourceNorm source point))
  rw [← FiniteSumAlgebra.sumMap_mul_left ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro source _
  calc
    ops.mul
        (TargetPolynomial.power ops.toOps gamma source.val)
        (ops.mul
          (SumCheckTruthPath.pointEquality ops
            (cubeSlice before after length point) alpha)
          _) =
      ops.mul
        (SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) alpha)
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma source.val) _) := by
        rw [← laws.mul_assoc, laws.mul_comm
          (TargetPolynomial.power ops.toOps gamma source.val),
          laws.mul_assoc]
    _ = _ := rfl

private theorem carriedSlice_represents
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (gamma : Field)
    (before after : List Field)
    (length : before.length + 1 + after.length = shape.cubeVariables) :
    Represents ops 2 fun point =>
      ProtocolPolynomial.carriedAtMessage ops data.toVerifierInput gamma
        (cubeSlice before after length point)
        (ProtocolPolynomial.messageAt ops data
          (cubeSlice before after length point)) := by
  let priorSelector : Field -> Field := fun point =>
    SumCheckTruthPath.pointEquality ops
      (cubeSlice before after length point) data.priorPoint
  have priorSelectorAffine : Represents ops 1 priorSelector :=
    selectorSlice_affine ops laws data.priorPoint before after length
  let image : CarriedCoordinate shape -> Field -> Field :=
    fun coordinate point =>
      (data.carriedImages coordinate).evaluate ops
        (cubeSlice before after length point)
  have imageAffine : forall coordinate,
      Represents ops 1 (image coordinate) := by
    intro coordinate
    exact tableSlice_affine ops laws (data.carriedImages coordinate)
      before after length
  have imageSum := weightedSum laws
    (canonicalCarriedCoordinates shape)
    (fun coordinate => TargetPolynomial.power ops.toOps gamma
      coordinate.localGammaExponent)
    image
    (by
      intro coordinate _
      exact imageAffine coordinate)
  have multiplied := Represents.mul laws priorSelectorAffine imageSum
  rcases multiplied with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  rfl

/-- Every one-coordinate slice of the actual paper polynomial has the exact
verifier-owned syntax-derived degree ceiling. -/
theorem polynomial_slice_represents
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (before after : List Field)
    (length : before.length + 1 + after.length = shape.cubeVariables) :
    Represents ops data.toVerifierInput.sumcheckDegreeBound fun point =>
      ProtocolPolynomial.polynomial ops data alpha gamma
        (before ++ point :: after) := by
  let ccsDegree :=
    data.constraintPolynomial.canonicalEqualityGatedDegreeBound
  let degree := data.toVerifierInput.sumcheckDegreeBound
  have ccsLe : ccsDegree <= degree := by
    unfold ccsDegree degree ProtocolPolynomial.VerifierInput.sumcheckDegreeBound
    exact Nat.le_max_left _ _
  have fourLe : 4 <= degree := by
    unfold degree ProtocolPolynomial.VerifierInput.sumcheckDegreeBound
    exact Nat.le_max_right _ _
  have twoLe : 2 <= degree := Nat.le_trans (by omega) fourLe
  have ccs := Represents.widen laws ccsLe
    (ccsGatedSlice_represents ops laws data alpha gamma before after length)
  have norm := Represents.widen laws fourLe
    (normGatedSlice_represents ops laws data alpha gamma before after length)
  have carried := Represents.widen laws twoLe
    (carriedSlice_represents ops laws data gamma before after length)
  have normScaled := Represents.scale laws
    (TargetPolynomial.power ops.toOps gamma shape.freshCount) norm
  have normShifted : Represents ops degree fun point =>
      ops.mul
        (SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) alpha)
        (SignedJointIdentity.gammaTerm ops gamma shape.freshCount
          (ProtocolPolynomial.normAtMessage ops gamma
            (ProtocolPolynomial.messageAt ops data
              (cubeSlice before after length point)))) := by
    rcases normScaled with ⟨polynomial, represents⟩
    refine ⟨polynomial, ?_⟩
    intro point
    rw [represents]
    calc
      ops.mul
          (TargetPolynomial.power ops.toOps gamma shape.freshCount)
          (ops.mul
            (SumCheckTruthPath.pointEquality ops
              (cubeSlice before after length point) alpha) _) =
        ops.mul
          (SumCheckTruthPath.pointEquality ops
            (cubeSlice before after length point) alpha)
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma shape.freshCount) _) := by
          rw [← laws.mul_assoc, laws.mul_comm
            (TargetPolynomial.power ops.toOps gamma shape.freshCount),
            laws.mul_assoc]
      _ = _ := rfl
  have first := Represents.add laws ccs normShifted
  have firstExact : Represents ops degree fun point =>
      ops.mul
        (SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) alpha)
        (ops.add
          (ProtocolPolynomial.ccsAtMessage ops data.toVerifierInput gamma
            (ProtocolPolynomial.messageAt ops data
              (cubeSlice before after length point)))
          (SignedJointIdentity.gammaTerm ops gamma shape.freshCount
            (ProtocolPolynomial.normAtMessage ops gamma
              (ProtocolPolynomial.messageAt ops data
                (cubeSlice before after length point))))) := by
    rcases first with ⟨polynomial, represents⟩
    refine ⟨polynomial, ?_⟩
    intro point
    rw [represents]
    exact (laws.left_distrib
      (SumCheckTruthPath.pointEquality ops
        (cubeSlice before after length point) alpha)
      (ProtocolPolynomial.ccsAtMessage ops data.toVerifierInput gamma
        (ProtocolPolynomial.messageAt ops data
          (cubeSlice before after length point)))
      (SignedJointIdentity.gammaTerm ops gamma shape.freshCount
        (ProtocolPolynomial.normAtMessage ops gamma
          (ProtocolPolynomial.messageAt ops data
            (cubeSlice before after length point))))).symm
  have carriedScaled := Represents.scale laws
    (TargetPolynomial.power ops.toOps gamma shape.carriedEvaluationOffset)
    carried
  have total := Represents.add laws firstExact carriedScaled
  rcases total with ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  change
    ops.add
        (ops.mul
          (SumCheckTruthPath.pointEquality ops
            (cubeSlice before after length point) alpha)
          (ops.add
            (ProtocolPolynomial.ccsAtMessage ops data.toVerifierInput gamma
              (ProtocolPolynomial.messageAt ops data
                (cubeSlice before after length point)))
            (SignedJointIdentity.gammaTerm ops gamma shape.freshCount
              (ProtocolPolynomial.normAtMessage ops gamma
                (ProtocolPolynomial.messageAt ops data
                  (cubeSlice before after length point))))))
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma
            shape.carriedEvaluationOffset)
          (ProtocolPolynomial.carriedAtMessage ops data.toVerifierInput gamma
            (cubeSlice before after length point)
            (ProtocolPolynomial.messageAt ops data
              (cubeSlice before after length point)))) =
      ProtocolPolynomial.polynomial ops data alpha gamma
        (before ++ point :: after)
  unfold ProtocolPolynomial.polynomial
  rw [dif_pos]
  rfl

/-- At every reachable challenge prefix, the next honest round polynomial is
representable without seeing any future challenge. This is the primary degree
theorem used by the causal message-before-challenge honest prover. -/
theorem sequentialRoundRepresentable
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    SumCheck.Finite.FixedPhase.Sequential.RoundRepresentable ops.toOps
      (ProtocolPolynomial.polynomial ops data alpha gamma)
      data.toVerifierInput.sumcheckDegreeBound shape.cubeVariables := by
  intro fixed remaining length
  exact sumCompletions_represents ops laws
    (ProtocolPolynomial.polynomial ops data alpha gamma)
    fixed remaining fun vertex => by
      simpa [List.append_assoc] using
        (polynomial_slice_represents ops laws data alpha gamma fixed
          (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex)
          (by
            have vertexLength :=
              SumCheckTruthPath.VertexEncoding.fieldCoordinates_length ops vertex
            rw [vertexLength]
            exact length))

private theorem roundRepresentable_expectedPolynomialsFrom
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (roundRepresentable :
      SumCheck.Finite.FixedPhase.Sequential.RoundRepresentable ops.toOps
        (ProtocolPolynomial.polynomial ops data alpha gamma)
        data.toVerifierInput.sumcheckDegreeBound shape.cubeVariables) :
    forall (fixed challenges : List Field),
      fixed.length + challenges.length = shape.cubeVariables ->
      forall expected,
        expected ∈ SumCheck.Finite.HypercubeTruth.expectedPolynomialsFrom
          ops.toOps (ProtocolPolynomial.polynomial ops data alpha gamma)
          fixed challenges ->
        Represents ops data.toVerifierInput.sumcheckDegreeBound expected
  | fixed, [], _, expected, member => by
      simp [SumCheck.Finite.HypercubeTruth.expectedPolynomialsFrom] at member
  | fixed, challenge :: challenges, totalLength, expected, member => by
      simp only [SumCheck.Finite.HypercubeTruth.expectedPolynomialsFrom,
        List.mem_cons] at member
      rcases member with rfl | member
      · exact roundRepresentable fixed challenges.length (by
          simp only [List.length_cons] at totalLength
          omega)
      · exact roundRepresentable_expectedPolynomialsFrom ops data alpha gamma
          roundRepresentable (fixed ++ [challenge]) challenges (by
            simp only [List.length_append, List.length_cons, List.length_nil]
            simp only [List.length_cons] at totalLength
            omega) expected member

/-- Every canonical expected round of the actual paper polynomial admits an
explicit verifier-visible polynomial at exactly the verifier-owned degree
ceiling. This closes the semantic degree premise left open by
`ProtocolPolynomial.canonicalGhosts_honest`. -/
theorem expectedRoundsRepresentable
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (roundPoint : CubePoint Field shape.cubeVariables) :
    SumCheck.Finite.FixedPhase.ExpectedRoundsRepresentable ops.toOps
      (ProtocolPolynomial.polynomial ops data alpha gamma)
      data.toVerifierInput.sumcheckDegreeBound
      roundPoint.coordinates := by
  intro expected member
  exact roundRepresentable_expectedPolynomialsFrom ops data alpha gamma
    (sequentialRoundRepresentable ops laws data alpha gamma)
    [] roundPoint.coordinates (by simpa using roundPoint.dimension)
    expected (by
      simpa [SumCheck.Finite.FixedPhase.expectedRounds,
        SumCheck.Finite.HypercubeTruth.expectedPolynomials] using member)

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree
