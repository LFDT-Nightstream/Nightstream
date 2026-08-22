import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanHypercubeSum
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanReproduction
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetPolynomial

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/SignedJointIdentity.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Exact signed joint identity for the paper-level one-SumCheck `Pi_CCS` model.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: pre-SumCheck construction of `T_abs(C) - sum_x Q(x,A,C)`.
Constraint family: signed composition of CCS, norm, and carried-evaluation
residuals before alpha/gamma sampling.

Owns: an explicit pointwise `F`, `NC`, `Eval`, and `Q`; the corrected shifted
target; the Boolean-hypercube sum of `Q`; three signed residual blocks; and
the exact identity between the paper difference and those blocks.

Does not own: construction of the input tables from concrete CCS/norm/ring
data, base-to-extension embeddings, the paper-source target audit,
SumCheck messages or soundness, Fiat--Shamir, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: `JointData` contains explicit Boolean tables and claimed
carried coefficients. It contains no evaluator and no proposition asserting
the desired identity. Every evaluation, finite sum, gamma exponent, sign, and
block offset is derived here from the shared typed indices. A later refinement
must prove that the concrete CCS, norm, and coefficient-matrix constructions
instantiate these tables exactly.

| Protocol object | Phase | Mathematical definition | Proven result |
|---|---|---|---|
| `F` | pointwise `Q` construction | `sum_i C^i * ccs_i(x)` | derived from typed fresh tables |
| `NC` | pointwise `Q` construction | `sum_i C^i * norm_i(x)` | shifted by `C^K` only in `Q` |
| `Eval` | pointwise `Q` construction | `eq(x,r) * sum_I C^I * image_I(x)` | derived from typed carried tables |
| `Q` | pointwise joint polynomial | `eq(x,A)*(F+C^K*NC)+C^(2K+k)*Eval` | exact finite definition |
| `T_abs` | corrected target | `C^(2K+k) * sum_I C^I * claimed_I` | reuses the target-convention owner |
| signed blocks | pre-SumCheck residual | `-CCS - norm + (claimed-evaluation)` | exact table-derived formula |
| `paperDifference_eq_signedResidualBlocks` | joint identity | `T_abs - sum_x Q` | equality for every `A,C` |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedJointIdentity

universe uField uIndex uLeft uRight

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open FiniteSumAlgebra

/-- Explicit extension-carrier tables entering the paper's one-joint
polynomial. Family cardinalities are intrinsic to `Shape`. -/
structure JointData (Field : Type uField) (shape : Shape) where
  ccs : Fin shape.freshCount -> BooleanTable Field shape.cubeVariables
  norm : Fin shape.sourceCount -> BooleanTable Field shape.cubeVariables
  priorPoint : CubePoint Field shape.cubeVariables
  carriedImage :
    CarriedCoordinate shape -> BooleanTable Field shape.cubeVariables
  claimedCoefficient : CarriedCoordinate shape -> Field

namespace JointData

/-- Equality of all five typed protocol families is equality of the complete
joint data object. -/
@[ext] theorem ext
    {Field : Type uField}
    {shape : Shape}
    (left right : JointData Field shape)
    (ccs : left.ccs = right.ccs)
    (norm : left.norm = right.norm)
    (priorPoint : left.priorPoint = right.priorPoint)
    (carriedImage : left.carriedImage = right.carriedImage)
    (claimedCoefficient :
      left.claimedCoefficient = right.claimedCoefficient) :
    left = right := by
  cases left
  cases right
  simp_all

end JointData

/-- The existing target-polynomial owner consumes exactly this data's claimed
carried coefficients. -/
def JointData.targetCoefficients
    {Field : Type uField}
    {shape : Shape}
    (data : JointData Field shape) :
    TargetPolynomial.CarriedTargetCoefficients Field shape where
  coefficient := data.claimedCoefficient

/-- Compatibility name retained at the signed-identity boundary. The shared
finite-sum owner proves every rearrangement law used below. Keeping this
definition local preserves the original reduction behavior of audit lemmas
which unfold `SignedJointIdentity.sumMap` explicitly. -/
def sumMap
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (indices : List Index)
    (value : Index -> Field) : Field :=
  BooleanTable.finiteSum ops (indices.map value)

private theorem sumMap_congr
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (indices : List Index)
    (left right : Index -> Field)
    (equal : forall index, index ∈ indices -> left index = right index) :
    sumMap ops indices left = sumMap ops indices right :=
  FiniteSumAlgebra.sumMap_congr ops indices left right equal

private theorem mul_sub
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left middle right : Field) :
    ops.mul left (ops.sub middle right) =
      ops.sub (ops.mul left middle) (ops.mul left right) :=
  FiniteSumAlgebra.mul_sub ops laws left middle right

private theorem sumMap_add
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (indices : List Index)
    (left right : Index -> Field) :
    sumMap ops indices (fun index => ops.add (left index) (right index)) =
      ops.add (sumMap ops indices left) (sumMap ops indices right) :=
  FiniteSumAlgebra.sumMap_add ops laws indices left right

private theorem sumMap_mul_left
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (factor : Field)
    (indices : List Index)
    (value : Index -> Field) :
    sumMap ops indices (fun index => ops.mul factor (value index)) =
      ops.mul factor (sumMap ops indices value) :=
  FiniteSumAlgebra.sumMap_mul_left ops laws factor indices value

private theorem sumMap_sub
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (indices : List Index)
    (left right : Index -> Field) :
    sumMap ops indices (fun index => ops.sub (left index) (right index)) =
      ops.sub (sumMap ops indices left) (sumMap ops indices right) :=
  FiniteSumAlgebra.sumMap_sub ops laws indices left right

private theorem sumMap_swap
    {Field : Type uField}
    {Left : Type uLeft}
    {Right : Type uRight}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (leftIndices : List Left)
    (rightIndices : List Right)
    (value : Left -> Right -> Field) :
    sumMap ops leftIndices (fun left => sumMap ops rightIndices (value left)) =
      sumMap ops rightIndices (fun right =>
        sumMap ops leftIndices (fun left => value left right)) :=
  FiniteSumAlgebra.sumMap_swap ops laws leftIndices rightIndices value

/-- Gamma monomial multiplied by one finite value. -/
def gammaTerm
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (gamma : Field)
    (exponent : Nat)
    (value : Field) : Field :=
  ops.mul (TargetPolynomial.power ops.toOps gamma exponent) value

/-- Pointwise `F(x,C)` from the `K` typed CCS tables. -/
def ccsAt
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) : Field :=
  sumMap ops (canonicalFinIndices shape.freshCount) fun source =>
    gammaTerm ops gamma source.val ((data.ccs source).valueAt vertex)

/-- Pointwise unshifted `NC(x,C)` from the `K+k` typed norm tables. -/
def normAt
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) : Field :=
  sumMap ops (canonicalFinIndices shape.sourceCount) fun source =>
    gammaTerm ops gamma source.val ((data.norm source).valueAt vertex)

/-- Pointwise unshifted `Eval(x,C)`, including the prior-point equality
factor and every typed carried matrix-image coefficient. -/
def carriedAt
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) : Field :=
  ops.mul (vertex.equalityWeight ops data.priorPoint) <|
    sumMap ops (canonicalCarriedCoordinates shape) fun coordinate =>
      gammaTerm ops gamma coordinate.localGammaExponent
        ((data.carriedImage coordinate).valueAt vertex)

/-- Literal pointwise paper `Q(x,A,C)` under the coherent absolute carried
offset. -/
def qAt
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) : Field :=
  ops.add
    (ops.mul (vertex.equalityWeight ops alpha)
      (ops.add
        (ccsAt ops data gamma vertex)
        (gammaTerm ops gamma shape.freshCount
          (normAt ops data gamma vertex))))
    (gammaTerm ops gamma shape.carriedEvaluationOffset
      (carriedAt ops data gamma vertex))

/-- Explicit Boolean-hypercube sum of the pointwise `Q`. -/
def summedQ
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) : Field :=
  sumMap ops (BooleanVertex.all shape.cubeVariables) fun vertex =>
    qAt ops data alpha gamma vertex

/-- Corrected absolute target. Its exponent convention is owned by
`TargetPolynomial`, not restated as caller data here. -/
def targetAbsolute
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (gamma : Field) : Field :=
  TargetPolynomial.evaluateShifted ops.toOps data.targetCoefficients gamma

/-- The exact pre-SumCheck equality residual. -/
def paperDifference
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) : Field :=
  ops.sub (targetAbsolute ops data gamma) (summedQ ops data alpha gamma)

private theorem weightedSum_indexedTables
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables : Nat}
    (indices : List Index)
    (tables : Index -> BooleanTable Field variables)
    (weights : Index -> Field)
    (point : CubePoint Field variables) :
    sumMap ops (BooleanVertex.all variables) (fun vertex =>
        ops.mul (vertex.equalityWeight ops point)
          (sumMap ops indices fun index =>
            ops.mul (weights index) ((tables index).valueAt vertex))) =
      sumMap ops indices fun index =>
        ops.mul (weights index)
          ((tables index).equalityWeightedSum ops point) := by
  exact BooleanReproduction.equalityWeighted_sumMap ops laws indices weights
    (fun index vertex => (tables index).valueAt vertex) point

/-- Alpha-specialized CCS residual block. -/
def ccsResidualBlock
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) : Field :=
  sumMap ops (canonicalFinIndices shape.freshCount) fun source =>
    gammaTerm ops gamma source.val
      ((data.ccs source).equalityWeightedSum ops alpha)

/-- Alpha-specialized norm residual block before the paper's `C^K` offset. -/
def normResidualLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) : Field :=
  sumMap ops (canonicalFinIndices shape.sourceCount) fun source =>
    gammaTerm ops gamma source.val
      ((data.norm source).equalityWeightedSum ops alpha)

/-- Alpha-specialized norm residual block, including the paper's `C^K`
offset. -/
def normResidualBlock
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) : Field :=
  gammaTerm ops gamma shape.freshCount <|
    normResidualLocal ops data alpha gamma

/-- Unshifted carried evaluation block after summing over the Boolean cube. -/
def carriedEvaluationLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (gamma : Field) : Field :=
  sumMap ops (canonicalCarriedCoordinates shape) fun coordinate =>
    gammaTerm ops gamma coordinate.localGammaExponent
      ((data.carriedImage coordinate).equalityWeightedSum ops data.priorPoint)

/-- One unshifted carried claimed-minus-derived residual block. -/
def carriedResidualLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (gamma : Field) : Field :=
  sumMap ops (canonicalCarriedCoordinates shape) fun coordinate =>
    gammaTerm ops gamma coordinate.localGammaExponent <|
      ops.sub
        (data.claimedCoefficient coordinate)
        ((data.carriedImage coordinate).equalityWeightedSum
          ops data.priorPoint)

/-- Absolute carried residual block, shifted to `2K+k+I`. -/
def carriedResidualBlock
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (gamma : Field) : Field :=
  gammaTerm ops gamma shape.carriedEvaluationOffset
    (carriedResidualLocal ops data gamma)

/-- The signed block composition forced by `T_abs - sum Q`: CCS and norm are
negative; claimed-minus-evaluation carried residuals are positive. -/
def signedResidualBlocks
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) : Field :=
  ops.add
    (ops.neg (ccsResidualBlock ops data alpha gamma))
    (ops.add
      (ops.neg (normResidualBlock ops data alpha gamma))
      (carriedResidualBlock ops data gamma))

private theorem summedQ_eq_residualBlocks
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    summedQ ops data alpha gamma =
      ops.add
        (ccsResidualBlock ops data alpha gamma)
        (ops.add
          (normResidualBlock ops data alpha gamma)
          (gammaTerm ops gamma shape.carriedEvaluationOffset
            (carriedEvaluationLocal ops data gamma))) := by
  unfold summedQ qAt
  rw [sumMap_add ops laws]
  have ccsExact := weightedSum_indexedTables ops laws
    (canonicalFinIndices shape.freshCount) data.ccs
    (fun source => TargetPolynomial.power ops.toOps gamma source.val) alpha
  have normExact := weightedSum_indexedTables ops laws
    (canonicalFinIndices shape.sourceCount) data.norm
    (fun source => TargetPolynomial.power ops.toOps gamma source.val) alpha
  have carriedExact := weightedSum_indexedTables ops laws
    (canonicalCarriedCoordinates shape) data.carriedImage
    (fun coordinate =>
      TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
    data.priorPoint
  change ops.add
      (sumMap ops (BooleanVertex.all shape.cubeVariables) fun vertex =>
        ops.mul (vertex.equalityWeight ops alpha)
          (ops.add
            (ccsAt ops data gamma vertex)
            (gammaTerm ops gamma shape.freshCount
              (normAt ops data gamma vertex))))
      (sumMap ops (BooleanVertex.all shape.cubeVariables) fun vertex =>
        gammaTerm ops gamma shape.carriedEvaluationOffset
          (carriedAt ops data gamma vertex)) = _
  rw [show
    sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
      ops.mul (vertex.equalityWeight ops alpha)
        (ops.add
          (ccsAt ops data gamma vertex)
          (gammaTerm ops gamma shape.freshCount
            (normAt ops data gamma vertex)))) =
      ops.add
        (sumMap ops (BooleanVertex.all shape.cubeVariables) fun vertex =>
          ops.mul (vertex.equalityWeight ops alpha)
            (ccsAt ops data gamma vertex))
        (sumMap ops (BooleanVertex.all shape.cubeVariables) fun vertex =>
          ops.mul (vertex.equalityWeight ops alpha)
            (gammaTerm ops gamma shape.freshCount
              (normAt ops data gamma vertex))) by
      rw [← sumMap_add ops laws]
      apply sumMap_congr
      intro vertex _
      exact laws.left_distrib _ _ _]
  rw [show
    sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
      ops.mul (vertex.equalityWeight ops alpha)
        (ccsAt ops data gamma vertex)) =
      ccsResidualBlock ops data alpha gamma by
      exact ccsExact]
  rw [show
    sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
      ops.mul (vertex.equalityWeight ops alpha)
        (gammaTerm ops gamma shape.freshCount
          (normAt ops data gamma vertex))) =
      normResidualBlock ops data alpha gamma by
      unfold normResidualBlock gammaTerm
      calc
        sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
            ops.mul (vertex.equalityWeight ops alpha)
              (ops.mul
                (TargetPolynomial.power ops.toOps gamma shape.freshCount)
                (normAt ops data gamma vertex))) =
          sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
            ops.mul
              (TargetPolynomial.power ops.toOps gamma shape.freshCount)
              (ops.mul (vertex.equalityWeight ops alpha)
                (normAt ops data gamma vertex))) := by
            apply sumMap_congr
            intro vertex _
            calc
              ops.mul (vertex.equalityWeight ops alpha)
                  (ops.mul
                    (TargetPolynomial.power ops.toOps gamma shape.freshCount)
                    (normAt ops data gamma vertex)) =
                ops.mul
                  (ops.mul (vertex.equalityWeight ops alpha)
                    (TargetPolynomial.power ops.toOps gamma shape.freshCount))
                  (normAt ops data gamma vertex) :=
                    (laws.mul_assoc _ _ _).symm
              _ = ops.mul
                  (ops.mul
                    (TargetPolynomial.power ops.toOps gamma shape.freshCount)
                    (vertex.equalityWeight ops alpha))
                  (normAt ops data gamma vertex) := by
                    rw [laws.mul_comm
                      (vertex.equalityWeight ops alpha)]
              _ = ops.mul
                  (TargetPolynomial.power ops.toOps gamma shape.freshCount)
                  (ops.mul (vertex.equalityWeight ops alpha)
                    (normAt ops data gamma vertex)) :=
                      laws.mul_assoc _ _ _
        _ = ops.mul
            (TargetPolynomial.power ops.toOps gamma shape.freshCount)
            (sumMap ops (BooleanVertex.all shape.cubeVariables) fun vertex =>
              ops.mul (vertex.equalityWeight ops alpha)
                (normAt ops data gamma vertex)) :=
          sumMap_mul_left ops laws _ _ _
        _ = _ := congrArg
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma shape.freshCount))
          normExact]
  rw [show
    sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
      gammaTerm ops gamma shape.carriedEvaluationOffset
        (carriedAt ops data gamma vertex)) =
      gammaTerm ops gamma shape.carriedEvaluationOffset
        (carriedEvaluationLocal ops data gamma) by
      unfold gammaTerm carriedAt carriedEvaluationLocal
      rw [sumMap_mul_left ops laws]
      exact congrArg
        (ops.mul
          (TargetPolynomial.power ops.toOps gamma shape.carriedEvaluationOffset))
        carriedExact]
  exact laws.add_assoc _ _ _

private def shiftLaws
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    TargetPolynomial.ShiftLaws ops.toOps where
  one_mul := laws.one_mul
  mul_assoc := laws.mul_assoc
  mul_zero := laws.mul_zero
  mul_add := laws.left_distrib

private def targetLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (gamma : Field) : Field :=
  sumMap ops (canonicalCarriedCoordinates shape) fun coordinate =>
    gammaTerm ops gamma coordinate.localGammaExponent
      (data.claimedCoefficient coordinate)

private theorem finiteSum_eq_foldr
    {Field : Type uField}
    (ops : InterpolationOps Field) : forall values : List Field,
    BooleanTable.finiteSum ops values = values.foldr ops.add ops.zero
  | [] => rfl
  | _ :: values => by
      simp only [BooleanTable.finiteSum, List.foldr]
      rw [finiteSum_eq_foldr ops values]

private theorem targetLocal_eq_evaluateLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : JointData Field shape)
    (gamma : Field) :
    targetLocal ops data gamma =
      TargetPolynomial.evaluateLocal ops.toOps data.targetCoefficients gamma := by
  rw [TargetPolynomial.evaluateLocal_eq_foldr]
  unfold targetLocal sumMap gammaTerm JointData.targetCoefficients
  rw [finiteSum_eq_foldr]
  rfl

private theorem targetAbsolute_eq_shift_mul_targetLocal
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : JointData Field shape)
    (gamma : Field) :
    targetAbsolute ops data gamma =
      gammaTerm ops gamma shape.carriedEvaluationOffset
        (targetLocal ops data gamma) := by
  unfold targetAbsolute gammaTerm
  rw [TargetPolynomial.evaluateShifted_eq_shift_mul_evaluateLocal
    ops.toOps (shiftLaws ops laws) data.targetCoefficients gamma]
  rw [targetLocal_eq_evaluateLocal]

private theorem carriedResidualLocal_eq_target_sub_evaluation
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : JointData Field shape)
    (gamma : Field) :
    carriedResidualLocal ops data gamma =
      ops.sub (targetLocal ops data gamma)
        (carriedEvaluationLocal ops data gamma) := by
  unfold carriedResidualLocal targetLocal carriedEvaluationLocal gammaTerm
  calc
    sumMap ops (canonicalCarriedCoordinates shape) (fun coordinate =>
        ops.mul
          (TargetPolynomial.power ops.toOps gamma
            coordinate.localGammaExponent)
          (ops.sub
            (data.claimedCoefficient coordinate)
            ((data.carriedImage coordinate).equalityWeightedSum
              ops data.priorPoint))) =
      sumMap ops (canonicalCarriedCoordinates shape) (fun coordinate =>
        ops.sub
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma
              coordinate.localGammaExponent)
            (data.claimedCoefficient coordinate))
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma
              coordinate.localGammaExponent)
            ((data.carriedImage coordinate).equalityWeightedSum
              ops data.priorPoint))) := by
        apply sumMap_congr
        intro coordinate _
        exact mul_sub ops laws _ _ _
    _ = _ := sumMap_sub ops laws _ _ _

/-- The shifted target minus the shifted carried `Eval` block is exactly the
positive claimed-minus-derived carried residual block. -/
theorem target_sub_carriedEvaluation_eq_carriedResidualBlock
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : JointData Field shape)
    (gamma : Field) :
    ops.sub
        (targetAbsolute ops data gamma)
        (gammaTerm ops gamma shape.carriedEvaluationOffset
          (carriedEvaluationLocal ops data gamma)) =
      carriedResidualBlock ops data gamma := by
  rw [targetAbsolute_eq_shift_mul_targetLocal ops laws]
  unfold carriedResidualBlock gammaTerm
  rw [carriedResidualLocal_eq_target_sub_evaluation ops laws]
  exact (mul_sub ops laws _ _ _).symm

private theorem sub_three_eq_signed
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (target ccs norm carried : Field) :
    ops.sub target (ops.add ccs (ops.add norm carried)) =
      ops.add (ops.neg ccs)
        (ops.add (ops.neg norm) (ops.sub target carried)) := by
  unfold InterpolationOps.sub
  rw [laws.neg_add ccs (ops.add norm carried), laws.neg_add norm carried]
  calc
    ops.add target
        (ops.add (ops.neg ccs)
          (ops.add (ops.neg norm) (ops.neg carried))) =
      ops.add (ops.add target (ops.neg ccs))
        (ops.add (ops.neg norm) (ops.neg carried)) :=
          (laws.add_assoc _ _ _).symm
    _ = ops.add (ops.add (ops.neg ccs) target)
        (ops.add (ops.neg norm) (ops.neg carried)) := by
          rw [laws.add_comm target (ops.neg ccs)]
    _ = ops.add (ops.neg ccs)
        (ops.add target
          (ops.add (ops.neg norm) (ops.neg carried))) :=
            laws.add_assoc _ _ _
    _ = ops.add (ops.neg ccs)
        (ops.add (ops.add target (ops.neg norm)) (ops.neg carried)) := by
          congr 1
          exact (laws.add_assoc _ _ _).symm
    _ = ops.add (ops.neg ccs)
        (ops.add (ops.add (ops.neg norm) target) (ops.neg carried)) := by
          rw [laws.add_comm target (ops.neg norm)]
    _ = ops.add (ops.neg ccs)
        (ops.add (ops.neg norm)
          (ops.add target (ops.neg carried))) := by
            congr 1
            exact laws.add_assoc _ _ _

/-- Exact finite signed joint identity under the coherent absolute target:

`T_abs(C) - sum_x Q(x,A,C) = -CCS(A,C) - Norm(A,C) + Carried(C)`.

The theorem holds for every dimension-checked `A` and every `C`. It does not
assume the identity, a supplied evaluator, or any implementation artifact. -/
theorem paperDifference_eq_signedResidualBlocks
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables)
    (gamma : Field) :
    paperDifference ops data alpha gamma =
      signedResidualBlocks ops data alpha gamma := by
  unfold paperDifference signedResidualBlocks
  rw [summedQ_eq_residualBlocks ops laws]
  rw [sub_three_eq_signed ops laws]
  rw [target_sub_carriedEvaluation_eq_carriedResidualBlock ops laws]

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedJointIdentity
