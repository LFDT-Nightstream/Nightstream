import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanHypercubeSum
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanReproduction
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetPolynomial

/-!
Exact finite SuperNeo v1.1 joint identity from Section 7.3 and Appendix B.2.
`Pad` and the 14 CCS matrices are separate evaluation families. This file
owns the pointwise four-term `Q`, its claimed target, the four residual
families, and their signed identity. It emits no constraints.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedJointIdentity

universe uField uIndex

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open FiniteSumAlgebra

structure JointData (Field : Type uField) (shape : Shape) where
  ccs : Fin shape.freshCount -> BooleanTable Field shape.cubeVariables
  norm : Fin shape.sourceCount -> BooleanTable Field shape.cubeVariables
  priorPoint : CubePoint Field shape.cubeVariables
  padImage : PadCoordinate shape -> BooleanTable Field shape.cubeVariables
  matrixImage : MatrixCoordinate shape -> BooleanTable Field shape.cubeVariables
  claimedPadCoefficient : PadCoordinate shape -> Field
  claimedMatrixCoefficient : MatrixCoordinate shape -> Field

namespace JointData

@[ext] theorem ext
    {Field : Type uField}
    {shape : Shape}
    (left right : JointData Field shape)
    (ccs : left.ccs = right.ccs)
    (norm : left.norm = right.norm)
    (priorPoint : left.priorPoint = right.priorPoint)
    (padImage : left.padImage = right.padImage)
    (matrixImage : left.matrixImage = right.matrixImage)
    (claimedPadCoefficient :
      left.claimedPadCoefficient = right.claimedPadCoefficient)
    (claimedMatrixCoefficient :
      left.claimedMatrixCoefficient = right.claimedMatrixCoefficient) :
    left = right := by
  cases left
  cases right
  simp_all

def targetCoefficients
    {Field : Type uField}
    {shape : Shape}
    (data : JointData Field shape) :
    TargetPolynomial.TargetCoefficients Field shape where
  pad := data.claimedPadCoefficient
  matrix := data.claimedMatrixCoefficient

end JointData

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

def gammaTerm
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (gamma : Field)
    (exponent : Nat)
    (value : Field) : Field :=
  ops.mul (TargetPolynomial.power ops.toOps gamma exponent) value

private theorem sumMap_gammaTerm
    {Field : Type uField}
    {Index : Type uIndex}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (gamma : Field)
    (exponent : Nat)
    (indices : List Index)
    (value : Index -> Field) :
    sumMap ops indices (fun index => gammaTerm ops gamma exponent (value index)) =
      gammaTerm ops gamma exponent (sumMap ops indices value) := by
  unfold gammaTerm
  exact sumMap_mul_left ops laws _ _ _

def ccsAt
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) (vertex : BooleanVertex shape.cubeVariables) : Field :=
  sumMap ops (canonicalFinIndices shape.freshCount) fun source =>
    gammaTerm ops gamma source.val ((data.ccs source).valueAt vertex)

def normAt
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) (vertex : BooleanVertex shape.cubeVariables) : Field :=
  sumMap ops (canonicalFinIndices shape.sourceCount) fun source =>
    gammaTerm ops gamma source.val ((data.norm source).valueAt vertex)

def padAt
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) (vertex : BooleanVertex shape.cubeVariables) : Field :=
  ops.mul (vertex.equalityWeight ops data.priorPoint) <|
    sumMap ops (canonicalPadCoordinates shape) fun coordinate =>
      gammaTerm ops gamma coordinate.localGammaExponent
        ((data.padImage coordinate).valueAt vertex)

def matrixAt
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) (vertex : BooleanVertex shape.cubeVariables) : Field :=
  ops.mul (vertex.equalityWeight ops data.priorPoint) <|
    sumMap ops (canonicalMatrixCoordinates shape) fun coordinate =>
      gammaTerm ops gamma coordinate.localGammaExponent
        ((data.matrixImage coordinate).valueAt vertex)

def constraintAt
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) : Field :=
  ops.mul (vertex.equalityWeight ops alpha)
    (ops.add (ccsAt ops data gamma vertex)
      (gammaTerm ops gamma shape.freshCount (normAt ops data gamma vertex)))

/-- Exact v1.1 four-term pointwise polynomial. -/
def qAt
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field)
    (vertex : BooleanVertex shape.cubeVariables) : Field :=
  ops.add (padAt ops data gamma vertex)
    (ops.add
      (gammaTerm ops gamma shape.matrixEvaluationOffset
        (matrixAt ops data gamma vertex))
      (gammaTerm ops gamma shape.constraintOffset
        (constraintAt ops data alpha gamma vertex)))

def summedQ
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) : Field :=
  sumMap ops (BooleanVertex.all shape.cubeVariables) fun vertex =>
    qAt ops data alpha gamma vertex

def targetAbsolute
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) : Field :=
  TargetPolynomial.evaluate ops.toOps data.targetCoefficients gamma

def paperDifference
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) : Field :=
  ops.sub (targetAbsolute ops data gamma) (summedQ ops data alpha gamma)

private theorem weightedSum_indexedTables
    {Field : Type uField} {Index : Type uIndex}
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

def ccsResidualBlock
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) : Field :=
  sumMap ops (canonicalFinIndices shape.freshCount) fun source =>
    gammaTerm ops gamma source.val
      ((data.ccs source).equalityWeightedSum ops alpha)

def normResidualLocal
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) : Field :=
  sumMap ops (canonicalFinIndices shape.sourceCount) fun source =>
    gammaTerm ops gamma source.val
      ((data.norm source).equalityWeightedSum ops alpha)

def normResidualBlock
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) : Field :=
  gammaTerm ops gamma shape.freshCount
    (normResidualLocal ops data alpha gamma)

def padEvaluationLocal
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) : Field :=
  sumMap ops (canonicalPadCoordinates shape) fun coordinate =>
    gammaTerm ops gamma coordinate.localGammaExponent
      ((data.padImage coordinate).equalityWeightedSum ops data.priorPoint)

def matrixEvaluationLocal
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) : Field :=
  sumMap ops (canonicalMatrixCoordinates shape) fun coordinate =>
    gammaTerm ops gamma coordinate.localGammaExponent
      ((data.matrixImage coordinate).equalityWeightedSum ops data.priorPoint)

def matrixEvaluationBlock
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) : Field :=
  gammaTerm ops gamma shape.matrixEvaluationOffset
    (matrixEvaluationLocal ops data gamma)

def constraintResidualBlock
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) : Field :=
  gammaTerm ops gamma shape.constraintOffset
    (ops.add (ccsResidualBlock ops data alpha gamma)
      (normResidualBlock ops data alpha gamma))

def padResidualLocal
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) : Field :=
  sumMap ops (canonicalPadCoordinates shape) fun coordinate =>
    gammaTerm ops gamma coordinate.localGammaExponent <|
      ops.sub (data.claimedPadCoefficient coordinate)
        ((data.padImage coordinate).equalityWeightedSum ops data.priorPoint)

def matrixResidualLocal
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) : Field :=
  sumMap ops (canonicalMatrixCoordinates shape) fun coordinate =>
    gammaTerm ops gamma coordinate.localGammaExponent <|
      ops.sub (data.claimedMatrixCoefficient coordinate)
        ((data.matrixImage coordinate).equalityWeightedSum ops data.priorPoint)

def matrixResidualBlock
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) : Field :=
  gammaTerm ops gamma shape.matrixEvaluationOffset
    (matrixResidualLocal ops data gamma)

def signedResidualBlocks
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) : Field :=
  ops.add (padResidualLocal ops data gamma)
    (ops.add (matrixResidualBlock ops data gamma)
      (ops.neg (constraintResidualBlock ops data alpha gamma)))

private theorem summedConstraintAt_eq
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) :
    sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
      constraintAt ops data alpha gamma vertex) =
      ops.add (ccsResidualBlock ops data alpha gamma)
        (normResidualBlock ops data alpha gamma) := by
  have ccsExact := weightedSum_indexedTables ops laws
    (canonicalFinIndices shape.freshCount) data.ccs
    (fun source => TargetPolynomial.power ops.toOps gamma source.val) alpha
  have normExact := weightedSum_indexedTables ops laws
    (canonicalFinIndices shape.sourceCount) data.norm
    (fun source => TargetPolynomial.power ops.toOps gamma source.val) alpha
  unfold constraintAt
  rw [show
    sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
      ops.mul (vertex.equalityWeight ops alpha)
        (ops.add (ccsAt ops data gamma vertex)
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
      ccsResidualBlock ops data alpha gamma by exact ccsExact]
  rw [show
    sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
      ops.mul (vertex.equalityWeight ops alpha)
        (gammaTerm ops gamma shape.freshCount
          (normAt ops data gamma vertex))) =
      normResidualBlock ops data alpha gamma by
      unfold normResidualBlock gammaTerm
      calc
        _ = sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
            ops.mul
              (TargetPolynomial.power ops.toOps gamma shape.freshCount)
              (ops.mul (vertex.equalityWeight ops alpha)
                (normAt ops data gamma vertex))) := by
          apply sumMap_congr
          intro vertex _
          calc
            _ = ops.mul
                (ops.mul (vertex.equalityWeight ops alpha)
                  (TargetPolynomial.power ops.toOps gamma shape.freshCount))
                (normAt ops data gamma vertex) := (laws.mul_assoc _ _ _).symm
            _ = ops.mul
                (ops.mul
                  (TargetPolynomial.power ops.toOps gamma shape.freshCount)
                  (vertex.equalityWeight ops alpha))
                (normAt ops data gamma vertex) := by
                  rw [laws.mul_comm (vertex.equalityWeight ops alpha)
                    (TargetPolynomial.power ops.toOps gamma shape.freshCount)]
            _ = _ := laws.mul_assoc _ _ _
        _ = ops.mul
            (TargetPolynomial.power ops.toOps gamma shape.freshCount)
            (sumMap ops (BooleanVertex.all shape.cubeVariables) fun vertex =>
              ops.mul (vertex.equalityWeight ops alpha)
                (normAt ops data gamma vertex)) := sumMap_mul_left ops laws _ _ _
        _ = _ := congrArg
          (ops.mul (TargetPolynomial.power ops.toOps gamma shape.freshCount))
          normExact]

private theorem summedQ_eq_blocks
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) :
    summedQ ops data alpha gamma =
      ops.add (padEvaluationLocal ops data gamma)
        (ops.add (matrixEvaluationBlock ops data gamma)
          (constraintResidualBlock ops data alpha gamma)) := by
  unfold summedQ qAt
  rw [sumMap_add ops laws, sumMap_add ops laws]
  have padExact := weightedSum_indexedTables ops laws
    (canonicalPadCoordinates shape) data.padImage
    (fun coordinate =>
      TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
    data.priorPoint
  have matrixExact := weightedSum_indexedTables ops laws
    (canonicalMatrixCoordinates shape) data.matrixImage
    (fun coordinate =>
      TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
    data.priorPoint
  rw [show
    sumMap ops (BooleanVertex.all shape.cubeVariables)
      (padAt ops data gamma) = padEvaluationLocal ops data gamma by
    exact padExact]
  rw [show
    sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
      gammaTerm ops gamma shape.matrixEvaluationOffset
        (matrixAt ops data gamma vertex)) =
      matrixEvaluationBlock ops data gamma by
    unfold matrixEvaluationBlock
    rw [sumMap_gammaTerm ops laws]
    exact congrArg (gammaTerm ops gamma shape.matrixEvaluationOffset)
      matrixExact]
  rw [show
    sumMap ops (BooleanVertex.all shape.cubeVariables) (fun vertex =>
      gammaTerm ops gamma shape.constraintOffset
        (constraintAt ops data alpha gamma vertex)) =
      constraintResidualBlock ops data alpha gamma by
    unfold constraintResidualBlock
    rw [sumMap_gammaTerm ops laws]
    exact congrArg (gammaTerm ops gamma shape.constraintOffset)
      (summedConstraintAt_eq ops laws data alpha gamma)]

private def shiftLaws
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) :
    TargetPolynomial.ShiftLaws ops.toOps where
  one_mul := laws.one_mul
  mul_assoc := laws.mul_assoc
  mul_zero := laws.mul_zero
  mul_add := laws.left_distrib

private def padTargetLocal
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) : Field :=
  sumMap ops (canonicalPadCoordinates shape) fun coordinate =>
    gammaTerm ops gamma coordinate.localGammaExponent
      (data.claimedPadCoefficient coordinate)

private def matrixTargetLocal
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) : Field :=
  sumMap ops (canonicalMatrixCoordinates shape) fun coordinate =>
    gammaTerm ops gamma coordinate.localGammaExponent
      (data.claimedMatrixCoefficient coordinate)

private theorem finiteSum_eq_foldr
    {Field : Type uField}
    (ops : InterpolationOps Field) : forall values : List Field,
    BooleanTable.finiteSum ops values = values.foldr ops.add ops.zero
  | [] => rfl
  | _ :: values => by
      simp only [BooleanTable.finiteSum, List.foldr]
      rw [finiteSum_eq_foldr ops values]

private theorem padTargetLocal_eq_evaluatePad
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) :
    padTargetLocal ops data gamma =
      TargetPolynomial.evaluatePad ops.toOps data.targetCoefficients gamma := by
  rw [TargetPolynomial.evaluatePad_eq_foldr]
  unfold padTargetLocal sumMap gammaTerm JointData.targetCoefficients
    TargetPolynomial.padTerm
  rw [finiteSum_eq_foldr]

private theorem matrixTargetLocal_eq_evaluateMatrixLocal
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field) (data : JointData Field shape)
    (gamma : Field) :
    matrixTargetLocal ops data gamma =
      TargetPolynomial.evaluateMatrixLocal ops.toOps data.targetCoefficients gamma := by
  rw [TargetPolynomial.evaluateMatrixLocal_eq_foldr]
  unfold matrixTargetLocal sumMap gammaTerm JointData.targetCoefficients
    TargetPolynomial.matrixLocalTerm
  rw [finiteSum_eq_foldr]

private theorem padResidualLocal_eq_target_sub_evaluation
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : JointData Field shape) (gamma : Field) :
    padResidualLocal ops data gamma =
      ops.sub (TargetPolynomial.evaluatePad ops.toOps data.targetCoefficients gamma)
        (padEvaluationLocal ops data gamma) := by
  rw [← padTargetLocal_eq_evaluatePad]
  unfold padResidualLocal padTargetLocal padEvaluationLocal gammaTerm
  calc
    _ = sumMap ops (canonicalPadCoordinates shape) (fun coordinate =>
        ops.sub
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
            (data.claimedPadCoefficient coordinate))
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
            ((data.padImage coordinate).equalityWeightedSum
              ops data.priorPoint))) := by
      apply sumMap_congr
      intro coordinate _
      exact mul_sub ops laws _ _ _
    _ = _ := sumMap_sub ops laws _ _ _

private theorem matrixResidualLocal_eq_target_sub_evaluation
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : JointData Field shape) (gamma : Field) :
    matrixResidualLocal ops data gamma =
      ops.sub
        (TargetPolynomial.evaluateMatrixLocal ops.toOps data.targetCoefficients gamma)
        (matrixEvaluationLocal ops data gamma) := by
  rw [← matrixTargetLocal_eq_evaluateMatrixLocal]
  unfold matrixResidualLocal matrixTargetLocal matrixEvaluationLocal gammaTerm
  calc
    _ = sumMap ops (canonicalMatrixCoordinates shape) (fun coordinate =>
        ops.sub
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
            (data.claimedMatrixCoefficient coordinate))
          (ops.mul
            (TargetPolynomial.power ops.toOps gamma coordinate.localGammaExponent)
            ((data.matrixImage coordinate).equalityWeightedSum
              ops data.priorPoint))) := by
      apply sumMap_congr
      intro coordinate _
      exact mul_sub ops laws _ _ _
    _ = _ := sumMap_sub ops laws _ _ _

private theorem matrixResidualBlock_eq_target_sub_evaluation
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : JointData Field shape) (gamma : Field) :
    matrixResidualBlock ops data gamma =
      ops.sub
        (TargetPolynomial.evaluateMatrix ops.toOps data.targetCoefficients gamma)
        (matrixEvaluationBlock ops data gamma) := by
  unfold matrixResidualBlock matrixEvaluationBlock gammaTerm
  rw [matrixResidualLocal_eq_target_sub_evaluation ops laws]
  rw [TargetPolynomial.evaluateMatrix_eq_shift_mul_evaluateMatrixLocal
    ops.toOps (shiftLaws ops laws)]
  exact mul_sub ops laws _ _ _

private theorem sub_targets_and_three
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (targetPad targetMatrix evalPad evalMatrix constraint : Field) :
    ops.sub (ops.add targetPad targetMatrix)
        (ops.add evalPad (ops.add evalMatrix constraint)) =
      ops.add (ops.sub targetPad evalPad)
        (ops.add (ops.sub targetMatrix evalMatrix) (ops.neg constraint)) := by
  unfold InterpolationOps.sub
  rw [laws.neg_add evalPad (ops.add evalMatrix constraint),
    laws.neg_add evalMatrix constraint]
  calc
    ops.add (ops.add targetPad targetMatrix)
        (ops.add (ops.neg evalPad)
          (ops.add (ops.neg evalMatrix) (ops.neg constraint))) =
      ops.add targetPad
        (ops.add targetMatrix
          (ops.add (ops.neg evalPad)
            (ops.add (ops.neg evalMatrix) (ops.neg constraint)))) :=
      laws.add_assoc _ _ _
    _ = ops.add targetPad
        (ops.add (ops.neg evalPad)
          (ops.add targetMatrix
            (ops.add (ops.neg evalMatrix) (ops.neg constraint)))) := by
      congr 1
      calc
        _ = ops.add (ops.add targetMatrix (ops.neg evalPad))
            (ops.add (ops.neg evalMatrix) (ops.neg constraint)) :=
          (laws.add_assoc _ _ _).symm
        _ = ops.add (ops.add (ops.neg evalPad) targetMatrix)
            (ops.add (ops.neg evalMatrix) (ops.neg constraint)) := by
          rw [laws.add_comm targetMatrix (ops.neg evalPad)]
        _ = _ := laws.add_assoc _ _ _
    _ = ops.add (ops.add targetPad (ops.neg evalPad))
        (ops.add targetMatrix
          (ops.add (ops.neg evalMatrix) (ops.neg constraint))) :=
      (laws.add_assoc _ _ _).symm
    _ = ops.add (ops.add targetPad (ops.neg evalPad))
        (ops.add (ops.add targetMatrix (ops.neg evalMatrix))
          (ops.neg constraint)) := by
      congr 1
      exact (laws.add_assoc _ _ _).symm

/-- Exact v1.1 finite joint identity, for every verifier challenge. -/
theorem paperDifference_eq_signedResidualBlocks
    {Field : Type uField} {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (data : JointData Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) :
    paperDifference ops data alpha gamma =
      signedResidualBlocks ops data alpha gamma := by
  unfold paperDifference targetAbsolute signedResidualBlocks
  rw [summedQ_eq_blocks ops laws]
  unfold TargetPolynomial.evaluate
  rw [sub_targets_and_three ops laws]
  rw [← padResidualLocal_eq_target_sub_evaluation ops laws]
  rw [← matrixResidualBlock_eq_target_sub_evaluation ops laws]

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedJointIdentity
