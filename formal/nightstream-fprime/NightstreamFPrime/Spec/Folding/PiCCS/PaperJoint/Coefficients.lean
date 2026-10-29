import NightstreamFPrime.Spec.SumCheck.Polynomial

/-! SuperNeo v1.2 Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

open NightstreamFPrime.Spec.SumCheck

universe uField uLeft uRight

structure Shape where
  cubeVariables : Nat
  freshCount : Nat
  runningCount : Nat
  matrixCount : Nat
  coefficientCount : Nat
deriving Repr, DecidableEq

namespace Shape

@[implicit_reducible]
def sourceCount (shape : Shape) : Nat :=
  shape.freshCount + shape.runningCount

def padEvaluationCount (shape : Shape) : Nat :=
  shape.runningCount * shape.coefficientCount

def matrixEvaluationCount (shape : Shape) : Nat :=
  shape.runningCount * shape.matrixCount * shape.coefficientCount

def matrixEvaluationOffset (shape : Shape) : Nat :=
  shape.padEvaluationCount

def constraintOffset (shape : Shape) : Nat :=
  shape.padEvaluationCount + shape.matrixEvaluationCount

def ccsOffset (shape : Shape) : Nat :=
  shape.constraintOffset

def normOffset (shape : Shape) : Nat :=
  shape.constraintOffset + shape.freshCount

def jointCoefficientCount (shape : Shape) : Nat :=
  shape.constraintOffset + shape.freshCount + shape.sourceCount

theorem constraintOffset_eq (shape : Shape) :
    shape.constraintOffset =
      shape.runningCount * shape.coefficientCount *
        (shape.matrixCount + 1) := by
  simp only [constraintOffset, padEvaluationCount, matrixEvaluationCount]
  calc
    shape.runningCount * shape.coefficientCount +
        shape.runningCount * shape.matrixCount * shape.coefficientCount =
      shape.runningCount * shape.coefficientCount +
        shape.runningCount * shape.coefficientCount * shape.matrixCount := by
      rw [Nat.mul_assoc shape.runningCount shape.matrixCount,
        Nat.mul_comm shape.matrixCount shape.coefficientCount,
        ← Nat.mul_assoc shape.runningCount shape.coefficientCount]
    _ = shape.runningCount * shape.coefficientCount *
        (shape.matrixCount + 1) := by
      rw [Nat.mul_add, Nat.mul_one]
      omega

end Shape

structure PadCoordinate (shape : Shape) where
  running : Fin shape.runningCount
  coefficient : Fin shape.coefficientCount
deriving Repr, DecidableEq

namespace PadCoordinate

def localGammaExponent {shape : Shape}
    (coordinate : PadCoordinate shape) : Nat :=
  coordinate.running.val + shape.runningCount * coordinate.coefficient.val

def gammaExponent {shape : Shape}
    (coordinate : PadCoordinate shape) : Nat :=
  coordinate.localGammaExponent

end PadCoordinate

structure MatrixCoordinate (shape : Shape) where
  running : Fin shape.runningCount
  matrix : Fin shape.matrixCount
  coefficient : Fin shape.coefficientCount
deriving Repr, DecidableEq

namespace MatrixCoordinate

def localGammaExponent {shape : Shape}
    (coordinate : MatrixCoordinate shape) : Nat :=
  coordinate.running.val + shape.runningCount * coordinate.matrix.val +
    shape.runningCount * shape.matrixCount * coordinate.coefficient.val

def gammaExponent {shape : Shape}
    (coordinate : MatrixCoordinate shape) : Nat :=
  shape.matrixEvaluationOffset + coordinate.localGammaExponent

end MatrixCoordinate

def canonicalFinIndices (count : Nat) : List (Fin count) :=
  List.ofFn id

theorem canonicalFinIndices_length (count : Nat) :
    (canonicalFinIndices count).length = count := by
  simp [canonicalFinIndices]

theorem canonicalFinIndices_getElem (count index : Nat)
    (bound : index < (canonicalFinIndices count).length) :
    (canonicalFinIndices count)[index] =
      ⟨index, by simpa [canonicalFinIndices] using bound⟩ := by
  simp [canonicalFinIndices]

theorem mem_canonicalFinIndices {count : Nat} (index : Fin count) :
    index ∈ canonicalFinIndices count := by
  simp [canonicalFinIndices]

theorem mem_take_canonicalFinIndices {count : Nat} (index : Fin count) {bound : Nat}
    (below : index.val < bound) : index ∈ (canonicalFinIndices count).take bound := by
  rw [List.mem_iff_getElem]
  exact ⟨index.val, by simp [canonicalFinIndices]; omega, by simp [canonicalFinIndices]⟩

theorem canonicalFinIndices_nodup (count : Nat) :
    (canonicalFinIndices count).Nodup := by
  induction count with
  | zero => simp [canonicalFinIndices]
  | succ count inductionHypothesis =>
      rw [canonicalFinIndices, List.ofFn_succ]
      apply List.nodup_cons.mpr
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index equal
      · have mapped :
            ((canonicalFinIndices count).map Fin.succ).Nodup := by
          exact inductionHypothesis.map Fin.succ (by
            intro left right different
            intro equal
            exact different (Fin.succ_inj.mp equal))
        simpa [canonicalFinIndices, Function.comp_def] using mapped

theorem canonicalFinIndices_values (count : Nat) :
    (canonicalFinIndices count).map (fun index => index.val) =
      List.range count := by
  apply List.ext_get
  · simp [canonicalFinIndices]
  · intro index leftLt rightLt
    simp [canonicalFinIndices]

private theorem canonical_length_flatMap_constant
    {Index : Type uLeft}
    {Value : Type uRight}
    (indices : List Index)
    (values : Index -> List Value)
    (count : Nat)
    (eachLength : forall index, (values index).length = count) :
    (indices.flatMap values).length = indices.length * count := by
  induction indices with
  | nil => simp
  | cons index indices inductionHypothesis =>
      simp [eachLength, inductionHypothesis, Nat.add_mul, Nat.add_comm]

def canonicalPadCoordinates (shape : Shape) : List (PadCoordinate shape) :=
  (canonicalFinIndices shape.coefficientCount).flatMap fun coefficient =>
    (canonicalFinIndices shape.runningCount).map fun running =>
      { running := running
        coefficient := coefficient }

def canonicalMatrixCoordinates (shape : Shape) :
    List (MatrixCoordinate shape) :=
  (canonicalFinIndices shape.coefficientCount).flatMap fun coefficient =>
    (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
      (canonicalFinIndices shape.runningCount).map fun running =>
        { running := running
          matrix := matrix
          coefficient := coefficient }

theorem canonicalPadCoordinates_length (shape : Shape) :
    (canonicalPadCoordinates shape).length = shape.padEvaluationCount := by
  unfold canonicalPadCoordinates
  calc
    _ = (canonicalFinIndices shape.coefficientCount).length *
        shape.runningCount := by
      apply canonical_length_flatMap_constant
      intro coefficient
      simp [canonicalFinIndices]
    _ = shape.padEvaluationCount := by
      simp [canonicalFinIndices, Shape.padEvaluationCount, Nat.mul_comm]

theorem canonicalMatrixCoordinates_length (shape : Shape) :
    (canonicalMatrixCoordinates shape).length =
      shape.matrixEvaluationCount := by
  unfold canonicalMatrixCoordinates
  calc
    _ = (canonicalFinIndices shape.coefficientCount).length *
        (shape.matrixCount * shape.runningCount) := by
      apply canonical_length_flatMap_constant
      intro coefficient
      calc
        _ = (canonicalFinIndices shape.matrixCount).length *
            shape.runningCount := by
          apply canonical_length_flatMap_constant
          intro matrix
          simp [canonicalFinIndices]
        _ = shape.matrixCount * shape.runningCount := by
          simp [canonicalFinIndices]
    _ = shape.matrixEvaluationCount := by
      simp [canonicalFinIndices, Shape.matrixEvaluationCount,
        Nat.mul_comm, Nat.mul_left_comm]

private theorem flatMap_congr_local
    {Index : Type uLeft}
    {Value : Type uRight}
    {indices : List Index}
    {left right : Index -> List Value}
    (equal : forall index, index ∈ indices -> left index = right index) :
    indices.flatMap left = indices.flatMap right := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.flatMap_cons]
      rw [equal index (by simp)]
      congr 1
      exact inductionHypothesis fun prior member =>
        equal prior (by simp [member])

private theorem range_flatMap_chunks
    (start chunk blockCount : Nat) :
    (List.range blockCount).flatMap (fun block =>
        List.range' (start + chunk * block) chunk) =
      List.range' start (blockCount * chunk) := by
  induction blockCount with
  | zero => simp
  | succ blockCount inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
      rw [inductionHypothesis]
      have blockStart :
          start + chunk * blockCount = start + blockCount * chunk := by
        rw [Nat.mul_comm chunk blockCount]
      have totalWidth :
          (blockCount + 1) * chunk = blockCount * chunk + chunk := by
        rw [Nat.add_mul]
        simp
      rw [blockStart, totalWidth]
      exact List.range'_append_1

private theorem runningExponentBlock
    (shape : Shape)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount) :
    (canonicalFinIndices shape.runningCount).map (fun running =>
        running.val + shape.runningCount * matrix.val +
          shape.runningCount * shape.matrixCount * coefficient.val) =
      List.range'
        (shape.runningCount * matrix.val +
          shape.runningCount * shape.matrixCount * coefficient.val)
        shape.runningCount := by
  rw [List.range'_eq_map_range, ← canonicalFinIndices_values]
  rw [List.map_map]
  apply List.map_congr_left
  intro running _
  simp only [Function.comp_apply]
  omega

private theorem padRunningExponentBlock
    (shape : Shape)
    (coefficient : Fin shape.coefficientCount) :
    (canonicalFinIndices shape.runningCount).map (fun running =>
        running.val + shape.runningCount * coefficient.val) =
      List.range' (shape.runningCount * coefficient.val)
        shape.runningCount := by
  rw [List.range'_eq_map_range, ← canonicalFinIndices_values]
  rw [List.map_map]
  apply List.map_congr_left
  intro running _
  simp only [Function.comp_apply]
  omega

private theorem matrixExponentBlock
    (shape : Shape)
    (coefficient : Fin shape.coefficientCount) :
    (canonicalFinIndices shape.matrixCount).flatMap (fun matrix =>
        (canonicalFinIndices shape.runningCount).map (fun running =>
          running.val + shape.runningCount * matrix.val +
            shape.runningCount * shape.matrixCount * coefficient.val)) =
      List.range'
        (shape.runningCount * shape.matrixCount * coefficient.val)
        (shape.runningCount * shape.matrixCount) := by
  calc
    _ = (canonicalFinIndices shape.matrixCount).flatMap (fun matrix =>
        List.range'
          (shape.runningCount * matrix.val +
            shape.runningCount * shape.matrixCount * coefficient.val)
          shape.runningCount) := by
      apply flatMap_congr_local
      intro matrix _
      exact runningExponentBlock shape matrix coefficient
    _ = (canonicalFinIndices shape.matrixCount).flatMap (fun matrix =>
        List.range'
          (shape.runningCount * shape.matrixCount * coefficient.val +
            shape.runningCount * matrix.val)
          shape.runningCount) := by
      apply flatMap_congr_local
      intro matrix _
      exact congrArg (fun start => List.range' start shape.runningCount)
        (Nat.add_comm
          (shape.runningCount * matrix.val)
          (shape.runningCount * shape.matrixCount * coefficient.val))
    _ = ((canonicalFinIndices shape.matrixCount).map (fun matrix =>
          matrix.val)).flatMap (fun matrix =>
        List.range'
          (shape.runningCount * shape.matrixCount * coefficient.val +
            shape.runningCount * matrix)
          shape.runningCount) := by
      exact (List.flatMap_map
        (fun matrix : Fin shape.matrixCount => matrix.val)
        (fun matrix =>
          List.range'
            (shape.runningCount * shape.matrixCount * coefficient.val +
              shape.runningCount * matrix)
            shape.runningCount)
        (canonicalFinIndices shape.matrixCount)).symm
    _ = (List.range shape.matrixCount).flatMap (fun matrix =>
        List.range'
          (shape.runningCount * shape.matrixCount * coefficient.val +
            shape.runningCount * matrix)
          shape.runningCount) := by
      rw [canonicalFinIndices_values]
    _ = List.range'
        (shape.runningCount * shape.matrixCount * coefficient.val)
        (shape.matrixCount * shape.runningCount) :=
      range_flatMap_chunks
        (shape.runningCount * shape.matrixCount * coefficient.val)
        shape.runningCount shape.matrixCount
    _ = List.range'
        (shape.runningCount * shape.matrixCount * coefficient.val)
        (shape.runningCount * shape.matrixCount) := by
      rw [Nat.mul_comm shape.matrixCount shape.runningCount]

theorem canonicalPadCoordinates_localGammaExponents (shape : Shape) :
    (canonicalPadCoordinates shape).map PadCoordinate.localGammaExponent =
      List.range shape.padEvaluationCount := by
  unfold canonicalPadCoordinates PadCoordinate.localGammaExponent
  simp only [List.map_flatMap, List.map_map]
  calc
    _ = (canonicalFinIndices shape.coefficientCount).flatMap
        (fun coefficient =>
          List.range' (shape.runningCount * coefficient.val)
            shape.runningCount) := by
      apply flatMap_congr_local
      intro coefficient _
      exact padRunningExponentBlock shape coefficient
    _ = ((canonicalFinIndices shape.coefficientCount).map (fun coefficient =>
          coefficient.val)).flatMap (fun coefficient =>
        List.range' (0 + shape.runningCount * coefficient)
          shape.runningCount) := by
      simpa only [Nat.zero_add] using (List.flatMap_map
        (fun coefficient : Fin shape.coefficientCount => coefficient.val)
        (fun coefficient =>
          List.range' (0 + shape.runningCount * coefficient)
            shape.runningCount)
        (canonicalFinIndices shape.coefficientCount)).symm
    _ = (List.range shape.coefficientCount).flatMap (fun coefficient =>
        List.range' (0 + shape.runningCount * coefficient)
          shape.runningCount) := by
      rw [canonicalFinIndices_values]
    _ = List.range' 0
        (shape.coefficientCount * shape.runningCount) :=
      range_flatMap_chunks 0 shape.runningCount shape.coefficientCount
    _ = List.range shape.padEvaluationCount := by
      rw [List.range_eq_range']
      apply congrArg (fun count => List.range' 0 count)
      unfold Shape.padEvaluationCount
      exact Nat.mul_comm shape.coefficientCount shape.runningCount

theorem canonicalMatrixCoordinates_localGammaExponents (shape : Shape) :
    (canonicalMatrixCoordinates shape).map
        MatrixCoordinate.localGammaExponent =
      List.range shape.matrixEvaluationCount := by
  unfold canonicalMatrixCoordinates MatrixCoordinate.localGammaExponent
  simp only [List.map_flatMap, List.map_map]
  calc
    _ = (canonicalFinIndices shape.coefficientCount).flatMap
        (fun coefficient =>
          List.range'
            (shape.runningCount * shape.matrixCount * coefficient.val)
            (shape.runningCount * shape.matrixCount)) := by
      apply flatMap_congr_local
      intro coefficient _
      exact matrixExponentBlock shape coefficient
    _ = ((canonicalFinIndices shape.coefficientCount).map (fun coefficient =>
          coefficient.val)).flatMap (fun coefficient =>
        List.range'
          (0 + shape.runningCount * shape.matrixCount * coefficient)
          (shape.runningCount * shape.matrixCount)) := by
      simpa only [Nat.zero_add] using (List.flatMap_map
        (fun coefficient : Fin shape.coefficientCount => coefficient.val)
        (fun coefficient =>
          List.range'
            (0 + shape.runningCount * shape.matrixCount * coefficient)
            (shape.runningCount * shape.matrixCount))
        (canonicalFinIndices shape.coefficientCount)).symm
    _ = (List.range shape.coefficientCount).flatMap (fun coefficient =>
        List.range'
          (0 + shape.runningCount * shape.matrixCount * coefficient)
          (shape.runningCount * shape.matrixCount)) := by
      rw [canonicalFinIndices_values]
    _ = List.range' 0
        (shape.coefficientCount *
          (shape.runningCount * shape.matrixCount)) :=
      range_flatMap_chunks 0
        (shape.runningCount * shape.matrixCount) shape.coefficientCount
    _ = List.range shape.matrixEvaluationCount := by
      rw [List.range_eq_range']
      apply congrArg (fun count => List.range' 0 count)
      unfold Shape.matrixEvaluationCount
      exact Nat.mul_comm
        shape.coefficientCount
        (shape.runningCount * shape.matrixCount)

structure AlphaMonomial (shape : Shape) where
  exponents : List Nat
  arity : exponents.length = shape.cubeVariables
deriving Repr, DecidableEq

/-- SuperNeo v1.2 Appendix B.2. -/
structure AlphaBasis (shape : Shape) where
  monomials : List (AlphaMonomial shape)
  nodup : monomials.Nodup

structure AlphaPolynomial
    (Field : Type uField)
    {shape : Shape}
    (basis : AlphaBasis shape) where
  coefficients : List Field
  coefficientCount : coefficients.length = basis.monomials.length

namespace AlphaPolynomial

def CoefficientZero
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (polynomial : AlphaPolynomial Field basis) : Prop :=
  forall coefficient, coefficient ∈ polynomial.coefficients -> coefficient = ops.zero

end AlphaPolynomial

structure Residuals
    (Field : Type uField)
    (shape : Shape)
    (basis : AlphaBasis shape) where
  padEvaluation : List Field
  padEvaluationCount :
    padEvaluation.length = shape.padEvaluationCount
  matrixEvaluation : List Field
  matrixEvaluationCount :
    matrixEvaluation.length = shape.matrixEvaluationCount
  ccs : List (AlphaPolynomial Field basis)
  ccsCount : ccs.length = shape.freshCount
  norm : List (AlphaPolynomial Field basis)
  normCount : norm.length = shape.sourceCount

structure Obligations (shape : Shape) where
  padEvaluation : List Prop
  padEvaluationCount :
    padEvaluation.length = shape.padEvaluationCount
  matrixEvaluation : List Prop
  matrixEvaluationCount :
    matrixEvaluation.length = shape.matrixEvaluationCount
  ccs : List Prop
  ccsCount : ccs.length = shape.freshCount
  norm : List Prop
  normCount : norm.length = shape.sourceCount

namespace Obligations

def AllHold {shape : Shape} (obligations : Obligations shape) : Prop :=
  (forall obligation, obligation ∈ obligations.padEvaluation -> obligation) ∧
  (forall obligation, obligation ∈ obligations.matrixEvaluation -> obligation) ∧
  (forall obligation, obligation ∈ obligations.ccs -> obligation) ∧
  forall obligation, obligation ∈ obligations.norm -> obligation

end Obligations

inductive Aligned {Left : Type uLeft} {Right : Type uRight}
    (relation : Left -> Right -> Prop) : List Left -> List Right -> Prop where
  | nil : Aligned relation [] []
  | cons (left : Left) (right : Right) (lefts : List Left)
      (rights : List Right) :
      relation left right -> Aligned relation lefts rights ->
        Aligned relation (left :: lefts) (right :: rights)

structure ResidualizationBoundary
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis)
    (obligations : Obligations shape) : Prop where
  padEvaluation : Aligned
    (fun (residual : Field) (obligation : Prop) =>
      residual = ops.zero ↔ obligation)
    residuals.padEvaluation obligations.padEvaluation
  matrixEvaluation : Aligned
    (fun (residual : Field) (obligation : Prop) =>
      residual = ops.zero ↔ obligation)
    residuals.matrixEvaluation obligations.matrixEvaluation
  ccs : Aligned
    (fun (residual : AlphaPolynomial Field basis) (obligation : Prop) =>
      residual.CoefficientZero ops ↔ obligation)
    residuals.ccs obligations.ccs
  norm : Aligned
    (fun (residual : AlphaPolynomial Field basis) (obligation : Prop) =>
      residual.CoefficientZero ops ↔ obligation)
    residuals.norm obligations.norm

inductive JointCoefficient
    (Field : Type uField)
    {shape : Shape}
    (basis : AlphaBasis shape) where
  | alpha (polynomial : AlphaPolynomial Field basis)
  | scalar (value : Field)

namespace JointCoefficient

def Zero
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field) :
    JointCoefficient Field basis -> Prop
  | .alpha polynomial => polynomial.CoefficientZero ops
  | .scalar value => value = ops.zero

end JointCoefficient

namespace Residuals

def jointCoefficients
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (residuals : Residuals Field shape basis) :
    List (JointCoefficient Field basis) :=
  residuals.padEvaluation.map JointCoefficient.scalar ++
    residuals.matrixEvaluation.map JointCoefficient.scalar ++
    residuals.ccs.map JointCoefficient.alpha ++
    residuals.norm.map JointCoefficient.alpha ++
    []

def CoefficientTruth
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis) : Prop :=
  forall coefficient,
    coefficient ∈ residuals.jointCoefficients -> coefficient.Zero ops

theorem jointCoefficients_length
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (residuals : Residuals Field shape basis) :
    residuals.jointCoefficients.length = shape.jointCoefficientCount := by
  simp [jointCoefficients, Shape.jointCoefficientCount,
    Shape.constraintOffset, residuals.padEvaluationCount,
    residuals.matrixEvaluationCount, residuals.ccsCount,
    residuals.normCount, Nat.add_assoc]

theorem coefficientTruth_iff_residualFamilies
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis) :
    CoefficientTruth ops residuals ↔
      (forall residual, residual ∈ residuals.padEvaluation ->
        residual = ops.zero) ∧
      (forall residual, residual ∈ residuals.matrixEvaluation ->
        residual = ops.zero) ∧
      (forall residual, residual ∈ residuals.ccs ->
        residual.CoefficientZero ops) ∧
      forall residual, residual ∈ residuals.norm ->
        residual.CoefficientZero ops := by
  constructor
  · intro truth
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro residual member
      exact truth (.scalar residual) (by
        simp [jointCoefficients, member])
    · intro residual member
      exact truth (.scalar residual) (by
        simp [jointCoefficients, member])
    · intro residual member
      exact truth (.alpha residual) (by
        simp [jointCoefficients, member])
    · intro residual member
      exact truth (.alpha residual) (by
        simp [jointCoefficients, member])
  · rintro ⟨padTruth, matrixTruth, ccsTruth, normTruth⟩ coefficient member
    simp only [jointCoefficients, List.mem_append, List.mem_map] at member
    rcases member with
      (((⟨residual, residualMember, rfl⟩ |
        ⟨residual, residualMember, rfl⟩) |
        ⟨residual, residualMember, rfl⟩) |
        ⟨residual, residualMember, rfl⟩) | member
    · exact padTruth residual residualMember
    · exact matrixTruth residual residualMember
    · exact ccsTruth residual residualMember
    · exact normTruth residual residualMember
    · simp at member

private theorem forall₂_iff_all
    {Left : Type uLeft}
    {Right : Type uRight}
    {left : List Left}
    {right : List Right}
    {leftHolds : Left -> Prop}
    {rightHolds : Right -> Prop}
    (exact : Aligned
      (fun leftValue rightValue =>
        leftHolds leftValue ↔ rightHolds rightValue)
      left right) :
    (forall leftValue, leftValue ∈ left -> leftHolds leftValue) ↔
      forall rightValue, rightValue ∈ right -> rightHolds rightValue := by
  induction exact with
  | nil => simp
  | cons left right lefts rights headExact tailExact inductionHypothesis =>
      simp only [List.mem_cons, forall_eq_or_imp]
      exact and_congr headExact inductionHypothesis

theorem coefficientTruth_iff_allObligations
    {Field : Type uField}
    {shape : Shape}
    {basis : AlphaBasis shape}
    (ops : SumCheck.Finite.Ops Field)
    (residuals : Residuals Field shape basis)
    (obligations : Obligations shape)
    (boundary : ResidualizationBoundary ops residuals obligations) :
    residuals.CoefficientTruth ops ↔ obligations.AllHold := by
  rw [coefficientTruth_iff_residualFamilies]
  unfold Obligations.AllHold
  exact and_congr
    (forall₂_iff_all
      (Left := Field)
      (Right := Prop)
      (left := residuals.padEvaluation)
      (right := obligations.padEvaluation)
      (leftHolds := fun residual : Field => residual = ops.zero)
      (rightHolds := fun obligation : Prop => obligation)
      boundary.padEvaluation)
    (and_congr
      (forall₂_iff_all
        (Left := Field)
        (Right := Prop)
        (left := residuals.matrixEvaluation)
        (right := obligations.matrixEvaluation)
        (leftHolds := fun residual : Field => residual = ops.zero)
        (rightHolds := fun obligation : Prop => obligation)
        boundary.matrixEvaluation)
      (and_congr
        (forall₂_iff_all
          (Left := AlphaPolynomial Field basis)
          (Right := Prop)
          (left := residuals.ccs)
          (right := obligations.ccs)
          (leftHolds := fun residual : AlphaPolynomial Field basis =>
            residual.CoefficientZero ops)
          (rightHolds := fun obligation : Prop => obligation)
          boundary.ccs)
        (forall₂_iff_all
          (Left := AlphaPolynomial Field basis)
          (Right := Prop)
          (left := residuals.norm)
          (right := obligations.norm)
          (leftHolds := fun residual : AlphaPolynomial Field basis =>
            residual.CoefficientZero ops)
          (rightHolds := fun obligation : Prop => obligation)
          boundary.norm)))

end Residuals

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
