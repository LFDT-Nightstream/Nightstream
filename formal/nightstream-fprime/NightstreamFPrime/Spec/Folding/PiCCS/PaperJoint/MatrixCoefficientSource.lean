import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

/-! SuperNeo v1.2 Section 5; Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra
open UnifiedSources

universe uBase uExtension

def sumRange
    {Base : Type uBase}
    (ops : InterpolationOps Base) : Nat -> (Nat -> Base) -> Base
  | 0, _ => ops.zero
  | count + 1, term =>
      ops.add (sumRange ops count term) (term count)

theorem sumRange_congr
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    (count : Nat)
    (left right : Nat -> Base)
    (equal : forall index, index < count -> left index = right index) :
    sumRange ops count left = sumRange ops count right := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [sumRange, sumRange, inductionHypothesis]
      · rw [equal count (Nat.lt_succ_self count)]
      · intro index indexLt
        exact equal index (Nat.lt_trans indexLt (Nat.lt_succ_self count))

theorem sumRange_eq_zero
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    (laws : InterpolationEvaluationLaws ops)
    (count : Nat)
    (term : Nat -> Base)
    (zero : forall index, index < count -> term index = ops.zero) :
    sumRange ops count term = ops.zero := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [sumRange, inductionHypothesis]
      · rw [zero count (Nat.lt_succ_self count)]
        exact laws.zero_add ops.zero
      · intro index indexLt
        exact zero index (Nat.lt_trans indexLt (Nat.lt_succ_self count))

theorem sumRange_select
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    (laws : InterpolationEvaluationLaws ops)
    (count selected : Nat)
    (term : Nat -> Base)
    (selectedLt : selected < count) :
    sumRange ops count (fun index =>
      if index = selected then term index else ops.zero) =
      term selected := by
  induction count with
  | zero => omega
  | succ count inductionHypothesis =>
      rw [sumRange]
      by_cases last : selected = count
      · subst selected
        have prefixZero :
            sumRange ops count (fun index =>
              if index = count then term index else ops.zero) =
              ops.zero := by
          apply sumRange_eq_zero ops laws
          intro index indexLt
          rw [ite_eq_right (Nat.ne_of_lt indexLt)]
        rw [prefixZero, ite_eq_left rfl]
        exact laws.zero_add _
      · have selectedEarlier : selected < count := by omega
        rw [inductionHypothesis selectedEarlier, ite_eq_right]
        · exact laws.add_zero _
        · exact Ne.symm last

structure RingColumnLayout
    (coefficientCount blockCount columns : Nat) where
  decode : Fin columns -> Fin blockCount × Fin coefficientCount
  encode? : Fin blockCount -> Fin coefficientCount -> Option (Fin columns)
  decode_encode : forall block coefficient column,
    encode? block coefficient = some column ->
      decode column = (block, coefficient)
  encode_decode : forall column,
    encode? (decode column).1 (decode column).2 = some column

structure CoefficientKernel (Base : Type uBase) (coefficientCount : Nat) where
  constant : Fin coefficientCount
  weight : Fin coefficientCount -> Fin coefficientCount ->
    Fin coefficientCount -> Base

/-- SuperNeo v1.2 Theorems 8 and 9. -/
structure ConstantTermLaw
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    {coefficientCount : Nat}
    (kernel : CoefficientKernel Base coefficientCount) : Prop where
  weight : forall row assignment,
    kernel.weight kernel.constant row assignment =
      if row = assignment then ops.one else ops.zero

structure MatrixSource
    (Base : Type uBase)
    (shape : Shape)
    (columns blockCount : Nat) where
  columnLayout :
    RingColumnLayout shape.coefficientCount blockCount columns
  matrices : Fin shape.matrixCount ->
    BooleanMatrix Base shape.cubeVariables columns
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial Base shape.matrixCount
  kernel : CoefficientKernel Base shape.coefficientCount

namespace MatrixSource

def system
    {Base : Type uBase}
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount) :
    CCSResidualTable.Structure Base shape columns where
  matrices := source.matrices
  constraintPolynomial := source.constraintPolynomial

def paddedEntry
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount)
    (matrix : BooleanMatrix Base shape.cubeVariables columns)
    (vertex : BooleanVertex shape.cubeVariables)
    (block : Fin blockCount)
    (coefficient : Fin shape.coefficientCount) : Base :=
  match source.columnLayout.encode? block coefficient with
  | some column => matrix vertex column
  | none => ops.zero

def paddedMatrixEntry
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.cubeVariables)
    (block : Fin blockCount)
    (coefficient : Fin shape.coefficientCount) : Base :=
  source.paddedEntry ops (source.matrices matrix) vertex block coefficient

def coefficientMatrixOf
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount)
    (matrix : BooleanMatrix Base shape.cubeVariables columns)
    (output : Fin shape.coefficientCount) :
    BooleanMatrix Base shape.cubeVariables columns :=
  fun vertex column =>
    let packed := source.columnLayout.decode column
    sumRange ops shape.coefficientCount fun rowIndex =>
      if rowLt : rowIndex < shape.coefficientCount then
        let row : Fin shape.coefficientCount :=
          ⟨rowIndex, rowLt⟩
        ops.mul
          (source.paddedEntry ops matrix vertex packed.1 row)
          (source.kernel.weight output row packed.2)
      else
        ops.zero

def coefficientMatrix
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount)
    (matrix : Fin shape.matrixCount)
    (output : Fin shape.coefficientCount) :
    BooleanMatrix Base shape.cubeVariables columns :=
  source.coefficientMatrixOf ops (source.matrices matrix) output

def coefficientMatrices
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount) :
    Fin shape.matrixCount -> Fin shape.coefficientCount ->
      BooleanMatrix Base shape.cubeVariables columns :=
  fun matrix coefficient => source.coefficientMatrix ops matrix coefficient

theorem coefficientMatrixOf_constant_apply
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    (laws : InterpolationEvaluationLaws ops)
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount)
    (constantLaw : ConstantTermLaw ops source.kernel)
    (matrix : BooleanMatrix Base shape.cubeVariables columns)
    (vertex : BooleanVertex shape.cubeVariables)
    (column : Fin columns) :
    source.coefficientMatrixOf ops matrix source.kernel.constant vertex column =
      matrix vertex column := by
  let packed := source.columnLayout.decode column
  let block := packed.1
  let selected := packed.2
  change
    sumRange ops shape.coefficientCount (fun rowIndex =>
      if rowLt : rowIndex < shape.coefficientCount then
        let row : Fin shape.coefficientCount := ⟨rowIndex, rowLt⟩
        ops.mul
          (source.paddedEntry ops matrix vertex block row)
          (source.kernel.weight source.kernel.constant row selected)
      else
        ops.zero) =
      matrix vertex column
  calc
    sumRange ops shape.coefficientCount (fun rowIndex =>
        if rowLt : rowIndex < shape.coefficientCount then
          let row : Fin shape.coefficientCount := ⟨rowIndex, rowLt⟩
          ops.mul
            (source.paddedEntry ops matrix vertex block row)
            (source.kernel.weight source.kernel.constant row selected)
        else
          ops.zero) =
        sumRange ops shape.coefficientCount (fun rowIndex =>
          if rowIndex = selected.val then
            source.paddedEntry ops matrix vertex block selected
          else
            ops.zero) := by
      apply sumRange_congr
      intro rowIndex rowLt
      rw [dite_eq_left rowLt]
      let row : Fin shape.coefficientCount := ⟨rowIndex, rowLt⟩
      change
        ops.mul
            (source.paddedEntry ops matrix vertex block row)
            (source.kernel.weight source.kernel.constant row selected) =
          if rowIndex = selected.val then
            source.paddedEntry ops matrix vertex block selected
          else
            ops.zero
      rw [constantLaw.weight]
      by_cases equal : row = selected
      · have valueEqual : rowIndex = selected.val :=
          congrArg Fin.val equal
        rw [ite_eq_left equal, ite_eq_left valueEqual, laws.mul_one]
        exact congrArg
          (fun coefficient =>
            source.paddedEntry ops matrix vertex block coefficient)
          equal
      · have valueDifferent : rowIndex ≠ selected.val := by
          intro valueEqual
          apply equal
          exact Fin.eq_of_val_eq valueEqual
        rw [ite_eq_right equal, ite_eq_right valueDifferent, laws.mul_zero]
    _ = source.paddedEntry ops matrix vertex block selected := by
      exact sumRange_select ops laws shape.coefficientCount selected.val
        (fun _ => source.paddedEntry ops matrix vertex block selected)
        selected.isLt
    _ = matrix vertex column := by
      simp only [paddedEntry]
      rw [source.columnLayout.encode_decode column]

theorem coefficientMatrix_constant_apply
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    (laws : InterpolationEvaluationLaws ops)
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount)
    (constantLaw : ConstantTermLaw ops source.kernel)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.cubeVariables)
    (column : Fin columns) :
    source.coefficientMatrix ops matrix source.kernel.constant vertex column =
      source.matrices matrix vertex column := by
  exact coefficientMatrixOf_constant_apply ops laws source constantLaw
    (source.matrices matrix) vertex column

theorem coefficientMatrixOf_constant_eq
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    (laws : InterpolationEvaluationLaws ops)
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount)
    (constantLaw : ConstantTermLaw ops source.kernel)
    (matrix : BooleanMatrix Base shape.cubeVariables columns) :
    source.coefficientMatrixOf ops matrix source.kernel.constant = matrix := by
  funext vertex column
  exact coefficientMatrixOf_constant_apply ops laws source constantLaw
    matrix vertex column

theorem coefficientMatrix_constant_eq
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    (laws : InterpolationEvaluationLaws ops)
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount)
    (constantLaw : ConstantTermLaw ops source.kernel)
    (matrix : Fin shape.matrixCount) :
    source.coefficientMatrix ops matrix source.kernel.constant =
      source.matrices matrix := by
  funext vertex column
  exact coefficientMatrixOf_constant_apply ops laws source constantLaw
    (source.matrices matrix) vertex column

end MatrixSource

structure ConnectedInputs
    (Extension : Type uExtension)
    (shape : Shape)
    (columns blockCount : Nat) where
  cubeLayout : ColumnLayout shape.cubeVariables columns
  matrixSource : MatrixSource F shape columns blockCount
  assignments : Fin shape.sourceCount -> Assignment F columns
  priorPoint : CubePoint Extension shape.cubeVariables
  claimedPadCoefficient : PadCoordinate shape -> Extension
  claimedMatrixCoefficient : MatrixCoordinate shape -> Extension

namespace ConnectedInputs

def padMatrix
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (data : ConnectedInputs Extension shape columns blockCount) :
    BooleanMatrix F shape.cubeVariables columns :=
  fun vertex column =>
    data.cubeLayout.paddedIdentityEntry
      baseOps.zero baseOps.one vertex column

def padCoefficientMatrices
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (data : ConnectedInputs Extension shape columns blockCount) :
    Fin shape.coefficientCount ->
      BooleanMatrix F shape.cubeVariables columns :=
  fun coefficient => data.matrixSource.coefficientMatrixOf baseOps
    (data.padMatrix baseOps) coefficient

def toUnifiedInputs
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (data : ConnectedInputs Extension shape columns blockCount) :
    UnifiedInputs Extension shape columns where
  layout := data.cubeLayout
  system := data.matrixSource.system
  assignments := data.assignments
  padCoefficientMatrices := data.padCoefficientMatrices baseOps
  matrixCoefficientMatrices :=
    data.matrixSource.coefficientMatrices baseOps
  priorPoint := data.priorPoint
  claimedPadCoefficient := data.claimedPadCoefficient
  claimedMatrixCoefficient := data.claimedMatrixCoefficient

theorem toUnifiedInputs_system_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (data : ConnectedInputs Extension shape columns blockCount) :
    (data.toUnifiedInputs baseOps).system = data.matrixSource.system := by
  rfl

theorem toUnifiedInputs_padCoefficientMatrices_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (data : ConnectedInputs Extension shape columns blockCount) :
    (data.toUnifiedInputs baseOps).padCoefficientMatrices =
      data.padCoefficientMatrices baseOps := by
  rfl

theorem toUnifiedInputs_matrixCoefficientMatrices_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (data : ConnectedInputs Extension shape columns blockCount) :
    (data.toUnifiedInputs baseOps).matrixCoefficientMatrices =
      data.matrixSource.coefficientMatrices baseOps := by
  rfl

theorem padCoefficientMatrix_constant_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (data : ConnectedInputs Extension shape columns blockCount)
    (constantLaw : ConstantTermLaw baseOps data.matrixSource.kernel) :
    data.padCoefficientMatrices baseOps data.matrixSource.kernel.constant =
      data.padMatrix baseOps := by
  exact data.matrixSource.coefficientMatrixOf_constant_eq
    baseOps baseLaws constantLaw (data.padMatrix baseOps)

def SemanticTruth
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (data : ConnectedInputs Extension shape columns blockCount)
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension) : Prop :=
  (data.toUnifiedInputs baseOps).SemanticTruth baseOps extensionOps lift

theorem matrixImageConstantAt_eq_ccsImageAt
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (lift : F -> Extension)
    (data : ConnectedInputs Extension shape columns blockCount)
    (constantLaw : ConstantTermLaw baseOps data.matrixSource.kernel)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (vertex : BooleanVertex shape.cubeVariables) :
    let coordinate : MatrixCoordinate shape :=
      { running := running
        matrix := matrix
        coefficient := data.matrixSource.kernel.constant }
    MatrixEvaluationResidual.imageCoefficientAt baseOps lift
        (data.toUnifiedInputs baseOps).matrixData coordinate vertex =
      lift (matrixVectorAt baseOps (data.matrixSource.matrices matrix)
        (data.assignments (runningSourceIndex running)) vertex) := by
  dsimp only
  unfold MatrixEvaluationResidual.imageCoefficientAt
  change
    lift (matrixVectorAt baseOps
        (data.matrixSource.coefficientMatrix baseOps matrix
          data.matrixSource.kernel.constant)
        (data.assignments (runningSourceIndex running)) vertex) =
      lift (matrixVectorAt baseOps (data.matrixSource.matrices matrix)
        (data.assignments (runningSourceIndex running)) vertex)
  rw [data.matrixSource.coefficientMatrix_constant_eq baseOps baseLaws
    constantLaw matrix]

theorem padImageConstantAt_eq_padMatrixImageAt
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (lift : F -> Extension)
    (data : ConnectedInputs Extension shape columns blockCount)
    (constantLaw : ConstantTermLaw baseOps data.matrixSource.kernel)
    (running : Fin shape.runningCount)
    (vertex : BooleanVertex shape.cubeVariables) :
    let coordinate : PadCoordinate shape :=
      { running := running
        coefficient := data.matrixSource.kernel.constant }
    PadEvaluationResidual.imageCoefficientAt baseOps lift
        (data.toUnifiedInputs baseOps).padData coordinate vertex =
      lift (matrixVectorAt baseOps (data.padMatrix baseOps)
        (data.assignments (runningSourceIndex running)) vertex) := by
  dsimp only
  unfold PadEvaluationResidual.imageCoefficientAt
  change
    lift (matrixVectorAt baseOps
        (data.padCoefficientMatrices baseOps
          data.matrixSource.kernel.constant)
        (data.assignments (runningSourceIndex running)) vertex) =
      lift (matrixVectorAt baseOps (data.padMatrix baseOps)
        (data.assignments (runningSourceIndex running)) vertex)
  rw [data.padCoefficientMatrix_constant_eq baseOps baseLaws constantLaw]

end ConnectedInputs

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
