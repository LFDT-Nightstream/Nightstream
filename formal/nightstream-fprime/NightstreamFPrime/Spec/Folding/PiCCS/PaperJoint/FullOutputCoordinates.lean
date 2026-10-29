import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolDataRefinement

/-! SuperNeo v1.2 Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FullOutputCoordinates

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open MatrixCoefficientSource
open PaperLinearAlgebra
open UnifiedSources

universe uExtension

structure FullOutput (Extension : Type uExtension) (shape : Shape) where
  padCoordinate : Fin shape.sourceCount ->
    Fin shape.coefficientCount -> Extension
  matrixCoordinate : Fin shape.sourceCount -> Fin shape.matrixCount ->
    Fin shape.coefficientCount -> Extension

theorem padMatrixVectorAt_eq_assignment
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (data : ConnectedInputs Extension shape columns blockCount)
    (source : Fin shape.sourceCount)
    (vertex : BooleanVertex shape.cubeVariables) :
    matrixVectorAt baseOps
        (data.padMatrix baseOps)
        (data.assignments source) vertex =
      data.cubeLayout.paddedValue baseOps.zero
        (data.assignments source) vertex := by
  cases decoded : data.cubeLayout.toColumn? vertex with
  | none =>
      simp only [ColumnLayout.paddedValue, decoded]
      apply matrixVectorAt_zeroRow baseOps baseLaws
      intro column
      simp [ConnectedInputs.padMatrix,
        ColumnLayout.paddedIdentityEntry, decoded]
  | some selected =>
      simp only [ColumnLayout.paddedValue, decoded]
      apply matrixVectorAt_identityRow baseOps baseLaws
        (data.padMatrix baseOps)
        (data.assignments source) vertex selected
      intro column
      simp [ConnectedInputs.padMatrix,
        ColumnLayout.paddedIdentityEntry, decoded]

namespace FullOutput

def honestAt
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (data : ConnectedInputs Extension shape columns blockCount)
    (point : CubePoint Extension shape.cubeVariables) :
    FullOutput Extension shape where
  padCoordinate := fun source coefficient =>
    (BooleanTable.tabulate fun vertex =>
      lift (matrixVectorAt baseOps
        (data.padCoefficientMatrices baseOps coefficient)
        (data.assignments source) vertex)).evaluate extensionOps point
  matrixCoordinate := fun source matrix coefficient =>
    (BooleanTable.tabulate fun vertex =>
      lift (matrixVectorAt baseOps
        (data.matrixSource.coefficientMatrix baseOps matrix coefficient)
        (data.assignments source) vertex)).evaluate extensionOps point

def toOutputMessage
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (data : ConnectedInputs Extension shape columns blockCount)
    (output : FullOutput Extension shape) :
    ProtocolPolynomial.OutputMessage Extension shape where
  freshMatrixImage := fun source matrix =>
    output.matrixCoordinate (freshSourceIndex source) matrix
      data.matrixSource.kernel.constant
  sourceAssignment := fun source =>
    output.padCoordinate source
      data.matrixSource.kernel.constant
  padImage := fun coordinate =>
    output.padCoordinate (runningSourceIndex coordinate.running)
      coordinate.coefficient
  matrixImage := fun coordinate =>
    output.matrixCoordinate (runningSourceIndex coordinate.running)
      coordinate.matrix coordinate.coefficient

theorem honestAt_freshMatrixImage_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (data : ConnectedInputs Extension shape columns blockCount)
    (constantLaw : ConstantTermLaw baseOps data.matrixSource.kernel)
    (point : CubePoint Extension shape.cubeVariables)
    (source : Fin shape.freshCount)
    (matrix : Fin shape.matrixCount) :
    ((honestAt baseOps extensionOps lift data point).toOutputMessage data).freshMatrixImage
        source matrix =
      (ProtocolPolynomial.messageAt extensionOps
          (ProtocolDataRefinement.toProtocolData baseOps lift
            (data.toUnifiedInputs baseOps)) point).freshMatrixImage
        source matrix := by
  unfold honestAt toOutputMessage ProtocolPolynomial.messageAt
    ProtocolDataRefinement.toProtocolData
  change
    (BooleanTable.tabulate fun vertex =>
      lift (matrixVectorAt baseOps
        (data.matrixSource.coefficientMatrix baseOps matrix
          data.matrixSource.kernel.constant)
        (data.assignments (freshSourceIndex source)) vertex)).evaluate
          extensionOps point =
      (BooleanTable.tabulate fun vertex =>
        lift (matrixVectorAt baseOps
          (data.matrixSource.matrices matrix)
          (data.assignments (freshSourceIndex source)) vertex)).evaluate
            extensionOps point
  rw [data.matrixSource.coefficientMatrix_constant_eq baseOps baseLaws
    constantLaw matrix]

theorem honestAt_sourceAssignment_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (data : ConnectedInputs Extension shape columns blockCount)
    (constantLaw : ConstantTermLaw baseOps data.matrixSource.kernel)
    (point : CubePoint Extension shape.cubeVariables)
    (source : Fin shape.sourceCount) :
    ((honestAt baseOps extensionOps lift data point).toOutputMessage data).sourceAssignment
        source =
      (ProtocolPolynomial.messageAt extensionOps
          (ProtocolDataRefinement.toProtocolData baseOps lift
            (data.toUnifiedInputs baseOps)) point).sourceAssignment source := by
  unfold honestAt toOutputMessage ProtocolPolynomial.messageAt
    ProtocolDataRefinement.toProtocolData
  change
    (BooleanTable.tabulate fun vertex =>
      lift (matrixVectorAt baseOps
        (data.padCoefficientMatrices baseOps
          data.matrixSource.kernel.constant)
        (data.assignments source) vertex)).evaluate extensionOps point =
      (BooleanTable.tabulate fun vertex =>
        lift (data.cubeLayout.paddedValue 0
          (data.assignments source) vertex)).evaluate extensionOps point
  rw [data.padCoefficientMatrix_constant_eq baseOps baseLaws constantLaw]
  apply congrArg (fun table : BooleanTable Extension shape.cubeVariables =>
    table.evaluate extensionOps point)
  apply congrArg (fun values : BooleanVertex shape.cubeVariables -> Extension =>
    BooleanTable.tabulate values)
  funext vertex
  rw [padMatrixVectorAt_eq_assignment baseOps baseLaws data source vertex]
  apply congrArg lift
  cases decoded : data.cubeLayout.toColumn? vertex with
  | none => simp [ColumnLayout.paddedValue, decoded, baseZero.zero_eq]
  | some column => simp [ColumnLayout.paddedValue, decoded]

theorem honestAt_padImage_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (data : ConnectedInputs Extension shape columns blockCount)
    (point : CubePoint Extension shape.cubeVariables)
    (coordinate : PadCoordinate shape) :
    ((honestAt baseOps extensionOps lift data point).toOutputMessage data).padImage
        coordinate =
      (ProtocolPolynomial.messageAt extensionOps
          (ProtocolDataRefinement.toProtocolData baseOps lift
          (data.toUnifiedInputs baseOps)) point).padImage coordinate := by
  rfl

theorem honestAt_matrixImage_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (data : ConnectedInputs Extension shape columns blockCount)
    (point : CubePoint Extension shape.cubeVariables)
    (coordinate : MatrixCoordinate shape) :
    ((honestAt baseOps extensionOps lift data point).toOutputMessage data).matrixImage
        coordinate =
      (ProtocolPolynomial.messageAt extensionOps
          (ProtocolDataRefinement.toProtocolData baseOps lift
            (data.toUnifiedInputs baseOps)) point).matrixImage coordinate := by
  rfl

theorem honestAt_toOutputMessage_eq_messageAt
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (data : ConnectedInputs Extension shape columns blockCount)
    (constantLaw : ConstantTermLaw baseOps data.matrixSource.kernel)
    (point : CubePoint Extension shape.cubeVariables) :
    (honestAt baseOps extensionOps lift data point).toOutputMessage data =
      ProtocolPolynomial.messageAt extensionOps
        (ProtocolDataRefinement.toProtocolData baseOps lift
          (data.toUnifiedInputs baseOps)) point := by
  apply ProtocolPolynomial.OutputMessage.ext
  · funext source matrix
    exact honestAt_freshMatrixImage_eq baseOps baseLaws extensionOps lift data
      constantLaw point source matrix
  · funext source
    exact honestAt_sourceAssignment_eq baseOps baseLaws baseZero extensionOps
      lift data constantLaw point source
  · funext coordinate
    exact honestAt_padImage_eq baseOps extensionOps lift data point coordinate
  · funext coordinate
    exact honestAt_matrixImage_eq baseOps extensionOps lift data point
      coordinate

end FullOutput

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FullOutputCoordinates
