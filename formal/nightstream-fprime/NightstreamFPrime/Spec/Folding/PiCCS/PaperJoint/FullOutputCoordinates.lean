import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolDataRefinement

/-! Provenance: adapted from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/FullOutputCoordinates.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; split into the
SuperNeo v1.1 Pad and 14-matrix output families. -/

/-!
The paper's complete `y'` family and its exact projection to the executable
`Pi_CCS` output message.

Protocol: SuperNeo v1.1 `Pi_CCS` (Section 7.3 / Appendix B.2).
Phase: honest output construction at the verifier-derived point.
Constraint family: semantic output coordinates only; this file emits no rows.

Owns: separate coefficient-complete Pad and 14-matrix value families for
every source; projection onto all four fields of
`ProtocolPolynomial.OutputMessage`; and proof that honest evaluation projects
to `ProtocolPolynomial.messageAt` at the same point.

Does not own: SumCheck acceptance, an operational game, probability,
Fiat--Shamir, a concrete Phi81 instantiation of the coefficient kernel,
production column decoding, Rust, R1CS, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: matrices and assignments are read only from
`MatrixCoefficientSource.ConnectedInputs`. The full output is computed from
that source through `coefficientMatrix`; it is never accepted as an
independent semantic oracle. The source-assignment component is obtained from
the canonical Pad matrix, not from a selected CCS matrix.

| Protocol path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.output.full` | all Pad and matrix coefficients share one source family | computed | `FullOutput` / `honestAt` |
| `pi_ccs.output.fresh` | fresh `y_(i,j)` is the constant coordinate of `y'_(i,j)` | derived | `toOutputMessage` |
| `pi_ccs.output.assignment` | `z_i(r')` is the Pad constant coordinate | derived | `toOutputMessage` |
| `pi_ccs.output.pad` | Pad coefficient images are the running-source coordinates | direct dataflow | `toOutputMessage` |
| `pi_ccs.output.matrix` | all matrix coefficient images are the running-source coordinates | direct dataflow | `toOutputMessage` |
| `pi_ccs.output.projection` | honest full output equals the canonical message at one point | derived | `honestAt_toOutputMessage_eq_messageAt` |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FullOutputCoordinates

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open MatrixCoefficientSource
open PaperLinearAlgebra
open UnifiedSources

universe uExtension

/-- The complete v1.1 output at one common point: every source and every
coefficient of both canonical Pad and all CCS matrix images. -/
structure FullOutput (Extension : Type uExtension) (shape : Shape) where
  padCoordinate : Fin shape.sourceCount ->
    Fin shape.coefficientCount -> Extension
  matrixCoordinate : Fin shape.sourceCount -> Fin shape.matrixCount ->
    Fin shape.coefficientCount -> Extension

/-- Canonical Pad acts as `[I; 0]` on every authoritative assignment. The
matrix-vector action is proved from the shared layout, not supplied as a
premise. -/
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

/-- Honest evaluation of the complete `y'` family at one typed point. Every
coordinate is computed from the same sole matrix source and authoritative
source assignment family. -/
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

/-- Project the complete paper output onto the executable message surface.
Fresh matrix images and assignment evaluations are constant-coordinate
views; Pad and matrix images retain every coefficient for running sources. -/
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

/-- Constant-coordinate projection of an honest fresh matrix image is the
canonical fresh matrix-image component of `messageAt`. -/
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

/-- Canonical Pad makes the honest constant coordinate exactly the canonical
source-assignment component of `messageAt`. -/
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

/-- Running-source Pad projection is definitionally the canonical Pad-image
component of `messageAt`. -/
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

/-- Running-source matrix projection is definitionally the canonical
matrix-image component of `messageAt`. -/
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

/-- Honest evaluation of the complete paper `y'` family projects exactly to
the canonical executable output message at the same point. The only bridge
premise is the paper constant-term kernel law. Canonical Pad is derived from
the shared layout. -/
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
