import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement

/-!
The paper's complete `y'` family and its exact projection to the executable
`Pi_CCS` output message.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: honest output construction at the verifier-derived point.
Constraint family: semantic output coordinates only; this file emits no rows.

Owns: one coefficient-complete value family for every source/matrix pair;
the explicit paper property that the first matrix of the sole
`MatrixSource` is the padded identity `[I; 0]`; projection of the full family onto all three
fields of `ProtocolPolynomial.OutputMessage`; and proof that honest evaluation
of the full family projects to `ProtocolPolynomial.messageAt` at the same
point.

Does not own: SumCheck acceptance, an operational game, probability,
Fiat--Shamir, a concrete Phi81 instantiation of the coefficient kernel,
production column decoding, Rust, R1CS, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: matrices and assignments are read only from
`MatrixCoefficientSource.ConnectedInputs`. The full output is computed from
that source through `coefficientMatrix`; it is never accepted as an
independent semantic oracle. The source-assignment component is obtained from
the entrywise `M_1 = [I; 0]` property, not from an assumed matrix-vector equation.

| Protocol path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.output.full` | all `y'_(i,j,l)(r')` share one source/matrix/coefficient family | computed | `FullOutput` / `honestAt` |
| `pi_ccs.structure.matrix.first` | the first sole-source matrix is entrywise `[I; 0]` | checked | `IdentityFirstMatrix` |
| `pi_ccs.output.fresh` | fresh `y_(i,j)` is the constant coordinate of `y'_(i,j)` | derived | `toOutputMessage` |
| `pi_ccs.output.assignment` | `z_i(r')` is the first-matrix constant coordinate | derived | `toOutputMessage` |
| `pi_ccs.output.carried` | carried coefficient images are the running-source coordinates | direct dataflow | `toOutputMessage` |
| `pi_ccs.output.projection` | honest full output equals the canonical message at one point | derived | `honestAt_toOutputMessage_eq_messageAt` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open MatrixCoefficientSource
open PaperLinearAlgebra
open UnifiedSources

universe uExtension

/-- The complete paper output at one common point: every source, every
matrix, and every coefficient of the coefficient-expanded matrix image. -/
structure FullOutput (Extension : Type uExtension) (shape : Shape) where
  coordinate : Fin shape.sourceCount -> Fin shape.matrixCount ->
    Fin shape.coefficientCount -> Extension

/-- The paper's `M_1 = [I; 0]` requirement attached to the sole matrix owner.
The live column for row `vertex`, or its padding status, comes from the same
layout used by the authoritative assignment family. -/
structure IdentityFirstMatrix
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (data : ConnectedInputs Extension shape columns blockCount) : Prop where
  matrixCountPositive : 0 < shape.matrixCount
  entry : forall
      (vertex : BooleanVertex shape.cubeVariables)
      (column : Fin columns),
    data.matrixSource.matrices
          ⟨0, matrixCountPositive⟩ vertex column =
      data.cubeLayout.paddedIdentityEntry baseOps.zero baseOps.one
        vertex column

namespace IdentityFirstMatrix

/-- Typed zero-based index of the paper's first matrix. -/
def index
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    {data : ConnectedInputs Extension shape columns blockCount}
    (identity : IdentityFirstMatrix baseOps data) : Fin shape.matrixCount :=
  ⟨0, identity.matrixCountPositive⟩

/-- The entrywise identity requirement derives the matrix-vector action; the
action itself is not supplied as a premise. -/
theorem matrixVectorAt_first_eq_assignment
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (data : ConnectedInputs Extension shape columns blockCount)
    (identity : IdentityFirstMatrix baseOps data)
    (source : Fin shape.sourceCount)
    (vertex : BooleanVertex shape.cubeVariables) :
    matrixVectorAt baseOps
        (data.matrixSource.matrices identity.index)
        (data.assignments source) vertex =
      data.cubeLayout.paddedValue baseOps.zero
        (data.assignments source) vertex := by
  cases decoded : data.cubeLayout.toColumn? vertex with
  | none =>
      simp only [ColumnLayout.paddedValue, decoded]
      apply matrixVectorAt_zeroRow baseOps baseLaws
      intro column
      simpa [ColumnLayout.paddedIdentityEntry, decoded] using
        identity.entry vertex column
  | some selected =>
      simp only [ColumnLayout.paddedValue, decoded]
      apply matrixVectorAt_identityRow baseOps baseLaws
        (data.matrixSource.matrices identity.index)
        (data.assignments source) vertex selected
      intro column
      simpa [ColumnLayout.paddedIdentityEntry, decoded] using
        identity.entry vertex column

end IdentityFirstMatrix

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
  coordinate := fun source matrix coefficient =>
    (BooleanTable.tabulate fun vertex =>
      lift (matrixVectorAt baseOps
        (data.matrixSource.coefficientMatrix baseOps matrix coefficient)
        (data.assignments source) vertex)).evaluate extensionOps point

/-- Project the complete paper output onto the executable message surface.
Fresh matrix images and assignment evaluations are both constant-coordinate
views; carried images retain every coefficient for the running sources. -/
def toOutputMessage
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    {data : ConnectedInputs Extension shape columns blockCount}
    (identity : IdentityFirstMatrix baseOps data)
    (output : FullOutput Extension shape) :
    ProtocolPolynomial.OutputMessage Extension shape where
  freshMatrixImage := fun source matrix =>
    output.coordinate (freshSourceIndex source) matrix
      data.matrixSource.kernel.constant
  sourceAssignment := fun source =>
    output.coordinate source identity.index
      data.matrixSource.kernel.constant
  carriedImage := fun coordinate =>
    output.coordinate (runningSourceIndex coordinate.running)
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
    (identity : IdentityFirstMatrix baseOps data)
    (point : CubePoint Extension shape.cubeVariables)
    (source : Fin shape.freshCount)
    (matrix : Fin shape.matrixCount) :
    ((honestAt baseOps extensionOps lift data point).toOutputMessage identity).freshMatrixImage
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

/-- The first identity matrix makes the honest full-output coordinate exactly
the canonical source-assignment component of `messageAt`. -/
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
    (identity : IdentityFirstMatrix baseOps data)
    (point : CubePoint Extension shape.cubeVariables)
    (source : Fin shape.sourceCount) :
    ((honestAt baseOps extensionOps lift data point).toOutputMessage identity).sourceAssignment
        source =
      (ProtocolPolynomial.messageAt extensionOps
          (ProtocolDataRefinement.toProtocolData baseOps lift
            (data.toUnifiedInputs baseOps)) point).sourceAssignment source := by
  unfold honestAt toOutputMessage ProtocolPolynomial.messageAt
    ProtocolDataRefinement.toProtocolData
  change
    (BooleanTable.tabulate fun vertex =>
      lift (matrixVectorAt baseOps
        (data.matrixSource.coefficientMatrix baseOps identity.index
          data.matrixSource.kernel.constant)
        (data.assignments source) vertex)).evaluate extensionOps point =
      (BooleanTable.tabulate fun vertex =>
        lift (data.cubeLayout.paddedValue 0
          (data.assignments source) vertex)).evaluate extensionOps point
  rw [data.matrixSource.coefficientMatrix_constant_eq baseOps baseLaws
    constantLaw identity.index]
  apply congrArg (fun table : BooleanTable Extension shape.cubeVariables =>
    table.evaluate extensionOps point)
  apply congrArg (fun values : BooleanVertex shape.cubeVariables -> Extension =>
    BooleanTable.tabulate values)
  funext vertex
  rw [identity.matrixVectorAt_first_eq_assignment baseOps baseLaws data
    source vertex]
  apply congrArg lift
  cases decoded : data.cubeLayout.toColumn? vertex with
  | none => simp [ColumnLayout.paddedValue, decoded, baseZero.zero_eq]
  | some column => simp [ColumnLayout.paddedValue, decoded]

/-- Running-source coefficient projection is definitionally the canonical
carried-image component of `messageAt`. -/
theorem honestAt_carriedImage_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension)
    (data : ConnectedInputs Extension shape columns blockCount)
    (identity : IdentityFirstMatrix baseOps data)
    (point : CubePoint Extension shape.cubeVariables)
    (coordinate : CarriedCoordinate shape) :
    ((honestAt baseOps extensionOps lift data point).toOutputMessage identity).carriedImage
        coordinate =
      (ProtocolPolynomial.messageAt extensionOps
          (ProtocolDataRefinement.toProtocolData baseOps lift
            (data.toUnifiedInputs baseOps)) point).carriedImage coordinate := by
  rfl

/-- Honest evaluation of the complete paper `y'` family projects exactly to
the canonical executable output message at the same point. The only bridge
premises are the paper constant-term kernel law and the entrywise
`M_1 = [I; 0]`
property of the sole matrix source. -/
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
    (identity : IdentityFirstMatrix baseOps data)
    (point : CubePoint Extension shape.cubeVariables) :
    (honestAt baseOps extensionOps lift data point).toOutputMessage identity =
      ProtocolPolynomial.messageAt extensionOps
        (ProtocolDataRefinement.toProtocolData baseOps lift
          (data.toUnifiedInputs baseOps)) point := by
  apply ProtocolPolynomial.OutputMessage.ext
  · funext source matrix
    exact honestAt_freshMatrixImage_eq baseOps baseLaws extensionOps lift data
      constantLaw identity point source matrix
  · funext source
    exact honestAt_sourceAssignment_eq baseOps baseLaws baseZero extensionOps
      lift data constantLaw identity point source
  · funext coordinate
    exact honestAt_carriedImage_eq baseOps extensionOps lift data identity point
      coordinate

end FullOutput

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates
