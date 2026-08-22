import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/MatrixCoefficientSource.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
One authoritative paper matrix for both CCS rows and carried ring
coefficients.

Protocol: SuperNeo coefficient embedding (Section 5) and `Pi_CCS`
(Section 7.3 / Appendix D.4).
Phase: structure ownership before CCS and carried-evaluation residuals.
Constraint family: field-matrix to coefficient-expanded matrix images.

Owns: an exact injection of logical field columns into padded
block/coefficient positions; explicit zero semantics for padding positions; a
finite bilinear coefficient kernel; derivation of every carried coefficient
matrix from the sole paper field matrix `M`; connected joint inputs that
contain no separately settable coefficient-matrix field; and the constant-term
connection to the original CCS matrix under the paper
inner-product-transform law.

Does not own: the concrete Phi81 inner-product transform, proof that the Rust
matrix cache instantiates this kernel, production padding/order, transcript,
SumCheck, R1CS, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `MatrixSource.matrices` is stored exactly once.
`coefficientMatrix` is a definition computed from it, the partial padded
column layout, and a named kernel. Missing final-block positions are zero by
definition rather than caller data. `ConnectedInputs` therefore makes the disconnected
countermodel in `Necessity.CoefficientConnectivity` unconstructible. The
constant-term theorem is conditional on the explicit kernel law; a production
assurance theorem must instantiate that law with the exact Phi81 transform.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| coefficient embedding | field layout | logical column / padded block and coefficient | `RingColumnLayout` is an exact partial inverse with explicit padding holes |
| coefficient embedding | ring action | output / row / assignment coefficient | `CoefficientKernel.weight` owns the bilinear map |
| coefficient embedding | constant term | transformed ring product | `ConstantTermLaw` is the Kronecker inner-product law |
| `Pi_CCS` | CCS source | structure matrix | `MatrixSource.matrices` is the sole stored `M` |
| `Pi_CCS` | carried source | coefficient matrices | `coefficientMatrix` is derived from `M` and the kernel |
| `Pi_CCS` | semantic input | all residual families | `ConnectedInputs.toUnifiedInputs` has no independent coefficient view |
| assurance | cross-view connection | coefficient zero / CCS image | `carriedImageConstantAt_eq_ccsImageAt` |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra
open UnifiedSources

universe uBase uExtension

/-- Explicit finite sum over the natural range `0 .. count - 1`. -/
def sumRange
    {Base : Type uBase}
    (ops : InterpolationOps Base) : Nat -> (Nat -> Base) -> Base
  | 0, _ => ops.zero
  | count + 1, term =>
      ops.add (sumRange ops count term) (term count)

/-- Pointwise equality over the active range preserves the finite sum. -/
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

/-- A finite sum vanishes when every active term vanishes. -/
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

/-- A finite sum with one selected natural index returns exactly that term. -/
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
          rw [if_neg (Nat.ne_of_lt indexLt)]
        rw [prefixZero, if_pos rfl]
        exact laws.zero_add _
      · have selectedEarlier : selected < count := by omega
        rw [inductionHypothesis selectedEarlier, if_neg]
        · exact laws.add_zero _
        · exact Ne.symm last

/-- Exact placement of logical field columns into padded ring positions.
`encode? = none` represents a final-block padding position. Every logical
column round-trips, and every present padded position decodes to its unique
logical column. -/
structure RingColumnLayout
    (coefficientCount blockCount columns : Nat) where
  decode : Fin columns -> Fin blockCount × Fin coefficientCount
  encode? : Fin blockCount -> Fin coefficientCount -> Option (Fin columns)
  decode_encode : forall block coefficient column,
    encode? block coefficient = some column ->
      decode column = (block, coefficient)
  encode_decode : forall column,
    encode? (decode column).1 (decode column).2 = some column

/-- Finite bilinear kernel for one coefficient of the transformed ring
product. `weight output row assignment` is the coefficient multiplying one
field-matrix entry and one assignment entry. -/
structure CoefficientKernel (Base : Type uBase) (coefficientCount : Nat) where
  constant : Fin coefficientCount
  weight : Fin coefficientCount -> Fin coefficientCount ->
    Fin coefficientCount -> Base

/-- Paper Theorem 3 / Theorem 4 obligation: the constant coefficient of the
transformed ring product is the ordinary field inner product. -/
structure ConstantTermLaw
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    {coefficientCount : Nat}
    (kernel : CoefficientKernel Base coefficientCount) : Prop where
  weight : forall row assignment,
    kernel.weight kernel.constant row assignment =
      if row = assignment then ops.one else ops.zero

/-- The sole paper structure owner. All carried coefficient matrices are
computed from these field matrices rather than supplied beside them. -/
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

/-- The CCS view is a direct projection of the sole stored field matrices. -/
def system
    {Base : Type uBase}
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount) :
    CCSResidualTable.Structure Base shape columns where
  matrices := source.matrices
  constraintPolynomial := source.constraintPolynomial

/-- One original matrix entry at a padded block/coefficient position. A
position beyond the logical column count is canonically zero. -/
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
  match source.columnLayout.encode? block coefficient with
  | some column => source.matrices matrix vertex column
  | none => ops.zero

/-- One coefficient-expanded matrix derived from the sole stored `M`.
The finite sum linearizes one blockwise transformed ring product against an
assignment coefficient. -/
def coefficientMatrix
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount)
    (matrix : Fin shape.matrixCount)
    (output : Fin shape.coefficientCount) :
    BooleanMatrix Base shape.cubeVariables columns :=
  fun vertex column =>
    let packed := source.columnLayout.decode column
    sumRange ops shape.coefficientCount fun rowIndex =>
      if rowLt : rowIndex < shape.coefficientCount then
        let row : Fin shape.coefficientCount :=
          ⟨rowIndex, rowLt⟩
        ops.mul
          (source.paddedMatrixEntry ops matrix vertex packed.1 row)
          (source.kernel.weight output row packed.2)
      else
        ops.zero

/-- Complete coefficient family, derived rather than stored. -/
def coefficientMatrices
    {Base : Type uBase}
    (ops : InterpolationOps Base)
    {shape : Shape}
    {columns blockCount : Nat}
    (source : MatrixSource Base shape columns blockCount) :
    Fin shape.matrixCount -> Fin shape.coefficientCount ->
      BooleanMatrix Base shape.cubeVariables columns :=
  fun matrix coefficient => source.coefficientMatrix ops matrix coefficient

/-- The constant derived coefficient at one leaf is exactly the original
field-matrix entry. -/
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
  let packed := source.columnLayout.decode column
  let block := packed.1
  let selected := packed.2
  change
    sumRange ops shape.coefficientCount (fun rowIndex =>
      if rowLt : rowIndex < shape.coefficientCount then
        let row : Fin shape.coefficientCount := ⟨rowIndex, rowLt⟩
        ops.mul
          (source.paddedMatrixEntry ops matrix vertex block row)
          (source.kernel.weight source.kernel.constant row selected)
      else
        ops.zero) =
      source.matrices matrix vertex column
  calc
    sumRange ops shape.coefficientCount (fun rowIndex =>
        if rowLt : rowIndex < shape.coefficientCount then
          let row : Fin shape.coefficientCount := ⟨rowIndex, rowLt⟩
          ops.mul
            (source.paddedMatrixEntry ops matrix vertex block row)
            (source.kernel.weight source.kernel.constant row selected)
        else
          ops.zero) =
        sumRange ops shape.coefficientCount (fun rowIndex =>
          if rowIndex = selected.val then
            source.paddedMatrixEntry ops matrix vertex block selected
          else
            ops.zero) := by
      apply sumRange_congr
      intro rowIndex rowLt
      rw [dif_pos rowLt]
      let row : Fin shape.coefficientCount := ⟨rowIndex, rowLt⟩
      change
        ops.mul
            (source.paddedMatrixEntry ops matrix vertex block row)
            (source.kernel.weight source.kernel.constant row selected) =
          if rowIndex = selected.val then
            source.paddedMatrixEntry ops matrix vertex block selected
          else
            ops.zero
      rw [constantLaw.weight]
      by_cases equal : row = selected
      · have valueEqual : rowIndex = selected.val :=
          congrArg Fin.val equal
        rw [if_pos equal, if_pos valueEqual, laws.mul_one]
        exact congrArg
          (fun coefficient =>
            source.paddedMatrixEntry ops matrix vertex block coefficient)
          equal
      · have valueDifferent : rowIndex ≠ selected.val := by
          intro valueEqual
          apply equal
          exact Fin.eq_of_val_eq valueEqual
        rw [if_neg equal, if_neg valueDifferent, laws.mul_zero]
    _ = source.paddedMatrixEntry ops matrix vertex block selected := by
      exact sumRange_select ops laws shape.coefficientCount selected.val
        (fun _ => source.paddedMatrixEntry ops matrix vertex block selected)
        selected.isLt
    _ = source.matrices matrix vertex column := by
      simp only [paddedMatrixEntry]
      rw [source.columnLayout.encode_decode column]

/-- Functional form of the constant-term connection for a whole matrix. -/
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
  exact coefficientMatrix_constant_apply ops laws source constantLaw
    matrix vertex column

end MatrixSource

/-- Connected paper inputs. Unlike `UnifiedInputs`, this structure has no
field in which a caller can place independent coefficient matrices. -/
structure ConnectedInputs
    (Extension : Type uExtension)
    (shape : Shape)
    (columns blockCount : Nat) where
  cubeLayout : ColumnLayout shape.cubeVariables columns
  matrixSource : MatrixSource F shape columns blockCount
  assignments : Fin shape.sourceCount -> Assignment F columns
  priorPoint : CubePoint Extension shape.cubeVariables
  claimedCoefficient : CarriedCoordinate shape -> Extension

namespace ConnectedInputs

/-- Internal projection into the already-proved joint residual stack. Every
coefficient matrix is derived from `matrixSource`; no field is copied from the
caller. -/
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
  coefficientMatrices := data.matrixSource.coefficientMatrices baseOps
  priorPoint := data.priorPoint
  claimedCoefficient := data.claimedCoefficient

/-- The CCS structure is exactly the sole stored matrix source. -/
theorem toUnifiedInputs_system_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (data : ConnectedInputs Extension shape columns blockCount) :
    (data.toUnifiedInputs baseOps).system = data.matrixSource.system := by
  rfl

/-- Every carried coefficient matrix is definitionally the source-derived
kernel expansion. -/
theorem toUnifiedInputs_coefficientMatrices_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (baseOps : InterpolationOps F)
    (data : ConnectedInputs Extension shape columns blockCount) :
    (data.toUnifiedInputs baseOps).coefficientMatrices =
      data.matrixSource.coefficientMatrices baseOps := by
  rfl

/-- Semantic truth exposed only through the connected projection. -/
def SemanticTruth
    {Extension : Type uExtension}
    {shape : Shape}
    {columns blockCount : Nat}
    (data : ConnectedInputs Extension shape columns blockCount)
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension) : Prop :=
  (data.toUnifiedInputs baseOps).SemanticTruth baseOps extensionOps lift

/-- The constant coefficient of a carried matrix image is exactly the CCS
matrix image at the same Boolean row and authoritative running assignment. -/
theorem carriedImageConstantAt_eq_ccsImageAt
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
    let coordinate : CarriedCoordinate shape :=
      { running := running
        matrix := matrix
        coefficient := data.matrixSource.kernel.constant }
    CarriedEvaluationResidual.imageCoefficientAt baseOps lift
        (data.toUnifiedInputs baseOps).carriedData coordinate vertex =
      lift (matrixVectorAt baseOps (data.matrixSource.matrices matrix)
        (data.assignments (runningSourceIndex running)) vertex) := by
  dsimp only
  unfold CarriedEvaluationResidual.imageCoefficientAt
  change
    lift (matrixVectorAt baseOps
        (data.matrixSource.coefficientMatrix baseOps matrix
          data.matrixSource.kernel.constant)
        (data.assignments (runningSourceIndex running)) vertex) =
      lift (matrixVectorAt baseOps (data.matrixSource.matrices matrix)
        (data.assignments (runningSourceIndex running)) vertex)
  rw [data.matrixSource.coefficientMatrix_constant_eq baseOps baseLaws
    constantLaw matrix]

end ConnectedInputs

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
