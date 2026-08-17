import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.LeanCompiler.DirectRows
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialPrepend
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PrefixLayout
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

/-!
Contract: model-level validation of the Nightstream `PaddedRowIdentity`
relation.

Owns:
- one fixed reference 24-variable row cube, 14,944,219-row logical prefix, and
  11,437,038-coordinate assignment prefix;
- zero-row padding of the exact thirteen-port, degree-eight selective CCS
  polynomial;
- the prepended matrix `M_0 = [I; 0]` and the lifted polynomial that ignores
  exactly that new input;
- exact logical-CCS/padded-CCS acceptance equivalence;
- the constant coefficient of the honest `M_0` ring output as the MLE of the
  same authoritative padded assignment used by the joint relation.

Does not own: Rust or R1CS conformance, a production matrix artifact, a
Fiat--Shamir reduction, Poseidon2 bytes, a commitment-security assumption, or
an end-to-end release claim.

Emits constraints: no.

Assurance tier: model-level reference snapshot. The polynomial is the
independent 74-term selective polynomial, but the matrix family remains an
explicit typed input. Therefore these theorems validate the construction for
every matrix family of these fixed dimensions; they do not select the active
verifier-key dimensions or claim that Rust emits a particular family.

The key theorem is an equivalence, not a one-way completeness result:
`identityFirstConstraintSatisfied_iff_logical`. If padding or the ignored
identity variable changes acceptance, this file cannot compile.

| Code owner | Protocol object | Mathematical obligation | Proven result |
|---|---|---|---|
| `applicationSystem` | zero-padded thirteen-matrix CCS relation | padding must preserve and reflect logical acceptance | exact iff |
| `identityFirstSystem` | `M_0 = [I; 0]` prepended to CCS | the lifted polynomial must ignore only `M_0` | exact iff with logical CCS |
| `identityMatrixVector_eq_paddedAssignment` | identity-matrix source | `M_0 z` must equal the canonical padded assignment | exact equality |
| `identityConstantOutput_eq_paddedAssignmentMLE` | joint norm input | the constant coefficient must open the same authoritative assignment | exact equality |
| `connectedSemanticTruth_iff_logicalSemanticTruth` | one-joint semantic relation | padded joint truth must equal logical CCS, norm, and carried claims | exact iff |
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FullOutputCoordinates
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial
open Nightstream.Implementation.R1CS.SelectiveCcs
open Nightstream.Implementation.R1CS.SelectiveCcs.LeanCompiler

/-- Fixed reference-snapshot row-cube dimension. -/
def rowVariables : Nat := 24

/-- Fixed reference-snapshot logical relation row count. -/
def logicalRows : Nat := 14944219

/-- Fixed reference-snapshot assignment width. This is 211,797 Phi81
ring columns. -/
def assignmentColumns : Nat := 11437038

/-- Selected application matrix count before `M_0` is prepended. -/
def applicationMatrixCount : Nat := 13

/-- Selected joint matrix count after `M_0` is prepended. -/
def jointMatrixCount : Nat := applicationMatrixCount + 1

/-- The selected row cube strictly contains the logical rows. -/
theorem logicalRows_lt_cube : logicalRows < 2 ^ rowVariables := by
  decide

/-- Every logical row fits in the selected row cube. -/
theorem logicalRows_covered : logicalRows <= 2 ^ rowVariables :=
  Nat.le_of_lt logicalRows_lt_cube

/-- The selected row cube strictly contains the assignment prefix. -/
theorem assignmentColumns_lt_cube :
    assignmentColumns < 2 ^ rowVariables := by
  decide

/-- Every assignment coordinate fits in the selected row cube. -/
theorem assignmentColumns_covered :
    assignmentColumns <= 2 ^ rowVariables :=
  Nat.le_of_lt assignmentColumns_lt_cube

/-- The assignment width is exactly divisible into Phi81 coefficient blocks. -/
theorem assignmentColumns_eq_ringBlocks :
    assignmentColumns =
      Phi81ColumnLayout.blockCount assignmentColumns * ringDegree := by
  decide

/-- Exact joint paper shape for the fixed reference snapshot. -/
def shape : Shape where
  cubeVariables := rowVariables
  freshCount := 1
  runningCount := 16
  matrixCount := jointMatrixCount
  coefficientCount := ringDegree

theorem shape_cubeVariables_exact : shape.cubeVariables = 24 := by
  rfl

theorem shape_matrixCount_exact : shape.matrixCount = 14 := by
  rfl

theorem shape_sourceCount_exact : shape.sourceCount = 17 := by
  rfl

theorem shape_carriedEvaluationCount_exact :
    shape.carriedEvaluationCount = 12096 := by
  rfl

theorem shape_jointCoefficientCount_exact :
    shape.jointCoefficientCount = 12114 := by
  rfl

theorem terminalRingValueCount_exact :
    shape.sourceCount * shape.matrixCount = 238 := by
  rfl

/-- Canonical little-endian assignment-prefix layout. -/
def assignmentLayout : ColumnLayout rowVariables assignmentColumns :=
  PrefixLayout.layout rowVariables assignmentColumns
    assignmentColumns_covered

/-- One exact thirteen-matrix finite application relation. -/
abbrev ApplicationMatrices :=
  RelationProfile.FiniteRelation logicalRows assignmentColumns

/-- The application-only paper shape used to state the intermediate padded
CCS relation. -/
def applicationShape : Shape where
  cubeVariables := rowVariables
  freshCount := 1
  runningCount := 16
  matrixCount := applicationMatrixCount
  coefficientCount := ringDegree

/-- The direct thirteen-matrix application system with zero rows after the
logical prefix. -/
def applicationSystem (matrices : ApplicationMatrices) :
    Structure F applicationShape assignmentColumns where
  matrices := fun matrix => RowPadding.padRows (matrices.matrixAt matrix)
  constraintPolynomial := Semantics.polynomial

/-- The padded identity matrix `M_0 = [I; 0]`. -/
def identityMatrix : BooleanMatrix F rowVariables assignmentColumns :=
  assignmentLayout.paddedIdentityEntry 0 1

/-- The selected fourteen-matrix family: identity first, then the thirteen
application matrices in their original order. -/
def identityFirstMatrices (matrices : ApplicationMatrices) :
    Fin jointMatrixCount -> BooleanMatrix F rowVariables assignmentColumns :=
  Fin.cases identityMatrix
    (fun matrix => RowPadding.padRows (matrices.matrixAt matrix))

/-- The selected joint CCS structure. Its polynomial ignores `M_0` by
construction and preserves the exact 74 application terms. -/
def identityFirstSystem (matrices : ApplicationMatrices) :
    Structure F shape assignmentColumns where
  matrices := identityFirstMatrices matrices
  constraintPolynomial :=
    ConstraintPolynomialPrepend.prependIgnoredVariable
      Semantics.polynomial

/-- The identity input does not change the syntax-derived SumCheck degree.
The selected joint polynomial therefore has degree ceiling nine. -/
theorem identityFirstDegree_exact (matrices : ApplicationMatrices) :
    ((identityFirstSystem matrices).constraintPolynomial).canonicalEqualityGatedDegreeBound =
      9 := by
  change
    (ConstraintPolynomialPrepend.prependIgnoredVariable
      Semantics.polynomial).canonicalEqualityGatedDegreeBound = 9
  rw [ConstraintPolynomialPrepend.prependIgnoredVariable_canonicalEqualityGatedDegreeBound]
  exact Semantics.canonicalEqualityGatedDegreeBound_exact

/-- Direct logical matrix images before Boolean-row padding. -/
def logicalMatrixImagesAt
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns)
    (row : Fin logicalRows) : Fin applicationMatrixCount -> F :=
  fun matrix =>
    matrixVectorAt baseOps (fun _ => matrices.matrixAt matrix row)
      assignment (.nil)

/-- Direct logical selective residual before Boolean-row padding. -/
def logicalResidualAt
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns)
    (row : Fin logicalRows) : F :=
  evaluatePolynomial baseOps Semantics.polynomial
    (logicalMatrixImagesAt matrices assignment row)

/-- Logical CCS acceptance before the row-cube injection. -/
def LogicalConstraintSatisfied
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns) : Prop :=
  forall row, logicalResidualAt matrices assignment row = 0

/-- The exact 74-term application polynomial has no constant term. This is
proved from its explicit syntax, not supplied as a profile premise. -/
theorem applicationPolynomial_at_zero :
    evaluatePolynomial baseOps Semantics.polynomial
        (fun _ => 0) = 0 := by
  change Semantics.evaluate (fun _ => 0) = 0
  rw [Components.evaluate_eq_combinedResidual]
  have canonicalZero :
      Components.canonicalResidual (fun _ => 0) = 0 :=
    Components.canonicalResidual_zero_of_generalSelector_zero
      (fun _ => 0) rfl
  simp only [Components.combinedResidual, Components.booleanResidual,
    Components.productResidual, Components.sboxResidual,
    Components.centeredResidual, Components.evaluationResidual,
    canonicalZero]
  decide

/-- Every live padded row has exactly the direct logical matrix-image tuple. -/
theorem applicationMatrixImagesAt_live
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns)
    (row : Fin logicalRows) :
    matrixImagesAt baseOps (applicationSystem matrices) assignment
        (RowPadding.numericRowVertex logicalRows_covered row) =
      logicalMatrixImagesAt matrices assignment row := by
  funext matrix
  unfold matrixImagesAt applicationSystem logicalMatrixImagesAt
  rw [DirectRows.matrixVectorAt_padRows_numeric]
  rfl

/-- Every row after the logical prefix has an all-zero application-image
tuple. -/
theorem applicationMatrixImagesAt_padding
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns)
    (vertex : BooleanVertex rowVariables)
    (padding : logicalRows <= rowIndex vertex) :
    matrixImagesAt baseOps (applicationSystem matrices) assignment vertex =
      fun _ => 0 := by
  funext matrix
  unfold matrixImagesAt applicationSystem
  exact DirectRows.matrixVectorAt_padRows_padding
    (matrices.matrixAt matrix) assignment vertex padding

/-- The padded residual at a live row is exactly its logical residual. -/
theorem applicationResidualAt_live
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns)
    (row : Fin logicalRows) :
    residualAt baseOps (applicationSystem matrices) assignment
        (RowPadding.numericRowVertex logicalRows_covered row) =
      logicalResidualAt matrices assignment row := by
  unfold residualAt logicalResidualAt
  change evaluatePolynomial baseOps Semantics.polynomial
      (matrixImagesAt baseOps (applicationSystem matrices) assignment
        (RowPadding.numericRowVertex logicalRows_covered row)) =
    evaluatePolynomial baseOps Semantics.polynomial
      (logicalMatrixImagesAt matrices assignment row)
  rw [applicationMatrixImagesAt_live]

/-- Every row after the logical prefix satisfies the actual application
polynomial because all matrix images are zero and its constant term is zero. -/
theorem applicationResidualAt_padding
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns)
    (vertex : BooleanVertex rowVariables)
    (padding : logicalRows <= rowIndex vertex) :
    residualAt baseOps (applicationSystem matrices) assignment vertex = 0 := by
  unfold residualAt
  rw [applicationMatrixImagesAt_padding matrices assignment vertex padding]
  exact applicationPolynomial_at_zero

/-- Zero-row padding neither adds nor removes an accepting application
assignment. -/
theorem applicationConstraintSatisfied_iff_logical
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns) :
    ConstraintSatisfied baseOps (applicationSystem matrices) assignment <->
      LogicalConstraintSatisfied matrices assignment := by
  constructor
  · intro padded row
    exact applicationResidualAt_live matrices assignment row |>.symm.trans
      (padded (RowPadding.numericRowVertex logicalRows_covered row))
  · intro logical vertex
    by_cases live : rowIndex vertex < logicalRows
    · let row : Fin logicalRows := ⟨rowIndex vertex, live⟩
      have vertexEq :
          RowPadding.numericRowVertex logicalRows_covered row = vertex := by
        unfold RowPadding.numericRowVertex
        simpa [row] using rowVertex_rowIndex vertex
      rw [← vertexEq]
      exact (applicationResidualAt_live matrices assignment row).trans
        (logical row)
    · exact applicationResidualAt_padding matrices assignment vertex
        (Nat.le_of_not_gt live)

/-- Prepending `M_0` changes no CCS residual at any Boolean row. -/
theorem identityFirstResidualAt_eq_applicationResidualAt
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns)
    (vertex : BooleanVertex rowVariables) :
    residualAt baseOps (identityFirstSystem matrices) assignment vertex =
      residualAt baseOps (applicationSystem matrices) assignment vertex := by
  unfold residualAt
  change evaluatePolynomial baseOps
      (ConstraintPolynomialPrepend.prependIgnoredVariable
        Semantics.polynomial)
      (matrixImagesAt baseOps (identityFirstSystem matrices) assignment vertex) =
    evaluatePolynomial baseOps Semantics.polynomial
      (matrixImagesAt baseOps (applicationSystem matrices) assignment vertex)
  rw [ConstraintPolynomialPrepend.evaluatePolynomial_prependIgnoredVariable
    baseOps baseLaws]
  apply congrArg (evaluatePolynomial baseOps Semantics.polynomial)
  funext matrix
  rfl

/-- Adding `M_0` and lifting the polynomial neither adds nor removes CCS
acceptance. -/
theorem identityFirstConstraintSatisfied_iff_application
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns) :
    ConstraintSatisfied baseOps (identityFirstSystem matrices) assignment <->
      ConstraintSatisfied baseOps (applicationSystem matrices) assignment := by
  constructor <;> intro satisfied vertex
  · rw [← identityFirstResidualAt_eq_applicationResidualAt]
    exact satisfied vertex
  · rw [identityFirstResidualAt_eq_applicationResidualAt]
    exact satisfied vertex

/-- Main exact relation gate: the selected identity-first, zero-padded CCS
relation accepts exactly the direct logical relation. -/
theorem identityFirstConstraintSatisfied_iff_logical
    (matrices : ApplicationMatrices)
    (assignment : Assignment F assignmentColumns) :
    ConstraintSatisfied baseOps (identityFirstSystem matrices) assignment <->
      LogicalConstraintSatisfied matrices assignment := by
  rw [identityFirstConstraintSatisfied_iff_application,
    applicationConstraintSatisfied_iff_logical]

/-- The identity matrix output is the same authoritative assignment on live
prefix rows and zero on all remaining rows. -/
theorem identityMatrixVector_eq_paddedAssignment
    (assignment : Assignment F assignmentColumns)
    (vertex : BooleanVertex rowVariables) :
    matrixVectorAt baseOps identityMatrix assignment vertex =
      assignmentLayout.paddedValue 0 assignment vertex := by
  cases decoded : assignmentLayout.toColumn? vertex with
  | none =>
      simp only [ColumnLayout.paddedValue, decoded]
      apply matrixVectorAt_zeroRow baseOps baseLaws
      intro column
      simp [identityMatrix, ColumnLayout.paddedIdentityEntry, decoded, baseOps]
  | some selected =>
      simp only [ColumnLayout.paddedValue, decoded]
      apply matrixVectorAt_identityRow baseOps baseLaws identityMatrix
        assignment vertex selected
      intro column
      simp [identityMatrix, ColumnLayout.paddedIdentityEntry, decoded, baseOps]

/-- The sole matrix source for the selected joint relation. All coefficient
matrices are derived through the Phi81 kernel. -/
def matrixSource (matrices : ApplicationMatrices) :
    MatrixSource F shape assignmentColumns
      (Phi81ColumnLayout.blockCount assignmentColumns) where
  columnLayout := Phi81ColumnLayout.layout assignmentColumns
  matrices := identityFirstMatrices matrices
  constraintPolynomial :=
    ConstraintPolynomialPrepend.prependIgnoredVariable
      Semantics.polynomial
  kernel := Phi81CoefficientKernel.phi81Kernel

/-- Connected inputs for the exact joint relation. -/
def connectedInputs
    (matrices : ApplicationMatrices)
    (assignments : Fin shape.sourceCount -> Assignment F assignmentColumns)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K) :
    ConnectedInputs K shape assignmentColumns
      (Phi81ColumnLayout.blockCount assignmentColumns) where
  cubeLayout := assignmentLayout
  matrixSource := matrixSource matrices
  assignments := assignments
  priorPoint := priorPoint
  claimedCoefficient := claimedCoefficient

/-- The connected matrix source satisfies the entrywise `M_0 = [I; 0]`
requirement from the paper strong reduction. -/
def identityFirst
    (matrices : ApplicationMatrices)
    (assignments : Fin shape.sourceCount -> Assignment F assignmentColumns)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K) :
    IdentityFirstMatrix baseOps
      (connectedInputs matrices assignments priorPoint claimedCoefficient) where
  matrixCountPositive := by decide
  entry := by
    intro vertex column
    rfl

/-- Exact norm-binding equation. The constant coefficient of the honest ring
output for `M_0` equals the MLE of the same authoritative assignment after the
canonical prefix-zero injection. No sidecar opening occurs in this statement. -/
theorem identityConstantOutput_eq_paddedAssignmentMLE
    (matrices : ApplicationMatrices)
    (assignments : Fin shape.sourceCount -> Assignment F assignmentColumns)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K)
    (point : CubePoint K rowVariables)
    (source : Fin shape.sourceCount) :
    let data :=
      connectedInputs matrices assignments priorPoint claimedCoefficient
    let identity :=
      identityFirst matrices assignments priorPoint claimedCoefficient
    (FullOutput.honestAt baseOps extensionOps K.embed data point).coordinate
        source identity.index data.matrixSource.kernel.constant =
      (BooleanTable.tabulate fun vertex =>
        K.embed (assignmentLayout.paddedValue 0
          (assignments source) vertex)).evaluate extensionOps point := by
  dsimp only
  unfold FullOutput.honestAt
  change
    (BooleanTable.tabulate fun vertex =>
      K.embed (matrixVectorAt baseOps
        ((matrixSource matrices).coefficientMatrix baseOps
          (identityFirst matrices assignments priorPoint
            claimedCoefficient).index
          (matrixSource matrices).kernel.constant)
        (assignments source) vertex)).evaluate extensionOps point =
      (BooleanTable.tabulate fun vertex =>
        K.embed (assignmentLayout.paddedValue 0
          (assignments source) vertex)).evaluate extensionOps point
  rw [(matrixSource matrices).coefficientMatrix_constant_eq baseOps baseLaws
    Phi81CoefficientKernel.phi81ConstantTermLaw
    (identityFirst matrices assignments priorPoint claimedCoefficient).index]
  apply congrArg (fun table : BooleanTable K rowVariables =>
    table.evaluate extensionOps point)
  apply congrArg (fun values : BooleanVertex rowVariables -> K =>
    BooleanTable.tabulate values)
  funext vertex
  change K.embed
      (matrixVectorAt baseOps identityMatrix (assignments source) vertex) =
    K.embed (assignmentLayout.paddedValue 0
      (assignments source) vertex)
  rw [identityMatrixVector_eq_paddedAssignment]

/-- Semantic truth stated against the direct logical CCS relation. The norm
and carried-evaluation components are unchanged from the connected joint paper
model. -/
def LogicalSemanticTruth
    (matrices : ApplicationMatrices)
    (assignments : Fin shape.sourceCount -> Assignment F assignmentColumns)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K) : Prop :=
  (forall fresh : Fin shape.freshCount,
    LogicalConstraintSatisfied matrices
      (assignments (freshSourceIndex fresh))) /\
  (forall source column,
    centeredMagnitude (assignments source column) < 2) /\
  CarriedEvaluationResidual.AllClaimsHold baseOps extensionOps K.embed
    ((connectedInputs matrices assignments priorPoint claimedCoefficient).toUnifiedInputs
      baseOps).carriedData

/-- The connected joint semantic relation is exactly the direct logical CCS
relation plus the unchanged norm and carried-evaluation obligations. This is
the bridge used to apply the generic paper strong reduction to the selected
padded relation. -/
theorem connectedSemanticTruth_iff_logicalSemanticTruth
    (matrices : ApplicationMatrices)
    (assignments : Fin shape.sourceCount -> Assignment F assignmentColumns)
    (priorPoint : CubePoint K rowVariables)
    (claimedCoefficient : CarriedCoordinate shape -> K) :
    (connectedInputs matrices assignments priorPoint claimedCoefficient).SemanticTruth
        baseOps extensionOps K.embed <->
      LogicalSemanticTruth matrices assignments priorPoint
        claimedCoefficient := by
  unfold ConnectedInputs.SemanticTruth UnifiedInputs.SemanticTruth
    LogicalSemanticTruth
  change
    ((forall fresh : Fin shape.freshCount,
        ConstraintSatisfied baseOps (identityFirstSystem matrices)
          (assignments (freshSourceIndex fresh))) /\
      (forall source column,
        centeredMagnitude (assignments source column) < 2) /\
      CarriedEvaluationResidual.AllClaimsHold baseOps extensionOps K.embed
        ((connectedInputs matrices assignments priorPoint
          claimedCoefficient).toUnifiedInputs baseOps).carriedData) <->
    ((forall fresh : Fin shape.freshCount,
        LogicalConstraintSatisfied matrices
          (assignments (freshSourceIndex fresh))) /\
      (forall source column,
        centeredMagnitude (assignments source column) < 2) /\
      CarriedEvaluationResidual.AllClaimsHold baseOps extensionOps K.embed
        ((connectedInputs matrices assignments priorPoint
          claimedCoefficient).toUnifiedInputs baseOps).carriedData)
  constructor
  · rintro ⟨ccs, norm, carried⟩
    exact ⟨fun fresh =>
      (identityFirstConstraintSatisfied_iff_logical matrices
        (assignments (freshSourceIndex fresh))).mp (ccs fresh), norm, carried⟩
  · rintro ⟨ccs, norm, carried⟩
    exact ⟨fun fresh =>
      (identityFirstConstraintSatisfied_iff_logical matrices
        (assignments (freshSourceIndex fresh))).mpr (ccs fresh), norm, carried⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
