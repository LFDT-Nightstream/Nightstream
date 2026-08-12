import Nightstream.Implementation.NebulaV2.NIFS.Core.PaperAlgebra

/-!
Contract: paper-NIFS algebra indexed by the exact augmented-relation row
exponent.

HyperNova Construction 2 folds the augmented relation itself. The same
`rowVariables` value therefore selects the generated Phi81 relation, the
paper cube, PiCCS, PiRLC, PiDEC, and NIFS. This module contains no fixed value
of 25 or 26.

The older `ProductPaperAlgebra` module remains the fixed-25 reference model.
Field-native production code must use this module and an artifact-selected
exponent.

Assurance tier: concrete semantic algebra.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.NebulaV2.ProductPaperAlgebraFor

open Nightstream.Implementation.NebulaV2
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction

/-- Full relation shape at the exact augmented-relation exponent. -/
def fullShape (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    Phi81Relation.Shape where
  rowVariables := rowVariables
  logicalWidth := logicalWidth
  matrixCount := 14
  publicRingColumns := 10
  publicFits := by simpa [ringDegree] using publicFits

@[simp] theorem fullShape_rowVariables
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    (fullShape rowVariables logicalWidth publicFits).rowVariables =
      rowVariables := rfl

@[simp] theorem fullShape_matrixCount
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    (fullShape rowVariables logicalWidth publicFits).matrixCount = 14 := rfl

@[simp] theorem fullShape_publicWidth
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    (fullShape rowVariables logicalWidth publicFits).publicWidth = 540 := rfl

def fullShapeContract
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    ProductNifsCodec.FullShapeContractFor rowVariables
      (fullShape rowVariables logicalWidth publicFits) where
  rowVariablesExact := rfl
  matrixCount := rfl
  publicRingColumns := rfl

abbrev FullShape (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  fullShape rowVariables logicalWidth publicFits

abbrev Config (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth)
    (operationsShape snapshotShape : Phi81Relation.Shape) :=
  ProductCommitmentAlgebra.Config
    (FullShape rowVariables logicalWidth publicFits)
    operationsShape snapshotShape

abbrev Structure (rowVariables logicalWidth : Nat) :=
  MatrixSource F (ProductNifsCodec.shapeFor rowVariables)
    (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth))

abbrev Assignment (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Phi81Relation.Assignment
    (FullShape rowVariables logicalWidth publicFits)

abbrev PublicInput (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Phi81Relation.PublicInput
    (FullShape rowVariables logicalWidth publicFits)

abbrev Point (rowVariables : Nat) := CubePoint K rowVariables
abbrev Evaluation (rowVariables : Nat) :=
  EvaluationFamily K (ProductNifsCodec.shapeFor rowVariables)
abbrev Commitment := ProductCommitmentAlgebra.BundleValue

/-- Paper source derived from the exact generated matrix family. -/
def matrixSource
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)) :
    Structure rowVariables logicalWidth :=
  Phi81MatrixSource.source rowVariables 1 14 14 logicalWidth system.matrices
    system.constraintPolynomial

def canonicalStructure
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : Structure rowVariables logicalWidth) :
    Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits) where
  matrices := fun matrix vertex column =>
    source.matrices matrix vertex (Phi81CarrierLayout.embedLogical column)
  constraintPolynomial := source.constraintPolynomial

theorem canonicalStructure_matrixSource
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits)) :
    canonicalStructure (matrixSource system) = system := by
  cases system with
  | mk matrices polynomial =>
      apply congrArg₂
        (@Phi81Relation.Structure.mk
          (FullShape rowVariables logicalWidth publicFits))
      · funext matrix vertex column
        exact Phi81MatrixSource.source_matrix_embedLogical
          rowVariables 1 14 14 logicalWidth matrices polynomial matrix vertex
            column
      · rfl

def openingMaps
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape) :
    OpeningMaps Commitment
      (PublicInput rowVariables logicalWidth publicFits)
      (FullShape rowVariables logicalWidth publicFits).carrierWidth where
  commit := ProductCommitmentAlgebra.commit config
  projectPublicInput := Phi81Relation.projectPublicInput

def evaluationFamily
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : Structure rowVariables logicalWidth)
    (assignment : Assignment rowVariables logicalWidth publicFits)
    (point : Point rowVariables) : Evaluation rowVariables :=
  fun matrix =>
    Phi81Relation.matrixEvaluation (canonicalStructure source) assignment
      point matrix

def semantics
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape) :
    RelationSemantics (Structure rowVariables logicalWidth)
      (Assignment rowVariables logicalWidth publicFits)
      (PublicInput rowVariables logicalWidth publicFits)
      (Point rowVariables) (Evaluation rowVariables) Commitment where
  commit := ProductCommitmentAlgebra.commit config
  projectPublicInput := Phi81Relation.projectPublicInput
  normBounded := Phi81Relation.assignmentNormBounded
  ccsSatisfied := fun source assignment =>
    CCSResidualTable.ConstraintSatisfied baseOps source.system assignment
  evaluationPointValid := fun _ _ => True
  evaluations := fun source assignment point =>
    #[evaluationFamily source assignment point]

@[simp] theorem semantics_evaluations_size
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape)
    (source : Structure rowVariables logicalWidth)
    (assignment : Assignment rowVariables logicalWidth publicFits)
    (point : Point rowVariables) :
    ((semantics config).evaluations source assignment point).size = 1 := rfl

theorem evaluationFamily_eq_paper
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits))
    (assignment : Assignment rowVariables logicalWidth publicFits)
    (point : Point rowVariables) :
    evaluationFamily (matrixSource system) assignment point =
      fun matrix coefficient =>
        (BooleanTable.tabulate fun vertex =>
          K.embed (matrixVectorAt baseOps
            ((matrixSource system).coefficientMatrix baseOps matrix
              coefficient)
            assignment vertex)).evaluate extensionOps point := by
  unfold evaluationFamily
  rw [canonicalStructure_matrixSource]
  funext matrix coefficient
  rfl

theorem evaluations_eq_paper
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape)
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits))
    (assignment : Assignment rowVariables logicalWidth publicFits)
    (point : Point rowVariables) :
    (semantics config).evaluations (matrixSource system) assignment point =
      (paperRelationSemantics baseOps extensionOps K.embed
        (shape := ProductNifsCodec.shapeFor rowVariables)
        (blockCount := Phi81ColumnLayout.blockCount
          (Phi81CarrierLayout.carrierWidth logicalWidth))
        (openingMaps config)).evaluations
          (matrixSource system) assignment point := by
  apply congrArg (fun family => #[family])
  exact evaluationFamily_eq_paper system assignment point

theorem ambientAgreement
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape)
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits))
    (statement : CE.Instance (Structure rowVariables logicalWidth)
      (PublicInput rowVariables logicalWidth publicFits)
      (Point rowVariables) (Evaluation rowVariables) Commitment)
    (assignment : Assignment rowVariables logicalWidth publicFits)
    (sourceEq : statement.constraintSystem = matrixSource system) :
    PiRLC.PaperCorrections.CorrectedAmbientHolds
        (paperRelationSemantics
          (shape := ProductNifsCodec.shapeFor rowVariables)
          (blockCount := Phi81ColumnLayout.blockCount
            (Phi81CarrierLayout.carrierWidth logicalWidth))
          baseOps extensionOps K.embed (openingMaps config))
        productionGlobalParams statement assignment <->
      PiRLC.PaperCorrections.CorrectedAmbientHolds
        (semantics config) productionGlobalParams statement assignment := by
  unfold PiRLC.PaperCorrections.CorrectedAmbientHolds Opening.Holds
  rw [sourceEq]
  change
    (ProductCommitmentAlgebra.commit config assignment = statement.commitment /\
      Phi81Relation.projectPublicInput assignment = statement.publicInput /\
      Phi81Relation.assignmentNormBounded
        (PiRLC.PaperCorrections.correctedAmbientBoundFor
          productionGlobalParams) assignment) /\
      True /\
      (paperRelationSemantics
        (shape := ProductNifsCodec.shapeFor rowVariables)
        (blockCount := Phi81ColumnLayout.blockCount
          (Phi81CarrierLayout.carrierWidth logicalWidth))
        baseOps extensionOps K.embed
        (openingMaps config)).evaluations (matrixSource system) assignment
          statement.point = statement.evaluations <->
    (ProductCommitmentAlgebra.commit config assignment = statement.commitment /\
      Phi81Relation.projectPublicInput assignment = statement.publicInput /\
      Phi81Relation.assignmentNormBounded
        (PiRLC.PaperCorrections.correctedAmbientBoundFor
          productionGlobalParams) assignment) /\
      True /\
      (semantics config).evaluations (matrixSource system) assignment
        statement.point = statement.evaluations
  rw [evaluations_eq_paper]

theorem openingAgreement
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape)
    (normBound : Nat) (commitment : Commitment)
    (publicInput : PublicInput rowVariables logicalWidth publicFits)
    (assignment : Assignment rowVariables logicalWidth publicFits) :
    Opening.Holds
        (paperRelationSemantics
          (shape := ProductNifsCodec.shapeFor rowVariables)
          (blockCount := Phi81ColumnLayout.blockCount
            (Phi81CarrierLayout.carrierWidth logicalWidth))
          baseOps extensionOps K.embed (openingMaps config))
        normBound commitment publicInput assignment <->
      Opening.Holds (semantics config) normBound commitment publicInput
        assignment := by
  change
    ((ProductCommitmentAlgebra.commit config assignment = commitment /\
        Phi81Relation.projectPublicInput assignment = publicInput /\
        (forall column,
          centeredMagnitude (assignment column) < normBound)) <->
      (ProductCommitmentAlgebra.commit config assignment = commitment /\
        Phi81Relation.projectPublicInput assignment = publicInput /\
        (forall column,
          centeredMagnitude (assignment column) < normBound)))
  exact Iff.rfl

/-! ## PiRLC algebra -/

def evaluationZero (rowVariables : Nat) : Evaluation rowVariables :=
  fun _ => BaseLinear.evaluationZero

def combineEvaluationFamily
    {rowVariables count : Nat}
    (challenges : Fin count -> RingF)
    (families : Fin count -> Evaluation rowVariables) :
    Evaluation rowVariables :=
  fun matrix => PiRLCFinite.combineEvaluation challenges
    (fun source => families source matrix)

def combineEvaluations (rowVariables : Nat) : {count : Nat} ->
    (Fin count -> RingF) ->
    (Fin count -> Array (Evaluation rowVariables)) ->
      Array (Evaluation rowVariables)
  | 0, _, _ => #[evaluationZero rowVariables]
  | count + 1, challenges, items =>
      Array.ofFn fun index : Fin (items 0).size =>
        combineEvaluationFamily challenges fun source =>
          (items source).getD index.val (evaluationZero rowVariables)

theorem evaluations_combine
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape)
    {count : Nat} (source : Structure rowVariables logicalWidth)
    (point : Point rowVariables) (challenges : Fin count -> RingF)
    (assignments : Fin count ->
      Assignment rowVariables logicalWidth publicFits) :
    (semantics config).evaluations source
        (PiRLCFinite.combineAssignments challenges assignments) point =
      combineEvaluations rowVariables challenges fun index =>
        (semantics config).evaluations source (assignments index) point := by
  cases count with
  | zero =>
      apply Array.ext
      · rfl
      · intro index leftLt rightLt
        have indexZero : index = 0 := by
          have indexLt : index < 1 := by
            simpa [semantics] using leftLt
          omega
        subst index
        change evaluationFamily source
            (PiRLCFinite.combineAssignments challenges assignments) point =
          evaluationZero rowVariables
        funext matrix
        exact BaseLinear.matrixEvaluation_zero
          (canonicalStructure source) point matrix
  | succ count =>
      apply Array.ext
      · simp [combineEvaluations, semantics]
      · intro index leftLt rightLt
        have indexZero : index = 0 := by
          have indexLt : index < 1 := by
            simpa [semantics] using leftLt
          omega
        subst index
        change
          evaluationFamily source
              (PiRLCFinite.combineAssignments challenges assignments) point =
            combineEvaluationFamily challenges fun index =>
              evaluationFamily source (assignments index) point
        funext matrix
        exact PiRLCFinite.matrixEvaluation_combine
          (canonicalStructure source) challenges assignments point matrix

def piRlcAlgebra
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape) :
    PiRLC.Algebra (Structure rowVariables logicalWidth)
      (Assignment rowVariables logicalWidth publicFits)
      (PublicInput rowVariables logicalWidth publicFits)
      (Point rowVariables) (Evaluation rowVariables) Commitment RingF
      (semantics config) productionGlobalParams where
  challengeValid := PiRLCAlgebra.Challenge.challengeValid
  combineAssignment := PiRLCFinite.combineAssignments
  combineCommitment := ProductCommitmentAlgebra.combineBundles
  combinePublicInput := PiRLCAlgebra.PublicInput.combinePublicInputs
  combineEvaluations := combineEvaluations rowVariables
  commit_hom := by
    intro count challenges assignments
    exact ProductCommitmentAlgebra.commit_combine config challenges assignments
  publicInput_hom := by
    intro count challenges assignments
    exact PiRLCAlgebra.PublicInput.relation_publicInput_hom
      (ProductCommitmentAlgebra.commit config) challenges assignments
  evaluations_hom := evaluations_combine config
  norm_growth := by
    intro count totalBound challenges assignments valid fresh
    exact PiRLCAlgebra.Norm.relation_norm_growth
      (ProductCommitmentAlgebra.commit config) totalBound challenges assignments
        valid fresh

/-! ## PiDEC algebra -/

def recomposeEvaluationFamily
    {rowVariables : Nat}
    (families : Fin productionGlobalParams.k -> Evaluation rowVariables) :
    Evaluation rowVariables :=
  fun matrix => BaseLinear.combineEvaluations
    EvaluationHomomorphism.PiDEC.radixWeight
    (fun child => families child matrix)

def recomposeEvaluations
    {rowVariables : Nat}
    (items : Fin productionGlobalParams.k ->
      Array (Evaluation rowVariables)) :
    Array (Evaluation rowVariables) :=
  #[recomposeEvaluationFamily
    (fun child => (items child).getD 0 (evaluationZero rowVariables))]

theorem evaluations_recompose
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape)
    (source : Structure rowVariables logicalWidth)
    (point : Point rowVariables)
    (assignments : Fin productionGlobalParams.k ->
      Assignment rowVariables logicalWidth publicFits) :
    (semantics config).evaluations source
        (PiDECAlgebra.Radix.recomposeAssignment assignments) point =
      recomposeEvaluations fun child =>
        (semantics config).evaluations source (assignments child) point := by
  apply Array.ext
  · rfl
  · intro index leftLt rightLt
    have indexLt : index < 1 := by
      simpa [semantics] using leftLt
    have indexZero : index = 0 := by omega
    subst index
    change
      evaluationFamily source
          (PiDECAlgebra.Radix.recomposeAssignment assignments) point =
        recomposeEvaluationFamily fun child =>
          evaluationFamily source (assignments child) point
    funext matrix
    exact BaseLinear.matrixEvaluation_combine
      (canonicalStructure source) EvaluationHomomorphism.PiDEC.radixWeight
      assignments point matrix

def piDecAlgebra
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape) :
    PiDEC.Algebra (Structure rowVariables logicalWidth)
      (Assignment rowVariables logicalWidth publicFits)
      (PublicInput rowVariables logicalWidth publicFits)
      (Point rowVariables) (Evaluation rowVariables) Commitment
      (semantics config) productionGlobalParams where
  splitAssignment := PiDECAlgebra.Radix.splitAssignment
  recomposeAssignment := PiDECAlgebra.Radix.recomposeAssignment
  recomposeCommitment := ProductCommitmentAlgebra.recomposeBundles
  recomposePublicInput := PiDECAlgebra.PublicInput.recomposePublicInput
  recomposeEvaluations := recomposeEvaluations
  split_recompose := PiDECAlgebra.Radix.split_recompose
  split_norm := PiDECAlgebra.Radix.split_norm
  recompose_norm := PiDECAlgebra.Radix.recompose_norm
  commit_hom := ProductCommitmentAlgebra.commit_recompose config
  publicInput_hom := PiDECAlgebra.PublicInput.relation_publicInput_hom
    (ProductCommitmentAlgebra.commit config)
  evaluations_hom := evaluations_recompose config

def publicInputSplit
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape) :
    PiDEC.PaperVerifier.PublicInputSplit (piDecAlgebra config) where
  split := fun input child =>
    PiDECAlgebra.PublicInput.splitPublicInput
      (shape := FullShape rowVariables logicalWidth publicFits) input child
  recompose_split := by
    intro input
    exact PiDECAlgebra.PublicInput.splitPublicInput_recompose input
  split_project := by
    intro assignment child
    exact PiDECAlgebra.PublicInput.splitPublicInput_project assignment child

/-- The production public split is the coordinatewise radix split. This
projection lemma keeps downstream verifier-output proofs from unfolding the
complete NIFS key. -/
@[simp] theorem publicInputSplit_coordinate
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape)
    (input : PublicInput rowVariables logicalWidth publicFits)
    (child : Fin productionGlobalParams.k)
    (column : Fin (FullShape rowVariables logicalWidth publicFits).publicWidth) :
    (publicInputSplit config).split input child column =
      PiDECAlgebra.Radix.splitScalar (input column) child := by
  rfl

def evaluationArity
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config rowVariables logicalWidth publicFits operationsShape
      snapshotShape) :
    PiDEC.PaperVerifier.EvaluationArity (semantics config) where
  count := fun _ => 1
  evaluations_size := fun _ _ _ => rfl

end Nightstream.Implementation.NebulaV2.ProductPaperAlgebraFor
