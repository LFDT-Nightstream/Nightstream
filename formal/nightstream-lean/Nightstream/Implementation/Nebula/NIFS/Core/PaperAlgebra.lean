import Nightstream.Implementation.Nebula.NIFS.Core.Poseidon2
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction

/-!
Contract: exact paper-NIFS algebra for the V2 product commitment.

Assurance tier: concrete semantic algebra.

Owns the 25-variable, fourteen-matrix, ten-public-ring full relation shape;
the canonical paper matrix source derived from one original relation; the
four-component opening map; the packed paper evaluation adapter; and its
agreement with the independent paper relation at the selected source.

Does not own the generated application relation, transcript rows, NIFS rows,
Module-SIS binding, Rust, or the deployed verifier.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.Nebula.ProductPaperAlgebra

open Nightstream.Implementation.Nebula
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

/-- V2 fixes all folding dimensions. Only the generated logical assignment
width remains an input to key generation. -/
def fullShape (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    Phi81Relation.Shape where
  rowVariables := 25
  logicalWidth := logicalWidth
  matrixCount := 14
  publicRingColumns := 10
  publicFits := by simpa [ringDegree] using publicFits

@[simp] theorem fullShape_rowVariables
    (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    (fullShape logicalWidth publicFits).rowVariables = 25 := rfl

@[simp] theorem fullShape_matrixCount
    (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    (fullShape logicalWidth publicFits).matrixCount = 14 := rfl

@[simp] theorem fullShape_publicWidth
    (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    (fullShape logicalWidth publicFits).publicWidth = 540 := rfl

def fullShapeContract
    (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    ProductNifsCodec.FullShapeContract (fullShape logicalWidth publicFits) where
  rowVariables := rfl
  matrixCount := rfl
  publicRingColumns := rfl

abbrev FullShape (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  fullShape logicalWidth publicFits

abbrev Config (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth)
    (operationsShape snapshotShape : Phi81Relation.Shape) :=
  ProductCommitmentAlgebra.Config
    (FullShape logicalWidth publicFits) operationsShape snapshotShape

abbrev Structure (logicalWidth : Nat) :=
  MatrixSource F ProductNifsCodec.shape
    (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth))

abbrev Assignment (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Phi81Relation.Assignment (FullShape logicalWidth publicFits)

abbrev PublicInput (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Phi81Relation.PublicInput (FullShape logicalWidth publicFits)

abbrev Point := CubePoint K ProductNifsCodec.shape.cubeVariables
abbrev Evaluation := EvaluationFamily K ProductNifsCodec.shape
abbrev Commitment := ProductCommitmentAlgebra.BundleValue

/-- The paper source is derived from the original generated matrix family.
Its fresh and running counts are type-level protocol dimensions only. -/
def matrixSource {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (system : Phi81Relation.Structure (FullShape logicalWidth publicFits)) :
    Structure logicalWidth :=
  Phi81MatrixSource.source 25 1 14 14 logicalWidth system.matrices
    system.constraintPolynomial

/-- Recover the original logical matrix family from a complete paper source.
This normalization makes the semantic algebra total on arbitrary source
values. At the selected source it is exactly the original system. -/
def canonicalStructure {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : Structure logicalWidth) :
    Phi81Relation.Structure (FullShape logicalWidth publicFits) where
  matrices := fun matrix vertex column =>
    source.matrices matrix vertex (Phi81CarrierLayout.embedLogical column)
  constraintPolynomial := source.constraintPolynomial

theorem canonicalStructure_matrixSource
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (system : Phi81Relation.Structure (FullShape logicalWidth publicFits)) :
    canonicalStructure (matrixSource system) = system := by
  cases system with
  | mk matrices polynomial =>
      apply congrArg₂
        (@Phi81Relation.Structure.mk (FullShape logicalWidth publicFits))
      · funext matrix vertex column
        exact Phi81MatrixSource.source_matrix_embedLogical
          25 1 14 14 logicalWidth matrices polynomial matrix vertex column
      · rfl

/-- Four-component paper opening maps. One assignment is the authority for
the full, operations, initial-snapshot, and final-snapshot commitments. -/
def openingMaps {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape) :
    OpeningMaps Commitment (PublicInput logicalWidth publicFits)
      (FullShape logicalWidth publicFits).carrierWidth where
  commit := ProductCommitmentAlgebra.commit config
  projectPublicInput := Phi81Relation.projectPublicInput

/-- One packed matrix/coefficient family derived through the normalized
Phi81 relation. -/
def evaluationFamily
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (source : Structure logicalWidth)
    (assignment : Assignment logicalWidth publicFits) (point : Point) :
    Evaluation :=
  fun matrix =>
    Phi81Relation.matrixEvaluation (canonicalStructure source) assignment
      point matrix

/-- Exact paper-carrier semantics used by PiRLC and PiDEC. -/
def semantics {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape) :
    RelationSemantics (Structure logicalWidth)
      (Assignment logicalWidth publicFits) (PublicInput logicalWidth publicFits)
      Point Evaluation
      Commitment where
  commit := ProductCommitmentAlgebra.commit config
  projectPublicInput := Phi81Relation.projectPublicInput
  normBounded := Phi81Relation.assignmentNormBounded
  ccsSatisfied := fun source assignment =>
    CCSResidualTable.ConstraintSatisfied baseOps source.system assignment
  evaluationPointValid := fun _ _ => True
  evaluations := fun source assignment point =>
    #[evaluationFamily source assignment point]

@[simp] theorem semantics_evaluations_size
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape)
    (source : Structure logicalWidth)
    (assignment : Assignment logicalWidth publicFits) (point : Point) :
    ((semantics config).evaluations source assignment point).size = 1 := rfl

/-- At the selected source, the normalized packed family is exactly the
independent paper formula. -/
theorem evaluationFamily_eq_paper
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (system : Phi81Relation.Structure (FullShape logicalWidth publicFits))
    (assignment : Assignment logicalWidth publicFits) (point : Point) :
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

/-- The semantic adapter and literal paper relation agree at the one
verifier-owned generated source. -/
theorem evaluations_eq_paper
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape)
    (system : Phi81Relation.Structure (FullShape logicalWidth publicFits))
    (assignment : Assignment logicalWidth publicFits) (point : Point) :
    (semantics config).evaluations (matrixSource system) assignment point =
      (paperRelationSemantics baseOps extensionOps K.embed
        (shape := ProductNifsCodec.shape)
        (blockCount := Phi81ColumnLayout.blockCount
          (Phi81CarrierLayout.carrierWidth logicalWidth))
        (openingMaps config)).evaluations
          (matrixSource system) assignment point := by
  apply congrArg (fun family => #[family])
  exact evaluationFamily_eq_paper system assignment point

/-- Exact corrected-ambient agreement at the verifier-owned generated
matrix source. This theorem changes only the representation of the complete
evaluation family. -/
theorem ambientAgreement
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape)
    (system : Phi81Relation.Structure (FullShape logicalWidth publicFits))
    (statement : CE.Instance (Structure logicalWidth)
      (PublicInput logicalWidth publicFits) Point Evaluation Commitment)
    (assignment : Assignment logicalWidth publicFits)
    (sourceEq : statement.constraintSystem = matrixSource system) :
    PiRLC.PaperCorrections.CorrectedAmbientHolds
        (paperRelationSemantics
          (shape := ProductNifsCodec.shape)
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
        (shape := ProductNifsCodec.shape)
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

/-- The authority-bearing opening fields are identical at every norm bound.
Only the packed evaluation representation differs. -/
theorem openingAgreement
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape)
    (normBound : Nat)
    (commitment : Commitment)
    (publicInput : PublicInput logicalWidth publicFits)
    (assignment : Assignment logicalWidth publicFits) :
    Opening.Holds
        (paperRelationSemantics
          (shape := ProductNifsCodec.shape)
          (blockCount := Phi81ColumnLayout.blockCount
            (Phi81CarrierLayout.carrierWidth logicalWidth))
          baseOps extensionOps K.embed
          (openingMaps config))
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

/-! ## Complete one-entry PiRLC evaluation algebra -/

/-- The canonical zero value for one complete matrix/coefficient family. -/
def evaluationZero : Evaluation := fun _ => BaseLinear.evaluationZero

/-- Apply one PiRLC challenge vector to every matrix in a complete family. -/
def combineEvaluationFamily {count : Nat}
    (challenges : Fin count -> RingF)
    (families : Fin count -> Evaluation) : Evaluation :=
  fun matrix => PiRLCFinite.combineEvaluation challenges
    (fun source => families source matrix)

/-- Combine the one-entry public evaluation arrays. The verifier checks the
source arities. The empty batch has the unique one-entry zero family. -/
def combineEvaluations : {count : Nat} ->
    (Fin count -> RingF) ->
    (Fin count -> Array Evaluation) -> Array Evaluation
  | 0, _, _ => #[evaluationZero]
  | count + 1, challenges, items =>
      Array.ofFn fun index : Fin (items 0).size =>
        combineEvaluationFamily challenges fun source =>
          (items source).getD index.val evaluationZero

theorem evaluations_combine
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape)
    {count : Nat} (source : Structure logicalWidth) (point : Point)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment logicalWidth publicFits) :
    (semantics config).evaluations source
        (PiRLCFinite.combineAssignments challenges assignments) point =
      combineEvaluations challenges fun index =>
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
          evaluationZero
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

/-- Complete V2 PiRLC algebra. Every bundle component uses the same
challenge vector and the same source order. -/
def piRlcAlgebra
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape) :
    PiRLC.Algebra (Structure logicalWidth)
      (Assignment logicalWidth publicFits) (PublicInput logicalWidth publicFits)
      Point Evaluation Commitment RingF (semantics config)
      productionGlobalParams where
  challengeValid := PiRLCAlgebra.Challenge.challengeValid
  combineAssignment := PiRLCFinite.combineAssignments
  combineCommitment := ProductCommitmentAlgebra.combineBundles
  combinePublicInput := PiRLCAlgebra.PublicInput.combinePublicInputs
  combineEvaluations := combineEvaluations
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

/-! ## Complete one-entry PiDEC evaluation algebra -/

/-- Recompose every matrix family with the verifier-owned binary weights. -/
def recomposeEvaluationFamily
    (families : Fin productionGlobalParams.k -> Evaluation) : Evaluation :=
  fun matrix => BaseLinear.combineEvaluations
    EvaluationHomomorphism.PiDEC.radixWeight
    (fun child => families child matrix)

/-- Recompose the one-entry public evaluation arrays. -/
def recomposeEvaluations
    (items : Fin productionGlobalParams.k -> Array Evaluation) :
    Array Evaluation :=
  #[recomposeEvaluationFamily
    (fun child => (items child).getD 0 evaluationZero)]

theorem evaluations_recompose
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape)
    (source : Structure logicalWidth) (point : Point)
    (assignments : Fin productionGlobalParams.k ->
      Assignment logicalWidth publicFits) :
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

/-- Complete V2 PiDEC algebra. Every child recomposes all four commitment
components against the same fourteen child assignments. -/
def piDecAlgebra
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape) :
    PiDEC.Algebra (Structure logicalWidth)
      (Assignment logicalWidth publicFits) (PublicInput logicalWidth publicFits)
      Point Evaluation Commitment (semantics config)
      productionGlobalParams where
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

/-- Verifier-owned public-input digit split for the exact V2 algebra. -/
def publicInputSplit
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape) :
    PiDEC.PaperVerifier.PublicInputSplit (piDecAlgebra config) where
  split := fun input child =>
    PiDECAlgebra.PublicInput.splitPublicInput
      (shape := FullShape logicalWidth publicFits) input child
  recompose_split := by
    intro input
    exact PiDECAlgebra.PublicInput.splitPublicInput_recompose input
  split_project := by
    intro assignment child
    exact PiDECAlgebra.PublicInput.splitPublicInput_project assignment child

/-- The packed complete family has exactly one evaluation value. -/
def evaluationArity
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config logicalWidth publicFits operationsShape snapshotShape) :
    PiDEC.PaperVerifier.EvaluationArity (semantics config) where
  count := fun _ => 1
  evaluations_size := fun _ _ _ => rfl

end Nightstream.Implementation.Nebula.ProductPaperAlgebra
