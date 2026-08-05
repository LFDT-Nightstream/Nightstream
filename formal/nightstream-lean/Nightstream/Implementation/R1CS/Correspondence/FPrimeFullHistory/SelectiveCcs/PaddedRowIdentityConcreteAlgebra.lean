import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentity
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
import Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections

/-!
Concrete Phi81 algebra for the selected padded one-joint protocol.

Owns: the exact 24-row-variable, 11,437,038-coordinate, 14-matrix Phi81
relation shape; the 18-row Ajtai commitment carrier; the 270-coordinate public
carrier; canonicalization of an untrusted `MatrixSource` layout and kernel;
the one-entry complete evaluation-family carrier used by paper `Pi_CCS`; and
complete `PiRLC.Algebra` and `PiDEC.Algebra` values over that carrier.

Does not own: Ajtai/MSIS binding, low-norm invertibility, Poseidon2 transcript
refinement, a generated production matrix payload, Rust, R1CS, or costs.

The semantic adapter keeps the original matrices and constraint polynomial.
It replaces only the redundant column-layout and coefficient-kernel fields by
the canonical Phi81 definitions. At the selected `matrixSource`, this view is
proved equal to the paper relation used by `Pi_CCS`.
-/

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 2000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteAlgebra

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
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity

/-- The exact production relation carrier. Five public ring columns contain
the complete 270-coordinate public prefix. -/
def relationShape : Phi81Relation.Shape where
  rowVariables := rowVariables
  logicalWidth := assignmentColumns
  matrixCount := jointMatrixCount
  publicRingColumns := 5
  publicFits := by decide

/-- Production Ajtai rank from SuperNeo Appendix B.2. -/
def verifierRows : Nat := 18

theorem relationShape_carrierWidth :
    relationShape.carrierWidth = assignmentColumns := by
  decide

theorem relationShape_publicWidth : relationShape.publicWidth = 270 := by
  rfl

abbrev Structure := MatrixSource F shape assignmentColumns
  (Phi81ColumnLayout.blockCount assignmentColumns)
abbrev Assignment := Phi81Relation.Assignment relationShape
abbrev PublicInput := Phi81Relation.PublicInput relationShape
abbrev Point := CubePoint K rowVariables
abbrev Evaluation := EvaluationFamily K shape
abbrev AjtaiKey := PiRLCAlgebra.Commitment.Key relationShape verifierRows
abbrev Commitment := PiRLCAlgebra.Commitment.Value verifierRows

/-- Remove caller authority over the redundant layout and coefficient kernel.
The original matrix family and polynomial remain unchanged. -/
def canonicalStructure (source : Structure) :
    Phi81Relation.Structure relationShape where
  matrices := source.matrices
  constraintPolynomial := source.constraintPolynomial

/-- Concrete opening maps used by all three reductions. -/
def openingMaps (key : AjtaiKey) :
    OpeningMaps Commitment PublicInput assignmentColumns where
  commit := PiRLCAlgebra.Commitment.commit key
  projectPublicInput := fun assignment =>
    Phi81Relation.projectPublicInput (shape := relationShape) assignment

/-- One complete matrix/coefficient family derived through the canonical
Phi81 structure. -/
def evaluationFamily
    (source : Structure) (assignment : Assignment) (point : Point) :
    Evaluation :=
  fun matrix =>
    Phi81Relation.matrixEvaluation (canonicalStructure source) assignment
      point matrix

/-- Canonical selected relation. The CCS predicate uses the original source;
only the carried-evaluation view is normalized. -/
def semantics (key : AjtaiKey) :
    RelationSemantics Structure Assignment PublicInput Point Evaluation
      Commitment where
  commit := PiRLCAlgebra.Commitment.commit key
  projectPublicInput := fun assignment =>
    Phi81Relation.projectPublicInput (shape := relationShape) assignment
  normBounded := Phi81Relation.assignmentNormBounded
  ccsSatisfied := fun source assignment =>
    CCSResidualTable.ConstraintSatisfied baseOps source.system assignment
  evaluationPointValid := fun _ _ => True
  evaluations := fun source assignment point =>
    #[evaluationFamily source assignment point]

@[simp] theorem semantics_evaluations_size
    (key : AjtaiKey) (source : Structure)
    (assignment : Assignment) (point : Point) :
    ((semantics key).evaluations source assignment point).size = 1 := by
  rfl

/-! ## Canonical Phi81 refinement at the selected source -/

private theorem extendMatrix_eq
    (matrix : BooleanMatrix F rowVariables assignmentColumns) :
    Phi81CarrierLayout.extendMatrix 0 matrix = matrix := by
  funext vertex column
  let logical : Fin assignmentColumns := ⟨column.val, column.isLt⟩
  have columnEq : column = Phi81CarrierLayout.embedLogical logical := by
    apply Fin.ext
    rfl
  rw [columnEq, Phi81CarrierLayout.extendMatrix_embedLogical]
  rfl

theorem canonical_matrices_eq (matrices : ApplicationMatrices) :
    (canonicalStructure (matrixSource matrices)).matrixSource.matrices =
      (matrixSource matrices).matrices := by
  funext matrix
  exact extendMatrix_eq (identityFirstMatrices matrices matrix)

theorem canonical_coefficientMatrix_eq
    (matrices : ApplicationMatrices)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount) :
    (canonicalStructure (matrixSource matrices)).matrixSource.coefficientMatrix
        baseOps matrix coefficient =
      (matrixSource matrices).coefficientMatrix baseOps
        matrix coefficient := by
  funext vertex column
  unfold MatrixSource.coefficientMatrix MatrixSource.paddedMatrixEntry
  rw [canonical_matrices_eq]
  rfl

/-- Every matrix and every coefficient in the normalized relation is exactly
the paper family's carried evaluation at the selected source. -/
theorem evaluationFamily_eq_paper
    (matrices : ApplicationMatrices)
    (assignment : Assignment) (point : Point) :
    evaluationFamily (matrixSource matrices) assignment point =
      fun matrix coefficient =>
        (BooleanTable.tabulate fun vertex =>
          K.embed (matrixVectorAt baseOps
            ((matrixSource matrices).coefficientMatrix baseOps matrix
              coefficient)
            assignment vertex)).evaluate extensionOps point := by
  funext matrix coefficient
  unfold evaluationFamily Phi81Relation.matrixEvaluation
    Phi81Evaluation.evaluate Phi81Evaluation.table
  exact congrArg
    (fun coefficientMatrix : BooleanMatrix F rowVariables assignmentColumns =>
      (BooleanTable.tabulate fun vertex =>
        K.embed (matrixVectorAt baseOps coefficientMatrix assignment vertex)
      ).evaluate extensionOps point)
    (canonical_coefficientMatrix_eq matrices matrix coefficient)

/-- The canonical relation and the literal paper relation agree on the sole
verifier-owned selected matrix source. -/
theorem evaluations_eq_paper
    (key : AjtaiKey) (matrices : ApplicationMatrices)
    (assignment : Assignment) (point : Point) :
    (semantics key).evaluations (matrixSource matrices) assignment point =
      (paperRelationSemantics baseOps extensionOps K.embed (openingMaps key)).evaluations
        (matrixSource matrices) assignment point := by
  apply congrArg (fun family => #[family])
  exact evaluationFamily_eq_paper matrices assignment point

/-- Exact corrected-ambient agreement needed by the adjacent strong/weak
composition. -/
theorem ambientAgreement
    (key : AjtaiKey) (matrices : ApplicationMatrices)
    (statement : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (sourceEq : statement.constraintSystem = matrixSource matrices) :
    PiRLC.PaperCorrections.CorrectedAmbientHolds
        (paperRelationSemantics baseOps extensionOps K.embed (openingMaps key))
        productionGlobalParams statement assignment <->
      PiRLC.PaperCorrections.CorrectedAmbientHolds
        (semantics key) productionGlobalParams statement assignment := by
  unfold PiRLC.PaperCorrections.CorrectedAmbientHolds Opening.Holds
  rw [sourceEq]
  change
    (PiRLCAlgebra.Commitment.commit key assignment = statement.commitment /\
      Phi81Relation.projectPublicInput assignment = statement.publicInput /\
      Phi81Relation.assignmentNormBounded
        (PiRLC.PaperCorrections.correctedAmbientBoundFor
          productionGlobalParams) assignment) /\
      True /\
      (paperRelationSemantics baseOps extensionOps K.embed
        (openingMaps key)).evaluations (matrixSource matrices) assignment
          statement.point = statement.evaluations <->
    (PiRLCAlgebra.Commitment.commit key assignment = statement.commitment /\
      Phi81Relation.projectPublicInput assignment = statement.publicInput /\
      Phi81Relation.assignmentNormBounded
        (PiRLC.PaperCorrections.correctedAmbientBoundFor
          productionGlobalParams) assignment) /\
      True /\
      (semantics key).evaluations (matrixSource matrices) assignment
          statement.point = statement.evaluations
  rw [evaluations_eq_paper]

/-- The semantic adapter preserves every authority-bearing opening field at
every norm bound. Only the evaluation representation changes. -/
theorem openingAgreement
    (key : AjtaiKey) (normBound : Nat)
    (commitment : Commitment) (publicInput : PublicInput)
    (assignment : Assignment) :
    Opening.Holds
        (paperRelationSemantics (shape := shape)
          (blockCount := Phi81ColumnLayout.blockCount assignmentColumns)
          baseOps extensionOps K.embed (openingMaps key))
        normBound commitment publicInput assignment <->
      Opening.Holds (semantics key) normBound commitment publicInput
        assignment := by
  change
    ((PiRLCAlgebra.Commitment.commit key assignment = commitment /\
        Phi81Relation.projectPublicInput assignment = publicInput /\
        (forall column, centeredMagnitude (assignment column) < normBound)) <->
      (PiRLCAlgebra.Commitment.commit key assignment = commitment /\
        Phi81Relation.projectPublicInput assignment = publicInput /\
        (forall column, centeredMagnitude (assignment column) < normBound)))
  exact Iff.rfl

/-! ## Packed one-entry PiRLC evaluation algebra -/

/-- Zero complete family. -/
def evaluationZero : Evaluation := fun _ => BaseLinear.evaluationZero

/-- Ring-combine every matrix of one complete family. -/
def combineEvaluationFamily {count : Nat}
    (challenges : Fin count -> RingF)
    (families : Fin count -> Evaluation) : Evaluation :=
  fun matrix => PiRLCFinite.combineEvaluation challenges
    (fun source => families source matrix)

/-- Combine public evaluation arrays coordinatewise. A nonempty batch keeps
the first input's length. The verifier separately requires all source lengths
to agree. The empty batch uses the selected relation's canonical one-entry
zero family. -/
def combineEvaluations : {count : Nat} ->
    (Fin count -> RingF) ->
    (Fin count -> Array Evaluation) -> Array Evaluation
  | 0, _, _ => #[evaluationZero]
  | count + 1, challenges, items =>
      Array.ofFn fun index : Fin (items 0).size =>
        combineEvaluationFamily challenges fun source =>
          (items source).getD index.val evaluationZero

private theorem semantic_getD
    (key : AjtaiKey) (source : Structure)
    (assignment : Assignment) (point : Point) :
    ((semantics key).evaluations source assignment point).getD 0
        evaluationZero =
      evaluationFamily source assignment point := by
  rfl

theorem evaluations_combine
    (key : AjtaiKey) {count : Nat}
    (source : Structure) (point : Point)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment) :
    (semantics key).evaluations source
        (PiRLCFinite.combineAssignments challenges assignments) point =
      combineEvaluations challenges fun index =>
        (semantics key).evaluations source (assignments index) point := by
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

/-- Complete selected `Pi_RLC` algebra. -/
def piRlcAlgebra (key : AjtaiKey) :
    PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment
      RingF (semantics key) productionGlobalParams where
  challengeValid := PiRLCAlgebra.Challenge.challengeValid
  combineAssignment := PiRLCFinite.combineAssignments
  combineCommitment := PiRLCAlgebra.Commitment.combineCommitments
  combinePublicInput := PiRLCAlgebra.PublicInput.combinePublicInputs
  combineEvaluations := combineEvaluations
  commit_hom := by
    intro count challenges assignments
    exact PiRLCAlgebra.Commitment.relation_commit_hom key challenges assignments
  publicInput_hom := by
    intro count challenges assignments
    exact PiRLCAlgebra.PublicInput.relation_publicInput_hom
      (PiRLCAlgebra.Commitment.commit key) challenges assignments
  evaluations_hom := evaluations_combine key
  norm_growth := by
    intro count totalBound challenges assignments valid fresh
    exact PiRLCAlgebra.Norm.relation_norm_growth
      (PiRLCAlgebra.Commitment.commit key) totalBound challenges assignments
        valid fresh

/-! ## Packed one-entry PiDEC evaluation algebra -/

/-- Recompose every matrix of one complete family with the verifier-owned
binary radix weights. -/
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
    (key : AjtaiKey) (source : Structure) (point : Point)
    (assignments : Fin productionGlobalParams.k -> Assignment) :
    (semantics key).evaluations source
        (PiDECAlgebra.Radix.recomposeAssignment assignments) point =
      recomposeEvaluations fun child =>
        (semantics key).evaluations source (assignments child) point := by
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

/-- Complete selected `Pi_DEC` algebra. -/
def piDecAlgebra (key : AjtaiKey) :
    PiDEC.Algebra Structure Assignment PublicInput Point Evaluation Commitment
      (semantics key) productionGlobalParams where
  splitAssignment := PiDECAlgebra.Radix.splitAssignment
  recomposeAssignment := PiDECAlgebra.Radix.recomposeAssignment
  recomposeCommitment := PiDECAlgebra.Commitment.recomposeCommitment
  recomposePublicInput := PiDECAlgebra.PublicInput.recomposePublicInput
  recomposeEvaluations := recomposeEvaluations
  split_recompose := PiDECAlgebra.Radix.split_recompose
  split_norm := PiDECAlgebra.Radix.split_norm
  recompose_norm := PiDECAlgebra.Radix.recompose_norm
  commit_hom := PiDECAlgebra.Commitment.relation_commit_hom key
  publicInput_hom := PiDECAlgebra.PublicInput.relation_publicInput_hom
    (PiRLCAlgebra.Commitment.commit key)
  evaluations_hom := evaluations_recompose key

/-- Verifier-owned public-input digit split for the selected algebra. -/
def publicInputSplit (key : AjtaiKey) :
    PiDEC.PaperVerifier.PublicInputSplit (piDecAlgebra key) where
  split := fun input child =>
    PiDECAlgebra.PublicInput.splitPublicInput
      (shape := relationShape) input child
  recompose_split := by
    intro input
    exact PiDECAlgebra.PublicInput.splitPublicInput_recompose input
  split_project := by
    intro assignment child
    exact PiDECAlgebra.PublicInput.splitPublicInput_project assignment child

/-- The packed complete family is one evaluation value. -/
def evaluationArity (key : AjtaiKey) :
    PiDEC.PaperVerifier.EvaluationArity (semantics key) where
  count := fun _ => 1
  evaluations_size := fun _ _ _ => rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteAlgebra
