import NightstreamFPrime.Lifecycle.Types

/-!
Owns the exact Stage 1 paper-NIFS algebra over the single Ajtai commitment:
the Φ₈₁ relation shape at the F′ row domain, the canonical paper matrix source
derived from one logical structure, the opening maps, the packed evaluation
adapter and its agreement with the independent paper relation, and the
complete Π_RLC and Π_DEC algebras the NIFS key consumes. The logical width is
the one open circuit parameter (closed by the recursive fixed point).

Provenance: adapted from
`formal/nightstream-lean/Nightstream/Implementation/Nebula/NIFS/Core/PaperAlgebra.lean`
at commit `f277c1d5e16b9f0d096d9b9da30baeb932af9be8`: the four-lane product commitment is replaced by the Ajtai
  commitment of `Spec.Phi81Relation.PiRLCAlgebra.Commitment`, the row domain is
  the Stage 1 `cubeVariables`, the public block is five ring columns, and the
SuperNeo v1.1 Pad evaluation family is separate from all CCS matrices.
-/

namespace NightstreamFPrime.Lifecycle.PaperAlgebra

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Lifecycle

/-- Five ring columns hold one marker, four canonical 64-bit digest encodings,
and a zero tail. Every nonzero coordinate has centered magnitude one. -/
def publicRingColumns : Nat := 5

/-- The Φ₈₁ relation shape of the Stage 1 F′ circuit at logical width
`logicalWidth`. All folding dimensions are fixed; only the width is open. -/
def fullShape (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    Phi81Relation.Shape where
  rowVariables := cubeVariables
  logicalWidth := logicalWidth
  matrixCount := productionProfile.ccsMatrices
  publicRingColumns := publicRingColumns
  publicFits := publicFits

abbrev FullShape (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  fullShape logicalWidth publicFits

abbrev MatrixStructure (logicalWidth : Nat) :=
  MatrixSource F productionShape
    (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth))

abbrev Structure (logicalWidth : Nat) :=
  RelationSource productionShape
    (Phi81CarrierLayout.carrierWidth logicalWidth)
    (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth))

section

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns <=
    Phi81CarrierLayout.carrierWidth logicalWidth}

abbrev Assignment := Phi81Relation.Assignment (FullShape logicalWidth publicFits)
abbrev PublicInput := Phi81Relation.PublicInput (FullShape logicalWidth publicFits)
abbrev Point := CubePoint K productionShape.cubeVariables
abbrev Evaluation := EvaluationFamily K productionShape
abbrev AjtaiKey := PiRLCAlgebra.Commitment.Key (FullShape logicalWidth publicFits)
  productionProfile.commitmentWidth
abbrev Commitment := PiRLCAlgebra.Commitment.Value productionProfile.commitmentWidth

private theorem evaluation_ext
    (left right : Evaluation)
    (pad : left.pad = right.pad)
    (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

/-- The paper source derived from the logical matrix family. -/
def matrixSource
    (system : Phi81Relation.Structure (FullShape logicalWidth publicFits)) :
    MatrixStructure logicalWidth :=
  Phi81MatrixSource.source cubeVariables productionProfile.freshSources
    productionProfile.runningSources productionProfile.ccsMatrices logicalWidth
    system.matrices system.constraintPolynomial

/-- Complete v1.1 source: canonical Pad layout plus all 14 CCS matrices. -/
def relationSource
    (layout : ColumnLayout productionShape.cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth))
    (system : Phi81Relation.Structure (FullShape logicalWidth publicFits)) :
    Structure logicalWidth where
  cubeLayout := layout
  matrixSource := matrixSource system

/-- Recover the logical family from a complete paper source; total on every
source, exact at the selected one. -/
def canonicalStructure (source : Structure logicalWidth) :
    Phi81Relation.Structure (FullShape logicalWidth publicFits) where
  matrices := fun matrix vertex column =>
    source.matrixSource.matrices matrix vertex
      (Phi81CarrierLayout.embedLogical column)
  constraintPolynomial := source.matrixSource.constraintPolynomial

theorem canonicalStructure_relationSource
    (layout : ColumnLayout productionShape.cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth))
    (system : Phi81Relation.Structure (FullShape logicalWidth publicFits)) :
    canonicalStructure (publicFits := publicFits)
      (relationSource layout system) = system := by
  cases system with
  | mk matrices polynomial =>
      apply congrArg₂ (@Phi81Relation.Structure.mk (FullShape logicalWidth publicFits))
      · funext matrix vertex column
        exact Phi81MatrixSource.source_matrix_embedLogical
          cubeVariables productionProfile.freshSources productionProfile.runningSources
          productionProfile.ccsMatrices logicalWidth matrices polynomial matrix vertex column
      · rfl

/-- The completed Pad matrix owned by the v1.1 relation source. Pad is not a
member of the CCS matrix family. -/
def padMatrix (source : Structure logicalWidth) :
    BooleanMatrix F productionShape.cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth) :=
  fun vertex column =>
    source.cubeLayout.paddedIdentityEntry
      baseOps.zero baseOps.one vertex column

/-- The independent v1.1 `Eval_K` family for Pad. -/
def padEvaluation
    (source : Structure logicalWidth)
    (assignment : Assignment (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    (point : Point) : Phi81Relation.Evaluation :=
  EvaluationHomomorphism.PiRLC.ExplicitMatrix.evaluate
    (canonicalStructure (publicFits := publicFits) source)
    (padMatrix source) assignment point

/-- Ajtai opening maps: one assignment opens one commitment. -/
def openingMaps (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    OpeningMaps Commitment (PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      (FullShape logicalWidth publicFits).carrierWidth where
  commit := PiRLCAlgebra.Commitment.commit key
  projectPublicInput := Phi81Relation.projectPublicInput

/-- One packed matrix/coefficient family through the normalized relation. -/
def evaluationFamily (source : Structure logicalWidth)
    (assignment : Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : Point) : Evaluation where
  pad := padEvaluation (publicFits := publicFits) source assignment point
  matrix := fun matrix =>
    Phi81Relation.matrixEvaluation
      (canonicalStructure (publicFits := publicFits) source)
      assignment point matrix

/-- Exact paper-carrier semantics used by Π_RLC and Π_DEC. -/
def semantics (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    RelationSemantics (Structure logicalWidth)
      (Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
      (PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      Point Evaluation Commitment where
  commit := PiRLCAlgebra.Commitment.commit key
  projectPublicInput := Phi81Relation.projectPublicInput
  normBounded := Phi81Relation.assignmentNormBounded
  ccsSatisfied := fun source assignment =>
    CCSResidualTable.ConstraintSatisfied baseOps
      source.matrixSource.system assignment
  evaluationPointValid := fun _ _ => True
  evaluations := fun source assignment point =>
    #[evaluationFamily (publicFits := publicFits) source assignment point]

@[simp] theorem semantics_evaluations_size
    (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Structure logicalWidth)
    (assignment : Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : Point) :
    ((semantics key).evaluations source assignment point).size = 1 := rfl

theorem evaluationFamily_eq_paper
    (layout : ColumnLayout productionShape.cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth))
    (system : Phi81Relation.Structure (FullShape logicalWidth publicFits))
    (assignment : Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : Point) :
    evaluationFamily (publicFits := publicFits)
        (relationSource layout system) assignment point = {
      pad := fun coefficient =>
        (BooleanTable.tabulate fun vertex =>
          K.embed (matrixVectorAt baseOps
            ((matrixSource system).coefficientMatrixOf baseOps
              (fun row column =>
                layout.paddedIdentityEntry
                  baseOps.zero baseOps.one row column)
              coefficient)
            assignment vertex)).evaluate extensionOps point
      matrix := fun matrix coefficient =>
        (BooleanTable.tabulate fun vertex =>
          K.embed (matrixVectorAt baseOps
            ((matrixSource system).coefficientMatrix baseOps matrix coefficient)
            assignment vertex)).evaluate extensionOps point
    } := by
  apply evaluation_ext
  · funext coefficient
    unfold evaluationFamily padEvaluation padMatrix
    rw [canonicalStructure_relationSource]
    rfl
  · funext matrix coefficient
    unfold evaluationFamily
    rw [canonicalStructure_relationSource]
    rfl

theorem evaluations_eq_paper
    (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (layout : ColumnLayout productionShape.cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth))
    (system : Phi81Relation.Structure (FullShape logicalWidth publicFits))
    (assignment : Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : Point) :
    (semantics key).evaluations (relationSource layout system) assignment point =
      (paperRelationSemantics baseOps extensionOps K.embed
        (shape := productionShape)
        (blockCount := Phi81ColumnLayout.blockCount
          (Phi81CarrierLayout.carrierWidth logicalWidth))
        (openingMaps key)).evaluations
          (relationSource layout system) assignment point := by
  apply congrArg (fun family => #[family])
  exact evaluationFamily_eq_paper layout system assignment point

theorem ambientAgreement
    (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (layout : ColumnLayout productionShape.cubeVariables
      (Phi81CarrierLayout.carrierWidth logicalWidth))
    (system : Phi81Relation.Structure (FullShape logicalWidth publicFits))
    (statement : CE.Instance (Structure logicalWidth)
      (PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      Point Evaluation Commitment)
    (assignment : Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (sourceEq : statement.constraintSystem = relationSource layout system) :
    PiRLC.PaperCorrections.CorrectedAmbientHolds
        (paperRelationSemantics (shape := productionShape)
          (blockCount := Phi81ColumnLayout.blockCount
            (Phi81CarrierLayout.carrierWidth logicalWidth))
          baseOps extensionOps K.embed (openingMaps key))
        productionGlobalParams statement assignment <->
      PiRLC.PaperCorrections.CorrectedAmbientHolds
        (semantics key) productionGlobalParams statement assignment := by
  unfold PiRLC.PaperCorrections.CorrectedAmbientHolds Opening.Holds
  rw [sourceEq]
  change
    (PiRLCAlgebra.Commitment.commit key assignment = statement.commitment /\
      Phi81Relation.projectPublicInput assignment = statement.publicInput /\
      Phi81Relation.assignmentNormBounded
        (PiRLC.PaperCorrections.correctedAmbientBoundFor productionGlobalParams)
        assignment) /\
      True /\
      (paperRelationSemantics (shape := productionShape)
        (blockCount := Phi81ColumnLayout.blockCount
          (Phi81CarrierLayout.carrierWidth logicalWidth))
        baseOps extensionOps K.embed (openingMaps key)).evaluations
          (relationSource layout system) assignment statement.point =
            statement.evaluations <->
    (PiRLCAlgebra.Commitment.commit key assignment = statement.commitment /\
      Phi81Relation.projectPublicInput assignment = statement.publicInput /\
      Phi81Relation.assignmentNormBounded
        (PiRLC.PaperCorrections.correctedAmbientBoundFor productionGlobalParams)
        assignment) /\
      True /\
      (semantics key).evaluations (relationSource layout system)
        assignment statement.point =
        statement.evaluations
  rw [evaluations_eq_paper key layout system]

theorem openingAgreement
    (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (normBound : Nat) (commitment : Commitment)
    (publicInput : PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    (assignment : Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Opening.Holds
        (paperRelationSemantics (shape := productionShape)
          (blockCount := Phi81ColumnLayout.blockCount
            (Phi81CarrierLayout.carrierWidth logicalWidth))
          baseOps extensionOps K.embed (openingMaps key))
        normBound commitment publicInput assignment <->
      Opening.Holds (semantics key) normBound commitment publicInput assignment := by
  change
    ((PiRLCAlgebra.Commitment.commit key assignment = commitment /\
        Phi81Relation.projectPublicInput assignment = publicInput /\
        (forall column, centeredMagnitude (assignment column) < normBound)) <->
      (PiRLCAlgebra.Commitment.commit key assignment = commitment /\
        Phi81Relation.projectPublicInput assignment = publicInput /\
        (forall column, centeredMagnitude (assignment column) < normBound)))
  exact Iff.rfl

/-! ## One-entry Π_RLC evaluation algebra -/

def evaluationZero : Evaluation where
  pad := BaseLinear.evaluationZero
  matrix := fun _ => BaseLinear.evaluationZero

def combineEvaluationFamily {count : Nat}
    (challenges : Fin count -> RingF) (families : Fin count -> Evaluation) : Evaluation :=
  {
    pad := PiRLCFinite.combineEvaluation challenges
      (fun source => (families source).pad)
    matrix := fun matrix => PiRLCFinite.combineEvaluation challenges
      (fun source => (families source).matrix matrix)
  }

def combineEvaluations : {count : Nat} ->
    (Fin count -> RingF) -> (Fin count -> Array Evaluation) -> Array Evaluation
  | 0, _, _ => #[evaluationZero]
  | count + 1, challenges, items =>
      Array.ofFn fun index : Fin (items 0).size =>
        combineEvaluationFamily challenges fun source =>
          (items source).getD index.val evaluationZero

/-- Combining singleton evaluation arrays preserves the v1.1 split and
produces exactly one combined Pad/matrix family. -/
theorem combineEvaluations_singletons
    {count : Nat} (positive : 0 < count)
    (challenges : Fin count → RingF) (families : Fin count → Evaluation) :
    combineEvaluations challenges (fun source => #[families source]) =
      #[combineEvaluationFamily challenges families] := by
  cases count with
  | zero => omega
  | succ count =>
      apply Array.ext
      · simp [combineEvaluations]
      · intro index leftLt rightLt
        have indexZero : index = 0 := by
          have indexLt : index < 1 := by
            simpa [combineEvaluations] using leftLt
          omega
        subst index
        simp [combineEvaluations]

theorem evaluations_combine
    (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    {count : Nat} (source : Structure logicalWidth) (point : Point)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
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
          have indexLt : index < 1 := by simpa [semantics] using leftLt
          omega
        subst index
        change evaluationFamily (publicFits := publicFits) source
            (PiRLCFinite.combineAssignments challenges assignments) point = evaluationZero
        apply evaluation_ext
        · exact EvaluationHomomorphism.PiRLC.ExplicitMatrix.evaluate_zero
            (canonicalStructure (publicFits := publicFits) source)
            (padMatrix source) point
        · funext matrix
          exact BaseLinear.matrixEvaluation_zero
            (canonicalStructure (publicFits := publicFits) source) point matrix
  | succ count =>
      apply Array.ext
      · simp [combineEvaluations, semantics]
      · intro index leftLt rightLt
        have indexZero : index = 0 := by
          have indexLt : index < 1 := by simpa [semantics] using leftLt
          omega
        subst index
        change
          evaluationFamily (publicFits := publicFits) source
              (PiRLCFinite.combineAssignments challenges assignments) point =
            combineEvaluationFamily challenges fun index =>
              evaluationFamily (publicFits := publicFits) source (assignments index) point
        apply evaluation_ext
        · exact PiRLCFinite.explicitMatrixEvaluation_combine
            (canonicalStructure (publicFits := publicFits) source)
            (padMatrix source) challenges assignments point
        · funext matrix
          exact PiRLCFinite.matrixEvaluation_combine
            (canonicalStructure (publicFits := publicFits) source)
            challenges assignments point matrix

/-- Complete Stage 1 Π_RLC algebra over the Ajtai commitment. -/
def piRlcAlgebra (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    PiRLC.Algebra (Structure logicalWidth)
      (Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
      (PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      Point Evaluation Commitment RingF (semantics key) productionGlobalParams where
  challengeValid := PiRLCAlgebra.Challenge.challengeValid
  combineAssignment := PiRLCFinite.combineAssignments
  combineCommitment := PiRLCAlgebra.Commitment.combineCommitments
  combinePublicInput := PiRLCAlgebra.PublicInput.combinePublicInputs
  combineEvaluations := combineEvaluations
  commit_hom := by
    intro count challenges assignments
    exact PiRLCAlgebra.Commitment.commit_combine key challenges assignments
  publicInput_hom := by
    intro count challenges assignments
    exact PiRLCAlgebra.PublicInput.relation_publicInput_hom
      (PiRLCAlgebra.Commitment.commit key) challenges assignments
  evaluations_hom := evaluations_combine key
  norm_growth := by
    intro count totalBound challenges assignments valid fresh
    exact PiRLCAlgebra.Norm.relation_norm_growth
      (PiRLCAlgebra.Commitment.commit key) totalBound challenges assignments valid fresh

/-! ## One-entry Π_DEC evaluation algebra -/

def recomposeEvaluationFamily
    (families : Fin productionGlobalParams.k -> Evaluation) : Evaluation :=
  {
    pad := BaseLinear.combineEvaluations
      EvaluationHomomorphism.PiDEC.radixWeight
      (fun child => (families child).pad)
    matrix := fun matrix => BaseLinear.combineEvaluations
      EvaluationHomomorphism.PiDEC.radixWeight
      (fun child => (families child).matrix matrix)
  }

def recomposeEvaluations
    (items : Fin productionGlobalParams.k -> Array Evaluation) : Array Evaluation :=
  #[recomposeEvaluationFamily (fun child => (items child).getD 0 evaluationZero)]

theorem evaluations_recompose
    (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Structure logicalWidth) (point : Point)
    (assignments : Fin productionGlobalParams.k ->
      Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (semantics key).evaluations source
        (PiDECAlgebra.Radix.recomposeAssignment assignments) point =
      recomposeEvaluations fun child =>
        (semantics key).evaluations source (assignments child) point := by
  apply Array.ext
  · rw [semantics_evaluations_size]
    rfl
  · intro index leftLt rightLt
    have indexLt : index < 1 := by simpa only [semantics_evaluations_size] using leftLt
    have indexZero : index = 0 := by omega
    subst index
    change
      evaluationFamily (publicFits := publicFits) source
          (PiDECAlgebra.Radix.recomposeAssignment assignments) point =
        recomposeEvaluationFamily fun child =>
          evaluationFamily (publicFits := publicFits) source (assignments child) point
    apply evaluation_ext
    · exact EvaluationHomomorphism.PiRLC.ExplicitMatrix.evaluate_baseCombine
        (canonicalStructure (publicFits := publicFits) source)
        (padMatrix source) EvaluationHomomorphism.PiDEC.radixWeight
        assignments point
    · funext matrix
      exact BaseLinear.matrixEvaluation_combine
        (canonicalStructure (publicFits := publicFits) source)
        EvaluationHomomorphism.PiDEC.radixWeight assignments point matrix

/-- Complete Stage 1 Π_DEC algebra over the Ajtai commitment. -/
def piDecAlgebra (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    PiDEC.Algebra (Structure logicalWidth)
      (Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
      (PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      Point Evaluation Commitment (semantics key) productionGlobalParams where
  splitAssignment := PiDECAlgebra.Radix.splitAssignment
  recomposeAssignment := PiDECAlgebra.Radix.recomposeAssignment
  recomposeCommitment := PiDECAlgebra.Commitment.recomposeCommitment
  recomposePublicInput := PiDECAlgebra.PublicInput.recomposePublicInput
  recomposeEvaluations := recomposeEvaluations
  split_recompose := PiDECAlgebra.Radix.split_recompose
  split_norm := PiDECAlgebra.Radix.split_norm
  recompose_norm := PiDECAlgebra.Radix.recompose_norm
  commit_hom := PiDECAlgebra.Commitment.commit_recompose key
  publicInput_hom := PiDECAlgebra.PublicInput.relation_publicInput_hom
    (PiRLCAlgebra.Commitment.commit key)
  evaluations_hom := evaluations_recompose key

def publicInputSplit (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    PiDEC.PaperVerifier.PublicInputSplit (piDecAlgebra key) where
  parentBounded := PiDECAlgebra.PublicInput.parentBounded
  parentBounded_decidable := PiDECAlgebra.PublicInput.parentBounded_decidable
  parentBounded_project := PiDECAlgebra.PublicInput.parentBounded_project
  split := fun input child =>
    PiDECAlgebra.PublicInput.splitPublicInput (shape := FullShape logicalWidth publicFits) input child
  recompose_split := fun input => PiDECAlgebra.PublicInput.splitPublicInput_recompose input
  split_project := fun assignment child =>
    PiDECAlgebra.PublicInput.splitPublicInput_project assignment child

def evaluationArity (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    PiDEC.PaperVerifier.EvaluationArity (semantics key) where
  count := fun _ => 1
  evaluations_size := fun _ _ _ => rfl

private def evaluationDecidableEq : DecidableEq Evaluation := by
  letI : DecidableEq (Fin productionShape.coefficientCount → K) :=
    Fintype.decidablePiFintype
  letI : DecidableEq
      (Fin productionShape.matrixCount →
        Fin productionShape.coefficientCount → K) :=
    Fintype.decidablePiFintype
  intro left right
  by_cases pad : left.pad = right.pad
  · by_cases matrix : left.matrix = right.matrix
    · exact isTrue (by cases left; cases right; simp_all)
    · exact isFalse fun equal =>
        matrix (congrArg EvaluationFamily.matrix equal)
  · exact isFalse fun equal =>
      pad (congrArg EvaluationFamily.pad equal)

/-- Constructive production decision for the exact PiDEC accepted language. -/
def piDecDecision
    (key : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (attempt : PiDEC.PaperVerifier.Attempt
      (Structure logicalWidth)
      (PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      Point Evaluation Commitment productionGlobalParams) :
    Decidable (PiDEC.PaperVerifier.Accepted (piDecAlgebra key)
      (publicInputSplit key) (evaluationArity key) attempt) := by
  letI : DecidableEq RingF := Fintype.decidablePiFintype
  letI : DecidableEq Commitment := Fintype.decidablePiFintype
  letI : DecidableEq Evaluation := evaluationDecidableEq
  exact PiDEC.PaperVerifier.acceptedDecision (piDecAlgebra key)
    (publicInputSplit key) (evaluationArity key) attempt

end

end NightstreamFPrime.Lifecycle.PaperAlgebra
