import NightstreamFPrime.Spec.Phi81Relation.Evaluation
import NightstreamFPrime.Spec.Relation

/-!
Typed CCS/CE membership for the batch-invariant paper Phi81 relation.

Protocol: SuperNeo v1.2 Definitions 19--21 specialized to the Phi81 carrier.

Owns: the independent norm and CCS predicates; the one concrete
`RelationSemantics` instantiation; exact membership expansions; canonical
honest statements; and an all-matrix, all-lane characterization of CE output
authority.

Does not own: an Ajtai construction or binding theorem, source-batch folding,
PiRLC/PiDEC homomorphism, `yZcol`, transcripts, Rust, R1CS, row removal, or
constraint counts.
-/

namespace NightstreamFPrime.Spec.Phi81Relation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

universe uCommitment

def assignmentNormBounded {shape : Shape}
    (bound : Nat) (assignment : Assignment shape) : Prop :=
  forall column, centeredMagnitude (assignment column) < bound

def publicInputMatches {shape : Shape}
    (assignment : Assignment shape) (publicInput : PublicInput shape) : Prop :=
  projectPublicInput assignment = publicInput

/-- SuperNeo v1.2 Definition 20's CCS predicate over the sole original matrix family and
explicit sparse constraint polynomial. -/
def ccsSatisfied {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape) : Prop :=
  CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps
    system.matrixSource.system assignment

def evaluationPointValid {shape : Shape}
    (_system : Structure shape) (point : Point shape) : Prop :=
  point.coordinates.length = shape.rowVariables

@[simp] theorem evaluationPointValid_holds {shape : Shape}
    (system : Structure shape) (point : Point shape) :
    evaluationPointValid system point := by
  exact point.dimension

def relationSemantics {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment) :
    RelationSemantics (Structure shape) (Assignment shape)
      (PublicInput shape) (Point shape) Evaluation Commitment where
  commit := commit
  projectPublicInput := projectPublicInput
  normBounded := assignmentNormBounded
  ccsSatisfied := ccsSatisfied
  evaluationPointValid := evaluationPointValid
  evaluations := evaluations

abbrev CCSStatement (shape : Shape) (Commitment : Type uCommitment) :=
  CCS.Instance (Structure shape) (PublicInput shape) Commitment

abbrev CEStatement (shape : Shape) (Commitment : Type uCommitment) :=
  CE.Instance (Structure shape) (PublicInput shape) (Point shape) Evaluation
    Commitment

theorem ccsMembership_iff {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams)
    (statement : CCSStatement shape Commitment)
    (assignment : Assignment shape) :
    CCS.Holds (relationSemantics commit) params statement assignment <->
      commit assignment = statement.commitment /\
      publicInputMatches assignment statement.publicInput /\
      assignmentNormBounded (statement.stage.bound params) assignment /\
      ccsSatisfied statement.constraintSystem assignment := by
  simp [CCS.Holds, Opening.Holds, relationSemantics, publicInputMatches,
    and_assoc]

theorem ceMembership_iff {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams)
    (statement : CEStatement shape Commitment)
    (assignment : Assignment shape) :
    CE.Holds (relationSemantics commit) params statement assignment <->
      commit assignment = statement.commitment /\
      publicInputMatches assignment statement.publicInput /\
      assignmentNormBounded (statement.stage.bound params) assignment /\
      evaluations statement.constraintSystem assignment statement.point =
        statement.evaluations := by
  constructor
  · intro holds
    exact ⟨holds.1.1, holds.1.2.1, holds.1.2.2, holds.2.2⟩
  · rintro ⟨hcommit, hpublic, hnorm, hevaluations⟩
    exact ⟨⟨hcommit, hpublic, hnorm⟩, statement.point.dimension,
      hevaluations⟩

def canonicalCCSStatement {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment)
    (system : Structure shape) (stage : NormStage)
    (assignment : Assignment shape) : CCSStatement shape Commitment where
  constraintSystem := system
  commitment := commit assignment
  publicInput := projectPublicInput assignment
  stage := stage

def canonicalCEStatement {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment)
    (system : Structure shape) (stage : NormStage) (point : Point shape)
    (assignment : Assignment shape) : CEStatement shape Commitment where
  constraintSystem := system
  commitment := commit assignment
  publicInput := projectPublicInput assignment
  point := point
  evaluations := evaluations system assignment point
  stage := stage

theorem canonicalCCS_holds {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams) (system : Structure shape) (stage : NormStage)
    (assignment : Assignment shape)
    (hnorm : assignmentNormBounded (stage.bound params) assignment)
    (hsatisfied : ccsSatisfied system assignment) :
    CCS.Holds (relationSemantics commit) params
      (canonicalCCSStatement commit system stage assignment) assignment := by
  exact ⟨⟨rfl, rfl, hnorm⟩, hsatisfied⟩

theorem canonicalCE_holds {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams) (system : Structure shape) (stage : NormStage)
    (point : Point shape) (assignment : Assignment shape)
    (hnorm : assignmentNormBounded (stage.bound params) assignment) :
    CE.Holds (relationSemantics commit) params
      (canonicalCEStatement commit system stage point assignment) assignment := by
  exact ⟨⟨rfl, rfl, hnorm⟩, point.dimension, rfl⟩

structure EvaluationsBound {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (point : Point shape) (claimed : Array Evaluation) : Prop where
  size_eq : claimed.size = shape.matrixCount
  lane_eq : forall (matrix : Fin shape.matrixCount) (lane : Fin ringDegree),
    (claimed[matrix.val]'(by
      rw [size_eq]
      exact matrix.isLt)) lane =
      matrixEvaluation system assignment point matrix lane

theorem evaluationsBound_iff_eq {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (point : Point shape) (claimed : Array Evaluation) :
    EvaluationsBound system assignment point claimed <->
      claimed = evaluations system assignment point := by
  constructor
  · intro bound
    apply Array.ext
    · simpa using bound.size_eq
    · intro index claimedLt canonicalLt
      have indexLt : index < shape.matrixCount := by
        simpa using canonicalLt
      let matrix : Fin shape.matrixCount := ⟨index, indexLt⟩
      funext lane
      calc
        (claimed[index]'claimedLt) lane =
            matrixEvaluation system assignment point matrix lane := by
          exact bound.lane_eq matrix lane
        _ = ((evaluations system assignment point)[index]'canonicalLt) lane := by
          symm
          exact congrFun (evaluations_get system assignment point matrix) lane
  · intro equal
    rw [equal]
    refine { size_eq := ?_, lane_eq := ?_ }
    · exact evaluations_size system assignment point
    · intro matrix lane
      exact congrFun (evaluations_get system assignment point matrix) lane

theorem ceMembership_iff_evaluationsBound
    {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams)
    (statement : CEStatement shape Commitment)
    (assignment : Assignment shape) :
    CE.Holds (relationSemantics commit) params statement assignment <->
      commit assignment = statement.commitment /\
      publicInputMatches assignment statement.publicInput /\
      assignmentNormBounded (statement.stage.bound params) assignment /\
      EvaluationsBound statement.constraintSystem assignment statement.point
        statement.evaluations := by
  rw [ceMembership_iff]
  constructor
  · rintro ⟨hcommit, hpublic, hnorm, hevaluations⟩
    exact ⟨hcommit, hpublic, hnorm,
      (evaluationsBound_iff_eq _ _ _ _).2 hevaluations.symm⟩
  · rintro ⟨hcommit, hpublic, hnorm, hbound⟩
    exact ⟨hcommit, hpublic, hnorm,
      (evaluationsBound_iff_eq _ _ _ _).1 hbound |>.symm⟩

theorem ce_evaluations_size_of_holds
    {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams)
    (statement : CEStatement shape Commitment)
    (assignment : Assignment shape)
    (holds : CE.Holds (relationSemantics commit) params statement assignment) :
    statement.evaluations.size = shape.matrixCount := by
  exact ((ceMembership_iff_evaluationsBound commit params statement assignment).1
    holds).2.2.2.size_eq

end NightstreamFPrime.Spec.Phi81Relation
