import Nightstream.SuperNeo.Concrete.Phi81Relation.Evaluation
import Nightstream.SuperNeo.Relations

/-!
Typed CCS/CE membership for the batch-invariant paper Phi81 relation.

Protocol: SuperNeo Definitions 11--13 specialized to the Phi81 carrier.
Phase: relation opening, CCS membership, and carried-evaluation membership.
Constraint family: semantic predicates only; this file emits no rows.

Owns: the independent norm and CCS predicates; the one concrete
`RelationSemantics` instantiation; exact membership expansions; canonical
honest statements; and an all-matrix, all-lane characterization of CE output
authority.

Does not own: an Ajtai construction or binding theorem, source-batch folding,
PiRLC/PiDEC homomorphism, `yZcol`, transcripts, Rust, R1CS, row removal, or
constraint counts.

Emits constraints: no.

Authority boundary: callers choose only the abstract commitment map. Public
inputs are the typed ring-aligned assignment prefix. CCS satisfaction is read
from the sole original matrix/polynomial source. Every CE matrix and every one
of its 54 lanes is definitionally derived from that same source. Evaluation
points require no extra runtime shape predicate because `Point shape` already
contains exactly `shape.rowVariables` coordinates.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| CCS/CE | opening | commitment / public input / norm | one assignment opens the public statement at its verifier-owned bound |
| CCS | relation | Boolean-row residuals | the explicit sparse polynomial vanishes at every row-cube vertex |
| CE | point | row-cube shape | malformed point dimensions are uninhabited by the typed carrier |
| CE | evaluations | matrix / Phi81 lane | every declared matrix and all 54 lanes equal the sole derived evaluation |
| assurance | canonical statement | completeness | every honest bounded opening with the required relation fact is accepted |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

universe uCommitment

/-- Independent paper norm predicate over the complete typed assignment. -/
def assignmentNormBounded {shape : Shape}
    (bound : Nat) (assignment : Assignment shape) : Prop :=
  forall column, centeredMagnitude (assignment column) < bound

/-- Exact verifier-owned public projection of the same assignment. -/
def publicInputMatches {shape : Shape}
    (assignment : Assignment shape) (publicInput : PublicInput shape) : Prop :=
  projectPublicInput assignment = publicInput

/-- Definition 12's CCS predicate over the sole original matrix family and
explicit sparse constraint polynomial. -/
def ccsSatisfied {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape) : Prop :=
  CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps
    system.matrixSource.system assignment

/-- Point validity is intrinsic to `Point shape = Fin rowVariables -> K`.
There is no raw length or domain tag left for a prover to choose. -/
def evaluationPointValid {shape : Shape}
    (_system : Structure shape) (point : Point shape) : Prop :=
  point.coordinates.length = shape.rowVariables

/-- Every typed point carries exactly the verifier-owned row dimension. -/
@[simp] theorem evaluationPointValid_holds {shape : Shape}
    (system : Structure shape) (point : Point shape) :
    evaluationPointValid system point := by
  exact point.dimension

/-- The typed paper relation. The commitment construction remains an explicit
parameter until a separate Ajtai refinement and binding theorem instantiate
it. No other semantic operation is caller supplied. -/
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

/-- Exact expansion of typed Phi81 CCS membership. -/
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

/-- Exact expansion of typed Phi81 CE membership. Point validity simplifies
to `True` because the point dimension is already encoded in its type. -/
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

/-- Canonical public CCS statement for one authoritative assignment. -/
def canonicalCCSStatement {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment)
    (system : Structure shape) (stage : NormStage)
    (assignment : Assignment shape) : CCSStatement shape Commitment where
  constraintSystem := system
  commitment := commit assignment
  publicInput := projectPublicInput assignment
  stage := stage

/-- Canonical public CE statement for one authoritative assignment and typed
row point. -/
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

/-- Completeness of the canonical CCS statement, conditional only on the two
mathematical facts that are not true by construction. -/
theorem canonicalCCS_holds {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams) (system : Structure shape) (stage : NormStage)
    (assignment : Assignment shape)
    (hnorm : assignmentNormBounded (stage.bound params) assignment)
    (hsatisfied : ccsSatisfied system assignment) :
    CCS.Holds (relationSemantics commit) params
      (canonicalCCSStatement commit system stage assignment) assignment := by
  exact ⟨⟨rfl, rfl, hnorm⟩, hsatisfied⟩

/-- Completeness of the canonical CE statement. The typed point discharges
the entire point-shape obligation by construction. -/
theorem canonicalCE_holds {shape : Shape} {Commitment : Type uCommitment}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams) (system : Structure shape) (stage : NormStage)
    (point : Point shape) (assignment : Assignment shape)
    (hnorm : assignmentNormBounded (stage.bound params) assignment) :
    CE.Holds (relationSemantics commit) params
      (canonicalCEStatement commit system stage point assignment) assignment := by
  exact ⟨⟨rfl, rfl, hnorm⟩, point.dimension, rfl⟩

/-- Explicit CE output authority. Exact array size prevents missing or extra
matrices; the leaf equation binds every one of the 54 lanes. -/
structure EvaluationsBound {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (point : Point shape) (claimed : Array Evaluation) : Prop where
  size_eq : claimed.size = shape.matrixCount
  lane_eq : forall (matrix : Fin shape.matrixCount) (lane : Fin ringDegree),
    (claimed[matrix.val]'(by
      rw [size_eq]
      exact matrix.isLt)) lane =
      matrixEvaluation system assignment point matrix lane

/-- The all-matrix, all-lane predicate is exactly equality with the canonical
evaluation array; it is not a weaker sampled or digest-only condition. -/
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

/-- CE membership restated as opening authority plus explicit coverage of
every matrix and every Phi81 lane. -/
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

/-- Any accepted CE statement contains exactly one claimed ring value per
matrix. This rules out default-filled short arrays at the semantic boundary. -/
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

end Nightstream.SuperNeo.Concrete.Phi81Relation
