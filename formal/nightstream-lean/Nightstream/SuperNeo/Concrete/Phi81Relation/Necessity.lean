import Nightstream.SuperNeo.Concrete.Necessity.Phi81OutputMismatch
import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.Concrete.Phi81Relation.Semantics

/-!
Concrete inclusion-necessity witnesses for typed Phi81 CCS/CE obligations.

Protocol: SuperNeo Definitions 12--13 specialized to the Phi81 relation.
Phase: opening and relation checks after independent semantics are fixed.
Constraint family: semantic check families only; this file emits no rows.

Owns: weakened verifier predicates that omit exactly one named family and
kernel-checked witnesses showing that the weakened predicate accepts while
the full typed relation rejects.

Does not own: cryptographic commitment binding, global protocol minimality,
PiRLC/PiDEC homomorphism, Rust/R1CS refinement, a gate-count lower bound, row
removal, or constraint counts.

Emits constraints: no.

Authority boundary: these are concrete countermodels, not wrappers saying
only that a bad value is rejected. Each theorem exhibits an actual typed
statement and assignment accepted after one family is omitted. The
commitment witness proves the equality check is necessary; it deliberately
does not claim that the abstract commitment map is binding. A production row
can be removed only after a separate refinement theorem proves another
retained family implies the omitted semantic obligation.

| Protocol | Phase | Omitted family | Kernel-checked invalid acceptance |
|---|---|---|---|
| CE | opening | commitment equality | wrong Boolean commitment passes all remaining CE families |
| CE | opening | public projection | zero public carrier replaces a nonzero authoritative lane |
| CE | opening | complete-carrier norm | magnitude-two coordinate passes at fresh bound two |
| CCS | relation | Boolean-row residual | constant-one residual passes all opening families |
| CE | evaluation | exact matrix count | oversized claim preserves every declared matrix lane when only exact size is omitted |
| CE | evaluation | every Phi81 lane | size-correct zero ring passes when only lane equations are omitted |
| CE | point | row-cube dimension | invariant is intrinsic to `CubePoint`; no malformed typed value exists |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Necessity.Phi81OutputMismatch
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

/-! ## One-family-omitted verifier predicates -/

/-- CE acceptance with only commitment equality omitted. -/
def CEWithoutCommitment {shape : Shape} {Commitment : Type}
    (params : GlobalParams) (statement : CEStatement shape Commitment)
    (assignment : Assignment shape) : Prop :=
  publicInputMatches assignment statement.publicInput /\
  assignmentNormBounded (statement.stage.bound params) assignment /\
  EvaluationsBound statement.constraintSystem assignment statement.point
    statement.evaluations

/-- CE acceptance with only public-input equality omitted. -/
def CEWithoutPublicInput {shape : Shape} {Commitment : Type}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams) (statement : CEStatement shape Commitment)
    (assignment : Assignment shape) : Prop :=
  commit assignment = statement.commitment /\
  assignmentNormBounded (statement.stage.bound params) assignment /\
  EvaluationsBound statement.constraintSystem assignment statement.point
    statement.evaluations

/-- CE acceptance with only complete-carrier norm checking omitted. -/
def CEWithoutNorm {shape : Shape} {Commitment : Type}
    (commit : Assignment shape -> Commitment)
    (statement : CEStatement shape Commitment)
    (assignment : Assignment shape) : Prop :=
  commit assignment = statement.commitment /\
  publicInputMatches assignment statement.publicInput /\
  EvaluationsBound statement.constraintSystem assignment statement.point
    statement.evaluations

/-- The declared matrix prefix remains authoritative, but extra trailing
evaluations are permitted. This isolates exact array size from lane equality. -/
structure DeclaredEvaluationsBound {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (point : Point shape) (claimed : Array Evaluation) : Prop where
  minimum_size : shape.matrixCount ≤ claimed.size
  lane_eq : forall (matrix : Fin shape.matrixCount) (lane : Fin ringDegree),
    (claimed[matrix.val]'(Nat.lt_of_lt_of_le matrix.isLt minimum_size)) lane =
      matrixEvaluation system assignment point matrix lane

/-- CE acceptance with only exact evaluation-array size omitted. -/
def CEWithoutEvaluationSize {shape : Shape} {Commitment : Type}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams) (statement : CEStatement shape Commitment)
    (assignment : Assignment shape) : Prop :=
  commit assignment = statement.commitment /\
  publicInputMatches assignment statement.publicInput /\
  assignmentNormBounded (statement.stage.bound params) assignment /\
  DeclaredEvaluationsBound statement.constraintSystem assignment
    statement.point statement.evaluations

/-- CE acceptance with only all-matrix, all-lane equality omitted. Exact array
size remains enforced. -/
def CEWithoutEvaluationLanes {shape : Shape} {Commitment : Type}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams) (statement : CEStatement shape Commitment)
    (assignment : Assignment shape) : Prop :=
  commit assignment = statement.commitment /\
  publicInputMatches assignment statement.publicInput /\
  assignmentNormBounded (statement.stage.bound params) assignment /\
  statement.evaluations.size = shape.matrixCount

/-- CCS acceptance with only the Boolean-row residual family omitted. -/
def CCSWithoutRelation {shape : Shape} {Commitment : Type}
    (commit : Assignment shape -> Commitment)
    (params : GlobalParams) (statement : CCSStatement shape Commitment)
    (assignment : Assignment shape) : Prop :=
  commit assignment = statement.commitment /\
  publicInputMatches assignment statement.publicInput /\
  assignmentNormBounded (statement.stage.bound params) assignment

/-! ## Shared nonzero Phi81 fixture -/

/-- One public ring over the existing nonzero completed-carrier fixture. -/
def witnessShape : Shape :=
  Shape.ofSemantic modelShape 1 (by decide)

def witnessSystem : Structure witnessShape :=
  Structure.ofSourceData 1 (by decide) sourceData

def witnessAssignment : Assignment witnessShape :=
  sourceData.assignment source

def witnessPoint : Point witnessShape := verifierPoints.rPrime

/-- The commitment carrier is intentionally elementary: this layer tests the
presence of equality, not cryptographic binding. -/
def booleanCommitment (_assignment : Assignment witnessShape) : Bool := false

def honestCE : CEStatement witnessShape Bool :=
  canonicalCEStatement booleanCommitment witnessSystem .fresh witnessPoint
    witnessAssignment

def runningSource : Fin modelShape.runningCount := ⟨0, by decide⟩

theorem source_eq_runningIndex :
    source =
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data.runningIndex
        runningSource := by
  apply Fin.ext
  rfl

theorem witnessAssignment_eq_runningAssignment :
    witnessAssignment = runningAssignment := by
  calc
    witnessAssignment = sourceData.assignment
        (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data.runningIndex
          runningSource) := by
      unfold witnessAssignment
      rw [source_eq_runningIndex]
    _ = sourceData.runningAssignments runningSource :=
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data.assignment_runningIndex
        sourceData runningSource
    _ = runningAssignment := rfl

theorem witnessAssignment_fresh_norm :
    assignmentNormBounded
      (NormStage.fresh.bound productionGlobalParams) witnessAssignment := by
  rw [witnessAssignment_eq_runningAssignment]
  intro column
  change centeredMagnitude (runningAssignment column) < 2
  by_cases equal : column = carrierColumnOne
  · subst column
    decide
  · have valueZero : runningAssignment column = 0 := by
      unfold runningAssignment oneHotAssignment
      split
      · contradiction
      · rfl
    rw [valueZero]
    decide

theorem honestEvaluationsBound :
    EvaluationsBound witnessSystem witnessAssignment witnessPoint
      honestCE.evaluations := by
  apply (evaluationsBound_iff_eq _ _ _ _).2
  rfl

/-! ## Commitment equality -/

def wrongCommitmentCE : CEStatement witnessShape Bool :=
  { honestCE with commitment := true }

/-- Omitting commitment equality accepts an actual statement that full CE
membership rejects while public input, norm, and all evaluations remain
canonical. -/
theorem commitment_check_is_necessary :
    CEWithoutCommitment productionGlobalParams wrongCommitmentCE
        witnessAssignment /\
      booleanCommitment witnessAssignment ≠ wrongCommitmentCE.commitment /\
      ¬ CE.Holds (relationSemantics booleanCommitment)
        productionGlobalParams wrongCommitmentCE witnessAssignment := by
  refine ⟨⟨rfl, witnessAssignment_fresh_norm, honestEvaluationsBound⟩,
    ?_, ?_⟩
  · decide
  · intro holds
    have commitment :=
      ((ceMembership_iff_evaluationsBound booleanCommitment
        productionGlobalParams wrongCommitmentCE witnessAssignment).1 holds).1
    have impossible : (false : Bool) = true := by
      simpa [booleanCommitment, wrongCommitmentCE, honestCE] using commitment
    exact Bool.noConfusion impossible

/-! ## Public-input projection -/

def zeroPublicInput : PublicInput witnessShape := fun _ => 0

def publicLaneOne : Fin witnessShape.publicWidth := ⟨1, by decide⟩

theorem witnessPublic_laneOne_eq_one :
    projectPublicInput witnessAssignment publicLaneOne = 1 := by
  decide

theorem witnessPublic_ne_zero :
    ¬ publicInputMatches witnessAssignment zeroPublicInput := by
  intro equal
  have laneEqual := congrFun equal publicLaneOne
  have impossible : (1 : F) = 0 := by
    simpa [zeroPublicInput, witnessPublic_laneOne_eq_one] using laneEqual
  exact (by decide : (1 : F) ≠ 0) impossible

def wrongPublicInputCE : CEStatement witnessShape Bool :=
  { honestCE with publicInput := zeroPublicInput }

/-- Omitting public projection accepts a statement whose public lane one is
zero although the authoritative assignment projects to one. -/
theorem public_input_check_is_necessary :
    CEWithoutPublicInput booleanCommitment productionGlobalParams
        wrongPublicInputCE witnessAssignment /\
      ¬ publicInputMatches witnessAssignment wrongPublicInputCE.publicInput /\
      ¬ CE.Holds (relationSemantics booleanCommitment)
        productionGlobalParams wrongPublicInputCE witnessAssignment := by
  refine ⟨⟨rfl, witnessAssignment_fresh_norm, honestEvaluationsBound⟩,
    ?_, ?_⟩
  · exact witnessPublic_ne_zero
  · intro holds
    exact witnessPublic_ne_zero
      (((ceMembership_iff_evaluationsBound booleanCommitment
        productionGlobalParams wrongPublicInputCE witnessAssignment).1
          holds).2.1)

/-! ## Complete-carrier norm -/

def carrierColumnZero : Fin witnessShape.carrierWidth := ⟨0, by decide⟩

def highNormAssignment : Assignment witnessShape := fun column =>
  if column = carrierColumnZero then 2 else 0

theorem highNormAssignment_not_fresh :
    ¬ assignmentNormBounded
      (NormStage.fresh.bound productionGlobalParams) highNormAssignment := by
  intro bounded
  have atZero := bounded carrierColumnZero
  change centeredMagnitude (2 : F) < 2 at atZero
  exact (by decide : ¬ centeredMagnitude (2 : F) < 2) atZero

def highNormCE : CEStatement witnessShape Bool :=
  canonicalCEStatement booleanCommitment witnessSystem .fresh witnessPoint
    highNormAssignment

theorem highNormEvaluationsBound :
    EvaluationsBound witnessSystem highNormAssignment witnessPoint
      highNormCE.evaluations := by
  apply (evaluationsBound_iff_eq _ _ _ _).2
  rfl

/-- Omitting the norm family accepts magnitude two at the strict fresh bound
two while every other CE family remains canonical. -/
theorem norm_check_is_necessary :
    CEWithoutNorm booleanCommitment highNormCE highNormAssignment /\
      ¬ assignmentNormBounded
        (highNormCE.stage.bound productionGlobalParams) highNormAssignment /\
      ¬ CE.Holds (relationSemantics booleanCommitment)
        productionGlobalParams highNormCE highNormAssignment := by
  refine ⟨⟨rfl, rfl, highNormEvaluationsBound⟩,
    highNormAssignment_not_fresh, ?_⟩
  intro holds
  exact highNormAssignment_not_fresh
    (((ceMembership_iff_evaluationsBound booleanCommitment
      productionGlobalParams highNormCE highNormAssignment).1 holds).2.2.1)

/-! ## CCS residual -/

/-- Constant one at the same nondegenerate matrix arity as the shared Phi81
fixture. Every matrix variable has exponent zero. -/
def constantOneMonomial :
    CCSResidualTable.Monomial F witnessShape.matrixCount where
  coefficient := 1
  exponents := fun _ => 0

def constantOnePolynomial :
    CCSResidualTable.ConstraintPolynomial F witnessShape.matrixCount where
  degreeBound := 1
  terms := [constantOneMonomial]
  termsBelowDegree := by
    intro term member
    simp only [List.mem_singleton] at member
    subst term
    decide

def falseCcsSystem : Structure witnessShape :=
  { witnessSystem with constraintPolynomial := constantOnePolynomial }

def falseCCS : CCSStatement witnessShape Bool :=
  canonicalCCSStatement booleanCommitment falseCcsSystem .fresh
    witnessAssignment

theorem falseCcsSystem_not_satisfied :
    ¬ ccsSatisfied falseCcsSystem witnessAssignment := by
  intro satisfied
  have atVertex := satisfied BooleanVertex.nil
  change (1 : F) = 0 at atVertex
  exact (by decide : (1 : F) ≠ 0) atVertex

/-- Omitting the CCS residual family accepts a constant-one residual while
commitment, public projection, and norm remain canonical. -/
theorem ccs_relation_check_is_necessary :
    CCSWithoutRelation booleanCommitment productionGlobalParams falseCCS
        witnessAssignment /\
      ¬ ccsSatisfied falseCcsSystem witnessAssignment /\
      ¬ CCS.Holds (relationSemantics booleanCommitment)
        productionGlobalParams falseCCS witnessAssignment := by
  refine ⟨⟨rfl, rfl, witnessAssignment_fresh_norm⟩,
    falseCcsSystem_not_satisfied, ?_⟩
  intro holds
  exact falseCcsSystem_not_satisfied
    (((ccsMembership_iff booleanCommitment productionGlobalParams falseCCS
      witnessAssignment).1 holds).2.2.2)

/-! ## Complete CE evaluation authority -/

/-- The canonical value for the sole declared matrix followed by one
unclaimed trailing value. -/
def oversizedEvaluations : Array Evaluation :=
  #[matrixEvaluation witnessSystem witnessAssignment witnessPoint matrix,
    ringKZero]

def oversizedEvaluationsCE : CEStatement witnessShape Bool :=
  { honestCE with evaluations := oversizedEvaluations }

theorem oversizedDeclaredEvaluationsBound :
    DeclaredEvaluationsBound witnessSystem witnessAssignment witnessPoint
      oversizedEvaluations := by
  refine { minimum_size := by decide, lane_eq := ?_ }
  intro matrixIndex lane
  have matrixIndex_eq : matrixIndex = matrix := by
    apply Fin.ext
    change matrixIndex.val = 0
    have indexLt : matrixIndex.val < 1 := by
      simpa [witnessShape, Shape.ofSemantic, modelShape] using matrixIndex.isLt
    omega
  subst matrixIndex
  rfl

/-- Omitting only exact size accepts a trailing evaluation while the value of
every declared matrix and every one of its lanes remains canonical. -/
theorem evaluation_size_check_is_necessary :
    CEWithoutEvaluationSize booleanCommitment productionGlobalParams
        oversizedEvaluationsCE witnessAssignment /\
      oversizedEvaluationsCE.evaluations.size ≠ witnessShape.matrixCount /\
      ¬ CE.Holds (relationSemantics booleanCommitment)
        productionGlobalParams oversizedEvaluationsCE witnessAssignment := by
  refine ⟨⟨rfl, rfl, witnessAssignment_fresh_norm,
    oversizedDeclaredEvaluationsBound⟩, ?_, ?_⟩
  · decide
  · intro holds
    have size := ce_evaluations_size_of_holds booleanCommitment
      productionGlobalParams oversizedEvaluationsCE witnessAssignment holds
    change 2 = 1 at size
    omega

def zeroRingEvaluations : Array Evaluation := #[ringKZero]

def wrongLaneCE : CEStatement witnessShape Bool :=
  { honestCE with evaluations := zeroRingEvaluations }

theorem wrongLane_not_bound :
    ¬ EvaluationsBound witnessSystem witnessAssignment witnessPoint
      zeroRingEvaluations := by
  intro bound
  have laneEqual := bound.lane_eq matrix laneOne
  have canonicalOne :
      matrixEvaluation witnessSystem witnessAssignment witnessPoint matrix
        laneOne = K.one := by
    change matrixEvaluation
      (Structure.ofSourceData 1 (by decide) sourceData)
      (sourceData.assignment source) verifierPoints.rPrime matrix laneOne = K.one
    rw [matrixEvaluation_apply_ofSourceData]
    exact canonicalYRing_laneOne_eq_one
  have impossible : K.zero = K.one := by
    simpa [zeroRingEvaluations, canonicalOne] using laneEqual
  exact (by decide : K.zero ≠ K.one) impossible

/-- Even a size-correct array is invalid if one Phi81 lane is not derived
from the sole matrix source and assignment. -/
theorem evaluation_lane_check_is_necessary :
    CEWithoutEvaluationLanes booleanCommitment productionGlobalParams wrongLaneCE
        witnessAssignment /\
      ¬ EvaluationsBound witnessSystem witnessAssignment witnessPoint
        wrongLaneCE.evaluations /\
      ¬ CE.Holds (relationSemantics booleanCommitment)
        productionGlobalParams wrongLaneCE witnessAssignment := by
  refine ⟨⟨rfl, rfl, witnessAssignment_fresh_norm, by decide⟩,
    wrongLane_not_bound, ?_⟩
  intro holds
  exact wrongLane_not_bound
    (((ceMembership_iff_evaluationsBound booleanCommitment
      productionGlobalParams wrongLaneCE witnessAssignment).1 holds).2.2.2)

/-! ## Point shape is intrinsic -/

/-- Unlike the five value-level families above, point dimension has no
omittable caller-selected value in this typed relation. Production may use
this fact only after raw decoding is proved to construct `Point shape`. -/
theorem point_shape_is_intrinsic {shape : Shape}
    (system : Structure shape) (point : Point shape) :
    evaluationPointValid system point :=
  evaluationPointValid_holds system point

/-- There is no typed counterexample to the point-dimension obligation. A raw
decoder must first prove that its input constructs `Point shape`; after that,
an independent runtime shape equation would be redundant. -/
theorem no_invalid_typed_point {shape : Shape} (system : Structure shape) :
    ¬ ∃ point : Point shape, ¬ evaluationPointValid system point := by
  rintro ⟨point, invalid⟩
  exact invalid (evaluationPointValid_holds system point)

end Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity
