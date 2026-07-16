import Nightstream.SuperNeo.CheckPlan
import Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity

/-!
Inclusion-minimal check plan for the concrete typed Phi81 CE relation.

Owns: a finite named CE obligation plan, exact equivalence between that plan
and typed CE membership, conversion between complete evaluation authority and
the separated exact-size/declared-lane leaves, and a concrete removal witness
for every retained family.

Does not own: CCS-family minimality, point decoding, commitment binding,
`PiRLC`/`PiDEC`, global verifier minimality, Rust/R1CS refinement, a gate-count
lower bound, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: `cePlan_inclusionMinimalSound` is semantic
inclusion-minimality for this concrete nonzero fixture. It proves that no named
family can be uniformly deleted from this plan while preserving soundness. It
does not prove that every production row implementing a family is necessary;
that requires exact lowering and per-row redundancy proofs.

| Protocol | Phase | Constraint family | Check-plan leaf | Removal witness |
|---|---|---|---|---|
| CE | opening | commitment equality | `CEFamily.commitment` | wrong Boolean commitment |
| CE | opening | public projection | `CEFamily.publicInput` | nonzero authoritative lane replaced by zero |
| CE | opening | complete-carrier norm | `CEFamily.norm` | magnitude-two coordinate at strict bound two |
| CE | evaluations | exact matrix count | `CEFamily.evaluationSize` | canonical declared prefix plus trailing ring |
| CE | evaluations | every matrix/lane | `CEFamily.evaluationLanes` | exact-size zero ring with one wrong lane |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.Minimality

open Nightstream.SuperNeo.CheckPlan
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity

inductive CEFamily where
  | commitment
  | publicInput
  | norm
  | evaluationSize
  | evaluationLanes
deriving DecidableEq

abbrev CEInput := CEStatement witnessShape Bool × Assignment witnessShape

def ceSemantics : CEFamily -> CEInput -> Prop
  | .commitment, input =>
      booleanCommitment input.2 = input.1.commitment
  | .publicInput, input =>
      publicInputMatches input.2 input.1.publicInput
  | .norm, input =>
      assignmentNormBounded
        (input.1.stage.bound productionGlobalParams) input.2
  | .evaluationSize, input =>
      input.1.evaluations.size = witnessShape.matrixCount
  | .evaluationLanes, input =>
      DeclaredEvaluationsBound input.1.constraintSystem input.2 input.1.point
        input.1.evaluations

def ceTarget (input : CEInput) : Prop :=
  CE.Holds (relationSemantics booleanCommitment)
    productionGlobalParams input.1 input.2

def cePlan : List CEFamily :=
  [.commitment, .publicInput, .norm, .evaluationSize, .evaluationLanes]

theorem declaredEvaluationsBound_of_evaluationsBound
    {shape : Shape} {system : Structure shape}
    {assignment : Assignment shape} {point : Point shape}
    {claimed : Array Evaluation}
    (bound : EvaluationsBound system assignment point claimed) :
    DeclaredEvaluationsBound system assignment point claimed := by
  refine { minimum_size := ?_, lane_eq := ?_ }
  · rw [bound.size_eq]
    exact Nat.le_refl _
  · intro matrix lane
    exact bound.lane_eq matrix lane

theorem evaluationsBound_of_size_and_declared
    {shape : Shape} {system : Structure shape}
    {assignment : Assignment shape} {point : Point shape}
    {claimed : Array Evaluation}
    (size : claimed.size = shape.matrixCount)
    (declared : DeclaredEvaluationsBound system assignment point claimed) :
    EvaluationsBound system assignment point claimed := by
  refine { size_eq := size, lane_eq := ?_ }
  intro matrix lane
  exact declared.lane_eq matrix lane

/-- The five-leaf plan accepts exactly the independently defined CE relation. -/
theorem cePlan_exact : Exact ceSemantics ceTarget cePlan := by
  intro input
  rcases input with ⟨statement, assignment⟩
  constructor
  · intro accepted
    apply (ceMembership_iff_evaluationsBound booleanCommitment
      productionGlobalParams statement assignment).2
    refine ⟨accepted .commitment (by simp [cePlan]),
      accepted .publicInput (by simp [cePlan]),
      accepted .norm (by simp [cePlan]), ?_⟩
    exact evaluationsBound_of_size_and_declared
      (accepted .evaluationSize (by simp [cePlan]))
      (accepted .evaluationLanes (by simp [cePlan]))
  · intro target family _member
    have expanded :=
      (ceMembership_iff_evaluationsBound booleanCommitment
        productionGlobalParams statement assignment).1 target
    cases family with
    | commitment => exact expanded.1
    | publicInput => exact expanded.2.1
    | norm => exact expanded.2.2.1
    | evaluationSize => exact expanded.2.2.2.size_eq
    | evaluationLanes =>
        exact declaredEvaluationsBound_of_evaluationsBound expanded.2.2.2

theorem cePlan_sound : Sound ceSemantics ceTarget cePlan :=
  (exact_iff_sound_and_complete.mp cePlan_exact).1

theorem commitment_necessary :
    NecessaryForSoundness ceSemantics ceTarget cePlan .commitment := by
  refine ⟨(wrongCommitmentCE, witnessAssignment), ?_,
    commitment_check_is_necessary.2.2⟩
  intro family member
  have retained := commitment_check_is_necessary.1
  cases family with
  | commitment => simp [cePlan, without] at member
  | publicInput => exact retained.1
  | norm => exact retained.2.1
  | evaluationSize => exact retained.2.2.size_eq
  | evaluationLanes =>
      exact declaredEvaluationsBound_of_evaluationsBound retained.2.2

theorem publicInput_necessary :
    NecessaryForSoundness ceSemantics ceTarget cePlan .publicInput := by
  refine ⟨(wrongPublicInputCE, witnessAssignment), ?_,
    public_input_check_is_necessary.2.2⟩
  intro family member
  have retained := public_input_check_is_necessary.1
  cases family with
  | commitment => exact retained.1
  | publicInput => simp [cePlan, without] at member
  | norm => exact retained.2.1
  | evaluationSize => exact retained.2.2.size_eq
  | evaluationLanes =>
      exact declaredEvaluationsBound_of_evaluationsBound retained.2.2

theorem norm_necessary :
    NecessaryForSoundness ceSemantics ceTarget cePlan .norm := by
  refine ⟨(highNormCE, highNormAssignment), ?_,
    norm_check_is_necessary.2.2⟩
  intro family member
  have retained := norm_check_is_necessary.1
  cases family with
  | commitment => exact retained.1
  | publicInput => exact retained.2.1
  | norm => simp [cePlan, without] at member
  | evaluationSize => exact retained.2.2.size_eq
  | evaluationLanes =>
      exact declaredEvaluationsBound_of_evaluationsBound retained.2.2

theorem evaluationSize_necessary :
    NecessaryForSoundness ceSemantics ceTarget cePlan .evaluationSize := by
  refine ⟨(oversizedEvaluationsCE, witnessAssignment), ?_,
    evaluation_size_check_is_necessary.2.2⟩
  intro family member
  have retained := evaluation_size_check_is_necessary.1
  cases family with
  | commitment => exact retained.1
  | publicInput => exact retained.2.1
  | norm => exact retained.2.2.1
  | evaluationSize => simp [cePlan, without] at member
  | evaluationLanes => exact retained.2.2.2

theorem evaluationLanes_necessary :
    NecessaryForSoundness ceSemantics ceTarget cePlan .evaluationLanes := by
  refine ⟨(wrongLaneCE, witnessAssignment), ?_,
    evaluation_lane_check_is_necessary.2.2⟩
  intro family member
  have retained := evaluation_lane_check_is_necessary.1
  cases family with
  | commitment => exact retained.1
  | publicInput => exact retained.2.1
  | norm => exact retained.2.2.1
  | evaluationSize => exact retained.2.2.2
  | evaluationLanes => simp [cePlan, without] at member

/-- The concrete CE obligation plan is sound and every leaf has a concrete
invalid acceptance after exactly that family is removed. -/
theorem cePlan_inclusionMinimalSound :
    InclusionMinimalSound ceSemantics ceTarget cePlan := by
  apply inclusionMinimalSound_of_witnesses cePlan_sound
  intro family _member
  cases family with
  | commitment => exact commitment_necessary
  | publicInput => exact publicInput_necessary
  | norm => exact norm_necessary
  | evaluationSize => exact evaluationSize_necessary
  | evaluationLanes => exact evaluationLanes_necessary

end Nightstream.SuperNeo.Concrete.Phi81Relation.Minimality
