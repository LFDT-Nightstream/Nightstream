import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.Folding.PiCCS
import Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier
import Nightstream.SuperNeo.Folding.PiRLC

/-!
Independent fixed-active SuperNeo NIFS paper profile.

Protocol: SuperNeo Sections 7.3--7.5.
Phase: `Pi_CCS -> Pi_RLC -> Pi_DEC` at the production-selected arity.
Constraint family: abstract relation obligations only; this file emits no
rows.

Assurance tier: model-level.

Owns: the exact `1 CCS + 14 CE -> 15 CE -> 1 CE -> 14 CE` semantic graph;
the abstract relation and tail algebras; source membership and common-shape
obligations; one new row point; fifteen strong-set challenges; the operational
Section-7.5 `Pi_DEC` verifier, including exact evaluation arity; and the
canonical honest-completeness constructor.

Does not own: Phi81, Split-NC, either SumCheck message flow, Fiat--Shamir,
commitment security, child-opening extraction, parent caches, HyperNova
lifecycle state, Rust, R1CS, costs, necessity, or row removal.

Authority boundary: this module depends only on the paper-level relation and
the independently specified `Pi_CCS`, `Pi_RLC`, and `Pi_DEC` algebras. It does
not import any concrete NIFS verifier or F-prime module. The combined parent
is an internal computed intermediate; only the fourteen children are public.

The final `-> 14 CE` edge comes from SuperNeo Section 7.5. Sections 7.3--7.4
alone end at the single combined `CE(B)` parent.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.paper.profile` | `K = 1`, `k = 14`, fifteen sources, fourteen outputs, `b = 2` | verifier parameters | profile count theorems |
| `nifs.paper.source` | every source is fresh-stage, valid, same-structure, and running claims share one prior point | independent specification | `Realization` |
| `nifs.paper.pi_ccs` | all sources are re-evaluated at one valid new point | computed | `outputs`, `Realization.outputsHold` |
| `nifs.paper.pi_rlc` | fifteen valid challenges determine one valid combined opening | checked/computed | `ChallengesValid`, `parentOf`, `Realization.parentOpening` |
| `nifs.paper.pi_dec` | child public inputs/structure/point/stage and evaluation arity are fixed; commitment/evaluation recomposition is checked | checked/computed | `Realization.piDecAccepted`, `childrenOf`, `complete` |
| `nifs.paper.transition` | public-verifier source-to-children acceptance relation | independent specification | `Transition` |
| `nifs.paper.completeness` | every complete realization premise constructs the canonical target | derived | `complete` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.PaperProfile

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar

/-- HyperNova selects one fresh relation instance; SuperNeo carries the full
production `k = 14` running product. -/
def arity : BatchArity productionGlobalParams :=
  BatchArity.active productionGlobalParams 1 (by decide) (by decide)

@[simp] theorem arity_freshCount : arity.freshCount = 1 := rfl

@[simp] theorem arity_runningCount :
    arity.mode.count productionGlobalParams = 14 := rfl

@[simp] theorem arity_total : arity.total = 15 := rfl

@[simp] theorem outputCount : productionGlobalParams.k = 14 := rfl

@[simp] theorem baseNormBound : productionGlobalParams.b = 2 := rfl

/-- Exact public paper input. The canonical source order is the one owned by
`PiCCS.InputProduct.source`. -/
abbrev Input
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment) :=
  PiCCS.InputProduct Structure PublicInput Point Evaluation Commitment
    productionGlobalParams arity

/-- Exact public paper target: the fourteen post-decomposition CE children. -/
abbrev Output
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment) :=
  Fin productionGlobalParams.k ->
    CE.Instance Structure PublicInput Point Evaluation Commitment

/-- Abstract relation operations and the two independently specified tail
algebras. No concrete verifier key or transcript state appears here. -/
structure Profile
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar) where
  semantics : RelationSemantics Structure Assignment PublicInput Point
    Evaluation Commitment
  rlcAlgebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation
    Commitment Scalar semantics productionGlobalParams
  decAlgebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
    Commitment semantics productionGlobalParams
  decPublicInputSplit : PiDEC.PaperVerifier.PublicInputSplit decAlgebra
  decEvaluationArity : PiDEC.PaperVerifier.EvaluationArity semantics

/-- Raw mathematical witness. Validity is deliberately not proof-carried so
each retained obligation can later be removed independently. -/
structure Witness
    (Assignment : Type uAssignment)
    (Point : Type uPoint)
    (Scalar : Type uScalar) where
  assignments : Fin arity.total -> Assignment
  point : Point
  challenges : Fin arity.total -> Scalar

/-- The unique fresh source used to derive the common relation structure. -/
def firstFresh : Fin arity.freshCount := ⟨0, by decide⟩

/-- Common structure is derived from the sole fresh source, never supplied as
an independent caller-controlled value. -/
def systemOf
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (input : Input Structure PublicInput Point Evaluation Commitment) :
    Structure :=
  (input.fresh firstFresh).constraintSystem

/-- Canonical `Pi_CCS` outputs at the one verifier-selected new point. -/
def outputs
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (input : Input Structure PublicInput Point Evaluation Commitment)
    (witness : Witness Assignment Point Scalar) :
    Fin arity.total ->
      CE.Instance Structure PublicInput Point Evaluation Commitment :=
  PiCCS.honestOutputs profile.semantics input witness.assignments witness.point

/-- Strong-set membership for every `Pi_RLC` challenge. -/
def ChallengesValid
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (witness : Witness Assignment Point Scalar) : Prop :=
  forall index,
    profile.rlcAlgebra.challengeValid (witness.challenges index)

/-- Canonical private opening of the internal combined parent. -/
def combinedAssignment
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (witness : Witness Assignment Point Scalar) : Assignment :=
  PiRLC.combinedWitness profile.rlcAlgebra witness.challenges
    witness.assignments

/-- Canonical internal `CE(B)` parent. -/
def parentOf
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (input : Input Structure PublicInput Point Evaluation Commitment)
    (witness : Witness Assignment Point Scalar) :
    CE.Instance Structure PublicInput Point Evaluation Commitment :=
  PiRLC.combinedOutput profile.rlcAlgebra (systemOf input) witness.point
    (outputs profile input witness) witness.challenges

/-- Canonical public children obtained by exact base-two decomposition. -/
def childrenOf
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (input : Input Structure PublicInput Point Evaluation Commitment)
    (witness : Witness Assignment Point Scalar) :
    Output Structure PublicInput Point Evaluation Commitment :=
  PiDEC.childrenOf profile.decAlgebra (parentOf profile input witness)
    (combinedAssignment profile witness)

/-- One indexed paper realization. `input`, `output`, and `witness` are fixed
outside the fields so later removal witnesses cannot change them. -/
structure Realization
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (input : Input Structure PublicInput Point Evaluation Commitment)
    (output : Output Structure PublicInput Point Evaluation Commitment)
    (witness : Witness Assignment Point Scalar) : Prop where
  sourceFresh : forall index, (input.source index).stage = .fresh
  sourceHolds : forall index,
    (input.source index).Holds profile.semantics productionGlobalParams
      (witness.assignments index)
  commonStructure : forall index,
    (input.source index).constraintSystem = systemOf input
  runningCommonPoint : forall left right,
    (input.running left).point = (input.running right).point
  newPointValid :
    profile.semantics.evaluationPointValid (systemOf input) witness.point
  challengesValid : ChallengesValid profile witness
  piDecAccepted : PiDEC.PaperVerifier.OutputAccepted profile.decAlgebra
    profile.decPublicInputSplit profile.decEvaluationArity
    (parentOf profile input witness) output

/-- Independent public-verifier acceptance relation. Child opening/membership
is a soundness premise of the reduction, not a check performed by `Pi_DEC`. -/
def Transition
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (input : Input Structure PublicInput Point Evaluation Commitment)
    (output : Output Structure PublicInput Point Evaluation Commitment) : Prop :=
  exists witness : Witness Assignment Point Scalar,
    Realization profile input output witness

/-- Conditional honest completeness: valid sources and challenges construct
the canonical public target, which satisfies the exact operational `Pi_DEC`
verifier. Challenge generation and sampler success are separate boundaries. -/
theorem complete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar)
    (input : Input Structure PublicInput Point Evaluation Commitment)
    (witness : Witness Assignment Point Scalar)
    (sourceFresh : forall index, (input.source index).stage = .fresh)
    (sourceHolds : forall index,
      (input.source index).Holds profile.semantics productionGlobalParams
        (witness.assignments index))
    (commonStructure : forall index,
      (input.source index).constraintSystem = systemOf input)
    (runningCommonPoint : forall left right,
      (input.running left).point = (input.running right).point)
    (newPointValid :
      profile.semantics.evaluationPointValid (systemOf input) witness.point)
    (challengesValid : ChallengesValid profile witness) :
    Transition profile input (childrenOf profile input witness) := by
  have outputsValid : forall index,
      CE.Holds profile.semantics productionGlobalParams
        (outputs profile input witness index) (witness.assignments index) := by
    apply PiCCS.product_complete profile.semantics productionGlobalParams arity
      input witness.assignments witness.point sourceFresh sourceHolds
    intro index
    rw [commonStructure index]
    exact newPointValid
  have parentValid :
      CE.Holds profile.semantics productionGlobalParams
        (parentOf profile input witness) (combinedAssignment profile witness) := by
    apply PiRLC.combinedOutput_holds profile.semantics productionGlobalParams
      profile.rlcAlgebra arity (systemOf input) witness.point
      (outputs profile input witness) witness.challenges witness.assignments
    · intro index
      rfl
    · intro index
      simpa [outputs, PiCCS.honestOutputs, PiCCS.honestOutput] using
        commonStructure index
    · intro index
      rfl
    · exact challengesValid
    · exact outputsValid
    · exact newPointValid
  have piDec := PiDEC.PaperVerifier.output_complete profile.semantics
    productionGlobalParams profile.decAlgebra profile.decPublicInputSplit
    profile.decEvaluationArity
    (parentOf profile input witness)
    (combinedAssignment profile witness) rfl parentValid
  refine ⟨witness, {
    sourceFresh := sourceFresh
    sourceHolds := sourceHolds
    commonStructure := commonStructure
    runningCommonPoint := runningCommonPoint
    newPointValid := newPointValid
    challengesValid := challengesValid
    piDecAccepted := ?_
  }⟩
  simpa only [childrenOf] using piDec.1

namespace Realization

/-- Every canonical `Pi_CCS` output has the corresponding authoritative
fresh-stage CE opening. -/
theorem outputsHold
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    {input : Input Structure PublicInput Point Evaluation Commitment}
    {output : Output Structure PublicInput Point Evaluation Commitment}
    {witness : Witness Assignment Point Scalar}
    (holds : Realization profile input output witness) :
    forall index,
      CE.Holds profile.semantics productionGlobalParams
        (outputs profile input witness index) (witness.assignments index) := by
  apply PiCCS.product_complete profile.semantics productionGlobalParams arity
    input witness.assignments witness.point holds.sourceFresh holds.sourceHolds
  intro index
  rw [holds.commonStructure index]
  exact holds.newPointValid

/-- The internal combined parent has the canonical challenge-folded opening. -/
theorem parentOpening
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    {input : Input Structure PublicInput Point Evaluation Commitment}
    {output : Output Structure PublicInput Point Evaluation Commitment}
    {witness : Witness Assignment Point Scalar}
    (holds : Realization profile input output witness) :
    CE.Holds profile.semantics productionGlobalParams
      (parentOf profile input witness) (combinedAssignment profile witness) := by
  apply PiRLC.combinedOutput_holds profile.semantics productionGlobalParams
    profile.rlcAlgebra arity (systemOf input) witness.point
    (outputs profile input witness) witness.challenges witness.assignments
  · intro index
    rfl
  · intro index
    simpa [outputs, PiCCS.honestOutputs, PiCCS.honestOutput] using
      holds.commonStructure index
  · intro index
    rfl
  · exact holds.challengesValid
  · exact holds.outputsHold
  · exact holds.newPointValid

/-- Exact operational `Pi_DEC` acceptance for the derived parent and candidate
target. Child public inputs and copied fields are verifier-computed; child
commitments/evaluations remain prover supplied. -/
theorem outputAccepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    {input : Input Structure PublicInput Point Evaluation Commitment}
    {output : Output Structure PublicInput Point Evaluation Commitment}
    {witness : Witness Assignment Point Scalar}
    (holds : Realization profile input output witness) :
    PiDEC.PaperVerifier.OutputAccepted profile.decAlgebra
      profile.decPublicInputSplit profile.decEvaluationArity
      (parentOf profile input witness) output :=
  holds.piDecAccepted

/-- Compatibility projection into the older public-recomposition relation. -/
theorem outputRecompositionAccepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    {input : Input Structure PublicInput Point Evaluation Commitment}
    {output : Output Structure PublicInput Point Evaluation Commitment}
    {witness : Witness Assignment Point Scalar}
    (holds : Realization profile input output witness) :
    PiDEC.Accepted profile.decAlgebra {
      parent := parentOf profile input witness
      children := output
    } :=
  holds.piDecAccepted.toRecompositionAccepted

/-- Valid openings of an accepted target reconstruct a valid combined parent
opening. This is the `Pi_DEC` reduction-of-knowledge direction and does not
claim that the children are the deterministic honest split. -/
theorem recomposedParentOpening
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    {input : Input Structure PublicInput Point Evaluation Commitment}
    {output : Output Structure PublicInput Point Evaluation Commitment}
    {witness : Witness Assignment Point Scalar}
    (holds : Realization profile input output witness)
    (childAssignments : Fin productionGlobalParams.k -> Assignment)
    (childrenValid : forall child,
      CE.Holds profile.semantics productionGlobalParams (output child)
        (childAssignments child)) :
    CE.Holds profile.semantics productionGlobalParams
      (parentOf profile input witness)
      (profile.decAlgebra.recomposeAssignment childAssignments) := by
  exact PiDEC.reduce_knowledge profile.semantics productionGlobalParams
    profile.decAlgebra {
      parent := parentOf profile input witness
      children := output
    } childAssignments (by decide) holds.outputRecompositionAccepted
      childrenValid

/-- The source-derived combined opening equals the target-child
recomposition, or the two valid openings expose the standard parent binding
collision. Noncanonical but exactly recomposing child families take the left
branch. -/
theorem parentOpening_eq_recompose_or_bindingCollision
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    {input : Input Structure PublicInput Point Evaluation Commitment}
    {output : Output Structure PublicInput Point Evaluation Commitment}
    {witness : Witness Assignment Point Scalar}
    (holds : Realization profile input output witness)
    (childAssignments : Fin productionGlobalParams.k -> Assignment)
    (childrenValid : forall child,
      CE.Holds profile.semantics productionGlobalParams (output child)
        (childAssignments child)) :
    combinedAssignment profile witness =
        profile.decAlgebra.recomposeAssignment childAssignments \/
      Nonempty (PiDEC.ParentOpeningBindingCollision profile.semantics
        productionGlobalParams
        (parentOf profile input witness).commitment) := by
  exact PiDEC.accepted_parent_eq_recompose_or_bindingCollision
    profile.semantics productionGlobalParams profile.decAlgebra {
      parent := parentOf profile input witness
      children := output
    } (combinedAssignment profile witness) childAssignments
      holds.outputRecompositionAccepted holds.parentOpening childrenValid

end Realization

namespace Transition

/-- Every paper transition contains one witness for which the target passes
the exact operational `Pi_DEC` verifier against the source-derived parent. -/
theorem outputAccepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {profile : Profile Structure Assignment PublicInput Point Evaluation
      Commitment Scalar}
    {input : Input Structure PublicInput Point Evaluation Commitment}
    {output : Output Structure PublicInput Point Evaluation Commitment}
    (transition : Transition profile input output) :
    exists witness : Witness Assignment Point Scalar,
      PiDEC.PaperVerifier.OutputAccepted profile.decAlgebra
        profile.decPublicInputSplit profile.decEvaluationArity
        (parentOf profile input witness) output := by
  rcases transition with ⟨witness, realized⟩
  exact ⟨witness, realized.piDecAccepted⟩

end Transition

end Nightstream.SuperNeo.Folding.Nifs.PaperProfile
