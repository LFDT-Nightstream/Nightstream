import NightstreamFPrime.Spec.Profile
import NightstreamFPrime.Spec.Folding.PiCCS.PaperProduct
import NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier
import NightstreamFPrime.Spec.Folding.PiRLC

/-! SuperNeo Sections 7.3–7.5. -/

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperProfile

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar

def arity : BatchArity productionGlobalParams :=
  BatchArity.active productionGlobalParams 1 (by decide) (by decide)

@[simp] theorem arity_freshCount : arity.freshCount = 1 := rfl

@[simp] theorem arity_runningCount :
    arity.mode.count productionGlobalParams = 16 := rfl

@[simp] theorem arity_total : arity.total = 17 := rfl

@[simp] theorem outputCount : productionGlobalParams.k = 16 := rfl

@[simp] theorem baseNormBound : productionGlobalParams.b = 2 := rfl

abbrev Input
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment) :=
  PiCCS.PaperProduct.InputProduct Structure PublicInput Point Evaluation Commitment
    productionGlobalParams arity

abbrev Output
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment) :=
  Fin productionGlobalParams.k ->
    CE.Instance Structure PublicInput Point Evaluation Commitment

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

structure Witness
    (Assignment : Type uAssignment)
    (Point : Type uPoint)
    (Scalar : Type uScalar) where
  assignments : Fin arity.total -> Assignment
  point : Point
  challenges : Fin arity.total -> Scalar

def firstFresh : Fin arity.freshCount := ⟨0, by decide⟩

def systemOf
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (input : Input Structure PublicInput Point Evaluation Commitment) :
    Structure :=
  (input.fresh firstFresh).constraintSystem

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
  PiCCS.PaperProduct.honestOutputs profile.semantics input
    witness.assignments witness.point

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
    apply PiCCS.PaperProduct.product_complete profile.semantics
      productionGlobalParams arity
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
      simpa [outputs, PiCCS.PaperProduct.honestOutputs,
        PiCCS.PaperProduct.honestOutput] using
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
  apply PiCCS.PaperProduct.product_complete profile.semantics
    productionGlobalParams arity
    input witness.assignments witness.point holds.sourceFresh holds.sourceHolds
  intro index
  rw [holds.commonStructure index]
  exact holds.newPointValid

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
    simpa [outputs, PiCCS.PaperProduct.honestOutputs,
      PiCCS.PaperProduct.honestOutput] using
      holds.commonStructure index
  · intro index
    rfl
  · exact holds.challengesValid
  · exact holds.outputsHold
  · exact holds.newPointValid

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

end NightstreamFPrime.Spec.Folding.Nifs.PaperProfile
