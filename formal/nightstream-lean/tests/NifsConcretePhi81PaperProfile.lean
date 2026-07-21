import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive
import Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution

/-!
Focused regressions for the independent fixed-active paper profile and its
concrete Phi81 refinement.

| Boundary | Property under test | Failure caught |
|---|---|---|
| abstract profile | exact `1 + 14 -> 14`, `b = 2` carrier | conflating HyperNova function slots, fresh arity, or SuperNeo fanout |
| abstract realization | fixed source, output, assignments, point, and challenges | existential witness substitution in later necessity proofs |
| abstract tail | paper output computes child public/copy fields and checks only commitment/evaluation recomposition | confusing relaxed recomposition with Section-7.5 acceptance |
| concrete refinement | Split-NC paper truth and source binding instantiate the abstract relation | treating the concrete verifier as semantic authority |
| lifecycle split | polynomial input, parent caches, and canonical-child strengthening are outside paper acceptance | counting implementation authority as intrinsic paper obligations |
-/

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

namespace Nightstream.SuperNeo.Folding.Nifs.PaperProfile

example : arity.freshCount = 1 := rfl

example : arity.mode.count productionGlobalParams = 14 := rfl

example : arity.total = 15 := rfl

example : productionGlobalParams.k = 14 := rfl

example : productionGlobalParams.b = 2 := rfl

#check Profile
#check Input
#check Output
#check Witness
#check systemOf
#check outputs
#check ChallengesValid
#check combinedAssignment
#check parentOf
#check childrenOf
#check Realization
#check Transition
#check complete
#check Realization.outputsHold
#check Realization.parentOpening
#check Realization.outputAccepted
#check Realization.recomposedParentOpening
#check Realization.parentOpening_eq_recompose_or_bindingCollision
#check Transition.outputAccepted
#check PiDEC.PaperVerifier.PublicInputSplit
#check PiDEC.PaperVerifier.EvaluationArity
#check PiDEC.PaperVerifier.ChildMessage
#check PiDEC.PaperVerifier.children
#check PiDEC.PaperVerifier.OutputAccepted
#check PiDEC.PaperVerifier.OutputAccepted.childPublicInput_eq
#check PiDEC.PaperVerifier.OutputAccepted.publicInputs_eq_of_parentPublicInput_eq
#check PiDEC.PaperVerifier.OutputAccepted.childEvaluations_size
#check PiDEC.PaperVerifier.OutputAccepted.parentEvaluations_size
#check PiDEC.PaperVerifier.OutputAccepted.toRecompositionAccepted
#check PiDEC.PaperVerifier.output_complete
#check PiDEC.PaperVerifier.reduce_knowledge

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar

example
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
    (realized : Realization profile input output witness) :
    PiDEC.PaperVerifier.OutputAccepted profile.decAlgebra
      profile.decPublicInputSplit profile.decEvaluationArity
      (parentOf profile input witness) output :=
  realized.piDecAccepted

example
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
    (realized : Realization profile input output witness) :
    PiDEC.Accepted profile.decAlgebra {
      parent := parentOf profile input witness
      children := output
    } :=
  realized.outputRecompositionAccepted

end Nightstream.SuperNeo.Folding.Nifs.PaperProfile

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

example : arity.freshCount = 1 := rfl

example : arity.mode.count productionGlobalParams = 14 := rfl

example : arity.total = 15 := rfl

#check PaperProfile.Profile
#check PaperProfile.Source
#check PaperProfile.Target
#check PaperProfile.Witness
#check PaperProfile.toGenericProfile
#check PaperProfile.toGenericWitness
#check PaperProfile.Realization
#check PaperProfile.Realization.toGeneric
#check PaperProfile.Transition
#check PaperProfile.complete
#check PaperProfile.Realization.outputAccepted
#check PaperProfile.Realization.parentOpening_eq_recompose_or_bindingCollision
#check Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput_project
#check Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput_recompose
#check Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PaperVerifier.publicInputSplit
#check Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PaperVerifier.evaluationArity
#check paperProfileOf
#check PaperDecomposition
#check resultTransition_iff_exists_paperDecomposition
#check ResultTransition.toPaperProfile

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

/- The old recomposition predicate permits a child-public-input substitution.
This is a regression witness separating it from the operational paper
verifier, not evidence that Section 7.5 accepts the substituted family. -/
#check Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.Fixture.rightAccepted_but_notCanonical
#check Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.Fixture.leftPaperAccepted
#check Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.Fixture.rightNotPaperAccepted
#check Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution.Fixture.trailingEvaluationChildren_notPaperAccepted
