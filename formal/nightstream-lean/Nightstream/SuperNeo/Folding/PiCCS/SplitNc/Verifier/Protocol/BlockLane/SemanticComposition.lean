import Nightstream.SuperNeo.Folding.Composition.ReferenceArithmetization
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth.Ambient
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.SemanticAttempt
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement

/-!
Composition of one concrete block/lane replay with `Pi_RLC` and `Pi_DEC`.

Assurance tier: model-level.

Owns: construction of the assignment-indexed FE and NC semantic views,
transport of the verifier-materialized output to a concrete ambient opening,
and replacement of the generic rewind-arithmetization callback by commitment
uniqueness against that opening.

Does not own: probability bounds for the named algebraic events,
Fiat--Shamir, commitment security, a concrete extractor, Rust, R1CS, costs, or
rows.

Emits constraints: no.

| Stage path | Owned equation | Authority |
|---|---|---|
| `piccs.split_nc.block_lane.semantic_composition` | accepted FE/NC replay composes with PiRLC/PiDEC without a rewind arithmetization callback | derived modulo named events |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.SemanticComposition

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

universe uCommitment uVerifierKey uInput uState uScalar

private abbrev sumcheckOps :=
  ConcreteCarrier.extensionOps.toOps.toSymbolic

/-- Concrete ideal-interactive composition for the actual FE-then-block/lane
NC replay. The theorem has no caller-supplied arithmetization callback:
arithmetization is reconstructed independently for FE and NC from the fixed
wire certificate and the authoritative source assignments. -/
theorem fold_extraction_or_bad_event
    {shape : SemanticShape}
    {domains : Domains}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {State : Type uState}
    {Scalar : Type uScalar}
    (covers : domains.nc.Covers shape)
    (projectInput : Input -> PublicInput shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (data : Data shape)
    (inputBound : projectInput statement.input = PublicInput.ofSources data)
    (certificate : Certificate (projectInput statement.input) domains)
    (challengeSetSize : Nat)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (inputAuthority :
      InputAuthority.BoundToSources publicRingColumns publicFits commit data
        alignment input)
    (freshBound : params.b = 2)
    (freshLeAmbient : params.b <= params.q / 2)
    (paper : SplitNc.Semantics.Paper.Holds data)
    (accepted :
      Accepted projectInput schedule priorState profile statement certificate)
    (outputBound :
      OutputBound covers data
        (derive projectInput schedule priorState profile statement certificate)
        certificate.output)
    (rlcAlgebra : PiRLC.Algebra
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (SourceAssignment shape)
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation Commitment Scalar
      (productSemantics publicRingColumns publicFits commit) params)
    (decAlgebra : PiDEC.Algebra
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (SourceAssignment shape)
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation Commitment
      (productSemantics publicRingColumns publicFits commit) params)
    (bindingOps :
      PiRLC.RelaxedBindingOps (SourceAssignment shape) Commitment Scalar)
    (sampling : PiRLC.SamplingBoundary arity.total)
    (rlcAttempt : PiRLC.Attempt
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation Commitment Scalar params arity)
    (decAttempt : PiDEC.Attempt
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation Commitment params)
    (finalAssignments : Fin params.k -> SourceAssignment shape)
    (kPositive : 0 < params.k)
    (sameRlcInputs : forall i,
      rlcAttempt.inputs i =
        (SemanticAttempt.attempt covers projectInput schedule priorState
          profile statement data inputBound certificate challengeSetSize
          publicRingColumns publicFits alignment input).outputs i)
    (sameDecParent : decAttempt.parent = rlcAttempt.output)
    (rlcAccepted : PiRLC.Accepted rlcAlgebra rlcAttempt)
    (decAccepted : PiDEC.Accepted decAlgebra decAttempt)
    (finalValid : forall i,
      CE.Holds (productSemantics publicRingColumns publicFits commit) params
        (decAttempt.children i) (finalAssignments i))
    (extractor : Composition.WeakExtractor
      (productSemantics publicRingColumns publicFits commit) params rlcAlgebra
      rlcAttempt sampling)
    (uniqueness : PiRLC.UniquenessBridge
      (productSemantics publicRingColumns publicFits commit) params bindingOps
      (n := arity.total)) :
    let ccsAttempt :=
      SemanticAttempt.attempt covers projectInput schedule priorState profile
        statement data inputBound certificate challengeSetSize
        publicRingColumns publicFits alignment input
    Nonempty (Composition.ExtractedBatch
      (productSemantics publicRingColumns publicFits commit) params
      ccsAttempt) ∨
      Composition.BadEvent
        (productSemantics publicRingColumns publicFits commit) params
        bindingOps sampling ccsAttempt rlcAttempt.inputs := by
  let ccsAttempt :=
    SemanticAttempt.attempt covers projectInput schedule priorState profile
      statement data inputBound certificate challengeSetSize
      publicRingColumns publicFits alignment input
  have ccsAccepted : PiCCS.Accepted sumcheckOps ccsAttempt :=
    SemanticAttempt.accepted_of_replay covers projectInput schedule priorState
      profile statement data inputBound certificate challengeSetSize
      publicRingColumns publicFits commit alignment input inputAuthority
      accepted outputBound
  have referenceArithmetization :
      PiCCS.Arithmetization
        (productSemantics publicRingColumns publicFits commit) params
        sumcheckOps ccsAttempt
        (InputAuthority.productAssignments data alignment) :=
    SemanticAttempt.arithmetization_of_replay covers projectInput schedule
      priorState profile statement data inputBound certificate
      challengeSetSize publicRingColumns publicFits commit alignment input
      inputAuthority freshBound accepted outputBound
  have freshBound_eq_two : NormStage.bound params .fresh = 2 := by
    simpa [NormStage.bound] using freshBound
  have productHolds :
      ProductHolds publicRingColumns publicFits commit ccsAttempt.outputs
        (InputAuthority.productAssignments data alignment) := by
    exact OutputRefinement.materializedOutputsHold_of_yRingEq
      publicRingColumns publicFits commit data alignment input
      (derive projectInput schedule priorState profile statement
        certificate).fePoint.row
      certificate.output freshBound_eq_two paper inputAuthority outputBound.1
  have outputFresh : forall source,
      (ccsAttempt.outputs source).stage = .fresh := by
    intro source
    exact OutputProduct.materialize_stage publicRingColumns publicFits
      alignment input
      (derive projectInput schedule priorState profile statement
        certificate).fePoint.row
      certificate.output source
  have ccsAmbient :
      PiRLC.AmbientOpenings
        (productSemantics publicRingColumns publicFits commit) params
        ccsAttempt.outputs
        (InputAuthority.productAssignments data alignment) :=
    ProductTruth.ambientOpenings_of_productHolds publicRingColumns publicFits
      commit ccsAttempt.outputs
      (InputAuthority.productAssignments data alignment) outputFresh
      freshLeAmbient productHolds
  have referenceAmbient :
      PiRLC.AmbientOpenings
        (productSemantics publicRingColumns publicFits commit) params
        rlcAttempt.inputs
        (InputAuthority.productAssignments data alignment) := by
    intro source
    rw [sameRlcInputs source]
    exact ccsAmbient source
  exact Composition.ReferenceArithmetization.fold_extraction_or_bad_event
    (productSemantics publicRingColumns publicFits commit) params sumcheckOps
    rlcAlgebra decAlgebra bindingOps arity sampling ccsAttempt rlcAttempt
    decAttempt finalAssignments (InputAuthority.productAssignments data
      alignment) kPositive sameRlcInputs sameDecParent ccsAccepted rlcAccepted
    decAccepted finalValid extractor uniqueness referenceAmbient
    referenceArithmetization

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.SemanticComposition
