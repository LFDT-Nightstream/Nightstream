import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputProduct
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth.Carried
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.SemanticAdapter

/-!
Generic `PiCCS.Attempt` view of one accepted production block/lane replay.

Assurance tier: model-level.

Owns: the proof-only attempt whose public inputs and materialized outputs are
the concrete product, while FE and NC semantic ghosts are independently
recomputed from the same fixed physical certificate.

Does not own: Fiat--Shamir, challenge probability, extraction, commitment
binding, PiRLC, PiDEC, Rust, R1CS, costs, or rows.

Emits constraints: no.

| Stage path | Owned equation | Authority |
|---|---|---|
| `piccs.split_nc.block_lane.semantic_attempt` | one fixed certificate induces accepted independent FE/NC semantic instances | derived |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.SemanticAttempt

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

universe uCommitment uVerifierKey uInput uState

private abbrev sumcheckOps :=
  ConcreteCarrier.extensionOps.toOps.toSymbolic

/-- The physical certificate with only its dependent FE input index transported
to the authoritative source projection. -/
abbrev SourceCertificate
    {shape : SemanticShape}
    {domains : Domains}
    {Input : Type uInput}
    {VerifierKey : Type uVerifierKey}
    (projectInput : Input -> PublicInput shape)
    (statement : Statement VerifierKey Input)
    (data : Data shape)
    (certificate : Certificate (projectInput statement.input) domains)
    (inputBound : projectInput statement.input = PublicInput.ofSources data) :=
  certificateAtSources data certificate inputBound

private theorem feSumCheckAccepted_atSources
    {shape : SemanticShape}
    {domains : Domains}
    {input : PublicInput shape}
    (data : Data shape)
    (certificate : Certificate input domains)
    (bound : input = PublicInput.ofSources data)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (coins : Polynomial.Fe.Coins shape domains.fe)
    (point : Polynomial.Fe.Point shape domains.fe)
    (message : OutputMessage shape)
    (accepted :
      SumCheck.Fe.Accepted
        (Polynomial.Fe.initial profile input coins)
        (Polynomial.Fe.terminalFromMessage profile input coins point message)
        point certificate.fe) :
    SumCheck.Fe.Accepted
      (Polynomial.Fe.initial profile (PublicInput.ofSources data) coins)
      (Polynomial.Fe.terminalFromMessage profile (PublicInput.ofSources data)
        coins point message)
      point (certificateAtSources data certificate bound).fe := by
  subst input
  exact accepted

private theorem ncCertificateAtSources
    {shape : SemanticShape}
    {domains : Domains}
    {input : PublicInput shape}
    (data : Data shape)
    (certificate : Certificate input domains)
    (bound : input = PublicInput.ofSources data) :
    (certificateAtSources data certificate bound).nc.toSumCheck =
      certificate.nc.toSumCheck := by
  subst input
  rfl

/-- One proof-only generic attempt. Its FE and NC instances share the actual
transcript-derived coin record but are built independently from their own
assignment-indexed polynomials. -/
def attempt
    {shape : SemanticShape}
    {domains : Domains}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {State : Type uState}
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
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity) :
    PiCCS.Attempt
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation Commitment K K params arity :=
  let pre := derivePreSumcheck schedule priorState statement
  let execution :=
    derive projectInput schedule priorState profile statement certificate
  let sourceCertificate :=
    certificateAtSources data certificate inputBound
  {
    inputs := input
    outputs :=
      OutputProduct.materialize publicRingColumns publicFits alignment input
        execution.fePoint.row certificate.output
    fe :=
      SumCheck.SemanticAdapter.feInstance profile data
        pre.challenges.feCoins execution.fePoint certificate.output
        sourceCertificate.fe challengeSetSize
    nc :=
      SumCheck.SemanticAdapter.ncInstance covers data
        pre.challenges.ncCoins execution.ncPoint certificate.output
        sourceCertificate.nc.toSumCheck challengeSetSize
  }

/-- Concrete replay acceptance and verifier-owned output binding transport to
the generic `PiCCS.Accepted` relation for the semantic attempt. -/
theorem accepted_of_replay
    {shape : SemanticShape}
    {domains : Domains}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {State : Type uState}
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
    (accepted :
      Accepted projectInput schedule priorState profile statement certificate)
    (outputBound :
      OutputBound covers data
        (derive projectInput schedule priorState profile statement certificate)
        certificate.output) :
    PiCCS.Accepted sumcheckOps
      (attempt covers projectInput schedule priorState profile statement data
        inputBound certificate challengeSetSize publicRingColumns publicFits
        alignment input) := by
  let pre := derivePreSumcheck schedule priorState statement
  let polynomialInput := projectInput statement.input
  let execution :=
    derive projectInput schedule priorState profile statement certificate
  let sourceCertificate :=
    certificateAtSources data certificate inputBound
  change
    Fe.Accepted (feMachine schedule
        (Polynomial.Fe.initial profile polynomialInput
          pre.challenges.feCoins))
        pre.state profile polynomialInput
        pre.challenges.feCoins certificate.output certificate.fe ∧
      Nc.BlockLane.Accepted (ncMachine schedule)
        (Transcript.Fe.derive
          (feMachine schedule
            (Polynomial.Fe.initial profile polynomialInput
              pre.challenges.feCoins))
          pre.state certificate.fe).finalState
        pre.challenges.ncCoins certificate.output certificate.nc at accepted
  rcases accepted with ⟨feAccepted, ncAccepted⟩
  have feAcceptedPhysical :
      SumCheck.Fe.Accepted
        (Polynomial.Fe.initial profile polynomialInput
          pre.challenges.feCoins)
        (Polynomial.Fe.terminalFromMessage profile polynomialInput
          pre.challenges.feCoins execution.fePoint certificate.output)
        execution.fePoint certificate.fe := by
    exact feAccepted
  have feAccepted' :
      SumCheck.Fe.Accepted
        (Polynomial.Fe.initial profile (PublicInput.ofSources data)
          pre.challenges.feCoins)
        (Polynomial.Fe.terminalFromMessage profile
          (PublicInput.ofSources data) pre.challenges.feCoins execution.fePoint
          certificate.output)
        execution.fePoint sourceCertificate.fe := by
    exact feSumCheckAccepted_atSources data certificate inputBound profile
      pre.challenges.feCoins execution.fePoint certificate.output
      feAcceptedPhysical
  have ncAccepted' :
      SumCheck.Nc.Accepted
        Polynomial.Nc.BlockLane.InitialSum.claimedInitial
        execution.ncPoint.coordinates
        (Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
          certificate.output pre.challenges.ncCoins execution.ncPoint)
        sourceCertificate.nc.toSumCheck := by
    rw [ncCertificateAtSources data certificate inputBound]
    exact ncAccepted
  refine ⟨?_, ?_, ?_⟩
  · simpa only [attempt] using
      OutputProduct.outputProduct_shape publicRingColumns publicFits alignment
        input execution.fePoint.row certificate.output
        (SumCheck.SemanticAdapter.feInstance profile data
          pre.challenges.feCoins execution.fePoint certificate.output
          sourceCertificate.fe challengeSetSize)
        (SumCheck.SemanticAdapter.ncInstance covers data
          pre.challenges.ncCoins execution.ncPoint certificate.output
          sourceCertificate.nc.toSumCheck challengeSetSize)
        (InputAuthority.BoundToSources.sourceFresh publicRingColumns publicFits
          commit data alignment input inputAuthority)
  · simpa only [attempt] using
      SumCheck.SemanticAdapter.feAccepted_implies_genericAccepted profile data
        pre.challenges.feCoins execution.fePoint certificate.output
        sourceCertificate.fe challengeSetSize feAccepted' outputBound.1
  · simpa only [attempt] using
      SumCheck.SemanticAdapter.ncAccepted_implies_genericAccepted covers data
        pre.challenges.ncCoins execution.ncPoint certificate.output
        sourceCertificate.nc.toSumCheck challengeSetSize ncAccepted' outputBound.2

/-- The fixed physical replay has an assignment-indexed arithmetization for
the exact authoritative source vector. FE and NC truth are derived
independently; no joint equality eliminates either mixing-root event. -/
theorem arithmetization_of_replay
    {shape : SemanticShape}
    {domains : Domains}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {State : Type uState}
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
    (accepted :
      Accepted projectInput schedule priorState profile statement certificate)
    (outputBound :
      OutputBound covers data
        (derive projectInput schedule priorState profile statement certificate)
        certificate.output) :
    PiCCS.Arithmetization
      (productSemantics publicRingColumns publicFits commit) params sumcheckOps
      (attempt covers projectInput schedule priorState profile statement data
        inputBound certificate challengeSetSize publicRingColumns publicFits
        alignment input)
      (InputAuthority.productAssignments data alignment) := by
  let pre := derivePreSumcheck schedule priorState statement
  let polynomialInput := projectInput statement.input
  let execution :=
    derive projectInput schedule priorState profile statement certificate
  let sourceCertificate :=
    certificateAtSources data certificate inputBound
  change
    Fe.Accepted (feMachine schedule
        (Polynomial.Fe.initial profile polynomialInput
          pre.challenges.feCoins))
        pre.state profile polynomialInput
        pre.challenges.feCoins certificate.output certificate.fe ∧
      Nc.BlockLane.Accepted (ncMachine schedule)
        (Transcript.Fe.derive
          (feMachine schedule
            (Polynomial.Fe.initial profile polynomialInput
              pre.challenges.feCoins))
          pre.state certificate.fe).finalState
        pre.challenges.ncCoins certificate.output certificate.nc at accepted
  rcases accepted with ⟨feAccepted, ncAccepted⟩
  have feAcceptedPhysical :
      SumCheck.Fe.Accepted
        (Polynomial.Fe.initial profile polynomialInput
          pre.challenges.feCoins)
        (Polynomial.Fe.terminalFromMessage profile polynomialInput
          pre.challenges.feCoins execution.fePoint certificate.output)
        execution.fePoint certificate.fe := by
    exact feAccepted
  have feAccepted' :
      SumCheck.Fe.Accepted
        (Polynomial.Fe.initial profile (PublicInput.ofSources data)
          pre.challenges.feCoins)
        (Polynomial.Fe.terminalFromMessage profile
          (PublicInput.ofSources data) pre.challenges.feCoins execution.fePoint
          certificate.output)
        execution.fePoint sourceCertificate.fe := by
    exact feSumCheckAccepted_atSources data certificate inputBound profile
      pre.challenges.feCoins execution.fePoint certificate.output
      feAcceptedPhysical
  have ncAccepted' :
      SumCheck.Nc.Accepted
        Polynomial.Nc.BlockLane.InitialSum.claimedInitial
        execution.ncPoint.coordinates
        (Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
          certificate.output pre.challenges.ncCoins execution.ncPoint)
        sourceCertificate.nc.toSumCheck := by
    rw [ncCertificateAtSources data certificate inputBound]
    exact ncAccepted
  refine {
    feTruthPath := ?_
    ncTruthPath := ?_
    feClaimTrue_of_payloads := ?_
    ncClaimTrue_of_norms := ?_
  }
  · simpa only [attempt] using
      SumCheck.SemanticAdapter.feAccepted_implies_truthPath profile data
        pre.challenges.feCoins execution.fePoint certificate.output
        sourceCertificate.fe challengeSetSize feAccepted' outputBound.1
  · simpa only [attempt] using
      SumCheck.SemanticAdapter.ncAccepted_implies_truthPath covers data
        pre.challenges.ncCoins execution.ncPoint certificate.output
        sourceCertificate.nc.toSumCheck challengeSetSize ncAccepted' outputBound.2
  · intro payloads
    change
      ProductTruth.PayloadsHold publicRingColumns publicFits commit data
        alignment input at payloads
    have feTruth : Semantics.Fe.Truth data := by
      exact ⟨
        ProductTruth.freshTruth_of_payloads publicRingColumns publicFits commit
          data alignment input inputAuthority payloads,
        ProductTruth.carriedTruth_of_payloads publicRingColumns publicFits
          commit data alignment input inputAuthority payloads⟩
    simpa only [attempt] using
      SumCheck.SemanticAdapter.feClaimTrue_of_truth profile data
        pre.challenges.feCoins execution.fePoint certificate.output
        sourceCertificate.fe challengeSetSize feTruth
  · intro norms
    change
      ∀ source,
        (productSemantics publicRingColumns publicFits commit).normBounded
          params.b
          (InputAuthority.productAssignments data alignment source) at norms
    have ncTruth : Semantics.Nc.Truth data :=
      ProductTruth.ncTruth_of_norms publicRingColumns publicFits commit data
        alignment freshBound norms
    simpa only [attempt] using
      SumCheck.SemanticAdapter.ncClaimTrue_of_truth covers data
        pre.challenges.ncCoins execution.ncPoint certificate.output
        sourceCertificate.nc.toSumCheck challengeSetSize ncTruth

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.SemanticAttempt
