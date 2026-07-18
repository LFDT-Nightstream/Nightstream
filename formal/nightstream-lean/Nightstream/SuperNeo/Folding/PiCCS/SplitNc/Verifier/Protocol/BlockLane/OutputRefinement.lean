import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement

/-!
Canonical block×lane Π_CCS output refinement into the CE product consumed by
Π_RLC.

Assurance tier: model-level.

Owns: composition of canonical protocol acceptance, explicit input/output
authority, CE materialization at the verifier-derived FE row point, named
phase bad events, and honest completeness of that handoff.

Does not own: Poseidon2 encoding, Fiat--Shamir probability, Π_RLC, Π_DEC,
Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: acceptance alone authorizes neither the source projection
nor the raw output. `inputBound` ties the enriched statement to independent
sources; `OutputBound` ties `yRing` and packed `yZcol` to those sources at the
two verifier-derived points. CE materialization consumes only the proved
`yRing` child. Packed `yZcol` closes the Split-NC terminal only: it is not a
paper CE field, is not carried into Π_RLC/Π_DEC, and is never promoted through
a digest.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.handoff.block_lane.acceptance` | exact FE then 3+6-round NC replay accepts | checked | `Accepted` premise |
| `nifs.pi_ccs.handoff.block_lane.input` | projected public input equals independent sources | checked | `inputBound` premise |
| `nifs.pi_ccs.handoff.block_lane.output.y_ring` | CE evaluations equal source evaluations at derived FE row | checked then derived | `accepted_and_outputBound_implies_outputsHold_or_badEvent` |
| `nifs.pi_ccs.handoff.block_lane.output.y_zcol` | all active packed lanes bind at derived block point | checked security boundary | `OutputBound` premise |
| `nifs.pi_ccs.handoff.block_lane.opening` | commitment, public input, norm, and evaluations form genuine CE openings | derived | `accepted_and_outputBound_implies_outputsHold_or_badEvent` |
| `nifs.pi_ccs.handoff.block_lane.completeness` | paper-valid authorized sources yield one accepted certificate and CE product | derived | `complete_of_paperObligations` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

universe uCommitment uVerifierKey uInput uState

/-- Accepted canonical replay plus explicit output authority yields the
complete CE product, unless the independent phase semantics exposes a named
FE/NC algebraic bad event. -/
theorem accepted_and_outputBound_implies_outputsHold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
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
    (publicFits : ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (freshBound_eq_two : NormStage.bound params .fresh = 2)
    (inputAuthority :
      InputAuthority.BoundToSources publicRingColumns publicFits commit data
        alignment input)
    (accepted :
      Accepted projectInput schedule priorState profile statement certificate)
    (outputBound :
      OutputBound covers data
        (derive projectInput schedule priorState profile statement certificate)
        certificate.output) :
    let pre := derivePreSumcheck schedule priorState statement
    let execution :=
      derive projectInput schedule priorState profile statement certificate
    ProductHolds publicRingColumns publicFits commit
      (OutputProduct.materialize publicRingColumns publicFits alignment input
        execution.fePoint.row certificate.output)
      (InputAuthority.productAssignments data alignment) ∨
      BadEvent profile covers data pre.challenges execution
        (certificateAtSources data certificate inputBound) challengeSetSize := by
  let pre := derivePreSumcheck schedule priorState statement
  let execution :=
    derive projectInput schedule priorState profile statement certificate
  change
    ProductHolds publicRingColumns publicFits commit
      (OutputProduct.materialize publicRingColumns publicFits alignment input
        execution.fePoint.row certificate.output)
      (InputAuthority.productAssignments data alignment) ∨
      BadEvent profile covers data pre.challenges execution
        (certificateAtSources data certificate inputBound) challengeSetSize
  have phaseSoundness :=
    accepted_implies_paperObligations_or_unbound_or_badEvent noZeroDivisors
      covers projectInput schedule priorState profile statement data inputBound
      certificate challengeSetSize accepted
  change
    SplitNc.Semantics.Paper.Holds data ∨
      ¬ OutputBound covers data execution certificate.output ∨
      BadEvent profile covers data pre.challenges execution
        (certificateAtSources data certificate inputBound) challengeSetSize
      at phaseSoundness
  rcases phaseSoundness with paper | unbound | bad
  · exact Or.inl <|
      Protocol.OutputRefinement.materializedOutputsHold_of_yRingEq
        publicRingColumns publicFits commit data alignment input
        execution.fePoint.row certificate.output freshBound_eq_two paper
        inputAuthority outputBound.1
  · exact False.elim (unbound outputBound)
  · exact Or.inr bad

/-- Every paper-valid source family at the explicit authority boundary has a
canonical message-only certificate whose replay accepts, whose raw output is
source-bound, and whose materialized CE product is genuine. -/
theorem complete_of_paperObligations
    {VerifierKey : Type uVerifierKey}
    {Input : Type uInput}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (covers : domains.nc.Covers shape)
    (projectInput : Input -> PublicInput shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (data : Data shape)
    (inputBound : projectInput statement.input = PublicInput.ofSources data)
    (publicRingColumns : Nat)
    (publicFits : ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (alignment : SourceAlignment shape params arity)
    (input :
      SourceProduct shape publicRingColumns publicFits Commitment params arity)
    (freshBound_eq_two : NormStage.bound params .fresh = 2)
    (paper : SplitNc.Semantics.Paper.Holds data)
    (inputAuthority :
      InputAuthority.BoundToSources publicRingColumns publicFits commit data
        alignment input) :
    ∃ certificate : Certificate (projectInput statement.input) domains,
      Accepted projectInput schedule priorState profile statement certificate ∧
        OutputBound covers data
          (derive projectInput schedule priorState profile statement certificate)
          certificate.output ∧
        ProductHolds publicRingColumns publicFits commit
          (OutputProduct.materialize publicRingColumns publicFits alignment input
            (derive projectInput schedule priorState profile statement
              certificate).fePoint.row
            certificate.output)
          (InputAuthority.productAssignments data alignment) := by
  rcases HonestProver.complete_of_paperObligations covers projectInput schedule
      priorState profile statement data inputBound paper with
    ⟨certificate, accepted, outputBound⟩
  refine ⟨certificate, accepted, outputBound, ?_⟩
  exact Protocol.OutputRefinement.materializedOutputsHold_of_yRingEq
    publicRingColumns publicFits commit data alignment input
    (derive projectInput schedule priorState profile statement
      certificate).fePoint.row
    certificate.output freshBound_eq_two paper inputAuthority outputBound.1

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement
