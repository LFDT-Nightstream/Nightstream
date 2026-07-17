import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane

/-!
Focused regressions for canonical FE-to-block×lane-NC protocol composition.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.verify.block_lane.input_projection` | the complete bound statement is distinct from its FE/NC projection | omission of source product or running parent from transcript binding |
| `nifs.pi_ccs.verify.block_lane.handoff` | NC consumes FE's exact outgoing state | reset, disconnected state, or alternate NC point |
| `nifs.pi_ccs.verify.block_lane.output` | output absorption occurs after NC | early, missing, or alternate output handoff |
| `nifs.pi_ccs.verify.block_lane.check` | executable and logical acceptance coincide | divergent verifier paths |
| `nifs.pi_ccs.verify.block_lane.soundness` | composed theorem retains output and phase-event boundaries | promotion of raw output or silent event loss |
-/

namespace NightstreamTests.PiCcsSplitNcVerifierProtocolBlockLane

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

universe uVerifierKey uState

/-- Test-only carrier proving that the transcript-bound input may contain data
that FE and NC do not consume. -/
structure EnrichedInput (shape : SemanticShape) where
  polynomial : PublicInput shape
  transcriptOnly : Nat

#check Certificate
#check Execution
#check derive_ncPoint
#check derive_finalState
#check check_eq_true_iff_accepted
#check OutputBound
#check BadEvent
#check certificateAtSources
#check accepted_implies_paperObligations_or_unbound_or_badEvent

example
    {VerifierKey : Type uVerifierKey}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type uState}
    (schedule : Schedule VerifierKey (EnrichedInput shape) shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey (EnrichedInput shape))
    (certificate : Certificate statement.input.polynomial domains) :
    check EnrichedInput.polynomial schedule priorState profile statement
          certificate = true ↔
      Accepted EnrichedInput.polynomial schedule priorState profile statement
        certificate := by
  exact check_eq_true_iff_accepted EnrichedInput.polynomial schedule priorState
    profile statement certificate

end NightstreamTests.PiCcsSplitNcVerifierProtocolBlockLane
