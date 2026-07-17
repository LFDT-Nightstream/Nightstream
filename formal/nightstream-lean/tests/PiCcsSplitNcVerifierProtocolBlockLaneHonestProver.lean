import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver

/-!
Focused regressions for canonical FE→block×lane-NC honest composition.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.prover.block_lane.input` | certificate is indexed by the explicitly bound projection | unconstrained semantic input |
| `nifs.pi_ccs.prover.block_lane.handoff` | NC starts from FE's exact successor state | disconnected phase replay |
| `nifs.pi_ccs.prover.block_lane.output` | source-derived output binds at both replay-derived points | caller-selected output points or unbound packed lanes |
| `nifs.pi_ccs.prover.block_lane.completeness` | paper obligations construct an accepted message-only certificate | soundness-only model |
-/

namespace NightstreamTests.PiCcsSplitNcVerifierProtocolBlockLaneHonestProver

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.HonestProver
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

example
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type}
    (covers : domains.nc.Covers shape)
    (data : Data shape)
    (execution : Execution shape domains State) :
    OutputBound covers data execution
      (canonicalOutput covers data execution.fePoint.row
        execution.ncPoint.block) :=
  canonicalOutput_bound covers data execution

example
    {VerifierKey Input : Type}
    {shape : SemanticShape}
    {domains : Domains}
    {State : Type}
    (covers : domains.nc.Covers shape)
    (projectInput : Input -> PublicInput shape)
    (schedule : Schedule VerifierKey Input shape domains State)
    (priorState : State)
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (statement : Statement VerifierKey Input)
    (data : Data shape)
    (inputBound : projectInput statement.input = PublicInput.ofSources data)
    (paper : Semantics.Paper.Holds data) :
    ∃ certificate : Certificate (projectInput statement.input) domains,
      Accepted projectInput schedule priorState profile statement certificate ∧
        OutputBound covers data
          (derive projectInput schedule priorState profile statement certificate)
          certificate.output :=
  complete_of_paperObligations covers projectInput schedule priorState profile
    statement data inputBound paper

end NightstreamTests.PiCcsSplitNcVerifierProtocolBlockLaneHonestProver
