import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.HonestProver

/-!
Focused regressions for message-before-challenge block×lane NC construction.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.prover.replay` | certificate honesty uses its own block-then-lane replay point | caller-supplied or future challenge |
| `nifs.pi_ccs.nc.block_lane.prover.completeness` | semantic NC truth yields exact transcript acceptance | symbolic-only prover disconnected from acceptance |
-/

namespace NightstreamTests.PiCcsSplitNcVerifierNcBlockLaneHonestProver

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.HonestProver
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain) :
    ∃ certificate : Transcript.Nc.BlockLane.Certificate domain,
      Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Honest
        Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
        (InitialSum.sumcheckPolynomial covers data coins)
        (Transcript.Nc.BlockLane.derive machine initialState
          certificate).challengePoint.coordinates
        certificate.toSumCheck :=
  exists_honest_certificate covers data machine initialState coins

example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data) :
    ∃ certificate : Transcript.Nc.BlockLane.Certificate domain,
      Transcript.Nc.BlockLane.Accepted machine initialState
        InitialSum.claimedInitial
        (InitialSum.sumcheckPolynomial covers data coins
          (Transcript.Nc.BlockLane.derive machine initialState
            certificate).challengePoint.coordinates)
        certificate :=
  complete_of_truth covers data machine initialState coins truth

end NightstreamTests.PiCcsSplitNcVerifierNcBlockLaneHonestProver
