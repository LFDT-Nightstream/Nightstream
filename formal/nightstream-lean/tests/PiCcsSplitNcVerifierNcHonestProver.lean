import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.HonestProver

/-!
Focused theorem regressions for sequential honest NC transcript construction.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.prover.replay` | generic and concrete replays are identical | alternate transcript semantics |
| `nifs.pi_ccs.nc.prover.honesty` | honesty is stated at replay-derived challenges | future-challenge fixed-point assumption |
| `nifs.pi_ccs.nc.prover.completeness` | semantic truth yields transcript-bound acceptance | post-challenge-only completeness |
-/

namespace NightstreamTests.PiCcsSplitNcVerifierNcHonestProver

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.HonestProver
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Polynomial.Nc.Mixing.Coins domain) :
    ∃ certificate : Transcript.Nc.Certificate domain,
      Nightstream.SuperNeo.SumCheck.Finite.FixedPhase.Honest
        ConcreteCarrier.extensionOps.toOps
        (Polynomial.Nc.InitialSum.sumcheckPolynomial
          convention covers data coins)
        (Transcript.Nc.derive machine initialState certificate).challengePoint.coordinates
        certificate.toSumCheck :=
  exists_honest_certificate convention covers data machine initialState coins

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Polynomial.Nc.Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data) :
    ∃ certificate : Transcript.Nc.Certificate domain,
      Transcript.Nc.Accepted machine initialState
        Polynomial.Nc.InitialSum.claimedInitial
        (Polynomial.Nc.InitialSum.sumcheckPolynomial
          convention covers data coins
          (Transcript.Nc.derive machine initialState certificate).challengePoint.coordinates)
        certificate :=
  complete_of_truth convention covers data machine initialState coins truth

end NightstreamTests.PiCcsSplitNcVerifierNcHonestProver
