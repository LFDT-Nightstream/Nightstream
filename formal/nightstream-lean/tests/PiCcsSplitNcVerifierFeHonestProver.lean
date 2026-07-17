import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver

/-!
Focused theorem regressions for sequential honest mixed-width FE replay.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.fe.prover.row` | row rounds retain the syntax-derived width | uniform-width transcript drift |
| `nifs.pi_ccs.fe.prover.phase_cut` | lane construction begins after row replay | transcript reset or future-challenge assumption |
| `nifs.pi_ccs.fe.prover.honesty` | honesty uses the replay-derived FE point | post-challenge-only completeness |
-/

namespace NightstreamTests.PiCcsSplitNcVerifierFeHonestProver

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Fe.HonestProver
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (coins : Coins shape domain) :
    ∃ certificate :
        SumCheck.Fe.Certificate (PublicInput.ofSources data) domain,
      SumCheck.Fe.HonestAt
        (InitialSum.sumcheckPolynomial profile data coins)
        (Transcript.Fe.derive machine initialState certificate).challengePoint
        certificate :=
  exists_honest_certificate profile data machine initialState coins

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (machine : Transcript.Fe.Machine State)
    (initialState : State)
    (coins : Coins shape domain)
    (truth : Semantics.Fe.Truth data) :
    ∃ certificate :
        SumCheck.Fe.Certificate (PublicInput.ofSources data) domain,
      Transcript.Fe.Accepted machine initialState
        (initial profile (PublicInput.ofSources data) coins)
        (InitialSum.sumcheckPolynomial profile data coins
          (Transcript.Fe.derive machine initialState certificate).challengePoint.coordinates)
        certificate :=
  complete_of_truth profile data machine initialState coins truth

end NightstreamTests.PiCcsSplitNcVerifierFeHonestProver
