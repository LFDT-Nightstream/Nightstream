import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.HonestProver

/-!
Focused theorem regression for the complete exact `Pi_CCS` honest prover.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.exact.prover.coins` | FE/NC coins follow the binding successor | caller-selected challenges |
| `nifs.pi_ccs.exact.prover.messages` | FE then NC messages precede their challenges | future-challenge fixed-point witness |
| `nifs.pi_ccs.exact.prover.carrier` | typed certificates enter the exact physical carrier losslessly | loose or reordered message carrier |
| `nifs.pi_ccs.exact.prover.output` | accepted output remains bound to semantic sources | caller-selected output claims |
-/

namespace NightstreamTests.PiCcsTranscriptExactHonestProver

open Nightstream.Implementation.R1CS.PiCcsTranscript.Exact
open Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.HonestProver
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (initialState : State)
    (binding :
      Nightstream.Implementation.R1CS.PiCcsTranscript.Binding.Input)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (obligations : Semantics.Paper.Holds data) :
    ∃ certificate :
        Protocol.Certificate (PublicInput.ofSources data) domain,
      let input :=
        completeInput initialState binding profile data certificate
      Protocol.Accepted
          (Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.machine
            input.expectedFeInitial)
          Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.machine
          (CompleteSchedule.challengeOutput input).state
          profile
          (PublicInput.ofSources data)
          (CompleteSchedule.feCoins input)
          (CompleteSchedule.ncCoins input)
          certificate /\
        BoundToSources covers data
          (Protocol.derive
            (Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.machine
              input.expectedFeInitial)
            Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.machine
            (CompleteSchedule.challengeOutput input).state
            certificate).outputPoints
          certificate.output /\
        Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.WellShaped
          (CompleteSchedule.scheduleInput input) :=
  complete_of_paperObligations covers initialState binding profile data
    obligations

end NightstreamTests.PiCcsTranscriptExactHonestProver
