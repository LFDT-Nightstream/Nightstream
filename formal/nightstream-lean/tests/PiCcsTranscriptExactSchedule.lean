import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact

/-!
Focused kernel regressions for the canonical exact FE-to-NC SumCheck
sub-schedule.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.exact.schedule.input` | schedule consumes the typed exact carrier directly | loose `Messages`/`WellShaped` authority |
| `nifs.pi_ccs.exact.schedule.encoding` | raw projection satisfies `ExactLanguage` and decodes losslessly | padding, trimming, or phase reshaping |
| `nifs.pi_ccs.exact.schedule.nc` | NC begins from FE's exact returned state | independently supplied phase-boundary state |
| `nifs.pi_ccs.exact.schedule.counts` | FE and NC challenges equal their typed round counts | missing or synthesized challenges |
| `nifs.pi_ccs.exact.schedule.nc.cursor` | a positive NC phase computes cursor zero | treating control flow as witness authority |
| `nifs.pi_ccs.exact.schedule.replay` | one exact input determines one joint trace | nondeterministic phase replay |
| `nifs.pi_ccs.exact.refinement.fe.encoding` | raw FE rounds equal the typed FE checker adapter | serialization drift |
| `nifs.pi_ccs.exact.refinement.nc.encoding` | raw NC rounds equal the typed NC checker adapter | serialization drift |
| `nifs.pi_ccs.exact.refinement.fe.derive` | typed FE derive equals the FE schedule projection | challenge or successor drift |
| `nifs.pi_ccs.exact.refinement.nc.derive` | typed NC derive equals the NC schedule projection from FE's successor | phase-boundary, challenge, or successor drift |
-/

namespace NightstreamTests.PiCcsTranscriptExactSchedule

open Nightstream.Implementation.R1CS.PiCcsTranscript.Exact
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain) :
    ExactLanguage publicInput domain input.expectedFeInitial
      (Schedule.rawMessages input) :=
  Schedule.rawMessages_exactLanguage input

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain) :
    decode publicInput domain input.expectedFeInitial
      (Schedule.rawMessages input) = some input.carrier := by
  simp

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain) :
    (Schedule.run input).afterNc =
      (Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc
        (Schedule.run input).afterFe (Schedule.rawMessages input)).1 := by
  simp

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain) :
    (Schedule.run input).feChallenges.length =
        shape.rowVariables + domain.laneVariables /\
      (Schedule.run input).ncChallenges.length =
        Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.roundCount
          domain :=
  Schedule.challengeCounts input

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain)
    (positive :
      0 <
        Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.roundCount
          domain) :
    (Schedule.run input).afterNc.absorbed.val = 0 :=
  Schedule.run_afterNc_absorbed_zero input positive

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain)
    (left right : Schedule.Trace)
    (leftEq : left = Schedule.run input)
    (rightEq : right = Schedule.run input) :
    left = right :=
  Schedule.replay_deterministic input left right leftEq rightEq

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain) :
    (Schedule.rawMessages input).feRounds =
      Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.concreteRounds
        input.carrier.toFeCertificate :=
  Refinement.encode_feRounds_eq_concreteRounds
    input.expectedFeInitial input.carrier

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain) :
    (Schedule.rawMessages input).ncRounds =
      Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.concreteRounds
        input.carrier.toNcCertificate :=
  Refinement.encode_ncRounds_eq_concreteRounds
    input.expectedFeInitial input.carrier

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain) :
    ((Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.derive
          (Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.machine
            input.expectedFeInitial)
          input.initialState input.carrier.toFeCertificate).challengePoint.coordinates,
        (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe.derive
          (Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.machine
            input.expectedFeInitial)
          input.initialState input.carrier.toFeCertificate).finalState) =
      ((Schedule.run input).feChallenges.map
          Nightstream.Implementation.R1CS.PiCcsTranscript.Transport.toK,
        (Schedule.run input).afterFe) :=
  Refinement.feDerive_refines_run input

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Schedule.Input publicInput domain) :
    ((Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.derive
          Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.machine
          (Schedule.run input).afterFe
          input.carrier.toNcCertificate).challengePoint.coordinates,
        (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.derive
          Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.machine
          (Schedule.run input).afterFe
          input.carrier.toNcCertificate).finalState) =
      ((Schedule.run input).ncChallenges.map
          Nightstream.Implementation.R1CS.PiCcsTranscript.Transport.toK,
        (Schedule.run input).afterNc) :=
  Refinement.ncDerive_refines_run input

end NightstreamTests.PiCcsTranscriptExactSchedule
