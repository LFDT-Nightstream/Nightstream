import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.FeRefinement

/-!
Focused model-level regressions for exact mixed-width FE transcript
refinement.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.transcript.transport` | semantic and implementation extensions round-trip | modulus or limb-order mismatch |
| `nifs.pi_ccs.fe.transcript.row.coefficients` | row width is exactly the syntax-derived `Drow + 1` | uniform or truncated row serialization |
| `nifs.pi_ccs.fe.transcript.lane.coefficients` | lane width is exactly three `K` / six base fields | proof-view widening entering the transcript |
| `nifs.pi_ccs.fe.transcript.phase_cut` | concrete lane replay starts at the row successor | hidden marker, reset, or second prologue |
| `nifs.pi_ccs.fe.transcript.phase` | typed semantic derive equals concrete `runFe` | prologue, ordering, challenge, point, or final-state drift |
-/

namespace NightstreamTests.PiCcsTranscriptFeRefinement

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiCcsTranscript
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
open Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Fe

example (value : K) : toK (toExtension value) = value := by
  simp

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain)
    (message : Nightstream.SuperNeo.SumCheck.Finite.Message K)
    (member : message ∈ certificate.rowRawRounds) :
    (toConcreteRound message).coefficients.length =
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Drow
        input + 1 :=
  concreteRowRound_coefficients_length certificate message member

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain)
    (message : Nightstream.SuperNeo.SumCheck.Finite.Message K)
    (member : message ∈ certificate.laneRawRounds) :
    (toConcreteRound message).coefficients.length = 3 ∧
      (SumCheck.roundFields (toConcreteRound message)).length = 6 := by
  exact ⟨concreteLaneRound_coefficients_length certificate message member,
    concreteLaneRound_fields_length certificate message member⟩

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (state : State)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain) :
    SumCheck.runRounds state (concreteRounds certificate) =
      let rowResult := SumCheck.runRounds state
        (concreteRowRounds certificate)
      let laneResult := SumCheck.runRounds rowResult.1
        (concreteLaneRounds certificate)
      (laneResult.1, rowResult.2 ++ laneResult.2) :=
  concreteReplay_eq_row_then_lane state certificate

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (state : State)
    (claimed : K)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain) :
    runRoundsFrom (machine claimed)
        (SumCheck.fePrologue state (toExtension claimed))
        certificate.rawRounds =
      let concrete := SumCheck.runRounds
        (SumCheck.fePrologue state (toExtension claimed))
        (concreteRounds certificate)
      (concrete.2.map toK, concrete.1) :=
  replay_refines_runRounds state claimed certificate

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (state : State)
    (claimed : K)
    (certificate :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Certificate
        input domain)
    (messages : SumCheck.Messages)
    (initialClaim : messages.feInitial = toExtension claimed)
    (feRounds : messages.feRounds = concreteRounds certificate) :
    ((derive (machine claimed) state certificate).challengePoint.coordinates,
        (derive (machine claimed) state certificate).finalState) =
      let concrete := SumCheck.runFe state messages
      (concrete.2.map toK, concrete.1) :=
  derive_refines_runFe state claimed certificate messages initialClaim feRounds

end NightstreamTests.PiCcsTranscriptFeRefinement
