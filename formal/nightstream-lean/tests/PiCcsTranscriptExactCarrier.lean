import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact

/-!
Focused regressions for the exact typed `Pi_CCS` SumCheck carrier.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.exact.fe.initial` | raw FE initial equals verifier-owned `toExtension` image | prover-selected initial claim |
| `nifs.pi_ccs.exact.fe.row` | exact row count and `Drow+1` widths | global-bound padding or malformed row phase |
| `nifs.pi_ccs.exact.fe.lane` | exact lane count and three-slot widths | widened lane serialization |
| `nifs.pi_ccs.exact.fe.total` | complete FE count follows from the two exact phases | duplicated or drifting total-count check |
| `nifs.pi_ccs.exact.nc` | exact NC count and five-slot widths | short, long, or loose NC messages |
| `nifs.pi_ccs.exact.codec` | decoding/encoding are lossless inverses | reordering, trimming, or synthesized rounds |
-/

namespace NightstreamTests.PiCcsTranscriptExactCarrier

open Nightstream.Implementation.R1CS.PiCcsTranscript
open Nightstream.Implementation.R1CS.PiCcsTranscript.Exact
open Nightstream.Implementation.R1CS.PiCcsTranscript.Transport
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expected : K)
    (carrier : Carrier input domain) :
    decode input domain expected (encode expected carrier) = some carrier := by
  simp

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expected : K)
    (carrier : Carrier input domain) :
    ExactLanguage input domain expected (encode expected carrier) :=
  exactLanguage_encode expected carrier

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expected : K)
    (carrier : Carrier input domain) :
    (encode expected carrier).feRounds.length =
      shape.rowVariables + domain.laneVariables := by
  simp

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    {expected : K}
    {messages : SumCheck.Messages}
    (language : ExactLanguage input domain expected messages) :
    messages.feRounds.length =
      shape.rowVariables + domain.laneVariables :=
  feRounds_length_of_exactLanguage language

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expected : K)
    (messages : SumCheck.Messages)
    (wrong : messages.feInitial ≠ toExtension expected) :
    decode input domain expected messages = none :=
  decode_none_of_feInitial_ne input domain expected messages wrong

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expected : K)
    (messages : SumCheck.Messages)
    (wrong :
      messages.feRounds.length ≠
        shape.rowVariables + domain.laneVariables) :
    decode input domain expected messages = none :=
  decode_none_of_feRoundCount_ne input domain expected messages wrong

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expected : K)
    (messages : SumCheck.Messages)
    (wrong :
      messages.ncRounds.length ≠
        Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.roundCount
          domain) :
    decode input domain expected messages = none :=
  decode_none_of_ncRoundCount_ne input domain expected messages wrong

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expected : K)
    (messages : SumCheck.Messages)
    (round : SumCheck.RoundMessage)
    (member : round ∈ messages.feRounds.take shape.rowVariables)
    (wrong :
      round.coefficients.length ≠
        Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Fe.Drow
          input + 1) :
    decode input domain expected messages = none :=
  decode_none_of_feRowWidth_ne input domain expected messages round member wrong

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expected : K)
    (messages : SumCheck.Messages)
    (round : SumCheck.RoundMessage)
    (member : round ∈ messages.feRounds.drop shape.rowVariables)
    (wrong : round.coefficients.length ≠ 3) :
    decode input domain expected messages = none :=
  decode_none_of_feLaneWidth_ne input domain expected messages round member wrong

example
    {shape : SemanticShape}
    {input : PublicInput shape}
    {domain : FlatNcDomain}
    (expected : K)
    (messages : SumCheck.Messages)
    (round : SumCheck.RoundMessage)
    (member : round ∈ messages.ncRounds)
    (wrong : round.coefficients.length ≠ 5) :
    decode input domain expected messages = none :=
  decode_none_of_ncWidth_ne input domain expected messages round member wrong

end NightstreamTests.PiCcsTranscriptExactCarrier
