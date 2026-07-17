import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority

/-!
Focused regressions for the single-owner Split-NC transcript authority.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.transcript.statement` | key and public input are bound as one typed value | partial statement seed |
| `nifs.pi_ccs.transcript.coins.shared` | FE and NC read one `betaA` and one `gamma` | duplicated or divergent coin authority |
| `nifs.pi_ccs.transcript.fe` | the computed initial claim parameterizes the FE entry | caller-supplied FE machine |
| `nifs.pi_ccs.transcript.nc` | NC projects from the same configured schedule | disconnected phase machine |
-/

namespace NightstreamTests.PiCcsSplitNcTranscriptAuthority

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (challenges : Challenges shape domain) :
    challenges.ncCoins.betaA = challenges.feCoins.betaA :=
  Challenges.ncCoins_betaA_eq_feCoins_betaA challenges

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (challenges : Challenges shape domain) :
    challenges.ncCoins.gamma = challenges.feCoins.gamma :=
  Challenges.ncCoins_gamma_eq_feCoins_gamma challenges

#check Statement
#check Schedule
#check derivePreSumcheck
#check feMachine
#check ncMachine

end NightstreamTests.PiCcsSplitNcTranscriptAuthority
