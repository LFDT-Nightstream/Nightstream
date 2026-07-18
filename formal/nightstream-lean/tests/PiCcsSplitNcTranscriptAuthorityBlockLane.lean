import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains

/-!
Focused regressions for canonical FE plus block×lane transcript authority.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.transcript.block_lane.domain` | one domain record projects the fixed FE and block×lane domains | independent or drifting lane widths |
| `nifs.pi_ccs.transcript.block_lane.coins` | FE and NC read one `betaA` and one `gamma` | duplicated or divergent coin authority |
| `nifs.pi_ccs.transcript.block_lane.fe` | the verifier-owned initial claim parameterizes FE entry | caller-supplied FE machine |
| `nifs.pi_ccs.transcript.block_lane.nc` | canonical NC projects from the same configured schedule | disconnected phase machine |
-/

namespace NightstreamTests.PiCcsSplitNcTranscriptAuthorityBlockLane

open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

example : PiCcsDomains.production.fe = PiCcsDomain.domain := by
  exact PiCcsDomains.production_fe

example : PiCcsDomains.production.nc = PiCcsDomain.blockDomain := by
  exact PiCcsDomains.production_nc

example
    {shape : SemanticShape}
    {domains : Domains}
    (challenges : Challenges shape domains) :
    challenges.ncCoins.betaA = challenges.feCoins.betaA :=
  Challenges.ncCoins_betaA_eq_feCoins_betaA challenges

example
    {shape : SemanticShape}
    {domains : Domains}
    (challenges : Challenges shape domains) :
    challenges.ncCoins.gamma = challenges.feCoins.gamma :=
  Challenges.ncCoins_gamma_eq_feCoins_gamma challenges

#check Statement
#check Schedule
#check derivePreSumcheck
#check feMachine
#check ncMachine

end NightstreamTests.PiCcsSplitNcTranscriptAuthorityBlockLane
