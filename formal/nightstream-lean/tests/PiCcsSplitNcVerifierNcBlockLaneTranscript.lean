import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain

/-!
Focused regressions for canonical block×lane NC transcript replay.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.transcript.certificate` | physical checker sees exactly the typed product arity | malformed round count |
| `nifs.pi_ccs.nc.block_lane.transcript.phase_cut` | lane replay starts from the block successor without re-entry | hidden transcript reset or boundary field |
| `nifs.pi_ccs.nc.block_lane.transcript.point` | typed serialization equals the one flat replay result | block/lane challenge reorder |
| `nifs.pi_ccs.nc.block_lane.transcript.fixed_profile` | production carrier domain has exactly `3 + 6 = 9` rounds | accidental reuse of the legacy 15-round flat path |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane.Tests

open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270

universe uState

/-- Projection preserves the certificate's structural arity. -/
example
    {domain : BlockNcDomain}
    (certificate : Certificate domain) :
    certificate.toSumCheck.rounds.length = roundCount domain :=
  Certificate.toSumCheck_rounds_length certificate

/-- The block/lane cut is only a view of one continuous replay. -/
example
    {State : Type uState}
    {domain : BlockNcDomain}
    (machine : Nc.Machine State)
    (initialState : State)
    (certificate : Certificate domain) :
    Nc.runRoundsFrom machine (machine.enterNc initialState)
        certificate.rawRounds =
      let blockResult := Nc.runRoundsFrom machine
        (machine.enterNc initialState) certificate.blockRounds
      let laneResult := Nc.runRoundsFrom machine blockResult.2
        certificate.laneRounds
      (blockResult.1 ++ laneResult.1, laneResult.2) :=
  replay_eq_block_then_lane machine initialState certificate

/-- Point coordinates and successor state come from the same replay. -/
example
    {State : Type uState}
    {domain : BlockNcDomain}
    (machine : Nc.Machine State)
    (initialState : State)
    (certificate : Certificate domain) :
    ((derive machine initialState certificate).challengePoint.coordinates,
        (derive machine initialState certificate).finalState) =
      Nc.runRoundsFrom machine (machine.enterNc initialState)
        certificate.rawRounds :=
  derive_coordinates_finalState machine initialState certificate

/-- The repaired 270-coordinate carrier uses three block and six lane
variables, never the old nine-column-plus-six-lane transcript. -/
example : roundCount PiCcsDomain.blockDomain = 9 :=
  PiCcsDomain.blockDomain_variableCount

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.BlockLane.Tests
