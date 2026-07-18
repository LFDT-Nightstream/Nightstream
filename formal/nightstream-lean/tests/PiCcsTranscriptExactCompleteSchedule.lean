import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.CompleteSchedule

/-!
Focused API regressions for the complete exact `Pi_CCS` transcript schedule.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.exact.complete.shape` | exact typed carrier implies the legacy schedule shape | caller-supplied loose shape evidence |
| `nifs.pi_ccs.exact.complete.coins` | schedule challenges equal the canonical coin execution | duplicated challenge authority |
| `nifs.pi_ccs.exact.complete.sumcheck` | complete FE/NC replay equals the exact sub-schedule | disconnected phase states |
| `nifs.pi_ccs.exact.complete.catchup` | catch-up consumes the exact NC successor | alternate terminal digest path |
-/

namespace NightstreamTests.PiCcsTranscriptExactCompleteSchedule

open Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.CompleteSchedule
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.WellShaped
      (scheduleInput input) :=
  scheduleInput_wellShaped input

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    (run input).challenges = challengeOutput input :=
  run_challenges input

example
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : Input publicInput domain) :
    (run input).afterNc =
      (Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule.run
        (exactInput input)).afterNc :=
  run_afterNc_eq_exact input

end NightstreamTests.PiCcsTranscriptExactCompleteSchedule
