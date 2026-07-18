import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FinalState

/-!
Focused regressions for the final terminal-NC state refinement.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.14.challenge.outputs` | outputs are columns `1692820..1692827` | tail-column drift |
| `nifs.pi_ccs.nc.post_state.retained_lanes` | accepted final call plus input refinement binds seven lanes | unproved carried post-state |
| `nifs.pi_ccs.nc.post_state` | cursor and retained lanes compose separately | blurred authority boundary |
-/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc

#check FinalState.finalSqueeze_outputColumn
#check FinalState.FinalPermutationBound
#check FinalState.FinalPermutationBound.cursorZero
#check FinalState.retainedLanes_of_accepted
#check FinalState.boundary_of_accepted
