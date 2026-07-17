import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.PostNcBoundary

/-!
Focused compile-time regression for the minimal post-NC catch-up boundary.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.post_state` | only cursor zero and lanes one through seven cross the boundary | unnecessary whole-state authority |
| `nifs.pi_ccs.nc.post_state.cursor` | exact positive NC replay constructs the cursor child | treating control flow as artifact authority |
| `nifs.pi_ccs.nc.post_state.retained_lanes` | seven surviving lanes remain a separate child | obscured authority ownership |
| `nifs.pi_ccs.catchup.lane0` | the marker overwrites the old lane-zero value | redundant lane-zero binding |
| `nifs.pi_ccs.catchup.marker` | the reduced boundary reconstructs the exact artifact call input | disconnected catch-up state |
-/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal

#check PostNcBoundary.CursorZero
#check PostNcBoundary.RetainedLanesBound
#check PostNcBoundary.Bound
#check PostNcBoundary.Bound.ofExactSchedule
#check PostNcBoundary.laneZero_irrelevant
#check PostNcBoundary.refines_catchupInput
