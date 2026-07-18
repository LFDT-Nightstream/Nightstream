import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage

/-!
Focused impossibility and coverage regressions for Split-NC domain ownership.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.paper.square_domain` | direct paper `ColumnLayout` cannot cover the Phi81 carrier | dishonest PaperJoint instantiation |
| `nifs.pi_ccs.nc.domain.columns` | logical-width cube omits completed coordinates | hidden running-carrier tail |
| `nifs.pi_ccs.nc.domain.lanes` | six lane bits cover all 54 real lanes | conflating lane and column failures |
-/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity.CarrierCoverage

#check logicalWidthCube_covers_lanes
#check no_paperColumnLayout_for_carrier
#check logicalWidth_lt_carrierWidth
#check firstCompletedTail_outside_columnCube
#check logicalWidthCube_does_not_cover
