import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.BlockLane

/-!
Compile-time surface checks for the independent block×lane NC semantics.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.layout` | canonical block/lane flattening is total and invertible | omitted or duplicated carrier coordinate |
| `nifs.pi_ccs.nc.block_lane.residual.exact` | block/lane cubics are equivalent to full-carrier norm truth | representation silently changes the relation |
-/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

#check Semantics.Nc.BlockLane.carrierColumn_decode
#check Semantics.Nc.BlockLane.blockCount_mul_ringDegree_eq_carrierWidth
#check Semantics.Nc.BlockLane.value_decode
#check Semantics.Nc.BlockLane.residualsZero_of_truth
#check Semantics.Nc.BlockLane.truth_of_residualsZero
#check Semantics.Nc.BlockLane.residualsZero_iff_truth
