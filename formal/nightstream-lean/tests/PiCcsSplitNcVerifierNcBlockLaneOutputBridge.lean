import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.OutputBridge

/-!
Focused surface check for the packed-output/source-polynomial anti-drift
bridge.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.output_bridge.lane` | packed output and NC source MLE use the same block layout and weights | bit-order, padding, or carrier drift |
-/

#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.OutputBridge.packedYZcol_lane_eq_blockValueAt
