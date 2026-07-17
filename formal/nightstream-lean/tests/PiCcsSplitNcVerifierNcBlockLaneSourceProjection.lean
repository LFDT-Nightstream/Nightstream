import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.SourceProjection

/-!
Focused surface checks for the source-derived block×lane NC polynomial.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.domain.decode` | block-then-lane points round-trip exactly | coordinate-order or arity drift |
| `nifs.pi_ccs.nc.block_lane.source.live` | live Boolean leaves recover authoritative coefficients | source or layout disconnect |
| `nifs.pi_ccs.nc.block_lane.source.padding` | padded blocks and lanes are computed zero | prover-controlled padding |
| `nifs.pi_ccs.nc.block_lane.range.exact` | padded Boolean cubics are equivalent to independent NC truth | changed or incomplete relation |
-/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

#check Point.decode_coordinates
#check SourceProjection.paddedValue_live
#check SourceProjection.paddedValue_block_padding
#check SourceProjection.paddedValue_lane_padding
#check SourceProjection.sourceValueAt_live
#check SourceProjection.rangeValueAt_live
#check SourceProjection.booleanResidualsZero_of_truth
#check SourceProjection.truth_of_booleanResidualsZero
#check SourceProjection.booleanResidualsZero_iff_truth
