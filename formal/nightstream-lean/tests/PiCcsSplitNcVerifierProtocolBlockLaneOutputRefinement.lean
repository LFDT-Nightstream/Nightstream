import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement

/-!
Focused compile regressions for the canonical block×lane Π_CCS-to-CE bridge.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.handoff.block_lane.output.y_ring` | CE materialization consumes the derived FE-row binding | packed sidecar substituted for CE authority |
| `nifs.pi_ccs.handoff.block_lane.output.y_zcol` | packed-lane authority remains a distinct checked boundary | output digest promoted to authority |
| `nifs.pi_ccs.handoff.block_lane.soundness` | accepted replay yields CE membership or a named FE/NC event | unnamed or unconditional soundness |
| `nifs.pi_ccs.handoff.block_lane.completeness` | paper-valid authorized sources construct acceptance and CE membership | certificate-shaped restatement without honest construction |
-/

open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.OutputRefinement

#check ProductHolds
#check accepted_and_outputBound_implies_outputsHold_or_badEvent
#check complete_of_paperObligations
