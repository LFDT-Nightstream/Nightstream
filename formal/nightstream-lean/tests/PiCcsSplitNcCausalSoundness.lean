import Nightstream.Protocol.FPrime.Frozen
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness

/-!
Focused interface and dimension regression for model-level production
Split-NC causal SumCheck soundness.
-/

namespace Nightstream.Tests.PiCcsSplitNcCausalSoundness

open Nightstream.Protocol.FPrime.Frozen.ProductionDeviations

#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.feRoundRepresentable
#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.fe_uniformRounds_eq_generated
#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.fe_roundCollision_implies_detects
#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.nc_roundCollision_implies_detects
#check Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane.CausalSoundness.split_detects_probability_le

#check BlockLaneCombinedNc.rawRoundRepresentable
#check BlockLaneCombinedNc.splitCollision_implies_detects
#check BlockLaneCombinedNc.splitCollision_probability_le
#check BlockLaneCombinedNc.accepted_implies_paper_or_algebraic_failure
#check BlockLaneCombinedNc.not_transcriptFailure
#check BlockLaneCombinedNc.not_bindingFailure
#check BlockLaneCombinedNc.blockLaneCombinedNc_refines_paperNc
#check BlockLaneCombinedNc.everyCoordinate_has_exact_owner
#check BlockLaneCombinedNc.delayedProjection_refines_rawRecomposition
#check BlockLaneCombinedNc.honest_complete_with_output
#check BlockLaneCombinedNc.accepted_output_suitable_for_piRlc

#check base_owns_no_predecessor
#check edge_owns_production_and_consumption
#check terminal_owns_discharge
#check terminalCount_eq_one
#check closedTrace_reduces_to_paper_transitions_or_named_failure

/-- Production block/lane NC consumes exactly 19 block challenges followed by
six lane challenges. -/
example : BlockLaneCombinedNc.ncRoundCount = 25 := by
  decide

/-- The explicit NC SumCheck numerator is 25 rounds times quartic degree. -/
example :
    BlockLaneCombinedNc.ncRoundCount *
        Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree.ncSumcheckDegreeBound =
      100 := by
  decide

end Nightstream.Tests.PiCcsSplitNcCausalSoundness
