import Ajtai.RankOneEstimator
import Nightstream.Assurance.NebulaV2.AjtaiBinding

/-!
Contract: connect the exact Nebula V2 compact-token matrix shapes to the
rank-aware Ajtai estimator results.

This proves the selected estimator arithmetic for the two rank-2 primary
roles and the two rank-1 short roles. It does not cover the three rank-18
bundle maps, whose final message widths remain generated-profile inputs.
-/

set_option autoImplicit false

namespace Ajtai.NebulaV2

open Ajtai.Parameters
open Nightstream.Assurance.NebulaV2.AjtaiBinding

theorem exact_token_shapes_match_estimators :
    primaryShape.rows = protocolBindingRank ∧
      primaryShape.columns = 738 ∧
      shortShape.rows = Ajtai.RankOneEstimator.rank ∧
      shortShape.columns = Ajtai.RankOneEstimator.ringColumns := by
  decide

/-- Both compact-token stages meet the common 131-bit raw Core-SVP threshold
inside the pinned estimator model. Computational Module-SIS hardness remains
an explicit assumption. -/
theorem token_widths_meet_selected_estimator :
    Ajtai.EstimatorModel.WidthAccepted primaryShape.columns ∧
      Ajtai.RankOneEstimator.WidthAccepted := by
  constructor
  · simpa [primaryShape] using
      Ajtai.EstimatorModel.compact_primary_width_accepted
  · exact Ajtai.RankOneEstimator.width_accepted

end Ajtai.NebulaV2
