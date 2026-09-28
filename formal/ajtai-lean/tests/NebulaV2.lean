import Ajtai.NebulaV2

namespace AjtaiTests.NebulaV2

example :
    Ajtai.EstimatorModel.WidthAccepted
        Nightstream.Assurance.NebulaV2.AjtaiBinding.primaryShape.columns ∧
      Ajtai.RankOneEstimator.WidthAccepted :=
  Ajtai.NebulaV2.token_widths_meet_selected_estimator

end AjtaiTests.NebulaV2
