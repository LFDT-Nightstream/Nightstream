import Ajtai.SecurityBoundary

namespace AjtaiTests.EstimatorModel

open Ajtai.EstimatorModel
open Ajtai.Parameters
open Ajtai.SecurityBoundary

example :
    ceilLogTwo setupAttackTargets = 3 ∧
      requiredRawBits = 131 ∧
      minimumAcceptedBeta = 495 ∧
      rejectedBeta = 494 :=
  selected_policy_values

example : computedMaxRingColumns = 50_371 :=
  computedMaxRingColumns_eq

example : WidthAccepted computedMaxRingColumns :=
  computedMaxRingColumns_is_largest.1

example {ringColumns : Nat}
    (above : computedMaxRingColumns < ringColumns) :
    ¬ WidthAccepted ringColumns :=
  computedMaxRingColumns_is_largest.2 ringColumns above

example : computedMaxSourceFields = 66_342 :=
  computedMaxSourceFields_eq

example :
    requiredRingColumns computedMaxSourceFields =
      computedMaxRingColumns :=
  computedBoundary_fits

example :
    requiredRingColumns (computedMaxSourceFields + 1) =
      computedMaxRingColumns + 1 :=
  computedBoundary_next_does_not_fit

example : protocolBindingParams.SideConditions :=
  protocolBindingParams_sideConditions

end AjtaiTests.EstimatorModel
