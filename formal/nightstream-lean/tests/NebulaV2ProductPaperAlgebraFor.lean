import Nightstream.Implementation.NebulaV2.ProductPaperAlgebraFor
import Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim

/-! Surface checks for the exponent-indexed product paper algebra. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductPaperAlgebraFor

open Nightstream.Implementation.NebulaV2.ProductPaperAlgebraFor
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

example {logicalWidth : Nat}
    {publicFits : 540 <=
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth} :
    (FullShape 26 logicalWidth publicFits).rowVariables = 26 := rfl

example {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (running :
      Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim.Running
        (FullShape 26 logicalWidth publicFits)) :
    (Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim.runningFields
      running).length = 83212 := by
  rw [Nightstream.Implementation.NebulaV2.ProductionFieldNativeFullClaim.runningFields_lengthFor
    (fullShapeContract 26 logicalWidth publicFits)]
  change 83160 + 2 * 26 = 83212
  decide

#check canonicalStructure_matrixSource
#check evaluations_eq_paper
#check ambientAgreement
#check openingAgreement
#check evaluations_combine
#check evaluations_recompose
#check piRlcAlgebra
#check piDecAlgebra

end tests.NebulaV2ProductPaperAlgebraFor
