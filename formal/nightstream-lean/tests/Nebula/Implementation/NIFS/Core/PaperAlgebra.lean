import Nightstream.Implementation.Nebula.NIFS.Core.PaperAlgebra

set_option autoImplicit false

namespace tests.NebulaProductPaperAlgebra

open Nightstream.Implementation.Nebula.ProductPaperAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

example : (fullShape 540 (by decide)).rowVariables = 25 := rfl
example : (fullShape 540 (by decide)).matrixCount = 14 := rfl
example : (fullShape 540 (by decide)).publicWidth = 540 := rfl

#check canonicalStructure_matrixSource
#check evaluationFamily_eq_paper
#check evaluations_eq_paper
#check ambientAgreement
#check openingAgreement
#check evaluations_combine
#check piRlcAlgebra
#check evaluations_recompose
#check piDecAlgebra
#check publicInputSplit
#check evaluationArity

end tests.NebulaProductPaperAlgebra
