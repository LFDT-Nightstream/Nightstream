import Nightstream.SuperNeo.Folding.PiRLC.PaperForkAlgebra

/-!
Focused interface regression for the artifact-independent `Pi_RLC` fork
algebra.
-/

namespace tests.PiRLCPaperForkAlgebra

open Nightstream.SuperNeo.Folding.PiRLC.PaperForkAlgebra

#check CommutativeRingOps
#check CommutativeRingLaws
#check ModuleOps
#check ModuleLaws
#check UnitWitness
#check linearCombination
#check AgreeExcept
#check coordinateIsolation
#check inverseActionCancellation

end tests.PiRLCPaperForkAlgebra
