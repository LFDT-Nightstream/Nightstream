import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81Evaluation

/-!
Focused compile-time surface check for the sole paper Phi81 evaluation leaf
and the compatibility wrappers retained by `OutputClaims`.
-/

namespace tests.PiCcsPaperJointPhi81Evaluation

open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

#check Phi81Evaluation.table
#check Phi81Evaluation.evaluate
#check yRingTableForMatrixSource
#check yRingForMatrixSource

end tests.PiCcsPaperJointPhi81Evaluation
