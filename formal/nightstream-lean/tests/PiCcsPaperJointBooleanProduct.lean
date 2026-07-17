import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanProduct

/-! Focused checks for structured Boolean product domains. -/

namespace tests.PiCcsPaperJointBooleanProduct

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

#check CubePoint.withLowPrefix
#check BooleanVertex.withLowPrefix
#check BooleanVertex.zeros
#check BooleanVertex.fieldCoordinates_withLowPrefix
#check NumericBooleanDomain.index_withLowPrefix
#check NumericBooleanDomain.index_zeros
#check BooleanTable.evaluate_tabulate_booleanPrefix

end tests.PiCcsPaperJointBooleanProduct
