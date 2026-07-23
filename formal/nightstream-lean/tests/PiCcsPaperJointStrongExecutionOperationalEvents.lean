import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents

/-! Focused interface regression for the literal operational `Pi_CCS` events. -/

namespace tests.PiCcsPaperJointStrongExecutionOperationalEvents

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents

#check witnessDisagreement
#check sourceExtracted
#check fixedFirstBad
#check outputPhiMismatch
#check witnessDisagreement_eq_true_iff
#check witnessDisagreement_implies_first_success
#check outputPhiMismatch_eq_false
#check targets_eq_of_success_of_no_disagreement
#check extraction_or_fixedFirstBad

end tests.PiCcsPaperJointStrongExecutionOperationalEvents
