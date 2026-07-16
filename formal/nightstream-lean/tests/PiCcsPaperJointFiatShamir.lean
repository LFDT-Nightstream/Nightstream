import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir

/-! Public theorem surface for verifier-owned paper joint-`Pi_CCS` coins. -/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir

#check Certificate.toFinite_rounds_length
#check derivePreSumcheck
#check derive
#check checkResidualTableAudit_eq_true_iff_accepted
#check checkResidualTableAudit_complete_of_accepted
#check checkResidualTableAudit_implies_semanticTruth_or_badEvent
