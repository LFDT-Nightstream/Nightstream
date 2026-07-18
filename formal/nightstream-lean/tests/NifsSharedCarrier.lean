import Nightstream.SuperNeo.Folding.Nifs.SharedCarrier

/-! Public theorem surface for the independent NIFS shared-carrier semantics. -/

open Nightstream.SuperNeo.Folding.Nifs

#check SharedAttempt.wiring
#check SharedAccepted.toAccepted
#check normalize_toAttempt_eq
#check accepted_normalize
#check sharedPaperNifsTransition_iff_paperNifsTransition
