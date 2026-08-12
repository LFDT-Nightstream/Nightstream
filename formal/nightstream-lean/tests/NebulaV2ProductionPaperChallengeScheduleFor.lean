import Nightstream.Implementation.NebulaV2.ProductionPaperChallengeScheduleFor

/-! Regression surface for the exponent-indexed challenge schedule. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionPaperChallengeScheduleFor

open Nightstream.Implementation.NebulaV2.ProductionPaperChallengeScheduleFor

#check AuthoritySchedule
#check derive
#check base_open_exact
#check continuation_exact

end tests.NebulaV2ProductionPaperChallengeScheduleFor
