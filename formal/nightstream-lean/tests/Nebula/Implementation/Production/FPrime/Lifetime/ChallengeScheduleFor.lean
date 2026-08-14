import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.ChallengeScheduleFor

/-! Regression surface for the exponent-indexed challenge schedule. -/

set_option autoImplicit false

namespace tests.NebulaProductionPaperChallengeScheduleFor

open Nightstream.Implementation.Nebula.ProductionPaperChallengeScheduleFor

#check AuthoritySchedule
#check derive
#check base_open_exact
#check continuation_exact

end tests.NebulaProductionPaperChallengeScheduleFor
