import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCSAuthority

/-! Regression surface for authoritative phased production PiCCS. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiCcsAuthority

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority

#check RoundPhaseRelation
#check runRounds_cursor
#check runRounds_exact
#check check_eq_fixedPhase_check
#check productionCheck_eq_piCcsCheck
#check production_round_count_exact
#check roundFrame_length
#check accepted_different_round_implies_collision

end tests.NebulaProductionStreamingPiCcsAuthority
