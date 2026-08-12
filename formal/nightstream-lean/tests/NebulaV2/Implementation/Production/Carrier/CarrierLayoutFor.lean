import Nightstream.Implementation.NebulaV2.Production.Carrier.CarrierLayoutFor

/-! Regression surface for the exponent-indexed physical full-claim layout. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionFullClaimCarrierLayoutFor

open Nightstream.Implementation.NebulaV2.ProductionFullClaimCarrierLayoutFor

#check section_offsets_exact
#check endOffset_exact
#check counter_intervals_exact
#check checkedMemoryPlacement
#check nifsRunningValues_eq_carrier
#check nifsBundleValues_eq_carrier
#check memoryNativeColumn_lt_end

end tests.NebulaV2ProductionFullClaimCarrierLayoutFor
