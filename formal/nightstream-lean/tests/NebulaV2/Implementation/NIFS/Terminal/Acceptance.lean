import Nightstream.Implementation.NebulaV2.NIFS.Terminal.Acceptance

/-! Focused gate for exact row-derived V2 terminal acceptance. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductTerminalAcceptance

open Nightstream.Implementation.NebulaV2

#check ProductTerminalAcceptance.acceptedOfRows
#check ProductTerminalAcceptance.consumes_exact_selected_trailing_claim
#check ProductTerminalAcceptance.common_product_witnesses

end tests.NebulaV2ProductTerminalAcceptance
