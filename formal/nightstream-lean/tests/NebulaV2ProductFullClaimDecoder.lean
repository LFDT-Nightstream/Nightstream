import Nightstream.Implementation.NebulaV2.ProductFullClaimDecoder

set_option autoImplicit false

namespace tests.NebulaV2ProductFullClaimDecoder

open Nightstream.Implementation.NebulaV2.ProductFullClaimDecoder

example : widths.ccsPublicBits = 540 := rfl
example : widths.applicationPublicBits = 7868 := rfl
example : widths.recursiveStateBits = 5325440 := rfl
example : widths.totalBits = 5587724 := widths_totalBits

#check applicationWord
#check decodeValue_block
#check decodeValue_success
#check decode_block
#check decode_success
#check claimDecoder

end tests.NebulaV2ProductFullClaimDecoder
