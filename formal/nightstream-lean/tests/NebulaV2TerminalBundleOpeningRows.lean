import Nightstream.Implementation.NebulaV2.TerminalBundleOpeningRows

/-! Focused gate for the exact common-witness terminal opening rows. -/

set_option autoImplicit false

namespace tests.NebulaV2TerminalBundleOpeningRows

open Nightstream.Implementation.NebulaV2.TerminalBundleOpeningRows

#check sound
#check Layout.numericColumn_injective
#check Layout.publicBundle_eq_codecBundle
#check sound_opens_codec_bundle
#check rows_length

end tests.NebulaV2TerminalBundleOpeningRows
