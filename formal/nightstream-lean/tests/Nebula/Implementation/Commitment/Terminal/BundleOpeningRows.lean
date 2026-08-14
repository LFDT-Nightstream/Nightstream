import Nightstream.Implementation.Nebula.Commitment.Terminal.BundleOpeningRows

/-! Focused gate for the exact common-witness terminal opening rows. -/

set_option autoImplicit false

namespace tests.NebulaTerminalBundleOpeningRows

open Nightstream.Implementation.Nebula.TerminalBundleOpeningRows

#check sound
#check Layout.numericColumn_injective
#check Layout.publicBundle_eq_codecBundle
#check sound_opens_codec_bundle
#check rows_length

end tests.NebulaTerminalBundleOpeningRows
