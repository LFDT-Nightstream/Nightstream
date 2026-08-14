import Nightstream.Implementation.Nebula.NIFS.Running.Codec

set_option autoImplicit false

namespace tests.NebulaProductNifsCodec

open Nightstream.Implementation.Nebula.ProductNifsCodec

example : shape.cubeVariables = 25 := rfl
example : shape.freshCount = 1 := rfl
example : shape.runningCount = 14 := rfl
example : shape.matrixCount = 14 := rfl
example : shape.coefficientCount = 54 := rfl
example : bundleCodec.width = 3888 := rfl
example : evaluationCodec.width = 1512 := rfl
example : runningFieldCount = 83210 := rfl
example : runningBitCount = 5325440 := rfl
example : runningFieldCountFor 25 = 83210 := by decide
example : runningFieldCountFor 26 = 83212 := by decide

#check runningCodec_width
#check runningBits_length
#check runningBits_injective
#check decodeRunning_blockOfRunning
#check decodeRunning_success_reencodes
#check codecBundle_injective
#check codecBundle_protocolBundleOf
#check protocolBundleOf_codecBundle
#check publicInputOf_injective
#check freshOf_commitment
#check freshOf_publicInput
#check freshOf_pair_injective
#check runningCodecFor_width
#check runningCodecFor_admissible
#check publicInputOfFor_injective
#check freshOfFor_pair_injective

end tests.NebulaProductNifsCodec
