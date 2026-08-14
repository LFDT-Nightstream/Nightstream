import Nightstream.Implementation.Nebula.NIFS.Core.Poseidon2

set_option autoImplicit false

namespace tests.NebulaProductPoseidon2

open Nightstream.Implementation.Nebula.ProductPoseidon2

example : eventSchedule.length = 55 := eventSchedule_length
example : eventScheduleFields.length = 313 := eventScheduleFields_length
example : statementIdentifierPrefixFields.length = 363 :=
  statementIdentifierPrefixFields_length
example : publicInputTag = 1314082354 := rfl
example : protocolVersion = 2 := rfl
example : profileName = 2 := rfl
example : checkedStepFactor = 1 := rfl
example : commitmentEncodingTag = 1 := rfl

#check bundleFields_length
#check runningFields_length
#check freshFields_length
#check publicNifsFields_length
#check outputFields_length
#check piDecOutputFields_length
#check piRlcResponse_valid

end tests.NebulaProductPoseidon2
