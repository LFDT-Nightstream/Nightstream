import Nightstream.Implementation.Nebula.Production.Carrier.CoordinateLocalRunning

/-! Regression surface for the coordinate-local running-claim view. -/

set_option autoImplicit false

namespace tests.NebulaProductionCoordinateLocalRunning

open Nightstream.Implementation.Nebula.ProductionCoordinateLocalRunning

#check toRunning_ofRunning
#check ofRunning_injective
#check coordinateLocalCodec_admissible
#check coordinateLocalCodec_width
#check encodeRunning_length
#check decodeRunning_encodeRunning
#check encodeRunning_injective
#check totalFieldCount_eq_runningFieldCountFor
#check totalFieldCount_r26
#check fullSourceRingWindowFieldCount_eq

end tests.NebulaProductionCoordinateLocalRunning
