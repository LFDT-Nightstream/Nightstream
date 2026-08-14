import Nightstream.Implementation.Nebula.NIFS.PiCCS.TranscriptGeometryFor

set_option autoImplicit false

namespace tests.NebulaProductPiCcsTranscriptGeometryFor

open Nightstream.Implementation.Nebula.ProductPiCcsTranscriptGeometryFor

#check rows_length_exact
#check rowCount_25
#check rowCount_26
#check exponent_26_adds_15177_rows

example : rowCount 26 = 12139373 := rowCount_26

end tests.NebulaProductPiCcsTranscriptGeometryFor
