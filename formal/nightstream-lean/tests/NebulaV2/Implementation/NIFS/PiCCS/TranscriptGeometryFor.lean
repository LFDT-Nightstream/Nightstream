import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TranscriptGeometryFor

set_option autoImplicit false

namespace tests.NebulaV2ProductPiCcsTranscriptGeometryFor

open Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptGeometryFor

#check rows_length_exact
#check rowCount_25
#check rowCount_26
#check exponent_26_adds_15177_rows

example : rowCount 26 = 12127977 := rowCount_26

end tests.NebulaV2ProductPiCcsTranscriptGeometryFor
