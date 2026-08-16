import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputResidualRows

/-! Regression surface for the production PiRLC residual-link rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcInputResidualRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidualRows

#check residualRow
#check rows_length
#check ColumnsPlaced
#check residualField_exact_of_row
#check rows_imply_addResidualFields
#check rows_imply_concreteResidualTransition
#check rows_complete

end tests.NebulaProductionStreamingPiRlcInputResidualRows
