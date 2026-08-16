import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedResidualRows

/-! Focused checks for normalized PiRLC residual-row semantics. -/

namespace tests.Nebula.Implementation.Production.Carrier.StreamingPiRLCNormalizedResidualRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedResidualRows.Normalized

example : productionRowCount = 108 := productionRowCount_exact

#check residualImage
#check ProductionAccepted
#check productionAccepted_implies_source_rows
#check StateColumnsPlaced
#check PhaseBindingPlaced
#check productionAccepted_implies_transition
#check receipt_geometry_exact

end tests.Nebula.Implementation.Production.Carrier.StreamingPiRLCNormalizedResidualRows
