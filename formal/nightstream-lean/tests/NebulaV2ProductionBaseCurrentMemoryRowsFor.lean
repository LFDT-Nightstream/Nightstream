import Nightstream.Implementation.NebulaV2.ProductionBaseCurrentMemoryRowsFor

/-! Regression surface for fixed base current-memory row ownership. -/

namespace tests.NebulaV2ProductionBaseCurrentMemoryRowsFor

open Nightstream.Implementation.NebulaV2.ProductionBaseCurrentMemoryRowsFor

#check Authority.satisfied
#check Authority.headersPlaced
#check Authority.result
#check Authority.outgoingValue_eq_firstBoundary

end tests.NebulaV2ProductionBaseCurrentMemoryRowsFor
