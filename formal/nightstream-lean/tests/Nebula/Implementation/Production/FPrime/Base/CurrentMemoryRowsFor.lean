import Nightstream.Implementation.Nebula.Production.FPrime.Base.CurrentMemoryRowsFor

/-! Regression surface for fixed base current-memory row ownership. -/

namespace tests.NebulaProductionBaseCurrentMemoryRowsFor

open Nightstream.Implementation.Nebula.ProductionBaseCurrentMemoryRowsFor

#check Authority.satisfied
#check Authority.headersPlaced
#check Authority.result
#check Authority.outgoingValue_eq_firstBoundary

end tests.NebulaProductionBaseCurrentMemoryRowsFor
