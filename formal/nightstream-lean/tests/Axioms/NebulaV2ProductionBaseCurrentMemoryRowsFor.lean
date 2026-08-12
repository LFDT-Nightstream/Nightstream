import Nightstream.Implementation.NebulaV2.ProductionBaseCurrentMemoryRowsFor
import tests.Axioms.Support

/-! Dependency audit for fixed base current-memory row ownership. -/

open Nightstream.Implementation.NebulaV2.ProductionBaseCurrentMemoryRowsFor

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionBaseCurrentMemoryRowsFor.Authority.satisfied' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Authority.satisfied

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionBaseCurrentMemoryRowsFor.Authority.headersPlaced' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Authority.headersPlaced

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionBaseCurrentMemoryRowsFor.Authority.result' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Authority.result

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionBaseCurrentMemoryRowsFor.Authority.outgoingValue_eq_firstBoundary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Authority.outgoingValue_eq_firstBoundary
