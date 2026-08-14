import Nightstream.Implementation.Nebula.Production.Carrier.CarrierLayoutFor
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.ProductionFullClaimCarrierLayoutFor

/-! Dependency audit for the exponent-indexed physical full-claim layout. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimCarrierLayoutFor.section_offsets_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms section_offsets_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimCarrierLayoutFor.endOffset_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms endOffset_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimCarrierLayoutFor.counter_intervals_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms counter_intervals_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimCarrierLayoutFor.checkedMemoryPlacement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms checkedMemoryPlacement

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimCarrierLayoutFor.nifsRunningValues_eq_carrier' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms nifsRunningValues_eq_carrier

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimCarrierLayoutFor.nifsBundleValues_eq_carrier' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms nifsBundleValues_eq_carrier

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimCarrierLayoutFor.memoryNativeColumn_lt_end' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms memoryNativeColumn_lt_end
