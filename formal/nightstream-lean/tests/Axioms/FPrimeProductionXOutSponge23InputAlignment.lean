import tests.FPrimeProductionXOutSponge23InputAlignment
import tests.Axioms.Support

/-!
Fail-closed guards for exact plain/stateless XOut source-vector alignment
with the selected fused 23-field Poseidon2 sponge recipe.
-/

namespace NightstreamTests.Axioms.FPrimeProductionXOutSponge23InputAlignment

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open ProductionXOutSponge23InputAlignment

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionXOutSponge23InputAlignment.Source.fields_eq_encodeStateXOutPreimage' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Source.fields_eq_encodeStateXOutPreimage

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionXOutSponge23InputAlignment.Source.fields_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Source.fields_length

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionXOutSponge23InputAlignment.Source.emptyTable_wellFormed_but_sourceWidth_seven' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Source.emptyTable_wellFormed_but_sourceWidth_seven

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionXOutSponge23InputAlignment.numericInputs_eq_sourceFields' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms numericInputs_eq_sourceFields

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionXOutSponge23InputAlignment.semanticLane_eq_sourceLane' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms semanticLane_eq_sourceLane

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionXOutSponge23InputAlignment.active_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms active_sound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionXOutSponge23InputAlignment.active_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms active_complete

end NightstreamTests.Axioms.FPrimeProductionXOutSponge23InputAlignment
