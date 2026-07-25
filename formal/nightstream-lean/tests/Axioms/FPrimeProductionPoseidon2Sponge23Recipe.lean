import tests.FPrimeProductionPoseidon2Sponge23Recipe
import tests.Axioms.Support

/-!
Fail-closed guards for the selected fused production Poseidon2 sponge
occurrence on the 23-field plain/stateless XOut preimage.
-/

namespace NightstreamTests.Axioms.FPrimeProductionPoseidon2Sponge23Recipe

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open ProductionPoseidon2Sponge23Recipe
open ProductionPoseidon2Sponge23Audit

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe.NumericSponge.trace_valid' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NumericSponge.trace_valid

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe.NumericSponge.emissionReceipt' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NumericSponge.emissionReceipt

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe.receipt_row_count' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms receipt_row_count

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe.rowIds_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rowIds_nodup

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe.active_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms active_sound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe.completedNumeric_eq_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms completedNumeric_eq_execution

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe.active_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms active_complete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Recipe.inactive_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms inactive_complete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Audit.Recipe.rows_supported' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Recipe.rows_supported

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Audit.Recipe.receipt_row_column_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Recipe.receipt_row_column_conservation

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Audit.Recipe.normalized_row_column_conservation' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Recipe.normalized_row_column_conservation

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Audit.RewriteClass.selected_cost' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms RewriteClass.selected_cost

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2Sponge23Audit.RewriteClass.selected_minimum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RewriteClass.selected_minimum

end NightstreamTests.Axioms.FPrimeProductionPoseidon2Sponge23Recipe
