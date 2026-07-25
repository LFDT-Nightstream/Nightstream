import tests.FPrimeProductionPoseidon2PermutationRecipe
import tests.Axioms.Support

/-!
Fail-closed guards for the activation-compatible exact production Poseidon2
width-eight permutation occurrence.
-/

namespace NightstreamTests.Axioms.FPrimeProductionPoseidon2PermutationRecipe

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.receipt_row_count' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms receipt_row_count

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.rows_owned' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_owned

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.rowIds_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rowIds_nodup

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.execution_output_eq_semantic' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms execution_output_eq_semantic

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.active_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms active_sound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.complete_changesOnly' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms complete_changesOnly

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.complete_agrees_visible' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms complete_agrees_visible

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.complete_temporary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms complete_temporary

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.completedNumeric_eq_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms completedNumeric_eq_execution

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.core_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms core_complete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.active_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms active_complete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionPoseidon2PermutationRecipe.inactive_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms inactive_complete

end NightstreamTests.Axioms.FPrimeProductionPoseidon2PermutationRecipe
