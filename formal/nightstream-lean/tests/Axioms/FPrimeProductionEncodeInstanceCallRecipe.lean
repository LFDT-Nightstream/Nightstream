import tests.FPrimeProductionEncodeInstanceCallRecipe
import tests.Axioms.Support

/-!
Fail-closed guards for the production fixed-one `encodeInstance` typed call
recipe.
-/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceCallRecipe.selected_footprint_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceCallRecipe.selected_footprint_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceCallRecipe.recipe_row_count' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceCallRecipe.recipe_row_count

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceCallRecipe.receipt_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodeInstanceCallRecipe.receipt_exact
