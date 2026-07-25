import tests.FPrimeProductionEncodedEqualCallRecipe
import tests.Axioms.Support

/-!
Fail-closed guards for the production fixed-one `encodedEqual` typed call
recipe.
-/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodedEqualCallRecipe.selected_footprint_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodedEqualCallRecipe.selected_footprint_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodedEqualCallRecipe.recipe_row_count' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodedEqualCallRecipe.recipe_row_count

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodedEqualCallRecipe.receipt_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionEncodedEqualCallRecipe.receipt_exact
