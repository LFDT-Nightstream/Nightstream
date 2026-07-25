import tests.FPrimeProductionIterationZeroCallRecipe
import tests.Axioms.Support

/-!
Fail-closed guards for the production fixed-one `iterationZero` typed call
recipe.
-/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionIterationZeroCallRecipe.selected_footprint_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionIterationZeroCallRecipe.selected_footprint_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionIterationZeroCallRecipe.recipe_row_count' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionIterationZeroCallRecipe.recipe_row_count

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionIterationZeroCallRecipe.receipt_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionIterationZeroCallRecipe.receipt_exact
