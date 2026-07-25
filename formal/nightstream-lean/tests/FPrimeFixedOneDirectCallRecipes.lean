import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

/-!
Focused model-level surface for the five selected direct fixed-one call
recipes.  Each `CallRecipe` contains its exact row-count, ownership,
row-support, active soundness, active honest-completeness, and inactive
satisfiability contracts.
-/

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

#check iterationZeroRecipe
#check stateEqualRecipe
#check freshPublicRecipe
#check encodeInstanceRecipe
#check encodedEqualRecipe
#check certifiedSubset
#check RemainingRecipes
#check allRecipes
#check remainingCalls_exact
