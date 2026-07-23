import Nightstream.SuperNeo.InteractiveReduction.StrongConditioningObstruction

/-!
Focused interface regression for the Appendix-D.4 first-success conditioning
counterexample.
-/

namespace tests.InteractiveReductionStrongConditioningObstruction

open Nightstream.SuperNeo.InteractiveReduction.StrongConditioningObstruction

#check exact_counts
#check raw_bound_does_not_bound_conditioned_disagreement
#check unchanged_raw_uniqueness_budget_counterexample

example : rawDisagreementCount = 2 :=
  exact_counts.2.2.2.2.1

example : conditionedDisagreementCount = 2 :=
  exact_counts.2.2.2.2.2

example : rawPairs.length = 16 :=
  exact_counts.2.2.1

example : firstSuccessConditionedPairs.length = 8 :=
  exact_counts.2.2.2.1

end tests.InteractiveReductionStrongConditioningObstruction
