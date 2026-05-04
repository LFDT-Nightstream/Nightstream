import DirectCcsFPrime.AggregateChildTableNecessity

/-!
Typed interface for aggregate-only child-table necessity.

Spec: `specs/AggregateChildTableNecessity.spec.md`
-/

namespace DirectCcsFPrime

namespace AggregateChildTableNecessityInterface

abbrev aggregateDigitSum :=
  @AggregateChildTableNecessity.aggregateDigitSum

abbrev aggregateNormSum :=
  @AggregateChildTableNecessity.aggregateNormSum

abbrev AggregateOnlyChildValidation :=
  @AggregateChildTableNecessity.AggregateOnlyChildValidation

abbrev AcceptedAggregateOnlyChildTable :=
  @AggregateChildTableNecessity.AcceptedAggregateOnlyChildTable

abbrev aggregate_digit_sum_not_functional_for_binary_fixed_length :=
  @AggregateChildTableNecessity.aggregate_digit_sum_not_functional_for_binary_fixed_length

abbrev aggregate_norm_sum_not_functional_for_fixed_child_count :=
  @AggregateChildTableNecessity.aggregate_norm_sum_not_functional_for_fixed_child_count

abbrev aggregate_only_validation_can_feed_different_next_inputs :=
  @AggregateChildTableNecessity.aggregate_only_validation_can_feed_different_next_inputs

end AggregateChildTableNecessityInterface

end DirectCcsFPrime
