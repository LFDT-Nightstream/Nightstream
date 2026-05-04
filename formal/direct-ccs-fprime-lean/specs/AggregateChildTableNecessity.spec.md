# Aggregate Child Table Necessity

`AggregateChildTableNecessity` specifies why aggregate summaries of private
post-DEC child tables are not sound authorization for the parent-only direct
CCS `F'` path.

The module defines an aggregate digit summary:

```text
aggregateDigitSum(children)(column) = sum(children[column])
```

and an intentionally weak aggregate-only validation predicate:

```text
binary child digits
exact fixed child-column length
aggregateDigitSum(children) = summary
```

The theorem `aggregate_digit_sum_not_functional_for_binary_fixed_length` states
that this validation shape is not functional. Two different binary fixed-length
child tables can have the same aggregate summary.

The theorem `aggregate_norm_sum_not_functional_for_fixed_child_count` states
the same necessity claim for a length-14 vector of child norms. The total norm
sum does not determine which child carries which norm, so it cannot bind child
identity.

The theorem `aggregate_only_validation_can_feed_different_next_inputs` gives
the concrete one-column counterexample:

```text
[1, 0, 0, ..., 0]
[0, 1, 0, ..., 0]
```

Both tables are binary, have length 14, and have aggregate digit sum 1. They
are different child tables and have different base-2 recompositions. Therefore
aggregate summaries cannot replace pointwise DEC recomposition, fixed child CE
membership, and wire identity into the next `Pi_CCS` inputs.
