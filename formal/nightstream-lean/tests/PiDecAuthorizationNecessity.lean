import Nightstream.SuperNeo.Folding.PiDEC.Necessity
import Nightstream.Implementation.R1CS.Correspondence.PiDecStrict.Necessity

/-!
Focused regressions for semantic and exact-implementation `Pi_DEC` child
authorization countermodels.

| Layer | Phase | Family | Regression |
|---|---|---|---|
| paper arithmetic | children | low norm / arity / range | each omitted obligation admits a concrete ambiguity |
| paper authority | children | aggregate summaries | totals do not determine pointwise child values or norms |
| implementation | strict `Pi_DEC` | recomposition + centered alphabet | exact Goldilocks predicates accept two different child vectors |
-/

#check Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization.signed_low_norm_base2_not_unique
#check Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization.binary_recomposition_not_unique_without_length
#check Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization.fixed_length_binary_mod_recomposition_not_unique_without_range
#check Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization.aggregate_digit_sum_not_functional_for_fixed_child_count
#check Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization.aggregate_norm_sum_not_functional_for_fixed_child_count
#check Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization.aggregate_only_validation_can_feed_different_next_inputs
#check Nightstream.Implementation.R1CS.PiDecStrictNecessity.recomposition_and_centered_alphabet_not_functional
