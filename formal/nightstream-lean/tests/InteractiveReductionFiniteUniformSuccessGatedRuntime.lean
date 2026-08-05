import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime

/-!
Focused interface regression for the success-gated extractor runtime theorem.
-/

namespace tests.InteractiveReductionFiniteUniformSuccessGatedRuntime

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime

#check Family
#check AdversaryExpectedPolynomialTime
#check gatedRetryExpectedWork_le_oneRun
#check expectedWork_le_twoRunWorkBound
#check terminatesAt
#check ExtractorExpectedPolynomialTime
#check extractorExpectedPolynomialTime

end tests.InteractiveReductionFiniteUniformSuccessGatedRuntime
