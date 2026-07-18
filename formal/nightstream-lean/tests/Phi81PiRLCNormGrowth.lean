import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm

/-!
Focused theorem-surface checks for concrete Phi81 `PiRLC` norm growth.

| Stage path | Regression |
|---|---|
| `nifs.pi_rlc.verify.norm_growth.centered.triangle` | active Goldilocks centered arithmetic has addition/subtraction triangle laws |
| `nifs.pi_rlc.verify.norm_growth.centered.symbol` | one five-symbol coefficient expands by at most two |
| `nifs.pi_rlc.verify.norm_growth.product.raw.support` | raw convolution exposes its exact active support |
| `nifs.pi_rlc.verify.norm_growth.product.reduction.support` | all 54 output-lane support totals are at most `2 * 54` |
| `nifs.pi_rlc.verify.norm_growth.product.expansion` | executable `ringFMul` is bounded by production `216` |
| `nifs.pi_rlc.verify.norm_growth.assignment.finite` | canonical finite assignment fold is bounded by `n * 216` |
| `nifs.pi_rlc.verify.norm_growth.algebra` | theorem has the exact concrete algebra-field signature |
-/

namespace tests.Phi81PiRLCNormGrowth

open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra

#check Norm.centeredMagnitude_add_le
#check Norm.centeredMagnitude_sub_le
#check Norm.embedCoefficient_mul_le_two
#check Norm.rawMulCoeffF_le_support
#check Norm.totalSupport_le_two_degrees
#check Norm.ringFMul_le_expansion
#check Norm.combineAssignments_le
#check Norm.production_total_bound
#check Norm.relation_norm_growth

end tests.Phi81PiRLCNormGrowth
