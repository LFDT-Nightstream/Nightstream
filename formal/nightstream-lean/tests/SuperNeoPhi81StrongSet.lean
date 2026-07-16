import Nightstream.SuperNeo.Concrete.Phi81StrongSet

/-! Dependency guards for the concrete Phi81 strong-set instantiation. -/

namespace tests.SuperNeoPhi81StrongSet

open Nightstream.SuperNeo.Concrete.Phi81StrongSet

#check embedCoefficient_values
#check embedScalar_injective
#check embeddedDifference_nonzero
#check embeddedDifference_normAtMostFour
#check theorem8Conditions_exact
#check differenceBound_below_goldilocks
#check LowNormInvertibility
#check productionSet_strong

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedCoefficient_values' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms embedCoefficient_values

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedScalar_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms embedScalar_injective

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81StrongSet.embeddedDifference_nonzero' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms embeddedDifference_nonzero

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81StrongSet.embeddedDifference_normAtMostFour' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms embeddedDifference_normAtMostFour

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81StrongSet.theorem8Conditions_exact' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms theorem8Conditions_exact

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81StrongSet.productionSet_strong' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms productionSet_strong

end tests.SuperNeoPhi81StrongSet
