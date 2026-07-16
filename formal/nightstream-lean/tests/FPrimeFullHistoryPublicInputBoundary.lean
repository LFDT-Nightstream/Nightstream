import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicInputBoundary

/-! Kernel checks for the fixed production public-input mismatch. -/

namespace tests.FPrimeFullHistoryPublicInputBoundary

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PublicInputBoundary

#check productionPublicWidth_not_aligned
#check publicProjection_not_injective
#check ringAction_enters_extra_coefficient
#check currentBinding_accepts_falseCeCarrier
#check distinguishedBinding_not_sufficient_for_ceCarrier

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PublicInputBoundary.ringAction_enters_extra_coefficient' does not depend on any axioms -/
#guard_msgs in
#print axioms ringAction_enters_extra_coefficient

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PublicInputBoundary.currentBinding_accepts_falseCeCarrier' depends on axioms: [propext] -/
#guard_msgs in
#print axioms currentBinding_accepts_falseCeCarrier

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PublicInputBoundary.distinguishedBinding_not_sufficient_for_ceCarrier' depends on axioms: [propext] -/
#guard_msgs in
#print axioms distinguishedBinding_not_sufficient_for_ceCarrier

end tests.FPrimeFullHistoryPublicInputBoundary
