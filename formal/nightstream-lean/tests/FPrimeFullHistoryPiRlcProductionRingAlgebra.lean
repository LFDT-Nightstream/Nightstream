import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.ProductionRingAlgebra

/-! Dependency guards for the production Phi81 paper-facing algebra. -/

namespace tests.FPrimeFullHistoryPiRlcProductionRingAlgebra

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ProductionRingAlgebra

#check ringOfList_canonicalRing
#check challengeMember_length
#check challengeMember_ringOfList
#check productionRingAlgebra_membership_iff
#check StrongChallengeSet
#check productionRingAlgebra_strong

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ProductionRingAlgebra.ringOfList_canonicalRing' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms ringOfList_canonicalRing

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.ProductionRingAlgebra.productionRingAlgebra_strong' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms productionRingAlgebra_strong

end tests.FPrimeFullHistoryPiRlcProductionRingAlgebra
