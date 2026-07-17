import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.RecursiveSamplerArtifact

/-!
Kernel theorem and dependency checks for the recursive paper sampler facade.

Assurance tier: artifact-checked theorem surface with fail-closed guards.
-/

namespace tests.FPrimeFullHistoryPiRlcRecursiveSamplerArtifact

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact

#check recursiveTrace_count
#check recursiveTotal_eq_one
#check publicRoleIndex_census
#check publicShared
#check decodedRing_eq_machineRing
#check machineRing_member
#check challengeWiringArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.publicRoleIndex_census' does not depend on any axioms -/
#guard_msgs in
#print axioms publicRoleIndex_census

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.publicShared' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms publicShared

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.decodedRing_eq_machineRing' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms decodedRing_eq_machineRing

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.challengeWiringArtifact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms challengeWiringArtifact

end tests.FPrimeFullHistoryPiRlcRecursiveSamplerArtifact
