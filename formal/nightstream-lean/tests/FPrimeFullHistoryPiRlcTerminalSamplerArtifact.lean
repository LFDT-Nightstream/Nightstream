import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.TerminalSamplerArtifact

/-!
Kernel theorem and dependency checks for the terminal paper sampler facade.

Assurance tier: artifact-checked theorem surface with fail-closed guards.
-/

namespace tests.FPrimeFullHistoryPiRlcTerminalSamplerArtifact

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact

#check terminalTrace_count
#check publicRoleIndex_census
#check publicShared
#check decodedRing_eq_machineRing
#check machineRing_member
#check challengeWiringArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact.publicRoleIndex_census' does not depend on any axioms -/
#guard_msgs in
#print axioms publicRoleIndex_census

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact.publicShared' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms publicShared

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact.decodedRing_eq_machineRing' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms decodedRing_eq_machineRing

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact.challengeWiringArtifact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms challengeWiringArtifact

end tests.FPrimeFullHistoryPiRlcTerminalSamplerArtifact
