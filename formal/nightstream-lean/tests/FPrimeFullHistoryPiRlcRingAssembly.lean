import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.RingAssembly

/-! Kernel theorem and dependency checks for terminal PiRLC ring assembly. -/

namespace tests.FPrimeFullHistoryPiRlcRingAssembly

open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.RingAssembly

#check challengeColumn_formula
#check embedCoefficient_eq_semantic
#check fieldResidue_centeredField_eq_embedCoefficient
#check embeddedAlphabet_values
#check machineChallenge_eq_embedScalar
#check decodedChallenge_eq_machineChallenge
#check decodedChallenges_eq_machineChallenges
#check machineChallenge_coefficients_valid

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.RingAssembly.challengeColumn_formula' depends on axioms: [propext] -/
#guard_msgs in
#print axioms challengeColumn_formula

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.RingAssembly.decodedChallenge_eq_machineChallenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms decodedChallenge_eq_machineChallenge

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.RingAssembly.decodedChallenges_eq_machineChallenges' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms decodedChallenges_eq_machineChallenges

end tests.FPrimeFullHistoryPiRlcRingAssembly
