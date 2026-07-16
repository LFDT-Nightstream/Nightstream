import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Recursive.RingAssembly

/-! Kernel theorem and dependency checks for recursive ring assembly. -/

namespace tests.FPrimeFullHistoryPiRlcRecursiveRingAssembly

open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Recursive.RingAssembly

#check challengeColumn_formula
#check challengeColumns_length
#check machineChallenge_eq_embedScalar
#check decodedChallenge_eq_machineChallenge
#check machineChallenge_coefficients_valid

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Recursive.RingAssembly.challengeColumn_formula' depends on axioms: [propext] -/
#guard_msgs in
#print axioms challengeColumn_formula

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Recursive.RingAssembly.decodedChallenge_eq_machineChallenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms decodedChallenge_eq_machineChallenge

end tests.FPrimeFullHistoryPiRlcRecursiveRingAssembly
