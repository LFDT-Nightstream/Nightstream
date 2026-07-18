import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.ChallengeWiring
import tests.Axioms.Support

/-! Fail-closed dependencies for the active PiRLC challenge wiring artifact. -/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ChallengeWiring

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ChallengeWiring.structure_check' does not depend on any axioms -/
#guard_msgs in
#audit_axioms structure_check

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ChallengeWiring.projection_consumers_are_sampler_outputs' does not depend on any axioms -/
#guard_msgs in
#audit_axioms projection_consumers_are_sampler_outputs
