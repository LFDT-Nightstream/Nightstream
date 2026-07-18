import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.ProjectionConsumer
import tests.Axioms.Support

/-! Fail-closed dependencies for the active PiRLC challenge handoff. -/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ProjectionConsumer

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ProjectionConsumer.sampler_output_column_eq_layout' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms sampler_output_column_eq_layout

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ProjectionConsumer.sampler_columns_eq_layout' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms sampler_columns_eq_layout

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ProjectionConsumer.sampler_columns_eq_projection_columns' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms sampler_columns_eq_projection_columns

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ProjectionConsumer.decodedSamplerChallenges_eq_decodedChallenges' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodedSamplerChallenges_eq_decodedChallenges

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ProjectionConsumer.samplerChallengesBound_fieldDerived' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms samplerChallengesBound_fieldDerived

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ProjectionConsumer.decodedChallenges_eq_of_samplerBound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodedChallenges_eq_of_samplerBound
