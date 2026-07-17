import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.ArtifactRefinement
import tests.Axioms.Support

/-! Fail-closed kernel dependency ownership for the generated aggregate leaf. -/

open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.generated_aggregate_shape_exact' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms generated_aggregate_shape_exact

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.generatedAggregateAcceptanceRows_iff' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generatedAggregateAcceptanceRows_iff

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.generatedAggregateAcceptanceRows_iff_verifierMeaning' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generatedAggregateAcceptanceRows_iff_verifierMeaning
