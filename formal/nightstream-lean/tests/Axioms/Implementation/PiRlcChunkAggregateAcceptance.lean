import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance
import tests.Axioms.Support

/-! Fail-closed kernel dependency ownership for the chunk aggregate relation. -/

open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.productTreeAggregateRow_iff' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms productTreeAggregateRow_iff

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.aggregateAcceptanceRows_iff_verifierMeaning' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms aggregateAcceptanceRows_iff_verifierMeaning

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.aggregateAcceptanceRows_extension_exact' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms aggregateAcceptanceRows_extension_exact

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.productTreeOutputBitRows_are_necessary' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms productTreeOutputBitRows_are_necessary

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.productTreeAggregateRow_is_necessary' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms productTreeAggregateRow_is_necessary

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.finalAcceptanceRow_is_necessary' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms finalAcceptanceRow_is_necessary
