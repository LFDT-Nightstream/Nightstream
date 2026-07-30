import Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe
import tests.Axioms.Support

/-!
Fail-closed axiom guard for the hashPrior / hashNext recipes.

Every report below is measured, not asserted: the expected text was produced by
running the audit and copying its output, so any drift fails the build.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2HashRecipe

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.spongeColumnTotal_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.spongeColumnTotal_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.relocate_pos' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.relocate_pos

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.relocate_injective_on_window' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.relocate_injective_on_window

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.hashProgram_length_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.hashProgram_length_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.hashPrior_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.hashPrior_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.hashNext_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.hashNext_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.ownedColumns_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.ownedColumns_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.ownedColumns_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.ownedColumns_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.mem_ownedColumns' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.mem_ownedColumns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.hashPrior_hashNext_disjoint' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.hashPrior_hashNext_disjoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.mentions_renameTerms' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.mentions_renameTerms

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.hashProgram_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.hashProgram_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.hashProgram_pull' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.hashProgram_pull

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.hashProgram_computes_digest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.hashProgram_computes_digest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.pull_honestAssignment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.pull_honestAssignment

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.hashProgram_honest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.hashProgram_honest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.hashCost_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.hashCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.hashCost_columns' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.hashCost_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.hashPairCost_rows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.hashPairCost_rows

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_rate' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_rate

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_capacity' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_capacity

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_partition' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_partition

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_digest_within_rate' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_digest_within_rate

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_arity' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_arity

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_chunking' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_chunking

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_permutationCalls' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_permutationCalls

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_capacity_untouched' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_capacity_untouched

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_padding_chunk' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_padding_chunk

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_padding_on_constant_wire' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_padding_on_constant_wire

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_padding_value' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_padding_value

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_padding_input_independent' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_padding_input_independent

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_padding_call' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_padding_call

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_absorbed_total' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_absorbed_total

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.separator_survives_absorption' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.separator_survives_absorption

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_single_arity' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_single_arity

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.digest_independent_of_placement' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.digest_independent_of_placement

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.chunkValue_at_index' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.chunkValue_at_index

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.chunkAt_determines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.chunkAt_determines

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.chunk_differs_at_index' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.chunk_differs_at_index

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.separator_reaches_chunk_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.separator_reaches_chunk_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.separatedPreimage_false' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.separatedPreimage_false

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.separatedPreimage_differs' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.separatedPreimage_differs

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.separatedPreimage_reaches_chunk_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.separatedPreimage_reaches_chunk_zero


/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe.committed_separation_survives' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2HashRecipe.committed_separation_survives

end NightstreamTests.Axioms.CanonicalPoseidon2HashRecipe
