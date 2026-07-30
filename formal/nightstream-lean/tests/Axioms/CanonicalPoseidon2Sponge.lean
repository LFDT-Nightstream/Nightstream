import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the reference sponge.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Sponge

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.capacity_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.capacity_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.rate_add_capacity' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.rate_add_capacity

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digest_within_rate' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.digest_within_rate

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorb_nil' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.absorb_nil

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorb_cons' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.absorb_cons

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbChunk_beyond_chunk' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.absorbChunk_beyond_chunk

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.pad_beyond_zero' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.pad_beyond_zero

/-! Permutation-call structure of the specification. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorb_permutation_count' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.absorb_permutation_count

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.permutationCalls_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.permutationCalls_eq

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.permutationCalls_empty' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.permutationCalls_empty

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeFinal_empty' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeFinal_empty

/-! Trailing-zero collision in the specification. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbChunk_trailing_zero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.absorbChunk_trailing_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.trailing_zero_inputs_differ' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.trailing_zero_inputs_differ

/-! Capacity lanes and the rate bound. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.oversized_chunk_touches_capacity' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.oversized_chunk_touches_capacity

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbChunk_capacity_untouched' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.absorbChunk_capacity_untouched

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.RateChunk_capacity_untouched' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.RateChunk_capacity_untouched

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbChunk_injective_at_lane' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.absorbChunk_injective_at_lane

/-! The fixed 23-field recipe. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.sponge23_permutationCalls' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.sponge23_permutationCalls

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.sponge23_chunk_arithmetic' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.sponge23_chunk_arithmetic

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.sponge23_single_arity' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.sponge23_single_arity

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.sponge23_final_chunk_bounded' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.sponge23_final_chunk_bounded

/-! Per-call layout chaining and the absorption entry. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.entryOf_mentions' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.entryOf_mentions

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.entryOf_beyond_chunk' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.entryOf_beyond_chunk

/-! The seven-call sponge row program. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.sponge23Program_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.sponge23Program_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeProgram_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeProgram_length

/-! The entry evaluates to the absorb step. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.entryOf_eval_is_absorbChunk' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.entryOf_eval_is_absorbChunk

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.entryOf_succ_eval_absorbed' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.entryOf_succ_eval_absorbed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.entryOf_succ_eval_carried' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.entryOf_succ_eval_carried

/-! From sponge satisfaction to per-call soundness. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeProgram_call_computes_reference' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeProgram_call_computes_reference

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeProgram_satisfies_call' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeProgram_satisfies_call

/-! The seven-call chain. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeChain' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeChain

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.entryOf_zero_eval_is_absorbChunk' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.entryOf_zero_eval_is_absorbChunk

/-! The chain is the sponge absorption. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeChain_is_absorption' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeChain_is_absorption

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.chainValues_permuted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.chainValues_permuted

/-! Padding is absorption of the chunk [1]. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.pad_eq_absorbChunk_one' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.pad_eq_absorbChunk_one

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.pad_eq_absorbChunk_one_funext' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.pad_eq_absorbChunk_one_funext

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.paddingChunk_absorbs' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.paddingChunk_absorbs

/-! The sponge is absorption alone. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeFinal_eq_absorb_padding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeFinal_eq_absorb_padding

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digest_eq_absorb_padding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.digest_eq_absorb_padding

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorb_append' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.absorb_append

/-! The indexing bridge and digest soundness. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeProgram_computes_digest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeProgram_computes_digest

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.absorbAt_eq_absorb' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.absorbAt_eq_absorb

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.chunkList_succ' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.chunkList_succ

/-! The sponge cost. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.sponge23Cost' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.sponge23Cost

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeCost_rows_eq_program' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeCost_rows_eq_program

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeCost_auxiliary_per_call' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeCost_auxiliary_per_call

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeCallTemporaryColumns_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeCallTemporaryColumns_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeTemporaryColumns_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeTemporaryColumns_length

/-! Sponge layout well-formedness. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.canonicalSpongeLayout_wellFormed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.canonicalSpongeLayout_wellFormed

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.SpongeLayout.WellFormed.auxDisjoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.SpongeLayout.WellFormed.auxDisjoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeStride_clears' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeStride_clears

/-! Sponge-level column classification. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeScheduleOf_columns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeScheduleOf_columns

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeScheduleOf_no_foreign_aux' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeScheduleOf_no_foreign_aux

/-! Assembled sponge conservation. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeProgram_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeProgram_conservation

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeCall_conservation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeCall_conservation

/-! Carried-entry and sponge ownership. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeProgram_eq_map_owners' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeProgram_eq_map_owners

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.spongeOwners_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Sponge.spongeOwners_length

end NightstreamTests.Axioms.CanonicalPoseidon2Sponge
