import Nightstream.Implementation.Nebula
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.SeedSchedule
open Nightstream.Implementation.Nebula.FieldCodec
open Nightstream.Implementation.Nebula.SnapshotRows
open Nightstream.Implementation.Nebula.WasmStateCodec
open Nightstream.Implementation.Nebula.WasmResultCodec
open Nightstream.Implementation.Nebula.WasmPublicStatementCodec
open Nightstream.Implementation.Nebula.WasmStatementBytes
open Nightstream.Implementation.Nebula.WasmStatementParser

/-- info: 'Nightstream.Implementation.Nebula.SeedSchedule.exact_fixed_role_geometry' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms exact_fixed_role_geometry

/-- info: 'Nightstream.Implementation.Nebula.SeedSchedule.Manifest.different_roles_have_different_seeds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Manifest.different_roles_have_different_seeds

/-- info: 'Nightstream.Implementation.Nebula.FieldCodec.CallSite.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms CallSite.sound

/-- info: 'Nightstream.Implementation.Nebula.BoundedWordRows.value_lt_twoPower' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.BoundedWordRows.value_lt_twoPower

/-- info: 'Nightstream.Implementation.Nebula.BoundedWordRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.BoundedWordRows.rows_complete

/-- info: 'Nightstream.Implementation.Nebula.LessThanConstantRows.value_lt_limit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.LessThanConstantRows.value_lt_limit

/-- info: 'Nightstream.Implementation.Nebula.LessThanConstantRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.LessThanConstantRows.rows_complete

/-- info: 'Nightstream.Implementation.Nebula.ConditionalEqualityRows.rows_sound_closed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ConditionalEqualityRows.rows_sound_closed

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimCounterRows.claim_canonical_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryClaimCounterRows.claim_canonical_of_rows

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimCounterRows.CallSite.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryClaimCounterRows.CallSite.sound

/-- info: 'Nightstream.Implementation.Nebula.FieldCodec.local_complete' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#audit_axioms local_complete

/-- info: 'Nightstream.Implementation.Nebula.SnapshotRows.accepts_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms accepts_sound

/-- info: 'Nightstream.Implementation.Nebula.SnapshotRows.accepts_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepts_complete

/-- info: 'Nightstream.Implementation.Nebula.WasmStateCodec.encode_injective_on_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms encode_injective_on_canonical

/-- info: 'Nightstream.Implementation.Nebula.WasmStateCodec.encode_exact_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms encode_exact_length

/-- info: 'Nightstream.Implementation.Nebula.WasmResultCodec.encodeDigest_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms encodeDigest_injective

/-- info: 'Nightstream.Implementation.Nebula.WasmResultCodec.encode_injective_of_decodes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.WasmResultCodec.encode_injective_of_decodes

/-- info: 'Nightstream.Implementation.Nebula.WasmPublicStatementCodec.encode_injective_of_decodes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.WasmPublicStatementCodec.encode_injective_of_decodes

/-- info: 'Nightstream.Implementation.Nebula.WasmPublicStatementCodec.encode_injective_of_decodesFor' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.WasmPublicStatementCodec.encode_injective_of_decodesFor

/-- info: 'Nightstream.Implementation.Nebula.WasmStatementBytes.join_split' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.WasmStatementBytes.join_split

/-- info: 'Nightstream.Implementation.Nebula.WasmStatementBytes.encode_injective_of_decodes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.WasmStatementBytes.encode_injective_of_decodes

/-- info: 'Nightstream.Implementation.Nebula.WasmStatementParser.parse_encode' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.WasmStatementParser.parse_encode

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimCodec.schema_width_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryClaimCodec.schema_width_exact

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimCodec.encode_injective_on_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryClaimCodec.encode_injective_on_canonical

/-- info: 'Nightstream.Implementation.Nebula.CommitmentBundleCodec.schema_width_exact' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.CommitmentBundleCodec.schema_width_exact

/-- info: 'Nightstream.Implementation.Nebula.CommitmentBundleCodec.encode_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.CommitmentBundleCodec.encode_injective

/-- info: 'Nightstream.Implementation.Nebula.BundleForwardingRows.exact_bundle_forwarding' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.BundleForwardingRows.exact_bundle_forwarding

/-- info: 'Nightstream.Implementation.Nebula.BundleForwardingRows.CallSite.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.BundleForwardingRows.CallSite.sound

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryCodec.schema_width_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryCodec.schema_width_exact

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryCodec.encode_injective_on_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryCodec.encode_injective_on_canonical

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryRows.value_canonical_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryRows.value_canonical_of_rows

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryRows.CallSite.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryRows.CallSite.sound

/-- info: 'Nightstream.Implementation.Nebula.CanonicalFieldSchemaRows.all_slots_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.CanonicalFieldSchemaRows.all_slots_sound

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimFieldRows.typed_columns_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryClaimFieldRows.typed_columns_of_rows

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimFieldRows.modulus_alias_impossible' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryClaimFieldRows.modulus_alias_impossible

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryFieldRows.typed_columns_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryFieldRows.typed_columns_of_rows

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryFieldRows.modulus_alias_impossible' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryFieldRows.modulus_alias_impossible

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimParser.parse_native_parses' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryClaimParser.parse_native_parses

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimParser.parse_claim_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryClaimParser.parse_claim_canonical

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimParser.parse_blockOfClaim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryClaimParser.parse_blockOfClaim

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimParser.rejects_modulus_alias' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryClaimParser.rejects_modulus_alias

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryParser.parse_native_parses' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryParser.parse_native_parses

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryParser.parse_value_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryParser.parse_value_canonical

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryParser.parse_blockOfValue' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryParser.parse_blockOfValue

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryParser.rejects_modulus_alias' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryParser.rejects_modulus_alias

/-- info: 'Nightstream.Implementation.Nebula.LessThanConstantLinkedRows.value_lt_limit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.LessThanConstantLinkedRows.value_lt_limit

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimRows.parsed_columns_match' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryClaimRows.parsed_columns_match

/-- info: 'Nightstream.Implementation.Nebula.MemoryProductBalanceRows.concreteBalanced_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryProductBalanceRows.concreteBalanced_of_rows

/-- info: 'Nightstream.Implementation.Nebula.MemoryProductBalanceRows.parsed_claim_balanced_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryProductBalanceRows.parsed_claim_balanced_of_rows

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryPublicRows.parsed_columns_match' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryPublicRows.parsed_columns_match

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryPublicRows.rows_force_parse' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryPublicRows.rows_force_parse

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryPublicRows.rows_force_parsed_columns_match' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryPublicRows.rows_force_parsed_columns_match

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperPriorStateAuthorityRowsFor.rows_imply_exact_prior_state_and_fullMatches' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperPriorStateAuthorityRowsFor.rows_imply_exact_prior_state_and_fullMatches

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryHashFrame.frame_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryHashFrame.frame_injective

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryHashBinding.parsed_value_eq_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryHashBinding.parsed_value_eq_or_collision

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryHashPackingRows.packed_columns_eq_encodePacked' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryHashPackingRows.packed_columns_eq_encodePacked

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryHashPackingRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryHashPackingRows.rows_complete

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryHashFrameRows.input_column_values' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryHashFrameRows.input_column_values

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryPoseidonRows.output_columns_eq_carryDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryPoseidonRows.output_columns_eq_carryDigest

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryPoseidonRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryPoseidonRows.rows_complete

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryPoseidonBinding.parsed_value_eq_or_poseidon_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryPoseidonBinding.parsed_value_eq_or_poseidon_collision

/-- info: 'Nightstream.Implementation.Nebula.StateOutputFrameRows.input_column_values' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.StateOutputFrameRows.input_column_values

/-- info: 'Nightstream.Implementation.Nebula.StateOutputFrameRows.canonical_shape_eq_v2_iff' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.StateOutputFrameRows.canonical_shape_eq_v2_iff

/-- info: 'Nightstream.Implementation.Nebula.StateOutputPoseidonRows.output_columns_eq_pureDigest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.StateOutputPoseidonRows.output_columns_eq_pureDigest

/-- info: 'Nightstream.Implementation.Nebula.MemoryCarryStateOutputRows.output_columns_eq_stateDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryCarryStateOutputRows.output_columns_eq_stateDigest

/-- info: 'Nightstream.Implementation.Nebula.StateOutputPoseidonBinding.satisfying_rows_bind_authority_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.StateOutputPoseidonBinding.satisfying_rows_bind_authority_or_collision

/-- info: 'Nightstream.Implementation.Nebula.StateOutputRowCensus.composed_rows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.StateOutputRowCensus.composed_rows_length

/-- info: 'Nightstream.Implementation.Nebula.U64HalvesRows.u64Halves_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.U64HalvesRows.u64Halves_injective

/-- info: 'Nightstream.Implementation.Nebula.StateOutputAuthorityRows.payloadFields_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.StateOutputAuthorityRows.payloadFields_injective

/-- info: 'Nightstream.Implementation.Nebula.StateOutputAuthorityRows.payload_column_values' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.StateOutputAuthorityRows.payload_column_values

/-- info: 'Nightstream.Implementation.Nebula.AuthoritativeStateOutputRows.output_columns_eq_typed_stateDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.AuthoritativeStateOutputRows.output_columns_eq_typed_stateDigest

/-- info: 'Nightstream.Implementation.Nebula.AuthoritativeStateOutputRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.AuthoritativeStateOutputRows.rows_complete

/-- info: 'Nightstream.Implementation.Nebula.AuthoritativeStateOutputBinding.satisfying_rows_bind_typed_authority_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.AuthoritativeStateOutputBinding.satisfying_rows_bind_typed_authority_or_collision

/-- info: 'Nightstream.Implementation.Nebula.UnsignedAdditionRows.output_eq_add' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.UnsignedAdditionRows.output_eq_add

/-- info: 'Nightstream.Implementation.Nebula.UnsignedLessOrEqualRows.left_le_right' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.UnsignedLessOrEqualRows.left_le_right

/-- info: 'Nightstream.Implementation.Nebula.ConditionalCarriedEqualityRows.rows_sound_closed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ConditionalCarriedEqualityRows.rows_sound_closed

/-- info: 'Nightstream.Implementation.Nebula.MemoryTransitionSound.core_evidence' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryTransitionSound.core_evidence

/-- info: 'Nightstream.Implementation.Nebula.MemoryTransitionSound.consumes_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.MemoryTransitionSound.consumes_of_rows
