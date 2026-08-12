import Nightstream.Implementation.NebulaV2
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.SeedSchedule
open Nightstream.Implementation.NebulaV2.FieldCodec
open Nightstream.Implementation.NebulaV2.SnapshotRows
open Nightstream.Implementation.NebulaV2.WasmStateCodec
open Nightstream.Implementation.NebulaV2.WasmResultCodec
open Nightstream.Implementation.NebulaV2.WasmPublicStatementCodec
open Nightstream.Implementation.NebulaV2.WasmStatementBytes
open Nightstream.Implementation.NebulaV2.WasmStatementParser

/-- info: 'Nightstream.Implementation.NebulaV2.SeedSchedule.exact_fixed_role_geometry' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms exact_fixed_role_geometry

/-- info: 'Nightstream.Implementation.NebulaV2.SeedSchedule.Manifest.different_roles_have_different_seeds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Manifest.different_roles_have_different_seeds

/-- info: 'Nightstream.Implementation.NebulaV2.FieldCodec.CallSite.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms CallSite.sound

/-- info: 'Nightstream.Implementation.NebulaV2.BoundedWordRows.value_lt_twoPower' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.BoundedWordRows.value_lt_twoPower

/-- info: 'Nightstream.Implementation.NebulaV2.BoundedWordRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.BoundedWordRows.rows_complete

/-- info: 'Nightstream.Implementation.NebulaV2.LessThanConstantRows.value_lt_limit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.LessThanConstantRows.value_lt_limit

/-- info: 'Nightstream.Implementation.NebulaV2.LessThanConstantRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.LessThanConstantRows.rows_complete

/-- info: 'Nightstream.Implementation.NebulaV2.ConditionalEqualityRows.rows_sound_closed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ConditionalEqualityRows.rows_sound_closed

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimCounterRows.claim_canonical_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryClaimCounterRows.claim_canonical_of_rows

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimCounterRows.CallSite.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryClaimCounterRows.CallSite.sound

/-- info: 'Nightstream.Implementation.NebulaV2.FieldCodec.local_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms local_complete

/-- info: 'Nightstream.Implementation.NebulaV2.SnapshotRows.accepts_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms accepts_sound

/-- info: 'Nightstream.Implementation.NebulaV2.SnapshotRows.accepts_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepts_complete

/-- info: 'Nightstream.Implementation.NebulaV2.WasmStateCodec.encode_injective_on_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms encode_injective_on_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.WasmStateCodec.encode_exact_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms encode_exact_length

/-- info: 'Nightstream.Implementation.NebulaV2.WasmResultCodec.encodeDigest_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms encodeDigest_injective

/-- info: 'Nightstream.Implementation.NebulaV2.WasmResultCodec.encode_injective_of_decodes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.WasmResultCodec.encode_injective_of_decodes

/-- info: 'Nightstream.Implementation.NebulaV2.WasmPublicStatementCodec.encode_injective_of_decodes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.WasmPublicStatementCodec.encode_injective_of_decodes

/-- info: 'Nightstream.Implementation.NebulaV2.WasmPublicStatementCodec.encode_injective_of_decodesFor' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.WasmPublicStatementCodec.encode_injective_of_decodesFor

/-- info: 'Nightstream.Implementation.NebulaV2.WasmStatementBytes.join_split' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.WasmStatementBytes.join_split

/-- info: 'Nightstream.Implementation.NebulaV2.WasmStatementBytes.encode_injective_of_decodes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.WasmStatementBytes.encode_injective_of_decodes

/-- info: 'Nightstream.Implementation.NebulaV2.WasmStatementParser.parse_encode' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.WasmStatementParser.parse_encode

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimCodec.schema_width_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryClaimCodec.schema_width_exact

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimCodec.encode_injective_on_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryClaimCodec.encode_injective_on_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.CommitmentBundleCodec.schema_width_exact' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.CommitmentBundleCodec.schema_width_exact

/-- info: 'Nightstream.Implementation.NebulaV2.CommitmentBundleCodec.encode_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.CommitmentBundleCodec.encode_injective

/-- info: 'Nightstream.Implementation.NebulaV2.BundleForwardingRows.exact_bundle_forwarding' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.BundleForwardingRows.exact_bundle_forwarding

/-- info: 'Nightstream.Implementation.NebulaV2.BundleForwardingRows.CallSite.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.BundleForwardingRows.CallSite.sound

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryCodec.schema_width_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryCodec.schema_width_exact

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryCodec.encode_injective_on_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryCodec.encode_injective_on_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryRows.value_canonical_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryRows.value_canonical_of_rows

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryRows.CallSite.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryRows.CallSite.sound

/-- info: 'Nightstream.Implementation.NebulaV2.CanonicalFieldSchemaRows.all_slots_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.CanonicalFieldSchemaRows.all_slots_sound

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimFieldRows.typed_columns_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryClaimFieldRows.typed_columns_of_rows

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimFieldRows.modulus_alias_impossible' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryClaimFieldRows.modulus_alias_impossible

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryFieldRows.typed_columns_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryFieldRows.typed_columns_of_rows

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryFieldRows.modulus_alias_impossible' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryFieldRows.modulus_alias_impossible

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimParser.parse_native_parses' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryClaimParser.parse_native_parses

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimParser.parse_claim_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryClaimParser.parse_claim_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimParser.parse_blockOfClaim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryClaimParser.parse_blockOfClaim

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimParser.rejects_modulus_alias' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryClaimParser.rejects_modulus_alias

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryParser.parse_native_parses' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryParser.parse_native_parses

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryParser.parse_value_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryParser.parse_value_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryParser.parse_blockOfValue' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryParser.parse_blockOfValue

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryParser.rejects_modulus_alias' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryParser.rejects_modulus_alias

/-- info: 'Nightstream.Implementation.NebulaV2.LessThanConstantLinkedRows.value_lt_limit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.LessThanConstantLinkedRows.value_lt_limit

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimRows.parsed_columns_match' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryClaimRows.parsed_columns_match

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductBalanceRows.concreteBalanced_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryProductBalanceRows.concreteBalanced_of_rows

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductBalanceRows.parsed_claim_balanced_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryProductBalanceRows.parsed_claim_balanced_of_rows

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows.parsed_columns_match' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows.parsed_columns_match

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows.rows_force_parse' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows.rows_force_parse

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows.rows_force_parsed_columns_match' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows.rows_force_parsed_columns_match

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperPriorStateAuthorityRowsFor.rows_imply_exact_prior_state_and_fullMatches' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperPriorStateAuthorityRowsFor.rows_imply_exact_prior_state_and_fullMatches

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryHashFrame.frame_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryHashFrame.frame_injective

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryHashBinding.parsed_value_eq_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryHashBinding.parsed_value_eq_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryHashPackingRows.packed_columns_eq_encodePacked' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryHashPackingRows.packed_columns_eq_encodePacked

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryHashPackingRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryHashPackingRows.rows_complete

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryHashFrameRows.input_column_values' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryHashFrameRows.input_column_values

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonRows.output_columns_eq_carryDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonRows.output_columns_eq_carryDigest

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonRows.rows_complete

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonBinding.parsed_value_eq_or_poseidon_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonBinding.parsed_value_eq_or_poseidon_collision

/-- info: 'Nightstream.Implementation.NebulaV2.StateOutputFrameRows.input_column_values' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.StateOutputFrameRows.input_column_values

/-- info: 'Nightstream.Implementation.NebulaV2.StateOutputFrameRows.canonical_shape_eq_v2_iff' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.StateOutputFrameRows.canonical_shape_eq_v2_iff

/-- info: 'Nightstream.Implementation.NebulaV2.StateOutputPoseidonRows.output_columns_eq_pureDigest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.StateOutputPoseidonRows.output_columns_eq_pureDigest

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryCarryStateOutputRows.output_columns_eq_stateDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryCarryStateOutputRows.output_columns_eq_stateDigest

/-- info: 'Nightstream.Implementation.NebulaV2.StateOutputPoseidonBinding.satisfying_rows_bind_authority_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.StateOutputPoseidonBinding.satisfying_rows_bind_authority_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.StateOutputRowCensus.composed_rows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.StateOutputRowCensus.composed_rows_length

/-- info: 'Nightstream.Implementation.NebulaV2.U64HalvesRows.u64Halves_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.U64HalvesRows.u64Halves_injective

/-- info: 'Nightstream.Implementation.NebulaV2.StateOutputAuthorityRows.payloadFields_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.StateOutputAuthorityRows.payloadFields_injective

/-- info: 'Nightstream.Implementation.NebulaV2.StateOutputAuthorityRows.payload_column_values' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.StateOutputAuthorityRows.payload_column_values

/-- info: 'Nightstream.Implementation.NebulaV2.AuthoritativeStateOutputRows.output_columns_eq_typed_stateDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.AuthoritativeStateOutputRows.output_columns_eq_typed_stateDigest

/-- info: 'Nightstream.Implementation.NebulaV2.AuthoritativeStateOutputRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.AuthoritativeStateOutputRows.rows_complete

/-- info: 'Nightstream.Implementation.NebulaV2.AuthoritativeStateOutputBinding.satisfying_rows_bind_typed_authority_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.AuthoritativeStateOutputBinding.satisfying_rows_bind_typed_authority_or_collision

/-- info: 'Nightstream.Implementation.NebulaV2.UnsignedAdditionRows.output_eq_add' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.UnsignedAdditionRows.output_eq_add

/-- info: 'Nightstream.Implementation.NebulaV2.UnsignedLessOrEqualRows.left_le_right' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.UnsignedLessOrEqualRows.left_le_right

/-- info: 'Nightstream.Implementation.NebulaV2.ConditionalCarriedEqualityRows.rows_sound_closed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ConditionalCarriedEqualityRows.rows_sound_closed

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryTransitionSound.core_evidence' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryTransitionSound.core_evidence

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryTransitionSound.consumes_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.MemoryTransitionSound.consumes_of_rows
