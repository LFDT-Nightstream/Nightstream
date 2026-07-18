import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.CarrierCodec
import tests.Axioms.Support

/-! Fail-closed dependency gate for the reduced accumulator field codec. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.encodePoint_eq_flatten_map' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.encodePoint_eq_flatten_map

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.encodeCommitment_decodeCommitmentOfLength' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.encodeCommitment_decodeCommitmentOfLength

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.encodeChildren_decodeCommitmentOfLength' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.encodeChildren_decodeCommitmentOfLength

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.encodeCommitmentFamily_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.encodeCommitmentFamily_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.encodeCanonicalParent_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.encodeCanonicalParent_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.commitmentFamilyScheme_no_encodingCollision' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.commitmentFamilyScheme_no_encodingCollision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.canonicalParentScheme_no_encodingCollision' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.canonicalParentScheme_no_encodingCollision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.commitmentFamily_claim_eq_or_hashCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.commitmentFamily_claim_eq_or_hashCollision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.canonicalParent_claim_eq_or_hashCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.canonicalParent_claim_eq_or_hashCollision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.fixed_commitment_family_field_count' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.fixed_commitment_family_field_count

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.fixed_canonical_parent_field_count' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.CarrierCodec.fixed_canonical_parent_field_count
