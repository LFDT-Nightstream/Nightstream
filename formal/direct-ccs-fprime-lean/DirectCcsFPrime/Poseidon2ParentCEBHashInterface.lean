import DirectCcsFPrime.Poseidon2ParentCEBHash

/-!
Typed interface for the Poseidon2 parent `CE(B)` hash boundary.

Spec: `specs/Poseidon2ParentCEBHash.spec.md`
-/

namespace DirectCcsFPrime

namespace Poseidon2ParentCEBHashInterface

abbrev BindingAssumption :=
  @Poseidon2ParentCEBHash.BindingAssumption

abbrev Hash :=
  Poseidon2ParentCEBHash.Hash

abbrev toParentCEBHash :=
  @Poseidon2ParentCEBHash.toParentCEBHash

abbrev encodedParentCEBDigestBinding :=
  @Poseidon2ParentCEBHash.encodedParentCEBDigestBinding

abbrev digest :=
  @Poseidon2ParentCEBHash.digest

abbrev source :=
  @Poseidon2ParentCEBHash.source

abbrev same_parentCEB_of_digest_eq :=
  @Poseidon2ParentCEBHash.same_parentCEB_of_digest_eq

abbrev projected_residue_source_functional :=
  @Poseidon2ParentCEBHash.projected_residue_source_functional

abbrev terminal_soundness_of_poseidon2_parent_hash_prior_authority_sound_and_msis :=
  @Poseidon2ParentCEBHash.terminal_soundness_of_poseidon2_parent_hash_prior_authority_sound_and_msis

abbrev terminal_soundness_of_poseidon2_parent_hash_sound_verifier_and_msis :=
  @Poseidon2ParentCEBHash.terminal_soundness_of_poseidon2_parent_hash_sound_verifier_and_msis

end Poseidon2ParentCEBHashInterface

end DirectCcsFPrime
