import DirectCcsFPrime.ParentCEBHashBinding

/-!
Interface for parent-CE(B) binding.

Spec: `specs/ParentCEBHashBinding.spec.md`
-/

namespace DirectCcsFPrime

namespace ParentCEBHashBindingInterface

abbrev ParentCEBHash :=
  ParentCEBHashBinding.ParentCEBHash

abbrev digest :=
  @ParentCEBHashBinding.digest

abbrev source :=
  @ParentCEBHashBinding.source

abbrev encodedParentCEBDigestBinding :=
  @ParentCEBHashBinding.encodedParentCEBDigestBinding

abbrev same_parentCEB_of_digest_eq :=
  @ParentCEBHashBinding.same_parentCEB_of_digest_eq

abbrev projected_residue_source_functional :=
  @ParentCEBHashBinding.projected_residue_source_functional

end ParentCEBHashBindingInterface

end DirectCcsFPrime
