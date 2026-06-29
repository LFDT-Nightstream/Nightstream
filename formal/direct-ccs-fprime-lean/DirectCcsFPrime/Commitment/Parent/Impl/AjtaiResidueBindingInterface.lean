import DirectCcsFPrime.Commitment.Parent.Impl.AjtaiResidueBinding

/-!
Typed interface for Ajtai-backed residue binding.

Spec: `specs/Commitment/Parent/Impl/AjtaiResidueBinding.spec.md`
-/

namespace DirectCcsFPrime

namespace AjtaiResidueBindingInterface

abbrev NoAjtaiBindingCollision :=
  AjtaiResidueBinding.NoAjtaiBindingCollision

abbrev AssignmentOpeningAdapter :=
  AjtaiResidueBinding.AssignmentOpeningAdapter

abbrev commitmentOfOpening :=
  @AjtaiResidueBinding.commitmentOfOpening

abbrev ajtaiCommitMap :=
  @AjtaiResidueBinding.ajtaiCommitMap

abbrev AjtaiBackedCommitMap :=
  AjtaiResidueBinding.AjtaiBackedCommitMap

abbrev opensTo_commitmentOfOpening :=
  @AjtaiResidueBinding.opensTo_commitmentOfOpening

abbrev assignmentOpeningAdapter_of_ajtaiBackedCommitMap :=
  @AjtaiResidueBinding.assignmentOpeningAdapter_of_ajtaiBackedCommitMap

abbrev ajtaiBackedCommitMap_of_ajtaiCommitMap :=
  @AjtaiResidueBinding.ajtaiBackedCommitMap_of_ajtaiCommitMap

abbrev CEOpeningAdapter :=
  AjtaiResidueBinding.CEOpeningAdapter

abbrev ceOpeningAdapter_of_assignmentOpeningAdapter :=
  @AjtaiResidueBinding.ceOpeningAdapter_of_assignmentOpeningAdapter

abbrev noAjtaiBindingCollision_of_advantageBound :=
  @AjtaiResidueBinding.noAjtaiBindingCollision_of_advantageBound

abbrev noAjtaiBindingCollision_of_ajtaiBindingAssumption :=
  @AjtaiResidueBinding.noAjtaiBindingCollision_of_ajtaiBindingAssumption

abbrev noAjtaiBindingCollision_of_msis :=
  @AjtaiResidueBinding.noAjtaiBindingCollision_of_msis

abbrev openingWitness_eq_of_noAjtaiBindingCollision :=
  @AjtaiResidueBinding.openingWitness_eq_of_noAjtaiBindingCollision

abbrev commitMapResiduesFunctional_of_noAjtaiBindingCollision :=
  @AjtaiResidueBinding.commitMapResiduesFunctional_of_noAjtaiBindingCollision

abbrev fixedCEOpeningResiduesFunctional_of_noAjtaiBindingCollision :=
  @AjtaiResidueBinding.fixedCEOpeningResiduesFunctional_of_noAjtaiBindingCollision

abbrev encodedParentCEBOpeningResiduesFunctionalFor_of_noAjtaiBindingCollision :=
  @AjtaiResidueBinding.encodedParentCEBOpeningResiduesFunctionalFor_of_noAjtaiBindingCollision

end AjtaiResidueBindingInterface

end DirectCcsFPrime
