import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.DerivedNegative
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LinearCompiler
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.Refinement.DerivedBorrow
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.Refinement.OrdinaryPrivateField
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LayoutManifest
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LayoutWidthFloor
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.SourceCensus
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.PackedSourceCensus
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.OrdinaryPlacement
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.InactiveNoninterference
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.Phi81ColumnLayoutRefinement
import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.OrdinaryPlacement
import tests.Axioms.Support

/-! Fail-closed axioms gate for ordinary private-field encoding. -/

/-- info: 'Nightstream.Implementation.R1CS.FreshAssignmentPacking.packAssignment_injective_of_length_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FreshAssignmentPacking.packAssignment_injective_of_length_eq

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryField.decode_encodeDigit' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryField.decode_encodeDigit

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryField.duplicate_words_decode_same' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryField.duplicate_words_decode_same

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryField.encodeChosenPrivate_decodeChosenPrivate' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryField.encodeChosenPrivate_decodeChosenPrivate

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryField.augmented_private_exists_iff_semantic_exists' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryField.augmented_private_exists_iff_semantic_exists

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged.accepts_iff_canonicalRows' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged.accepts_iff_canonicalRows

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative.derivedNegative_eq_indicator' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative.derivedNegative_eq_indicator

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative.conservative_iff_derived_and_materialized' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative.conservative_iff_derived_and_materialized

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative.materialized_accepts_iff_derived' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryDerivedNegative.materialized_accepts_iff_derived

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler.reemit_parsed_projection' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler.reemit_parsed_projection

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler.loweredRows_iff_sourceRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler.loweredRows_iff_sourceRows

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler.honest_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler.honest_complete

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.derivedBorrowEquation_holds_iff' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.derivedBorrowEquation_holds_iff

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.derivedBorrowEquation_degree_le_three' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.derivedBorrowEquation_degree_le_three

/-- info: 'Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.derivedAccepts_iff_polynomial_schedule' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CenteredTernaryDerivedBorrow.derivedAccepts_iff_polynomial_schedule

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.ExactPartition.existsUniqueOwner' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.ExactPartition.existsUniqueOwner

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.ExactPartition.distinctOwnersDisjoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.ExactPartition.distinctOwnersDisjoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.existsUniqueSlotForSource' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.existsUniqueSlotForSource

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.encodedCoordinateHasUniqueOwner' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.encodedCoordinateHasUniqueOwner

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.ceCoordinateHasUniqueOwner' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.ceCoordinateHasUniqueOwner

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.ordinaryOwnerFor' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.ordinaryOwnerFor

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.coordinateOnlyOwnerIsExcluded' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.coordinateOnlyOwnerIsExcluded

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.sourceZeroHasConstantOneOwner' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.sourceZeroHasConstantOneOwner

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.encodedZeroHasExcludedOwner' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.encodedZeroHasExcludedOwner

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.ceZeroHasExcludedOwner' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.Manifest.Valid.ceZeroHasExcludedOwner

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.eligibleSlots_share_committed_freshCe_assignment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.eligibleSlots_share_committed_freshCe_assignment

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.normBounded_word_can_decode_nonCentered_source' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.normBounded_word_can_decode_nonCentered_source

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.ExactPartition.totalRunLength_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.ExactPartition.totalRunLength_eq

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.encodedEligibleLength_le_total' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.encodedEligibleLength_le_total

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.encoded_width_floor' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.encoded_width_floor

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.sourceColumn_hasUniqueRole' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.sourceColumn_hasUniqueRole

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.sourceColumn_hasExactEligibilityClass' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.sourceColumn_hasExactEligibilityClass

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.eligibleCount_eq_ordinaryRunSubtotal' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.eligibleCount_eq_ordinaryRunSubtotal

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.declaredRoleTotal_eq_sourceColumnCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.declaredRoleTotal_eq_sourceColumnCount

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.sourceColumnCount_eq_eligibleCount_add_excludedCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.sourceColumnCount_eq_eligibleCount_add_excludedCount

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.budget_below_perField41_is_no_go' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.budget_below_perField41_is_no_go

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.PackedSourceCensus.Data.check_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.PackedSourceCensus.Data.check_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.PackedSourceCensus.Data.toSourceCensusArtifact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.PackedSourceCensus.Data.toSourceCensusArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement.segmentPlacementStart_some_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement.segmentPlacementStart_some_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement.sameSegment_wordRun_before' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement.sameSegment_wordRun_before

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement.Metadata.check_sound' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement.Metadata.check_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.recursive_data_check' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.recursive_data_check

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.recursive_lastPlacement' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.recursive_lastPlacement

/-- info: 'Nightstream.Implementation.R1CS.InactiveFieldNoninterference.selectorComposed_sound' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.InactiveFieldNoninterference.selectorComposed_sound

/-- info: 'Nightstream.Implementation.R1CS.InactiveFieldNoninterference.selectorComposed_complete' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.InactiveFieldNoninterference.selectorComposed_complete

/-- info: 'Nightstream.Implementation.R1CS.InactiveFieldNoninterference.selectorComposed_acceptance_iff' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.InactiveFieldNoninterference.selectorComposed_acceptance_iff

/-- info: 'Nightstream.Implementation.R1CS.InactiveFieldNoninterference.authorityOutput_invariant' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.InactiveFieldNoninterference.authorityOutput_invariant

/-- info: 'Nightstream.Implementation.R1CS.InactiveFieldNoninterference.inactiveNoninterference' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.InactiveFieldNoninterference.inactiveNoninterference

/-- info: 'Nightstream.Implementation.R1CS.OrdinaryPrivateFieldRefinement.materializeWord_represents' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.OrdinaryPrivateFieldRefinement.materializeWord_represents

/-- info: 'Nightstream.Implementation.R1CS.OrdinaryPrivateFieldRefinement.safeAccepts_iff' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.OrdinaryPrivateFieldRefinement.safeAccepts_iff

/-- info: 'Nightstream.Implementation.R1CS.OrdinaryPrivateFieldRefinement.normDischargedLowering_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.OrdinaryPrivateFieldRefinement.normDischargedLowering_sound

/-- info: 'Nightstream.Implementation.R1CS.OrdinaryPrivateFieldRefinement.freshCcsAuthority_privateNorm' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.OrdinaryPrivateFieldRefinement.freshCcsAuthority_privateNorm

/-- info: 'Nightstream.Implementation.R1CS.FreshAssignmentPacking.packAssignment_coordinate' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FreshAssignmentPacking.packAssignment_coordinate

/-! Phi81 logical-column packing refinement. -/

/-- info: 'Nightstream.Implementation.R1CS.Phi81ColumnLayoutRefinement.packAssignment_length_eq_blockCount' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81ColumnLayoutRefinement.packAssignment_length_eq_blockCount

/-- info: 'Nightstream.Implementation.R1CS.Phi81ColumnLayoutRefinement.packedCell_eq_layout' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81ColumnLayoutRefinement.packedCell_eq_layout

/-- info: 'Nightstream.Implementation.R1CS.Phi81ColumnLayoutRefinement.logicalColumn_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81ColumnLayoutRefinement.logicalColumn_exact

/-- info: 'Nightstream.Implementation.R1CS.Phi81ColumnLayoutRefinement.paddingCell_zero' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Phi81ColumnLayoutRefinement.paddingCell_zero
