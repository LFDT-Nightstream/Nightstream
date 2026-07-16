import Nightstream.Implementation
import tests.Axioms.Support

/-!
Fail-closed dependency guards for the fixed-carrier Π_CCS, NIFS, recursive F′,
and aligned-repair boundary.

| Guard family | What it audits |
| --- | --- |
| Public projection | The logical projection does not determine all packed coefficients |
| Fixed-width Π_CCS | The implementation accepts a carrier that violates paper NC |
| Fixed-carrier Π_CCS | The exact current carrier accepts that witness |
| Fixed NIFS | The exact current NIFS accepts that witness |
| Recursive F′ | The exact current recursive relation accepts the linked non-norm carrier |
| Aligned dimensions | The fixed carrier becomes paper-aligned after authoritative insertion |
| Aligned completeness | Honest legacy witnesses transport into the repaired relation and NIFS model |
| Aligned rejection | The exact linked bad carrier fails repaired norm, CCS, and CE membership |
| Aligned compiler map | Old columns map injectively around exactly thirteen fixed padding positions |
| Aligned assignment map | Every old scalar is preserved and every new padding scalar is zero |
| Aligned matrix map | Every row scalar comes from one old coefficient or fixed-zero padding |
| Aligned commitment shape | Packing and key widths agree, but equal dimensions do not authorize commitment reuse |
| PiDEC evaluation bridge | Independent typed evaluation recomposition equals the fixed public-carrier operation |
| Recursive Pi_RLC sampler | Bootstrap arity, shared challenge columns, machine decoding, and paper artifact |
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDecEvaluationBridge.semantic_evaluations_hom' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDecEvaluationBridge.semantic_evaluations_hom

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.distinguishedProjection_does_not_determine_packedInput' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.distinguishedProjection_does_not_determine_packedInput

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedWidth_pi_ccs_artifact_accepts_nc_false_carrier' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedWidth_pi_ccs_artifact_accepts_nc_false_carrier

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedCarrier_pi_ccs_artifact_accepts_nc_false_carrier' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedCarrier_pi_ccs_artifact_accepts_nc_false_carrier

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedCarrier_nifs_artifact_accepts_nc_false_carrier' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedCarrier_nifs_artifact_accepts_nc_false_carrier

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedCarrier_recursive_f_prime_artifact_accepts_non_norm_carrier' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedCarrier_recursive_f_prime_artifact_accepts_non_norm_carrier

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrier_paperDimensions' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrier_paperDimensions

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.alignedFreshStatement_holds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.alignedFreshStatement_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.alignedNifsTransition_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.alignedNifsTransition_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrierTail_not_normBounded' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrierTail_not_normBounded

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrierTail_not_alignedCCS' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrierTail_not_alignedCCS

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrierTail_not_alignedCE' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedNifsRepair.fixedCarrierTail_not_alignedCE

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.alignedIndex_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.alignedIndex_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.alignColumn_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.alignColumn_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.unalignIndex?_alignedIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.unalignIndex?_alignedIndex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.unalignIndex?_eq_none_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.unalignIndex?_eq_none_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.alignedIndex_of_unalignIndex?_eq_some' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.alignedIndex_of_unalignIndex?_eq_some

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.unalignedIndex_lt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.unalignedIndex_lt

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.packedFlatIndex' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.packedFlatIndex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.boundary_coordinates' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.boundary_coordinates

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.fixedCarrier_blockCounts' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap.fixedCarrier_blockCounts

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.getElem?_alignedIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.getElem?_alignedIndex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.getD_alignedIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.getD_alignedIndex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.getD_padding_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.getD_padding_zero

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.packedCoeff_alignedIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap.packedCoeff_alignedIndex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.compiledRowValue_old' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.compiledRowValue_old

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.compiledRowValue_padding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.compiledRowValue_padding

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.compiledRowValue_cases' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.compiledRowValue_cases

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.alignRow_getD_eq_compiledRowValue' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap.alignRow_getD_eq_compiledRowValue

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.setupColumns_eq_blockCount' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.setupColumns_eq_blockCount

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.packAssignment_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.packAssignment_length

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.alignedKeyShape_matches_packing' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.alignedKeyShape_matches_packing

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.setupColumns_exactRingWidth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.setupColumns_exactRingWidth

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.paperDimensions_setupColumns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.paperDimensions_setupColumns

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.fixedPublicSetupColumns' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.fixedPublicSetupColumns

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.sameShape_setupFacts' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.sameShape_setupFacts

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.sameShape_legacyKey_commitment_changes' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.sameShape_legacyKey_commitment_changes

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.sameShape_legacyKey_commitment_not_preserved' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.CommitmentShape.sameShape_legacyKey_commitment_not_preserved

/-! ## Recursive `Pi_RLC` sampler bridge -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.publicRoleIndex_census' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.publicRoleIndex_census

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.publicShared' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.publicShared

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.decodedRing_eq_machineRing' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.decodedRing_eq_machineRing

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.samplerArtifact_of_membership' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.samplerArtifact_of_membership
