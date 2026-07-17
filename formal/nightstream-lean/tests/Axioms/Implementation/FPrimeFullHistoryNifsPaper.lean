import Nightstream.Implementation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedNifsRepair
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.ColumnMap
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.AssignmentMap
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.MatrixMap
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.CommitmentShape
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
| PiDEC public-input bridge | Exact 270-coordinate transpose and typed recomposition refinement |
| PiRLC public-input bridge | Exact five-ring decoding and typed Phi81 combination refinement |
| PiRLC ring transport | Shared list-to-typed product sum beneath public identity branches |
| PiRLC commitment bridge | Exact eighteen-ring decoding and typed commitment combination refinement |
| PiRLC evaluation bridge | Exact six-limb decoding and typed `RingK` evaluation combination refinement |
| PiRLC point bridge | Checked point decoding and propagation without caller-owned dimension evidence |
| PiRLC equation profiles | Minimal equation wiring, separate parent binding, both sampler profiles, and conditional terminal handoff |
| Selective polynomial | Exact 13-port syntax, degree, padding specialization, and canonical padding completeness |
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDecEvaluationBridge.semantic_evaluations_hom' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDecEvaluationBridge.semantic_evaluations_hom

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.PublicInputBridge.packedSlot_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.PublicInputBridge.packedSlot_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.PublicInputBridge.decode_injective_of_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.PublicInputBridge.decode_injective_of_length

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.PublicInputBridge.strictAccepted_typedPublicInputEquation' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.PublicInputBridge.strictAccepted_typedPublicInputEquation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.decodeXRings_phi81Combine' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.decodeXRings_phi81Combine

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RingTransport.ringOfList_phi81Combine' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RingTransport.ringOfList_phi81Combine

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.CommitmentBridge.decodeCommitmentRings_phi81Combine' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.CommitmentBridge.decodeCommitmentRings_phi81Combine

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge.pairRingF_action' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge.pairRingF_action

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge.decodeYRingRings_phi81Combine' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge.decodeYRingRings_phi81Combine

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge.typedEvaluationEquation_of_refinement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.EvaluationBridge.typedEvaluationEquation_of_refinement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PointBridge.bound_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PointBridge.bound_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PointBridge.inputPointBound_of_outputPointBound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PointBridge.inputPointBound_of_outputPointBound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PointBridge.parentPointBound_of_outputPointBound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PointBridge.parentPointBound_of_outputPointBound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.typedPublicInputEquation_of_refinement' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.typedPublicInputEquation_of_refinement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.decode_assembledX' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.decode_assembledX

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.typedPiRlcToPiDecParentEquation_of_refinement' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.typedPiRlcToPiDecParentEquation_of_refinement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.typedPiRlcPiDecPublicInputComposition' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.typedPiRlcPiDecPublicInputComposition

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.RelabeledCarrier.decodedPackedInput_relabel' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.RelabeledCarrier.decodedPackedInput_relabel

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.typedPiRlcPiDecPublicInputComposition_relabel' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.typedPiRlcPiDecPublicInputComposition_relabel

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.equations_of_refinement' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.equations_of_refinement

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.output_eq_parent_of_artifacts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.output_eq_parent_of_artifacts

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.typedOutput_eq_parent_of_wiring' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge.typedOutput_eq_parent_of_wiring

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.CarrierCodec.canonical_artifact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.CarrierCodec.canonical_artifact

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

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.challengeWiringArtifact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveSamplerArtifact.challengeWiringArtifact

/-! ## Exact recursive and terminal `Pi_RLC` active carriers -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact.parentArtifact' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact.parentArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact.carrierArtifact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact.carrierArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact.challengeWiringArtifact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact.challengeWiringArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact.equationWiringArtifact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.RecursiveCarrierArtifact.equationWiringArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalCarrierArtifact.parentArtifact' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalCarrierArtifact.parentArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalCarrierArtifact.carrierArtifact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalCarrierArtifact.carrierArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalCarrierArtifact.challengeWiringArtifact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalCarrierArtifact.challengeWiringArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalCarrierArtifact.equationWiringArtifact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalCarrierArtifact.equationWiringArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact.publicShared' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact.publicShared

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact.decodedRing_eq_machineRing' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.TerminalSamplerArtifact.decodedRing_eq_machineRing

/-! ## Exact `Pi_RLC` reduction and typed composition -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction.exact_output_eq_phi81Combine' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction.exact_output_eq_phi81Combine

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction.Profiles.recursiveReduction_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction.Profiles.recursiveReduction_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction.Profiles.terminalReduction_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Reduction.Profiles.terminalReduction_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles.recursiveEquationRefinement_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles.recursiveEquationRefinement_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles.recursiveTypedComposition_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles.recursiveTypedComposition_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles.terminalEquationRefinement_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles.terminalEquationRefinement_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles.terminalTypedComposition_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles.terminalTypedComposition_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.accepted_refines_certificateAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.accepted_refines_certificateAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.withDecodedChallenges_challenge_eq_columns' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.withDecodedChallenges_challenge_eq_columns

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.accepted_refines_withDecodedChallenges' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.accepted_refines_withDecodedChallenges

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles.terminalSampler_refines_decodedBatchChallenges' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles.terminalSampler_refines_decodedBatchChallenges

/-! ## Typed selective carrier boundary -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.SelectiveLayout.exact_layout' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.SelectiveLayout.exact_layout

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Semantics.zeroPinHolds_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Semantics.zeroPinHolds_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Semantics.canonicalAssignment_complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Semantics.canonicalAssignment_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Necessity.eachRawPaddingCheck_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Necessity.eachRawPaddingCheck_necessary

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.term_count_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.term_count_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.canonicalEqualityGatedDegreeBound_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.canonicalEqualityGatedDegreeBound_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Refinement.evaluate_paddingPortPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Refinement.evaluate_paddingPortPoint

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Refinement.evaluate_paddingRowPoint_eq_zero_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Refinement.evaluate_paddingRowPoint_eq_zero_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Refinement.canonicalAssignment_sparse_complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Refinement.canonicalAssignment_sparse_complete
