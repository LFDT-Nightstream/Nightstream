import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConcreteCompleteness
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityArtifact
import tests.Axioms.Support

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCodec
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteAlgebra
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteComposition
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteCompleteness
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteExtraction
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteNifs
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteAlgebra.evaluations_eq_paper' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms evaluations_eq_paper

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteCompleteness.logicalSourceHolds_iff_sourceValid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms logicalSourceHolds_iff_sourceValid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteCompleteness.logicalSource_exists_verifiedTransition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms logicalSource_exists_verifiedTransition

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteComposition.existsFiniteReductionThroughPiDec' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms existsFiniteReductionThroughPiDec

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteExtraction.extractionStrongSetUnits' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms extractionStrongSetUnits

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2.piRlcResponse_refines_of_no_shortfall' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms piRlcResponse_refines_of_no_shortfall

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteNifs.boundedSampler_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms boundedSampler_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity.completeSamplerShortfallBound_le_target' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms completeSamplerShortfallBound_le_target

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity.idealCandidateTriple_joint_probability' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms idealCandidateTriple_joint_probability

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity.idealThreeRejections_probability' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms idealThreeRejections_probability

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity.samplerShortfall_probability_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms samplerShortfall_probability_le

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity.samplerShortfall_probability_le_182_bits' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms samplerShortfall_probability_le_182_bits

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCodec.publicWireFields_injective_on_admissible' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms publicWireFields_injective_on_admissible

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCodec.proofWireFields_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms proofWireFields_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteNifs.concreteFullOracleSoundness' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms concreteFullOracleSoundness

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityArtifact.FixedSnapshotMatrixRefinement.matrices_eq_compiled' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FixedSnapshotMatrixRefinement.matrices_eq_compiled

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityArtifact.FixedSnapshotMatrixRefinement.decoder_accepts' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FixedSnapshotMatrixRefinement.decoder_accepts
