import Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline
import tests.Axioms.Support

/-! Dependency audit for the direct exact-production F-prime release path. -/

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.baseChallengeAuthorityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.baseChallengeAuthorityExact

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.exactClaimSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.exactClaimSchedule

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.fixedBranchSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.fixedBranchSchedule

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.consumerInvocationIndicesExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.consumerInvocationIndicesExact

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.fullStateContinuityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.fullStateContinuityExact

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.receiptsExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.receiptsExact

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.exactClaimOpenings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.exactClaimOpenings

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.everyConsumedBundleOpens' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.everyConsumedBundleOpens

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.everyConsumedClaimHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.everyConsumedClaimHolds

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.everyConsumedClaimSelectsGeneratedCore' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.Certificate.everyConsumedClaimSelectsGeneratedCore

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.GeneratedCertificate.recursivePayloadCanonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.GeneratedCertificate.recursivePayloadCanonical

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.GeneratedCertificate.rowsAndCarrierFit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.GeneratedCertificate.rowsAndCarrierFit

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.GeneratedCertificate.terminalRowsAndCarrierFit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.GeneratedCertificate.terminalRowsAndCarrierFit

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.GeneratedCertificate.verifierKeyIdentityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.GeneratedCertificate.verifierKeyIdentityExact

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_certificate_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_certificate_or_failure

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_execution_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_execution_or_failure

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_any_bad_or_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_any_bad_or_execution

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_and_no_bad_implies_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_and_no_bad_implies_execution

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_wasm_result_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_wasm_result_or_failure

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_certificate_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_certificate_or_failure

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_any_bad_or_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_any_bad_or_execution

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_and_no_bad_implies_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_and_no_bad_implies_execution

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_wasm_result_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_wasm_result_or_failure
