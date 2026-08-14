import Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline
import tests.Axioms.Support

/-! Dependency audit for the direct exact-production F-prime release path. -/

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.baseChallengeAuthorityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.baseChallengeAuthorityExact

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.exactClaimSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.exactClaimSchedule

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.fixedBranchSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.fixedBranchSchedule

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.consumerInvocationIndicesExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.consumerInvocationIndicesExact

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.fullStateContinuityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.fullStateContinuityExact

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.receiptsExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.receiptsExact

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.exactClaimOpenings' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.exactClaimOpenings

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.everyConsumedBundleOpens' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.everyConsumedBundleOpens

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.everyConsumedClaimHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.everyConsumedClaimHolds

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.everyConsumedClaimSelectsGeneratedCore' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.Certificate.everyConsumedClaimSelectsGeneratedCore

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.GeneratedCertificate.recursivePayloadCanonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.GeneratedCertificate.recursivePayloadCanonical

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.GeneratedCertificate.rowsAndCarrierFit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.GeneratedCertificate.rowsAndCarrierFit

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.GeneratedCertificate.terminalRowsAndCarrierFit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.GeneratedCertificate.terminalRowsAndCarrierFit

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.GeneratedCertificate.verifierKeyIdentityExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.GeneratedCertificate.verifierKeyIdentityExact

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_certificate_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_certificate_or_failure

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_execution_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_execution_or_failure

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_any_bad_or_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_any_bad_or_execution

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_and_no_bad_implies_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_and_no_bad_implies_execution

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_wasm_result_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.acceptance_under_staged_refinement_implies_wasm_result_or_failure

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_certificate_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_certificate_or_failure

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_any_bad_or_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_any_bad_or_execution

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_and_no_bad_implies_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_and_no_bad_implies_execution

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_wasm_result_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleasePipeline.generated_acceptance_under_staged_refinement_implies_wasm_result_or_failure
