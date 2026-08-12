import Nightstream.Implementation.NebulaV2.FPrime.Claim.NifsCall
import Nightstream.Implementation.NebulaV2.FPrime.Claim.GlobalFPrime
import Nightstream.Implementation.NebulaV2.FPrime.Manifest.RecursiveNifsCall
import tests.Axioms.Support

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimEnvelope.section_width_sum' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimEnvelope.section_width_sum

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimEnvelope.Value.encode_slice' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimEnvelope.Value.encode_slice

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimEnvelope.Value.encode_get_section' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimEnvelope.Value.encode_get_section

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimEnvelope.Value.encode_injective_on_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimEnvelope.Value.encode_injective_on_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimEnvelopeRows.input_eq_block' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimEnvelopeRows.input_eq_block

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimEnvelopeRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimEnvelopeRows.rows_complete

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimEnvelopeRows.CallSite.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimEnvelopeRows.CallSite.sound

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt.transition_accepts_and_consumes_same_full_claim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt.transition_accepts_and_consumes_same_full_claim

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimNifsCall.CircuitCall.input_is_exact_full_claim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimNifsCall.CircuitCall.input_is_exact_full_claim

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimNifsCall.satisfying_call_and_transition_bind_exact_claim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimNifsCall.satisfying_call_and_transition_bind_exact_claim

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimGlobalFPrime.Chain.exactClaimCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimGlobalFPrime.Chain.exactClaimCount

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimGlobalFPrime.Chain.completeDelayedSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimGlobalFPrime.Chain.completeDelayedSchedule

/-- info: 'Nightstream.Implementation.NebulaV2.FullClaimGlobalFPrime.Chain.terminalConsumesExactTrailingReceipt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FullClaimGlobalFPrime.Chain.terminalConsumesExactTrailingReceipt

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.rowRanges_exact_cover' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.rowRanges_exact_cover

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.target_has_unique_window' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.target_has_unique_window

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.fullClaimCallSite' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.fullClaimCallSite

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.selected_identity_is_exact_v2' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.selected_identity_is_exact_v2

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.satisfying_manifest_binds_exact_nifs_input' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.satisfying_manifest_binds_exact_nifs_input

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.exactPaperInputMatchesRunningRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.exactPaperInputMatchesRunningRows

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.memoryBlockPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.memoryBlockPlaced

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.memoryClaimColumnsMatch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.memoryClaimColumnsMatch

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.memoryBalance_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.memoryBalance_satisfied

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.closingProductsBalanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.closingProductsBalanced

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.bundleInputPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.bundleInputPlaced

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.forwardedBundleExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.forwardedBundleExact

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.exactMemoryTransition_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.exactMemoryTransition_satisfied

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.priorCarryColumnsMatch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.priorCarryColumnsMatch

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.outgoingCarryColumnsMatch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.outgoingCarryColumnsMatch

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.consumesExactAcceptedMemoryClaim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.consumesExactAcceptedMemoryClaim

/-- info: 'Nightstream.Implementation.NebulaV2.PriorStateLinkRows.claimCcsPublicExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.PriorStateLinkRows.claimCcsPublicExact

/-- info: 'Nightstream.Implementation.NebulaV2.PriorStateLinkRows.CcsPublicExact.ccsPublic_eq_ccsPublicWord' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.PriorStateLinkRows.CcsPublicExact.ccsPublic_eq_ccsPublicWord

/-- info: 'Nightstream.Implementation.NebulaV2.PriorStateLinkRows.outputDigest_eq_typedPriorState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.PriorStateLinkRows.outputDigest_eq_typedPriorState

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.priorStateLink_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.priorStateLink_satisfied

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.knownRows_lower_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.knownRows_lower_bound

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.rows_above_known_minimum_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.rows_above_known_minimum_bound

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.priorStateCcsPublicExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.priorStateCcsPublicExact

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.priorStateDigestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.priorStateDigestExact

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.priorStateLinkedAndConsumesExactAcceptedMemoryClaim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall.Call.priorStateLinkedAndConsumesExactAcceptedMemoryClaim
