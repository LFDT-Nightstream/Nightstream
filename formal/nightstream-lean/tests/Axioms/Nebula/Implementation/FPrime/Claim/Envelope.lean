import Nightstream.Implementation.Nebula.FPrime.Claim.NifsCall
import Nightstream.Implementation.Nebula.FPrime.Claim.GlobalFPrime
import Nightstream.Implementation.Nebula.FPrime.Manifest.RecursiveNifsCall
import tests.Axioms.Support

/-- info: 'Nightstream.Implementation.Nebula.FullClaimEnvelope.section_width_sum' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimEnvelope.section_width_sum

/-- info: 'Nightstream.Implementation.Nebula.FullClaimEnvelope.Value.encode_slice' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimEnvelope.Value.encode_slice

/-- info: 'Nightstream.Implementation.Nebula.FullClaimEnvelope.Value.encode_get_section' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimEnvelope.Value.encode_get_section

/-- info: 'Nightstream.Implementation.Nebula.FullClaimEnvelope.Value.encode_injective_on_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimEnvelope.Value.encode_injective_on_canonical

/-- info: 'Nightstream.Implementation.Nebula.FullClaimEnvelopeRows.input_eq_block' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimEnvelopeRows.input_eq_block

/-- info: 'Nightstream.Implementation.Nebula.FullClaimEnvelopeRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimEnvelopeRows.rows_complete

/-- info: 'Nightstream.Implementation.Nebula.FullClaimEnvelopeRows.CallSite.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimEnvelopeRows.CallSite.sound

/-- info: 'Nightstream.Implementation.Nebula.FullClaimNifsReceipt.transition_accepts_and_consumes_same_full_claim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimNifsReceipt.transition_accepts_and_consumes_same_full_claim

/-- info: 'Nightstream.Implementation.Nebula.FullClaimNifsCall.CircuitCall.input_is_exact_full_claim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimNifsCall.CircuitCall.input_is_exact_full_claim

/-- info: 'Nightstream.Implementation.Nebula.FullClaimNifsCall.satisfying_call_and_transition_bind_exact_claim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimNifsCall.satisfying_call_and_transition_bind_exact_claim

/-- info: 'Nightstream.Implementation.Nebula.FullClaimGlobalFPrime.Chain.exactClaimCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimGlobalFPrime.Chain.exactClaimCount

/-- info: 'Nightstream.Implementation.Nebula.FullClaimGlobalFPrime.Chain.completeDelayedSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimGlobalFPrime.Chain.completeDelayedSchedule

/-- info: 'Nightstream.Implementation.Nebula.FullClaimGlobalFPrime.Chain.terminalConsumesExactTrailingReceipt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FullClaimGlobalFPrime.Chain.terminalConsumesExactTrailingReceipt

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.rowRanges_exact_cover' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.rowRanges_exact_cover

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.target_has_unique_window' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.target_has_unique_window

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.fullClaimCallSite' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.fullClaimCallSite

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.selected_identity_is_exact_v2' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.selected_identity_is_exact_v2

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.satisfying_manifest_binds_exact_nifs_input' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.satisfying_manifest_binds_exact_nifs_input

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.exactPaperInputMatchesRunningRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.exactPaperInputMatchesRunningRows

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.memoryBlockPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.memoryBlockPlaced

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.memoryClaimColumnsMatch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.memoryClaimColumnsMatch

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.memoryBalance_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.memoryBalance_satisfied

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.closingProductsBalanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.closingProductsBalanced

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.bundleInputPlaced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.bundleInputPlaced

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.forwardedBundleExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.forwardedBundleExact

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.exactMemoryTransition_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.exactMemoryTransition_satisfied

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.priorCarryColumnsMatch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.priorCarryColumnsMatch

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.outgoingCarryColumnsMatch' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.outgoingCarryColumnsMatch

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.consumesExactAcceptedMemoryClaim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.consumesExactAcceptedMemoryClaim

/-- info: 'Nightstream.Implementation.Nebula.PriorStateLinkRows.claimCcsPublicExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.PriorStateLinkRows.claimCcsPublicExact

/-- info: 'Nightstream.Implementation.Nebula.PriorStateLinkRows.CcsPublicExact.ccsPublic_eq_ccsPublicWord' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.PriorStateLinkRows.CcsPublicExact.ccsPublic_eq_ccsPublicWord

/-- info: 'Nightstream.Implementation.Nebula.PriorStateLinkRows.outputDigest_eq_typedPriorState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.PriorStateLinkRows.outputDigest_eq_typedPriorState

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.priorStateLink_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.priorStateLink_satisfied

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.knownRows_lower_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.knownRows_lower_bound

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.rows_above_known_minimum_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestSchema.Artifact.rows_above_known_minimum_bound

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.priorStateCcsPublicExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.priorStateCcsPublicExact

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.priorStateDigestExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.priorStateDigestExact

/-- info: 'Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.priorStateLinkedAndConsumesExactAcceptedMemoryClaim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.RecursiveManifestNifsCall.Call.priorStateLinkedAndConsumesExactAcceptedMemoryClaim
