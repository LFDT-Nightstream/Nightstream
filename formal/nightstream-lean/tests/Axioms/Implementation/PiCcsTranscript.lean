import Nightstream.Implementation
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.FeRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.NcRefinement
import tests.Axioms.Implementation.PiCcsTranscriptBlockLaneNcRefinement
import tests.Axioms.Implementation.PiCcsTranscriptCoins
import tests.Axioms.Implementation.PiCcsTranscriptExactCompleteSchedule
import tests.Axioms.Implementation.PiCcsTranscriptExactHonestProver
import tests.Axioms.Implementation.PiCcsTranscriptPostNcBoundary
import tests.Axioms.Implementation.PiCcsTranscriptTerminalFeRows
import tests.Axioms.Implementation.PiCcsTranscriptTerminalFeSchedule
import tests.Axioms.Implementation.PiCcsTranscriptTerminalFeWireFormat
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcCarrier
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcFinalRoundArtifact
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcFinalState
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcFirstRoundArtifact
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcFirstRoundConnectivity
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcFirstRoundExecution
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcFirstRoundSource
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcLaterRoundArtifact
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcLaterRoundConnectivity
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcLaterRoundExecution
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcLaterRoundReplay
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcLaterRoundSource
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcPrologueArtifact
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcPrologueExecution
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcReplay
import tests.Axioms.Implementation.PiCcsTranscriptTerminalNcSchedule
import tests.Axioms.Implementation.PiCcsTranscriptTerminalRoundExecution
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the independently stated
minimal mixed-width `Pi_CCS` candidate transcript schedule and the separate
legacy artifact refinements.

Owns: dependency guards for transcript primitives, authority-prefix ordering,
verifier-derived challenge partitioning, FE/NC response cardinality, catch-up
handoff, and deterministic digest uniqueness.

Does not own: paper-protocol soundness, authority of the incoming state and
binding values, native/gadget/R1CS refinement, cost totals, or row removal.

| Protocol | Phase | Guarded mathematical obligation | Emits constraints? |
|---|---|---|---|
| `Pi_CCS` | primitives | squeeze cardinality and extension round-trip | no |
| `Pi_CCS` | primitives | two-field challenge decoding has no default branch | no |
| `Pi_CCS` | raw absorption | lazy/eager states compose through normalization and later observers | no |
| `Pi_CCS` | binding | final checked-parent payload follows all prior bindings | no |
| `Pi_CCS` | challenges | verifier-owned bundle partition and state threading | no |
| `Pi_CCS` | FE/NC | one derived response per shaped round | no |
| `Pi_CCS` | FE/NC control flow | every nonempty round sequence computes cursor zero | no |
| `Pi_CCS` | exact messages | exact physical width/count and lossless round trips | no |
| `Pi_CCS` | exact carrier | exact mixed-width phase language, derived FE total, and lossless whole-carrier codec | no |
| `Pi_CCS` | exact schedule | one FE execution threads its successor directly into one NC execution | no |
| `Pi_CCS` | exact refinement | exact carrier serialization and typed FE/NC derives equal the joint schedule | no |
| `Pi_CCS` | FE refinement | exact mixed-width serialization and semantic/concrete replay equality | no |
| `Pi_CCS` | NC refinement | exact five-slot serialization and semantic/concrete replay equality | no |
| `Pi_CCS` | catch-up | joint state/digest derivation and digest uniqueness | no |
-/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.squeezeBlocks_fields_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.squeezeBlocks_fields_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.pairFields_extensionFields' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.pairFields_extensionFields

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.extensionFields_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.extensionFields_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.squeezeN_two_exact' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.squeezeN_two_exact

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.squeezeN_two_absorbed_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.squeezeN_two_absorbed_zero

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Transport.toK_toExtension' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Transport.toK_toExtension

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.catchup_eq_digest' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.catchup_eq_digest

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.absorbElem_normalizeFull' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.absorbElem_normalizeFull

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.constant_normalizes_to_native' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.constant_normalizes_to_native

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.variable_eq_native' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.variable_eq_native

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.appendRaw_eq_of_normalizedEq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.appendRaw_eq_of_normalizedEq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.constantAppend_normalizedEq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.constantAppend_normalizedEq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.variableAppend_eq_of_normalizedEq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.variableAppend_eq_of_normalizedEq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.digest_eq_of_normalizedEq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.digest_eq_of_normalizedEq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.constant_then_append_eq_native' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.constant_then_append_eq_native

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.constant_then_digest_eq_native' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.RawAbsorption.constant_then_digest_eq_native

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Binding.trace_afterParentHandle' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Binding.trace_afterParentHandle

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Binding.run_eq_appendParentHandle' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Binding.run_eq_appendParentHandle

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_state' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_state

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_alpha' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_alpha

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_gamma' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Challenges.run_gamma

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runRounds_challengeCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runRounds_challengeCount

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runRounds_append' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runRounds_append

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runRounds_cons_absorbed_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runRounds_cons_absorbed_zero

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runFe_shape' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runFe_shape

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc_shape' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc_shape

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.concreteLaneRound_fields_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.concreteLaneRound_fields_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.concreteReplay_eq_row_then_lane' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.concreteReplay_eq_row_then_lane

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.replay_refines_runFe' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.replay_refines_runFe

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.derive_refines_runFe' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.FeRefinement.derive_refines_runFe

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.toConcreteRound_fields_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.toConcreteRound_fields_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.runRound_refines' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.runRound_refines

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.derive_refines_runNc' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.NcRefinement.derive_refines_runNc

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.run_afterBinding' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.run_afterBinding

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.run_afterFe' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.run_afterFe

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.run_afterNc' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.run_afterNc

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.run_catchup_joint' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.run_catchup_joint

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.headerDigest_unique' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.headerDigest_unique

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.feChallengeCount' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.feChallengeCount

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.ncChallengeCount' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.ncChallengeCount

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.replay_deterministic' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.replay_deterministic

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Rows.ownerPieces_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Rows.ownerPieces_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Schedule.familyLengths' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Schedule.familyLengths

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Schedule.phaseTree_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Schedule.phaseTree_eq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.DigestRounds.scheduledCallsAccepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.DigestRounds.scheduledCallsAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PinSchedule.Artifact.pinTree_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PinSchedule.Artifact.pinTree_eq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PinSchedule.facts' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PinSchedule.facts

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages.decodeFixed_encodeFixed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages.decodeFixed_encodeFixed

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages.encodeFixed_of_decodeFixed' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages.encodeFixed_of_decodeFixed

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages.decodeExact_encode' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages.decodeExact_encode

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages.encodeExact_of_decodeExact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.ExactMessages.encodeExact_of_decodeExact

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.decode_isSome_iff_exactLanguage' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.decode_isSome_iff_exactLanguage

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.feRounds_length_of_exactLanguage' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.feRounds_length_of_exactLanguage

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.decode_encode' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.decode_encode

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.encode_of_decode' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.encode_of_decode

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.ExactRoundProjection.ofFn_toFunction' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.ExactRoundProjection.ofFn_toFunction

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule.rawMessages_exactLanguage' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule.rawMessages_exactLanguage

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule.run_afterNc_uses_afterFe' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule.run_afterNc_uses_afterFe

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule.challengeCounts' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule.challengeCounts

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule.replay_deterministic' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Schedule.replay_deterministic

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Refinement.encode_feRounds_eq_concreteRounds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Refinement.encode_feRounds_eq_concreteRounds

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Refinement.encode_ncRounds_eq_concreteRounds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Refinement.encode_ncRounds_eq_concreteRounds

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Refinement.feDerive_refines_run' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Refinement.feDerive_refines_run

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Refinement.ncDerive_refines_run' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.Refinement.ncDerive_refines_run
