import Nightstream.Implementation
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the independently stated
production-shaped `Pi_CCS` transcript schedule.

Owns: dependency guards for transcript primitives, authority-prefix ordering,
verifier-derived challenge partitioning, FE/NC response cardinality, catch-up
handoff, and deterministic digest uniqueness.

Does not own: paper-protocol soundness, authority of the incoming state and
binding values, native/gadget/R1CS refinement, cost totals, or row removal.

| Protocol | Phase | Guarded mathematical obligation | Emits constraints? |
|---|---|---|---|
| `Pi_CCS` | primitives | squeeze cardinality and extension round-trip | no |
| `Pi_CCS` | binding | final checked-parent payload follows all prior bindings | no |
| `Pi_CCS` | challenges | verifier-owned bundle partition and state threading | no |
| `Pi_CCS` | FE/NC | one derived response per shaped round | no |
| `Pi_CCS` | catch-up | joint state/digest derivation and digest uniqueness | no |
-/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.squeezeBlocks_fields_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.squeezeBlocks_fields_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.pairFields_extensionFields' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives.pairFields_extensionFields

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

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runFe_shape' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runFe_shape

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc_shape' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck.runNc_shape

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
