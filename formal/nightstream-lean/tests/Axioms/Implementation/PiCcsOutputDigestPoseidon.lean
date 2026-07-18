import Nightstream.Implementation
import tests.Axioms.Support

/-!
Fail-closed kernel dependency expectations for the exact terminal `Pi_CCS`
output-digest envelope and its conditional handoff into `Pi_RLC`.

Owns: dependency guards for the independent envelope shape, exact generated
Poseidon2 trace, full typed-serialization/SIS/sponge composition, and the
downstream transcript handoff.

Does not own: dynamic output authority, SIS seed/native conformance, complete
`Pi_CCS` soundness, row necessity, row removal, or cost totals.

| Protocol | Phase | Guarded obligation | Emits constraints? |
|---|---|---|---|
| `Pi_CCS` | output digest | independent envelope and exact 17-round trace | no |
| `Pi_CCS` | output digest | typed serialization through both SIS maps and Poseidon2 | no |
| `Pi_RLC` | digest handoff | the recomputed four lanes enter the audited sampler state | no |
-/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.EnvelopeSemantics.envelope_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.EnvelopeSemantics.envelope_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.Schedule.trace_valid' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.Schedule.trace_valid

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.Refinement.accepted_composedDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.Refinement.accepted_composedDigest

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Handoff.accepted_conditionalDigestHandoff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Handoff.accepted_conditionalDigestHandoff
