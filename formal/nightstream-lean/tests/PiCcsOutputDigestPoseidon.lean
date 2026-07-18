import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Handoff

/-!
Kernel-facing surface for the typed terminal `Pi_CCS` output-digest envelope,
its exact generated Poseidon2 trace, and the conditional `Pi_RLC` handoff.
-/

namespace tests.PiCcsOutputDigestPoseidon

open Nightstream.Implementation.R1CS.PiCcsOutputDigest

#check Poseidon.EnvelopeSemantics.diagnosticEnvelope_length
#check Poseidon.Schedule.trace_valid
#check Poseidon.Refinement.accepted_digestEnvelope
#check Poseidon.Refinement.accepted_composedDigest
#check Handoff.accepted_digestFieldValues
#check Handoff.accepted_conditionalDigestHandoff

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.EnvelopeSemantics.diagnosticEnvelope_length' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Poseidon.EnvelopeSemantics.diagnosticEnvelope_length

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.Schedule.trace_valid' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Poseidon.Schedule.trace_valid

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.Refinement.accepted_digestEnvelope' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Poseidon.Refinement.accepted_digestEnvelope

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.Refinement.accepted_composedDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms Poseidon.Refinement.accepted_composedDigest

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Handoff.accepted_digestFieldValues' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms Handoff.accepted_digestFieldValues

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Handoff.accepted_conditionalDigestHandoff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms Handoff.accepted_conditionalDigestHandoff

end tests.PiCcsOutputDigestPoseidon
