import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics
import tests.Axioms.Support

/-! Fail-closed dependency checks for the active output envelope. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics.sourceFieldCount_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics.sourceFieldCount_eq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics.envelopePrefix_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics.envelopePrefix_eq

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics.envelope_length_of_compression' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics.envelope_length_of_compression

/-- info: 'Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics.active_ne_diagnostic_prefix' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics.active_ne_diagnostic_prefix
