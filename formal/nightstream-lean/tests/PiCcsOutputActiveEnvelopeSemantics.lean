import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Poseidon.ActiveEnvelopeSemantics

/-! Focused checks for the artifact-independent active output envelope. -/

open Nightstream.Implementation.R1CS.PiCcsOutputDigest.Poseidon

#check ActiveEnvelopeSemantics.sourceFieldCount_eq
#check ActiveEnvelopeSemantics.envelopePrefix_eq
#check ActiveEnvelopeSemantics.envelope_length_of_compression
#check ActiveEnvelopeSemantics.active_ne_diagnostic_prefix

#guard ActiveEnvelopeSemantics.sourceFieldCount = 23033
#guard ActiveEnvelopeSemantics.envelopePrefix.length = 10
#guard ActiveEnvelopeSemantics.envelopePrefix !=
  EnvelopeSemantics.diagnosticEnvelopePrefix

example (compressionOutput : List Nat)
    (length : compressionOutput.length = 54) :
    (ActiveEnvelopeSemantics.envelope compressionOutput).length = 64 :=
  ActiveEnvelopeSemantics.envelope_length_of_compression
    compressionOutput length
