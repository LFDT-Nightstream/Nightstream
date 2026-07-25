import tests.FPrimeFullHistoryProductionDigestCodec
import tests.Axioms.Support

/-!
Fail-closed guards for the exact recursive-output and terminal-link
refinement into the selected production digest codec.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.codec_values_eq_outputDigest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.codec_values_eq_outputDigest

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.codec_roundtrip' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.codec_roundtrip

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.decodedDigest_eq_logicalLinkDigest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.decodedDigest_eq_logicalLinkDigest

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.terminalLogicalPublic_eq_encodePublicInput' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.terminalLogicalPublic_eq_encodePublicInput

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.rows_decode_exact_xOut' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.rows_decode_exact_xOut

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.output_and_terminal_rows_decode_same_digest' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.output_and_terminal_rows_decode_same_digest

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.output_and_terminal_rows_decode_linked_digest' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProductionDigestCodec.output_and_terminal_rows_decode_linked_digest
