import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5
import tests.Axioms.Support

/-! Fail-closed kernel dependency ownership for the packed Mod-5 leaf. -/

open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.packedRows_iff_directRows' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms packedRows_iff_directRows

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.generated_shape_exact' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms generated_shape_exact

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.generated_polynomial_degrees_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms generated_polynomial_degrees_exact

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.generated_polynomial_degree_at_most_eight' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms generated_polynomial_degree_at_most_eight

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.generated_source_rows_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generated_source_rows_exact

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.generatedSourceAccepts_iff_candidateZero' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generatedSourceAccepts_iff_candidateZero

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.generated_bit_polynomial' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generated_bit_polynomial

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.generated_residue_polynomial' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generated_residue_polynomial

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.generatedHighDecoder_fieldTerms_exact' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms generatedHighDecoder_fieldTerms_exact

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.generatedHighDecoder_output_eq_derived' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms generatedHighDecoder_output_eq_derived
