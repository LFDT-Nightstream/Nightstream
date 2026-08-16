import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetained
import tests.Axioms.Support

/-! Dependency audit for the normalized PiRLC carry retained-row scan. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.nonzero_census_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms nonzero_census_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.decoder_run_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms decoder_run_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.retained_intervals_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms retained_intervals_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.audit_valid' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms audit_valid
