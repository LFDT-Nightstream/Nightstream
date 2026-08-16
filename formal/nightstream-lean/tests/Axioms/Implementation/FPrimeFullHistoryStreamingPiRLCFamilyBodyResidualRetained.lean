import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetained
import tests.Axioms.Support

/-! Dependency audit for the normalized PiRLC residual retained-row scan. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained.nonzero_census_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms nonzero_census_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained.decoder_run_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms decoder_run_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained.retained_intervals_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms retained_intervals_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained.audit_valid' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms audit_valid
