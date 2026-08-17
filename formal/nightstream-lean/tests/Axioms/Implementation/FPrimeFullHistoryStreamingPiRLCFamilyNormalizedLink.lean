import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLink
import tests.Axioms.Support

/-! Dependency audit for the normalized PiRLC body-overlay link receipt. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.source_geometry_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms source_geometry_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.final_geometry_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms final_geometry_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.link_census_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms link_census_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.cross_receipts_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms cross_receipts_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink.audit_valid' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms audit_valid
