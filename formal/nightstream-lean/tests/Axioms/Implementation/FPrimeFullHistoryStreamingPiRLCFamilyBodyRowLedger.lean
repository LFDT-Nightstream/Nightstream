import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger
import tests.Axioms.Support

/-! Dependency audit for the production PiRLC body row ledger. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger.dimensions_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms dimensions_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger.family_census_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms family_census_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger.maximum_check_run_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms maximum_check_run_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger.ledger_valid' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ledger_valid
