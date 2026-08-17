import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetained
import tests.Axioms.Support

/-! Dependency audit for the normalized PiRLC overlay receipt. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayScheduleCertificate.schedule_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayScheduleCertificate.schedule_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.seed_schedule_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms seed_schedule_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.affine_geometry_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms affine_geometry_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.nonzero_census_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms nonzero_census_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained.audit_valid' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms audit_valid
