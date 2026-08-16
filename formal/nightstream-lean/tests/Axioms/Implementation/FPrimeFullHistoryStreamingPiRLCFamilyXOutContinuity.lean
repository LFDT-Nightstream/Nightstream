import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyXOutContinuity
import tests.Axioms.Support

/-! Fail-closed axiom guard for complete `x_out` family continuity. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity.familyState_eq_or_poseidon2_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms familyState_eq_or_poseidon2_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity.local_digest_eq_or_phase_envelope_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms local_digest_eq_or_phase_envelope_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity.semantic_digest_eq_or_xOut_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms semantic_digest_eq_or_xOut_failure

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity.familyState_eq_or_continuity_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms familyState_eq_or_continuity_failure
