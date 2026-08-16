import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyContinuity
import tests.Axioms.Support

/-! Fail-closed axiom guard for PiRLC family-state continuity. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.familyStateFields_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms familyStateFields_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity.state_digest_eq_family_state_digest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms state_digest_eq_family_state_digest

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity.accepted_public_continuity' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepted_public_continuity
