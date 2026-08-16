import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPhysicalOverlayRelation
import tests.Axioms.Support

/-! Fail-closed axiom guard for physical-overlay source soundness. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhysicalOverlayRelation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhysicalOverlayRelation.SourceSoundness.localSound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SourceSoundness.localSound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhysicalOverlayRelation.linkedAccepts_implies_step' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms linkedAccepts_implies_step
