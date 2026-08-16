import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPhasedRelation
import tests.Axioms.Support

/-! Fail-closed axiom guard for the exact phased F-prime schedule model. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.exists_phaseAtArm_iff_step' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.exists_phaseAtArm_iff_step

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.exists_linkedAccepts_iff_armSemantics' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.exists_linkedAccepts_iff_armSemantics

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.linkedAccepts_implies_step' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.linkedAccepts_implies_step

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.terminal_complete_steps_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation.terminal_complete_steps_exact
