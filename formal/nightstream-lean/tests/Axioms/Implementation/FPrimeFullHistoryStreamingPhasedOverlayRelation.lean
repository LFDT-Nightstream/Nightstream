import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPhasedOverlayRelation
import tests.Axioms.Support

/-! Fail-closed axiom guard for the production base-plus-overlay model. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.overlayKind_claim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.overlayKind_claim

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.overlayKind_piRlcFamily' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.overlayKind_piRlcFamily

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.exists_linkedAccepts_iff_jointArmSemantics' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.exists_linkedAccepts_iff_jointArmSemantics

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.linkedAccepts_implies_step_of_joint_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.linkedAccepts_implies_step_of_joint_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.exists_linkedAccepts_iff_overlayArmSemantics' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.exists_linkedAccepts_iff_overlayArmSemantics

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.linkedAccepts_implies_step' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.linkedAccepts_implies_step
