import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrongBoundary
import tests.Axioms.Support

/-! Fail-closed guards for the exact finite PiCCS runtime boundary. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrongBoundary.extractorRuntime_iff_uniformTruncatedWorkBound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrongBoundary.extractorRuntime_iff_uniformTruncatedWorkBound

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrongBoundary.extractorRuntime_iff_all_finite_cutoffs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrongBoundary.extractorRuntime_iff_all_finite_cutoffs
