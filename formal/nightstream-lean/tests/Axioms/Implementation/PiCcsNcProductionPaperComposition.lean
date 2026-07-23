import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperChecker
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperSequence
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperTerminal
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperTrace
import tests.Axioms.Support

/-! Axiom-dependency probes for the opening-derived paper checker and its
base/recursive/terminal composition.

These probes intentionally remain unguarded until a focused run exposes the
actual kernel dependency sets. Replace each probe with a fail-closed
`#guard_msgs` expectation only after inspecting that output; `sorryAx` is
never acceptable. -/

#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperChecker.check_eq_true_iff_accepted

#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperSequence.checkedPair_of_nextPacked_implies_previousPackedAndPaper_or_namedFailure

#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperTerminal.checkedTerminal_implies_packedAndPaper_or_namedFailure

#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperTrace.closedTrace_implies_baseAndAllClosed_or_failure
