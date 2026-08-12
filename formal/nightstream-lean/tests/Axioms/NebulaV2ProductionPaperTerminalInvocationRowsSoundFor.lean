import Nightstream.Implementation.NebulaV2.ProductionPaperTerminalInvocationRowsSoundFor
import tests.Axioms.Support

/-! Fail-closed kernel-dependency guards for the indexed terminal branch. -/

set_option pp.universes true in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperTerminalInvocationRowsSoundFor.exactOfHolds
