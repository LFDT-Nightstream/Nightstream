import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.InvocationRowsSoundFor
import tests.Axioms.Support

/-! Fail-closed kernel-dependency guards for the indexed terminal branch. -/

set_option pp.universes true in
#audit_axioms Nightstream.Implementation.Nebula.ProductionPaperTerminalInvocationRowsSoundFor.exactOfHolds
