import Nightstream.Implementation.NebulaV2.Production.FPrime.Lifetime.DefaultRunningFor
import tests.Axioms.Support

set_option pp.universes true in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperDefaultRunningFor.commit_zero

set_option pp.universes true in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperDefaultRunningFor.slot_holds
