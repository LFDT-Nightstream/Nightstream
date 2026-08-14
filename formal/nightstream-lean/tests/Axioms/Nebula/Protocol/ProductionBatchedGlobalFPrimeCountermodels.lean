import Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrimeCountermodels
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionBatchedGlobalFPrimeCountermodels

open Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrimeCountermodels

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrimeCountermodels.weak_local_checks_accept_changed_authority' does not depend on any axioms -/
#guard_msgs in
#audit_axioms weak_local_checks_accept_changed_authority

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrimeCountermodels.fixed_lifetime_authority_rejects_countermodel' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms fixed_lifetime_authority_rejects_countermodel

end tests.Axioms.NebulaProductionBatchedGlobalFPrimeCountermodels
