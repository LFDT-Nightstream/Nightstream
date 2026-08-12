import Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrimeCountermodels
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionBatchedGlobalFPrimeCountermodels

open Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrimeCountermodels

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrimeCountermodels.weak_local_checks_accept_changed_authority' does not depend on any axioms -/
#guard_msgs in
#audit_axioms weak_local_checks_accept_changed_authority

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrimeCountermodels.fixed_lifetime_authority_rejects_countermodel' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms fixed_lifetime_authority_rejects_countermodel

end tests.Axioms.NebulaV2ProductionBatchedGlobalFPrimeCountermodels
