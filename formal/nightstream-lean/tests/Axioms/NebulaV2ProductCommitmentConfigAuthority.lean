import Nightstream.Implementation.NebulaV2.ProductCommitmentConfigAuthority
import tests.Axioms.Support

/-! Axiom gate for verifier-owned product-commitment authority. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductCommitmentConfigAuthority

open Nightstream.Implementation.NebulaV2.ProductCommitmentConfigAuthority

/-- info: 'Nightstream.Implementation.NebulaV2.ProductCommitmentConfigAuthority.config_ofTerminalLayout' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms config_ofTerminalLayout

/-- info: 'Nightstream.Implementation.NebulaV2.ProductCommitmentConfigAuthority.config_lanes' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms config_lanes

/-- info: 'Nightstream.Implementation.NebulaV2.ProductCommitmentConfigAuthority.config_fullKey' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms config_fullKey

/-- info: 'Nightstream.Implementation.NebulaV2.ProductCommitmentConfigAuthority.config_operationsKey' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms config_operationsKey

/-- info: 'Nightstream.Implementation.NebulaV2.ProductCommitmentConfigAuthority.config_snapshotKey' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms config_snapshotKey

end tests.Axioms.NebulaV2ProductCommitmentConfigAuthority
