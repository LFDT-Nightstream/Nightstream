import Nightstream.Implementation.Nebula.Commitment.Core.ConfigAuthority
import tests.Axioms.Support

/-! Axiom gate for verifier-owned product-commitment authority. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaProductCommitmentConfigAuthority

open Nightstream.Implementation.Nebula.ProductCommitmentConfigAuthority

/-- info: 'Nightstream.Implementation.Nebula.ProductCommitmentConfigAuthority.config_ofTerminalLayout' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms config_ofTerminalLayout

/-- info: 'Nightstream.Implementation.Nebula.ProductCommitmentConfigAuthority.config_lanes' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms config_lanes

/-- info: 'Nightstream.Implementation.Nebula.ProductCommitmentConfigAuthority.config_fullKey' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms config_fullKey

/-- info: 'Nightstream.Implementation.Nebula.ProductCommitmentConfigAuthority.config_operationsKey' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms config_operationsKey

/-- info: 'Nightstream.Implementation.Nebula.ProductCommitmentConfigAuthority.config_snapshotKey' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms config_snapshotKey

end tests.Axioms.NebulaProductCommitmentConfigAuthority
