import Nightstream.Implementation.NebulaV2.FPrime.Claim.IndexCountermodels
import tests.Axioms.Support

/-! Dependency audit for the F-prime fresh-claim index countermodels. -/

/-- info: 'Nightstream.Implementation.NebulaV2.FreshClaimIndexCountermodels.consumed_bridge_does_not_imply_produced_bridge' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FreshClaimIndexCountermodels.consumed_bridge_does_not_imply_produced_bridge

/-- info: 'Nightstream.Implementation.NebulaV2.FreshClaimIndexCountermodels.consumed_bridge_rejects_valid_successor' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FreshClaimIndexCountermodels.consumed_bridge_rejects_valid_successor
