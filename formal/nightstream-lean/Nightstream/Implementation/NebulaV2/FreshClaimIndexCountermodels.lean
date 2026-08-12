/-!
Contract: countermodels for the fresh-claim index in one recursive F-prime
invocation.

HyperNova Construction 2 verifies the claim produced by the preceding
invocation, executes the current application witness, and then produces the
claim for the current invocation. Therefore the current application batch
must bind to the produced claim. Binding it to the consumed claim is both
unsound and incomplete.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.FreshClaimIndexCountermodels

/-- The three distinct claim-time values in one recursive invocation. -/
structure Invocation where
  consumedBatch : Bool
  currentBatch : Bool
  producedBatch : Bool
deriving DecidableEq, Repr

/-- Rejected off-by-one bridge: the current batch is equated with the batch
inside the claim that this invocation consumes. -/
def ConsumedClaimBridge (invocation : Invocation) : Prop :=
  invocation.currentBatch = invocation.consumedBatch

/-- Required bridge: the current batch is equated with the batch inside the
fresh claim that this invocation produces. -/
def ProducedClaimBridge (invocation : Invocation) : Prop :=
  invocation.currentBatch = invocation.producedBatch

/-- The rejected bridge accepts a produced claim that does not contain the
current batch. -/
def falseAccept : Invocation :=
  { consumedBatch := false
    currentBatch := false
    producedBatch := true }

theorem consumed_bridge_does_not_imply_produced_bridge :
    ConsumedClaimBridge falseAccept /\
      ¬ ProducedClaimBridge falseAccept := by
  simp [ConsumedClaimBridge, ProducedClaimBridge, falseAccept]

/-- A valid recursive invocation can consume a different prior batch while
its produced claim contains the current batch. -/
def falseReject : Invocation :=
  { consumedBatch := false
    currentBatch := true
    producedBatch := true }

theorem consumed_bridge_rejects_valid_successor :
    ProducedClaimBridge falseReject /\
      ¬ ConsumedClaimBridge falseReject := by
  simp [ConsumedClaimBridge, ProducedClaimBridge, falseReject]

end Nightstream.Implementation.NebulaV2.FreshClaimIndexCountermodels
