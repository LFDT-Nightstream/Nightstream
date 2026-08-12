import Nightstream.Implementation.NebulaV2.BundleForwardingRows
import tests.NebulaV2CommitmentBundleCodec

set_option autoImplicit false

namespace tests.NebulaV2BundleForwardingRows

open Nightstream.Implementation.NebulaV2.BundleForwardingRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry

def layout : Layout :=
  { inputStart := 1
    outputStart := 1 + mandatoryBundleBits }

theorem exact_number_of_forwarding_rows :
    (rows layout).length = 248832 := by
  rw [rows_length]
  exact mandatoryBundleBits_exact

/-- The equality rows cannot accept a mutation of only the operations
component when both bundle values are linked to their codec bits. -/
theorem changed_operations_component_is_rejected
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment
      tests.NebulaV2CommitmentBundleCodec.zeroValue
      tests.NebulaV2CommitmentBundleCodec.changedOperations) :
    ¬ RowsHold layout assignment := by
  intro holds
  have forwarded := exact_bundle_forwarding canonical one placed holds
  exact tests.NebulaV2CommitmentBundleCodec.changed_component_changes_bundle
    forwarded

end tests.NebulaV2BundleForwardingRows
