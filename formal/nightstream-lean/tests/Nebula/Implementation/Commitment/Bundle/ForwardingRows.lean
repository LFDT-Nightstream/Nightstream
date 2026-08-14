import Nightstream.Implementation.Nebula.Commitment.Bundle.ForwardingRows
import tests.Nebula.Implementation.Commitment.Bundle.Codec

set_option autoImplicit false

namespace tests.NebulaBundleForwardingRows

open Nightstream.Implementation.Nebula.BundleForwardingRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.MemoryWireGeometry

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
      tests.NebulaCommitmentBundleCodec.zeroValue
      tests.NebulaCommitmentBundleCodec.changedOperations) :
    ¬ RowsHold layout assignment := by
  intro holds
  have forwarded := exact_bundle_forwarding canonical one placed holds
  exact tests.NebulaCommitmentBundleCodec.changed_component_changes_bundle
    forwarded

end tests.NebulaBundleForwardingRows
