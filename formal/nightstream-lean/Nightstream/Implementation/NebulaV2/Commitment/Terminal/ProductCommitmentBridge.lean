import Nightstream.Implementation.NebulaV2.Commitment.Core.Algebra
import Nightstream.Implementation.NebulaV2.Commitment.Terminal.BundleOpeningRows

/-!
Contract: identity between the V2 product-commitment algebra and the exact
typed terminal opening compiler.

Assurance tier: implementation-to-algebra bridge.

Owns construction of the semantic product configuration from the terminal
layout's verifier-key-selected full, operations, and shared snapshot keys; and
proof that its commitment of the terminal full witness is exactly the bundle
computed by the terminal rows.

Does not own accumulator output binding, terminal CE membership, generated
numeric-to-typed assignment refinement, cryptographic binding, Rust, or the
deployed verifier.

Emits constraints: no additional rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.TerminalProductCommitmentBridge

open Nightstream.SuperNeo.Concrete.Phi81Relation

variable {manifest : SeedSchedule.Manifest}
variable {fullShape operationsShape snapshotShape : Shape}

/-- The exact semantic configuration selected by one typed terminal layout. -/
def config
    (layout : TerminalBundleOpeningRows.Layout manifest fullShape
      operationsShape snapshotShape) :
    ProductCommitmentAlgebra.Config fullShape operationsShape snapshotShape where
  lanes := layout.lanes
  fullKey := layout.agreement.fullKey
  operationsKey := layout.agreement.operationsKey
  snapshotKey := layout.agreement.snapshotKey

/-- The terminal compiler and product algebra read the same full assignment,
the same aligned lane projections, and the same three verifier-selected key
roles. -/
theorem commit_eq_exactBundle
    (layout : TerminalBundleOpeningRows.Layout manifest fullShape
      operationsShape snapshotShape)
    (assignment :
      Nightstream.Implementation.Lowering.Goldilocks.ColumnId →
        Nightstream.SuperNeo.Concrete.F) :
    ProductCommitmentAlgebra.commit (config layout)
        (layout.fullAssignment assignment) =
      TerminalBundleOpeningRows.exactBundle layout assignment := by
  funext component
  cases component <;> rfl

/-- A row-derived terminal opening is an opening under the exact product map
used by the concrete PiRLC/PiDEC algebra. -/
theorem product_opening_of_terminal_rows
    (layout : TerminalBundleOpeningRows.Layout manifest fullShape
      operationsShape snapshotShape)
    (assignment :
      Nightstream.Implementation.Lowering.Goldilocks.ColumnId →
        Nightstream.SuperNeo.Concrete.F)
    (bundle : ProductCommitmentAlgebra.BundleValue)
    (opens : TerminalBundleOpeningRows.exactBundle layout assignment = bundle) :
    ProductCommitmentAlgebra.commit (config layout)
        (layout.fullAssignment assignment) = bundle := by
  rw [commit_eq_exactBundle]
  exact opens

end Nightstream.Implementation.NebulaV2.TerminalProductCommitmentBridge
