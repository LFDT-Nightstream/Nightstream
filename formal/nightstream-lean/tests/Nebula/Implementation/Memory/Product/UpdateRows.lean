import Nightstream.Implementation.Nebula.Memory.Product.UpdateRows

/-! Focused checks for the fixed eight product-update chains. -/

set_option autoImplicit false

namespace tests.NebulaMemoryProductUpdateRows

open Nightstream.Implementation.Nebula.MemoryProductUpdateRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner

theorem exact_row_count (layout : Layout) :
    (rows layout).length = 4072 :=
  rows_length_exact layout

theorem every_operation_endpoint_is_derived
    {layout : Layout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (repetition : Fin 2) (role : OperationRole) :
    carriedValue assignment
        (layout.operationChain repetition role).final =
      Nightstream.Implementation.Nebula.MemoryProductChainRows.productValue
        assignment
        (layout.operationChain repetition role).gamma1
        (layout.operationChain repetition role).gamma2
        (carriedValue assignment
          (layout.operationChain repetition role).initial)
        (layout.operationChain repetition role).entries :=
  operationChain_sound one holds repetition role

theorem every_snapshot_endpoint_is_derived
    {layout : Layout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (repetition : Fin 2) (role : SnapshotRole) :
    carriedValue assignment
        (layout.snapshotChain repetition role).final =
      Nightstream.Implementation.Nebula.MemoryProductChainRows.productValue
        assignment
        (layout.snapshotChain repetition role).gamma1
        (layout.snapshotChain repetition role).gamma2
        (carriedValue assignment
          (layout.snapshotChain repetition role).initial)
        (layout.snapshotChain repetition role).entries :=
  snapshotChain_sound one holds repetition role

end tests.NebulaMemoryProductUpdateRows
