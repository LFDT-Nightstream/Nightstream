import Nightstream.Implementation.NebulaV2.MemoryClaimHashFrameRows

/-! Focused gates for the exact 91-field memory-claim hash frame. -/

set_option autoImplicit false

namespace tests.NebulaV2MemoryClaimHashFrameRows

open Nightstream.Implementation.NebulaV2

example (layout : MemoryClaimHashFrameRows.Layout) :
    (MemoryClaimHashFrameRows.inputColumns layout).length = 91 :=
  MemoryClaimHashFrameRows.inputColumns_length layout

example (layout : MemoryClaimHashFrameRows.Layout) :
    (MemoryClaimHashFrameRows.rows layout).length = 8 :=
  MemoryClaimHashFrameRows.rows_length_exact layout

#check MemoryClaimHashFrameRows.input_column_values
#check MemoryClaimHashFrameRows.rows_complete

end tests.NebulaV2MemoryClaimHashFrameRows
