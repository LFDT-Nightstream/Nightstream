import Nightstream.Implementation.NebulaV2.Memory.Claim.PoseidonRows

/-! Focused gates for the exact memory-claim Poseidon2 row relation. -/

set_option autoImplicit false

namespace tests.NebulaV2MemoryClaimPoseidonRows

open Nightstream.Implementation.NebulaV2

theorem exact_round_count
    {layout : MemoryClaimPoseidonRows.Layout}
    (valid : layout.Valid) : layout.trace.rounds.length = 24 :=
  valid.round_count_exact

theorem exact_row_count
    {layout : MemoryClaimPoseidonRows.Layout}
    (valid : layout.Valid) :
    (MemoryClaimPoseidonRows.rows layout).length = 14501 :=
  MemoryClaimPoseidonRows.rows_length_exact valid

#check MemoryClaimPoseidonRows.output_columns_eq_digest
#check MemoryClaimPoseidonRows.rows_complete

end tests.NebulaV2MemoryClaimPoseidonRows
