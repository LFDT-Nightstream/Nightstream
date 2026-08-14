import Nightstream.Protocol.Nebula.ScanSchedule

/-! Focused gates for the exact V2 full-memory scan schedule. -/

set_option autoImplicit false

namespace tests.NebulaScanSchedule

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ScanSchedule

theorem exact_scan_capacity :
    Nightstream.Protocol.Nebula.Lifecycle.claimsPerSegment * scanSlots =
      scannedCells :=
  scan_capacity

theorem every_address_has_one_position :
    Function.Bijective Position.globalIndex :=
  globalIndex_bijective

theorem count_alone_allows_repeated_steps :
    repeatedStepIndexes.length =
        Nightstream.Protocol.Nebula.Lifecycle.claimsPerSegment ∧
      repeatedStepIndexes ≠
        List.range Nightstream.Protocol.Nebula.Lifecycle.claimsPerSegment :=
  ⟨repeatedStepIndexes_has_exact_count,
    repeatedStepIndexes_is_not_canonical⟩

#check verifiedRun_claim_segment_bounds_at

end tests.NebulaScanSchedule
