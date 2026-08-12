import Nightstream.Protocol.NebulaV2.Snapshot

/-!
Contract: independent semantic validity of one closed V2 memory segment.

Assurance tier: model-level.

Owns the conjunction between complete boundary snapshots, the global integer
timestamp schedule, and exact multiset balance. It derives an operational
execution in the application access order.

Does not own the fingerprint reduction that establishes balance, the
commitment reduction that establishes boundary authority, or circuit rows.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2

open Memory

/-- Semantic obligations for one closed segment after all computational bad
events have been excluded. -/
structure ValidSegment
    (initial final : Snapshot)
    (timestampIn : Nat)
    (accesses : List Access)
    (timestampOut : Nat) : Prop where
  initialValid : initial.ValidAt timestampIn
  finalValid : final.ValidAt timestampOut
  ordered : Ordered timestampIn accesses timestampOut
  balanced : Balanced initial.tuples accesses final.tuples

namespace ValidSegment

/-- Final snapshot validity follows from the authoritative initial bound,
ordered writes, and exact multiset balance. The direct `finalValid` field is a
useful circuit conformance check, but it is not an extra semantic assumption.
-/
theorem finalValid_of_balance
    {initial final : Snapshot}
    {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (initialValid : initial.ValidAt timestampIn)
    (ordered : Ordered timestampIn accesses timestampOut)
    (balanced : Balanced initial.tuples accesses final.tuples) :
    final.ValidAt timestampOut := by
  intro index
  have member := final.tupleAt_mem index
  rcases balanced_final_member_origin balanced member with
    initialMember | writeMember
  · have validEntry := initial.tuple_mem_validAt initialValid initialMember
    have timestampOrder : timestampIn ≤ timestampOut := by
      rw [ordered.timestampOut_eq]
      omega
    constructor
    · simpa [Snapshot.tupleAt] using validEntry.1
    · have timestampBound := validEntry.2
      simp only [Snapshot.tupleAt] at timestampBound
      exact Nat.le_trans timestampBound timestampOrder
  · simpa [Snapshot.tupleAt] using
      ordered_write_valid_at_output ordered writeMember

theorem executes
    {initial final : Snapshot}
    {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (segment : ValidSegment initial final timestampIn accesses timestampOut) :
    Executes initial.tuples timestampIn accesses final.tuples timestampOut :=
  balanced_implies_executes segment.ordered segment.balanced

theorem timestampOut_eq
    {initial final : Snapshot}
    {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (segment : ValidSegment initial final timestampIn accesses timestampOut) :
    timestampOut = timestampIn + accesses.length :=
  segment.ordered.timestampOut_eq

end ValidSegment

end Nightstream.Protocol.NebulaV2
