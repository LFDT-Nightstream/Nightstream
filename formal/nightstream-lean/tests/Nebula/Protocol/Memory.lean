import Nightstream.Protocol.Nebula

set_option autoImplicit false

namespace tests.NebulaMemory

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Memory

def initial : MemTuple := ⟨0, 0, 7⟩

def afterRead : MemTuple := ⟨1, 0, 7⟩

def readAccess : Access :=
  { space := .rom
    address := 0
    kind := .read
    read := initial
    write := afterRead }

def readAccessValid : readAccess.ValidAt 0 where
  wellFormed :=
    { addressInRange := by decide
      readIndex := rfl
      writeIndex := rfl
      readValueInRange := by decide
      writeValueInRange := by decide
      valueRule := rfl }
  timestampInRange := by decide
  timestampOutRange := by decide
  readBeforeWrite := by decide
  writeTimestamp := rfl

def orderedRead : Ordered 0 [readAccess] 1 :=
  .cons readAccessValid (.nil 1)

theorem one_read_balance :
    Balanced ({initial} : Multiset MemTuple) [readAccess]
      ({afterRead} : Multiset MemTuple) := by
  simp [Balanced, readTuples, writeTuples, readAccess]

theorem one_read_reconstructs :
    Executes ({initial} : Multiset MemTuple) 0 [readAccess]
      ({afterRead} : Multiset MemTuple) 1 :=
  balanced_implies_executes orderedRead one_read_balance

theorem one_read_final_record_has_canonical_bounds :
    afterRead.value < valueLimit ∧ afterRead.timestamp ≤ 1 := by
  apply ordered_write_valid_at_output orderedRead
  simp [writeTuples, readAccess, afterRead]

theorem one_read_final_record_has_an_authoritative_origin :
    afterRead ∈ ({initial} : Multiset MemTuple) ∨
      afterRead ∈ (writeTuples [readAccess] : Multiset MemTuple) := by
  apply balanced_final_member_origin one_read_balance
  simp

/- A disconnected two-edge cycle satisfies the multiset equation. It cannot
execute from the authoritative initial state. This is the countermodel that
requires strict, globally ordered integer write timestamps. -/
namespace MissingTimestampOrder

def isolatedInitial : MemTuple := ⟨0, 0, 3⟩
def future : MemTuple := ⟨2, 0, 4⟩
def past : MemTuple := ⟨1, 0, 5⟩

def first : Access :=
  { space := .ram
    address := 0
    kind := .write 5
    read := future
    write := past }

def second : Access :=
  { space := .ram
    address := 0
    kind := .write 4
    read := past
    write := future }

theorem cycle_balances :
    Balanced ({isolatedInitial} : Multiset MemTuple) [first, second]
      ({isolatedInitial} : Multiset MemTuple) := by
  change
    isolatedInitial ::ₘ past ::ₘ future ::ₘ 0 =
      future ::ₘ past ::ₘ isolatedInitial ::ₘ 0
  calc
    isolatedInitial ::ₘ past ::ₘ future ::ₘ 0 =
        past ::ₘ isolatedInitial ::ₘ future ::ₘ 0 :=
      Multiset.cons_swap _ _ _
    _ = past ::ₘ future ::ₘ isolatedInitial ::ₘ 0 := by
      rw [Multiset.cons_swap isolatedInitial future]
    _ = future ::ₘ past ::ₘ isolatedInitial ::ₘ 0 :=
      Multiset.cons_swap _ _ _

theorem cycle_is_not_sequential :
    ¬ Executes ({isolatedInitial} : Multiset MemTuple) 0 [first, second]
      ({isolatedInitial} : Multiset MemTuple) 2 := by
  intro execution
  cases execution with
  | cons applies _ =>
      have present := applies.readPresent
      have equal : future = isolatedInitial := by
        simpa using present
      have timestamps := congrArg MemTuple.timestamp equal
      change 2 = 0 at timestamps
      omega

end MissingTimestampOrder

end tests.NebulaMemory
