import Mathlib.Data.Multiset.AddSub
import Nightstream.Protocol.Nebula.Types

/-!
Contract: reverse sequential-consistency semantics for one V2 memory segment.

Assurance tier: model-level.

Owns multiset memory states, exact access execution, the IS/WS versus RS/FS
balance predicate, and reconstruction of an execution from balance plus the
V2 ordered integer timestamp schedule.

Does not own complete-scan extraction, application-port coverage,
fingerprints, commitments, circuit rows, or cryptographic probabilities.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.Memory

open Nightstream.Protocol.Nebula

def readTuples (accesses : List Access) : List MemTuple :=
  accesses.map Access.read

def writeTuples (accesses : List Access) : List MemTuple :=
  accesses.map Access.write

/-- Exact multiset equation checked by a sound fingerprint opening. -/
def Balanced
    (initial : Multiset MemTuple)
    (accesses : List Access)
    (final : Multiset MemTuple) : Prop :=
  initial + (writeTuples accesses : Multiset MemTuple) =
    (readTuples accesses : Multiset MemTuple) + final

/-- One operational transition removes the record read by the application and
installs its checked write record. -/
structure Applies
    (before : Multiset MemTuple)
    (timestampIn : Nat)
    (access : Access)
    (after : Multiset MemTuple) : Prop where
  valid : access.ValidAt timestampIn
  readPresent : access.read ∈ before
  afterExact :
    after = ({access.write} : Multiset MemTuple) + before.erase access.read

/-- Exact sequential execution in the application-provided access order. -/
inductive Executes :
    Multiset MemTuple → Nat → List Access → Multiset MemTuple → Nat → Prop
  | nil (state : Multiset MemTuple) (timestamp : Nat) :
      Executes state timestamp [] state timestamp
  | cons
      {before middle final : Multiset MemTuple}
      {timestampIn timestampOut : Nat}
      {access : Access}
      {rest : List Access}
      (applies : Applies before timestampIn access middle)
      (tail : Executes middle (timestampIn + 1) rest final timestampOut) :
      Executes before timestampIn (access :: rest) final timestampOut

/-- Every write in an ordered suffix occurs after the suffix input timestamp. -/
theorem ordered_write_timestamp_gt
    {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (ordered : Ordered timestampIn accesses timestampOut)
    {entry : MemTuple}
    (member : entry ∈ (writeTuples accesses : Multiset MemTuple)) :
    timestampIn < entry.timestamp := by
  induction ordered with
  | nil =>
      simp [writeTuples] at member
  | @cons timestampIn timestampOut access rest valid tail inductionHypothesis =>
      have memberList : entry ∈ writeTuples (access :: rest) := by
        exact member
      simp only [writeTuples, List.map_cons, List.mem_cons] at memberList
      rcases memberList with equal | memberList
      · subst entry
        simpa only [valid.writeTimestamp] using Nat.lt_succ_self timestampIn
      · have tailMember : entry ∈ (writeTuples rest : Multiset MemTuple) := by
          exact memberList
        have later := inductionHypothesis tailMember
        exact Nat.lt_trans (Nat.lt_succ_self timestampIn) later

/-- Every committed write record is a canonical value and its timestamp is at
most the segment output timestamp. -/
theorem ordered_write_valid_at_output
    {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (ordered : Ordered timestampIn accesses timestampOut)
    {entry : MemTuple}
    (member : entry ∈ (writeTuples accesses : Multiset MemTuple)) :
    entry.value < valueLimit ∧ entry.timestamp ≤ timestampOut := by
  induction ordered with
  | nil =>
      simp [writeTuples] at member
  | @cons timestampIn timestampOut access rest valid tail
      inductionHypothesis =>
      have memberList : entry ∈ writeTuples (access :: rest) := member
      simp only [writeTuples, List.map_cons, List.mem_cons] at memberList
      rcases memberList with equal | memberList
      · subst entry
        constructor
        · exact valid.wellFormed.writeValueInRange
        · rw [valid.writeTimestamp, tail.timestampOut_eq]
          omega
      · exact inductionHypothesis memberList

/-- Exact balance makes each final record originate in either the initial
snapshot or the write-record sequence. Read records cannot create a final
record. -/
theorem balanced_final_member_origin
    {initial final : Multiset MemTuple}
    {accesses : List Access}
    (balance : Balanced initial accesses final)
    {entry : MemTuple}
    (member : entry ∈ final) :
    entry ∈ initial ∨
      entry ∈ (writeTuples accesses : Multiset MemTuple) := by
  have onRight :
      entry ∈ (readTuples accesses : Multiset MemTuple) + final :=
    Multiset.mem_add.mpr (Or.inr member)
  have onLeft :
      entry ∈ initial + (writeTuples accesses : Multiset MemTuple) := by
    rw [balance]
    exact onRight
  exact Multiset.mem_add.mp onLeft

private theorem head_read_mem_initial
    {initial final : Multiset MemTuple}
    {timestampIn timestampOut : Nat}
    {access : Access}
    {rest : List Access}
    (ordered : Ordered timestampIn (access :: rest) timestampOut)
    (balance : Balanced initial (access :: rest) final) :
    access.read ∈ initial := by
  have readOnRight :
      access.read ∈
        (readTuples (access :: rest) : Multiset MemTuple) + final := by
    simp [readTuples]
  have readOnLeft :
      access.read ∈
        initial + (writeTuples (access :: rest) : Multiset MemTuple) := by
    rw [balance]
    exact readOnRight
  rcases Multiset.mem_add.mp readOnLeft with readInInitial | readInWrites
  · exact readInInitial
  · have writeAfter := ordered_write_timestamp_gt ordered readInWrites
    cases ordered with
    | cons valid _ =>
        have readBefore := valid.readBeforeWrite
        omega

private theorem cancel_head
    {initial final : Multiset MemTuple}
    {access : Access}
    {rest : List Access}
    (readPresent : access.read ∈ initial)
    (balance : Balanced initial (access :: rest) final) :
    Balanced
      (({access.write} : Multiset MemTuple) + initial.erase access.read)
      rest
      final := by
  unfold Balanced at balance ⊢
  have initialDecomposition :
      ({access.read} : Multiset MemTuple) + initial.erase access.read = initial := by
    simpa only [Multiset.singleton_add] using Multiset.cons_erase readPresent
  apply Multiset.add_right_inj.mp
  calc
    ({access.read} : Multiset MemTuple) +
        ((({access.write} : Multiset MemTuple) + initial.erase access.read) +
          (writeTuples rest : Multiset MemTuple)) =
      (({access.read} : Multiset MemTuple) + initial.erase access.read) +
        (({access.write} : Multiset MemTuple) +
          (writeTuples rest : Multiset MemTuple)) := by
            rw [Multiset.add_comm
              ({access.write} : Multiset MemTuple)
              (initial.erase access.read)]
            rw [Multiset.add_assoc]
            rw [← Multiset.add_assoc]
    _ = initial +
        (({access.write} : Multiset MemTuple) +
          (writeTuples rest : Multiset MemTuple)) := by
          rw [initialDecomposition]
    _ = initial + (writeTuples (access :: rest) : Multiset MemTuple) := by
          simp only [writeTuples, List.map_cons, ← Multiset.cons_coe,
            ← Multiset.singleton_add]
    _ = (readTuples (access :: rest) : Multiset MemTuple) + final := balance
    _ = ({access.read} : Multiset MemTuple) +
        ((readTuples rest : Multiset MemTuple) + final) := by
          simp only [readTuples, List.map_cons, ← Multiset.cons_coe,
            ← Multiset.singleton_add]
          exact Multiset.add_assoc _ _ _

/-- Reverse direction needed by V2 soundness: exact multiset balance cannot
hide a future read when write timestamps follow the global integer access
order. The proof reconstructs the application access sequence one operation at
a time. -/
theorem balanced_implies_executes
    {initial final : Multiset MemTuple}
    {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (ordered : Ordered timestampIn accesses timestampOut)
    (balance : Balanced initial accesses final) :
    Executes initial timestampIn accesses final timestampOut := by
  induction ordered generalizing initial with
  | nil timestamp =>
      have finalExact : initial = final := by
        simpa [Balanced, writeTuples, readTuples] using balance
      subst final
      exact .nil initial timestamp
  | @cons timestampIn timestampOut access rest valid tail inductionHypothesis =>
      have readPresent := head_read_mem_initial (.cons valid tail) balance
      let middle : Multiset MemTuple :=
        ({access.write} : Multiset MemTuple) + initial.erase access.read
      have tailBalance : Balanced middle rest final := by
        exact cancel_head readPresent balance
      exact .cons
        { valid := valid
          readPresent := readPresent
          afterExact := rfl }
        (inductionHypothesis tailBalance)

theorem balanced_implies_timestamp_exact
    {initial final : Multiset MemTuple}
    {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (ordered : Ordered timestampIn accesses timestampOut)
    (_balance : Balanced initial accesses final) :
    timestampOut = timestampIn + accesses.length :=
  ordered.timestampOut_eq

end Nightstream.Protocol.Nebula.Memory
