import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingFPrimeProgramSchema

/-!
Contract: structural support for bounded streaming-program leaf certificates.

Assurance tier: artifact-checked certificate support.

Owns exact, nonoverlapping 64-entry partitions for 400-entry schedule data and
86-entry claim-link data. Each certificate checks the complete source and
target lengths and the exact final remainder length.

Does not own generated data, semantic schedule definitions, or relation rows.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramLeafCertificateSupport

def tail64 {α : Type} (items : List α) : Nat → List α
  | 0 => items
  | index + 1 => (tail64 items index).drop 64

def chunk64 {α : Type} (items : List α) (index : Nat) : List α :=
  (tail64 items index).take 64

private theorem list_eq_of_take_drop
    {α : Type} {left right : List α} {count : Nat}
    (head : left.take count = right.take count)
    (tail : left.drop count = right.drop count) :
    left = right := by
  rw [← List.take_append_drop count left,
    ← List.take_append_drop count right, head, tail]

private theorem all_of_take_drop
    {α : Type} {items : List α} {predicate : α → Bool} {count : Nat}
    (head : (items.take count).all predicate = true)
    (tail : (items.drop count).all predicate = true) :
    items.all predicate = true := by
  rw [← List.take_append_drop count items, List.all_append, head, tail]
  rfl

/-- Six 64-entry leaves and one 16-entry remainder cover an exact 400-entry
list. The recursive `tail64` definition makes the partitions adjacent and
nonoverlapping. -/
structure Chunked400Eq {α : Type} (left right : List α) : Prop where
  leftLength : left.length = 400
  rightLength : right.length = 400
  chunk0 : chunk64 left 0 = chunk64 right 0
  chunk1 : chunk64 left 1 = chunk64 right 1
  chunk2 : chunk64 left 2 = chunk64 right 2
  chunk3 : chunk64 left 3 = chunk64 right 3
  chunk4 : chunk64 left 4 = chunk64 right 4
  chunk5 : chunk64 left 5 = chunk64 right 5
  remainder : tail64 left 6 = tail64 right 6
  leftRemainderLength : (tail64 left 6).length = 16
  rightRemainderLength : (tail64 right 6).length = 16

theorem Chunked400Eq.sound
    {α : Type} {left right : List α}
    (certificate : Chunked400Eq left right) :
    left = right := by
  have tail5 : tail64 left 5 = tail64 right 5 :=
    list_eq_of_take_drop certificate.chunk5 certificate.remainder
  have tail4 : tail64 left 4 = tail64 right 4 :=
    list_eq_of_take_drop certificate.chunk4 tail5
  have tail3 : tail64 left 3 = tail64 right 3 :=
    list_eq_of_take_drop certificate.chunk3 tail4
  have tail2 : tail64 left 2 = tail64 right 2 :=
    list_eq_of_take_drop certificate.chunk2 tail3
  have tail1 : tail64 left 1 = tail64 right 1 :=
    list_eq_of_take_drop certificate.chunk1 tail2
  exact list_eq_of_take_drop certificate.chunk0 tail1

/-- Bounded predicate checks over the same exact 400-entry partition. -/
structure Chunked400All {α : Type}
    (items : List α) (predicate : α → Bool) : Prop where
  length : items.length = 400
  chunk0 : (chunk64 items 0).all predicate = true
  chunk1 : (chunk64 items 1).all predicate = true
  chunk2 : (chunk64 items 2).all predicate = true
  chunk3 : (chunk64 items 3).all predicate = true
  chunk4 : (chunk64 items 4).all predicate = true
  chunk5 : (chunk64 items 5).all predicate = true
  remainder : (tail64 items 6).all predicate = true
  remainderLength : (tail64 items 6).length = 16

theorem Chunked400All.sound
    {α : Type} {items : List α} {predicate : α → Bool}
    (certificate : Chunked400All items predicate) :
    items.all predicate = true := by
  have tail5 : (tail64 items 5).all predicate = true :=
    all_of_take_drop certificate.chunk5 certificate.remainder
  have tail4 : (tail64 items 4).all predicate = true :=
    all_of_take_drop certificate.chunk4 tail5
  have tail3 : (tail64 items 3).all predicate = true :=
    all_of_take_drop certificate.chunk3 tail4
  have tail2 : (tail64 items 2).all predicate = true :=
    all_of_take_drop certificate.chunk2 tail3
  have tail1 : (tail64 items 1).all predicate = true :=
    all_of_take_drop certificate.chunk1 tail2
  exact all_of_take_drop certificate.chunk0 tail1

/-- One 64-entry leaf and one 22-entry remainder cover an exact 86-entry
list. -/
structure Chunked86Eq {α : Type} (left right : List α) : Prop where
  leftLength : left.length = 86
  rightLength : right.length = 86
  chunk : chunk64 left 0 = chunk64 right 0
  remainder : tail64 left 1 = tail64 right 1
  leftRemainderLength : (tail64 left 1).length = 22
  rightRemainderLength : (tail64 right 1).length = 22

theorem Chunked86Eq.sound
    {α : Type} {left right : List α}
    (certificate : Chunked86Eq left right) :
    left = right :=
  list_eq_of_take_drop certificate.chunk certificate.remainder

/-- Bounded predicate checks over the exact 86-entry partition. -/
structure Chunked86All {α : Type}
    (items : List α) (predicate : α → Bool) : Prop where
  length : items.length = 86
  chunk : (chunk64 items 0).all predicate = true
  remainder : (tail64 items 1).all predicate = true
  remainderLength : (tail64 items 1).length = 22

theorem Chunked86All.sound
    {α : Type} {items : List α} {predicate : α → Bool}
    (certificate : Chunked86All items predicate) :
    items.all predicate = true :=
  all_of_take_drop certificate.chunk certificate.remainder

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramLeafCertificateSupport
