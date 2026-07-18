import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.SumCheck
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.DigestRounds

/-!
Pure transcript-machine lemmas shared by terminal FE and NC SumCheck rounds.

Assurance tier: executable implementation semantics.

Owns: exact four-word absorption from cursor zero; rate-boundary
normalization; ten-field raw-message serialization into two full
permutations plus a three-field tail; and field interpretation of accepted
small constants.

Does not own: artifact rows or columns; any particular round index; typed
message authority; Poseidon2 call acceptance; SumCheck algebra; costs;
necessity; or row removal.

Emits constraints: no.

Authority boundary: these theorems mention only the canonical transcript
machine. They expose no generated artifact values and therefore cannot grant
authority to an implementation trace.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.sumcheck.round.message.block` | four absorbed words fill the rate from cursor zero | computed | `absorbAll_four_of_cursorZero` |
| `nifs.pi_ccs.sumcheck.round.0.message.block` | three absorbed words fill the rate from cursor one without overwriting lane zero | computed | `absorbAll_three_of_cursorOne` |
| `nifs.pi_ccs.sumcheck.round.message.boundary` | the next word crosses a full-rate boundary through one permutation | computed | `absorbAll_cons_of_full` |
| `nifs.pi_ccs.sumcheck.prologue.serialize.0` | two singleton raw messages fill and normalize one rate block | computed | `appendRaw_singletons_of_cursorZero` |
| `nifs.pi_ccs.sumcheck.prologue.serialize.1` | a raw pair plus singleton fills one block and retains the singleton payload | computed | `appendRaw_pair_then_singleton_of_cursorZero` |
| `nifs.pi_ccs.sumcheck.round.message.serialize` | ten fields become length + 3, then 4, then 3 fields | computed | `appendRaw_ten_of_cursorZero` |
| `nifs.pi_ccs.sumcheck.round.0.message.serialize` | ten fields from a cursor-one prologue become three exact full-rate permutations | computed | `appendRaw_ten_of_cursorOne` |
| `nifs.pi_ccs.sumcheck.round.constant` | a canonical accepted small integer column equals its transcript word | derived | `fieldAt_eq_wordField` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives

set_option maxHeartbeats 1000000

theorem stateExt
    {left right : State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) :
    left = right := by
  cases left
  cases right
  simp_all

theorem laneValueCases (lane : Fin width) :
    lane.val = 0 ∨ lane.val = 1 ∨ lane.val = 2 ∨ lane.val = 3 ∨
    lane.val = 4 ∨ lane.val = 5 ∨ lane.val = 6 ∨ lane.val = 7 := by
  have laneLt : lane.val < 8 := by
    simpa [width] using lane.isLt
  omega

theorem fieldAt_eq_wordField
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    {column value : Nat}
    (equal : assignment column = value)
    (valueLtU64 : value < u64Modulus)
    (valueLtField : value < goldilocksP) :
    fieldAt assignment canonical column = wordField value := by
  apply Fin.ext
  simp [fieldAt, wordField, fieldValue, equal,
    Nat.mod_eq_of_lt valueLtU64, Nat.mod_eq_of_lt valueLtField]

theorem absorbAll_append
    (initial : State)
    (first rest : List Field) :
    absorbAll initial (first ++ rest) =
      absorbAll (absorbAll initial first) rest := by
  induction first generalizing initial with
  | nil => rfl
  | cons value first inductionHypothesis =>
      simp only [List.cons_append, absorbAll]
      exact inductionHypothesis (absorbElem initial value)

theorem absorbElem_lanes_of_room
    (state : State)
    (value : Field)
    (room : state.absorbed.val < rate) :
    (absorbElem state value).lanes =
      overwriteLane state.lanes state.absorbed.val value := by
  unfold absorbElem
  rw [dif_pos room]

theorem absorbElem_absorbed_of_room
    (state : State)
    (value : Field)
    (room : state.absorbed.val < rate) :
    (absorbElem state value).absorbed.val =
      state.absorbed.val + 1 := by
  unfold absorbElem
  rw [dif_pos room]

theorem absorbElem_lanes_of_full
    (state : State)
    (value : Field)
    (full : ¬ state.absorbed.val < rate) :
    (absorbElem state value).lanes =
      overwriteLane (permute state).lanes 0 value := by
  unfold absorbElem
  rw [dif_neg full]

theorem absorbElem_absorbed_of_full
    (state : State)
    (value : Field)
    (full : ¬ state.absorbed.val < rate) :
    (absorbElem state value).absorbed.val = 1 := by
  unfold absorbElem
  rw [dif_neg full]

def fullBuffer
    (initial : State)
    (a b c d : Field) : State where
  lanes :=
    overwriteLane
      (overwriteLane
        (overwriteLane
          (overwriteLane initial.lanes 0 a)
          1 b)
        2 c)
      3 d
  absorbed := ⟨rate, Nat.lt_succ_self rate⟩

def threeBuffer
    (initial : State)
    (a b c : Field) : State where
  lanes :=
    overwriteLane
      (overwriteLane
        (overwriteLane initial.lanes 0 a)
        1 b)
      2 c
  absorbed := ⟨3, by decide⟩

def cursorOneFullBuffer
    (initial : State)
    (a b c : Field) : State where
  lanes :=
    overwriteLane
      (overwriteLane
        (overwriteLane initial.lanes 1 a)
        2 b)
      3 c
  absorbed := ⟨rate, Nat.lt_succ_self rate⟩

theorem absorbAll_four_of_cursorZero
    (initial : State)
    (cursorZero : initial.absorbed.val = 0)
    (a b c d : Field) :
    absorbAll initial [a, b, c, d] =
      fullBuffer initial a b c d := by
  have room0 : initial.absorbed.val < rate := by
    rw [cursorZero]
    decide
  have cursor1 : (absorbElem initial a).absorbed.val = 1 := by
    rw [absorbElem_absorbed_of_room initial a room0]
    rw [cursorZero]
  have room1 : (absorbElem initial a).absorbed.val < rate := by
    rw [cursor1]
    decide
  have cursor2 :
      (absorbElem (absorbElem initial a) b).absorbed.val = 2 := by
    rw [absorbElem_absorbed_of_room (absorbElem initial a) b room1]
    rw [cursor1]
  have room2 :
      (absorbElem (absorbElem initial a) b).absorbed.val < rate := by
    rw [cursor2]
    decide
  have cursor3 :
      (absorbElem
        (absorbElem (absorbElem initial a) b) c).absorbed.val = 3 := by
    rw [absorbElem_absorbed_of_room
      (absorbElem (absorbElem initial a) b) c room2]
    rw [cursor2]
  have room3 :
      (absorbElem
        (absorbElem (absorbElem initial a) b) c).absorbed.val < rate := by
    rw [cursor3]
    decide
  apply stateExt
  · funext lane
    simp only [absorbAll]
    rw [absorbElem_lanes_of_room
      (absorbElem (absorbElem (absorbElem initial a) b) c) d room3]
    rw [absorbElem_lanes_of_room
      (absorbElem (absorbElem initial a) b) c room2]
    rw [absorbElem_lanes_of_room (absorbElem initial a) b room1]
    rw [absorbElem_lanes_of_room initial a room0]
    rw [cursorZero, cursor1, cursor2, cursor3]
    rfl
  · apply Fin.ext
    simp only [absorbAll]
    rw [absorbElem_absorbed_of_room
      (absorbElem (absorbElem (absorbElem initial a) b) c) d room3]
    rw [absorbElem_absorbed_of_room
      (absorbElem (absorbElem initial a) b) c room2]
    rw [absorbElem_absorbed_of_room (absorbElem initial a) b room1]
    rw [absorbElem_absorbed_of_room initial a room0]
    change initial.absorbed.val + 1 + 1 + 1 + 1 = rate
    rw [cursorZero]
    decide

theorem absorbAll_three_of_cursorZero
    (initial : State)
    (cursorZero : initial.absorbed.val = 0)
    (a b c : Field) :
    absorbAll initial [a, b, c] =
      threeBuffer initial a b c := by
  have room0 : initial.absorbed.val < rate := by
    rw [cursorZero]
    decide
  have cursor1 : (absorbElem initial a).absorbed.val = 1 := by
    rw [absorbElem_absorbed_of_room initial a room0]
    rw [cursorZero]
  have room1 : (absorbElem initial a).absorbed.val < rate := by
    rw [cursor1]
    decide
  have cursor2 :
      (absorbElem (absorbElem initial a) b).absorbed.val = 2 := by
    rw [absorbElem_absorbed_of_room (absorbElem initial a) b room1]
    rw [cursor1]
  have room2 :
      (absorbElem (absorbElem initial a) b).absorbed.val < rate := by
    rw [cursor2]
    decide
  apply stateExt
  · funext lane
    simp only [absorbAll]
    rw [absorbElem_lanes_of_room
      (absorbElem (absorbElem initial a) b) c room2]
    rw [absorbElem_lanes_of_room (absorbElem initial a) b room1]
    rw [absorbElem_lanes_of_room initial a room0]
    rw [cursorZero, cursor1, cursor2]
    rfl
  · apply Fin.ext
    simp only [absorbAll]
    rw [absorbElem_absorbed_of_room
      (absorbElem (absorbElem initial a) b) c room2]
    rw [absorbElem_absorbed_of_room (absorbElem initial a) b room1]
    rw [absorbElem_absorbed_of_room initial a room0]
    change initial.absorbed.val + 1 + 1 + 1 = 3
    rw [cursorZero]

/-- Three words from cursor one fill lanes one through three while retaining
lane zero from the NC prologue. -/
theorem absorbAll_three_of_cursorOne
    (initial : State)
    (cursorOne : initial.absorbed.val = 1)
    (a b c : Field) :
    absorbAll initial [a, b, c] =
      cursorOneFullBuffer initial a b c := by
  have room1 : initial.absorbed.val < rate := by
    rw [cursorOne]
    decide
  have cursor2 : (absorbElem initial a).absorbed.val = 2 := by
    rw [absorbElem_absorbed_of_room initial a room1]
    rw [cursorOne]
  have room2 : (absorbElem initial a).absorbed.val < rate := by
    rw [cursor2]
    decide
  have cursor3 :
      (absorbElem (absorbElem initial a) b).absorbed.val = 3 := by
    rw [absorbElem_absorbed_of_room (absorbElem initial a) b room2]
    rw [cursor2]
  have room3 :
      (absorbElem (absorbElem initial a) b).absorbed.val < rate := by
    rw [cursor3]
    decide
  apply stateExt
  · funext lane
    simp only [absorbAll]
    rw [absorbElem_lanes_of_room
      (absorbElem (absorbElem initial a) b) c room3]
    rw [absorbElem_lanes_of_room (absorbElem initial a) b room2]
    rw [absorbElem_lanes_of_room initial a room1]
    rw [cursorOne, cursor2, cursor3]
    rfl
  · apply Fin.ext
    simp only [absorbAll]
    rw [absorbElem_absorbed_of_room
      (absorbElem (absorbElem initial a) b) c room3]
    rw [absorbElem_absorbed_of_room (absorbElem initial a) b room2]
    rw [absorbElem_absorbed_of_room initial a room1]
    change initial.absorbed.val + 1 + 1 + 1 = rate
    rw [cursorOne]
    rfl

/-- Three words fill the remaining rate slots from the cursor-one state
produced by the NC prologue. -/
theorem absorbAll_three_of_cursorOne_cursorFull
    (initial : State)
    (cursorOne : initial.absorbed.val = 1)
    (a b c : Field) :
    (absorbAll initial [a, b, c]).absorbed.val = rate := by
  rw [absorbAll_three_of_cursorOne initial cursorOne]
  rfl

theorem absorbAll_cons_of_full
    (initial : State)
    (cursorFull : initial.absorbed.val = rate)
    (value : Field)
    (rest : List Field) :
    absorbAll initial (value :: rest) =
      absorbAll (permute initial) (value :: rest) := by
  have noRoom : ¬ initial.absorbed.val < rate := by
    omega
  have permuteRoom : (permute initial).absorbed.val < rate := by
    simp [permute, rate]
  apply congrArg (fun state => absorbAll state rest)
  apply stateExt
  · rw [absorbElem_lanes_of_full initial value noRoom]
    rw [absorbElem_lanes_of_room (permute initial) value permuteRoom]
    simp [permute]
  · apply Fin.ext
    rw [absorbElem_absorbed_of_full initial value noRoom]
    rw [absorbElem_absorbed_of_room
      (permute initial) value permuteRoom]
    simp [permute]

/-- Two one-field raw messages from cursor zero contribute
`[1,a,1,b]`, fill one rate block, and execute native full-buffer
normalization. -/
theorem appendRaw_singletons_of_cursorZero
    (initial : State)
    (cursorZero : initial.absorbed.val = 0)
    (a b : Field) :
    appendRaw (appendRaw initial [a]) [b] =
      permute
        (absorbAll initial [wordField 1, a, wordField 1, b]) := by
  have firstCursor :
      (absorbAll initial [wordField 1, a]).absorbed.val = 2 := by
    simp [absorbAll, absorbElem, cursorZero, rate]
  have firstNotFull :
      (absorbAll initial [wordField 1, a]).absorbed.val ≠ rate := by
    rw [firstCursor]
    decide
  have secondCursor :
      (absorbAll
        (absorbAll initial [wordField 1, a])
        [wordField 1, b]).absorbed.val = rate := by
    have room2 :
        (absorbAll initial [wordField 1, a]).absorbed.val < rate := by
      rw [firstCursor]
      decide
    have cursor3 :
        (absorbElem
          (absorbAll initial [wordField 1, a])
          (wordField 1)).absorbed.val = 3 := by
      rw [absorbElem_absorbed_of_room _ _ room2]
      rw [firstCursor]
    have room3 :
        (absorbElem
          (absorbAll initial [wordField 1, a])
          (wordField 1)).absorbed.val < rate := by
      rw [cursor3]
      decide
    change
      (absorbElem
        (absorbElem
          (absorbAll initial [wordField 1, a])
          (wordField 1))
        b).absorbed.val = rate
    rw [absorbElem_absorbed_of_room _ _ room3]
    rw [cursor3]
    rfl
  unfold appendRaw appendRawLazy
  simp only [List.length_cons, List.length_nil, Nat.reduceAdd]
  rw [show
    normalizeFull (absorbAll initial [wordField 1, a]) =
      absorbAll initial [wordField 1, a] by
    unfold normalizeFull
    rw [if_neg firstNotFull]]
  rw [show
    normalizeFull
        (absorbAll
          (absorbAll initial [wordField 1, a])
          [wordField 1, b]) =
      permute
        (absorbAll
          (absorbAll initial [wordField 1, a])
          [wordField 1, b]) by
    unfold normalizeFull
    rw [if_pos secondCursor]]
  rw [show
    [wordField 1, a, wordField 1, b] =
      [wordField 1, a] ++ [wordField 1, b] by rfl]
  rw [absorbAll_append]

/-- A two-field raw message followed by a singleton raw message from cursor
zero contributes `[2,a,b,1]`, executes one full permutation, and retains the
singleton payload at cursor one. -/
theorem appendRaw_pair_then_singleton_of_cursorZero
    (initial : State)
    (cursorZero : initial.absorbed.val = 0)
    (a b c : Field) :
    appendRaw (appendRaw initial [a, b]) [c] =
      absorbElem
        (permute
          (absorbAll initial [wordField 2, a, b, wordField 1]))
        c := by
  have firstCursor :
      (absorbAll initial [wordField 2, a, b]).absorbed.val = 3 := by
    rw [absorbAll_three_of_cursorZero initial cursorZero]
    rfl
  have firstNotFull :
      (absorbAll initial [wordField 2, a, b]).absorbed.val ≠ rate := by
    rw [firstCursor]
    decide
  have fullCursor :
      (absorbAll
        (absorbAll initial [wordField 2, a, b])
        [wordField 1]).absorbed.val = rate := by
    have room3 :
        (absorbAll initial [wordField 2, a, b]).absorbed.val < rate := by
      rw [firstCursor]
      decide
    change
      (absorbElem
        (absorbAll initial [wordField 2, a, b])
        (wordField 1)).absorbed.val = rate
    rw [absorbElem_absorbed_of_room _ _ room3]
    rw [firstCursor]
    rfl
  have crossed :
      absorbAll
          (absorbAll initial [wordField 2, a, b])
          [wordField 1, c] =
        absorbElem
          (permute
            (absorbAll initial [wordField 2, a, b, wordField 1]))
          c := by
    rw [show
      [wordField 1, c] = [wordField 1] ++ [c] by rfl]
    rw [absorbAll_append]
    rw [show
      [wordField 2, a, b, wordField 1] =
        [wordField 2, a, b] ++ [wordField 1] by rfl]
    rw [absorbAll_append]
    change
      absorbAll
          (absorbAll
            (absorbAll initial [wordField 2, a, b])
            [wordField 1])
          [c] =
        absorbElem
          (permute
            (absorbAll
              (absorbAll initial [wordField 2, a, b])
              [wordField 1]))
          c
    simpa only [absorbAll] using
      absorbAll_cons_of_full
        (absorbAll
          (absorbAll initial [wordField 2, a, b])
          [wordField 1])
        fullCursor c []
  have finalCursor :
      (absorbElem
        (permute
          (absorbAll initial [wordField 2, a, b, wordField 1]))
        c).absorbed.val = 1 := by
    have room :
        (permute
          (absorbAll initial
            [wordField 2, a, b, wordField 1])).absorbed.val < rate := by
      simp [permute, rate]
    rw [absorbElem_absorbed_of_room _ c room]
    rfl
  unfold appendRaw appendRawLazy
  simp only [List.length_cons, List.length_nil, Nat.reduceAdd]
  rw [show
    normalizeFull (absorbAll initial [wordField 2, a, b]) =
      absorbAll initial [wordField 2, a, b] by
    unfold normalizeFull
    rw [if_neg firstNotFull]]
  rw [crossed]
  unfold normalizeFull
  rw [if_neg (by rw [finalCursor]; decide)]

/-- Exact ten-field raw-message schedule from a cursor-zero state. -/
theorem appendRaw_ten_of_cursorZero
    (initial : State)
    (cursorZero : initial.absorbed.val = 0)
    (f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 : Field) :
    appendRaw initial [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] =
      absorbAll
        (permute
          (absorbAll
            (permute
              (absorbAll initial [wordField 10, f0, f1, f2]))
            [f3, f4, f5, f6]))
        [f7, f8, f9] := by
  have firstFull :
      (absorbAll initial
        [wordField 10, f0, f1, f2]).absorbed.val = rate := by
    rw [absorbAll_four_of_cursorZero initial cursorZero]
    rfl
  have secondFull :
      (absorbAll
        (permute
          (absorbAll initial [wordField 10, f0, f1, f2]))
        [f3, f4, f5, f6]).absorbed.val = rate := by
    rw [absorbAll_four_of_cursorZero
      (permute
        (absorbAll initial [wordField 10, f0, f1, f2]))
      (by rfl)]
    rfl
  have thirdCursor :
      (absorbAll
        (permute
          (absorbAll
            (permute
              (absorbAll initial [wordField 10, f0, f1, f2]))
            [f3, f4, f5, f6]))
        [f7, f8, f9]).absorbed.val = 3 := by
    rw [absorbAll_three_of_cursorZero
      (permute
        (absorbAll
          (permute
            (absorbAll initial [wordField 10, f0, f1, f2]))
          [f3, f4, f5, f6]))
      (by rfl)]
    rfl
  have thirdNotFull :
      (absorbAll
        (permute
          (absorbAll
            (permute
              (absorbAll initial [wordField 10, f0, f1, f2]))
            [f3, f4, f5, f6]))
        [f7, f8, f9]).absorbed.val ≠ rate := by
    rw [thirdCursor]
    decide
  unfold appendRaw appendRawLazy
  change
    normalizeFull
        (absorbAll initial
          ([wordField 10, f0, f1, f2] ++
            ([f3, f4, f5, f6] ++ [f7, f8, f9]))) =
      absorbAll
        (permute
          (absorbAll
            (permute
              (absorbAll initial [wordField 10, f0, f1, f2]))
            [f3, f4, f5, f6]))
        [f7, f8, f9]
  rw [absorbAll_append]
  have crossFirst :=
    absorbAll_cons_of_full
      (absorbAll initial [wordField 10, f0, f1, f2])
      firstFull f3
      ([f4, f5, f6] ++ [f7, f8, f9])
  rw [show
    absorbAll
        (absorbAll initial [wordField 10, f0, f1, f2])
        ([f3, f4, f5, f6] ++ [f7, f8, f9]) =
      absorbAll
        (permute
          (absorbAll initial [wordField 10, f0, f1, f2]))
        ([f3, f4, f5, f6] ++ [f7, f8, f9]) by
    simpa only [List.cons_append] using crossFirst]
  rw [absorbAll_append]
  rw [absorbAll_cons_of_full
    (absorbAll
      (permute
        (absorbAll initial [wordField 10, f0, f1, f2]))
      [f3, f4, f5, f6])
    secondFull]
  unfold normalizeFull
  rw [if_neg thirdNotFull]

/-- Exact ten-field raw-message schedule from the cursor-one state left by
the NC prologue. The length word and first two fields complete the first
rate block; the remaining eight fields form two further complete blocks,
and native full-buffer normalization executes the third permutation. -/
theorem appendRaw_ten_of_cursorOne
    (initial : State)
    (cursorOne : initial.absorbed.val = 1)
    (f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 : Field) :
    appendRaw initial [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] =
      permute
        (absorbAll
          (permute
            (absorbAll
              (permute
                (absorbAll initial [wordField 10, f0, f1]))
              [f2, f3, f4, f5]))
          [f6, f7, f8, f9]) := by
  have firstFull :
      (absorbAll initial [wordField 10, f0, f1]).absorbed.val =
        rate :=
    absorbAll_three_of_cursorOne_cursorFull initial cursorOne
      (wordField 10) f0 f1
  have secondFull :
      (absorbAll
        (permute
          (absorbAll initial [wordField 10, f0, f1]))
        [f2, f3, f4, f5]).absorbed.val = rate := by
    rw [absorbAll_four_of_cursorZero
      (permute
        (absorbAll initial [wordField 10, f0, f1]))
      (by rfl)]
    rfl
  have thirdFull :
      (absorbAll
        (permute
          (absorbAll
            (permute
              (absorbAll initial [wordField 10, f0, f1]))
            [f2, f3, f4, f5]))
        [f6, f7, f8, f9]).absorbed.val = rate := by
    rw [absorbAll_four_of_cursorZero
      (permute
        (absorbAll
          (permute
            (absorbAll initial [wordField 10, f0, f1]))
          [f2, f3, f4, f5]))
      (by rfl)]
    rfl
  unfold appendRaw appendRawLazy
  change
    normalizeFull
        (absorbAll initial
          ([wordField 10, f0, f1] ++
            ([f2, f3, f4, f5] ++ [f6, f7, f8, f9]))) =
      permute
        (absorbAll
          (permute
            (absorbAll
              (permute
                (absorbAll initial [wordField 10, f0, f1]))
              [f2, f3, f4, f5]))
          [f6, f7, f8, f9])
  rw [absorbAll_append]
  have crossFirst :=
    absorbAll_cons_of_full
      (absorbAll initial [wordField 10, f0, f1])
      firstFull f2
      ([f3, f4, f5] ++ [f6, f7, f8, f9])
  rw [show
    absorbAll
        (absorbAll initial [wordField 10, f0, f1])
        ([f2, f3, f4, f5] ++ [f6, f7, f8, f9]) =
      absorbAll
        (permute
          (absorbAll initial [wordField 10, f0, f1]))
        ([f2, f3, f4, f5] ++ [f6, f7, f8, f9]) by
    simpa only [List.cons_append] using crossFirst]
  rw [absorbAll_append]
  rw [absorbAll_cons_of_full
    (absorbAll
      (permute
        (absorbAll initial [wordField 10, f0, f1]))
      [f2, f3, f4, f5])
    secondFull]
  unfold normalizeFull
  rw [if_pos thirdFull]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution
