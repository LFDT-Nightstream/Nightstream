import Nightstream.Implementation.R1CS.Canonical.KLowNorm
import Nightstream.Implementation.R1CS.Canonical.KMulHonest
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: the low-norm check over many digits at once.

Owns: the batched row program, its folded row count, its allocation and the
non-collision of that allocation, batched soundness, and a single honest witness
that satisfies every digit's rows simultaneously.

Does not own: the per-digit rows, their cube identity, or the centered-window
argument — those are `KLowNorm`, and this module consumes them.

## Why the batch is not just a `flatMap`

Rows batch trivially. The **witness** does not.

`KLowNorm.lowNormWitness` writes one square to one allocated column, and each
digit needs its own. Composing those writes is only sound when two conditions
hold, and both are hypotheses here rather than assumptions:

- **Freshness across the batch.** No checked combination may read *any* square
  column, not merely its own. A combination that reads a later digit's column
  would have its value changed by that digit's write, and its own rows need not
  survive.
- **Distinct columns.** If two digits shared a square column, the later write
  would clobber the earlier square and the earlier digit's rows would fail.

`Π_DEC` decomposes every entry of the public `X` matrix into `k_rho = 14`
balanced base-2 digits, so this batch is the check's dominant cost and the
non-collision is the load-bearing part of its column ownership.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KLowNormBatch

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-- One checked digit: a combination and the column its square is written to. -/
structure Digit where
  value : LinComb
  squareColumn : Nat

/-- **The emitted batch.** -/
def batchRows (digits : List Digit) : List Row :=
  digits.flatMap (fun digit => KLowNorm.lowNormRows digit.value digit.squareColumn)

/-- **The derived row count, as a fold over per-digit receipts.** -/
theorem batchRows_length (digits : List Digit) :
    (batchRows digits).length = (digits.map (fun _ => 2)).sum := by
  unfold batchRows
  rw [List.length_flatMap]
  exact congrArg List.sum
    (List.map_congr_left (fun digit _ =>
      KLowNorm.lowNormRows_length digit.value digit.squareColumn))

theorem batchRows_length_eq (digits : List Digit) :
    (batchRows digits).length = 2 * digits.length := by
  rw [batchRows_length]
  induction digits with
  | nil => rfl
  | cons digit rest inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis]
      omega

/-! ## Allocation

One column per digit, and they must not collide. -/

def batchColumns (digits : List Digit) : List Nat :=
  digits.map Digit.squareColumn

/-- **Every declared column is used by some emitted row.**

The batch form of `KLowNorm.lowNormRows_use_squareColumn`.  With
`batchRows_conservation` this pins the allocation from both sides: no emitted row
reaches outside the declared columns, and no declared column is unreached. -/
theorem batchRows_use_columns
    (digits : List Digit) (column : Nat) (member : column ∈ batchColumns digits) :
    ∃ row ∈ batchRows digits, Mentions row.c column := by
  unfold batchColumns at member
  rcases List.mem_map.1 member with ⟨digit, digitMember, rfl⟩
  rcases KLowNorm.lowNormRows_use_squareColumn digit.value digit.squareColumn
    with ⟨row, rowMember, mentions⟩
  exact ⟨row, List.mem_flatMap.2 ⟨digit, digitMember, rowMember⟩, mentions⟩

theorem batchColumns_length (digits : List Digit) :
    (batchColumns digits).length = digits.length :=
  List.length_map _

/-- **The batch is well formed** when its allocation does not collide and no
checked combination reads any allocated column. Both arms are used by honest
completeness, and neither is decorative. -/
structure WellFormed (digits : List Digit) : Prop where
  distinct : (batchColumns digits).Nodup
  fresh : ∀ digit ∈ digits, ∀ other ∈ digits,
    ¬ Mentions digit.value other.squareColumn

theorem WellFormed.tail {digit : Digit} {rest : List Digit}
    (wellFormed : WellFormed (digit :: rest)) : WellFormed rest where
  distinct := by
    have := wellFormed.distinct
    unfold batchColumns at this ⊢
    rw [List.map_cons, List.nodup_cons] at this
    exact this.2
  fresh := fun first firstMember second secondMember =>
    wellFormed.fresh first (List.mem_cons_of_mem digit firstMember)
      second (List.mem_cons_of_mem digit secondMember)

/-! ## Soundness -/

/-- **Satisfaction forces the cube identity on every digit.** -/
theorem batchRows_sound
    (z : Nat → Nat) (digits : List Digit)
    (satisfied : Satisfies (batchRows digits) z)
    (digit : Digit) (member : digit ∈ digits) :
    lcEval z digit.value * lcEval z digit.value % goldilocksP
        * lcEval z digit.value % goldilocksP
      = lcEval z digit.value :=
  KLowNorm.lowNormRows_sound z digit.value digit.squareColumn
    (fun row rowMember =>
      satisfied row (List.mem_flatMap.2 ⟨digit, member, rowMember⟩))

/-! ## The batched honest witness

One assignment that satisfies every digit's rows.  Built by folding the
per-digit writes, which is only correct under `WellFormed`. -/

/-- **The batched witness.**  Each digit's square, on its own column. -/
def batchWitness (z : Nat → Nat) : List Digit → (Nat → Nat)
  | [] => z
  | digit :: rest =>
      KLowNorm.lowNormWitness (batchWitness z rest) digit.value digit.squareColumn

theorem batchWitness_nil (z : Nat → Nat) : batchWitness z [] = z := rfl

/-- **Nothing outside the allocation moves.** -/
theorem batchWitness_off_columns
    (z : Nat → Nat) (digits : List Digit) (column : Nat)
    (off : ∀ digit ∈ digits, column ≠ digit.squareColumn) :
    batchWitness z digits column = z column := by
  induction digits with
  | nil => rfl
  | cons head rest inductionHypothesis =>
      rw [show batchWitness z (head :: rest)
          = KLowNorm.lowNormWitness (batchWitness z rest) head.value
              head.squareColumn from rfl,
        KLowNorm.lowNormWitness_off_column _ head.value head.squareColumn column
          (off head List.mem_cons_self)]
      exact inductionHypothesis
        (fun digit member => off digit (List.mem_cons_of_mem head member))

/-- **The batch leaves every checked combination alone.**

This is where cross-digit freshness earns its place: no digit's write may
disturb any other digit's value. -/
theorem lcEval_batchWitness
    (z : Nat → Nat) (digits : List Digit) (wellFormed : WellFormed digits)
    (digit : Digit) (member : digit ∈ digits) :
    lcEval (batchWitness z digits) digit.value = lcEval z digit.value := by
  refine KMulHonest.lcEval_congr _ z digit.value (fun column mentioned => ?_)
  refine batchWitness_off_columns z digits column
    (fun other otherMember equal => ?_)
  exact wellFormed.fresh digit member other otherMember (equal ▸ mentioned)

/-- **Each allocated column carries its own digit's square.**

This is where non-collision earns its place: if two digits shared a column the
outer write would clobber the inner one and the inner digit's rows would
fail. -/
theorem batchWitness_at_column
    (z : Nat → Nat) (digits : List Digit) (wellFormed : WellFormed digits)
    (digit : Digit) (member : digit ∈ digits) :
    batchWitness z digits digit.squareColumn
      = lcEval z digit.value * lcEval z digit.value % goldilocksP := by
  induction digits with
  | nil => cases member
  | cons head rest inductionHypothesis =>
      have restWellFormed := wellFormed.tail
      have unfoldStep : batchWitness z (head :: rest)
          = KLowNorm.lowNormWitness (batchWitness z rest) head.value
              head.squareColumn := rfl
      rcases List.mem_cons.1 member with rfl | inRest
      · have headValue : lcEval (batchWitness z rest) digit.value
            = lcEval z digit.value := by
          refine KMulHonest.lcEval_congr _ z digit.value
            (fun column mentioned => ?_)
          refine batchWitness_off_columns z rest column
            (fun other otherMember equal => ?_)
          exact wellFormed.fresh digit member other
            (List.mem_cons_of_mem digit otherMember) (equal ▸ mentioned)
        rw [unfoldStep]
        unfold KLowNorm.lowNormWitness
        rw [if_pos rfl, headValue]
      · have distinctColumn : digit.squareColumn ≠ head.squareColumn := by
          have nodup := wellFormed.distinct
          unfold batchColumns at nodup
          rw [List.map_cons, List.nodup_cons] at nodup
          intro equal
          exact nodup.1 (equal ▸ List.mem_map.2 ⟨digit, inRest, rfl⟩)
        rw [unfoldStep, KLowNorm.lowNormWitness_off_column _ head.value
          head.squareColumn digit.squareColumn distinctColumn]
        exact inductionHypothesis restWellFormed inRest

/-! ## Honest completeness

One witness for the whole batch.  Both `WellFormed` arms are consumed here, and
the cube hypothesis is exactly `batchRows_sound`'s conclusion, so the batch is
complete for precisely the digits it accepts. -/

/-- **Every in-window digit is satisfied by the one batched witness.** -/
theorem batchRows_honest
    (z : Nat → Nat) (digits : List Digit) (wellFormed : WellFormed digits)
    (cubes : ∀ digit ∈ digits,
      lcEval z digit.value * lcEval z digit.value % goldilocksP
          * lcEval z digit.value % goldilocksP
        = lcEval z digit.value) :
    Satisfies (batchRows digits) (batchWitness z digits) := by
  intro row member
  rcases List.mem_flatMap.1 member with ⟨digit, digitMember, rowMember⟩
  have value := lcEval_batchWitness z digits wellFormed digit digitMember
  have square := batchWitness_at_column z digits wellFormed digit digitMember
  have columnEval :
      lcEval (batchWitness z digits) [(digit.squareColumn, 1)]
        = lcEval z digit.value * lcEval z digit.value % goldilocksP := by
    rw [KMul.lcEval_singleton_col, square, Nat.mod_mod]
  simp only [KLowNorm.lowNormRows, List.mem_cons, List.not_mem_nil,
    or_false] at rowMember
  rcases rowMember with rfl | rfl
  · unfold RowHolds
    simp only
    rw [value, columnEval]
  · unfold RowHolds
    simp only
    rw [value, columnEval]
    exact cubes digit digitMember

/-! ## Conservation -/

/-- **Every column is a checked combination's or an allocated square.** -/
theorem batchRows_conservation
    (digits : List Digit) (row : Row) (member : row ∈ batchRows digits)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    ∃ digit ∈ digits,
      Mentions digit.value column ∨ column = digit.squareColumn := by
  rcases List.mem_flatMap.1 member with ⟨digit, digitMember, rowMember⟩
  exact ⟨digit, digitMember,
    KLowNorm.lowNormRows_conservation digit.value digit.squareColumn row
      rowMember column mentioned⟩

/-! ## Cost -/

/-- **The batch's cost**, folded over digits.  One column per digit, which is
what the non-collision proof is about. -/
def batchCost (digits : List Digit) : Lowering.Typed.Cost where
  recurringRows := 2 * digits.length
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := digits.length

theorem batchCost_rows (digits : List Digit) :
    (batchRows digits).length = (batchCost digits).recurringRows :=
  batchRows_length_eq digits

theorem batchCost_columns (digits : List Digit) :
    (batchColumns digits).length = (batchCost digits).auxiliaryColumns :=
  batchColumns_length digits

/-! ## The canonical allocation

`WellFormed` has two arms and they are not the same kind of obligation.

`distinct` is about the **allocation**, which the encoder chooses.  `fresh` is
about what the checked **values** read, which it does not.  Leaving both to the
caller — as "supply a well-formed batch" does — overstates what a deployment has
to establish.

`canonicalDigits` picks the allocation: digit `i` takes column `base + i + 1`.
That discharges `distinct` outright and supplies the width bound
`ColumnWindows.placeAll_columns` needs, leaving exactly one caller obligation
instead of three.

The remaining one is genuinely the caller's.  No numbering can make a value stop
reading a column; only knowing where the values live can. -/

/-- **The canonical allocation.**  Consecutive columns from `base + 1`. -/
def canonicalDigits (base : Nat) : List LinComb → List Digit
  | [] => []
  | value :: rest => ⟨value, base + 1⟩ :: canonicalDigits (base + 1) rest

theorem canonicalDigits_length (base : Nat) (values : List LinComb) :
    (canonicalDigits base values).length = values.length := by
  induction values generalizing base with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp only [canonicalDigits, List.length_cons, inductionHypothesis]

/-- Every canonical column lies strictly above the base. -/
theorem canonicalDigits_column_gt (base : Nat) (values : List LinComb) :
    ∀ digit ∈ canonicalDigits base values, base < digit.squareColumn := by
  induction values generalizing base with
  | nil => intro digit member; cases member
  | cons value rest inductionHypothesis =>
      intro digit member
      rcases List.mem_cons.1 member with rfl | inTail
      · exact Nat.lt_succ_self base
      · exact Nat.lt_trans (Nat.lt_succ_self base)
          (inductionHypothesis (base + 1) digit inTail)

/-- **The canonical columns fit the allocation's width.**  This is the bound
`ColumnWindows.placeAll_columns` asks for, on the allocated side. -/
theorem canonicalDigits_column_le (base : Nat) (values : List LinComb) :
    ∀ digit ∈ canonicalDigits base values,
      digit.squareColumn ≤ base + values.length := by
  induction values generalizing base with
  | nil => intro digit member; cases member
  | cons value rest inductionHypothesis =>
      intro digit member
      rcases List.mem_cons.1 member with rfl | inTail
      · exact Nat.add_le_add_left (Nat.succ_le_succ (Nat.zero_le _)) base
      · have := inductionHypothesis (base + 1) digit inTail
        simp only [List.length_cons]
        omega

/-- **The canonical allocation never collides.**

`WellFormed.distinct` for free: consecutive columns are distinct because each
tail column is strictly above the head's. -/
theorem canonicalDigits_nodup (base : Nat) (values : List LinComb) :
    (batchColumns (canonicalDigits base values)).Nodup := by
  induction values generalizing base with
  | nil => exact List.nodup_nil
  | cons value rest inductionHypothesis =>
      unfold batchColumns canonicalDigits
      rw [List.map_cons, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_map.1 member with ⟨digit, digitMember, columnEq⟩
        have above := canonicalDigits_column_gt (base + 1) rest digit digitMember
        have headColumn :
            ({ value := value, squareColumn := base + 1 } : Digit).squareColumn
              = base + 1 := rfl
        rw [headColumn] at columnEq
        omega
      · exact inductionHypothesis (base + 1)

/-- **One caller obligation, not three.**

Given only that no checked value reads an allocated column, the canonical
allocation is well formed.  `distinct` is discharged; `fresh` is what remains,
and it is about the values rather than the numbering. -/
theorem canonicalDigits_wellFormed
    (base : Nat) (values : List LinComb)
    (fresh : ∀ digit ∈ canonicalDigits base values,
      ∀ other ∈ canonicalDigits base values,
        ¬ Mentions digit.value other.squareColumn) :
    WellFormed (canonicalDigits base values) where
  distinct := canonicalDigits_nodup base values
  fresh := fresh

/-! ## Row ownership

Section 2 item 3: every emitted row belongs to exactly **one** receipt.

Existence is `List.mem_flatMap` and was always available.  Uniqueness is the
content, and it is where the allocation earns its second keep: a row carries its
digit's square column in a field position, so the column is recoverable from the
row, and `distinct` then recovers the digit.

Note what this does *not* need: the values.  Two digits with identical
combinations are still distinguishable, because the allocation distinguishes
them.  That is why the argument works for a batch whose values a caller
supplies. -/

/-- Both emitted rows expose the allocated column, so a row shared between two
digits forces their columns equal. -/
theorem lowNormRows_determines_column
    (firstValue secondValue : LinComb) (firstColumn secondColumn : Nat)
    (row : Row)
    (inFirst : row ∈ KLowNorm.lowNormRows firstValue firstColumn)
    (inSecond : row ∈ KLowNorm.lowNormRows secondValue secondColumn) :
    firstColumn = secondColumn := by
  simp only [KLowNorm.lowNormRows, List.mem_cons, List.not_mem_nil,
    or_false] at inFirst inSecond
  rcases inFirst with firstEq | firstEq <;> rcases inSecond with secondEq | secondEq <;>
    rw [firstEq] at secondEq <;> injection secondEq with aEq bEq cEq <;>
    · have extract : ∀ {left right : Nat},
          ([(left, 1)] : LinComb) = [(right, 1)] → left = right := by
        intro left right equal
        simp only [List.cons.injEq, Prod.mk.injEq] at equal
        exact equal.1.1
      first
        | exact extract cEq
        | exact extract aEq
        | exact extract (aEq.trans (bEq.symm.trans cEq))
        | exact extract (cEq.trans (bEq.symm.trans aEq))

/-- A `Nodup` allocation makes the digit recoverable from its column. -/
theorem column_determines_digit
    (digits : List Digit) (wellFormed : WellFormed digits)
    (first second : Digit)
    (firstMember : first ∈ digits) (secondMember : second ∈ digits)
    (sameColumn : first.squareColumn = second.squareColumn) :
    first = second := by
  have nodup := wellFormed.distinct
  unfold batchColumns at nodup
  induction digits with
  | nil => cases firstMember
  | cons head rest inductionHypothesis =>
      rw [List.map_cons, List.nodup_cons] at nodup
      rcases List.mem_cons.1 firstMember with rfl | firstTail
      · rcases List.mem_cons.1 secondMember with rfl | secondTail
        · rfl
        · refine absurd ?_ nodup.1
          rw [sameColumn]
          exact List.mem_map.2 ⟨second, secondTail, rfl⟩
      · rcases List.mem_cons.1 secondMember with rfl | secondTail
        · refine absurd ?_ nodup.1
          rw [← sameColumn]
          exact List.mem_map.2 ⟨first, firstTail, rfl⟩
        · exact inductionHypothesis
            { distinct := by unfold batchColumns; exact nodup.2
              fresh := fun a am b bm =>
                wellFormed.fresh a (List.mem_cons_of_mem head am) b
                  (List.mem_cons_of_mem head bm) }
            firstTail secondTail nodup.2

/-- **Every emitted row belongs to exactly one digit.**

Section 2 item 3 for this recipe: existence from `List.mem_flatMap`, uniqueness
from the allocation. -/
theorem batchRows_owner_unique
    (digits : List Digit) (wellFormed : WellFormed digits) (row : Row)
    (first second : Digit)
    (firstMember : first ∈ digits) (secondMember : second ∈ digits)
    (inFirst : row ∈ KLowNorm.lowNormRows first.value first.squareColumn)
    (inSecond : row ∈ KLowNorm.lowNormRows second.value second.squareColumn) :
    first = second :=
  column_determines_digit digits wellFormed first second firstMember
    secondMember
    (lowNormRows_determines_column first.value second.value first.squareColumn
      second.squareColumn row inFirst inSecond)

/-- **Existence and uniqueness together.** -/
theorem batchRows_owned
    (digits : List Digit) (wellFormed : WellFormed digits)
    (row : Row) (member : row ∈ batchRows digits) :
    ∃ digit, digit ∈ digits
      ∧ row ∈ KLowNorm.lowNormRows digit.value digit.squareColumn
      ∧ ∀ other ∈ digits,
          row ∈ KLowNorm.lowNormRows other.value other.squareColumn →
            other = digit := by
  rcases List.mem_flatMap.1 member with ⟨digit, digitMember, rowMember⟩
  exact ⟨digit, digitMember, rowMember, fun other otherMember inOther =>
    batchRows_owner_unique digits wellFormed row other digit otherMember
      digitMember inOther rowMember⟩

end Nightstream.Implementation.R1CS.Canonical.KLowNormBatch
