import Nightstream.Implementation.R1CS.Canonical.KMulOwnership

/-!
Contract: the two facts a Horner completeness induction needs, and neither of
which belongs to a single multiplication.

Owns: preservation of satisfaction under extension at untouched columns, and
the classification of every column a carried Horner value can mention.

Does **not** own, and does not prove: honest completeness for `KHorner`. These
are its ingredients. Naming them separately is deliberate — this project has
five recorded instances of ingredients being reported as an assembled result,
and the module boundary is the cheapest way to keep the distinction visible.

## Why the induction needs both

The witness for a Horner evaluation is built inside out: the deepest step's
frame is written first, then each enclosing step extends it. Two things must
hold at every extension.

**The already-satisfied inner rows must stay satisfied.** The enclosing step
writes only its own frame, so this is `satisfies_extend` plus the fact that the
inner rows never mention that frame.

**The enclosing multiplication's operands must be fresh for its frame.** One
operand is the carried suffix, whose columns are whatever
`hornerCarried_mentions` says — the coefficients, and the frames of strictly
later steps. With the canonical allocator's ordering, none of those is the
current step's frame.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KHornerSupport

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-! ## Preservation -/

/-- **Satisfaction survives an extension that misses every referenced column.**
The reusable core of any inside-out witness construction. -/
theorem satisfies_extend
    (program : List Row) (z z' : Nat → Nat)
    (agree : ∀ row ∈ program, ∀ column,
      (Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column)
        → z column = z' column)
    (satisfied : Satisfies program z) :
    Satisfies program z' := by
  intro row member
  have aEq : lcEval z row.a = lcEval z' row.a :=
    KMulHonest.lcEval_congr z z' row.a
      (fun column mentioned => agree row member column (Or.inl mentioned))
  have bEq : lcEval z row.b = lcEval z' row.b :=
    KMulHonest.lcEval_congr z z' row.b
      (fun column mentioned => agree row member column (Or.inr (Or.inl mentioned)))
  have cEq : lcEval z row.c = lcEval z' row.c :=
    KMulHonest.lcEval_congr z z' row.c
      (fun column mentioned =>
        agree row member column (Or.inr (Or.inr mentioned)))
  have holds := satisfied row member
  unfold RowHolds at holds ⊢
  rw [← aEq, ← bEq, ← cEq]
  exact holds

/-! ## What a carried Horner value can mention -/

/-- A column belonging to one of the frames this run actually uses: steps
`step` through `step + coefficients.length − 2`, since an `n`-coefficient
evaluation performs `n − 1` multiplications.

The **upper** bound is what makes conservation provable. Without it the witness
places a column somewhere at or after `step` with no ceiling, which cannot be
shown to lie inside a finite allocation. The lower bound alone is enough for the
freshness argument in `KHornerHonest`, which is why this was originally stated
one-sided. -/
def FrameOfRun (frames : Nat → Frame) (coefficients : List Carried)
    (step column : Nat) : Prop :=
  ∃ later, step ≤ later ∧ later + 1 < step + coefficients.length ∧
    (column = (frames later).lowLow ∨ column = (frames later).highHigh
      ∨ column = (frames later).cross)

/-- A column mentioned by one of the coefficient combinations. -/
def CoefficientColumn (coefficients : List Carried) (column : Nat) : Prop :=
  ∃ c ∈ coefficients, Mentions c.low column ∨ Mentions c.high column

/-- **The carried value reaches only coefficients and later frames.**  This is
what lets the allocator's ordering discharge freshness at every step: the
current step's frame is strictly earlier than any frame the suffix mentions. -/
theorem hornerCarried_mentions
    (beta : Carried) (frames : Nat → Frame) :
    ∀ (coefficients : List Carried) (step column : Nat),
      (Mentions (hornerCarried beta frames coefficients step).low column
        ∨ Mentions (hornerCarried beta frames coefficients step).high column) →
      CoefficientColumn coefficients column
        ∨ FrameOfRun frames coefficients step column
  | [], _, _, mentioned => by
      simp [hornerCarried, Mentions] at mentioned
  | [c], _, column, mentioned => by
      exact Or.inl ⟨c, by simp, mentioned⟩
  | c :: next :: rest, step, column, mentioned => by
      simp only [hornerCarried, Mentions, List.map_append,
        List.mem_append] at mentioned
      rcases mentioned with (inC | inFrame) | (inC | inFrame)
      · exact Or.inl ⟨c, by simp, Or.inl inC⟩
      · refine Or.inr ⟨step, Nat.le_refl _, by simp only [List.length_cons]; omega, ?_⟩
        simp only [outLow, Mentions, List.map_cons, List.map_nil,
          List.mem_cons, List.not_mem_nil, or_false] at inFrame
        rcases inFrame with h | h
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
      · exact Or.inl ⟨c, by simp, Or.inr inC⟩
      · refine Or.inr ⟨step, Nat.le_refl _, by simp only [List.length_cons]; omega, ?_⟩
        simp only [outHigh, Mentions, List.map_cons, List.map_nil,
          List.mem_cons, List.not_mem_nil, or_false] at inFrame
        rcases inFrame with h | h | h
        · exact Or.inr (Or.inr h)
        · exact Or.inl h
        · exact Or.inr (Or.inl h)

/-- **What an emitted evaluation row can mention.**  The row analogue of
`hornerCarried_mentions`, and what lets an enclosing step extend the witness
without disturbing the rows already satisfied. -/
theorem hornerRows_mentions
    (beta : Carried) (frames : Nat → Frame) :
    ∀ (coefficients : List Carried) (step : Nat) (row : Row),
      row ∈ hornerRows beta frames coefficients step →
      ∀ column, (Mentions row.a column ∨ Mentions row.b column
          ∨ Mentions row.c column) →
        (Mentions beta.low column ∨ Mentions beta.high column)
          ∨ CoefficientColumn coefficients column
          ∨ FrameOfRun frames coefficients step column
  | [], _, _, member, _, _ => by simp [hornerRows] at member
  | [_], _, _, member, _, _ => by simp [hornerRows] at member
  | c :: next :: rest, step, row, member, column, mentioned => by
      simp only [hornerRows, List.mem_append] at member
      rcases member with inMul | inTail
      · rcases KMulOwnership.rows_conservation beta
          (hornerCarried beta frames (next :: rest) (step + 1)) (frames step)
          row inMul column mentioned with operand | frameCol
        · rcases operand with bl | bh | sl | sh
          · exact Or.inl (Or.inl bl)
          · exact Or.inl (Or.inr bh)
          · rcases hornerCarried_mentions beta frames (next :: rest) (step + 1)
              column (Or.inl sl) with ⟨d, memberD, inD⟩ | ⟨later, ordered, bounded, inF⟩
            · exact Or.inr (Or.inl ⟨d, List.mem_cons_of_mem _ memberD, inD⟩)
            · exact Or.inr (Or.inr ⟨later, by omega,
                by simp only [List.length_cons] at bounded ⊢; omega, inF⟩)
          · rcases hornerCarried_mentions beta frames (next :: rest) (step + 1)
              column (Or.inr sh) with ⟨d, memberD, inD⟩ | ⟨later, ordered, bounded, inF⟩
            · exact Or.inr (Or.inl ⟨d, List.mem_cons_of_mem _ memberD, inD⟩)
            · exact Or.inr (Or.inr ⟨later, by omega,
                by simp only [List.length_cons] at bounded ⊢; omega, inF⟩)
        · exact Or.inr (Or.inr ⟨step, Nat.le_refl _,
            by simp only [List.length_cons]; omega, frameCol⟩)
      · rcases hornerRows_mentions beta frames (next :: rest) (step + 1) row
          inTail column mentioned with
            inBeta | ⟨d, memberD, inD⟩ | ⟨later, ordered, bounded, inF⟩
        · exact Or.inl inBeta
        · exact Or.inr (Or.inl ⟨d, List.mem_cons_of_mem _ memberD, inD⟩)
        · exact Or.inr (Or.inr ⟨later, by omega,
            by simp only [List.length_cons] at bounded ⊢; omega, inF⟩)

end Nightstream.Implementation.R1CS.Canonical.KHornerSupport
