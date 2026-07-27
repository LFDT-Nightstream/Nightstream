import Nightstream.Implementation.R1CS.Canonical.Poseidon2Support

/-!
Contract: the coefficient count of the canonical Poseidon2 program.

Owns: what "a coefficient" means for a never-materialize encoding, and the
decomposition of the program total into per-receipt contributions.

Does not own: row counts (`Poseidon2Program`), the support recurrence
(`Poseidon2Support`), or semantics.

## Why the count is over normalized rows

`applyMatrix` concatenates and never aggregates, so a carried combination's
*syntactic* length grows without bound across rounds.  Counting those entries
would measure the intermediate representation, not the encoding.  The
implementable form is `LinCombNormal.normalize`, which
`LinCombNormal.lcEval_normalize` proves semantics-preserving and
`LinCombNormal.normalize_nodup` proves has exactly one entry per referenced
column.  `rowTermCount` therefore counts normalized entries.

## Term count versus nonzero-coefficient count

These are not the same number and the difference is stated rather than hidden.
A normalized entry could still carry a coefficient that vanishes modulo the
prime, if the matrix products happen to cancel.  So

    nonzero coefficients  ≤  rowTermCount

with equality exactly when no cancellation occurs.  Both matrices are dense
(`externalMatrix_nonzero`, `internalMatrix_nonzero`), which rules out the
trivial source of zeros but not cancellation in a sum of products; proving
none occurs is a separate arithmetic obligation over the concrete constants,
recorded as `POSEIDON2-NO-CANCELLATION`.

Every singleton entry the encoding emits carries coefficient `1`, so the
uncertainty is confined to the scheduled input combinations —
`sboxRows_termCount` shows exactly `8` of each S-box's entries are singletons
and only `4` copies of the input combination are in question.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-! ## Counting -/

/-- Normalized entries a single row contributes across its three operands. -/
def rowTermCount (row : Row) : Nat :=
  (normalize row.a).length + (normalize row.b).length + (normalize row.c).length

def programTermCount (rows : List Row) : Nat := (rows.map rowTermCount).sum

theorem programTermCount_append (left right : List Row) :
    programTermCount (left ++ right)
      = programTermCount left + programTermCount right := by
  simp [programTermCount]

theorem programTermCount_flatMap {α : Type} (list : List α) (f : α → List Row) :
    programTermCount (list.flatMap f)
      = ((list.map (fun x => programTermCount (f x))).sum) := by
  induction list with
  | nil => simp [programTermCount]
  | cons head tail hypothesis =>
      rw [List.flatMap_cons, programTermCount_append, hypothesis]
      simp

/-- A singleton is already normal. -/
theorem normalize_singleton (column coefficient : Nat) :
    (normalize [(column, coefficient)]).length = 1 := rfl

/-! ## Per-receipt contributions -/

/-- **One S-box costs `3·|input| + 9`.**  Under production's `1 → 2 → 4 → 6 → 7`
chain the input combination appears in only three operand positions — both
operands of the squaring row and the multiplicand of the seventh-power row —
and the remaining nine entries are singletons with coefficient `1`.

The `1 → 2 → 3 → 6 → 7` variant this replaced put the input in four positions,
costing `4·|input| + 8`.  Both are four rows, so the row and column counts were
unaffected, but at `|input| = 31` the difference is 132 against 102 per S-box.

That split is also what confines the term-count/nonzero-count gap: the nine
singletons are certainly nonzero, so only the three input copies could
cancel. -/
theorem sboxRows_termCount (frame : SboxFrame) :
    programTermCount (sboxRows frame)
      = 3 * (normalize frame.input).length + 9 := by
  simp only [programTermCount, sboxRows, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, rowTermCount, rowSquare, rowFourth, rowSixth,
    rowSeventh, normalize_singleton]
  omega

/-- **One terminal binding row costs `|state| + 2`.** -/
theorem bindRow_termCount (comb : Poseidon2Core.LinComb) (port : Nat) :
    rowTermCount (bindRow comb port) = (normalize comb).length + 2 := by
  simp only [rowTermCount, bindRow, normalize_singleton]

/-! ## Uniform summation

Each receipt family contributes an affine function of its combination size, so
the family total is a sum plus a multiple of the family's length. -/

theorem sum_map_three_plus_nine {α : Type} (list : List α) (f : α → Nat) :
    ((list.map (fun x => 3 * f x + 9)).sum)
      = 3 * (list.map f).sum + 9 * list.length := by
  induction list with
  | nil => simp
  | cons head tail hypothesis =>
      simp only [List.map_cons, List.sum_cons, hypothesis, List.length_cons]
      omega

theorem sum_map_plus_two {α : Type} (list : List α) (f : α → Nat) :
    ((list.map (fun x => f x + 2)).sum)
      = (list.map f).sum + 2 * list.length := by
  induction list with
  | nil => simp
  | cons head tail hypothesis =>
      simp only [List.map_cons, List.sum_cons, hypothesis, List.length_cons]
      omega

/-! ## The program total

Reduced to the per-S-box scheduled combination sizes and the final state's
lane sizes.  Both are supplied by the support recurrence: full-round states
reference eight columns, partial-round state `r` references `8 + r`, and the
scheduled input adds the constant wire. -/

def scheduledSizes (layout : Layout) (constants : Constants) : List Nat :=
  (List.finRange sboxCount).map
    (fun index => (normalize (scheduleOf layout constants index)).length)

def finalSizes (layout : Layout) : List Nat :=
  (List.finRange width).map
    (fun lane => (normalize (finalState layout lane)).length)

theorem sboxProgram_termCount (layout : Layout) (constants : Constants) :
    programTermCount (sboxProgram layout (scheduleOf layout constants))
      = 3 * (scheduledSizes layout constants).sum + 9 * sboxCount := by
  unfold sboxProgram scheduledSizes
  rw [programTermCount_flatMap]
  simp only [sboxRows_termCount, frameAt]
  rw [sum_map_three_plus_nine, List.length_finRange]

theorem bindingProgram_termCount (layout : Layout) (final : State) :
    programTermCount (bindingProgram layout final)
      = ((List.finRange width).map
          (fun lane => (normalize (final lane)).length)).sum + 2 * width := by
  unfold bindingProgram terminalBindingRows programTermCount
  rw [List.map_map]
  simp only [Function.comp_def, bindRow_termCount]
  rw [sum_map_plus_two, List.length_finRange]

/-- **The program total, decomposed into receipts.**  Every term is derived
from the emitted rows; nothing is declared. -/
theorem canonicalProgram_termCount (layout : Layout) (constants : Constants) :
    programTermCount (canonicalProgram layout constants)
      = 3 * (scheduledSizes layout constants).sum + 9 * sboxCount
        + ((finalSizes layout).sum + 2 * width) := by
  unfold canonicalProgram permutationProgram finalSizes
  rw [programTermCount_append, sboxProgram_termCount, bindingProgram_termCount]


/-! ## Exact sizes for the full-round families

A state produced by a linear layer over eight *distinct* fresh columns
normalizes to exactly eight entries — the raw form is already duplicate-free,
so nothing merges.  This pins every full-round state and the final state.

`List.Nodup.map` is not available without Mathlib; `LinCombNormal.nodup_map`
supplies it. -/

theorem flatten_map_singleton {α β : Type} (list : List α) (g : α → β) :
    (list.map (fun x => [g x])).flatten = list.map g := by
  induction list with
  | nil => simp
  | cons head tail hypothesis => simp [hypothesis]

/-- **A linear layer over distinct fresh columns normalizes to `width`
entries.**  This is where matrix density stops mattering: the columns are
already distinct, so `normalize` merges nothing regardless of coefficients. -/
theorem normalize_length_applyMatrix_singletons
    (matrix : Fin width → Fin width → Nat) (f : Fin width → Nat)
    (inj : ∀ a b : Fin width, f a = f b → a = b) (target : Fin width) :
    (normalize (applyMatrix matrix (fun lane => [(f lane, 1)]) target)).length
      = width := by
  let raw :=
    (List.finRange width).map
      (fun source => (f source, matrix target source * 1 % goldilocksP))
  have rawNodup : (raw.map Prod.fst).Nodup := by
    unfold raw
    rw [List.map_map]
    exact nodup_map _ _ (fun a b image => inj a b image) (by decide)
  change
    (normalize
      (normalize
        ((List.finRange width).flatMap
          (fun source => scale (matrix target source) [(f source, 1)])))).length
      = width
  rw [normalize_length_of_nodup _ (normalize_nodup _)]
  have flattened :
      (List.finRange width).flatMap
          (fun source => scale (matrix target source) [(f source, 1)])
        = raw := by
    unfold raw scale
    simp only [List.map_cons, List.map_nil]
    exact flatten_map_singleton _ _
  rw [flattened, normalize_length_of_nodup raw rawNodup,
    List.length_map, List.length_finRange]

/-- The eight terminal S-box outputs feeding the final layer are distinct. -/
theorem terminalOutput_injective (layout : Layout) :
    ∀ a b : Fin width,
      sboxOutput layout (terminalSboxIndex 3 a.val)
        = sboxOutput layout (terminalSboxIndex 3 b.val) → a = b := by
  intro a b image
  simp only [sboxOutput, terminalSboxIndex, columnsPerSbox, halfFullRounds,
    width, partialRounds] at image
  exact Fin.ext (by omega)

/-- **Every lane of the final state normalizes to eight entries.** -/
theorem finalState_normalize_length (layout : Layout) (lane : Fin width) :
    (normalize (finalState layout lane)).length = width := by
  show (normalize (applyMatrix externalMatrix
    (fun l => [(sboxOutput layout (terminalSboxIndex 3 l.val), 1)]) lane)).length
      = width
  exact normalize_length_applyMatrix_singletons _ _
    (terminalOutput_injective layout) lane

/-- **The terminal binding contribution is exactly 64.**  Derived from the
construction, not declared. -/
theorem finalSizes_sum (layout : Layout) : (finalSizes layout).sum = 64 := by
  unfold finalSizes
  rw [show (fun lane => (normalize (finalState layout lane)).length)
        = (fun _ : Fin width => width) from
      funext (fun lane => finalState_normalize_length layout lane)]
  simp only [width]
  decide


/-! ## No cancellation in a full-round state

A full-round state is one entry per column, so `normalize` merges nothing and
the coefficients are the matrix entries themselves.  Density then rules out
cancellation with no computation and no reference to any round constant — this
is the structural third of `POSEIDON2-NO-CANCELLATION`. -/

theorem fieldNormalize_length_applyMatrix_singletons
    (matrix : Fin width → Fin width → Nat) (f : Fin width → Nat)
    (inj : ∀ a b : Fin width, f a = f b → a = b)
    (nonzero : ∀ a b : Fin width, matrix a b ≠ 0)
    (bounded : ∀ a b : Fin width, matrix a b < goldilocksP)
    (target : Fin width) :
    (fieldNormalize (applyMatrix matrix (fun lane => [(f lane, 1)]) target)).length
      = width := by
  let raw :=
    (List.finRange width).map
      (fun source => (f source, matrix target source * 1 % goldilocksP))
  have rawNodup : (raw.map Prod.fst).Nodup := by
    unfold raw
    rw [List.map_map]
    exact nodup_map _ _ (fun a b image => inj a b image) (by decide)
  have flattened :
      (List.finRange width).flatMap
          (fun source => scale (matrix target source) [(f source, 1)])
        = raw := by
    unfold raw scale
    simp only [List.map_cons, List.map_nil]
    exact flatten_map_singleton _ _
  rw [fieldNormalize_length_of_nonzero, normalize_length_applyMatrix_singletons _ _ inj]
  intro entry member
  change entry ∈
    normalize
      (normalize
        ((List.finRange width).flatMap
          (fun source => scale (matrix target source) [(f source, 1)]))) at member
  have firstNodup :=
    normalize_nodup
      ((List.finRange width).flatMap
        (fun source => scale (matrix target source) [(f source, 1)]))
  have inFirst := normalize_entries_of_nodup _ firstNodup entry member
  rw [flattened] at inFirst
  have inRaw := normalize_entries_of_nodup raw rawNodup entry inFirst
  rcases List.mem_map.1 inRaw with ⟨source, _, image⟩
  rw [← image]
  simp only [Nat.mul_one, Nat.mod_mod]
  rw [Nat.mod_eq_of_lt (bounded target source)]
  exact nonzero target source

/-- **The final state's eight coefficients all survive.**  No round constant is
involved: the coefficients are external-matrix entries, and the matrix is
dense. -/
theorem finalState_fieldNormalize_length (layout : Layout) (lane : Fin width) :
    (fieldNormalize (finalState layout lane)).length = width := by
  show (fieldNormalize (applyMatrix externalMatrix
    (fun l => [(sboxOutput layout (terminalSboxIndex 3 l.val), 1)]) lane)).length
      = width
  exact fieldNormalize_length_applyMatrix_singletons _ _
    (terminalOutput_injective layout)
    (fun a b => externalMatrix_nonzero a b)
    (fun a b => externalMatrix_lt a b) lane

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients
