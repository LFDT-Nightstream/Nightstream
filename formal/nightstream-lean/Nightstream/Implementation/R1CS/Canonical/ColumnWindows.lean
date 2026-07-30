import Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-!
Contract: consecutive column windows, and the proof that distinct owners never
collide.

Owns: the window type, the consecutive layout, and pairwise disjointness.

Does not own: any recipe's own allocation.  Each recipe proves its columns are
`Nodup` within its own window; this module proves the windows themselves do not
overlap, and the two compose into the section-2 "exact column ownership with no
collision" item for an assembled program.

## Why a separate module

`Poseidon2HashRecipe` proved two hash instances disjoint by giving each a base
and checking the arithmetic by hand.  That works for two.  The canonical program
has eight recipes, and hand-checking pairs does not scale — nor does it produce
a statement a future recipe can be added to.

`windowsOf` lays widths out consecutively from a base, and
`windowsOf_no_collision` says any column belongs to at most one window.  Adding
a recipe means adding a width, not extending a proof.

## Column zero is shared

The constant wire belongs to no window: a window owns `(base, base + width]`,
which is strictly above its base, and the first window's base is the layout's
starting point.  Starting a layout at `0` therefore leaves column `0` outside
every window, which is the shared-read convention the recipes already use.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.ColumnWindows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-- A half-open column window: owns `(base, base + width]`. -/
structure Window where
  base : Nat
  width : Nat
deriving DecidableEq, Repr

/-- The columns a window owns.  Strictly above the base, so a layout starting at
`0` never claims the constant wire. -/
def Window.owns (window : Window) (column : Nat) : Prop :=
  window.base < column ∧ column ≤ window.base + window.width

instance (window : Window) (column : Nat) : Decidable (window.owns column) := by
  unfold Window.owns; infer_instance

/-- **The consecutive layout.**  Each recipe starts where the previous one
ended. -/
def windowsOf : Nat → List Nat → List Window
  | _, [] => []
  | base, width :: rest => ⟨base, width⟩ :: windowsOf (base + width) rest

theorem windowsOf_length (base : Nat) (widths : List Nat) :
    (windowsOf base widths).length = widths.length := by
  induction widths generalizing base with
  | nil => rfl
  | cons width rest inductionHypothesis =>
      simp only [windowsOf, List.length_cons, inductionHypothesis]

/-- Every window in a layout starts at or after the layout's base. -/
theorem mem_windowsOf_base_ge (base : Nat) (widths : List Nat) :
    ∀ window ∈ windowsOf base widths, base ≤ window.base := by
  induction widths generalizing base with
  | nil => intro window member; cases member
  | cons width rest inductionHypothesis =>
      intro window member
      rcases List.mem_cons.1 member with rfl | inTail
      · exact Nat.le_refl _
      · exact Nat.le_trans (Nat.le_add_right base width)
          (inductionHypothesis (base + width) window inTail)

/-- **No column belongs to two windows.**

This is the section-2 non-collision item for an assembled program: adding a
recipe means adding a width to the layout, not extending this proof. -/
theorem windowsOf_no_collision (base : Nat) (widths : List Nat) :
    ∀ first ∈ windowsOf base widths, ∀ second ∈ windowsOf base widths,
      ∀ column, first.owns column → second.owns column → first = second := by
  induction widths generalizing base with
  | nil => intro first member; cases member
  | cons width rest inductionHypothesis =>
      intro first firstMember second secondMember column inFirst inSecond
      rcases List.mem_cons.1 firstMember with rfl | firstTail
      · rcases List.mem_cons.1 secondMember with rfl | secondTail
        · rfl
        · have secondBase := mem_windowsOf_base_ge (base + width) rest second
            secondTail
          have firstUpper : column ≤ base + width := inFirst.2
          have secondLower : second.base < column := inSecond.1
          omega
      · rcases List.mem_cons.1 secondMember with rfl | secondTail
        · have firstBase := mem_windowsOf_base_ge (base + width) rest first
            firstTail
          have secondUpper : column ≤ base + width := inSecond.2
          have firstLower : first.base < column := inFirst.1
          omega
        · exact inductionHypothesis (base + width) first firstTail second
            secondTail column inFirst inSecond

/-- **The layout spans exactly the sum of its widths.**  So a total column count
is the widths' sum, and under `windowsOf_no_collision` that sum counts distinct
columns rather than merely adding per-recipe figures. -/
theorem windowsOf_span (base : Nat) (widths : List Nat) :
    ∀ window ∈ windowsOf base widths,
      window.base + window.width ≤ base + widths.sum := by
  induction widths generalizing base with
  | nil => intro window member; cases member
  | cons width rest inductionHypothesis =>
      intro window member
      rcases List.mem_cons.1 member with rfl | inTail
      · simp only [List.sum_cons]
        omega
      · have := inductionHypothesis (base + width) window inTail
        simp only [List.sum_cons]
        omega

/-- **The constant wire is outside every window of a layout based at zero.** -/
theorem constantWire_unowned (widths : List Nat) :
    ∀ window ∈ windowsOf 0 widths, ¬ window.owns 0 := by
  intro window _ owns
  exact absurd owns.1 (Nat.not_lt_zero _)

/-! ## Placement

A window is a plan until something is put in it.  `relocate` is the map that
does so — the same one `Poseidon2HashRecipe` uses for its two hash instances,
lifted here so it serves any number of recipes. -/

/-- Shift every column except the constant wire, which is shared. -/
def relocate (base : Nat) : Nat → Nat :=
  fun column => if column = 0 then 0 else base + column

theorem relocate_zero (base : Nat) : relocate base 0 = 0 := rfl

theorem relocate_pos (base column : Nat) (nonZero : column ≠ 0) :
    relocate base column = base + column := if_neg nonZero

/-- **A relocated column lands in its window.**

The bound `column ≤ width` is what the caller owes: a recipe that numbers its
own columns `1 … width` satisfies it, and one that does not has no width to be
placed by. -/
theorem relocate_owns
    (base width column : Nat) (nonZero : column ≠ 0) (bounded : column ≤ width) :
    (Window.mk base width).owns (relocate base column) := by
  rw [relocate_pos base column nonZero]
  constructor
  · simp only []
    omega
  · simp only []
    omega

/-- **Place a sequence of programs in consecutive windows.** -/
def placeAll : Nat → List (List Row × Nat) → List Row
  | _, [] => []
  | base, part :: rest =>
      part.1.map (renameRow (relocate base)) ++ placeAll (base + part.2) rest

theorem mentions_renameTerms_relocate
    (base : Nat) (terms : List (Nat × Nat)) (column : Nat)
    (mentioned : Mentions (renameTerms (relocate base) terms) column) :
    ∃ source, Mentions terms source ∧ column = relocate base source := by
  simp only [Mentions, renameTerms, List.map_map, List.mem_map,
    Function.comp] at mentioned ⊢
  rcases mentioned with ⟨term, member, rfl⟩
  exact ⟨term.1, ⟨term, member, rfl⟩, rfl⟩

/-- **Every column of a placed program is the constant wire or lies in exactly
one window.**

Combined with `windowsOf_no_collision`, this is the section-2 "exact column
ownership with no collision" item for an assembled program: the existential
gives membership, the collision theorem gives uniqueness.

The `bounded` hypothesis is each recipe's own conservation, restated against its
declared width.  A recipe that cannot state one has not declared a width, and a
program cannot be laid out from widths it does not have. -/
theorem placeAll_columns
    (base : Nat) (parts : List (List Row × Nat))
    (bounded : ∀ part ∈ parts, ∀ row ∈ part.1, ∀ column,
      (Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) →
        column = 0 ∨ column ≤ part.2) :
    ∀ row ∈ placeAll base parts, ∀ column,
      (Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) →
        column = 0
          ∨ ∃ window ∈ windowsOf base (parts.map Prod.snd), window.owns column := by
  induction parts generalizing base with
  | nil => intro row member; cases member
  | cons part rest inductionHypothesis =>
      intro row member column mentioned
      rcases List.mem_append.1 member with placed | later
      · rcases List.mem_map.1 placed with ⟨raw, rawMember, rfl⟩
        simp only [renameRow] at mentioned
        have sourceOf : ∃ source,
            (Mentions raw.a source ∨ Mentions raw.b source ∨ Mentions raw.c source)
              ∧ column = relocate base source := by
          rcases mentioned with inA | inB | inC
          · rcases mentions_renameTerms_relocate base raw.a column inA with
              ⟨source, m, same⟩
            exact ⟨source, Or.inl m, same⟩
          · rcases mentions_renameTerms_relocate base raw.b column inB with
              ⟨source, m, same⟩
            exact ⟨source, Or.inr (Or.inl m), same⟩
          · rcases mentions_renameTerms_relocate base raw.c column inC with
              ⟨source, m, same⟩
            exact ⟨source, Or.inr (Or.inr m), same⟩
        rcases sourceOf with ⟨source, sourceMentioned, rfl⟩
        rcases bounded part List.mem_cons_self raw rawMember source
          sourceMentioned with isWire | inWidth
        · exact Or.inl (by rw [isWire, relocate_zero])
        · by_cases sourceZero : source = 0
          · exact Or.inl (by rw [sourceZero, relocate_zero])
          · refine Or.inr ⟨⟨base, part.2⟩, ?_, relocate_owns base part.2 source
              sourceZero inWidth⟩
            simp only [windowsOf, List.map_cons, List.mem_cons]
            exact Or.inl trivial
      · have tail := inductionHypothesis (base + part.2)
          (fun other otherMember => bounded other
            (List.mem_cons_of_mem part otherMember)) row later column mentioned
        rcases tail with isWire | ⟨window, windowMember, owns⟩
        · exact Or.inl isWire
        · refine Or.inr ⟨window, ?_, owns⟩
          simp only [windowsOf, List.map_cons, List.mem_cons]
          exact Or.inr windowMember

/-! ## Targets only

`placeAll_columns` bounds **every** column a part mentions.  That is the right
condition for a part that owns everything it touches, and the wrong one for a
part that *reads* a value another part owns.

Most recipes in this line read: `FoldDigestRecipe` compares caller-supplied lane
combinations, `KRecomposition` reads children and a parent, `CommitmentMixerRecipe`
reads commitment coordinates.  None allocates, so bounding their mentioned
columns by their own width — zero — is unsatisfiable, and `placeAll_columns`
cannot be applied to them at all.

The fix is the one `TranscriptRecipe` already found: **track the target, not
every column.**  A row's `c` field names what the row writes, and writes are what
a window owns.  Reads cross windows legitimately — that is what makes a program
a program rather than a list of independent parts.

`placeAll_targets` is `placeAll_columns` restricted to `c`, and it is the form an
assembly of reading parts can actually use. -/

/-- **Every target of a placed program lies in exactly one window.**

The reading counterpart of `placeAll_columns`: reads may cross windows, writes
may not. -/
theorem placeAll_targets
    (base : Nat) (parts : List (List Row × Nat))
    (bounded : ∀ part ∈ parts, ∀ row ∈ part.1, ∀ column,
      Mentions row.c column → column = 0 ∨ column ≤ part.2) :
    ∀ row ∈ placeAll base parts, ∀ column, Mentions row.c column →
      column = 0
        ∨ ∃ window ∈ windowsOf base (parts.map Prod.snd), window.owns column := by
  induction parts generalizing base with
  | nil => intro row member; cases member
  | cons part rest inductionHypothesis =>
      intro row member column mentioned
      rcases List.mem_append.1 member with placed | later
      · rcases List.mem_map.1 placed with ⟨raw, rawMember, rfl⟩
        simp only [renameRow] at mentioned
        rcases mentions_renameTerms_relocate base raw.c column mentioned with
          ⟨source, sourceMentioned, rfl⟩
        rcases bounded part List.mem_cons_self raw rawMember source
          sourceMentioned with isWire | inWidth
        · exact Or.inl (by rw [isWire, relocate_zero])
        · by_cases sourceZero : source = 0
          · exact Or.inl (by rw [sourceZero, relocate_zero])
          · refine Or.inr ⟨⟨base, part.2⟩, ?_, relocate_owns base part.2 source
              sourceZero inWidth⟩
            simp only [windowsOf, List.map_cons, List.mem_cons]
            exact Or.inl trivial
      · have tail := inductionHypothesis (base + part.2)
          (fun other otherMember => bounded other
            (List.mem_cons_of_mem part otherMember)) row later column mentioned
        rcases tail with isWire | ⟨window, windowMember, owns⟩
        · exact Or.inl isWire
        · refine Or.inr ⟨window, ?_, owns⟩
          simp only [windowsOf, List.map_cons, List.mem_cons]
          exact Or.inr windowMember

/-! ## `c` is not a target either

`placeAll_targets` was written on the reading that a row's `c` field "names what
the row writes".  Applying it to `KLowNormBatch` shows that is false.

`KLowNorm.lowNormRows value column` emits

```text
⟨value, value, [(column, 1)]⟩      -- c is the allocated column
⟨[(column, 1)], value, value⟩      -- c is the checked combination
```

The second row's `c` is `value` — a **read**.  In `A·z * B·z = C·z` the `C`
side is a linear combination like the others; it is a single allocated column
only when a recipe happens to emit one there, and `KEquality.equalityRow left
right` has `c = right`, also a read.

**No syntactic field of `Row` identifies what a row writes.**  That is a
property of the emission, not of the row.

## What placement actually needs

Each recipe already declares its allocation separately —
`KLowNormBatch.batchColumns`, `KRecomposition.recompositionColumns`,
`FoldDigestRecipe.digestColumns`, and so on.  Placement should relocate a part
and check that **its declared allocation** lands in its window, which is:

- vacuous for the allocation-free recipes, whose declared allocation is empty;
- `KLowNormBatch.canonicalDigits_column_le` for the one that allocates.

That is a statement about allocations, not about rows, and it needs neither
`placeAll_columns`' unsatisfiable read bound nor `placeAll_targets`' false
premise about `c`.

Both earlier tools are kept: `placeAll_columns` is correct for a part that owns
everything it touches, and `placeAll_targets` is correct for a part whose `c` is
always a declared column — `TranscriptRecipe` is one, which is why the argument
worked there.  Neither is correct in general, and the general case is the
allocation. -/

/-- **The allocation-based placement obligation**, stated so the next
construction has the right shape to aim at: a part's declared allocation, after
relocation, lies inside its own window. -/
def AllocationPlaced (base : Nat) (allocation : List Nat) (width : Nat) : Prop :=
  ∀ column ∈ allocation,
    (Window.mk base width).owns (relocate base column)

/-- **An empty allocation is placed anywhere.**  Every reading recipe in this
line satisfies the obligation vacuously, which is why the general case is about
allocations rather than rows. -/
theorem allocationPlaced_nil (base width : Nat) :
    AllocationPlaced base [] width := by
  intro column member
  cases member

/-- **A bounded allocation is placed.**  The one condition an allocating recipe
must supply, and `KLowNormBatch.canonicalDigits_column_le` is exactly its
shape. -/
theorem allocationPlaced_of_bounded
    (base width : Nat) (allocation : List Nat)
    (bounded : ∀ column ∈ allocation, column ≠ 0 ∧ column ≤ width) :
    AllocationPlaced base allocation width := by
  intro column member
  exact relocate_owns base width column (bounded column member).1
    (bounded column member).2

/-! ## Sizing a window from its allocation

`CANONICAL-PROGRAM-WINDOW-SIZING` recorded the transcript's sparse allocation as
blocked between two repairs, each said to change a published number.  That
framing was wrong, and the correction is worth more than the choice: **the
window is not a published number.**

`Typed.Cost.auxiliaryColumns` is a count and stays one.  `Window.width` is an
abstraction introduced here, and nothing requires it to equal that count.  What
was really needed is a width the allocation determines.

`spanOf` is that width: the largest allocated column.  Sizing a window by it
makes placement hold **for any allocation**, sparse or not, with one obligation —
no allocated column is the constant wire.  Nothing has to be contiguous, and no
count changes meaning. -/

/-- The width an allocation needs: its largest column, or zero when empty. -/
def spanOf (allocation : List Nat) : Nat :=
  allocation.foldl max 0

theorem spanOf_nil : spanOf [] = 0 := rfl

/-- Every allocated column is within the span. -/
theorem le_spanOf (allocation : List Nat) :
    ∀ column ∈ allocation, column ≤ spanOf allocation := by
  unfold spanOf
  suffices general : ∀ (accumulator : Nat) (columns : List Nat),
      ∀ column ∈ columns, column ≤ columns.foldl max accumulator by
    exact general 0 allocation
  intro accumulator columns
  induction columns generalizing accumulator with
  | nil => intro column member; cases member
  | cons head tail inductionHypothesis =>
      intro column member
      rcases List.mem_cons.1 member with rfl | inTail
      · have grows : ∀ (start : Nat) (rest : List Nat),
            start ≤ rest.foldl max start := by
          intro start rest
          induction rest generalizing start with
          | nil => exact Nat.le_refl _
          | cons next remaining hypothesis =>
              exact Nat.le_trans (Nat.le_max_left start next)
                (hypothesis (max start next))
        exact Nat.le_trans (Nat.le_max_right accumulator column)
          (grows (max accumulator column) tail)
      · exact inductionHypothesis (max accumulator head) column inTail

/-- **An allocation is placed in a window sized by its own span.**

The only obligation is that no allocated column is the constant wire.
Contiguity is not required, so a sparse allocation — the transcript's — is
placed exactly as a dense one is. -/
theorem allocationPlaced_spanOf
    (base : Nat) (allocation : List Nat)
    (noWire : ∀ column ∈ allocation, column ≠ 0) :
    AllocationPlaced base allocation (spanOf allocation) := by
  intro column member
  exact relocate_owns base (spanOf allocation) column (noWire column member)
    (le_spanOf allocation column member)

/-! ### Placement plus window-disjointness is allocation-disjointness

`windowsOf_no_collision` says two windows sharing a column are the same window.
`AllocationPlaced` says a part's columns land in its own window.  Neither is
what section 2 item 4 asks for — that is **no column is allocated by two
recipes**, and it is the composition of the two.

Kept as a separate step because the composition is where the quantifiers have to
line up: the disjointness is about *relocated* columns, since a placed program is
what the assembly emits and two parts may perfectly well allocate the same raw
column before relocation. -/

/-- **Two placed allocations in distinct windows share no relocated column.** -/
theorem placed_allocations_disjoint
    (first second : Window) (firstAllocation secondAllocation : List Nat)
    (firstPlaced : AllocationPlaced first.base firstAllocation first.width)
    (secondPlaced : AllocationPlaced second.base secondAllocation second.width)
    (noCollision : ∀ column, first.owns column → second.owns column →
      first = second)
    (different : first ≠ second)
    (firstColumn : Nat) (firstMember : firstColumn ∈ firstAllocation)
    (secondColumn : Nat) (secondMember : secondColumn ∈ secondAllocation) :
    relocate first.base firstColumn ≠ relocate second.base secondColumn := by
  intro equal
  have inFirst : first.owns (relocate first.base firstColumn) :=
    firstPlaced firstColumn firstMember
  have inSecond : second.owns (relocate second.base secondColumn) :=
    secondPlaced secondColumn secondMember
  rw [← equal] at inSecond
  exact different (noCollision _ inFirst inSecond)

/-! ### The count against the span

A `Nodup` allocation of `n` nonzero columns has span at least `n`, by
pigeonhole.  **That argument is not written here**: it needs
`List.Nodup.length_le_of_subset` or an equivalent, which this tree does not
have — the same gap that keeps `EuclidPrime goldilocksP` a hypothesis.

It is not needed for placement.  `allocationPlaced_spanOf` holds without it, and
`Typed.Cost.auxiliaryColumns` keeps its meaning as a count regardless of how the
window is sized.  Recorded so the relationship is not assumed in either
direction. -/

/-! ## Satisfaction across a placement

`placeAll` relocates each part by its own base, so satisfaction does not
transport by a single pullback — each part needs its own.  `rowHolds_pull_iff`
supplies the step; this threads it through the consecutive bases.

Without this a placed program has a row count and no semantics, which is the
shape cycle 353 found in the assembly's selection interface: capabilities
enumerated on one object and the other left without them. -/

/-- The base at which `placeAll` puts the part at position `index`. -/
def baseAt : Nat → List (List Row × Nat) → Nat → Nat
  | base, _, 0 => base
  | base, [], _ + 1 => base
  | base, part :: rest, index + 1 => baseAt (base + part.2) rest index

/-- **Satisfying a placed program satisfies each part**, under that part's own
pullback.

Stated for the head, and applied by peeling: the tail of a `placeAll` is itself
a `placeAll` at the shifted base, so an index-`n` part is reached by `n`
applications rather than by an induction on a position argument. -/
theorem placeAll_satisfies_head
    (base : Nat) (part : List Row × Nat) (rest : List (List Row × Nat))
    (z : Nat → Nat)
    (satisfied : Satisfies (placeAll base (part :: rest)) z) :
    Satisfies part.1 (pullAssignment z (relocate base)) := by
  intro row member
  refine (rowHolds_pull_iff z (relocate base) row).2 ?_
  exact satisfied _ (List.mem_append_left _
    (List.mem_map.2 ⟨row, member, rfl⟩))

/-- **The tail of a placed program is a placed program.**  This is what lets the
head lemma reach every part. -/
theorem placeAll_satisfies_tail
    (base : Nat) (part : List Row × Nat) (rest : List (List Row × Nat))
    (z : Nat → Nat)
    (satisfied : Satisfies (placeAll base (part :: rest)) z) :
    Satisfies (placeAll (base + part.2) rest) z := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

/-- **Every part of a placed program is recovered, at its own base.**

The indexed form.  Six further instances of `placeAll_satisfies_head` would have
been six copies of one argument; this is the argument, and `baseAt` is the base
the placement puts each part at.

Cycle 379 called the remaining instances mechanical.  They were — but writing
them out would have been the mechanical thing, and generalising was not: the
statement needs `baseAt` to describe where a part lands, which no instance
needs. -/
theorem placeAll_satisfies_index
    (z : Nat → Nat) :
    ∀ (parts : List (List Row × Nat)) (base index : Nat) (part : List Row × Nat),
      parts[index]? = some part →
      Satisfies (placeAll base parts) z →
      Satisfies part.1 (pullAssignment z (relocate (baseAt base parts index))) := by
  intro parts
  induction parts with
  | nil =>
      intro base index part found _
      simp only [List.getElem?_nil] at found
      cases found
  | cons head rest inductionHypothesis =>
      intro base index part found satisfied
      cases index with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at found
          rw [← found]
          exact placeAll_satisfies_head base head rest z satisfied
      | succ previous =>
          simp only [List.getElem?_cons_succ] at found
          exact inductionHypothesis (base + head.2) previous part found
            (placeAll_satisfies_tail base head rest z satisfied)

/-- **A placed program is satisfied when every part is**, each under its own
pullback.

The converse of `placeAll_satisfies_index`.  Without it a placed program can be
taken apart and never put together, which is the directional gap cycle 354 found
for the unplaced assembly — recurring here because `placedRows` was a new object
and only the direction that had been needed was proved. -/
theorem placeAll_honest
    (z : Nat → Nat) :
    ∀ (parts : List (List Row × Nat)) (base : Nat),
      (∀ (index : Nat) (part : List Row × Nat), parts[index]? = some part →
        Satisfies part.1 (pullAssignment z (relocate (baseAt base parts index)))) →
      Satisfies (placeAll base parts) z := by
  intro parts
  induction parts with
  | nil => intro base _ row member; cases member
  | cons head rest inductionHypothesis =>
      intro base each row member
      rcases List.mem_append.1 member with placed | later
      · rcases List.mem_map.1 placed with ⟨raw, rawMember, rfl⟩
        refine (rowHolds_pull_iff z (relocate base) raw).1 ?_
        exact each 0 head rfl raw rawMember
      · exact inductionHypothesis (base + head.2)
          (fun index part found =>
            each (index + 1) part (by simpa using found)) row later

end Nightstream.Implementation.R1CS.Canonical.ColumnWindows
