import Nightstream.Implementation.R1CS.Canonical.PiDecRecipe
import Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe
import Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe
import Nightstream.Implementation.R1CS.Canonical.TranscriptRecipe
import Nightstream.Implementation.R1CS.Canonical.ColumnWindows

/-!
Contract: the canonical program and its cost, assembled from every recipe that
exists.

Owns: the assembly, the emitted row list, and `N_canonical` as a receipt fold
whose row component is proved against that list.

## What was actually missing

`allRecipes`, `CanonicalProgram` and `N_canonical` are named by the phase plan.
Before this module **none of the three existed anywhere in the tree** — the only
`CanonicalProgram`-shaped name was `Poseidon2Normalized.normalizedCanonicalProgram`,
which is the permutation and unrelated.

They had been reported as "blocked pending selection" for many cycles.  That was
true of four *components* and not of the assembly, and the distinction was never
checked.  It is the same defect as the four inherited Rust dependencies this
session corrected, with the difference that this one was self-inflicted.

## The selections are inputs, and they are proof-carrying

`step`, `nifsVerify`, `runningCheck` and `freshCheck` are setup selections.
`TerminalCheckSelectionBoundary` proves two lawful checkers disagree on the same
argument, so no encoding derives them.

They therefore enter as `SelectedRecipe`: rows, a cost, **and a proof that the
rows match the cost**.  That is not an obligation moved to a premise no consumer
constructs — the consumer is a deployment that has made the selection, and it
must supply the certificate along with the recipe.  A selection without one is
not a recipe.

So the blockage is now exactly four proof-carrying inputs rather than an
unbounded "downstream of the selection".

## What the fold does and does not include

Every recipe built in this line contributes: Π_DEC's decomposition algebra, the
fold-digest lane equalities, the commitment mixer, and the Fiat–Shamir
transcript.  The Poseidon2 hash calls enter through the transcript's rounds.

`N_canonical.recurringRows` is proved equal to the emitted list's length, and
`auxiliaryColumns_eq_counts_sum` relates the column component to the per-part
counts.  `CANONICAL-PROGRAM-COLUMN-DISJOINTNESS` was closed in cycle 344 at the
general level and the layout was connected to an emitted program —
`placedRows` — in cycle 377.  This header said the disjointness entry was open
until cycle 384.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.CanonicalProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-- A setup-selected recipe, carrying its own certificates.

`rowsCertified` is what makes a selection usable: without it the assembly could
not derive its own row count, and a declared count is exactly what this project
does not accept.

`allocation` and `allocationCertified` were added in cycle 372, after the
placement work found the gap.  The first draft carried only
`cost.auxiliaryColumns` — a **number**, not a list — so a selection declared how
many columns it allocates and never which.  The layout sized its window from
that number while nothing tied the number to any column, and
`ColumnWindows.AllocationPlaced` could not even be *stated* for a selection.

A count without the columns it counts is the same defect as a row count without
the rows.  The structure now carries both. -/
structure SelectedRecipe where
  rows : List Row
  cost : Lowering.Typed.Cost
  allocation : List Nat
  rowsCertified : rows.length = cost.recurringRows
  allocationCertified : allocation.length = cost.auxiliaryColumns

/-- The zero selection, for stating the built part in isolation. -/
def SelectedRecipe.empty : SelectedRecipe where
  rows := []
  cost := ⟨0, 0, 0, 0⟩
  allocation := []
  rowsCertified := rfl
  allocationCertified := rfl

/-- **Every recipe of the canonical program.**

The first four fields are built and proved in this line.  The last four are the
setup selections. -/
structure Recipes where
  /-- Π_DEC's decomposition algebra. -/
  piDec : PiDecRecipe.Decomposition
  /-- Fold-digest lane equalities. -/
  foldDigest : List (LinComb × LinComb)
  /-- The commitment mixer, coordinate by coordinate. -/
  mixer : List CommitmentMixerRecipe.Coordinate
  /-- The mixer's radix. -/
  mixerBase : Nat
  /-- Transcript layouts, absorb schedule, constants and round count. -/
  transcriptLayouts : Nat → Poseidon2Program.Layout
  transcriptSchedule : TranscriptRecipe.Schedule
  transcriptConstants : Poseidon2Schedule.Constants
  transcriptRounds : Nat
  /-- The four setup selections. -/
  step : SelectedRecipe
  nifsVerify : SelectedRecipe
  runningCheck : SelectedRecipe
  freshCheck : SelectedRecipe

namespace Recipes

/-- **The emitted canonical program.** -/
def rows (recipes : Recipes) : List Row :=
  PiDecRecipe.rows recipes.piDec
    ++ FoldDigestRecipe.digestRows recipes.foldDigest
    ++ CommitmentMixerRecipe.mixerRows recipes.mixerBase recipes.mixer
    ++ TranscriptRecipe.transcriptRows recipes.transcriptLayouts
        recipes.transcriptSchedule recipes.transcriptConstants
        recipes.transcriptRounds
    ++ recipes.step.rows
    ++ recipes.nifsVerify.rows
    ++ recipes.runningCheck.rows
    ++ recipes.freshCheck.rows

/-- **`N_canonical`.**  Every component a receipt fold: the four built recipes'
costs plus the four selections' certified costs. -/
def N_canonical (recipes : Recipes) : Lowering.Typed.Cost where
  recurringRows :=
    (PiDecRecipe.cost recipes.piDec).recurringRows
      + (FoldDigestRecipe.digestCost recipes.foldDigest).recurringRows
      + (CommitmentMixerRecipe.mixerCost recipes.mixer).recurringRows
      + (TranscriptRecipe.transcriptCost recipes.transcriptRounds).recurringRows
      + recipes.step.cost.recurringRows
      + recipes.nifsVerify.cost.recurringRows
      + recipes.runningCheck.cost.recurringRows
      + recipes.freshCheck.cost.recurringRows
  committedColumns :=
    recipes.step.cost.committedColumns
      + recipes.nifsVerify.cost.committedColumns
      + recipes.runningCheck.cost.committedColumns
      + recipes.freshCheck.cost.committedColumns
  publicColumns :=
    recipes.step.cost.publicColumns
      + recipes.nifsVerify.cost.publicColumns
      + recipes.runningCheck.cost.publicColumns
      + recipes.freshCheck.cost.publicColumns
  auxiliaryColumns :=
    (PiDecRecipe.cost recipes.piDec).auxiliaryColumns
      + (TranscriptRecipe.transcriptCost recipes.transcriptRounds).auxiliaryColumns
      + recipes.step.cost.auxiliaryColumns
      + recipes.nifsVerify.cost.auxiliaryColumns
      + recipes.runningCheck.cost.auxiliaryColumns
      + recipes.freshCheck.cost.auxiliaryColumns

/-- **The row component is derived from the emitted list**, not declared.

The four built recipes contribute their own proved counts; the four selections
contribute theirs through `rowsCertified`.  Nothing here is measured. -/
theorem rows_length (recipes : Recipes) :
    (rows recipes).length = (N_canonical recipes).recurringRows := by
  unfold rows N_canonical
  simp only [List.length_append,
    PiDecRecipe.cost_rows, FoldDigestRecipe.digestCost_rows,
    CommitmentMixerRecipe.mixerCost_rows,
    TranscriptRecipe.transcriptCost_rows,
    recipes.step.rowsCertified, recipes.nifsVerify.rowsCertified,
    recipes.runningCheck.rowsCertified, recipes.freshCheck.rowsCertified]

/-- **With no selection made, the built part still has a derived count.**

This is what the assembly buys: the four selections are isolated to four
summands, so everything else is stated and proved today. -/
theorem rows_length_built_only (recipes : Recipes)
    (noStep : recipes.step = SelectedRecipe.empty)
    (noNifs : recipes.nifsVerify = SelectedRecipe.empty)
    (noRunning : recipes.runningCheck = SelectedRecipe.empty)
    (noFresh : recipes.freshCheck = SelectedRecipe.empty) :
    (rows recipes).length
      = (PiDecRecipe.cost recipes.piDec).recurringRows
        + (FoldDigestRecipe.digestCost recipes.foldDigest).recurringRows
        + (CommitmentMixerRecipe.mixerCost recipes.mixer).recurringRows
        + (TranscriptRecipe.transcriptCost recipes.transcriptRounds).recurringRows := by
  rw [rows_length]
  unfold N_canonical
  simp only [noStep, noNifs, noRunning, noFresh, SelectedRecipe.empty]
  omega


/-! ## The column layout

`CANONICAL-PROGRAM-COLUMN-DISJOINTNESS` asked whether `N_canonical`'s column
components count distinct columns or merely add per-recipe figures.  The layout
below answers half of it, and the half it does not answer is stated rather than
glossed.

**Proved here:** the eight recipes' auxiliary widths, laid out consecutively
from zero, give windows no two of which share a column, and the constant wire
falls outside all of them.  So the widths' sum *is* a distinct-column count for
that layout.

**Where the layout is applied:** `placedRows`, below.  Each part's allocation is
placed in its own window by the `*_placed_at_its_base` family, at the base
`placeAll` assigns it.  What the layout governs is allocations rather than every
column a row mentions — reads cross windows by design. -/

/-- The eight declared allocations, in program order. -/
def allocations (recipes : Recipes) : List (List Nat) :=
  [ PiDecRecipe.columns recipes.piDec,
    FoldDigestRecipe.digestColumns,
    CommitmentMixerRecipe.mixerColumns,
    TranscriptRecipe.transcriptColumns recipes.transcriptRounds,
    recipes.step.allocation,
    recipes.nifsVerify.allocation,
    recipes.runningCheck.allocation,
    recipes.freshCheck.allocation ]

/-! ## Counts and spans are different quantities

`Typed.Cost.auxiliaryColumns` counts columns.  A window needs to know **where**
they sit, which is `ColumnWindows.spanOf`.  The two coincide only for a
`1 … n` allocation and the transcript's is not one, so both are stated and
neither is used for the other's job.

`CANONICAL-PROGRAM-WINDOW-SIZING` records why sizing windows by the count was
the error. -/

/-- Per-part column counts, in program order.  These are what `N_canonical`
sums. -/
def counts (recipes : Recipes) : List Nat :=
  (allocations recipes).map List.length

/-- Per-part window widths: each allocation's own span. -/
def widths (recipes : Recipes) : List Nat :=
  (allocations recipes).map ColumnWindows.spanOf

/-- The consecutive layout, based at zero so the constant wire is unowned. -/
def windows (recipes : Recipes) : List ColumnWindows.Window :=
  ColumnWindows.windowsOf 0 (widths recipes)

theorem windows_length (recipes : Recipes) :
    (windows recipes).length = 8 := by
  unfold windows widths allocations
  rw [ColumnWindows.windowsOf_length, List.length_map]
  rfl

/-- **No two parts' windows share a column.** -/
theorem windows_no_collision (recipes : Recipes) :
    ∀ first ∈ windows recipes, ∀ second ∈ windows recipes,
      ∀ column, first.owns column → second.owns column → first = second :=
  ColumnWindows.windowsOf_no_collision 0 (widths recipes)

/-- **The constant wire belongs to no part.** -/
theorem windows_exclude_constantWire (recipes : Recipes) :
    ∀ window ∈ windows recipes, ¬ window.owns 0 :=
  ColumnWindows.constantWire_unowned (widths recipes)

/-- **`N_canonical`'s auxiliary component is the counts' sum.**

Stated against `counts`, not `widths`: it is a number of columns, and the
layout's spans are a different quantity. -/
theorem auxiliaryColumns_eq_counts_sum (recipes : Recipes) :
    (N_canonical recipes).auxiliaryColumns = (counts recipes).sum := by
  unfold N_canonical counts allocations
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    PiDecRecipe.columns_length, PiDecRecipe.cost,
    FoldDigestRecipe.digestColumns,
    CommitmentMixerRecipe.mixerColumns,
    TranscriptRecipe.transcriptColumns_length_eq,
    List.length_nil, recipes.step.allocationCertified,
    recipes.nifsVerify.allocationCertified,
    recipes.runningCheck.allocationCertified,
    recipes.freshCheck.allocationCertified]
  omega

/-! ## Placement, discharged where it can be

`ColumnWindows.allocationPlaced_spanOf` needs one thing per part: no allocated
column is the constant wire.  Two parts allocate nothing, one is proved, and the
rest are the claim's or the deployment's. -/

theorem foldDigest_placed (base : Nat) :
    ColumnWindows.AllocationPlaced base FoldDigestRecipe.digestColumns
      (ColumnWindows.spanOf FoldDigestRecipe.digestColumns) :=
  ColumnWindows.allocationPlaced_spanOf base _ (by
    intro column member
    simp only [FoldDigestRecipe.digestColumns] at member
    cases member)

theorem mixer_placed (base : Nat) :
    ColumnWindows.AllocationPlaced base CommitmentMixerRecipe.mixerColumns
      (ColumnWindows.spanOf CommitmentMixerRecipe.mixerColumns) :=
  ColumnWindows.allocationPlaced_spanOf base _ (by
    intro column member
    simp only [CommitmentMixerRecipe.mixerColumns] at member
    cases member)

/-- **The transcript is placed**, sparse allocation and all.  This is the case a
count-sized window could not hold. -/
theorem transcript_placed (recipes : Recipes) (base : Nat) :
    ColumnWindows.AllocationPlaced base
      (TranscriptRecipe.transcriptColumns recipes.transcriptRounds)
      (ColumnWindows.spanOf
        (TranscriptRecipe.transcriptColumns recipes.transcriptRounds)) :=
  ColumnWindows.allocationPlaced_spanOf base _
    (TranscriptRecipe.transcriptColumns_nonzero recipes.transcriptRounds)

/-- **A selection is placed** once the deployment says its columns are real. -/
theorem selection_placed
    (selection : SelectedRecipe) (base : Nat)
    (noWire : ∀ column ∈ selection.allocation, column ≠ 0) :
    ColumnWindows.AllocationPlaced base selection.allocation
      (ColumnWindows.spanOf selection.allocation) :=
  ColumnWindows.allocationPlaced_spanOf base _ noWire

/-- **Π_DEC is placed** once its digits allocate real columns. -/
theorem piDec_placed (recipes : Recipes) (base : Nat)
    (noWire : ∀ column ∈ PiDecRecipe.columns recipes.piDec, column ≠ 0) :
    ColumnWindows.AllocationPlaced base (PiDecRecipe.columns recipes.piDec)
      (ColumnWindows.spanOf (PiDecRecipe.columns recipes.piDec)) :=
  ColumnWindows.allocationPlaced_spanOf base _ noWire

/-! ## Soundness restricts to every recipe

The assembly adds no constraint of its own, so satisfying it satisfies each
part.  Each part's own soundness theorem then applies unchanged. -/

theorem satisfies_piDec (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (rows recipes) z) :
    Satisfies (PiDecRecipe.rows recipes.piDec) z := by
  intro row member
  exact satisfied row (by
    unfold rows
    exact List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_left _ (List.mem_append_left _
        (List.mem_append_left _ (List.mem_append_left _
          (List.mem_append_left _ member)))))))

theorem satisfies_foldDigest (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (rows recipes) z) :
    Satisfies (FoldDigestRecipe.digestRows recipes.foldDigest) z := by
  intro row member
  exact satisfied row (by
    unfold rows
    exact List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_left _ (List.mem_append_left _
        (List.mem_append_left _ (List.mem_append_left _
          (List.mem_append_right _ member)))))))

theorem satisfies_mixer (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (rows recipes) z) :
    Satisfies (CommitmentMixerRecipe.mixerRows recipes.mixerBase recipes.mixer) z := by
  intro row member
  exact satisfied row (by
    unfold rows
    exact List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_left _ (List.mem_append_left _
        (List.mem_append_left _ (List.mem_append_right _ member))))))

theorem satisfies_transcript (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (rows recipes) z) :
    Satisfies (TranscriptRecipe.transcriptRows recipes.transcriptLayouts
      recipes.transcriptSchedule recipes.transcriptConstants
      recipes.transcriptRounds) z := by
  intro row member
  exact satisfied row (by
    unfold rows
    exact List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_left _ (List.mem_append_left _
        (List.mem_append_right _ member)))))

/-! ## Soundness restricts to the selections too

The four built parts had restriction lemmas from the start; the four selections
did not.  That made `SelectedRecipe` an interface a deployment could put rows
*into* and not get conclusions *out of* — a selection could be assembled and
then never used, which is not what a proof-carrying interface is for.

These four complete it.  A deployment that supplies `step` can now conclude its
own rows hold from the whole program holding, and apply whatever soundness
theorem it proved about them. -/

theorem satisfies_step (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (rows recipes) z) :
    Satisfies recipes.step.rows z := by
  intro row member
  exact satisfied row (by
    unfold rows
    exact List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_left _ (List.mem_append_right _ member))))

theorem satisfies_nifsVerify (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (rows recipes) z) :
    Satisfies recipes.nifsVerify.rows z := by
  intro row member
  exact satisfied row (by
    unfold rows
    exact List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_right _ member)))

theorem satisfies_runningCheck (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (rows recipes) z) :
    Satisfies recipes.runningCheck.rows z := by
  intro row member
  exact satisfied row (by
    unfold rows
    exact List.mem_append_left _ (List.mem_append_right _ member))

theorem satisfies_freshCheck (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (rows recipes) z) :
    Satisfies recipes.freshCheck.rows z := by
  intro row member
  exact satisfied row (by
    unfold rows
    exact List.mem_append_right _ member)

/-- **Every part is recoverable.**

Stated as the conjunction so the completeness of the restriction is visible in
one place: satisfying the canonical program satisfies each of its eight parts,
built and selected alike. -/
theorem satisfies_every_part (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (rows recipes) z) :
    Satisfies (PiDecRecipe.rows recipes.piDec) z
      ∧ Satisfies (FoldDigestRecipe.digestRows recipes.foldDigest) z
      ∧ Satisfies (CommitmentMixerRecipe.mixerRows recipes.mixerBase
          recipes.mixer) z
      ∧ Satisfies (TranscriptRecipe.transcriptRows recipes.transcriptLayouts
          recipes.transcriptSchedule recipes.transcriptConstants
          recipes.transcriptRounds) z
      ∧ Satisfies recipes.step.rows z
      ∧ Satisfies recipes.nifsVerify.rows z
      ∧ Satisfies recipes.runningCheck.rows z
      ∧ Satisfies recipes.freshCheck.rows z :=
  ⟨satisfies_piDec recipes z satisfied,
    satisfies_foldDigest recipes z satisfied,
    satisfies_mixer recipes z satisfied,
    satisfies_transcript recipes z satisfied,
    satisfies_step recipes z satisfied,
    satisfies_nifsVerify recipes z satisfied,
    satisfies_runningCheck recipes z satisfied,
    satisfies_freshCheck recipes z satisfied⟩

/-! ## Honest completeness for the assembly

The restriction lemmas take satisfaction apart.  This puts it back together, and
it is the direction a deployment needs to conclude its program is satisfiable at
all.

The easy half is that concatenation is conjunction.  The half worth stating is
the hypothesis: **one** assignment must satisfy all eight parts.  Each recipe's
own honest completeness produces its *own* witness — `KLowNormBatch` extends the
caller's on its allocated columns, `TranscriptRecipe` needs its per-round values
— so composing them is only possible because those extensions touch disjoint
columns.  `windows_no_collision` is what makes a common `z` constructible;
without it the eight witnesses could disagree and no assembly would exist. -/

/-- **A common satisfying assignment satisfies the assembly.** -/
theorem rows_honest (recipes : Recipes) (z : Nat → Nat)
    (piDec : Satisfies (PiDecRecipe.rows recipes.piDec) z)
    (foldDigest : Satisfies (FoldDigestRecipe.digestRows recipes.foldDigest) z)
    (mixer : Satisfies (CommitmentMixerRecipe.mixerRows recipes.mixerBase
      recipes.mixer) z)
    (transcript : Satisfies (TranscriptRecipe.transcriptRows
      recipes.transcriptLayouts recipes.transcriptSchedule
      recipes.transcriptConstants recipes.transcriptRounds) z)
    (step : Satisfies recipes.step.rows z)
    (nifsVerify : Satisfies recipes.nifsVerify.rows z)
    (runningCheck : Satisfies recipes.runningCheck.rows z)
    (freshCheck : Satisfies recipes.freshCheck.rows z) :
    Satisfies (rows recipes) z := by
  intro row member
  unfold rows at member
  rcases List.mem_append.1 member with inSeven | inFresh
  · rcases List.mem_append.1 inSeven with inSix | inRunning
    · rcases List.mem_append.1 inSix with inFive | inNifs
      · rcases List.mem_append.1 inFive with inFour | inStep
        · rcases List.mem_append.1 inFour with inThree | inTranscript
          · rcases List.mem_append.1 inThree with inTwo | inMixer
            · rcases List.mem_append.1 inTwo with inPiDec | inFoldDigest
              · exact piDec row inPiDec
              · exact foldDigest row inFoldDigest
            · exact mixer row inMixer
          · exact transcript row inTranscript
        · exact step row inStep
      · exact nifsVerify row inNifs
    · exact runningCheck row inRunning
  · exact freshCheck row inFresh

/-- **Satisfaction of the assembly is exactly satisfaction of every part.**

The two directions together, so the assembly's semantics is pinned rather than
bounded on one side.  The assembly adds no constraint and drops none. -/
theorem rows_iff_every_part (recipes : Recipes) (z : Nat → Nat) :
    Satisfies (rows recipes) z ↔
      (Satisfies (PiDecRecipe.rows recipes.piDec) z
        ∧ Satisfies (FoldDigestRecipe.digestRows recipes.foldDigest) z
        ∧ Satisfies (CommitmentMixerRecipe.mixerRows recipes.mixerBase
            recipes.mixer) z
        ∧ Satisfies (TranscriptRecipe.transcriptRows recipes.transcriptLayouts
            recipes.transcriptSchedule recipes.transcriptConstants
            recipes.transcriptRounds) z
        ∧ Satisfies recipes.step.rows z
        ∧ Satisfies recipes.nifsVerify.rows z
        ∧ Satisfies recipes.runningCheck.rows z
        ∧ Satisfies recipes.freshCheck.rows z) := by
  constructor
  · exact satisfies_every_part recipes z
  · rintro ⟨piDec, foldDigest, mixer, transcript, step, nifs, running, fresh⟩
    exact rows_honest recipes z piDec foldDigest mixer transcript step nifs
      running fresh

/-! ## Every component accounted for

`recurringRows` is proved against the emitted list and `auxiliaryColumns`
against the window layout.  The other two components had no statement at all —
they are sums of the four selections' figures, and that the built recipes
contribute nothing to them was true by construction and nowhere written.

"True by construction" is what a reader has to verify by unfolding four costs.
These two make it checkable instead. -/

/-- The built recipes commit no columns, so this component is the selections'
fold alone. -/
theorem committedColumns_from_selections (recipes : Recipes) :
    (N_canonical recipes).committedColumns
      = recipes.step.cost.committedColumns
        + recipes.nifsVerify.cost.committedColumns
        + recipes.runningCheck.cost.committedColumns
        + recipes.freshCheck.cost.committedColumns := rfl

/-- The built recipes publish no columns, so this component is the selections'
fold alone. -/
theorem publicColumns_from_selections (recipes : Recipes) :
    (N_canonical recipes).publicColumns
      = recipes.step.cost.publicColumns
        + recipes.nifsVerify.cost.publicColumns
        + recipes.runningCheck.cost.publicColumns
        + recipes.freshCheck.cost.publicColumns := rfl

/-- **The built recipes contribute nothing to either committed or public
columns.**

Stated over the four built costs rather than read off `N_canonical`'s
definition, so it is a fact about the recipes and not about how the tuple
happens to be written. -/
theorem built_recipes_allocate_no_public_columns (recipes : Recipes) :
    (PiDecRecipe.cost recipes.piDec).committedColumns = 0
      ∧ (PiDecRecipe.cost recipes.piDec).publicColumns = 0
      ∧ (FoldDigestRecipe.digestCost recipes.foldDigest).committedColumns = 0
      ∧ (FoldDigestRecipe.digestCost recipes.foldDigest).publicColumns = 0
      ∧ (CommitmentMixerRecipe.mixerCost recipes.mixer).committedColumns = 0
      ∧ (CommitmentMixerRecipe.mixerCost recipes.mixer).publicColumns = 0
      ∧ (TranscriptRecipe.transcriptCost recipes.transcriptRounds).committedColumns = 0
      ∧ (TranscriptRecipe.transcriptCost recipes.transcriptRounds).publicColumns = 0 :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- **All four components are receipt folds.**

`recurringRows` from the emitted list, `auxiliaryColumns` from the window
layout, and the two public components from the selections — with
`built_recipes_allocate_no_public_columns` giving the reason the last two omit
the built recipes rather than dropping them. -/
theorem N_canonical_components (recipes : Recipes) :
    (rows recipes).length = (N_canonical recipes).recurringRows
      ∧ (N_canonical recipes).auxiliaryColumns = (counts recipes).sum
      ∧ (N_canonical recipes).committedColumns
          = recipes.step.cost.committedColumns
            + recipes.nifsVerify.cost.committedColumns
            + recipes.runningCheck.cost.committedColumns
            + recipes.freshCheck.cost.committedColumns
      ∧ (N_canonical recipes).publicColumns
          = recipes.step.cost.publicColumns
            + recipes.nifsVerify.cost.publicColumns
            + recipes.runningCheck.cost.publicColumns
            + recipes.freshCheck.cost.publicColumns :=
  ⟨rows_length recipes, auxiliaryColumns_eq_counts_sum recipes,
    committedColumns_from_selections recipes,
    publicColumns_from_selections recipes⟩

/-! ## No column is allocated by two parts

Section 2 item 4 asks for column ownership **with no collision**.  What was
proved was `windows_no_collision` — two windows sharing a column are the same
window — and, separately, that each part's allocation lands in its own window.

Neither is the item.  The item is that no column is allocated by two recipes,
and it is the composition: `ColumnWindows.placed_allocations_disjoint` supplies
the step, `windows_no_collision` supplies its hypothesis.

The conclusion is about **relocated** columns, because a placed program is what
the assembly emits and two parts may perfectly well allocate the same raw column
before relocation — that is precisely what relocation exists to fix. -/

/-- **Two parts with distinct windows share no allocated column once placed.** -/
theorem allocations_disjoint (recipes : Recipes)
    (first second : ColumnWindows.Window)
    (firstMember : first ∈ windows recipes)
    (secondMember : second ∈ windows recipes)
    (different : first ≠ second)
    (firstAllocation secondAllocation : List Nat)
    (firstPlaced : ColumnWindows.AllocationPlaced first.base firstAllocation
      first.width)
    (secondPlaced : ColumnWindows.AllocationPlaced second.base secondAllocation
      second.width)
    (firstColumn : Nat) (firstIn : firstColumn ∈ firstAllocation)
    (secondColumn : Nat) (secondIn : secondColumn ∈ secondAllocation) :
    ColumnWindows.relocate first.base firstColumn
      ≠ ColumnWindows.relocate second.base secondColumn :=
  ColumnWindows.placed_allocations_disjoint first second firstAllocation
    secondAllocation firstPlaced secondPlaced
    (fun column inFirst inSecond =>
      windows_no_collision recipes first firstMember second secondMember column
        inFirst inSecond)
    different firstColumn firstIn secondColumn secondIn

/-- **Only two built parts allocate at all.**

So the assembly's only possible built-part collision is Π_DEC against the
transcript.  Derived from the two recipes' own allocation lemmas rather than read
off their definitions. -/
theorem foldDigest_and_mixer_allocate_nothing (recipes : Recipes) :
    (counts recipes)[1]? = some 0 ∧ (counts recipes)[2]? = some 0 := by
  unfold counts allocations
  simp only [List.map_cons, List.getElem?_cons_succ, List.getElem?_cons_zero,
    Option.some.injEq]
  exact ⟨FoldDigestRecipe.digestColumns_length,
    CommitmentMixerRecipe.mixerColumns_length⟩

/-! ## Conservation for the assembly

Section 2 item 5.  Each built part conserves against its own claim; the assembly
concatenates them, so a column mentioned anywhere in the built program traces to
whichever part emitted the row.

**The four selections are excluded, and that is a statement rather than an
omission.**  `SelectedRecipe` carries rows a deployment supplies and says nothing
about which columns they mention, so no conservation is derivable for them here —
the same interface gap `rows_owner_not_unique` records for item 3.  A deployment
that supplies a selection owes its own conservation, and `built_rows` below is
the part of the program this module can speak for. -/

/-- Every column the four built parts may touch, traced to the claim that
supplied it. -/
def BuiltTouches (recipes : Recipes) (column : Nat) : Prop :=
  PiDecRecipe.Touches recipes.piDec column
    ∨ (∃ pair ∈ recipes.foldDigest,
        Mentions pair.1 column ∨ Mentions pair.2 column ∨ column = 0)
    ∨ (∃ coordinate ∈ recipes.mixer,
        column = 0 ∨ Mentions coordinate.parent column
          ∨ ∃ child ∈ coordinate.children, Mentions child column)
    ∨ (∃ round, round < recipes.transcriptRounds ∧
        (column = 0
          ∨ (∃ source : Fin Poseidon2Core.width,
              Mentions (TranscriptRecipe.entryAt recipes.transcriptLayouts
                recipes.transcriptSchedule round source) column)
          ∨ (∃ index, index < Poseidon2Core.sboxCount
              ∧ column = Poseidon2Schedule.sboxOutput
                  (recipes.transcriptLayouts round) index)
          ∨ (∃ slot : Fin Poseidon2Program.columnsPerSbox,
              ∃ index : Fin Poseidon2Core.sboxCount,
                column = Poseidon2Program.sboxColumn
                  (recipes.transcriptLayouts round) index slot)
          ∨ (∃ lane : Fin Poseidon2Core.width,
              column = (recipes.transcriptLayouts round).outputPort lane)))

/-- The four built parts of the emitted program, in program order. -/
def builtRows (recipes : Recipes) : List Row :=
  PiDecRecipe.rows recipes.piDec
    ++ FoldDigestRecipe.digestRows recipes.foldDigest
    ++ CommitmentMixerRecipe.mixerRows recipes.mixerBase recipes.mixer
    ++ TranscriptRecipe.transcriptRows recipes.transcriptLayouts
        recipes.transcriptSchedule recipes.transcriptConstants
        recipes.transcriptRounds

/-- **No row of the built program reaches a column no claim supplied.** -/
theorem builtRows_conservation
    (recipes : Recipes) (row : Row) (member : row ∈ builtRows recipes)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    BuiltTouches recipes column := by
  unfold BuiltTouches
  unfold builtRows at member
  rcases List.mem_append.1 member with inThree | inTranscript
  · rcases List.mem_append.1 inThree with inTwo | inMixer
    · rcases List.mem_append.1 inTwo with inPiDec | inFoldDigest
      · exact Or.inl (PiDecRecipe.rows_conservation recipes.piDec row inPiDec
          column mentioned)
      · exact Or.inr (Or.inl (FoldDigestRecipe.digestRows_conservation
          recipes.foldDigest row inFoldDigest column mentioned))
    · exact Or.inr (Or.inr (Or.inl
        (CommitmentMixerRecipe.mixerRows_conservation recipes.mixerBase
          recipes.mixer row inMixer column mentioned)))
  · exact Or.inr (Or.inr (Or.inr
      (TranscriptRecipe.transcriptRows_conservation recipes.transcriptLayouts
        recipes.transcriptSchedule recipes.transcriptConstants
        recipes.transcriptRounds row inTranscript column mentioned)))

/-- **The built program is exactly the assembly minus the four selections.**

So `builtRows_conservation` is a statement about `rows` on the part this module
owns, and not about a different program that happens to look similar. -/
theorem rows_eq_builtRows_append_selections (recipes : Recipes) :
    rows recipes
      = builtRows recipes
          ++ recipes.step.rows ++ recipes.nifsVerify.rows
          ++ recipes.runningCheck.rows ++ recipes.freshCheck.rows := by
  unfold rows builtRows
  simp only [List.append_assoc]

/-! ## Is the auxiliary count a count of the emitted program's columns?

`auxiliaryColumns_eq_counts_sum` relates the cost tuple to the **declared**
allocations.  Conservation bounds the emitted rows' columns from above.  Neither
says a declared column is one the program actually uses, and without that the
count is a figure about a declaration rather than about an emitted row program —
which is the first thing the trap list rules out.

Both built allocating parts are now exact.  `PiDecRecipe.columns_exact` gives
both inclusions for Π_DEC.  For the transcript,
`TranscriptRecipe.transcriptColumns_eq_canonical_sbox` shows the declared list is
exactly `canonicalLayouts`' S-box columns — and no other layout's, by
`transcriptColumns_not_layout_generic` — while
`TranscriptRecipe.transcriptColumns_written` shows each is written by an emitted
row.

The selections' allocations remain the deployment's, as everywhere else.

The two selections' allocations are the deployment's, as everywhere else. -/

/-- **Π_DEC's declared columns are used by the built program.** -/
theorem builtRows_use_piDec_columns (recipes : Recipes)
    (column : Nat) (member : column ∈ PiDecRecipe.columns recipes.piDec) :
    ∃ row ∈ builtRows recipes, Mentions row.c column := by
  rcases PiDecRecipe.rows_use_columns recipes.piDec column member with
    ⟨row, rowMember, mentions⟩
  refine ⟨row, ?_, mentions⟩
  unfold builtRows
  exact List.mem_append_left _ (List.mem_append_left _
    (List.mem_append_left _ rowMember))

/-! ## Row ownership for the assembly

Section 2 item 3 was settled for the six recipes.  The assembly is a seventh
object and its status was never recorded, which is the gap this section closes.

**Both of its reasons are about row values**, and value equality is not the
ownership ABI — `Poseidon2Ownership`'s header settled that, and
`CanonicalProgramOwnership` applies it here.

First, inherited: `PiDecRecipe.rows_owner_not_unique` exhibits a row attributable
to two of its own receipts.  Answered in cycle 393 by
`PiDecOwnership.ownership_is_positional`.

Second, and new at this level: **a selection's rows are unconstrained.**  A
deployment may supply a `SelectedRecipe` whose rows coincide with another
selection's, or with a built recipe's.  That was recorded as not inherited, not
repairable by strengthening the built recipes, and a property of the interface a
deployment fills.  All three are true *of value-based ownership*.  Under
positional ownership a duplicating selection occupies different positions, so its
rows are owned separately and nothing needs to be prevented — see
`CanonicalProgramOwnership.duplicating_selection_has_distinct_receipts`.

A deployment supplying duplicate rows may still be a *waste*, the same constraint
asserted twice, but that is a cost question and `N_canonical` counts both. -/

/-- **A selection may duplicate a built recipe's rows.**

Take the fold-digest program as a selection's rows.  Nothing rejects it, and
every row is then attributable to two parts of the assembly.  The witness needs
no exotic construction — it is the interface behaving as specified. -/
theorem selection_may_duplicate_built_rows
    (recipes : Recipes)
    (duplicated : recipes.step.rows
      = FoldDigestRecipe.digestRows recipes.foldDigest) :
    ∀ row ∈ FoldDigestRecipe.digestRows recipes.foldDigest,
      row ∈ recipes.step.rows
        ∧ row ∈ FoldDigestRecipe.digestRows recipes.foldDigest := by
  intro row member
  exact ⟨duplicated ▸ member, member⟩

/-- **The assembly does not meet item 3**, and the obligation is the
deployment's.

Stated as the two reasons rather than one, because they are discharged
differently: the first by whatever supplies Π_DEC's checks, the second by
whatever supplies the selections. -/
theorem rows_owner_not_unique :
    ∃ claim : PiDecRecipe.Decomposition, ∃ row : Row,
      row ∈ PiDecRecipe.rows claim
        ∧ row ∈ KRecomposition.recompositionRows claim.base
            KRecomposition.witnessCheckA.1 KRecomposition.witnessCheckA.2
        ∧ row ∈ KRecomposition.recompositionRows claim.base
            KRecomposition.witnessCheckB.1 KRecomposition.witnessCheckB.2 := by
  refine ⟨PiDecRecipe.witnessDecomposition, ⟨[(1, 1)], [(0, 1)], [(3, 1)]⟩,
    ?_, ?_, ?_⟩ <;> decide

/-! ## `rows` is the unplaced program, `placedRows` the placed one

`rows` concatenates its parts **unrelocated** — it calls neither `renameRow` nor
`ColumnWindows.placeAll`.  That is not a gap; it is what the two objects are for.
`placedRows` below is the program the layout describes, and both are kept because
each recipe's own soundness theorem is stated about *its* rows, which is what
`rows` contains verbatim.

`auxiliaryColumns_eq_counts_sum` relates the cost tuple to the **counts**, and
`widths` to the layout's spans.  Those are different quantities and neither is
used for the other's job. -/

/-! ## Placement, uniform across the eight parts

`ColumnWindows.AllocationPlaced` asks that a part's declared allocation land in
its window after relocation.  With `SelectedRecipe.allocation` in place the
obligation is now *statable* for every part, which it was not before.

The four reading recipes discharge it vacuously; `KLowNormBatch` discharges it
from a bound; a selection discharges it by supplying one.  That uniformity is
what `SelectedRecipe`'s missing field was blocking. -/

/-! ## Why windows are sized by span and not by count

Discharging `ColumnWindows.AllocationPlaced` part by part once reached the
transcript and stopped.

`ColumnWindows.Window` owns a **contiguous** range `(base, base + width]`.  Sized
by `cost.auxiliaryColumns` — a **count** — a window holds an allocation only when
that allocation is contiguous.  The transcript's is not: round `r` allocates 344
columns starting at `r · 369 + 17`, so at one round the allocation is `17 … 360`,
**344 columns spanning 361**.  A window of width 344 ends at column 344, and
column 360 is allocated and outside it.  The sponge stride leaves gaps for chunk
columns, so no bound on the count places it.

`CANONICAL-PROGRAM-WINDOW-SIZING` recorded the choice and it is now made:
`widths` sizes by `ColumnWindows.spanOf`.  The cost tuple keeps counting columns
— `auxiliaryColumns_eq_counts_sum` — and `transcript_placed` is the sparse case
the count-sized window could not hold. -/

/-! ## The placed program

`CANONICAL-PROGRAM-LAYOUT-UNAPPLIED` recorded that `rows` concatenates its parts
unrelocated while `windows` described a layout nothing used.  `placedRows` is
the program the layout describes.

Relocation drops no row and adds none, so the row count is unchanged and every
row-side theorem — `rows_length`, `rows_iff_every_part`, the eight restriction
lemmas — continues to describe the same program up to renaming.

What the layout governs is **allocations**, not every column a row mentions.
Reads cross windows by design: a part reads carried values another part owns,
and `CANONICAL-PROGRAM-PLACEMENT-IS-ABOUT-ALLOCATIONS` records why no syntactic
field distinguishes a read from a write here. -/

/-- Each part paired with its window width. -/
def parts (recipes : Recipes) : List (List Row × Nat) :=
  [ (PiDecRecipe.rows recipes.piDec,
      ColumnWindows.spanOf (PiDecRecipe.columns recipes.piDec)),
    (FoldDigestRecipe.digestRows recipes.foldDigest,
      ColumnWindows.spanOf FoldDigestRecipe.digestColumns),
    (CommitmentMixerRecipe.mixerRows recipes.mixerBase recipes.mixer,
      ColumnWindows.spanOf CommitmentMixerRecipe.mixerColumns),
    (TranscriptRecipe.transcriptRows recipes.transcriptLayouts
        recipes.transcriptSchedule recipes.transcriptConstants
        recipes.transcriptRounds,
      ColumnWindows.spanOf
        (TranscriptRecipe.transcriptColumns recipes.transcriptRounds)),
    (recipes.step.rows, ColumnWindows.spanOf recipes.step.allocation),
    (recipes.nifsVerify.rows, ColumnWindows.spanOf recipes.nifsVerify.allocation),
    (recipes.runningCheck.rows,
      ColumnWindows.spanOf recipes.runningCheck.allocation),
    (recipes.freshCheck.rows,
      ColumnWindows.spanOf recipes.freshCheck.allocation) ]

/-- **The placed canonical program.** -/
def placedRows (recipes : Recipes) : List Row :=
  ColumnWindows.placeAll 0 (parts recipes)

/-- The part widths are the layout's widths, so `placedRows` and `windows`
describe the same layout. -/
theorem parts_widths (recipes : Recipes) :
    (parts recipes).map Prod.snd = widths recipes := by
  simp only [parts, widths, allocations, List.map_cons, List.map_nil]

/-- **Relocation drops no row and adds none.** -/
theorem placeAll_length (base : Nat) (someParts : List (List Row × Nat)) :
    (ColumnWindows.placeAll base someParts).length
      = (someParts.map (fun part => part.1.length)).sum := by
  induction someParts generalizing base with
  | nil => rfl
  | cons part rest inductionHypothesis =>
      simp only [ColumnWindows.placeAll, List.length_append, List.length_map,
        List.map_cons, List.sum_cons, inductionHypothesis]

/-- **The placed program has the same row count as the concatenated one.**

So `N_canonical.recurringRows` describes the placed program too, and the
placement changes where rows sit rather than how many there are. -/
theorem placedRows_length (recipes : Recipes) :
    (placedRows recipes).length = (rows recipes).length := by
  unfold placedRows parts rows
  rw [placeAll_length]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    List.length_append]
  omega

/-- **The placed program's row count is `N_canonical`'s.** -/
theorem placedRows_length_eq (recipes : Recipes) :
    (placedRows recipes).length = (N_canonical recipes).recurringRows := by
  rw [placedRows_length, rows_length]

/-! ## The placed program's semantics

`placedRows_length` gave the placed program a row count.  These give it
soundness: each part is recovered under its own pullback, so a deployment that
satisfies the placed program satisfies every part exactly as it does for
`rows`. -/

/-- **Π_DEC is recovered from the placed program.** -/
theorem placed_satisfies_piDec (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (placedRows recipes) z) :
    Satisfies (PiDecRecipe.rows recipes.piDec)
      (pullAssignment z (ColumnWindows.relocate 0)) :=
  ColumnWindows.placeAll_satisfies_head 0 _ _ z satisfied

/-- **The fold digest is recovered**, at its own base. -/
theorem placed_satisfies_foldDigest (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (placedRows recipes) z) :
    Satisfies (FoldDigestRecipe.digestRows recipes.foldDigest)
      (pullAssignment z (ColumnWindows.relocate
        (0 + ColumnWindows.spanOf (PiDecRecipe.columns recipes.piDec)))) :=
  ColumnWindows.placeAll_satisfies_head _ _ _ z
    (ColumnWindows.placeAll_satisfies_tail 0 _ _ z satisfied)

/-! ### All eight, not the first two

`placed_has_semantics` stated two parts and said the remaining six were "the
same pattern" — the unplaced twin `satisfies_every_part` states all eight.  A
recipe recovered from `rows` and not from `placedRows` has no soundness under the
layout the assembly publishes, so the six omitted parts were six recipes whose
placed semantics nothing asserted.

The reason given for stopping — further applications of
`placeAll_satisfies_tail` — described the route before
`placeAll_satisfies_index` existed.  With the indexed form each part is one
application at its own `baseAt`, and the six are no longer a pattern to be
trusted. -/

/-- **Every part of the placed program is recovered at the base the placement
assigns it.**

The general form: `placedRows` is a `placeAll` at base zero over `parts`, so
`ColumnWindows.placeAll_satisfies_index` applies directly. -/
theorem placed_satisfies_at (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (placedRows recipes) z)
    (index : Nat) (part : List Row × Nat)
    (found : (parts recipes)[index]? = some part) :
    Satisfies part.1 (pullAssignment z
      (ColumnWindows.relocate (ColumnWindows.baseAt 0 (parts recipes) index))) :=
  ColumnWindows.placeAll_satisfies_index z (parts recipes) 0 index part found
    satisfied

theorem placed_satisfies_mixer (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (placedRows recipes) z) :
    Satisfies (CommitmentMixerRecipe.mixerRows recipes.mixerBase recipes.mixer)
      (pullAssignment z (ColumnWindows.relocate
        (ColumnWindows.baseAt 0 (parts recipes) 2))) :=
  placed_satisfies_at recipes z satisfied 2 _ rfl

theorem placed_satisfies_transcript (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (placedRows recipes) z) :
    Satisfies (TranscriptRecipe.transcriptRows recipes.transcriptLayouts
        recipes.transcriptSchedule recipes.transcriptConstants
        recipes.transcriptRounds)
      (pullAssignment z (ColumnWindows.relocate
        (ColumnWindows.baseAt 0 (parts recipes) 3))) :=
  placed_satisfies_at recipes z satisfied 3 _ rfl

theorem placed_satisfies_step (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (placedRows recipes) z) :
    Satisfies recipes.step.rows
      (pullAssignment z (ColumnWindows.relocate
        (ColumnWindows.baseAt 0 (parts recipes) 4))) :=
  placed_satisfies_at recipes z satisfied 4 _ rfl

theorem placed_satisfies_nifsVerify (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (placedRows recipes) z) :
    Satisfies recipes.nifsVerify.rows
      (pullAssignment z (ColumnWindows.relocate
        (ColumnWindows.baseAt 0 (parts recipes) 5))) :=
  placed_satisfies_at recipes z satisfied 5 _ rfl

theorem placed_satisfies_runningCheck (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (placedRows recipes) z) :
    Satisfies recipes.runningCheck.rows
      (pullAssignment z (ColumnWindows.relocate
        (ColumnWindows.baseAt 0 (parts recipes) 6))) :=
  placed_satisfies_at recipes z satisfied 6 _ rfl

theorem placed_satisfies_freshCheck (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (placedRows recipes) z) :
    Satisfies recipes.freshCheck.rows
      (pullAssignment z (ColumnWindows.relocate
        (ColumnWindows.baseAt 0 (parts recipes) 7))) :=
  placed_satisfies_at recipes z satisfied 7 _ rfl

/-- **Every one of the eight parts is recoverable from the placed program.**

The placed counterpart of `satisfies_every_part`, and stated at the same width:
eight conjuncts, each at the base `placeAll` assigns that part. -/
theorem placed_satisfies_every_part (recipes : Recipes) (z : Nat → Nat)
    (satisfied : Satisfies (placedRows recipes) z) :
    Satisfies (PiDecRecipe.rows recipes.piDec)
        (pullAssignment z (ColumnWindows.relocate
          (ColumnWindows.baseAt 0 (parts recipes) 0)))
      ∧ Satisfies (FoldDigestRecipe.digestRows recipes.foldDigest)
        (pullAssignment z (ColumnWindows.relocate
          (ColumnWindows.baseAt 0 (parts recipes) 1)))
      ∧ Satisfies (CommitmentMixerRecipe.mixerRows recipes.mixerBase
          recipes.mixer)
        (pullAssignment z (ColumnWindows.relocate
          (ColumnWindows.baseAt 0 (parts recipes) 2)))
      ∧ Satisfies (TranscriptRecipe.transcriptRows recipes.transcriptLayouts
          recipes.transcriptSchedule recipes.transcriptConstants
          recipes.transcriptRounds)
        (pullAssignment z (ColumnWindows.relocate
          (ColumnWindows.baseAt 0 (parts recipes) 3)))
      ∧ Satisfies recipes.step.rows
        (pullAssignment z (ColumnWindows.relocate
          (ColumnWindows.baseAt 0 (parts recipes) 4)))
      ∧ Satisfies recipes.nifsVerify.rows
        (pullAssignment z (ColumnWindows.relocate
          (ColumnWindows.baseAt 0 (parts recipes) 5)))
      ∧ Satisfies recipes.runningCheck.rows
        (pullAssignment z (ColumnWindows.relocate
          (ColumnWindows.baseAt 0 (parts recipes) 6)))
      ∧ Satisfies recipes.freshCheck.rows
        (pullAssignment z (ColumnWindows.relocate
          (ColumnWindows.baseAt 0 (parts recipes) 7))) :=
  ⟨placed_satisfies_at recipes z satisfied 0 _ rfl,
    placed_satisfies_at recipes z satisfied 1 _ rfl,
    placed_satisfies_mixer recipes z satisfied,
    placed_satisfies_transcript recipes z satisfied,
    placed_satisfies_step recipes z satisfied,
    placed_satisfies_nifsVerify recipes z satisfied,
    placed_satisfies_runningCheck recipes z satisfied,
    placed_satisfies_freshCheck recipes z satisfied⟩

/-- **The placed program is satisfied when every part is.**

The converse of `placeAll_satisfies_index` at the assembly, so `placedRows` has
both directions and not only the one cycle 379 needed. -/
theorem placedRows_honest (recipes : Recipes) (z : Nat → Nat)
    (each : ∀ (index : Nat) (part : List Row × Nat),
      (parts recipes)[index]? = some part →
      Satisfies part.1 (pullAssignment z
        (ColumnWindows.relocate (ColumnWindows.baseAt 0 (parts recipes) index)))) :
    Satisfies (placedRows recipes) z :=
  ColumnWindows.placeAll_honest z (parts recipes) 0 each

/-! ## Allocations placed at the bases the placement uses

The `*_placed` theorems above are universally quantified over `base`, so they
apply at any base — including the ones `placeAll` uses.  **Nothing said so.**

That is the same disconnect cycle 369 found between `rows` and `windows`, one
level in: two families of theorems about the same layout, correct separately and
never related.  A general theorem without its instantiation leaves the record
with no statement that *this* allocation lands in *this* program's window.

These instantiate at `ColumnWindows.baseAt 0 (parts recipes) index`, which is
where `placeAll` puts part `index`.  No new mathematics — the content is the
connection. -/

/-- **The fold digest's allocation is placed where the placement puts it.** -/
theorem foldDigest_placed_at_its_base (recipes : Recipes) :
    ColumnWindows.AllocationPlaced
      (ColumnWindows.baseAt 0 (parts recipes) 1)
      FoldDigestRecipe.digestColumns
      (ColumnWindows.spanOf FoldDigestRecipe.digestColumns) :=
  foldDigest_placed _

/-- **The mixer's allocation is placed where the placement puts it.** -/
theorem mixer_placed_at_its_base (recipes : Recipes) :
    ColumnWindows.AllocationPlaced
      (ColumnWindows.baseAt 0 (parts recipes) 2)
      CommitmentMixerRecipe.mixerColumns
      (ColumnWindows.spanOf CommitmentMixerRecipe.mixerColumns) :=
  mixer_placed _

/-- **The transcript's allocation is placed where the placement puts it.**

The sparse one — 344 columns spanning 361 at a round — now tied to the base
`placeAll` assigns it rather than to an arbitrary one. -/
theorem transcript_placed_at_its_base (recipes : Recipes) :
    ColumnWindows.AllocationPlaced
      (ColumnWindows.baseAt 0 (parts recipes) 3)
      (TranscriptRecipe.transcriptColumns recipes.transcriptRounds)
      (ColumnWindows.spanOf
        (TranscriptRecipe.transcriptColumns recipes.transcriptRounds)) :=
  transcript_placed recipes _

/-- **Π_DEC's allocation is placed where the placement puts it**, given its
digits allocate real columns. -/
theorem piDec_placed_at_its_base (recipes : Recipes)
    (noWire : ∀ column ∈ PiDecRecipe.columns recipes.piDec, column ≠ 0) :
    ColumnWindows.AllocationPlaced
      (ColumnWindows.baseAt 0 (parts recipes) 0)
      (PiDecRecipe.columns recipes.piDec)
      (ColumnWindows.spanOf (PiDecRecipe.columns recipes.piDec)) :=
  piDec_placed recipes _ noWire

/-- **A selection's allocation is placed where the placement puts it**, given
the deployment's columns are real. -/
theorem selection_placed_at_its_base (recipes : Recipes) (index : Nat)
    (selection : SelectedRecipe)
    (noWire : ∀ column ∈ selection.allocation, column ≠ 0) :
    ColumnWindows.AllocationPlaced
      (ColumnWindows.baseAt 0 (parts recipes) index)
      selection.allocation
      (ColumnWindows.spanOf selection.allocation) :=
  selection_placed selection _ noWire

end Recipes

end Nightstream.Implementation.R1CS.Canonical.CanonicalProgram
