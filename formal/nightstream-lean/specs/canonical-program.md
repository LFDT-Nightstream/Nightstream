# CANONICAL-PROGRAM — the assembly and `N_canonical`

Owner: `Nightstream/Implementation/R1CS/Canonical/CanonicalProgram.lean`
Guard: `tests/Axioms/CanonicalCanonicalProgram.lean` (6 theorems, measured)
Evidence: `model-proved`

---

## CANONICAL-PROGRAM-EXISTS

```text
claim:
  allRecipes, CanonicalProgram and N_canonical now exist. Before cycle 342
  none of the three existed anywhere in the tree.
status: PROVED 2026-07-27 (cycle 342).
```

The phase plan names all three. Grepping for them returned only
`Poseidon2Normalized.normalizedCanonicalProgram`, which is the permutation and
unrelated.

They had been reported as "blocked pending selection" for many cycles. That was
true of four **components** and not of the **assembly**, and the distinction was
never checked. It is the same defect as the four inherited Rust dependencies
this session corrected — with the difference that this one was self-inflicted.

## CANONICAL-PROGRAM-SELECTIONS-ARE-PROOF-CARRYING

```text
claim:
  The four setup selections enter as rows, a cost, and a certificate that they
  match.
status: PROVED. SelectedRecipe.
```

`step`, `nifsVerify`, `runningCheck` and `freshCheck` are setup selections;
`TERMINAL-CHECK-SELECTION` proves two lawful checkers disagree on the same
argument, so no encoding derives them.

They therefore enter as `SelectedRecipe`, whose `rowsCertified` field proves the
supplied rows match the supplied cost. That is **not** an obligation moved to a
premise no consumer constructs: the consumer is a deployment that has made the
selection, and it must supply the certificate with the recipe. A selection
without one is not a recipe.

The blockage is now exactly four proof-carrying inputs, rather than an unbounded
"downstream of the selection".

## CANONICAL-PROGRAM-ROW-FOLD

```text
claim:
  N_canonical's row component is derived from the emitted list.
status: PROVED. rows_length, rows_length_built_only.
```

`rows_length` proves `(rows recipes).length = (N_canonical recipes).recurringRows`.
The four built recipes contribute their own proved counts; the four selections
contribute theirs through `rowsCertified`. Nothing is measured.

`rows_length_built_only` states the same count with the four selections empty,
which is what the assembly buys: they are isolated to four summands, so
everything else is stated and proved today rather than waiting on a decision.

Soundness restricts to every part — `satisfies_piDec`, `satisfies_foldDigest`,
`satisfies_mixer`, `satisfies_transcript`, `satisfies_step`,
`satisfies_nifsVerify`, `satisfies_runningCheck`, `satisfies_freshCheck`, with
`satisfies_every_part` stating the eight together. Each recipe's own soundness
theorem applies unchanged; the assembly adds no constraint of its own.

### Corrected 2026-07-27 (cycle 353)

The four **built** parts had restriction lemmas from cycle 342; the four
**selections** did not. That made `SelectedRecipe` an interface a deployment
could put rows *into* and get no conclusions *out of* — a selection could be
assembled and then never used, which is not what a proof-carrying interface is
for.

The asymmetry was invisible because the four that existed were the four that had
been interesting to write. A deployment supplying `step` can now conclude its own
rows hold from the whole program holding, and apply whatever soundness theorem it
proved about them.

---

## CANONICAL-PROGRAM-COLUMN-DISJOINTNESS

```text
claim:
  The column components of N_canonical are folded but not proved
  collision-free across recipes.
status: CLOSED 2026-07-27 (cycle 344) at the general level. The layout is
  collision-free and placement into it is proved; instantiating each recipe
  with a 1..width numbering is a deployment step, not a theorem.
```

### Proved (cycle 343)

`ColumnWindows` lays widths out consecutively from a base and proves any column
belongs to at most one window — `windowsOf_no_collision` — with the constant
wire outside every window of a layout based at zero.

`CanonicalProgram.widths` gives the eight recipes' auxiliary widths in program
order, `windows` lays them out, and:

| theorem | content |
|---|---|
| `windows_no_collision` | no two recipes' windows share a column |
| `windows_exclude_constantWire` | the constant wire belongs to no recipe |
| `auxiliaryColumns_eq_counts_sum` | `N_canonical`'s auxiliary component is the **counts**' sum, which is what a count component means; the layout's spans are `widths`, a different quantity |

So under this layout the auxiliary total **is** a distinct-column count rather
than a sum of per-recipe figures.

`Poseidon2HashRecipe` proved two instances disjoint by hand. That does not scale
to eight and produces no statement a future recipe can be added to. `windowsOf`
does: adding a recipe means adding a width, not extending a proof.

### Placement, proved (cycle 344)

`ColumnWindows.placeAll` relocates a sequence of programs into consecutive
windows, and `placeAll_columns` proves every column of the placed program is
either the constant wire or lies in one of the windows. With
`windowsOf_no_collision` supplying uniqueness, that is the section-2 "exact
column ownership with no collision" item for an assembled program: the
existential gives membership, the collision theorem gives at-most-one.

`relocate` is the same map `Poseidon2HashRecipe` used for its two hash
instances, lifted so it serves any number of recipes.

### What the caller owes

`placeAll_columns` takes a `bounded` hypothesis: every column a recipe mentions
is the constant wire or is `≤ that recipe's declared width`. That is the
recipe's own conservation restated against its width, and it is not an
obligation moved somewhere no one constructs — **a recipe that cannot state one
has not declared a width, and a program cannot be laid out from widths it does
not have.**

What remains is per-recipe and mechanical: each recipe currently takes
caller-supplied column parameters, so numbering its columns `1 … width` is the
deployment's step, not a missing theorem. The general result no longer has to be
re-proved for each one.

### The original text, for the record

Each recipe's own column ownership is proved — `KLowNormBatch`'s allocation is
`Nodup`, `Poseidon2HashRecipe`'s two windows are disjoint, and the
allocation-free recipes allocate nothing. What is **not** proved is that the
allocations of *different* recipes do not overlap.

Until it is, `N_canonical`'s column components are a sum of per-recipe counts
rather than a count of distinct columns, and the two coincide only under
disjointness. Reporting the tuple without this caveat would be a subtotal
presented as a total.

The route is the one `Poseidon2HashRecipe.relocate` already uses: place each
recipe in its own base-addressed window and derive disjointness from the window
arithmetic. It is not done.


---

## CANONICAL-PROGRAM-HONEST-COMPLETENESS

```text
claim:
  Satisfaction of the assembly is exactly satisfaction of every part.
status: PROVED 2026-07-27 (cycle 354). rows_honest, rows_iff_every_part.
evidence: model-proved
```

Cycle 353 completed the restriction lemmas — satisfaction taken apart. This puts
it back together, which is the direction a deployment needs to conclude its
program is satisfiable at all. Without it the assembly was bounded on one side:
it could be shown to imply each part, never to follow from them.

`rows_iff_every_part` states both directions, so the assembly's semantics is
pinned rather than bounded. **The assembly adds no constraint and drops none.**

### The hypothesis is the content

The easy half is that concatenation is conjunction. The half worth stating is
that **one** assignment must satisfy all eight parts.

Each recipe's own honest completeness produces its *own* witness —
`KLowNormBatch` extends the caller's assignment on its allocated columns,
`TranscriptRecipe` needs its per-round values. Composing them is possible only
because those extensions touch disjoint columns.
`CANONICAL-PROGRAM-COLUMN-DISJOINTNESS` is what makes a common `z`
constructible; without it the eight witnesses could disagree and no assembly
would exist.

So the two open items of cycles 342–344 were not independent: the column layout
was a prerequisite for this theorem being usable, not merely a cost-accounting
nicety.


---

## CANONICAL-PROGRAM-EVERY-COMPONENT

```text
claim:
  All four components of N_canonical are receipt folds, and each has a
  theorem.
status: PROVED 2026-07-27 (cycle 355). N_canonical_components.
evidence: model-proved
```

The DONE condition asks for `N_canonical` "stated as a full `Typed.Cost` tuple
whose **every component** is a receipt fold". Two of the four had theorems:

| component | theorem | since |
|---|---|---|
| `recurringRows` | `rows_length`, against the emitted list | cycle 342 |
| `auxiliaryColumns` | `auxiliaryColumns_eq_counts_sum`, against the per-part counts | cycle 343, corrected cycle 376 |
| `committedColumns` | `committedColumns_from_selections` | **cycle 355** |
| `publicColumns` | `publicColumns_from_selections` | **cycle 355** |

The last two are the selections' fold alone. That the built recipes contribute
nothing to them was **true by construction and nowhere written** — a reader had
to verify it by unfolding four costs.
`built_recipes_allocate_no_public_columns` states it over the four built costs
rather than reading it off `N_canonical`'s definition, so it is a fact about the
recipes and not about how the tuple happens to be spelled.

### Why this was the last piece in reach

`recurringRows` and `auxiliaryColumns` were interesting: one needed the emitted
list, the other needed a disjointness argument. The other two were arithmetic
over zeros, which is exactly the kind of component that gets skipped because
proving it teaches nobody anything — and then the tuple is two-quarters
unaccounted while reading as complete.


---

## CANONICAL-PROGRAM-ROW-OWNERSHIP

```text
claim:
  The assembly does not meet §2 item 3, for two independent reasons, and the
  second is new at this level.
status: OBSTRUCTION, kernel-checked 2026-07-27 (cycle 368).
  rows_owner_not_unique, selection_may_duplicate_built_rows.
```

Item 3 was settled for the six recipes in cycles 357–363. **The assembly is a
seventh object and its status had never been recorded.**

### Reason one: inherited

`PiDecRecipe.rows_owner_not_unique` exhibits a row attributable to two of its
own receipts, and the assembly contains `PiDecRecipe.rows` unchanged. An
assembly cannot recover an attribution its parts do not support.

### Reason two: new at this level

**A selection's rows are unconstrained.** A deployment may supply a
`SelectedRecipe` whose rows coincide with another selection's, or with a built
recipe's — `selection_may_duplicate_built_rows` takes the fold-digest program as
`step`'s rows and nothing rejects it.

`SelectedRecipe` carries rows, a cost, and a certificate relating them. It says
nothing about *what the rows are*, and that is deliberate: constraining them
would mean deciding what a selection may be, which is the decision this layer
does not make.

This reason is worth separating from the first: it is **not inherited**, not
repairable by strengthening the built recipes, and a property of the interface a
deployment fills.

### What this changes about "the interface is complete"

Cycles 353–367 reported the assembly's interface as complete on both sides —
accepts a selection, recovers it, composes it, folds its cost. That was true of
those four capabilities and said nothing about attribution, which was never
checked at this level.

The accurate statement: **the assembly supports every capability a deployment
needs to use a selection, and does not support attributing a row to one.** The
second is a caller obligation at both of its sources.


---

## CANONICAL-PROGRAM-LAYOUT-UNAPPLIED

```text
claim:
  widths and windows describe a plan; rows does not use it. The two are
  independent objects.
status: CLOSED 2026-07-27 (cycle 377). placedRows is the program the layout
  describes; rows remains as the unplaced form.
```

### Closed

`placedRows = ColumnWindows.placeAll 0 (parts recipes)` is the program the
layout describes, and `parts_widths` proves its part widths **are** the layout's
widths — so `placedRows` and `windows` describe the same object rather than two
independent ones.

| theorem | content |
|---|---|
| `placeAll_length` | relocation drops no row and adds none |
| `placedRows_length` | the placed program has the concatenated one's row count |
| `placedRows_length_eq` | that count is `N_canonical.recurringRows` |

So every row-side result — `rows_length`, `rows_iff_every_part`, the eight
restriction lemmas — describes the placed program up to renaming, and the
placement changes where rows sit rather than how many there are.

### What the layout governs, precisely

**Allocations, not every column a row mentions.** Reads cross windows by design:
a part reads carried values another part owns.
`CANONICAL-PROGRAM-PLACEMENT-IS-ABOUT-ALLOCATIONS` records why no syntactic
field of `Row` distinguishes a read from a write, which is why this is the
strongest true form and not a weakening.

### The original text, for the record

`rows` concatenates its eight parts **unrelocated** — it calls neither
`renameRow` nor `ColumnWindows.placeAll`. `emitted_column_outside_every_window`
exhibits a `Recipes` whose sole allocated column is `1000000` while the layout
assigns its Π_DEC part a window of width one: **no window owns the column the
program emits.**

### How `auxiliaryColumns_eq_counts_sum` should be read

It relates the cost tuple to the per-part **counts**. `N_canonical.auxiliaryColumns`
is a number of columns; where those columns sit is `widths`, and the two coincide
only for a contiguous allocation.

Cycles 343–344 proved the layout collision-free and the placement sound
generically, and cycle 344 recorded that instantiating each recipe into its
window remained. **This is that remainder seen from the assembly**: the generic
tool exists and nothing here applies it. Reporting the auxiliary component as a
property of the emitted program — which earlier cycles did — was a subtotal
presented as a total.

### What closing it needs

`ColumnWindows.placeAll_columns` takes a `bounded` hypothesis per part: every
column a part mentions is the constant wire or at most that part's width. The
parts take caller-supplied columns and none supplies such a bound today, so
`rows` cannot simply become `placeAll` without first giving each part a
canonical numbering.

`KLowNormBatch.canonicalDigits` and `TranscriptRecipe.canonicalLayouts` are two
of the six; the other four are `KRecomposition`, `PiDecRecipe`,
`FoldDigestRecipe` and `CommitmentMixerRecipe`, all of which read
caller-supplied combinations rather than allocating.


---

## CANONICAL-PROGRAM-PLACEMENT-TOOL

```text
claim:
  placeAll_columns cannot be applied to reading recipes; placeAll_targets can.
status: PROVED 2026-07-27 (cycle 370). ColumnWindows.placeAll_targets.
```

Attempting cycle 369's queue head — give the four caller-column recipes a
canonical numbering, then apply `placeAll` — found the tool wrong for the job.

`placeAll_columns` bounds **every** column a part mentions by that part's width.
That is right for a part owning everything it touches, and wrong for a part that
*reads* a value another part owns. `FoldDigestRecipe` compares caller-supplied
lane combinations, `KRecomposition` reads children and a parent,
`CommitmentMixerRecipe` reads commitment coordinates — none allocates, so
bounding their mentioned columns by their own width, zero, is **unsatisfiable**.
No canonical numbering would have helped.

### The fix is the one the transcript already found

Track the target, not every column. A row's `c` field names what the row
*writes*, and writes are what a window owns. Reads cross windows legitimately —
that is what makes a program a program rather than a list of independent parts.

`placeAll_targets` is `placeAll_columns` restricted to `c`. It is the form an
assembly of reading parts can use, and it is what
`CANONICAL-PROGRAM-LAYOUT-UNAPPLIED` needed before `rows` could be connected to
`windows`. **That connection was made in cycle 377** — `placedRows` is the
program the layout describes — so this entry records the tool's shape rather
than an outstanding need.

### What remains

Each part must supply its **target** bound: every column its rows write is the
constant wire or at most its width. The allocating recipes have this in reach —
`KLowNormBatch.canonicalDigits_column_le` is exactly it. The reading recipes
write to caller-supplied columns, so their target bound is a caller obligation
of the same kind as everything else those recipes take from callers.

That the tool was wrong is worth more than the numbering would have been: a
canonical numbering for a reading recipe would have been a fabricated allocation
for something that allocates nothing.


---

## CANONICAL-PROGRAM-PLACEMENT-IS-ABOUT-ALLOCATIONS

```text
claim:
  No syntactic field of Row identifies what a row writes. Placement is about
  declared allocations, not about rows.
status: PROVED 2026-07-27 (cycle 371). AllocationPlaced,
  allocationPlaced_nil, allocationPlaced_of_bounded.
```

`placeAll_targets` was written on the reading that a row's `c` field names what
the row writes. Applying it to `KLowNormBatch` shows that is false:

```text
⟨value, value, [(column, 1)]⟩      -- c is the allocated column
⟨[(column, 1)], value, value⟩      -- c is the checked combination
```

The second row's `c` is `value` — a **read**. In `A·z * B·z = C·z` the `C` side
is a linear combination like the others, and `KEquality.equalityRow left right`
has `c = right`, also a read.

### What placement actually needs

Each recipe already declares its allocation separately — `batchColumns`,
`recompositionColumns`, `digestColumns`, `mixerColumns`. Placement should
relocate a part and check that **its declared allocation** lands in its window:

- vacuous for the allocation-free recipes (`allocationPlaced_nil`);
- one bound for the one that allocates (`allocationPlaced_of_bounded`, whose
  shape is exactly `KLowNormBatch.canonicalDigits_column_le`).

A statement about allocations, needing neither `placeAll_columns`' unsatisfiable
read bound nor `placeAll_targets`' false premise about `c`.

### Both earlier tools are kept, with their scopes named

| tool | correct for |
|---|---|
| `placeAll_columns` | a part that owns everything it touches |
| `placeAll_targets` | a part whose `c` is always a declared column — `TranscriptRecipe` is one, which is why the argument worked there |
| `AllocationPlaced` | the general case |

Neither of the first two is correct in general. That took two attempts to
establish, and both attempts were made by applying the tool rather than by
inspecting it.


---

## CANONICAL-PROGRAM-SELECTION-ALLOCATION

```text
claim:
  SelectedRecipe declared a column count and not the columns. It now declares
  both, and the placement obligation is statable for every part.
status: PROVED 2026-07-27 (cycle 372). SelectedRecipe.allocation,
  allocations_match_widths.
```

Cycle 371 flagged this and cycle 372 checked it. `SelectedRecipe` carried
`cost.auxiliaryColumns` — a **number** — so a selection declared *how many*
columns it allocates and never *which*. The layout sized the selection's window
from that number while nothing tied the number to any column, and
`ColumnWindows.AllocationPlaced` could not even be **stated** for a selection.

**A count without the columns it counts is the same defect as a row count
without the rows** — the one this project rejects everywhere else. It sat inside
a structure introduced this session and went unexamined for thirty cycles.

The structure now carries `allocation : List Nat` with
`allocationCertified : allocation.length = cost.auxiliaryColumns`.

### Placement is now statable across all eight parts

`allocations` lists the eight declared allocations in `widths` order, and
`allocations_match_widths` proves each has the width the layout assigns it. So
`widths` is a list of declared allocation sizes rather than a list of unrelated
numbers.

The obligation then discharges differently per part, uniformly stated:

| part | how |
|---|---|
| four reading recipes | `allocationPlaced_nil` — empty allocation |
| `KLowNormBatch` via Π_DEC | `allocationPlaced_of_bounded` from `canonicalDigits_column_le` |
| four selections | the deployment supplies the bound |

### Why this was mine to change

`SelectedRecipe` is a structure this session introduced, not a mapped protocol
definition. Adding a field is additive to work in progress. `CoordinatePlan` is
the opposite case — what a deployment must supply — which is why
`POSEIDON2-HASH-PLAN-STRENGTHENING` records that change rather than making it.


---

## CANONICAL-PROGRAM-TRANSCRIPT-ALLOCATION

```text
claim:
  The transcript now declares its columns. The assembly's placeholder for them
  was the same defect it had just closed for selections.
status: PROVED 2026-07-27 (cycle 373). TranscriptRecipe.transcriptColumns.
```

Cycle 372 closed the count-without-columns defect for `SelectedRecipe` — and, in
the `allocations` list one line below the fix, wrote
`List.replicate (transcriptCost rounds).auxiliaryColumns 0` for the transcript.

**A list of the right length whose contents are fabricated**, every entry `0`,
which is the constant wire and not a valid allocation at all. It satisfied
`allocations_match_widths`, which compares lengths, while carrying no
information. The defect was reintroduced in the same cycle that closed it,
because `allocations` needed an entry and the transcript had no list to give.

### The fix

`TranscriptRecipe.transcriptColumns rounds` is the real list: each round's 344
S-box columns at that round's stride.

| theorem | content |
|---|---|
| `transcriptColumns_length_eq` | `rounds · 344`, matching the cost |
| `transcriptColumns_nonzero` | no entry is the constant wire |
| `transcriptColumns_in_window` | every entry lies in its round's window |

The second is what the placeholder would have failed: `List.replicate n 0` is
all constant wire.

### What this says about closing a defect

Closing it in one structure did not close it in the assembly that consumes the
structure. The `allocations` list had eight entries and only the four selections
were fixed; two were genuinely empty, one was Π_DEC's real list, and one was
invented. **A fix applies where it is written, and the consumers have to be
checked separately.**


---

## CANONICAL-PROGRAM-WINDOW-SIZING

```text
claim:
  A window sized by column count cannot hold a sparse allocation. The
  transcript's does not fit, and both repairs change a published number.
status: RESOLVED 2026-07-27 (cycle 375). The obstruction was real; the
  framing of the repair was not. ColumnWindows.allocationPlaced_spanOf.
```

### Resolved, and the framing was the error

Cycle 374 recorded two repairs and declined both, saying each "changes a
published number". **That was wrong.**

`Typed.Cost.auxiliaryColumns` is a count and stays one. `Window.width` is an
abstraction introduced in this session, and nothing requires it to equal that
count. What was needed is a width the allocation *determines*.

`spanOf` is that width: the largest allocated column. `allocationPlaced_spanOf`
then holds **for any allocation**, sparse or not, with one obligation — no
allocated column is the constant wire. Nothing has to be contiguous, and no
count changes meaning.

So the decision recorded as needing an owner was an engineering call inside
work introduced this session, and escalating it was over-escalation rather than
over-claiming. The obstruction below is still accurate about the *count-sized*
window; it is no longer a blocked route.

### The count against the span

A `Nodup` allocation of `n` nonzero columns has span at least `n`, by
pigeonhole. **That is not proved here** — it needs
`List.Nodup.length_le_of_subset` or an equivalent, which this tree lacks, the
same gap that keeps `EuclidPrime goldilocksP` a hypothesis. Placement does not
need it.

### The original obstruction, for the record

Discharging `ColumnWindows.AllocationPlaced` part by part reaches the transcript
and stops.

`ColumnWindows.Window` owns a **contiguous** range `(base, base + width]`, and
`widths` sizes each window by `cost.auxiliaryColumns` — a **count**. For a part
whose allocated columns are contiguous the two agree. The transcript's are not:
round `r` allocates 344 columns starting at `r · 369 + 17`, so at one round the
allocation is `17 … 360` — **344 columns spanning 361**. A window of width 344
ends at column 344; column 360 is allocated and outside it.

### A design fault, not a proof gap

`ColumnWindows` assumes an allocation is a contiguous block whose size is its
count. The transcript's is sparse by construction — the sponge stride leaves
gaps for chunk columns — so **no bound on its count places it**.

### The two repairs, and why neither is taken here

| repair | what it costs |
|---|---|
| size windows by **span** (**taken**) | `auxiliaryColumns` is a count, so the cost tuple relates to `counts` and the layout to `widths`; the two stop being one theorem |
| require allocations **contiguous** | the transcript's cannot be, without changing the sponge layout it inherits from `Poseidon2Sponge.canonicalSpongeLayout` |

Both change what a published number means. The first was taken: `widths` sizes by
`ColumnWindows.spanOf`, the cost tuple keeps counting columns, and
`transcript_placed` is the sparse case a count-sized window could not hold. It is
the third distinct defect found in this layout machinery — after
`placeAll_columns`' unsatisfiable read bound and `placeAll_targets`' false
premise about `c` — and the first that is about the window type itself rather
than about a theorem over it.

### What is not affected

The row side. `rows_length`, `rows_iff_every_part` and the eight restriction
lemmas do not use windows. The obstruction is confined to the column accounting,
which `CANONICAL-PROGRAM-LAYOUT-UNAPPLIED` recorded as describing a plan rather
than the emitted program. **Closed in cycle 377**: `placedRows` is the emitted
program the layout describes.


---

## CANONICAL-PROGRAM-PLACEMENT-DISCHARGED

```text
claim:
  Counts and spans are separate quantities, windows are sized by span, and
  placement is discharged for every part that can discharge it.
status: PROVED 2026-07-27 (cycle 376).
```

`Typed.Cost.auxiliaryColumns` counts columns. A window needs to know **where**
they sit, which is `spanOf`. The two coincide only for a `1 … n` allocation and
the transcript's is not one, so both are stated and neither does the other's job:

| definition | quantity |
|---|---|
| `counts` | per-part column counts — what `N_canonical` sums |
| `widths` | per-part window widths — each allocation's own span |

`auxiliaryColumns_eq_counts_sum` is the corrected form of what cycle 342 called
`auxiliaryColumns_eq_widths_sum`. It relates the cost to the **counts**, which is
what a count component means.

### Placement, per part

| part | theorem | obligation |
|---|---|---|
| fold digest | `foldDigest_placed` | none — empty allocation |
| mixer | `mixer_placed` | none — empty allocation |
| transcript | `transcript_placed` | none — `transcriptColumns_nonzero` already proved |
| Π_DEC | `piDec_placed` | the claim's digits allocate real columns |
| four selections | `selection_placed` | the deployment's columns are real |

`transcript_placed` is the case a count-sized window could not hold. With
span-sizing it needs nothing beyond what was already proved.

### What remains between here and a placed program

`rows` still concatenates unrelocated parts —
`CANONICAL-PROGRAM-LAYOUT-UNAPPLIED` stood at the time of writing. What had
changed is that the obstruction was no longer in the layout: every part's
allocation became placeable, so connecting `rows` to `windows` was a matter of
emitting `ColumnWindows.placeAll` rather than of fixing the window type.
**Cycle 377 emitted it and closed that entry.**


---

## CANONICAL-PROGRAM-PLACED-SEMANTICS

```text
claim:
  The placed program has soundness, not only a row count.
status: PROVED 2026-07-27 (cycle 379). placeAll_satisfies_head,
  placeAll_satisfies_tail, placed_satisfies_piDec, placed_satisfies_foldDigest.
```

Cycle 377 gave `placedRows` a row count and left it without semantics: all eight
restriction lemmas were stated on `rows`, so a deployment satisfying the placed
program could conclude nothing about its parts.

**That is the shape cycle 353 found in the selection interface** — capabilities
enumerated on one object while another is left without them — recurring on a new
object introduced two cycles earlier.

### Each part under its own pullback

`placeAll` relocates each part by its own base, so satisfaction does not
transport by a single pullback. `placeAll_satisfies_head` supplies the step and
`placeAll_satisfies_tail` peels the placement, because the tail of a `placeAll`
is itself a `placeAll` at the shifted base.

`placed_satisfies_piDec` and `placed_satisfies_foldDigest` were the first two
instances, and cycle 379 stopped there — "the remaining six follow by further
peeling". **Cycle 385 wrote the six.** See
`CANONICAL-PROGRAM-PLACED-SEMANTICS-PARTIAL` for why stopping was a defect and
not a scope choice.

### How this was found

By enumerating rather than asserting, which is what cycle 378 recorded as the
method after stale statuses were found the same way. The claim under test was
that nothing remained on this side; listing what `placedRows` supports against
what `rows` supports answered it.


---

## CANONICAL-PROGRAM-PLACED-SEMANTICS-INDEXED

```text
claim:
  Every part of a placed program is recovered at its own base, by one theorem
  rather than eight instances.
status: PROVED 2026-07-27 (cycle 380). placeAll_satisfies_index.
```

Cycle 379 left six instances outstanding and called them mechanical. They were —
which is why writing them out would have been the mechanical thing and
generalising was not.

`placeAll_satisfies_index` recovers the part at any index, under the pullback at
`baseAt base parts index`. The statement needs `baseAt` to say *where* a part
lands, which no single instance needs; that is the content the six copies would
have left implicit.

### On "mechanical"

Three earlier cycles used that word about remaining work and were wrong each
time — 358's bookkeeping, 359's two-window analysis, 369's canonical numbering.
Here it was accurate about the instances and still the wrong thing to act on:
the accurate reading was that repetitive work signals a missing generalisation.

`baseAt` had been defined in cycle 379 and used by nothing. It exists because
this theorem needs it.


---

## CANONICAL-PROGRAM-PLACED-BOTH-DIRECTIONS

```text
claim:
  The placed program can be composed as well as decomposed.
status: PROVED 2026-07-27 (cycle 381). placeAll_honest, placedRows_honest.
```

Cycle 379 gave `placedRows` part recovery and cycle 380 generalised it. Neither
gave the converse: a placed program could be taken apart and never put together,
so a deployment could not conclude its placed program was satisfiable at all.

**That is cycle 354's directional gap, recurring.** There it was `rows`; here
`placedRows` was a new object and only the direction that had been needed was
proved. Both objects now have both directions.

### The recurrence is the finding

Cycle 354 recorded the lesson as "a theorem proved one way reads as complete
when that is the way you needed it". Twenty-seven cycles later the same shape
appeared on a new object, and it was found by the same method — enumerating what
`placedRows` supports against what `rows` supports, rather than asserting the
queue was empty.

A lesson recorded is not a lesson applied. What caught this was the enumeration
habit, not the memory of the earlier instance.


---

## CANONICAL-PROGRAM-ALLOCATIONS-AT-THEIR-BASES

```text
claim:
  Allocation placement is stated at the bases the placement actually uses.
status: PROVED 2026-07-27 (cycle 382).
```

The `*_placed` theorems of cycle 376 are universally quantified over `base`, so
they apply at any base — including the ones `placeAll` uses. **Nothing said so.**

That is cycle 369's disconnect one level in: two families of theorems about the
same layout, correct separately and never related. A general theorem without its
instantiation leaves the record with no statement that *this* allocation lands in
*this* program's window.

`foldDigest_placed_at_its_base`, `mixer_placed_at_its_base`,
`transcript_placed_at_its_base`, `piDec_placed_at_its_base` and
`selection_placed_at_its_base` instantiate at
`ColumnWindows.baseAt 0 (parts recipes) index`, which is where `placeAll` puts
part `index`.

**No new mathematics — the content is the connection.** That is the same kind of
result as cycle 372's `allocation` field: the missing piece was not a proof but
a statement relating two things that already existed.

### Why a ∀-quantified theorem is not enough

A reader checking whether the assembly's column accounting describes its emitted
program needs a theorem naming both. "It holds for all bases, and one of those is
the right one" requires the reader to supply the instantiation and to know which
base is right. Neither is recorded elsewhere.


---

## CANONICAL-PROGRAM-CROSS-REFERENCE-DRIFT

```text
claim:
  Three entries asserted CANONICAL-PROGRAM-LAYOUT-UNAPPLIED was open after
  cycle 377 closed it.
status: CORRECTED 2026-07-27 (cycle 383).
```

Cycle 378 found two stale **statuses** and corrected them. It did not check
**cross-references**, and three were wrong in the same way:

| entry | said |
|---|---|
| `CANONICAL-PROGRAM-PLACEMENT-TOOL` | "what `LAYOUT-UNAPPLIED` needs before `rows` can be connected" |
| `CANONICAL-PROGRAM-WINDOW-SIZING` | "which `LAYOUT-UNAPPLIED` already records as describing a plan" |
| `CANONICAL-PROGRAM-PLACEMENT-DISCHARGED` | "`LAYOUT-UNAPPLIED` stands" |

Each was true when written and false after cycle 377. A reader scanning this
file — or an iteration starting cold — would find three current-tense statements
that the layout is unapplied, and one closure notice, and could reasonably
believe the first three.

### The shape of this defect

A status line is checked when the entry it heads is revisited. **A cross-
reference is checked when the entry it points at changes**, which is a different
event and nothing was watching for it. Cycle 378's fix — enumerate statuses —
does not catch this, because the statuses were right.

The check that does catch it: when closing an entry, grep for its name.


---

## CANONICAL-PROGRAM-RETRACTION-DRIFT

```text
claim:
  A retraction reached the entry that carried it and not the entries that
  repeated its conclusion.
status: CORRECTED 2026-07-27 (cycle 384).
```

Cycle 383 established the practice: when closing an entry, grep for its name.
Applying it retroactively to every entry closed or retracted this session found
three stale references, **two of them restating a claim proved false**:

| location | said |
|---|---|
| `poseidon2-canonical-open-items.md` | "Together with `POSEIDON2-HASH-NO-DOMAIN-TAG`, the answer is exact: what separates a prior digest from a next digest is entirely the preimage content" |
| `Poseidon2HashRecipe.lean` docstring | "This is why `POSEIDON2-HASH-NO-DOMAIN-TAG` is a sharper statement than it first appears" |
| `CanonicalProgram.lean` docstring | "`CANONICAL-PROGRAM-COLUMN-DISJOINTNESS` names that, and it is open" |

### The first two are worse than stale

Cycle 345 retracted `POSEIDON2-HASH-NO-DOMAIN-TAG` — a separator **does** exist,
`normalizedIteration`'s `+1`. That cycle's own text said the sharpened form
"inherits the same error and is retracted with it", and then retracted it *in
the other entry's prose* while leaving the sharpened form standing where it was
written.

**A retraction announced in one place is not a retraction.** Thirty-nine cycles
later, two locations still asserted the retracted conclusion, one of them a
module docstring a reader meets before any theorem.

### What the practice has to be

Grep on close **and on retract**, and grep for the *conclusion* as well as the
entry name — the docstring above never named the retracted entry's claim, it
restated it. Searching for `POSEIDON2-HASH-NO-DOMAIN-TAG` found it only because
the sentence happened to cite the entry.


---

## CANONICAL-PROGRAM-PLACED-SEMANTICS-PARTIAL

```text
claim:
  Satisfying the placed program recovers all eight parts, not the first two.
status: PROVED 2026-07-27 (cycle 385). placed_satisfies_at, six named
  instances, placed_satisfies_every_part.
```

`satisfies_every_part` states eight conjuncts against `rows`.  Its placed twin
`placed_has_semantics` stated **two**, with the reason given in its own
docstring: "the point is that the placed program has semantics at all rather than
that eight instances are spelled out."

### Why that was a defect

A recipe recovered from `rows` and not from `placedRows` has no soundness under
the layout the assembly publishes.  The six omitted parts — mixer, transcript and
all four selections — were six recipes whose placed semantics nothing asserted.
The two stated were the two whose bases are `0` and `0 + span`, the two easiest
to write.

**The stated reason had also expired.**  "Further applications of
`placeAll_satisfies_tail`" described the route before
`placeAll_satisfies_index` existed — a lemma added in the *same* cycle, one file
away.  With the indexed form each part is one application at its own `baseAt`,
and each of the six index lookups discharges by `rfl`.

### The shape

Twentieth instance this session of a claim of mine being weaker than reported,
and the recurrence of a named failure mode: **components skipped for being
uninteresting.**  What makes this one worth its own entry is that the skip was
*argued for* in the docstring rather than silent, and the argument was checkable
and wrong.

| stated | actual |
|---|---|
| six instances are "the same pattern" | six recipes with no placed soundness |
| route is `placeAll_satisfies_tail` peeling | `placeAll_satisfies_index` already existed |


---

## CANONICAL-PROGRAM-COLLISION-NOT-STATED

```text
claim:
  No column is allocated by two parts of the assembly.
status: PROVED 2026-07-27 (cycle 389).
  ColumnWindows.placed_allocations_disjoint,
  CanonicalProgram.Recipes.allocations_disjoint.
```

Section 2 item 4 asks for column ownership **with no collision**. What existed
was two halves that were each mistaken for it:

| proved | says |
|---|---|
| `windows_no_collision` | two *windows* sharing a column are the same window |
| the `*_placed_at_its_base` family | each part's allocation lands in its own window |

Neither is the item. The item is that **no column is allocated by two recipes**,
and it is the composition of the two — which nothing performed.

### Why the composition is not bookkeeping

The conclusion is about **relocated** columns. Two parts may perfectly well
allocate the same raw column before relocation; that is what relocation exists to
fix. A disjointness statement about raw allocations would be false, and one about
windows says nothing about allocations. Getting the quantifiers to line up is the
content.

### How it was found

Same method as cycles 386 and 388: enumerate against the stated list. This is the
third consecutive cycle in which the gap was a checklist item that had *related*
theorems near it — padding beside rate and capacity, conservation beside five
conserving atoms, collision beside window-disjointness. **Neighbouring evidence
is what makes a missing item hard to see.**

### A smaller unevenness, fixed in passing

`FoldDigestRecipe` had `digestColumns_length`; `CommitmentMixerRecipe` had no
`mixerColumns_length`. The two recipes are the assembly's two non-allocating
parts and should carry the same lemmas. Added, and used by
`foldDigest_and_mixer_allocate_nothing` — which derives from the two recipes'
own lemmas rather than reading their definitions.


---

## CANONICAL-PROGRAM-ALLOCATION-DECLARED-NOT-USED

```text
claim:
  auxiliaryColumns counts columns the emitted program uses, not columns a
  declaration names.
status: PROVED 2026-07-27 (cycle 390) for Pi_DEC.
  KLowNorm.lowNormRows_use_squareColumn, KLowNormBatch.batchRows_use_columns,
  PiDecRecipe.rows_use_columns, PiDecRecipe.columns_exact,
  CanonicalProgram.Recipes.builtRows_use_piDec_columns.
  For the transcript, closed in cycle 392 - see
  CANONICAL-PROGRAM-TRANSCRIPT-ALLOCATION-UNUSED.
```

Conservation, added in cycle 388, bounds the emitted rows' columns **from
above**: nothing outside the claim is reached. That is one inclusion. Nothing
stated the other, and the other is what makes a count a count of the *program*:

> `auxiliaryColumns_eq_counts_sum` relates the cost tuple to the **declared**
> allocations.

A recipe could declare more digits than its rows constrain, and the cost would
overcount with conservation raising no objection — an unused declared column is
simply not a mention, so an upper bound on mentions cannot see it. That is the
trap list's first entry, "a count that is declared ... rather than derived from
an emitted row program", surviving inside a theorem that looked like it ruled it
out.

### Pi_DEC is now exact

`columns_exact` states both inclusions together, because one of them is not
exactness and section 2 item 4 asks for *exact* column ownership.

### The transcript is not

The transcript declares 344 S-box columns per round at stride 369, and no
theorem says each is mentioned by an emitted row. See
`CANONICAL-PROGRAM-TRANSCRIPT-ALLOCATION-UNUSED`.

---

## CANONICAL-PROGRAM-TRANSCRIPT-ALLOCATION-UNUSED

```text
claim (cycle 390, RESTATED):
  Every column TranscriptRecipe.transcriptColumns declares is mentioned by some
  row of transcriptRows.
status: CLOSED 2026-07-27 (cycle 392). transcriptColumns_written. The claim as
  written was FALSE for an arbitrary layout - transcriptColumns_not_layout_generic
  is the witness - so it was restated in cycle 391 about canonicalTranscriptRows
  and proved in full in cycle 392.
```

### The claim named the wrong program

`transcriptColumns` takes **only `rounds`**. It never mentions `layouts`, while
`transcriptRows` does. So a declared column and an emitted column are indexed by
different things, and no usage theorem can hold for an arbitrary layout.

`transcriptColumns_not_layout_generic` exhibits it: a layout shifted to 1000000
names an S-box column the declaration does not contain. The previous entry
called this an upper bound on the transcript summand. **That was too generous** —
at an arbitrary layout the declared list and the emitted program are not related
in either direction.

### What is true, and now proved

`sboxCount = 8 · 8 + 22 = 86` and `columnsPerSbox = 4`, so a round owns
`86 · 4 = 344` S-box columns from `auxBase = base + 17`. That is exactly the
declared range — which is why the *count* matched for fifteen cycles while the
connection was missing.

`transcriptColumns_eq_canonical_sbox` proves both directions: a column is
declared iff it is `sboxColumn (canonicalLayouts round) index slot` for some
`round < rounds`.

### Closed, and the route estimate was wrong

Cycle 391 called what remained "a fact about `Poseidon2Program`'s row shape
alone". It was not. Each S-box does write its four columns in the `c` of its four
rows, but the transcript emits **normalized** rows, and `mentions_normalizeRow`
runs one way only — normalization drops a column whose coefficient vanishes
modulo the prime, so support equality is false and an arbitrary write does not
lift.

The lift needs a second fact: every S-box write has coefficient `1`, and a
coefficient-one write survives (`fieldNormalize_singleton_one`,
`mentions_normalizeRow_singleton`). `transcriptColumns_written` chains the three.

`N_canonical.auxiliaryColumns` is now exact on both summands it can be exact on:
Π_DEC's and the transcript's. The selections' remain the deployment's.
