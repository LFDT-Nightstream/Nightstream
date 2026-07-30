# RECIPE-ROW-OWNERSHIP — §2 item 3, retracted as claimed and stated as open

Evidence: `planned`

---

## RECIPE-ROW-OWNERSHIP-RETRACTION

```text
claim:
  The recipes built in this line do NOT all meet all ten section-2 items.
  Item 3, exact row ownership, is absent from every one of them.
status: RETRACTED 2026-07-27 (cycle 356) by grepping for ownership theorems.
```

Section 2's ten items include:

> **Exact row ownership: every emitted row belongs to exactly one receipt.**

Six recipes were built in this line — `KRecomposition`, `KLowNormBatch`,
`PiDecRecipe`, `FoldDigestRecipe`, `CommitmentMixerRecipe`, `TranscriptRecipe`.
**None of them has a row-ownership theorem.** Searching each for one returns
nothing; the incidental matches are the word "owner" in prose.

Row *counts* were derived, and column ownership and conservation were proved.
Item 3 is a distinct obligation and it was never discharged.

### The claim that was wrong

"Every recipe this session has produced now meets all ten §2 items" — reported
from cycle 339 onward and repeated in every summary since. It was wrong for all
six, on the same item, for eighteen cycles.

This is the largest-scope error of the session. It was not detectable from any
summary because the summary is what carried it; finding it took listing the ten
items and checking each against the modules, which had never been done.

---

## RECIPE-ROW-OWNERSHIP-SHAPE

```text
claim:
  What item 3 needs, and which half of it is already available.
status: SETTLED 2026-07-27 (cycle 363), superseded by
  RECIPE-ROW-OWNERSHIP-STATUS. Left in place because the analysis below is
  what the per-recipe work was aimed at, and because it was written before
  the answer was known.
```

**Stale status corrected in cycle 378.** This entry sat at `OPEN` for
twenty-two cycles after the question it poses was answered — three recipes
proved, two proved impossible with witnesses, one proved. A spec status is what
a cold-start iteration reads, and this one would have sent it to re-open a
settled question.

`KEquality.rows_eq_map_owners` is the model the project already has: `rows` is
defined as an explicit list and the theorem reconciles it with a `map` over an
enumerated `RowOwner` type. That reconciliation is the content.

For the six recipes here the shape differs, and so does the work:

| recipe | emission | attribution | uniqueness |
|---|---|---|---|
| `FoldDigestRecipe`, `CommitmentMixerRecipe` | `receipts.map f` | definitional | needs receipts distinguishable |
| `KLowNormBatch`, `KRecomposition`, `TranscriptRecipe` | `receipts.flatMap f` | `List.mem_flatMap` | needs per-receipt row sets disjoint |
| `PiDecRecipe` | five-part concatenation | `List.mem_append` | needs the five parts disjoint |

**Existence of attribution is available for all six** — every emitted row comes
from some receipt, by the membership lemma for the emission shape.

**Uniqueness is not.** Two receipts emitting identical rows would defeat it, and
nothing currently rules that out. For the allocating recipes it should follow
from the allocation being `Nodup` — distinct square columns make distinct rows —
but that argument is not written, and for the allocation-free recipes it depends
on the caller's combinations being distinct, which is a caller property.

### Why a `map`-shaped theorem would not close it

For a program *defined* as `receipts.map f`, stating "rows = receipts.map f" is
`rfl` — a definitional theorem read as a semantic fact, which is the trap
section 3 names. The content is in the uniqueness direction, and that is exactly
the half that is missing.

### Not claimed

That the recipes are wrong, or that any row is misattributed. Only that item 3
was reported as met and is not, and that closing it needs the uniqueness
argument rather than a restatement of the definition.


---

## RECIPE-ROW-OWNERSHIP-LOWNORM

```text
claim:
  KLowNormBatch meets item 3. Every emitted row belongs to exactly one digit.
status: PROVED 2026-07-27 (cycle 357). batchRows_owned.
evidence: model-proved
```

The first of the six, and the one where the uniqueness argument is real rather
than definitional.

| theorem | content |
|---|---|
| `lowNormRows_determines_column` | a row shared between two digits forces their columns equal |
| `column_determines_digit` | a `Nodup` allocation recovers the digit from its column |
| `batchRows_owner_unique` | the two composed |
| `batchRows_owned` | existence and uniqueness together |

### Where the allocation earns its second keep

`WellFormed.distinct` was introduced for honest completeness — a shared column
would let the outer write clobber the inner square. It turns out to carry
ownership as well: **a row exposes its digit's allocated column in a field
position**, so the column is recoverable from the row, and `distinct` recovers
the digit.

Both emitted rows expose it, in different positions — `⟨value, value, [(col,1)]⟩`
in `c`, and `⟨[(col,1)], value, value⟩` in `a` — and the cross case, where one
digit's first row equals another's second, forces the columns equal through the
value. All four cases close.

### What it does not need

The values. Two digits with identical combinations remain distinguishable
because the *allocation* distinguishes them. That is why the argument works for
a batch whose values a caller supplies.

### The remaining five

`KRecomposition`, `PiDecRecipe`, `FoldDigestRecipe`, `CommitmentMixerRecipe`,
`TranscriptRecipe`. The allocation-free ones have no column to recover a receipt
from, so their uniqueness is a property of the caller's combinations being
distinct — a stated caller obligation rather than a theorem, and it must be
written as one rather than assumed. `TranscriptRecipe`'s rounds are separated by
`canonicalLayouts_disjoint`, which is the same shape as this argument and should
carry the same conclusion.


---

## RECIPE-ROW-OWNERSHIP-TRANSCRIPT

```text
claim:
  Round separation on every column family, which is the ingredient row
  ownership for the transcript consumes.
status: CLOSED 2026-07-27 (cycle 363) by
  TranscriptRecipe.transcriptRows_owner_unique. Status corrected in cycle 378.
evidence: model-proved
```

**Stale status corrected in cycle 378.** This entry read `PARTIAL — the
row-level statement is not yet assembled` for fifteen cycles after
`transcriptRows_owner_unique` assembled it. Worse than merely stale: cycle 359
recorded the row-level step as needing a two-window argument, cycle 363 proved
that framing wrong and closed it by tracking the target column alone, and this
entry still describes the superseded plan as the remaining work.

`canonicalLayouts_disjoint` covered **S-box columns only** — enough for the
sponge's cost fold, not enough for ownership, because a row can mention an input
or an output port too.

`shiftedLayout` is explicit: `inputPort lane = base + 1 + lane`,
`outputPort lane = base + 9 + lane`, `sboxColumn index slot = base + 17 +
4·index + slot`. So every column a round names lies in `(base, base + 361]`, and
a stride of 369 separates rounds on every family at once.

| theorem | content |
|---|---|
| `canonicalLayouts_column_window` | every column of a round lies in that round's window, across all three families |
| `canonicalLayouts_windows_disjoint` | distinct rounds' windows share nothing |
| `canonicalLayouts_no_shared_column` | two rounds cannot name the same output port |

Stated over the whole space rather than one family, so adding a row shape later
does not reopen it.

### What is still not assembled

The row-level statement — *every emitted row belongs to exactly one round*.

### Corrected 2026-07-27 (cycle 359): it is not bookkeeping

Cycle 358 called the remaining step "bookkeeping rather than a new argument".
That was wrong, and the reason is in this recipe's own design.

`entryAt layouts absorbedAt (previous + 1)` reads
`(layouts previous).outputPort lane` on unwritten lanes — **the previous
round's window**. So a row of round `r` legitimately mentions columns of window
`r - 1` as well as window `r`, and the natural argument, *all of a row's columns
lie in one window, so two windows force the rounds equal*, is false here.

The chaining that makes the duplex a duplex is exactly what defeats
single-window containment. That is not a defect: `chain_value` is the theorem
that a round reads the previous round's output, and it is the point of the
construction.

### What the argument actually needs

Two facts, and only the first is one line:

1. **Every row touches its own window.** The four S-box rows target
   `sboxColumn (layouts r) index slot`; the binding row targets
   `(layouts r).outputPort lane`. Both are in window `r`, and
   `mentions_normalizeRow` carries this through normalization.
2. **Every row's columns lie in window `r - 1` ∪ window `r`.** This needs the
   entry classification per round, which `transcriptRows_conservation` supplies
   only with the round existentially bound.

With both, a shared row would put a window-`r` column inside
window `r' - 1` ∪ window `r'`, and the disjointness forces `r ∈ {r' - 1, r'}`
and symmetrically `r' ∈ {r - 1, r}`, which closes only after ruling out the
crossed case. That is a real argument, not a restatement.

**Still not claimed:** that `TranscriptRecipe` meets item 3.

**Not claimed:** that `TranscriptRecipe` meets item 3. It does not yet. This
entry records the ingredient, and cycle 356's retraction stands for this recipe
until the row-level theorem exists.


---

## RECIPE-ROW-OWNERSHIP-MAP-SHAPED

```text
claim:
  FoldDigestRecipe and CommitmentMixerRecipe meet item 3.
status: PROVED 2026-07-27 (cycle 360). digestRows_owned, mixerRows_owned.
evidence: model-proved
```

Both emit `receipts.map f`, so existence is `List.mem_map`. Uniqueness differs:

**`FoldDigestRecipe` needs no hypothesis.** `equalityRow left right` is
`⟨left, [(0,1)], right⟩`, so two lane pairs emit the same row only if they *are*
the same pair — the row is the pair. `equalityRow_injective` gives it outright.

**`CommitmentMixerRecipe` needs distinct parents.** A coordinate's row carries
its parent in the `c` field, so `(coordinates.map Coordinate.parent).Nodup`
recovers the coordinate. A decoder meets it by construction: each coordinate is
a distinct commitment position with its own parent column.

### A premise that consumed nothing

The first draft gave **both** a `Nodup` hypothesis. The linter reported both
unused, and it was right for different reasons:

- for the fold digest the hypothesis was genuinely unnecessary, and dropping it
  made the theorem stronger;
- for the mixer the *conclusion* had been weakened to `other.parent =
  coordinate.parent`, which follows from the row alone — so the hypothesis was
  unused because the statement had been trimmed to avoid needing it.

Section 3 names "an obligation moved to a new premise that no real consumer
constructs". This was that in reverse: a premise nothing consumes, in one case
hiding a weakened conclusion. Both are now correct — one hypothesis-free, one
using its hypothesis to conclude `other = coordinate`.

### The remaining three

`KLowNormBatch` was proved in cycle 357. `TranscriptRecipe` needs the two-window
argument. `KRecomposition` and `PiDecRecipe` are `flatMap` and concatenation
respectively and are not yet attempted.


---

## RECIPE-ROW-OWNERSHIP-RECOMPOSITION

```text
claim:
  KRecomposition cannot meet item 3 unconditionally. A row belonging to two
  distinct checks exists, and is exhibited.
status: OBSTRUCTION, kernel-checked 2026-07-27 (cycle 361).
  recompositionRows_owner_not_unique.
```

A check emits **two** rows carrying **different** coordinates:
`⟨recomposed.low, [(0,1)], parent.low⟩` and
`⟨recomposed.high, [(0,1)], parent.high⟩`. Nothing stops one check's low row
from coinciding with another check's high row, because the two rows expose
unrelated halves of unrelated carriers.

`witnessCheckA` and `witnessCheckB` are two distinct checks sharing exactly one
row, and `recompositionRows_owner_not_unique` exhibits it. So item 3 for this
recipe is a caller obligation, and the obligation is **not vacuous** — it rules
out configurations that would otherwise typecheck.

### Why uniqueness came cheaply elsewhere

| recipe | why |
|---|---|
| `KLowNormBatch` | each receipt **allocates** a column, and the row exposes it |
| `FoldDigestRecipe` | each receipt emits **one** row, and the row *is* the receipt |
| `CommitmentMixerRecipe` | each receipt emits **one** row, carrying the whole parent |
| `KRecomposition` | each receipt emits **two** rows carrying different halves — none of the above applies |

That is the structural rule this session's ownership work produced: **a receipt
is recoverable from its row when it either allocates, or emits a single row that
determines it.** A receipt emitting several rows of differing content is
recoverable from none of them individually.

### Not claimed

That the recipe is unsound, or that any deployment hits the collision. Only that
attributing a row to a receipt is impossible from the row alone here, so item 3
needs a stated hypothesis rather than a proof — and that a proof was attempted
and found to be unavailable, rather than skipped.


---

## RECIPE-ROW-OWNERSHIP-PIDEC

```text
claim:
  PiDecRecipe inherits the recomposition's ownership obstruction.
status: OBSTRUCTION, kernel-checked 2026-07-27 (cycle 362).
  rows_owner_not_unique.
```

The assembly **contains** `recompositionsRows`, so the two colliding checks of
`RECIPE-ROW-OWNERSHIP-RECOMPOSITION` placed in `xEntries` put the same row in
`rows`, attributable to either. Nothing in the concatenation distinguishes them.

**An assembly cannot repair an attribution its parts do not support.** The
obstruction is inherited, not newly introduced, and item 3 for Π_DEC carries the
same caller obligation the recomposition does — discharged wherever the checks
are supplied, not here.

---

## RECIPE-ROW-OWNERSHIP-STATUS

```text
claim:
  Item 3, per recipe, as of cycle 363.
status: ALL SIX SETTLED.
```

| recipe | item 3 | why |
|---|---|---|
| `KLowNormBatch` | **proved** | receipt allocates a column; the row exposes it |
| `FoldDigestRecipe` | **proved** | one row per receipt, and the row *is* the receipt |
| `CommitmentMixerRecipe` | **proved** | one row per receipt, carrying the whole parent |
| `KRecomposition` | **impossible unconditionally** | two rows per receipt, differing halves; witness exhibited |
| `PiDecRecipe` | **impossible unconditionally** | inherits the above |
| `TranscriptRecipe` | **proved** | each round allocates a window, and every row targets it |

### The rule the work produced

A receipt is recoverable from its row when it **either allocates a column the
row exposes, or emits a single row that determines it**. A receipt emitting
several rows of differing content is recoverable from none of them individually,
and no assembly containing such a receipt can recover it either.

That is a statement about row programs generally, not about these six, and it is
what the five settled cases have in common.

### What cycle 356's retraction now amounts to

"All recipes meet all ten items" was wrong. The accurate replacement is: **item
3 holds for three, is provably unavailable without a caller obligation for two,
and is open for one** — and the two unavailable cases are a property of their
emission shape rather than a gap in the proofs.


---

## RECIPE-ROW-OWNERSHIP-TRANSCRIPT-CLOSED

```text
claim:
  TranscriptRecipe meets item 3, and the two-window difficulty was an artifact.
status: PROVED 2026-07-27 (cycle 363). transcriptRows_owner_unique.
```

Cycle 359 recorded that a row of round `r` mentions window `r - 1` too — the
entry reads the previous round's output ports — and concluded the argument
needed a two-window analysis with a crossed case to rule out.

**That was over-complicated, and the correction is the useful part: only the
target column needs tracking.**

Every emitted row's `c` field names a column in its *own* window — the four
S-box rows target `sboxColumn (layouts r) index slot`, the binding row targets
`(layouts r).outputPort lane`. The entry appears only in `a` and `b`. So a
shared row puts one column in two windows, and
`canonicalLayouts_windows_disjoint` closes it. No two-window analysis, no
crossed case.

`rawRow_target_in_window` does the five-shape case analysis;
`row_target_in_window` carries it through normalization via
`mentions_normalizeRow`; `transcriptRows_owner_unique` concludes.

### The rule predicted this

`RECIPE-ROW-OWNERSHIP-STATUS` says a receipt is recoverable when it allocates a
column the row exposes. A round allocates its **whole window**, and every one of
its 352 rows exposes that window in its target — so the many-rows-per-receipt
shape that defeats `KRecomposition` does not defeat this one. There, two rows
expose unrelated halves; here, all rows expose the same round.

That the rule predicted the outcome before the proof was attempted is the first
time this session a general statement earned its keep by forecasting rather than
summarising.


---

## PIDEC-CONSERVATION-ABSENT

```text
claim:
  Every recipe in the assembly satisfies section 2 item 5 (conservation).
status: PROVED 2026-07-27 (cycle 388) for Pi_DEC and the built assembly.
  PiDecRecipe.Touches, PiDecRecipe.rows_conservation,
  CanonicalProgram.Recipes.BuiltTouches, builtRows_conservation.
```

### How it was found

By building the ten-item matrix mechanically instead of reasoning about it.
Grepping each recipe module for a theorem matching each section 2 item produced
one column with two dots:

| recipe | conservation |
|---|---|
| FoldDigest, Mixer, Transcript, Poseidon2Hash, KRecomposition, KLowNormBatch | present |
| **PiDecRecipe** | **absent** |
| **CanonicalProgram** | **absent** |

Π_DEC is the recipe most discussed in this session — its `rows_owner_not_unique`
is cited in nearly every assessment — and the missing item was never once named.
Attention had gone to the item known to fail, and the item nobody had checked
stayed unchecked. Same shape as cycle 386's padding.

### Why five conserving atoms are not a conserving recipe

Each of Π_DEC's five parts had its own conservation, which is what made the
absence invisible. It does not compose on its own: `Touches` has to name the
claim's own data, and the theorem has to say that *no emitted row* reaches
outside it. Without that, Π_DEC could emit a row mentioning an arbitrary
column — reading another recipe's allocation — and nothing in the record would
object. Under `placedRows`, where each part sits in its own window, a stray
column is a cross-window read.

### Four sub-lemmas were missing too

Composing needed conservation for parts that had none:

- `KZeroCheck.carriedZeroRows_conservation` — the `K`-valued form; a `K` zero is
  two rows, so the carried value contributes two combinations
- `KZeroCheck.paddingRows_conservation`
- `KConsistency.consistencyRows_conservation`
- `KRecomposition.recompositionsRows_conservation` — the fold over checks; the
  single-check form does not compose on its own

### The assembly states its exclusion rather than omitting it

`builtRows_conservation` covers the four built parts. The four selections carry
rows a deployment supplies, and `SelectedRecipe` says nothing about which columns
they mention, so no conservation is derivable for them — the same interface gap
`rows_owner_not_unique` records for item 3.
`rows_eq_builtRows_append_selections` pins that `builtRows` is `rows` minus the
selections and not a different program that happens to look similar.


---

## PIDEC-ROW-OWNERSHIP-WRONG-CONTRACT

```text
claim:
  Section 2 item 3 holds for Pi_DEC.
status: PROVED 2026-07-27 (cycle 393).
  PiDecOwnership.ownership_is_positional, owners_nodup, rows_eq_map_owners.
```

`PiDecRecipe.rows_owner_not_unique` exhibits one row value attributable to two of
Π_DEC's own recomposition receipts. It has been cited in every assessment since
cycle 368 as a kernel-checked obstruction to item 3.

**It is an obstruction to the wrong contract**, and this tree had already said so
— in `Poseidon2Ownership`'s header, under the heading *"Why positional rather
than by row value"*:

> making structural `Row` equality the ABI ... two receipts emitting equal rows
> is degenerate rather than incoherent

Two Π_DEC checks that constrain the same relation *should* emit the same row. A
program that deduplicated them would be a different program. What
`rows_owner_not_unique` refutes is value-based ownership, which is not the
contract.

### The contract that was never applied here

A row program has positions, and position is what a receipt owns.
`PiDecOwnership` builds for Π_DEC exactly what `Poseidon2Ownership` built for the
permutation: a receipt type, the row each receipt emits, and

- the program and the receipt list have equal length,
- the receipt list repeats nothing,
- position `i` is emitted by receipt `i`.

Nothing compares row values, so the two colliding checks are receipts
`recomposition i` and `recomposition j` with `i ≠ j` — both present, both
distinct, one row each.

### How it was found

By re-reading a blocker instead of re-citing it. Twenty-five cycles of
assessments repeated "kernel-checked obstruction" without asking whether the
obstruction was to something this tree wanted. The theorem was true the whole
time; the inference from it was not.

`Classical.choice` appears in `owners_nodup` and `ownership_is_positional`. That
is parity with `Poseidon2Ownership.ownership_is_positional`, which carries it
too — measured, not assumed.

### What remains

The assembly's `CanonicalProgram.Recipes.rows_owner_not_unique` gives two
reasons: the inherited one, now answered, and unconstrained selection rows. The
second is also a value-level objection — a selection duplicating a built
recipe's rows occupies different *positions* — so the same construction should
discharge it. `CANONICAL-PROGRAM-POSITIONAL-OWNERSHIP` was that step, and it is
closed — see below.


---

## CANONICAL-PROGRAM-POSITIONAL-OWNERSHIP

```text
claim:
  Section 2 item 3 holds for the assembled canonical program.
status: PROVED 2026-07-27 (cycle 394).
  CanonicalProgramOwnership.ownership_is_positional, owners_nodup,
  rows_eq_map_owners.
```

`CanonicalProgram.Recipes.rows_owner_not_unique` gave two reasons the assembly
fails item 3. Cycle 393 answered the first. This closes the second.

### The second reason had the same shape, and that was not obvious

It was recorded as:

> not inherited, not repairable by strengthening the built recipes, and a
> property of the interface a deployment fills

All three are true — **of value-based ownership**. A deployment may hand `step`
the fold-digest program, and no strengthening of `FoldDigestRecipe` prevents it.
Under positional ownership those rows are receipts `step i` and `foldDigest i`:
distinct receipts, distinct positions, one owner each.
`duplicating_selection_has_distinct_receipts` states it.

What made this one harder to see than the inherited reason is that it is
genuinely a property of the interface. It just is not a property that matters
for ownership. A deployment supplying duplicate rows may be *wasting* rows — the
same constraint asserted twice — and `N_canonical` counts both, correctly. That
is a cost question, not an attribution one.

### The construction

Eight constructors, one per part, each carrying the position within that part's
own emitted list; the program is the receipt list's image; the receipt list
repeats nothing; position `i` is emitted by receipt `i`. Identical in shape to
`PiDecOwnership` and to `Poseidon2Ownership` before it.

`Classical.choice` appears in `owners_nodup` and `ownership_is_positional` —
parity with both predecessors, measured.

### Where item 3 now stands

| object | item 3 |
|---|---|
| Poseidon2 permutation | `Poseidon2Ownership.ownership_is_positional` |
| Π_DEC | `PiDecOwnership.ownership_is_positional` (cycle 393) |
| the assembly | `CanonicalProgramOwnership.ownership_is_positional` (cycle 394) |

`rows_owner_not_unique` and `selection_may_duplicate_built_rows` stay true and
stay guarded. They are statements about row values, which is not the ABI.


---

## SEPARATION-RECIPE-ITEM-3-NOT-WALKED

```text
claim:
  Section 2 item 3 holds for the separation recipe.
status: PROVED 2026-07-27 (cycle 398).
  Poseidon2HashSeparation.ownership_is_positional.
```

`Poseidon2HashSeparation` was added in cycle 396 with nine of §2's ten items.
Item 3 was not written, and nothing noticed for a cycle.

**A new recipe does not inherit the checklist from the matrix that lists the old
ones.** Cycle 388 built the coverage matrix over eight recipes; cycle 396 added a
ninth and the matrix was not re-run. Re-running it is what found this.

Closed the same way as its three predecessors: a receipt type — the slot-zero
increment, and one equality per later slot — the program as the receipt list's
image, receipts duplicate-free, position `i` emitted by receipt `i`.

### A shared lemma instead of a third copy

`List.Nodup.map` is Mathlib's. Both `PiDecOwnership` and
`CanonicalProgramOwnership` had written a private `nodup_map_of_injective`. This
would have been the third. It now lives once, in `Poseidon2Ownership`, beside its
converse `nodup_of_map_nodup`. The two private copies remain and are the obvious
next consolidation — they are reachable only from modules that do not import
`Poseidon2Ownership`.

---

## RECIPE-ROW-OWNERSHIP-RECOMPOSITION-POSITIONAL

```text
claim:
  Section 2 item 3 holds for KRecomposition.
status: PROVED 2026-07-27 (cycle 399).
  KRecomposition.ownership_is_positional, owners_nodup,
  recompositionsRows_eq_map_owners, flatMap_getD_range.
```

`KRecomposition.recompositionRows_owner_not_unique` is the value-level negative,
and it is the one Π_DEC inherits. Cycle 393 answered it **for Π_DEC** by giving
Π_DEC positional ownership; `KRecomposition` itself never got the analogue.

The route is the one used four times now: `recompositionsRows base checks` is
`checks.flatMap (fun check => recompositionRows base check.1 check.2)` with each
inner list of length two, so a receipt is a check index paired with
`KEquality.RowOwner` — and `KEquality.allOwners`, `ownedRow` and
`rows_eq_map_owners` already exist to build on.

It was the last §2 item outstanding in this line, and it is closed.

### The receipt carries structure, deliberately

A position-only receipt — `RowOwner := Nat`, `ownedRow i = rows.getD i` — would
satisfy "exactly one receipt per row" and say nothing; it is the definitional
`rfl` the trap list rules out. The receipt here is a **check index paired with
one of `KEquality`'s two halves**, so the statement is that the emitted program is
the image of *that* list, which is a fact about how the rows are laid out.

### Four core lemmas were unavailable

`List.flatMap_congr`, `List.nodup_flatMap_of_disjoint`, `List.Nodup.map` and
`List.disjoint_left` are Mathlib's. The proof was restructured around what core
has: `List.flatMap_map`, `List.range_succ`, `List.nodup_append` as an iff, and
`flatMap_getD_range` written here as the `flatMap` analogue of
`PiDecOwnership.map_getD_range`.

`owners_nodup` came out **without** `Classical.choice`, unlike the three sibling
ownership proofs — the `range_succ` induction avoids the classical step that
`simp` was reaching for elsewhere.

### Item 3, everywhere

| object | theorem |
|---|---|
| Poseidon2 permutation | `Poseidon2Ownership.ownership_is_positional` |
| `KRecomposition` | `KRecomposition.ownership_is_positional` (399) |
| Π_DEC | `PiDecOwnership.ownership_is_positional` (393) |
| the separation recipe | `Poseidon2HashSeparation.ownership_is_positional` (398) |
| the assembly | `CanonicalProgramOwnership.ownership_is_positional` (394) |
| the transcript | `TranscriptRecipe.transcriptRows_owner_unique` (value-level, and it holds) |
