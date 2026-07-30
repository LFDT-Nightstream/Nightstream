# PIDEC-RECOMPOSITION — radix-`b` recomposition as Π_DEC's algebraic core

Owner: `Nightstream/Implementation/R1CS/Canonical/KRecomposition.lean`
Guard: `tests/Axioms/CanonicalKRecomposition.lean` (23 theorems, measured)
Evidence: `model-proved`

---

## PIDEC-RECOMPOSITION-SHAPE

```text
claim:
  Pi_DEC's algebraic content is a single relation, parent = sum_i b^i * child_i,
  instantiated on four carriers. The remaining validators are guards.
status: DETERMINED 2026-07-27 (cycle 327) by reading
  neo_reductions::api::dec::verify_dec_public in full.
```

`verify_dec_public` is 268 lines, of which the great majority are shape guards
and error formatting. Its constraint content is:

| carrier | site | count |
|---|---|---|
| every entry of the public `X` matrix | `split_b_matrix_k(parent.X, k, b)` | `D · m_in` |
| every lane of every `y_ring` row | the `y_lhs` loop | `t · d_pad` |
| every lane of `y_zcol` | guarded by `enforce_y_zcol_recomposition` | `d_pad` |
| every `aux_openings` entry | the final scalar loop | `|aux_openings|` |

all four of which are the same relation.

**Corrected 2026-07-27 (cycle 334): three of those four are live.** The table
above is exactly right about `verify_dec_public` read on its own, and wrong
about `pi_dec::verify`, which is the entry point. `pi_dec::verify` calls
`validate_supported_sidecars` *before* the engine, and that validator rejects
any claim with a non-empty `aux_openings`. By the time the `aux_openings`
recomposition loop is reached, the list it iterates is empty; the loop runs
zero times, always. See `PIDEC-SIDECARS` below.

What sits outside it:

| obligation | owner |
|---|---|
| digit range on `X` (balanced base 2 gives `{-1, 0, 1}`) | `KLowNorm` |
| `ct[j] = y_ring[j][0]` | `KConsistency` |
| `y_zcol` lanes at index `≥ D` are zero | `KZeroCheck.paddingRows` |
| `s_col` shared between parent and children | `KConsistency` |
| `combine_b_pows(children.c, b) = parent.c` | not this layer — Ajtai |

### Supersedes

`PIDEC-CHECK-CLASSIFICATION` recorded `engine::verify_pi_dec` as **unexamined**
and named it "the real core". It has now been read. The correction to that
entry is one of *size*, not direction: the core is real, and it is smaller than
its guard surface — one relation, four instantiations.

---

## PIDEC-RECOMPOSITION-COST

```text
claim:
  One recomposition costs two rows and allocates nothing, at every child count.
status: PROVED. recompositionRows_length, recompositionCost_rows,
  recompositionColumns_nodup.
```

`b` is public, so `b^i` are public constants and scaling is a **coefficient
rewrite**, not a multiplication. Nothing is allocated and no product row is
emitted for the sum itself. The whole cost is the equality against the parent:
two rows, one per extension coordinate.

This is the non-obvious direction. A `K`-valued recomposition over `k_rho = 14`
children *looks* like it needs 14 extension multiplications; it needs **none**,
because the scalars live in the base field and act coordinatewise.

### Not a claim about combination length

The emitted combination's syntactic length is the sum of the children's — it
grows with `k`. Its **row cost** does not. Prompt §4.7: a concatenating
combination grows without bound while its support does not, and neither figure
is evidence about the other.

---

## PIDEC-RECOMPOSITION-CONFORMANCE

```text
claim:
  The Horner combination the encoder emits computes the same function as the
  verifier's explicit power accumulator.
status: PROVED. powerSumFrom_eq_hornerValue, powerSum_one.
```

The encoder builds `c₀ + b·(c₁ + b·(…))`, because each level reuses the previous
level's combination instead of tabulating powers.

`verify_dec_public` does not do that. It builds a table with `p_k *= b_k` and
then takes `Σ pow_i · child_i`. Same function, different expression.

`powerSumFrom` transcribes the verifier's loop, accumulator and all, and
`powerSumFrom_eq_hornerValue` proves the two agree from an arbitrary starting
power; `powerSum_one` specialises to the verifier's `p_k = 1`. **Without this
theorem the encoding would be checking a relation the verifier does not check**,
and the agreement would be an assumption wearing a definition's clothes.

Not vacuous: `hornerValue 2 [1,1,1] = 7` and `hornerValue 3 [1,1,1] = 13`, so
neither side is constant in the base.

---

## PIDEC-RECOMPOSITION-SOUNDNESS

```text
claim:
  Satisfaction forces the parent to be the radix-b recomposition of the
  children, and an honest decomposition satisfies the rows.
status: PROVED. recompositionRows_sound, recompositionRows_honest.
```

Soundness quantifies over all assignments with `z 0 = 1`. Completeness is under
the caller's own assignment: nothing is allocated, so there is no witness to
extend — the strongest form completeness can take.

The consumer that constructs the completeness premise is a Π_DEC prover, which
produced the children as balanced radix-`b` digits of the parent. The premise is
therefore discharged by construction rather than moved.

### Negative controls

The honest fixture is accepted, and four independent mutations are each
rejected: parent low coordinate, parent high coordinate, one child digit, and
the base itself (2 → 3). The base control matters most — it confirms the
coefficient rewrite is load-bearing rather than decorative.

---

## PIDEC-RECOMPOSITION-CONSERVATION

```text
claim:
  Every column of every emitted row is a child's, the parent's, or the
  constant wire.
status: PROVED. recompositionRows_conservation, via mentions_recomposeComb.
```

Scaling changes no column (`mentions_map_scale`) and concatenation introduces
none (`mentions_append`), so the recomposition's support is exactly the union of
the children's. The constant-wire arm is real: `KEquality` reads a literal `1`
in its `B` operand.

---

## Open

- The commitment fold `combine_b_pows` is not arithmetic this layer owns; it
  needs the Ajtai layer.
- Which carriers are recomposed — `D · m_in`, `t · d_pad`, `d_pad`,
  `|aux_openings|` — is decoder work from a claim's shape. The check-level count
  is therefore stated as a fold over per-carrier receipts
  (`recompositionsRows_length`), not a closed formula. A closed formula would be
  a subtotal presented as a total.
- `verify_dec_public` does **not** validate the parent's old-point `y_zcol`.
  Rust's own comment says so and says the delayed-projection authority bridge
  must close it. That is a recorded gap in shipping Rust, not in this encoding.

---

## PIDEC-LOWNORM-BATCH

```text
claim:
  The low-norm check over many digits is one row program with one witness, and
  both of its well-formedness arms are necessary.
status: PROVED 2026-07-27 (cycle 330). KLowNormBatch.
evidence: model-proved
```

Π_DEC decomposes every entry of the public `X` matrix into `k_rho = 14`
balanced base-2 digits, so the low-norm check is the decomposition's dominant
row cost. `KLowNorm` owns one digit; this owns the batch.

### Rows batch trivially; the witness does not

`KLowNorm.lowNormWitness` writes one square to one allocated column, and each
digit needs its own. Composing those writes is sound only under `WellFormed`:

| arm | statement | why it is needed |
|---|---|---|
| `distinct` | the allocated columns are `Nodup` | a shared column means the outer write clobbers the inner square and the inner digit's rows fail |
| `fresh` | no checked combination reads *any* allocated column | a combination reading another digit's column has its value changed by that digit's write |

### Both arms are demonstrated, not asserted

Three fixtures, all with honest cubes under the same base assignment, so no
failure below is the cube test failing:

| fixture | columns nodup | satisfied |
|---|---|---|
| distinct columns, no cross-reads | yes | **yes** |
| two digits sharing column 3, squares 1 and 0 | no | **no** |
| tail digit reads the head's square column | **yes** | **no** |

The third fixture is the one that matters: its allocation *is* collision-free,
so its failure isolates `fresh` rather than `distinct`. Between them the two
controls show that "exact column ownership with no collision" is a load-bearing
constraint here and not a naming convention.

### Cost

`2 · digits.length` rows and `digits.length` columns, both derived from the
emitted program. Stated as a fold over per-digit receipts first, because how
many digits there are is a property of the claim's shape.

---

## PIDEC-ASSEMBLY

```text
claim:
  Pi_DEC's decomposition algebra is one emitted row program with one honest
  witness, and only its digit range check allocates.
status: PROVED 2026-07-27 (cycle 331). PiDecRecipe.
evidence: model-proved
```

### The program

```text
rows claim =
    KRecomposition.recompositionsRows base (xEntries ++ yRingLanes
                                            ++ yZcolLanes ++ auxOpenings)
 ++ KLowNormBatch.batchRows xDigits
 ++ inactiveX.flatMap KZeroCheck.zeroRows
 ++ KZeroCheck.paddingRows yRingPadding
 ++ KConsistency.consistencyRows consistency
```

The four recomposed carriers are **one** `recompositionsRows` call, not four,
because `PIDEC-RECOMPOSITION` proved they are one relation.

### The count is a fold, and it was checked

```text
2·|recompositions| + 2·|xDigits| + |inactiveX|
  + 2·|yRingPadding| + 2·|consistency|
```

Every term is a per-atom receipt. On a fixture with 4 recompositions, 2 digits,
3 inactive entries, 1 padded lane and 1 consistency pair, the emitted program
has **19** rows and `cost.recurringRows` is 19 — derived from the list, then
cross-checked against the cost tuple rather than asserted from it.

### One allocator, and that is the whole composition problem

Every atom but one is allocation-free: recompositions, zero checks, padding and
consistency read carried values and write nothing. `KLowNormBatch` is the sole
allocator, one column per digit.

So the assembled honest witness **is** the batch's witness — there is nothing
else to extend — and every other part must still hold under that extension. It
does, but only if no other part reads an allocated column. That is `Fresh`, and
it is a hypothesis rather than an assumption: `PIDEC-LOWNORM-BATCH` already
demonstrated that a violated freshness arm makes a batch unsatisfiable even
when its allocation is collision-free.

`columns` is therefore `batchColumns xDigits` and `columns_nodup` is the
batch's `distinct` arm, unchanged. The program allocates nothing else.

### Honest completeness is exactly soundness' converse

Each field of `Honest` is verbatim the conclusion of the corresponding
`rows_sound_*` theorem, so the recipe is complete for precisely the transitions
it accepts — no gap in either direction.

### Still outside

`combine_b_pows(children.c, b) = parent.c` needs the Ajtai layer, and the
parent's old-point `y_zcol` is not validated by `verify_dec_public` at all.
Both stay named in `PIDEC-RECOMPOSITION` rather than being silently absorbed
into this assembly.


---

## PIDEC-SIDECARS

```text
claim:
  supported_sidecars is two emptiness assertions, decoder-side, no rows - and
  its consequence is that the aux_openings recomposition carrier is dead.
status: DETERMINED 2026-07-27 (cycle 334) by reading pi_dec::verify's order.
```

### The check

`validate_supported_sidecars_one` rejects a claim when `aux_openings` is
non-empty, and again when `c_step_coords` is non-empty. Two list-emptiness
assertions. Like the five length assertions in `PIDEC-CHECK-CLASSIFICATION`,
these are **decoder-side**: emitting rows for them would fabricate constraints
the protocol does not have.

The check was queued as possible row work. Reading it first — rather than
building and then discovering — is the lesson `PIDEC-FOLD-DIGEST` recorded one
cycle earlier, applied.

### The consequence is the interesting half

`pi_dec::verify` runs, in order: `validate_supported_sidecars` at line 320, then
`engine::verify_pi_dec` at line 327, which delegates to `verify_dec_public` and
its `aux_openings` recomposition loop.

So the loop is unreachable with a non-empty list. **In the shipping Π_DEC path
the recomposition has three live carriers, not four.**

`PiDecRecipe.recompositions_length_without_sidecars` and
`rows_length_without_sidecars` state this as theorems, so the dead carrier is
visible in the count rather than hidden inside a fold. The recipe itself needs
no change — an empty carrier list contributes no rows — but a reader deriving a
figure from "four carriers" would overcount by `2 · |aux_openings|`.

Prompt section 4.3: a number can be exactly right about something narrower than
the sentence containing it. "Four carriers" was right about a function and wrong
about the verifier that calls it.

---

## PIDEC-ADV-PRESENCE

```text
claim:
  adv_recomposition has three branches and only one reaches the commitment
  layer. The other two were never blocked.
status: PROVED 2026-07-27 (cycle 340). AdvPresenceBoundary.
evidence: model-proved
```

### The carried dependency was half right

Every ledger entry since cycle 326 has said "`adv_recomposition`, needing the
Ajtai commitment layer". Reading `recompose_adv` shows three branches:

| branch | result | reaches the mixer |
|---|---|---|
| no child carries a sidecar | `Ok(None)` | no |
| some but not all carry one | `Err(AdvPresence)` | no |
| every child carries one | `Ok(Some(combine …))` | yes |

`require_homogeneous`'s own comment names the first: "`Ok(None)` means a plain
(non-Nebula) fold".

So the all-or-nothing presence rule is structure, not arithmetic.
`plain_profile_is_structural` proves that in the plain profile the check reduces
to "the parent carries no sidecar either", and
`mixer_unreached_unless_all_present` proves that a single absent child rejects
the fold before any commitment arithmetic happens.

### Consistent with the sidecar finding

`PIDEC-SIDECARS` established that `validate_supported_sidecars` rejects any claim
with a non-empty `aux_openings` or `c_step_coords`, making the fourth
recomposition carrier dead in the shipping path. This is the same profile seen
from another side: the shipping Π_DEC fold is the plain one, and both the
`aux_openings` recomposition and the `adv` commitment fold are inert in it.

Two independent readings agreeing on which profile ships is worth more than
either alone.

### What stays outside

The mixer itself, in the all-present branch. That is the Ajtai fold and this
layer does not own it. The correction is to the *scope* of the blockage, not to
its existence.

---

## PIDEC-COMMITMENT-MIXER

```text
claim:
  The commitment mixer's arithmetic is the radix-b recomposition over
  commitment coordinates. It was never blocked on the Ajtai layer.
status: PROVED 2026-07-27 (cycle 341). CommitmentMixerRecipe.
evidence: model-proved
```

### The mixer is the recomposition

`combine_b_pows` is `acc = 0; pow = 1; for c in cs { acc += pow · c; pow *= b }`,
and `scale_commitment_add_inplace(acc, scalar, c)` is `acc += scalar · c`
coordinatewise — the `ZERO`, `ONE` and `-1` cases in `neo-ajtai/src/commit.rs`
are fast paths for the same map. A `Commitment` is a flat `d × kappa` vector of
field elements (`neo-ajtai/src/types.rs`).

So the mixer is `Σ_i b^i · c_i`, coordinate by coordinate — exactly the relation
`KRecomposition` owns, with `powerSumFrom_eq_hornerValue` already proving the
accumulator loop and Horner form agree. That proof is reused, not repeated.

**One row per coordinate**, because a commitment coordinate is a base-field
value rather than an extension element.

### The correction

Every ledger entry from cycle 326 to 340 said the mixer needed "the Ajtai
commitment layer". `PIDEC-ADV-PRESENCE` narrowed that to one of three branches.
This narrows it again: the arithmetic of that branch is not blocked either.

That is the **fourth** inherited dependency this session to fail on reading, and
the four form a pattern worth naming: each was carried forward in summaries for
many cycles, each named a real component, and each turned out to describe
something narrower than the sentence containing it.

---

## COMMITMENT-MIXER-NOT-BINDING

```text
claim:
  The mixer rows establish recomposition and nothing about binding.
status: RECORDED 2026-07-27 (cycle 341).
```

The rows say the parent commitment is the radix-`b` combination of the
children's, coordinate by coordinate. They say nothing about whether a
commitment determines its opening.

`mixing_alone_does_not_bind` makes the gap concrete rather than rhetorical: at
base two the digit lists `[2]` and `[0, 1]` both recompose to `2`, and neither is
degenerate or all-zero. The mixer is linear, so collisions are expected.

What confines digits is `KLowNorm`'s centered window; what makes a commitment
determine its opening is the Ajtai binding assumption. Recomposition alone does
neither, and this module claims neither. Prompt section 4.6 states the general
rule: shared arithmetic does not transfer a security contract.

**The Ajtai dependency is real and remains** — it is the binding property, not
the fold.

### Corrected 2026-07-27 (cycle 347): the owner already exists

"A hardness assumption, not something a row program establishes" was right about
what binding *is* and wrong about where it stands. It is not unmodelled:

`Nightstream/Protocol/FPrime/ConcretePhi81/AccumulatorBinding.lean` owns it, with
a named failure event and a stated reduction. Its own ownership table records the
row this module needs:

| stage path | obligation | class | owner |
|---|---|---|---|
| `fprime.accumulator.commitments.binding` | equal handles recover the strict Π_DEC view **or expose compression/opening failure** | security reduction | `parent_children_eq_or_commitmentFailure` |

and its header states the governing principle directly: *"Neither digest is
authority unless recomputed from its carrier and reduced through the
corresponding failure partition… public Π_DEC recomposition alone is
insufficient."*

That last clause is exactly what `COMMITMENT-MIXER-NOT-BINDING` says. This
module re-derived, with a two-digit collision, a fact the owning module already
states as its premise. The collision is still a fine demonstration; what was
wrong is the implication that nothing in the tree covered it.

### Withdrawn 2026-07-27 (cycle 348): the proposed wiring is the wrong shape

Cycle 347 proposed connecting `CommitmentMixerRecipe.mixerRows_sound` to
`parent_children_eq_or_commitmentFailure`, and said the reduction had to be read
in full first. It was, and the proposal does not survive the reading.

The reduction's hypotheses are **two** accepted Π_DEC transitions plus a digest
collision between them:

```text
(leftAccepted  : PiDEC.Accepted algebra ⟨leftParent,  left⟩)
(rightAccepted : PiDEC.Accepted algebra ⟨rightParent, right⟩)
(sameDigest    : commitmentFamilyDigest scheme leftParent left
                   = commitmentFamilyDigest scheme rightParent right)
⊢ (leftParent = rightParent ∧ left = right)
    ∨ CommitmentFamilyFailure semantics params scheme left
```

`mixerRows_sound` is about **one** transition and **one** check. Feeding a
single-transition arithmetic fact into a two-transition collision reduction is a
category error, not a stronger theorem. And `PiDEC.Accepted` is everything Π_DEC
checks, of which the mixer is one component.

### Where the named-event form actually belongs

At a claim that **carries a digest**. The mixer carries none: it relates
commitment coordinates by radix-`b` recomposition and asserts nothing about what
any digest compresses. Soundness to the frozen relation is the right and
complete form at that scope.

`FoldDigestRecipe` does carry digests, and now says so:
`digestRows_claim_is_lane_equality` states that lane agreement is the whole
claim, with the module recording that authority is discharged at
`AccumulatorBinding` and why its two-transition reduction does not take these
rows as input.

The value of cycle 347's queue item was in the instruction attached to it —
*read the reduction in full before wiring* — which is what prevented the error.


---

## PIDEC-LOWNORM-CANONICAL-ALLOCATION

```text
claim:
  WellFormed's two arms are different kinds of obligation, and the encoder can
  discharge one of them.
status: PROVED 2026-07-27 (cycle 349). KLowNormBatch.canonicalDigits.
evidence: model-proved
```

`WellFormed` asks for `distinct` and `fresh`, and "supply a well-formed batch"
treats them alike. They are not alike:

- `distinct` is about the **allocation**, which the encoder chooses;
- `fresh` is about what the checked **values** read, which it does not.

`canonicalDigits base` picks the allocation — digit `i` takes column
`base + i + 1` — and:

| theorem | content |
|---|---|
| `canonicalDigits_nodup` | `distinct` discharged outright |
| `canonicalDigits_column_gt`, `canonicalDigits_column_le` | every column lies in `(base, base + n]`, which is the bound `ColumnWindows.placeAll_columns` asks for |
| `canonicalDigits_wellFormed` | `WellFormed` from `fresh` alone |

So a deployment owes **one** condition where it previously owed three, and the
one that remains is genuinely its own: no numbering can make a value stop
reading a column; only knowing where the values live can.

### Why this was worth checking

The previous cycle recorded per-recipe column numbering as "a deployment
instantiation rather than a theorem". That was half right — the *choice* of
numbering is a deployment's, but its **properties** are theorems, and two of the
three obligations were provable without knowing anything about the deployment.

Ninth instance this session of a carried claim that named something real and
placed it further away than it was.
