# Poseidon2 canonical encoding — open items raised by external review 2026-07-26

Three defects found by review of cycles 167–173. Each is a named property here
rather than a comment in the source.

---

## POSEIDON2-NORMALIZED-EMISSION

```text
claim:
  The program the encoding EMITS must be the program whose cost is counted.
status: CLOSED 2026-07-26 (cycle 176).
```

`Poseidon2Coefficients.rowTermCount` applies `normalize` *inside the metric*,
while `canonicalProgram` still emits raw `flatMap` combinations. So
`canonicalProgram_termCount` is the cost of a representation that is never
constructed. Under the governing prompt's §3 that is "a count without a
construction" — the count is not derived from an emitted row program.

What is actually established: rows 352 and auxiliary columns 344 describe the
emitted program and are unaffected. Only the *coefficient* figure is attached to
a representation that does not exist as a term.

**Resolved.** `Poseidon2Normalized` emits `normalizedCanonicalProgram` and
measures it with `rawTermCount`, which counts entries as they stand and does no
normalizing of its own. `rowHolds_normalizeRow` and `satisfies_normalizeProgram`
give satisfaction in both directions from `lcEval_normalize`, so soundness
(`normalizedCanonicalProgram_computes_reference`) and honest completeness
(`honest_satisfies_normalized`) transfer without reproving anything.
`mentions_normalizeRow` carries ownership and conservation over operandwise.
Length is still 352 and the cost decomposition is unchanged — it now attaches to
a constructed term.

---

## POSEIDON2-SBOX-TRACE-ALIGNMENT

```text
claim:
  The canonical S-box should allocate the same intermediates as production.
status: CLOSED 2026-07-26 (cycle 175). The canonical encoding now uses
        production's chain.
```

Production (`enforce_sbox_x7`, `r1cs_circuit/poseidon2.rs:133`) allocates

    x² = x·x     x⁴ = x²·x²     x⁶ = x²·x⁴     x⁷ = x·x⁶

The canonical encoding allocates

    x² = x·x     x³ = x²·x      x⁶ = x³·x³     x⁷ = x⁶·x

Both are four multiplications, so **row and column counts are unaffected** —
344 S-box rows and 344 auxiliary columns stand either way.

The coefficient cost is not equal. Counting normalized operand entries, with
`|c|` the scheduled input combination's size:

| chain | per S-box | at `|c| = 31` |
|---|---|---|
| production `x²/x⁴/x⁶/x⁷` | `3·|c| + 9` | 102 |
| canonical `x²/x³/x⁶/x⁷` | `4·|c| + 8` | 132 |

Production's chain places the input in three operand positions; the canonical
one placed it in four. Since `|c|` reaches 31 in the terminal block, adopting
production's chain is strictly cheaper *and* buys trace alignment — no tradeoff
to weigh.

This reversed the direction of the finding: the review raised alignment, but the
stronger reason was cost.

**Resolved.** `SboxFrame` now carries `square, fourth, sixth, output`;
`rowFourth` replaces `rowCube`; `sboxRows_chain`, `sbox7`, `chainSlot` and the
honest row proofs follow production. `sboxRows_termCount` is now
`3·|input| + 9`. Row count 352 and auxiliary column count 344 are unchanged, as
predicted. Soundness, honest completeness and the axiom gate all survived the
change without weakening.

---

## POSEIDON2-ROW-OWNERSHIP-UNIQUENESS

```text
claim:
  Distinct receipts emit distinct rows.
status: CLOSED 2026-07-26 (cycle 176) as stated — but see
        POSEIDON2-POSITIONAL-OWNERSHIP below, which supersedes it. What is
        proved is pairwise row-value distinctness, not an assembled
        exactly-one-owner theorem over the emitted program.
```

`everyPermutationRow_has_owner` (renamed 2026-07-26 from
`..._has_exact_owner`) concludes a disjunction of existentials. Nothing forbids
two receipts from emitting the same row *value*. The docstring claimed "exactly
one owner"; it did not prove it.

The column analogue `everyPermutationColumn_has_exact_owner` is genuinely
stronger — it carries a `∀ otherIndex otherSlot` clause — so the two should not
be cited together as one ownership result.

**Resolved** in `Poseidon2Layout`, through the `c` operand — every emitted row
writes to exactly one column and that column names its receipt.
`sboxRows_target` pins each S-box row's write target to one of its own four
frame columns; `sboxRows_disjoint` rules out two S-boxes sharing a row via
`sboxColumn_injective`; `bindRow_not_sboxRow` separates binding rows from S-box
rows, since well-formedness puts output ports below `auxBase` and frame columns
at or above it; `bindRow_injective` separates binding rows from each other via
`outputInjective`.

The `WellFormed` hypothesis is load-bearing for the last two and not removable:
a layout aliasing an output port into the auxiliary block genuinely does emit
two receipts writing the same column. `canonicalLayout_wellFormed` discharges
it.

---

## POSEIDON2-FIELD-CANONICAL-COUNT

```text
claim:
  Stored coefficients must be canonical residues, and the entry count must be
  the nonzero count rather than an upper bound.
status: CLOSED 2026-07-26 (cycle 178). The emitted program now carries the
        field-canonical form.
```

`normalize` merges duplicate columns but adds coefficients as unbounded
naturals. Two consequences, neither affecting `lcEval`:

- a merged coefficient can exceed the prime, so a stored entry need not be a
  canonical residue — and Rust stores canonical residues, so eventual row-level
  equality against production could fail on representation alone;
- a merged coefficient congruent to zero is retained, so the entry count bounds
  the nonzero count instead of being it.

`POSEIDON2-NO-CANCELLATION` as previously framed covered only the second. The
first was not recorded at all.

**Closed:** `LinCombNormal.fieldNormalize` reduces every coefficient modulo the
prime and drops the zeros. `lcEval_fieldNormalize` proves it semantics
preserving, `fieldNormalize_canonical` that every stored coefficient is a
canonical residue, and `fieldNormalize_nonzero` that none vanishes — so its
length *is* the nonzero coefficient count, not a bound.

**Applied.** `Poseidon2Normalized.normalizeRow` now uses `fieldNormalize`, so
every entry of every emitted row is a canonical nonzero residue
(`normalizeRow_entries`). Soundness and honest completeness transferred
unchanged, since `lcEval_fieldNormalize` has the same shape as
`lcEval_normalize`.

Two statements had to weaken to stay true, and both weakened in the direction
that is actually needed:

- `mentions_normalizeRow` is now an implication, not an iff. `fieldNormalize`
  drops a column whose coefficient vanishes, so the emitted row can reference
  strictly fewer columns. Conservation needs exactly "introduces no column", so
  nothing is lost; support *equality* would simply be false.
- `normalizedCanonicalProgram_termCount` became
  `..._termCount_le`. The emitted count is the nonzero coefficient count; the
  merge-only decomposition bounds it from above, and the gap is exactly the
  coefficients that cancel. Turning the bound into an equality is
  `POSEIDON2-NO-CANCELLATION`.

This is the honest shape of the result: the emitted program's nonzero count is
now a property of a real term, but it is not a closed numeral, and it cannot
become one from structure alone — cancellation is an arithmetic fact about the
selected constants.

---

## POSEIDON2-POSITIONAL-OWNERSHIP

```text
claim:
  Every row of the emitted program has exactly one receipt, assigned by
  position rather than by row value.
status: CLOSED 2026-07-26 (cycle 179).
```

Cycle 176 proved the ingredients — `sboxRows_disjoint`, `bindRow_not_sboxRow`,
`bindRow_injective` — and the spec recorded
`POSEIDON2-ROW-OWNERSHIP-UNIQUENESS` as closed. That was premature in two
respects:

- there is no assembled theorem over `normalizedCanonicalProgram` stating that
  every row has exactly one owner; the lemmas are ingredients;
- the lemmas are stated on row *values*, which makes structural `Row` equality
  the ownership ABI. That is the wrong ABI: two receipts emitting equal rows is
  a degenerate but not incoherent situation, and position is what a row program
  actually has.

**Resolved** in `Poseidon2Ownership`, though not with an `ownerAt : Fin 352 →
RowOwner`. Indexing into a `flatMap` positionally needs a block-decomposition
lemma core does not supply, and the same content is available more directly:
`ownership_is_positional` states that the emitted program *is*
`allOwners.map (normalizeRow ∘ ownedRow ...)`, that `allOwners` is duplicate-free,
and that the lengths agree. Position `i` of the program is therefore emitted by
receipt `i` and by no other — which is what `ownerAt` was for, obtained by
construction rather than by indexing.

`allOwners_nodup` goes through `ownerIndex : RowOwner → Nat` (S-box `index` step
`s` at `4·index + s`, binding lane `l` at `344 + l`) and
`allOwners.map ownerIndex = List.range 352`, kernel-checked by `decide` under a
raised `maxRecDepth`. No `native_decide`, and the axiom report is unchanged.

Nothing in the statement compares row values, so two receipts emitting
structurally equal rows would still be owned separately. The cycle-176
value-level lemmas remain true and are now corollaries rather than the
contract.


---

## POSEIDON2-NO-CANCELLATION

```text
claim:
  No coefficient of the emitted program vanishes modulo the prime, so the
  4397 upper bound is the exact nonzero count.
status: OPEN, but narrower than cycles 178/181/182 stated.
```

Those cycles said the exact nonzero count "cannot close from structure."
That is too strong and is corrected here. The obligation splits three ways and
only one part is genuinely arithmetic:

**Structural — no constants involved.** For a full-round state, `applyMatrix`
over singleton lanes produces exactly one entry per column,
`[(col_j, M i j % p)]`, with no merging. The coefficients are therefore the
matrix entries themselves, and `externalMatrix_nonzero` and `externalMatrix_lt`
(both proved) say those are nonzero canonical residues. `fieldNormalize` drops
nothing. This covers the state part of all 56 full-round S-boxes and the 8
terminal binding rows.

**One bit per round constant.** The constant wire carries the round constant
itself. It survives iff that constant is nonzero modulo the prime — 86 scalar
checks, not an argument about products. Nothing is known about them until
`POSEIDON2-ROUND-CONSTANT-CONFORMANCE` pins the values.

**Genuinely arithmetic.** Inside the partial block coefficients accumulate
across rounds as sums of products of internal matrix entries, so a vanishing
sum is possible in principle and only computation rules it out. This covers the
22 partial S-boxes and the 8 terminal round-0 S-boxes — 30 of 86, and exactly
the ones with large inputs.

So the honest statement is that the structural part of no-cancellation is
already within reach of proved lemmas, the constant-wire part is 86 scalar
non-zero checks, and only the partial block needs an arithmetic argument over
concrete constants. The earlier blanket claim understated what is reachable.

---

## POSEIDON2-SPONGE-CHUNKING

```text
claim:
  Splitting a flat field list into rate-sized chunks is a fixed, stated choice.
status: OPEN — deliberately not folded into the sponge.
```

`Poseidon2Sponge.absorb` takes the input already split into chunks. That is not
laziness: chunking fixes what happens to a final short chunk, and therefore what
the padding has to be. Folding it into the sponge would hide that choice inside
a definition instead of naming it.

Rust uses `input.chunks(POSEIDON2_RATE)`, which yields a final short chunk when
the length is not a multiple of the rate, and then pads with a single `1` into
lane 0 regardless. Whether that is the intended domain separation — and whether
it is injective on inputs of different lengths — is exactly the question this
property has to answer, and it is a security question, not a bookkeeping one.

---

## POSEIDON2-SPONGE-INTER-PERMUTATION-BINDING

```text
claim:
  Whether the sponge must materialize state between permutation calls.
status: DECIDED 2026-07-26 (cycle 191) — adopt the BOUND shape.
```

**Decision: bind after every permutation.** Three reasons, in order of weight:

1. It matches production, which materializes state between calls. A canonical
   encoding whose whole purpose is comparison against production should not
   differ in structure on a point that is not forced.
2. The trade is 8 rows against 96 coefficients per boundary — the unbound shape
   is *worse* on coefficients by a factor of twelve. Taking it would require
   knowing a row costs more than twelve coefficients in the target prover, and
   that is not known.
3. The bound shape reuses `canonicalProgram` unchanged, so soundness, honest
   completeness, ownership and conservation all compose rather than needing the
   boundary support argument restated.

The unbound shape and its arithmetic stay recorded below. If a prover
measurement later shows a row costs more than twelve coefficients, this
decision should be revisited — the analysis is done and only the choice would
change.

In the never-materialize encoding, absorption is `state[lane] += chunk[lane]`,
which adds a term to a carried combination and emits **no row** — exactly like a
linear layer. Padding is the same: `+1` on the constant wire. So neither
absorption nor padding is inherently a row cost.

But `canonicalProgram`'s 352 rows include 8 terminal binding rows that collapse
the final combination into declared output ports. Inside a sponge those ports
are consumed immediately by the next absorption, so the binding may be
unnecessary between calls and needed only once, at the digest.

Two shapes, and the difference is not small:

| shape | rows |
|---|---|
| bind after every permutation | `(chunks + 1) · 352` |
| bind only at the digest | `(chunks + 1) · 344 + 8` |

For a 4-chunk input that is 1,760 against 1,384 — a 21% difference decided
entirely by whether inter-permutation binding is required.

It is not obviously free to drop. Carrying combinations across a permutation
boundary means the support bound must be re-established across the join: the
support recurrence in `Poseidon2Support` is stated within one permutation and
resets at each full round, and a sponge that never binds would need that
argument extended across calls. Whether support still collapses at the first
full round of the next permutation is exactly the question, and it decides
whether the cheaper shape is implementable.

**This must be settled before `hashPrior` and `hashNext` are written**, because
the two shapes are different row programs, not the same program counted two
ways.

### Answer: support does collapse across the join

Tracing the carried support across a permutation boundary, using lemmas already
proved:

| point | support | why |
|---|---|---|
| `finalState` of permutation *k* | 8 | full round resets (`terminalState_succ_mentions`) |
| after absorbing a ≤4 chunk | ≤ 12 | up to four input columns join |
| permutation *k+1* pre-layer | ≤ 12 | matrix application unions, adds nothing (`mentions_applyMatrix`) |
| its round-0 S-box inputs | ≤ 13 | plus the constant wire |
| `initialState 1` | **8** | round 0 S-boxes every lane; outputs are fresh |

So the recurrence survives the join and the unbound shape is implementable. Only
each permutation's *first* round sees a wider input; everything after it is
unchanged.

### The trade, per boundary

Dropping the binding saves the 8 terminal binding rows. It widens the next
permutation's 8 round-0 S-box inputs from 9 columns to 13, and by
`sboxRows_termCount` each S-box costs `3·|input| + 9`:

    bound:    3·9  + 9 = 36 per S-box
    unbound:  3·13 + 9 = 48 per S-box

So **8 rows saved against 96 coefficients added, per boundary.**

Whether that is worth taking depends on the relative cost of a row versus a
coefficient in the target proof system. That is an engineering judgement about
the prover, not something derivable here, and it should be made deliberately
rather than by whichever recipe gets written first.

**Not yet proved in Lean.** The supporting lemmas all exist
(`mentions_applyMatrix`, `terminalState_succ_mentions`,
`partialState_mentions_subset`, `sboxRows_termCount`), but stating the boundary
argument formally needs the sponge encoding, which does not exist. This section
is a derivation on paper from proved lemmas, and is marked as such.

---

## POSEIDON2-INITIAL-STATE-GENERALIZATION

```text
claim:
  Absorption is free only if `initialState` accepts a carried state rather than
  declared input ports.
status: OPEN — corrects a claim made in cycle 190 without checking the interface.
```

Cycle 190 asserted that sponge absorption "adds a term to a carried combination
and emits no row, exactly like a linear layer." That is true of the
never-materialize *idea* but false of the interface as written:

```lean
def initialState (layout : Layout) : Nat → State
  | 0 => applyMatrix externalMatrix (fun lane => [(layout.inputPort lane, 1)])
```

The base case takes **one column per lane**. Absorption needs
`state[lane] + chunk[lane]` — two terms — which this cannot express. So with the
current interface absorption costs 8 materialization rows per chunk, and the
cycle-190 row arithmetic is wrong.

This is the same defect class as the original `SboxFrame.input : Nat` problem
that forced `Poseidon2Core` to be rebuilt: a property was claimed of the
encoding without checking that the representation could express it. It is worth
noting that the identical mistake recurred, in the identical shape, in a module
whose header documents the first occurrence.

**The fix is small and strictly improving.** Replace the base case's derived
state with a parameter:

```lean
def initialState (layout : Layout) (entry : State) : Nat → State
  | 0 => applyMatrix externalMatrix entry
  | round + 1 => ...unchanged...
```

`canonicalProgram` then instantiates `entry := fun lane => [(inputPort lane, 1)]`
and every existing theorem holds verbatim, since none of them inspects the base
case beyond its being a `State`. The sponge instantiates
`entry := fun lane => [(outPort lane, 1), (chunkColumn lane, 1)]` and absorption
is genuinely free.

**Two axes were conflated in cycle 190** and should be kept apart:

| axis | choice |
|---|---|
| bind between permutations | yes (cycle 191, matches production) |
| absorb symbolically vs materialize | independent, and symbolic is free once generalized |

Binding between permutations does not force materialized absorption. The bound
shape with generalized entry costs `(chunks + 1) · 352` rows with absorption
free; the bound shape without it costs `(chunks + 1) · 352 + 8 · chunks`.

Do this generalization **before** writing `hashPrior`.

---

## POSEIDON2-SPONGE-PREIMAGE-WIDTH

```text
claim:
  Whether the trailing-zero digest collision is reachable in production.
status: CORRECTED 2026-07-26 (cycle 204). The collision is real but affects the
        v1/configurable path only; the selected complete F′ relation is immune.
```

### Correction (cycle 204) — the cycle-200 audit was too narrow

Cycles 200 and 201 claimed "the live call path is `encode_poseidon_trace`". That
was a scoped-search error: it grepped callers of one hash entry point and missed
the complete field-R1CS relation, which uses a different one.

`full_relation.rs:816` — *"Arity and representation are absorbed before values"*:

```rust
pub fn semantic_state_digest_fields(values: &[F]) -> [F; DIGEST_LEN] {
    let mut preimage = pack_bytes_as_fields(SEMANTIC_STATE_FIELDS_TAG);
    preimage.push(F::from_u64(SEMANTIC_STATE_FIELDS_SCHEMA));
    preimage.push(F::from_u64(values.len() as u64));   // length prefix
    preimage.extend_from_slice(values);
    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&preimage)
}
```

So the v2 encoding is `TAG ‖ SCHEMA ‖ len ‖ values`. `[42]` and `[42, 0]` produce
different preimages and cannot collide. It also validates state arities
(`full_relation.rs:829` onward).

**Revised status of the collision:**

| path | encoding | collision |
|---|---|---|
| v1 configurable (`recursive_plan.rs:158` → `poseidon_trace.rs:111`) | `TAG ‖ app_state` | **reachable** |
| v2 complete F′ relation (`full_relation.rs:816`) | `TAG ‖ SCHEMA ‖ len ‖ values` | **immune** |

The Lean collision remains kernel-checked and correct about the *generic*
sponge. What changes is its production reach: it is a **legacy/configurable
interface defect**, not a defect in the selected shipping path.

**Reported but not verified here:** a red-team regression at
`redteam_f_prime_encoding_gap.rs:133` is said to construct `[42] → [42, 0]`
under the configurable compiler, and to currently fail to compile because an
unrelated `StepProof` initializer lacks a `nebula_open` field. Neither the test's
content nor the compile failure was checked in this cycle; both are recorded as
claims to verify, not findings.

### What this changes for Phase 3

`hashPrior` and `hashNext` must be audited against **their own** Construction-2
preimages. The semantic-state builder is not a substitute — it was used here as
a proxy for "what gets hashed" and that proxy has now been shown to be the wrong
one for the selected path. The right requirement set is:

- typed, rate-bounded chunking (done, cycle 199/204);
- injective preimage serialization *including length*, as v2 already does;
- Poseidon collision resistance retained as a named cryptographic event.

"Padding proves length separation" was never achievable and is withdrawn as a
requirement.

`absorbChunk_trailing_zero` (cycle 197) proves `[v]` and `[v, 0]` reach the same
digest. Whether that is a production defect depends on the preimages actually
hashed.

**What the audit found.** The live call path is
`encode_poseidon_trace(preimage: &[F])` in
`paper/f_prime/poseidon_trace.rs:111`, and every caller routes through a
structured builder. The main one is

```rust
pub fn build_semantic_state_preimage_fields(app_state: &[F]) -> Vec<F> {
    let mut p = pack_bytes_as_fields(SEMANTIC_STATE_TAG);
    p.extend_from_slice(app_state);
    p
}
```

So the preimage is `TAG || app_state`: a fixed domain tag, then the application
state. Two consequences:

- the tag separates *different hash uses*, not *different lengths within a use*,
  so it does not close the collision;
- the preimage length varies with `app_state.len()`, so the collision is
  reachable exactly when an application's state encoding admits two values
  differing only by trailing zeros.

`PoseidonTraceLayout::from_preimage_len` derives the circuit layout from the
length, so the two preimages produce *different circuit shapes* — they cannot be
substituted within one circuit instance. That limits the attack surface but does
not remove it: the digests are equal, and a digest carried across a trust
boundary is compared as a value, not as a circuit.

**Not settled, and deliberately not overstated.** This is a conditional finding:
*if* any application has variable-arity state where a trailing zero is
meaningful, digests collide. Establishing that requires enumerating the
application state encodings actually in use, which this audit did not do. It is
therefore recorded as a conditional, not reported under §7 as a production
defect — §7 requires that shipping Rust be *found* not to enforce a frozen
relation, and that has not been shown.

### Arity enumeration (cycle 201)

Three call-site shapes reach the builder:

| site | arity |
|---|---|
| `state_lanes56_fields` (metal-bench, cuda parity) | **fixed** — asserts `state.len() == 32` |
| `neo-wasm::semantic_state_digest` | fixed *per relation*, from `build_wasm_relation_layout()` |
| `semantic_state_digest_for_fields` / `..._for_assignment` | caller's `indices.len()` |

So **within a single relation the arity is fixed and the collision is
unreachable** — the two preimages would need different lengths, hence different
`PoseidonTraceLayout`s, hence different circuits.

**Across relations the arity varies, and the digest does not encode it.** Two
relations whose states are `s` and `s ++ [0]` produce equal semantic-state
digests under the same `SEMANTIC_STATE_TAG`, because the tag separates hash
*uses*, not arities.

### The one remaining question

Is a semantic-state digest ever accepted without also binding the relation or
layout it came from? If the relation identity is committed alongside the digest
at every trust boundary, the cross-relation collision is unreachable and this
property closes. If a digest is ever compared or substituted on its own, it is a
reportable finding under §7.

That is a question about the verifier's binding surface, not about the sponge,
and it is where this audit stops.

---

## POSEIDON2-RUST-TARGET-BOUNDARY

```text
claim:
  Which Rust implementation the canonical recipes correspond to.
status: STATED 2026-07-26 (cycle 202), EXTENDED to the sponge entrypoint in
        cycle 205. Both the shared-matrix and shared-sponge assumptions are
        verified rather than assumed.
```

### Sponge entrypoint (cycle 205)

Three hash entry points exist. `neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash`
(`neo-ccs`, a different crate) is what the complete F′ relation uses, and its
sponge is **identical** to `ccs_native::build_bit_backed_poseidon2_hash_values`:

```rust
state = 0
for chunk in input.chunks(RATE) { state[i] += chunk[i]; state = permute(state) }
state[0] += ONE; state = permute(state)
out = state[..DIGEST_LEN]
```

So `Poseidon2Sponge` is a correct model of **both**, and the v1/v2 difference is
**not the sponge** — it is the preimage construction feeding it. That is why the
length prefix fixes the collision without touching the sponge at all.

**Selected entrypoint for Phase 3:** the complete F′ relation's
`semantic_state_digest_fields` → `poseidon2_hash`, with the v2 preimage
discipline `TAG ‖ SCHEMA ‖ len ‖ values`. `hashPrior` and `hashNext` target that
sponge, which `Poseidon2Sponge` already specifies, under a preimage serialization
that is injective by construction.

**Noted, not acted on:** the same sponge is implemented twice, in `neo-ccs` and
in `neo-fold-clean`. Any change to one must track the other. That is a
maintenance observation about Rust, not a formal obligation here.

Two Poseidon2 implementations exist and the work has drawn from both without
ever saying so:

| path | shape |
|---|---|
| `r1cs_circuit/poseidon2.rs` | field-valued R1CS, state carried as `[Lc; WIDTH]` |
| `ccs_native/poseidon2.rs` | bit-backed low-norm CCS, `b = 2`, bit-decomposed witness |

**The canonical encoding targets the field-valued R1CS path.** `Row` is
`(a, b, c)` sparse combinations and `RowHolds` is `A·z * B·z = C·z` over
Goldilocks — that is `r1cs_circuit`'s model, not a bit-backed CCS one. A
352-row field-valued recipe cannot replace a bit-backed `b = 2` source trace,
and nothing here should be read as doing so.

**The shared-matrix assumption, now checked.** The round structure and both
matrices were transcribed from `ccs_native`'s `value_*` functions while the
S-box gate chain came from `r1cs_circuit`'s `enforce_sbox_x7`. That is only
sound if both compute the same permutation. Verified: `r1cs_circuit` carries
`MATRIX_DIAG_8_GOLDILOCKS = [-2, 1, 2, 1/2, 3, -1/2, -3, -4]`, `apply_mat4`,
`external_linear_layer` and `internal_linear_layer` — the same matrices, in the
same arrangement. The transcription is valid for both.

**Why `r1cs_circuit` is the right comparison target.** It already carries state
as linear combinations rather than materializing every round, so it is the same
normal form the canonical encoding uses; its extra rows come from explicit
`materialize_state` calls. That is what makes 608-versus-352 a comparison of
materialization policy rather than of two unrelated encodings.

**How `ccs_native` remains related.** It computes the same arithmetic
permutation, which is why its value functions are a legitimate source for the
matrices and round order. It is not what the recipes replace, and any claim
about the bit-backed path's cost or witness would need its own derivation.

---

## POSEIDON2-ENCODING-CLASS-AND-ORDER

```text
claim:
  The bound sponge shape is the selected comparison encoding, not a canonical
  minimum.
status: RELABELLED 2026-07-26 (cycle 202), per the cycle-197 finding.
```

Cycle 191 chose the bound shape citing production resemblance, coefficient cost
and reuse. Under the project's own stated optimization order — recurring rows
first, then committed, public and auxiliary columns — the **unbound** shape wins
on rows: `(chunks + 1) · 344 + 8` against `(chunks + 1) · 352`.

So the bound choice is **not** the canonical minimum under the project's own
criterion. It stands as the *selected comparison encoding*, chosen to match
production's materialization policy so that a cost comparison is meaningful.

Calling it canonical would require a finite encoding class, an explicit
optimization order, and a selection theorem over that class — the shape
`CIR-POSEIDON2-SPONGE23-RECIPE` uses for its 128-member rewrite class. None of
those exists here, and none is claimed.

---

## POSEIDON2-ROUND-CONSTANT-CONFORMANCE

```text
claim:
  Pin the 86 concrete round constants so the exact nonzero coefficient count
  and any Rust-conformance claim become reachable.
status: CLOSED 2026-07-26 (cycle 230), with the authority direction inverted.
        Lean owns the table in `Poseidon2CanonicalConstants`; Rust is checked
        against it in `Poseidon2RustConformance`. Supersedes the BLOCKED status
        recorded at cycle 203 and the "read them from Rust" decision of cycle
        191.
```

### How it closed, and why the closure is not the one cycle 191 planned

Cycle 191 decided to pin the constants by reading them from Rust, which §4.1 of
the governing prompt permitted. The user has since stated the governing intent
directly: *define the constraints from Lean; do not import from Rust and
validate them; Rust is not correct.* That supersedes §4.1 for constants. The
route cycle 191 chose would have satisfied the letter of §4.1 and defeated the
goal.

The values were exported by the Rust generator and are now **inlined as Lean
data**. What changed is not the numbers but which file decides them:

| | before | after |
|---|---|---|
| Table lives in | generated artifact | `Poseidon2CanonicalConstants` |
| Headline path imports the artifact | yes, transitively | **no** |
| Generator changes | canonical theorems silently follow | `Poseidon2RustConformance` fails |
| A failure means | Lean re-derives | **Rust is wrong** |

The inversion is mechanically checkable rather than declared: exactly three
modules reach the generated artifact — the artifact, its facade, and the
conformance checker. No module in the headline path does.

### What this does not establish

It does not make Lean authoritative over the *derivation* of the table. Lean now
fixes which table F′ uses; it does not prove that table is the ChaCha8 seed's
image. See `POSEIDON2-CONSTANT-DERIVATION`.

It also does not establish that Rust's permutation *uses* the table it exports;
that remains `POSEIDON2-RUST-TARGET-BOUNDARY`.

### The original blockage, retained for the record

**The values are not reachable.** In
`crates/neo-fold-clean/src/engine/ccs_native/poseidon2.rs`:

```rust
struct Poseidon2Constants { ... }          // private
fn poseidon2_constants() -> ...            // private
fn sample_poseidon2_constants() -> ...     // private
```

Only `POSEIDON2_HALF_FULL_ROUNDS` and `POSEIDON2_PARTIAL_ROUNDS` are `pub`. The
constants are generated by ChaCha8 from `neo_params::poseidon2_goldilocks::SEED`
at first use and never exposed.

Two routes, both closed:

1. **Expose or print them from Rust** — add `pub`, or add a test that prints the
   table. This is modifying Rust, which §7 makes a stop-and-report rather than
   something to do unilaterally.
2. **Generate them in Lean** — implement ChaCha8 and the Goldilocks sampler,
   then prove the Lean stream matches Rust's. That reduces "pin 86 constants" to
   "formalize a stream cipher and prove it conformant", which is a strictly
   larger obligation than the one it discharges. Blocked under §5.

**What this blocks and what it does not.** Blocked: the constant-wire and
partial-block thirds of `POSEIDON2-NO-CANCELLATION`, hence Phase 2's exact
nonzero coefficient count, hence any bit-for-bit Rust conformance. Not blocked:
everything else. Every theorem in the project is universally quantified over
`Constants` and none weakens. Phase 3's sponge rows, Phase 4's recipes and the
whole assembly are independent of this.

**The decision needed.** Someone with authority to change Rust adds one of:

```rust
pub fn poseidon2_round_constants() -> Poseidon2Constants   // accessor
```

or a printing test whose output can be transcribed. Either is a small,
read-only change; neither alters behaviour. Until then the exact nonzero count
is unreachable and the derived upper bound of 4397 stands as the strongest
available statement.

**Resolved.** An accessor and a byte-for-byte drift test now exist
(`cargo test -p neo-ccs --release --test poseidon2_round_constants`), so route 1
was taken. The 4397 upper bound has been superseded by the exact count: 31,139
nonzero coefficients over the seven-call fixed-23 sponge
(`Poseidon2ExactCoefficients.program_nonzero_coefficient_count`).

---

## POSEIDON2-CONSTANT-DERIVATION

```text
claim:
  The selected 86-constant table is the image of the published ChaCha8 seed
  under the Goldilocks sampler.
status: OPEN 2026-07-26 (cycle 230). Deliberately not attempted.
```

`Poseidon2CanonicalConstants` makes Lean authoritative over **which** table the
protocol uses. It says nothing about where those values came from.

Closing this would mean formalizing ChaCha8 and the rejection sampler in Lean
and proving the stream reproduces the table. That is the route cycle 203 marked
blocked under §5, and the reasoning still holds: it reduces "pin 86 constants"
to "formalize a stream cipher", a strictly larger obligation than the one it
discharges.

**Why leaving it open costs nothing the goal needs.** The encoding needs the
table *pinned*, not *derived* — every canonical theorem is quantified over
`Constants`, and the exact coefficient count needs only that the selected
constants are nonzero, which is 86 scalar checks over Lean-owned data. A seed
derivation would replace one selection (these 86 values) with another (this
seed, this sampler). It would move the trust boundary, not remove it.

What it would buy is auditability against the published parameter set, which is
a real but separate property. It is not on the Phase 2–5 critical path.

---

## POSEIDON2-PHASE3-TARGET-REDEFINITION

```text
claim:
  What `hashPrior` and `hashNext` actually are.
status: FOUND 2026-07-26 (cycle 206). Phase 3's target is narrower and more
        concrete than the generic sponge modelled so far.
```

`hashPrior` and `hashNext` have **zero occurrences in Rust**. They are existing
Lean-side `CallRecipe` slots, and `fpr-production-program-instantiation.md`
already documents them:

- typed semantics `FixedOneLoweringAdapter.CallAlignment.hashPrior`;
- semantic operation: totalized `paperHash` at iteration `i`, with presence of
  `current` and exact iteration, initial-state, running and program-counter
  alignment;
- current Rust rows: `fprime.recursive.step.prior_link.digest`;
- existing Lean slice: `FPrimeFullHistoryXOutSpongeReceipts.priorReceipt`, which
  owns **only the nonoptional 23-field Poseidon2 sponge core**;
- verdict already recorded there: **no existing complete physical owner**.

### Three consequences

**1. The input is fixed-width at 23 fields, not a variable list.** That is the
same 23-field sponge as `CIR-POSEIDON2-SPONGE23-RECIPE`. `23 = 5·4 + 3`, so six
absorb chunks plus the padding block — **seven permutation calls**, matching that
recipe's "seven 600-row permutation programs".

**2. Two open properties close immediately.** `POSEIDON2-SPONGE-CHUNKING` is
determined by the fixed arity, and `POSEIDON2-SPONGE-PREIMAGE-WIDTH` is
unreachable at fixed width — the trailing-zero collision needs two lengths.

**3. `Poseidon2Sponge` models the wrong granularity.** It specifies a generic
sponge over `List RateChunk`. The recipe needs a fixed 23-field input with a
specific call-frame serialization. The generic module is not wrong, but it is not
the target; it should become the substrate a 23-field instantiation sits on.

### Revised Phase 3 estimate

With the bound shape, `7 × 352 = 2,464` rows against production's `4,229` for
the same recipe. That comparison is now available in principle because both sides
are the same fixed 23-field call — unlike the permutation comparison, this one
does not depend on any materialization-policy caveat.

**Not claimed:** that 2,464 is derived. It is `7 ×` a proved figure, but the
seven-call decomposition and the absorption rows have not been constructed.

### What the next session should do first

Read `FPrimeFullHistoryXOutSpongeReceipts` and
`ProductionXOutSponge23InputAlignment` before writing anything. Substantial Lean
work already exists on this exact recipe, and the missing pieces are enumerated
in `fpr-production-program-instantiation.md`: the optional presence coordinate,
failed-preimage alignment, complete call-frame serialization, and one reusable
receipt for both Step and Terminal occurrences.

---

## POSEIDON2-PHASE3-INDEPENDENCE

```text
claim:
  The canonical 23-field recipe must not build on the existing XOut receipts.
status: CONSTRAINT IDENTIFIED 2026-07-26 (cycle 207). Load-bearing.
```

Cycle 206 recommended reading `FPrimeFullHistoryXOutSpongeReceipts` before
writing Phase 3. Doing so surfaces a constraint that would have been easy to
violate:

```lean
theorem priorReceipt :
    EmissionReceipt priorTrace FPrimeFullHistoryPriorLink.rows
      inputFields 218 868073 := by
  constructor <;> native_decide
```

The existing receipts are **`artifact-checked`**: proved by `native_decide`,
hence depending on `Lean.trustCompiler`, and pinning *measured* row slices and
column intervals — `recursiveOutputReceipt` owns `[11, 4236)`.

The canonical work is **`model-proved`**: every figure derived from an emitted
row program, no `native_decide` anywhere, `[propext, Quot.sound]` throughout.

**These must not be composed.** Building `hashPrior` on `priorReceipt` would
import `Lean.trustCompiler` and artifact-measured row counts into a derivation
whose entire purpose is to avoid both. The permutation's `4397` is worth having
precisely because nothing in its dependency graph was measured; a sponge figure
resting on `priorReceipt` would have no such standing.

**The correct relationship** is the same one the permutation has with
`CIR-POSEIDON2`: the existing receipts are the **comparison target**, not the
foundation. Canonical `hashPrior` derives its seven permutation calls and its
absorption rows independently, and the resulting figure is then compared against
the artifact-checked `~4,229`.

**Practical check for the next session:** after building any Phase 3 theorem,
run raw `#print axioms` on it. If `Lean.trustCompiler` appears, an artifact-
checked result has been composed in and the derivation is no longer independent.
The `#audit_axioms` guard normalizes `native_decide` axioms, so the raw form is
the one that catches this.

---

## POSEIDON2-SPONGE-SOUNDNESS-SHAPE

```text
claim:
  What generalizing the round induction to a carried entry actually requires.
status: ANALYSED 2026-07-26 (cycle 212). Not a mechanical lift.
```

Sponge soundness composes `canonicalProgram_computes_reference` per call. That
theorem is stated over `scheduleOf` (port entry), so it needs a `From` version.
Cycle 196 showed the partial and terminal *states* are entry-independent, which
suggested only the initial phase would need work. That is true of the **terms**
and false of the **values**.

**What generalizes for free.** `initialStateFrom_halfFull_eq` proves
`initialStateFrom layout entry halfFullRounds = initialState layout halfFullRounds`
as *terms*. So `partialState` and `terminalState` are literally the same
combinations whatever the entry, and their column identities are unchanged.

**What does not.** `partialState_eval` concludes
`lcEval z (partialState …) = refPartial constants (inputValues layout z) …`.
The reference side is `refPartial` at the **port** values. For a sponge the
reference is `refPartial` at the **entry** values, and those differ — because
`refPartial constants v 0 = refInitial constants v halfFullRounds`, which
depends on `v`. The encoding's terms are entry-independent; the values they
denote under a satisfying assignment are not, because satisfaction chains back
through the S-box outputs to the entry.

**So the required generalization is the whole chain, not one phase:**

- `SboxChainFrom`, since `SboxChain` is stated over `scheduleOf`;
- `initialState_eval` taking `entry : State` and `entryValues : Values` with
  `∀ s, lcEval z (entry s) = entryValues s` — the base case becomes that
  hypothesis directly, and the successor case is unchanged;
- `partialState_eval` and `terminalState_eval` restated at `entryValues`, their
  proofs unchanged since only the reference index moves;
- `canonicalProgramFrom_computes_reference` as the end-to-end statement.

Then the sponge composes: call `k`'s output ports give `entryValues` for call
`k + 1` through `entryOf`, and the chain telescopes to `spongeFinal`.

**Estimate.** Four theorems restated, three proofs unchanged in body. Mechanical
but not free, and it must be done before any sponge soundness claim — the
tempting shortcut of reusing `partialState_eval` at the port values would prove
the sponge computes the wrong function.

---

## POSEIDON2-PHASE3-CORE-AND-CALL-BOUNDARY

```text
claim:
  Exact status after constructing the selected fixed-23 arithmetic core.
status:
  CORE MODEL-PROVED 2026-07-26; COMPLETE HASH CALLS OPEN AT AN EXACT
  SEMANTIC/PROFILE BOUNDARY.
```

The selected fixed-23 core is now constructive and kernel-checked:

- `Poseidon2Sponge23.program` emits exactly 2,464 normalized permutation rows;
- `Poseidon2Sponge23.program_computes_digest` proves soundness;
- `Poseidon2Sponge23Honest.honest_satisfies` constructs an honest assignment;
- `Poseidon2Sponge23Ownership.program_conservation` owns the complete numeric
  column space;
- `Poseidon2ExactCoefficients.program_nonzero_coefficient_count` proves the
  exact core coefficient count is 31,139;
- `CanonicalPoseidon2Sponge23Recipe` adds four activation-gated auxiliary
  outputs, giving an exact standalone receipt and cost
  `(2468, 0, 0, 2468)` with 31,151 nonzero A/B/C terms.

The Rust-exported round-constant artifact is now selected by
`Poseidon2ProductionConstants.selected`. The focused Rust drift test proves the
artifact still equals `neo_ccs::crypto::poseidon2_goldilocks::round_constants`
and reconstructs the same permutation. Constant generation/import remains the
explicit temporary TCB; no claim derives the table from the seed in Lean. This
supersedes cycle 203's source-access obstruction and cycle 222's stale
2,408-auxiliary count.

This does **not** complete `hashPrior` or `hashNext`. Their frozen result type is
an optional digest encoded in five coordinates. On an absent current state or
failed duplicated-carrier alignment, `paperHash` must return `none`; on an
aligned present state it returns presence one plus the four-lane XOut result.
The new arithmetic core is deliberately always present and exposes four
auxiliary lanes. The exact theorem
`ProductionHashCallBoundary.no_nonoptionalCoreRefines` proves that no such core,
under any preimage serializer, can implement the total frozen call by itself.

Two production selections also remain absent:

1. `FixedOneCanonicalAdapter.Parameters.hash` is still an abstract
   `XOut.Semantics`; no concrete theorem identifies its state-output message
   with the selected 23-field Poseidon2 calculation.
2. The repository has no complete production `DataCodecs`/`Profile` instance.
   State, witness, running, fresh, NIFS-proof, and terminal-witness codecs are
   not selected, so the complete call frame and its alignment rows cannot be
   defined without inventing a production profile.

Therefore the exact assurance claim is **canonical arithmetic core
model-proved; complete hash-call recipes open**. Neither call may be entered in
`DirectCalls.RemainingRecipes`, removed from `remainingCalls`, or counted as a
complete F-prime call until the optional/alignment wrapper, concrete XOut
selection, codecs, ownership, soundness, and honest completeness all exist.

## POSEIDON2-PHASE4-TERMINAL-SELECTION-BOUNDARY

```text
claim:
  Whether canonical runningCheck and freshCheck recipes can currently be
  instantiated.
status:
  PRECISELY BLOCKED AT MODEL SELECTION.
```

The frozen terminal verifier requires explicit independent `TerminalRelations`
and exact Boolean `RelationChecks`. The production lowering
`FixedOneLoweringAdapter.Configuration` still accepts those relations, checker
implementations, and both witness types as parameters; no fixed-one
ConcretePhi81 production value selects them.

`FPrimeProductionOwnerProgramBoundary.TerminalSelectionOpacity.
terminalFacts_do_not_select_relationChecks` is the kernel-checked boundary:
the same physical `TerminalFacts` is compatible with one exact checker that
accepts and another exact checker that rejects. Consequently the current
terminal facts cannot determine either required unary relation. This does not
prove production unsound; it proves that a `runningCheck` or `freshCheck`
recipe constructed now would have to invent the semantic relation it claims
to encode.

Phase 4 can resume only after the shipping application selects the concrete
running/fresh relations, checkers, witness types, and canonical codecs. The
frozen terminal relation must not be weakened and the final NIFS fold must not
be relabelled as either independent unary check.

## POSEIDON2-PRODUCTION-APPLICATION-SELECTION-BOUNDARY

```text
claim:
  Whether fixed-one, plain, and the 270-coordinate recursive carrier determine
  one closed production profile for the remaining Phase 3/4 recipes.
status:
  SOURCE-AUDITED EXTERNAL SELECTION REQUIRED.
```

They do not. The public Rust constructor
`R1csIvcPreprocessing::new(params, app, plan)` accepts an arbitrary
`R1csShape`, and `R1csIvcRelation::compile_fixed_point` compiles that supplied
application into the recursive relation. Consequently fixed-one/plain/270
selects the recursive carrier shape but does not select the application
relation itself.

The repository contains materially different callers. The bounded terminal
diagnostic uses `one_product_r1cs`; the SHA-256 system path supplies a
Bellpepper-derived sparse R1CS. Neither may be promoted to “the shipping
application” merely because it is executable. In particular, the
one-product diagnostic remains diagnostic-only.

The Lean boundary matches the Rust API. `Vocabulary.Parameters` still owns
arbitrary `State`, `Witness`, `Running`, `Fresh`, `NifsProof`,
`RunningWitness`, and `FreshWitness` types together with the application
machine and the two terminal relations/checkers. `Encoding.DataCodecs`
requires a codec for each of those selected types. No concrete production
value closes those fields.

This blocks both remaining phases:

- a total `hashPrior`/`hashNext` recipe needs the selected state/running
  representations, complete optional/alignment frame, and a concrete
  `XOut.Semantics` theorem identifying the exact 23-field state-output
  preimage;
- `runningCheck` and `freshCheck` need the selected application relations,
  witness types, executable checkers, and canonical codecs.

The exact next input is one named deployment application: its authoritative
`R1csShape` and lifecycle plan, the corresponding Lean semantic carrier and
terminal relations/checkers, and the canonical serialization of every
application-owned value. Alternatively, supporting arbitrary applications
requires an application compiler interface that supplies those certified
objects; the existing parametric lowering already states that boundary but
does not provide a closed program or a unique concrete cost.

No recipe, row count, or production theorem may choose one of the test
fixtures implicitly. This is a missing application selection, not permission
to weaken the frozen relation or to insert opaque call outcomes.

## POSEIDON2-PHASE3-4-APPLICATION-CERTIFICATION

```text
claim:
  Whether the application-compiler alternative can certify the four
  application-dependent Phase 3/4 calls without selecting a test fixture.
status:
  MODEL-PROVED, PROFILE-INDEXED.
```

`Poseidon23ApplicationProfile` is now the finite certification boundary for
an application supplied to HyperNova Construction 2. It owns the complete
codecs, a fixed 23-coordinate projection, prior/next hash serialization
equations, the rejecting alignment condition, and separate exact running and
fresh checker equations. These are serialization and checker-refinement
facts; no accepted result or semantic conclusion is stored in the profile.

From such a profile, `ApplicationCertification.poseidon23` constructs four
complete `CallRecipe`s:

- `Poseidon23HashPriorRecipe.recipe`, whose digest output is auxiliary;
- `Poseidon23HashNextRecipe.recipe`, whose digest output is public;
- `RunningCheckRecipe.recipe`;
- `FreshCheckRecipe.recipe`.

Each recipe carries the existing artifact-independent row-count, ownership,
support, active-soundness, honest-active-completeness,
inactive-satisfiability, and emission-receipt contract. The terminal recipes
are separate equality checks over their respective authoritative codecs; a
NIFS result or one terminal check cannot substitute for the other. Successful
decoding implies exact codec width.

The typed call list invokes all four exactly once.
`ApplicationCertification.phase34Cost_exact` derives the resulting rows and
committed/public/auxiliary column counts from
`alignmentWidth`, `running.width`, and `fresh.width`. During validation, the
first proposed hash auxiliary formula was rejected: the footprint owns two,
not three, alignment-width temporary bundles. The final guarded formula is
therefore `2 * alignmentWidth + alignmentWidth.pred + constant`.

This closes the application-compiler alternative only. It does not contradict
the source audit above: the repository still does not select a unique
deployment application, and therefore does not supply a single numeric
deployment cost or a Rust equality theorem. It also does not construct
`step`, `nifsVerify`, `DirectCalls.allRecipes`, a complete F-prime program, or
generated rows. Cycles 226 and 227 remain valid for claims about one selected
shipping profile; they are superseded only as claims that no honest
profile-indexed certification interface can be built.

## POSEIDON2-PHASE5-COMPLETE-APPLICATION-CERTIFICATION

```text
claim:
  The profile-indexed assembly boundary constructs complete fixed-one Step
  and Terminal programs after a deployment supplies certified `step` and
  `nifsVerify` recipes.
status:
  CONDITIONAL INTERFACE MODEL-PROVED; NO DEPLOYMENT INHABITANT.
```

`Phase5CallCertification` does not construct the two remaining setup-selected
physical programs. It is a premise containing a complete `CallRecipe` for the
HyperNova application `step` and another for the selected `nifsVerify`. The
conditional boundary is sound: once those proof-carrying programs exist, it
receives executable rows plus exact footprints, receipts, active soundness,
honest active completeness, and inactive satisfiability—not an accepted
Boolean, semantic conclusion, opaque call outcome, or generated artifact.

`CompleteApplicationCertification.allRecipes` combines those programs with
the four Phase 3/4 recipes and five direct recipes only for an inhabitant of
that premise. No deployment inhabitant exists in the repository. Its
eleven-entry recipe-family domain is duplicate-free, and runtime
multiplicities are not inferred from that domain:
`stepProgramCalls_exact` and `terminalProgramCalls_exact` traverse the actual
typed ASTs and expose their distinct nine-call and eight-call sequences. The
conditional Step and Terminal programs then prove:

- physical acceptance iff the unchanged frozen checker accepts, for values in
  the selected codecs' honest domain;
- honest accepted values construct satisfying assignments;
- every row and allocated column has one source-aligned receipt owner;
- no rows or columns exist outside the receipt lists;
- exact cost is the receipt fold;
- the selected Step and Terminal costs attain the existing finite
  rewrite-class minima.

The public PiRLC projection occurrence is now constructive and event-aware:
`KPiRlcTrace.occurrence_rows_length` derives the exact subtotal
`(23 + 2 * matrixCount) * (321 * arity + 482)`, while
`KTraceBadRootFixture.not_eventFreeOccurrenceSoundness` proves that its
occurrence-bound bad-root branch cannot be erased. This is only one component
of `nifsVerify`; it omits PiCCS, PiDEC, transcript, point-binding, accumulator,
residual, and call-framing rows.

Two kernel boundaries prevent promotion to a complete Phase-5 deployment:

- `NifsCompletionBoundary.publicOccurrence_does_not_determine_completeNifs`
  proves the public occurrence alone cannot determine an arbitrary
  setup-selected HyperNova verifier result.
- `DeploymentSelectionBoundary.fixedProfile_does_not_select_step_or_nifs_cost`
  proves the fixed call footprint leaves the setup-owned `step` and
  `nifsVerify` costs independent.

Consequently no concrete `Phase5CallCertification`, complete deployment
`allRecipes`, unique numeric deployment cost, or Rust replacement exists.
The current `CallRecipe` contract is also exact-only, so event propagation
through calls and branches must wait until the complete NIFS occurrence exists
and can own that event. The conditional assembly theorem remains valid; it
does not close canonical Phase 5 for a selected deployment.

---

## KMUL-PRODUCT-COUNT

```text
claim:
  One K multiplication costs four emitted rows under schoolbook expansion.
status: DECIDED 2026-07-26 (cycle 239). Karatsuba, three rows. Implemented,
        not merely selected.
```

**Decision: Karatsuba.** It strictly dominates, so no optimization order needed
to be consulted:

| | rows | aux columns | operand entries |
|---|---|---|---|
| schoolbook | 4 | 4 | `4|cL| + 4|cR| + 4` |
| Karatsuba | **3** | **3** | `4|cL| + 4|cR| + 3` |

The naive objection is that Karatsuba's third row takes *summed* operands and so
carries `2|cL|` and `2|cR|` entries instead of `|cL|` and `|cR|`. Working the
totals out, that costs one entry less overall, not more, and `outHigh` spends
that one back on its extra term. Equal coefficients, one fewer row and one fewer
column.

`outHigh` recovers `l0r1 + l1r0` by subtraction. In a `Nat` encoding that is the
coefficient `goldilocksP - 1`, which stays linear and still emits no row.
`karatsuba_identity` discharges it without any `Nat` subtraction at all: the
`goldilocksP - 1` coefficients become exact multiples of the prime that the
final reduction discards.

Settled now rather than later because the three-product form changes which
intermediate values exist; selecting it after identity frames are laid out would
invalidate them.

`KMul.rows_length` derives four from the emitted list: the schoolbook products
`l0r0`, `l1r1`, `l0r1`, `l1r0`. Both output coordinates are linear in those and
emit nothing, so the whole nonlinear cost of a `K` multiplication is four rows
and four auxiliary columns.

Karatsuba computes the same product with **three**: `l0r0`, `l1r1`, and
`(l0+l1)(r0+r1)`, recovering `l0r1 + l1r0` by subtraction. That is a 25% row
saving on every projection multiplication, and the projection check is expected
to dominate the NIFS row program.

It is not taken here because the three-product form changes which intermediate
values exist, so it must be selected once for the whole projection encoding
rather than per gadget — the same discipline that made the Poseidon2 S-box chain
a single global choice (`x²/x⁴/x⁶/x⁷` at three operand positions rather than
`x²/x³/x⁶/x⁷` at four). Selecting it after identities are written would
invalidate their frame layouts.

Deciding this is a prerequisite for any NIFS row count, not a later
optimization.

---

## POSEIDON2-EXEC-SPEC-SHAPE

```text
claim:
  `permutationProgram_exec_iff_spec` does not exist, and its literal form
  would be false. The correspondence holds as a pair of directions.
status: RECORDED 2026-07-27 (cycle 312). Both directions already proved.
```

### The name does not exist

`permutationProgram_exec_iff_spec` appears exactly once in the tree, in a
`Poseidon2Program` docstring, as a forward reference to
`POSEIDON2-ROUND-INDUCTION`. There is no theorem of that name, and grepping
`specs/` for that identifier returns nothing either.

### Its literal form would be false

A biconditional `Satisfies program z ↔ outputs z = reference` over a single `z`
is **not** true of this program, and should not be. An assignment can carry the
correct output ports while carrying wrong intermediate S-box columns; it hits
the right spec value and fails the row program. Soundness is the direction that
quantifies over all `z`; completeness is an existence statement about the
honest assignment. They are not two halves of one iff.

### Both directions are proved

| direction | theorem |
|---|---|
| soundness — any satisfying `z` computes the reference | `Poseidon2RoundInduction.canonicalProgram_computes_reference` (its own docstring says "Soundness direction only") |
| completeness — a reference execution yields a satisfying assignment | `Poseidon2Honest.honest_satisfies` |
| the two agree | `Poseidon2Honest.honest_directions_agree` |

So the substance the DONE condition asks for is present in the only shape it can
honestly take. What is absent is a theorem with that name, and adding one under
that name would misdescribe what is proved.

### Superseded 2026-07-27 (cycle 328)

The finding above was right about the *literal* form and wrong to stop there.
There is a biconditional that is both true and stronger, and it is now proved:
`Poseidon2Uniqueness.canonicalProgram_exec_iff_spec`. See
`POSEIDON2-WITNESS-UNIQUENESS` below. What this section correctly rejected was
an iff whose right-hand side is *output equality*; what it missed is that the
right-hand side should be *agreement with the honest execution on every
column*, which soundness does establish.

### How this was found

By checking the name against code rather than against a carried summary. Several
reports in this session asserted Phase 2 complete "including
`permutationProgram_exec_iff_spec`" on the strength of a pre-compaction summary.
That assertion was inherited, not verified, and was wrong about the name.


---

## POSEIDON2-WITNESS-UNIQUENESS

```text
claim:
  The canonical program's satisfying assignments are exactly its honest
  executions, on every column of the declared 361-column space.
status: PROVED 2026-07-27 (cycle 328).
  Poseidon2Uniqueness.canonicalProgram_exec_iff_spec.
evidence: model-proved
```

### The statement

```text
Satisfies (canonicalProgram canonicalLayout constants) z
  <-> forall column < canonicalColumnTotal,
        z column = honestAssignment constants (inputValues canonicalLayout z) column
```

under `z 0 = 1` and canonical residues. `canonicalColumnTotal` is 361: constant
wire, eight inputs, eight outputs, 344 S-box auxiliaries.

Forward is **witness uniqueness**. Backward is honest completeness transported
along assignment congruence. The input ports are shared reads rather than
allocations, so their arm is definitional — `inputValues` *is* the assignment
restricted to them. Uniqueness is therefore a statement about the 352 allocated
columns, given the inputs.

### Why this is the honest form and output-equality is not

An assignment can carry the correct S-box **output** while carrying a wrong
**intermediate**. Demonstrated, not argued: on a one-S-box fixture that is
otherwise accepted, mutating only the square intermediate is rejected while the
output column is provably unchanged; same for the sixth intermediate. An
`outputs = reference` biconditional would have accepted both. It would say the
encoding accepts assignments it must reject.

### What made uniqueness available

`Poseidon2Schedule.canonicalProgram_sbox_chains` forces **all four** chain
columns per S-box — square, fourth, sixth, output — not merely the output. Every
allocated column is one of those four or an output port, so pinning them pins
the whole witness. Had the chain lemma forced only the S-box output, uniqueness
would be false and the two separate directions would have been the ceiling.

### The one piece that had to be built

Soundness needs
`lcEval z (scheduleOf i) = sboxInputValue constants (inputValues z) i` for an
arbitrary satisfying `z`. `Poseidon2Honest.honest_scheduleOf` proves that for the
honest assignment only. `Poseidon2Uniqueness.scheduleOf_eval` is its mirror
through the round inductions `initialState_eval`, `partialState_eval` and
`terminalState_eval`. Neither existing direction had it.

### Supporting results, also new

| result | content |
|---|---|
| `rowHolds_congr`, `satisfies_congr` | satisfaction depends on an assignment only through the columns the rows mention |
| `canonicalProgram_conservation` | the raw program's whole-program conservation, previously stated only for the normalized program |
| `sboxColumn_forced` | every chain column, not just the output, equals its honest value |

### Scope

This is the width-8 permutation program. It says nothing about the sponge, about
domain separation, or about binding or random-oracle security — see
`4.6 Two hashes, one permutation`. Arithmetic correctness proves neither.

---

## POSEIDON2-HASH-PLACEMENT

```text
claim:
  hashPrior and hashNext are emitted row programs: the fixed-23 sponge placed
  at disjoint column windows.
status: PROVED 2026-07-27 (cycle 329).
  Poseidon2HashRecipe.
evidence: model-proved
```

### What was missing, precisely

`Poseidon2Sponge23` already owns absorption, padding, rate, capacity and the
digest. What it does not own is **placement**: it is written at fixed absolute
columns — seven permutation blocks of stride 361, then 23 preimage columns at
2527. That is sufficient for one hash. The step program contains two, and two
instances of a fixed-column program collide on every column they use.

So the Phase-3 obligation not already discharged by the sponge is exactly:
place two instances so that "distinct owners never collide" is a theorem, and
carry soundness and completeness across the placement.

### The numbers, derived

| quantity | value | source |
|---|---|---|
| sponge column window | 2550 | `inputBase + sponge23Fields`, evaluated |
| columns owned per instance | 2549 | window minus the shared constant wire |
| rows per instance | 2464 | `program_length`, through `List.length_map` |
| `hashPriorBase` | 0 | first placement |
| `hashNextBase` | 2550 | one full window later |

`relocate base` fixes column `0` and shifts everything else, so the constant
wire is a **shared read** and belongs to neither owner's allocation.
`hashPrior_hashNext_disjoint` is then a theorem, not a naming convention.

### Transport

`rowHolds_pull_iff` carries both directions across the placement:

- soundness — `hashProgram_pull` reduces satisfaction of a placed instance to
  satisfaction of the sponge under `pullAssignment z (relocate base)`, so
  `program_computes_digest` applies unchanged;
- completeness — `pull_honestAssignment` shows the placed honest witness pulls
  back to the sponge's own, so `honest_satisfies` transports.

Conservation is stated as what the placement does — every column of a relocated
row is the relocation of an `Allocated` source column — and composes with
`Poseidon2Sponge23Ownership.program_conservation`. The placement introduces
nothing.

---

## POSEIDON2-HASH-NO-DOMAIN-TAG

```text
claim:
  There is no domain separator between hashPrior and hashNext. The separation
  the recipe owns is the sponge's padding rule.
status: RECORDED 2026-07-27 (cycle 329) by reading the call vocabulary.
```

`Vocabulary`'s own docstring states it: `hashPrior` and `hashNext` "have the
same semantics but different ownership: the former is an auxiliary link value
and the latter is the sole public step output." Both invoke
`parameters.machine.hash` on the same record shape, with the same
`callInputs`.

**This encoding does not invent a separator.** Adding one would change a mapped
definition, which is a change-control decision (§16) and not a formalization
one. What distinguishes the two calls is their preimage and their output port,
both of which the caller owns.

### RETRACTED 2026-07-27 (cycle 345)

**A domain separator exists, and this entry was wrong.**

`Poseidon23ApplicationProfile.normalizedIteration`:

```text
normalizedIteration next iteration =
  let coordinate = (boundedNatCodec.encode iteration).getD 0 0
  if next then coordinate + 1 else coordinate
```

with its own docstring: "Prior and next hashes share one physical projection.
The only semantic difference is the **normalized first coordinate**."

The wiring is explicit and was checked, not inferred:

| recipe | call |
|---|---|
| `Poseidon23HashPriorRecipe.lean:135` | `sourceCoordinates profile.codecs **false**` |
| `Poseidon23HashNextRecipe.lean:135` | `sourceCoordinates profile.codecs **true**` |

So the two hashes are separated by a `+1` on the iteration coordinate. Prior and
next digests over otherwise-identical data are **not** equal.

### How the error was made

By reading `Vocabulary`'s docstring — "`hashPrior` and `hashNext` have the same
semantics but different ownership" — and treating it as the whole story. That
sentence is true *of the typed call vocabulary*, where both invoke
`machine.hash`. The separator lives one layer down, in the encoding profile that
builds the source coordinates, and that layer was never read.

The consequence was not cosmetic: this entry was cited three times to claim that
nothing but preimage content distinguishes a prior digest from a next one, and
the sharpened form in `POSEIDON2-HASH-COMMITMENTS` — "what separates a prior
digest from a next digest is entirely the preimage content" — inherits the same
error and is retracted with it.

### What survives

`committed_single_arity` and `digest_independent_of_placement` are unaffected:
the sponge does admit one preimage length, and placement does carry no
separation. What was wrong is the conclusion drawn from them, not the theorems.

---

## POSEIDON2-HASH-PROJECTION-INJECTIVITY

```text
claim:
  The 23-slot preimage is a projection from a wider source, and nothing in the
  profile requires it to be injective.
status: CLOSED 2026-07-27 (cycle 346). The constraint exists, one layer up.
```

### Resolved

`CoordinatePlan` alone is unconstrained — that part of the observation stands.
But the plan is never used alone. `Poseidon23ApplicationProfile` carries it
together with

```text
hashPrior_exact :
  forall iteration z0 current running,
    codecs.digest.encode (parameters.machine.hash { ... }) =
      Poseidon23Hash.resultCoordinates hashPlan
        (Poseidon23Hash.sourceCoordinates codecs false iteration z0 current running)
```

and the matching `hashNext_exact`. The profile's own docstring names them: "The
two semantic equations bind the finite projections to the unchanged frozen
machine hash. They are serialization/refinement facts, not supplied acceptance
conclusions."

Universally quantified over all inputs. So a plan whose projection dropped a
source coordinate that the frozen `machine.hash` reads **cannot inhabit the
profile** — the two sides would disagree on inputs differing only in that
coordinate. The constraint is a refinement obligation on the profile rather than
a condition on the plan's type, which is where a refinement obligation belongs.

### What this does and does not say

It says the encoding faithfully refines `machine.hash`. It does **not** say
`machine.hash` reads every coordinate: if the frozen definition ignores one, the
plan may too, and the profile is still satisfied. Any loss there would be in the
frozen definition, not in the encoding, and would need to be raised as a
change-control question rather than an encoding defect.

### The original text, for the record

`Poseidon23ApplicationProfile.CoordinatePlan` carries

```text
preimage : Fin 23 -> Fin sourceWidth
```

where `sourceWidth = 1 + state.width + state.width + running.width`. So 23 is
**not** the source arity — it is the number of coordinates selected from a wider
source, and `select` simply reads them by index.

No injectivity or surjectivity condition on `preimage` appears in the profile.
A plan that maps two slots to one source coordinate, or that omits a source
coordinate entirely, type-checks. A source coordinate outside the image is not
hashed.

**This is recorded, not alleged.** Whether any deployed `CoordinatePlan` is
deficient has not been checked, and a plan being unconstrained in the type is
not the same as a plan being wrong. What the record establishes is that the
constraint is not in the interface, so it must be checked per deployment — and
that a future reader should not assume the 23 slots cover the source.

### What separation does exist

The sponge's padding rule. `Poseidon2Sponge.pad` absorbs a single `1`, which is
what makes preimages of different lengths distinct;
`absorbChunk_trailing_zero` and `trailing_zero_inputs_differ` are the sponge's
record of that. Length separation is real and proved. Inter-call separation
does not exist, and this is recorded as a named property rather than a comment
so that a reader can tell it was checked rather than overlooked.

---

## POSEIDON2-HASH-COMMITMENTS

```text
claim:
  The hash recipe states and discharges its own rate, capacity, absorption,
  padding and separation commitments.
status: PROVED 2026-07-27 (cycle 332). Poseidon2HashRecipe.
```

Absorption, padding, rate and capacity are `Poseidon2Sponge`'s to *prove*. They
are the recipe's to *commit to*, and the difference is auditability: a reader
checking `hashPrior` should be able to see what it promises without first
reconstructing which sponge it was built on. Each commitment below is
discharged from the sponge; none is re-proved.

| commitment | theorem |
|---|---|
| rate 4, capacity 4, `rate + capacity = width` | `committed_rate`, `committed_capacity`, `committed_partition` |
| digest reads only absorbed lanes | `committed_digest_within_rate` |
| arity 23, chunked `5·4 + 3`, seven permutation calls | `committed_arity`, `committed_chunking`, `committed_permutationCalls` |
| absorption never writes capacity | `committed_capacity_untouched` |

### Separation, sharpened

Two new theorems answer what prose had been asserting.

`committed_single_arity` — the recipe admits exactly **one** preimage length. So
the padding rule's length-distinguishing job is *vacuous here*: there are no two
lengths to distinguish. This is stronger than the earlier statement that padding
is what separates; at fixed arity it separates nothing, because nothing needs
separating.

`digest_independent_of_placement` — two placed instances compute the same digest
from the same preimage. The base is a column offset and nothing more, so nothing
about *where* a hash sits enters *what* it computes. This is the formal content
of "`hashPrior` and `hashNext` have the same semantics."

**Retracted 2026-07-27 (cycle 384).** This paragraph read: "Together with
`POSEIDON2-HASH-NO-DOMAIN-TAG`, the answer is exact — what separates a prior
digest from a next digest is entirely the preimage content."

`POSEIDON2-HASH-NO-DOMAIN-TAG` was retracted in cycle 345: a separator **does**
exist, `normalizedIteration`'s `+1` on the iteration coordinate. That cycle said
this paragraph "inherits the same error and is retracted with it" — and retracted
it in the other entry's prose while leaving this one standing. Corrected here.

What survives is `digest_independent_of_placement` itself: placement carries no
separation. Two call sites hashing equal 23-field preimages do obtain equal
digests — but the two calls do not hash equal preimages, because the profile
separates them at slot zero.

---

## TERMINAL-CHECK-SELECTION

```text
claim:
  runningCheck and freshCheck are setup selections. No row program for them
  can be derived from this encoding.
status: OBSTRUCTION, kernel-checked 2026-07-27 (cycle 332).
  TerminalCheckSelectionBoundary.
```

`Vocabulary.callEval` sends `Call.runningCheck` and `Call.freshCheck` to
`parameters.terminalChecks.runningCheck` and `.freshCheck`. Those are **fields**
of `CanonicalTerminalVerifier.RelationChecks`, whose `_iff` fields tie them to
`TerminalRelations.runningHolds` and `.freshHolds` — themselves setup-supplied
`Prop`s. The pair `(relations, checks)` is an input.

`runningCheck_is_a_real_choice` and `freshCheck_is_a_real_choice` exhibit two
legitimate inhabitants over the same carriers that give opposite verdicts on the
same argument. `lawful_checkers_still_disagree` adds that both are fully lawful —
each satisfies its own `_iff` — so the disagreement is not an artefact of an
unconstrained field.

### Not the vacuous shape

`NifsCompletionBoundary` withdrew a theorem of the form "X does not determine Y"
for independent X and Y, which is provable for every X and says nothing. These
statements are not that: they fix everything except the checks and compare two
concrete inhabitants at one concrete argument, so what varies is exactly the
object the claim is about.

### The selection surface

`step`, `nifsVerify`, `runningCheck`, `freshCheck`. All four are setup-selected;
none is derivable.

**Completed 2026-07-27 (cycle 352).** Three of the four were kernel-checked;
`step` was asserted from the first cycle and never proved. It is now
`StepSelectionBoundary.step_is_a_real_choice`: two legitimate `Machine`s over
the same carriers take the same state and witness to different states.

| call | boundary |
|---|---|
| `nifsVerify` | `NifsCompletionBoundary.setupVerifier_is_a_real_choice` |
| `runningCheck` | `TerminalCheckSelectionBoundary.runningCheck_is_a_real_choice` |
| `freshCheck` | `TerminalCheckSelectionBoundary.freshCheck_is_a_real_choice` |
| `step` | `StepSelectionBoundary.step_is_a_real_choice` |

"Is a field" had been read rather than checked, and the other three had already
shown what checking looks like. `allRecipes` and `N_canonical` are downstream of
all four, and `CanonicalProgram.SelectedRecipe` is the interface each enters
through.

---

## TERMINAL-CHECK-RECEIPT-CARRIED

```text
claim:
  The one native-conformance instantiation of these checks returns a carried
  verdict rather than recomputing it.
status: RECORDED 2026-07-27 (cycle 332) by reading the instantiation.
```

`Implementation/Rust/CanonicalConformance/OneSlot.lean` defines

```text
runningReceiptCheck case key value witness =
  match case.trace with
  | .base => false
  | .recursive _ _ _ _ receipt =>
      decide (key = receipt.key ∧ value = receipt.value ∧
        witness = receipt.witness) && receipt.accepted
```

and `terminalRelations` defines `runningHolds` as that Bool being `true`.

So in this lane the relation *is* "the receipt says accepted, and its fields
match". The project rule is that digests compress but never authorise, and a
check whose verdict is a carried `accepted` bit is on the wrong side of that
line for anything crossing a trust boundary.

**Scope, stated precisely.** This is a fact about the Lean conformance model of
the native F′ lane, not a claim that shipping Rust fails to enforce a frozen
relation.

### RETRACTED 2026-07-27 (cycle 335)

The framing above was wrong, and the error was mine: I read a **differential-test
schema** as a model of the protocol's checks.

`OneSlot.lean`'s own contract header says what it is — "a proof-free schema for
recording the primitive calls made by one canonical step or terminal check, and
executable comparison against an externally supplied Rust Boolean" — and states
the discipline explicitly: "`rustAccepted` is never an input to a reconstructed
setup, machine, primitive receipt, or canonical evaluation; it occurs only on
the right-hand side of the two differential comparisons."

Checked rather than taken on trust: `rustAccepted` occurs exactly twice outside
the docstring, both as the right operand of `decide (… = case.rustAccepted)`.
It never feeds a reconstruction.

So `runningReceiptCheck` returning `receipt.accepted` is not the protocol
authorising on a carried bit. It is the recorded side of a differential
comparison — the thing being tested, which is where a recorded outcome belongs.

**There is no production defect here.** The concern is withdrawn. What survives
untouched is `TERMINAL-CHECK-SELECTION`: `runningCheck` and `freshCheck` are
setup selections, kernel-checked, and that claim never depended on this
reading.

The residual lesson is about method, not about the code: a module whose header
says "does not own Rust correctness" and "emits constraints: no" is not a place
to read protocol authority from, and the header said so before the finding was
written.

---

## TRANSCRIPT-MODE-BOUNDARY

```text
claim:
  The Fiat-Shamir transcript is a different construction from the binding
  sponge. No transcript recipe may be built on Poseidon2Sponge.
status: BLOCKED ROUTE, kernel-checked 2026-07-27 (cycle 336).
  TranscriptModeBoundary.
```

### The absorb modes differ

`Poseidon2Sponge.absorbChunk` **adds** into the rate lanes:
`(state lane + value) % p`.

`neo-transcript/src/poseidon2.rs` **overwrites** them. `absorb_elem` is
`self.st[self.absorbed] = x`, and `absorb_slice`'s unrolled fast path carries
the comment "We use assignment (overwrite) to match absorb_elem behavior".

| theorem | content |
|---|---|
| `modes_agree_on_initial_state` | from the all-zero state the two coincide — which is why the difference is easy to miss, since it is invisible on the first chunk |
| `modes_differ` | from any state carrying a value they diverge — which is every chunk after the first |
| `divergent_values` | the concrete lane values, `6` versus `1`, so the fixture cannot pass by both sides being equal for an unrelated reason |

### Three further differences

- **Arity.** The sponge is fixed at 23 fields. The transcript is a duplex over
  variable-length input with a cursor.
- **Padding.** `Poseidon2Sponge.pad` adds a single `1` to lane 0 before a final
  permutation. The transcript has no such step; `absorb_packed_bytes_with_len`
  absorbs the byte length as a field element instead.
- **Length separation.** The transcript therefore separates lengths by absorbing
  one. `POSEIDON2-HASH-COMMITMENTS` records that at fixed arity padding
  separates nothing; that argument is about the binding hash and does **not**
  transfer here.

### Why this is a result

Prompt section 4.6: Construction 2's binding hash and the Fiat–Shamir random
oracle are distinct objects with distinct security contracts, and may share
arithmetic. They do share the permutation. They do **not** share the sponge.

Building the step transcript on `Poseidon2Sponge` would encode a different
function from the one the verifier computes — the same defect
`KRecomposition.powerSumFrom_eq_hornerValue` was written to prevent for the
radix-`b` relation. There the two expressions were proved equal; here they are
proved unequal.

### What a transcript recipe would need

A duplex model with a cursor: overwrite absorption, variable arity, and
length-prefixed encoding. That model does not exist in this tree. The route is
recorded as blocked at this construction rather than worked around.

Note also that `Core/TranscriptCertificate` exists but is built on the 600-row
**artifact** permutation, which prompt section 4.1 lists as a measured value.
It is not the Phase-2 program and cannot supply a derived count.

---

## TRANSCRIPT-DUPLEX-MODEL

```text
claim:
  The value-level duplex the Fiat-Shamir transcript runs, with its cursor
  invariant proved unconditional.
status: PROVED 2026-07-27 (cycle 337). Poseidon2Duplex.
evidence: model-proved
```

`TRANSCRIPT-MODE-BOUNDARY` recorded that no transcript recipe could be built on
`Poseidon2Sponge`, and named what was missing: a duplex with a cursor. This is
that model. It is **value-level and owns no row program** — emitting rows before
the model is right is exactly how a recipe ends up computing a different
function from the verifier.

### Transcribed from `neo-transcript/src/poseidon2.rs`

```text
fn absorb_elem(&mut self, x) {
    if self.absorbed >= RATE { self.permute(); }
    self.st[self.absorbed] = x;      // overwrite
    self.absorbed += 1;
}
fn permute(&mut self) { self.st = perm(self.st); self.absorbed = 0; }
```

and before every squeeze, `absorb_elem(ONE); permute();` — the source's own
comment calls that pair a "domain gate before squeezing to avoid state reuse
issues".

### The cursor invariant is unconditional

The interesting result, and it is stronger than expected.
`guarded_absorbed_lt` takes **no hypothesis**: either the guard fires and the
cursor becomes `0`, or it did not fire and the cursor was already below the
rate. So `cursor_le_rate`, `capacity_untouched` and `duplex_absorb_is_overwrite`
need no invariant premise either.

The first draft carried `s.absorbed ≤ rate` through all of them. It was
redundant — a premise no consumer would have needed to construct — and dropping
it made every statement stronger. Rust carries the same fact as an `assert!` in
`from_state_and_absorbed`; here nothing has to be assumed.

### The capacity guarantee

`capacity_untouched`: absorption never reaches a capacity lane, because the
guard fires first. This is the duplex's analogue of
`Poseidon2Sponge.RateChunk_capacity_untouched`, and it is the security-relevant
corollary of the cursor bound rather than a separate assumption.

### The transcript's separation mechanism

The pre-squeeze gate, not the sponge's padding rule. `gate_absorbed` and
`challengeField_cursor` record that a challenge is always read from a freshly
permuted state with the cursor at zero, never from one absorption has just
written into. `POSEIDON2-HASH-COMMITMENTS`' fixed-arity argument is about the
binding hash and does not apply here; this construction separates by gating
every squeeze.

### What is still missing for a recipe

The row program. This model says what to encode; it does not encode it. A
transcript recipe would place one permutation program per `permute` call — the
placement machinery `POSEIDON2-HASH-PLACEMENT` already owns — and emit the
overwrite and cursor logic as rows.

---

## TRANSCRIPT-RECIPE

```text
claim:
  The Fiat-Shamir duplex as an emitted row program: one permutation per round,
  overwrite absorption free, and the chain that makes it a duplex.
status: PROVED 2026-07-27 (cycle 338), eight of ten section-2 items.
  TranscriptRecipe.
evidence: model-proved
```

### The entry is where the mode lives

A duplex round is one permutation applied to an entry state. Each entry lane is

- the **absorbed column alone**, when the round writes that lane
  (`entry_overwritten`), or
- the previous round's **output port alone**, when it does not
  (`entry_carried`), or
- the **empty combination** at round zero, which is the all-zero initial state
  (`entry_initial`).

Neither is a row. That is the same "absorption is free" fact the sponge has, but
the combination differs: add mode carries `[(chunk, 1), (previousPort, 1)]`
where this carries one term or the other.

`TRANSCRIPT-MODE-BOUNDARY` proves those denote different values from any carried
state. Here the difference is visible in the emitted syntax, which is where an
encoder can actually get it wrong.

### The cursor is static

`Poseidon2Duplex`'s cursor is a runtime value. In a row program it is not: the
absorb schedule is fixed at encoding time, so no row implements the
`if absorbed >= RATE then permute` guard — the encoder has already placed the
permutations where the guard would have fired.

`Schedule` is that parameter. A caller supplying one inconsistent with the
duplex's cursor arithmetic gets a program that is internally sound and models
something else, so the schedule is named as the caller's obligation rather than
assumed away.

### Counts

`rounds · 352` rows and `rounds · 344` auxiliaries, both derived through
`canonicalProgramFrom_length`, which already records that a carried entry costs
no extra row.

---

## TRANSCRIPT-RECIPE-OPEN

```text
claim:
  Honest completeness and conservation are not proved for TranscriptRecipe.
status: CLOSED 2026-07-27 (cycle 339). Both are now proved;
  TranscriptRecipe meets all ten section-2 items.
```

### Closed

`transcriptRows_honest` and `transcriptRows_conservation` discharge the two
items this entry named one cycle earlier.

Neither needed new machinery. `Poseidon2HonestFrom.honest_satisfies_normalizedFrom`
and `Poseidon2Conservation.scheduleOfFrom_columns` were already **layout-generic
and entry-generic** — written for the sponge's carried entry, and the overwrite
entry is just another entry. What the transcript had to supply was the per-round
packaging (`RoundHonest`) and the case analysis over the four S-box row shapes
and the binding row.

That the carried-entry lemmas transferred unchanged is the useful fact: the
add/overwrite distinction lives entirely in the entry *combination*, and nothing
downstream of it depends on which mode produced that combination.

### The original text, for the record

Eight of the ten section-2 items hold: constructive row program, derived count,
row ownership, column ownership, soundness, the chain, `Typed.Cost`, fail-closed
guard, spec and ledger.

**Honest completeness and conservation are not among them.** They follow
`Poseidon2Sponge`'s pattern and need its per-call witness machinery lifted to
the overwrite entry, which is not done. The module states this in its own header
rather than leaving a reader to infer it from what is absent, and this entry
names it as a property so it is not a comment.

Recording a recipe at eight of ten is the honest form. Claiming ten would be the
subtotal-presented-as-a-total defect the prompt names.


---

## POSEIDON2-HASH-SEPARATION-OWNERSHIP

```text
claim:
  Domain separation is owned by the encoding profile, not by the placement
  recipe, and that is the correct layering.
status: DETERMINED 2026-07-27 (cycle 346).
```

The DONE condition asks that `hashPrior` and `hashNext` own "absorption,
padding, rate, capacity and domain separation". Four of those five are owned by
`Poseidon2HashRecipe` as explicit commitments — `POSEIDON2-HASH-COMMITMENTS`.
The fifth is not, and putting it there would be wrong.

**Separation is a property of the preimage.** `Poseidon2HashRecipe` owns the
*placement of a sponge*: which columns a hash instance occupies and that two
instances do not collide. It never sees a preimage — its input is a
`Poseidon2Sponge23.Preimage`, an abstract `Fin 23 → Nat` supplied by a caller.

The preimage is built one layer up, by `Poseidon23ApplicationProfile`, which is
also where `normalizedIteration` applies the separator and where
`hashPrior_exact` and `hashNext_exact` bind the result to the frozen hash. That
is the only layer with the information the property is about.

Re-exporting the separator downward would invert the dependency —
`Encoding` imports `Canonical`, not the reverse.

### Amended 2026-07-27 (cycle 351)

The layering argument above said the recipe "cannot see a preimage". **That was
wrong** — `hashProgram_computes_digest` takes one as a parameter. The layering
conclusion survives, but the reason given for it did not, and the correct
version yields something the recipe *does* own.

Separation by preimage content only means anything if the preimage is actually
absorbed. A recipe that silently dropped a field would make two distinct
preimages hash identically no matter what the profile did upstream. That is this
layer's contribution, and it is now proved:

| theorem | content |
|---|---|
| `chunkValue_at_index` | field `call · 4 + lane` is exactly what chunk `call` lane `lane` reads |
| `chunkAt_determines` | two preimages with the same chunks are the same preimage |

Calls `0 … 4` carry four lanes each and call `5` carries three — `5 · 4 + 3 =
23`, every index exactly once. **The chunking loses nothing**, so whatever the
profile distinguishes upstream survives absorption.

So the split is: the profile *applies* the separator; this recipe *preserves*
it. Neither statement is the other, and the second was missing.

---

## TRANSCRIPT-CANONICAL-LAYOUT

```text
claim:
  The transcript's per-round layout is the encoder's choice, and the
  collision-freedom that choice buys is a theorem.
status: PROVED 2026-07-27 (cycle 350). TranscriptRecipe.canonicalLayouts.
evidence: model-proved
```

`transcriptRows` takes `layouts` as a parameter. The distinction
`PIDEC-LOWNORM-CANONICAL-ALLOCATION` drew applies here too: *which* layout each
round gets is the encoder's, and the properties of that choice are theorems.

`canonicalLayouts` places round `r` at stride `r`, reusing the stride
`Poseidon2Sponge` already proved clears a full column space.

| theorem | content |
|---|---|
| `canonicalLayouts_eq_spongeCall` | it *is* the canonical sponge's own call layout, so nothing new is introduced |
| `canonicalLayouts_wellFormed` | each round's layout is coherent |
| `canonicalLayouts_disjoint` | distinct rounds allocate disjoint columns |

Disjointness is not a new argument: it is
`SpongeLayout.WellFormed.auxDisjoint`, which the canonical sponge layout already
satisfies and which is definitionally the same placement. The transcript needed
the fact, not a proof of it — and finding that out took looking, because the
previous cycle's queue described this as work rather than as a lookup.

### What a caller still supplies

The **absorb schedule**: which lane each round overwrites, and from which
column. That is about the data being absorbed, and no layout choice determines
it. One obligation, and it is genuinely the caller's — the same shape the
low-norm batch reached.


---

## POSEIDON2-HASH-SEPARATOR-DELIVERY

```text
claim:
  The recipe delivers the profile's separator to a specific chunk lane.
status: PROVED 2026-07-27 (cycle 364). chunk_differs_at_index,
  separator_reaches_chunk_zero.
evidence: model-proved
```

`chunkAt_determines` says the chunking is lossless — some chunk differs when the
preimages do. This is sharper: **the chunk carrying that index** differs.

Field `i` is read by chunk `i / 4` at lane `i % 4`, and by nothing else
(`chunkValue_at_index`). So a difference at index `i` reaches chunk `i / 4`
specifically. `separator_reaches_chunk_zero` states it for slot zero, the case
the production projection uses.

### The composition, complete

`Poseidon23ApplicationProfile.normalizedIteration` differs between the two calls
in the **first source coordinate**. The profile's projection places source
coordinates into preimage slots. This recipe then carries a difference in a slot
into a difference in that slot's chunk lane.

| layer | owns |
|---|---|
| encoding profile | **applies** the separator — `+1` on the iteration coordinate |
| this recipe | **delivers** it — into a lane of a chunk the sponge absorbs |
| `Poseidon2Sponge` | absorbs, pads, and fixes rate and capacity |

Each half is where it belongs, and neither layer can make the other's statement:
the profile cannot see chunks, and the recipe cannot see the iteration.

### What is still not claimed

That the *digests* differ. That is collision resistance of the permutation — a
hardness assumption, not an arithmetic fact, and section 4.6's rule applies:
shared arithmetic does not transfer a security contract. What is proved is that
the encoding does not lose the distinction before the permutation sees it.


---

## POSEIDON2-HASH-SEPARATOR-APPLICATION

```text
claim:
  The recipe can apply a first-slot separator, not only preserve one — and
  what that establishes is bounded.
status: PROVED 2026-07-27 (cycle 365). separatedPreimage_differs,
  separatedPreimage_reaches_chunk_zero.
evidence: model-proved
```

Everything before this said the recipe *preserves* a separation applied
elsewhere. It can also *apply* one. `separatedPreimage next input` adds one to
slot zero in the field, leaving the rest alone.

### The reduction is not decoration

`Poseidon23ApplicationProfile.normalizedIteration` adds one to a **field**
coordinate. A `Nat` increment would be a different function wherever the
coordinate is `p - 1` — and `separatedPreimage_differs` covers exactly that
case: at `p - 1` the field sum is `0`, which still differs.

That is the `TRANSCRIPT-MODE-BOUNDARY` defect in miniature — an encoding that
looks like the shipped function and is not — avoided by construction rather than
discovered later.

### What this does and does not establish

**Does:** a separator of this shape moves slot zero for *every* residue, and
reaches chunk zero. So a profile applying one loses nothing at this layer.

**Does not:** that this *is* the profile's separator. The profile's version is
authoritative; whether the two agree is a conformance obligation between layers
and is not established here, because this module cannot see an iteration.

That boundary is why `POSEIDON2-HASH-SEPARATION-OWNERSHIP` still stands: the
recipe can supply the *action*, and only the profile can supply the *argument*
it is applied to. Requirement 2's clause is met to the extent one layer can meet
it, and the residue is a named conformance obligation rather than an absence.


---

## POSEIDON2-HASH-SEPARATOR-CONFORMANCE

```text
claim:
  The profile's separator is confined to the slots its plan points at the
  iteration, and slot zero carries it when the plan sends it there.
status: PROVED 2026-07-27 (cycle 366).
  Encoding.Poseidon23SeparatorConformance.
evidence: model-proved
```

Cycle 365 named this as the residue: `Poseidon2HashRecipe.separatedPreimage`
supplies the *action* of a first-slot separator, and cannot say the action is
the profile's, because `Canonical` cannot see an iteration. This module can see
both.

| theorem | content |
|---|---|
| `selected_slot_zero` | slot zero carries `normalizedIteration next iteration`, when the plan sends it to source coordinate zero |
| `prior_next_differ_at_slot_zero` | the two calls' slot-zero entries are exactly `normalizedIteration true` against `false` |
| `prior_next_agree_off_slot_zero` | every slot pointing away from source coordinate zero carries identical data between the calls |

The third is what makes the separation *exact* rather than merely present: the
difference is confined, so nothing else about the two calls varies.

### The hypothesis is a plan property, and it is checkable

A `CoordinatePlan` may send slot zero anywhere; nothing in the type stops it.
The bridge therefore takes `(plan.preimage ⟨0,_⟩).val = 0` explicitly. A
deployment checks that of its own plan — which is the same gap
`POSEIDON2-HASH-PROJECTION-INJECTIVITY` recorded about plans generally, now
surfaced at the one slot where it changes what the protocol separates.

### Requirement 2, as far as the layering permits

| layer | statement |
|---|---|
| encoding profile | applies the separator, confined to the iteration's slots |
| this bridge | connects that application to a named slot |
| `Poseidon2HashRecipe` | supplies the action, delivers a slot difference to a chunk lane, preserves every preimage distinction |
| `Poseidon2Sponge` | absorbs, pads, fixes rate and capacity |

Nothing in this chain is asserted where it cannot be seen, and the one residue —
that a deployment's plan sends slot zero to source coordinate zero — is a
hypothesis a deployment discharges rather than an omission.


---

## POSEIDON2-HASH-PLAN-STRENGTHENING

```text
claim:
  The plan hypothesis can be carried by a type. Strengthening CoordinatePlan
  itself would be stronger and is a change-control question, not taken here.
status: REFINEMENT PROVED 2026-07-27 (cycle 367).
  SeparatingPlan and its two restated bridges.
```

The bridge theorems took `(plan.preimage ⟨0,_⟩).val = 0` as a loose side
condition. A deployment that forgets to check it gets a plan that typechecks and
a separator that lands nowhere in particular.

`SeparatingPlan` bundles the plan with the property, and
`SeparatingPlan.selected_slot_zero` and `SeparatingPlan.prior_next_differ`
restate the bridge with **no side condition at all**. A deployment that
constructs one cannot omit the check.

### Why `CoordinatePlan` itself was not changed

Adding the field there would be the stronger fix. It is **not** taken
unilaterally: `CoordinatePlan` is what a deployment must supply, so requiring a
new field changes what counts as a valid deployment — a change-control question
under spec §16, not a formalization one.

Recorded as the alternative rather than performed. A refinement is additive:
existing plans still typecheck, and a deployment opts in by proving one equation
about its own.

**The decision this needs:** whether `CoordinatePlan` should require slot zero
to read source coordinate zero. If yes, the refinement folds into the structure
and the hypothesis disappears everywhere. That is a protocol-shape call.


---

## POSEIDON2-HASH-PADDING-UNCOMMITTED

```text
claim:
  hashPrior and hashNext commit to what their padding is and to the fact that a
  prover cannot choose it.
status: PROVED 2026-07-27 (cycle 386). committed_padding_chunk,
  committed_padding_on_constant_wire, committed_padding_value,
  committed_padding_input_independent, committed_padding_call,
  committed_absorbed_total.
```

The parameter-commitments section opened by naming what it commits to:

> Absorption, padding, rate and capacity are `Poseidon2Sponge`'s to prove. They
> are *this recipe's to commit to*.

Four items named. Walking the eight theorems below it against that list:

| named | committed |
|---|---|
| absorption | `committed_chunking`, `committed_permutationCalls`, `chunkValue_at_index` |
| rate | `committed_rate`, `committed_partition`, `committed_digest_within_rate` |
| capacity | `committed_capacity`, `committed_capacity_untouched` |
| **padding** | **nothing** |

### Padding was the one with a soundness consequence

The other three are shape commitments — a reader checks a width against a
constant. Padding is different: an absorbed lane a prover can choose is a lane
that lets one preimage be extended into another, and the digest then fails to
determine what it compresses.

The scheme is sound. `chunkLength` gives seven calls — six of data carrying
`4+4+4+4+4+3 = 23` fields, and a seventh absorbing `[1]`. `layout.chunkColumn 6`
is `0`, the constant wire. So the padding value is pinned by `z 0 = 1` and there
is no witness column to vary.

**None of that was stated at this layer.** It was reachable by opening
`Poseidon2Sponge23` and reading `chunkAt_padding` and `chunkColumn_padding` —
which is exactly what the commitments section exists to spare a reader.

### What the existing padding text said instead

`committed_single_arity` and the header discussed padding only in its
*separation* role, and correctly: at fixed arity the length-distinguishing job
is vacuous. Separating nothing is not the same as doing nothing, and the header
recorded the vacuous half while the load-bearing half went unwritten.

### Method

Requirement 2 names five things — absorption, padding, rate, capacity, domain
separation. Enumerating the commitments against that list, rather than against a
sense of whether the section looked complete, is what surfaced it. The prior
cycle's lesson applied forward: enumerate against a list, not against a feeling.
Domain separation had been examined for eleven cycles; padding, sitting beside
it in the same five-item list, had not been looked at once.


---

## POSEIDON2-HASH-ABSORPTION-MASKING

```text
claim:
  A difference between two preimages survives absorption into the state, not
  only the chunk decomposition.
status: PROVED 2026-07-27 (cycle 387).
  Poseidon2Sponge.absorbChunk_injective_at_lane,
  Poseidon2HashRecipe.separator_survives_absorption.
```

`chunkAt_determines` proves two preimages with the same chunks are the same
preimage. Its docstring drew a further conclusion:

> whatever the profile distinguishes upstream survives absorption

**Chunking and absorption are two steps.** Chunking arranges a preimage into
blocks; absorption folds a block into the state. A recipe with injective
chunking could still absorb two distinct chunks into one state, and the
separator would be lost before a single permutation ran — a *structural* failure,
not a cryptographic one. Nothing ruled it out.

### The step that was missing

`absorbChunk` adds the chunk into the rate lane modulo the prime, and addition is
injective: two canonical values absorbed at the same lane of the same state give
different lanes exactly when they differ. Stated at a lane rather than on whole
chunks, because that is where a separator lands — one slot, not a block.

`separator_survives_absorption` composes it at slot zero: the state after
absorbing chunk zero differs between a preimage and its slot-zero increment.

### Where the separation chain now stands

| step | status |
|---|---|
| preimages differ at slot zero | `separatedPreimage_differs` |
| chunk zero differs | `separatedPreimage_reaches_chunk_zero` |
| **state after absorbing chunk zero differs** | **`separator_survives_absorption`** |
| digest differs | Poseidon2's, not this recipe's |

The chain is now arithmetic-complete: every step that *can* be settled by
arithmetic is. That sharpens what "domain separation is split across layers"
means — the split sits precisely at the permutation, not vaguely at "the
profile".

### How it was found

By reading a docstring against the theorem under it. The theorem is about
chunking; the sentence claimed absorption. This is the second cycle running in
which prose asserted a conclusion no theorem carried — cycle 386's was a
commitments header naming padding, this one a docstring naming absorption.


---

## POSEIDON2-HASH-SEPARATION-NOT-COMMITTED

```text
claim:
  The fifth of Phase 3's five items is committed to at the recipe, in the same
  form as the other four.
status: PROVED 2026-07-27 (cycle 395). committed_separation_survives, with
  digest_independent_of_placement as its fourth element.
```

Nine cycles of assessments recorded requirement 2 as "four of five", with domain
separation "split across layers by design". Re-reading the split instead of
re-citing it — the practice that dissolved the item 3 blocker in cycles 393 and
394 — shows what was actually missing.

### What "owning" means in this file

The module's own commitments header says it:

> Absorption, padding, rate and capacity are `Poseidon2Sponge`'s to prove. They
> are *this recipe's to commit to*.

So four of five are owned in the sense of **committed to, discharged from
elsewhere, not re-proved**. Nothing in that sense was impossible for the fifth.
The separation facts existed — `chunkAt_determines`,
`separatedPreimage_reaches_chunk_zero`, `separator_survives_absorption`,
`digest_independent_of_placement` — scattered across three sections, none framed
as a commitment. Same shape as cycle 386's padding: the facts were there and the
commitment was not.

### The correction to the stated reason

The recorded justification was that "this module cannot see a preimage". **That
is false** — `separatedPreimage`, `chunkValue_at_index` and `chunkAt_determines`
all take a `Preimage`. What the module cannot see is an **iteration coordinate**.
The boundary is therefore not preimage-versus-placement but *applied*-versus-*not
lost*:

- **the profile owns that a separator is applied**, and that its argument is the
  iteration — `Poseidon23SeparatorConformance.SeparatingPlan`;
- **this recipe owns that an applied separator is not lost** — by placement, by
  chunking, or by absorption.

### Not a subtotal

`committed_separation_survives` carries three of the four elements. The fourth,
`digest_independent_of_placement`, is proved and guarded above and is named in
the docstring rather than restated: its thirteen-argument signature inside a
conjunction would obscure rather than sharpen. The theorem's *name* says what it
carries and does not claim to be the whole commitment.


---

## POSEIDON2-HASH-SEPARATION-APPLIED

```text
claim:
  The 23-row program applies an iteration separator only when both hash calls
  use the same state and running payload.
status: RECLASSIFIED 2026-07-29. It is a same-payload diagnostic and is not a
  relation between the actual HyperNova hashPrior and hashNext calls.
```

The actual calls use `(i, z0, zi, U_i)` and
`(i + 1, z0, zNext, U_next)`. Their payloads can differ. Placing these rows
between those calls would incorrectly require `zi = zNext` and
`U_i = U_next`. `changed_tail_rejected` makes this boundary executable.
The complete F-prime assembly does not import or emit these rows. The
remaining text in this historical section describes only the same-payload
fixture and must not be read as a production composition claim.

Ten cycles recorded separation as split because "this module cannot see an
iteration coordinate". Cycle 395 corrected the stated reason and kept the split.
**It does not need to see one.**

### What the profile actually specifies

`Poseidon23ApplicationProfile.sourceCoordinates` is

```text
normalizedIteration next iteration :: (encode z0 ++ encode current ++ encode running)
```

with its own docstring saying the rest: *"Prior and next hashes share one
physical projection. The only semantic difference is the normalized first
coordinate."*

So the two preimages are related by **exactly** `separatedPreimage true` — equal
at every slot but the first, `+1` at the first — and that relation is expressible
in columns without naming an iteration. Twenty-three rows enforce it: one
increment at slot zero, twenty-two equalities after it.

### What changes

Before, a prover could feed the two placed sponges unrelated preimages and the
emitted program would not object; separation held because the profile promised
to build the preimages correctly. `separationRows_applies` makes the program
reject that prover.

Nothing about the frozen relation changes. These rows *enforce* what the profile
already specifies, and the honest profile satisfies them by construction —
`separationRows_honest` is that direction.

### Where the boundary sits now

| owned by | what |
|---|---|
| the recipe | that a separator is applied — `separationRows_applies` |
| the recipe | that it is not lost — `committed_separation_survives` |
| the profile | that the moved coordinate *is* the iteration |

The remaining profile obligation is about **meaning**, not about whether the
circuit enforces separation. `Poseidon23SeparatorConformance.SeparatingPlan` is
where it sits, and `CoordinatePlan`'s slot-zero question is its §16 half.


---

## POSEIDON2-HASH-SEPARATION-CONFORMANCE-UNCOMPOSED

```text
claim:
  For one fixed payload tuple, the profile's next-mode preimage changes only
  the normalized iteration coordinate.
status: RECLASSIFIED 2026-07-29. This does not compare the two real F-prime
  calls, which use different current and running values.
  selected_at_iteration_slot, SeparatingPlan.prior_next_agree,
  SeparatingPlan.next_is_separated.
```

Cycle 396 left one profile-side obligation and called it a §16 decision. It was
not a decision. It was an uncomposed pair.

### Three bridges, two restated

`Poseidon23SeparatorConformance` proved three things about a `CoordinatePlan`
with the slot-zero hypothesis: slot zero carries the normalized iteration, the
two calls differ there, and away from the iteration coordinate they agree.
`SeparatingPlan` bundles the hypothesis and restates the bridges without it —
**two of the three.** `prior_next_agree_off_slot_zero` was never restated.

That unevenness made the composition look unavailable: the "differ here" half was
reachable without a side condition and the "agree elsewhere" half was not, so
joining them appeared to need the hypothesis back, and pushing the hypothesis
into `CoordinatePlan` looked like the way to get it.

### The composition needs nothing new

`SeparatingPlan` already carries the hypothesis. Restating the third bridge over
it and joining the two gives `next_is_separated`: at every slot, the next
selection equals the prior one unless that slot points at source coordinate zero,
where it is `normalizedIteration true`.

`selected_slot_zero` generalises to `selected_at_iteration_slot` on the way —
separation is located by *which source coordinate a slot pulls from*, not by the
slot's own index, and only the general form composes.

### What §16 is actually about

`POSEIDON2-HASH-PLAN-STRENGTHENING` asks whether `CoordinatePlan` should carry
the slot-zero field. That remains a change-control question and is not taken
here. But it is a question about **where a hypothesis is stored**, not about
whether the conformance holds — and it had been recorded as the latter for
eleven cycles.

### The pair this completes

| side | theorem |
|---|---|
| recipe enforces the relation | `Poseidon2HashSeparation.separationRows_applies` |
| profile's data satisfies it | `SeparatingPlan.next_is_separated` |
