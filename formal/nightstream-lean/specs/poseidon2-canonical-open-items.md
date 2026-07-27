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
  Whether the profile-indexed application-compiler boundary can close the
  remaining setup-owned calls and assemble the complete canonical fixed-one
  Step and Terminal programs without selecting a test fixture.
status:
  MODEL-PROVED, PROFILE-INDEXED.
```

`Phase5CallCertification` owns the two remaining setup-selected physical
programs: the HyperNova application `step` and the selected `nifsVerify`.
Both are complete existing `CallRecipe` values, so the boundary receives
executable rows plus exact footprint, receipt, soundness, honest-completeness,
and inactive-satisfiability proofs. It receives no accepted Boolean, semantic
conclusion, opaque call outcome, or generated artifact. A deployment must
construct these recipes from the Lean-owned application/setup encoding; Rust
is subsequently compared with that selected program and is never authority
for the recipes, layouts, rows, or costs.

`CompleteApplicationCertification.allRecipes` combines those programs with
the four Phase 3/4 recipes and five direct recipes. Its eleven-entry
recipe-family domain is duplicate-free and supplies one recipe at every typed
call position. Runtime multiplicities are not inferred from that domain:
`stepProgramCalls_exact` and `terminalProgramCalls_exact` traverse the actual
typed ASTs and expose their distinct nine-call and eight-call sequences. The
constructed canonical Step and Terminal programs prove:

- physical acceptance iff the unchanged frozen checker accepts, for values in
  the selected codecs' honest domain;
- honest accepted values construct satisfying assignments;
- every row and allocated column has one source-aligned receipt owner;
- no rows or columns exist outside the receipt lists;
- exact cost is the receipt fold;
- the selected Step and Terminal costs attain the existing finite
  rewrite-class minima.

This closes canonical Phase 5 at the profile-indexed model boundary. HyperNova
intentionally leaves the application circuit and NIFS verifier setup-selected,
so a deployment must still supply one concrete certified profile and the two
physical programs before Lean can kernel-evaluate a unique deployment number.
Rust-emitted-program equality, generated-row equality, current-production
selection, named-event bounds, and cryptographic security remain separate.
