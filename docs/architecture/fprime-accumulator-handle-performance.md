# F' accumulator-handle performance plan

Date: 2026-05-31
Status: unsafe experiment; do not ship or commit as a protocol plan
Scope: `crates/neo-fold-clean` recursive F' performance

> Soundness note: this file records a performance experiment from the dirty
> worktree. It removes or delays pieces of the HyperNova Construction-2
> accumulator binding. Treat it as evidence for where the cost lives, not as a
> shippable design. The clean rebuild plan is
> `docs/architecture/uncommitted-soundness-salvage-2026-05-31.md`.

## Problem

Before the delayed-handle experiment, the canonical Fibonacci R1CS-F'
recursive step had a source-image width of `6,029,252` columns. A static
layout check showed that the recursive accumulator Poseidon2 trace alone
accounted for `5,406,336` columns, or about `89.7%` of the image.

After removing that producer-side trace from the canonical unified plan,
removing the redundant `public_trace_update` chain, and dropping
`structure_digest` from the `state_x_out` preimage (it is already absorbed by
`vk_fs_digest`), dropping the duplicate `public_trace` absorb (the F'
structure now enforces `new_public_trace == new_z_i` directly), dropping
the batched-build-only `chunk_count`/`pc` absorb in favor of `step_count` as the
paper-level iteration counter, replacing the F' ASCII domain tags
with compact single-field Poseidon2 domain IDs, and eliding the now-dead
unified-mode NIFS payload region from the source image, the same static layout
check reported `244,228` columns.

The next cut removed the old `boundary_update = H(prev_z_i, chunk_digest)`
trace from the canonical image. The step now carries
`new_z_i = chunk_digest` directly and the F' structure enforces that equality
with four linear rows.

An attempted follow-up cut removed `z_0` from the `state_x_out` preimage, but
that shortcut was rejected on soundness grounds. `state_x_out` is the public
link between folded F' instances; once historical steps are folded, base and
terminal pins are not enough to bind each step's private `z_0` witness. This
matches HyperNova Construction 2, where the recursive hash includes both
`vk_fs` and `z_0`.

The current static layout check reports `156,676` columns. A local
two-step Fibonacci R1CS-F' run now measures the recursive append at
`0.26s`, terminal fold at `0.25s`, and terminal-only verification at
`0.11s` on the target machine. That proves the near-term Fibonacci
folding target for this fixture, but it is not a proof that larger app
circuits or future compressed/on-chain wrapping are sub-second.

The remaining canonical width now breaks down as:

| Region | Columns | Share |
|---|---:|---:|
| Poseidon2 traces total | `131,328` | `97.43%` |
| NIFS payloads | `0` | `0.00%` |
| state/chunk digests | `3,200` | `2.37%` |
| boundary/app/control | `259` | `0.19%` |
| CCS constant column | `1` | `<0.01%` |
| **Total** | **`134,788`** | **`100%`** |

The one remaining canonical Poseidon2 trace is:

| Trace | Preimage lanes | Absorbs | Columns | Lifted non-bit rows |
|---|---:|---:|---:|---:|
| state_x_out | `19` | `6` | `131,328` | `2,052` |

For a larger app circuit, the shape changes. The SHA-256 Bellpepper
R1CS-F' fixture is a real recursive chain (not direct-CCS). Under its
test-only tiny Ajtai profile, the current two-step local run measured:

| Phase | Time |
|---|---:|
| preprocess R1CS-F' | `0.90s` |
| step 0 append (base) | `0.27s` |
| step 1 append (recursive) | `0.77s` |
| terminal `chain.finish()` | `1.78s` |
| `verify_uncompressed` | `0.23s` |
| total | `3.98s` |

The SHA run reported `structure.n = 49,354`, `structure.m = 3,133,506`,
and `plan.limbs = 1,576,001`; max RSS was about `19.9GB`. The online
recursive append is therefore already near the one-second target for this
fixture, while terminal finalization and memory footprint remain too high.

The typed app-private SHA slice changes that shape substantially. The static
layout guard now reports:

```text
typed app_private bits: current=1576000 typed=1167256 (vars=24625);
F' n=27808 m=1345818 r_len=15 s_col_len=21
```

This is a column/memory cut, not a complete wall-clock result yet. It reduces
the committed source width by about `408,744` coordinates and lowers the full
F' column domain from the old `3,133,506`-column SHA shape to `1,345,818`.
The first typed slice accidentally added bitness rows for every app-private
coordinate because the app-private region was no longer 64-bit aligned; that
briefly pushed `F'.n` to `1,211,192`. The current version fixes ownership:
R1CS-F' may use a one-bit slot only when the R1CS shape has an explicit
Boolean row for that variable, so the app R1CS row owns the bitness constraint
and the F' shell does not duplicate it. The static guard now reports
`F'.n = 27,808`, close to the SHA app R1CS row count plus shell rows.

The plan derives the recursive CE challenge lengths from the generated F'
structure (`r_len=15`, `s_col_len=21`) instead of reusing legacy constants;
this prevents stale `PostParentShapeMismatch` failures as the layout changes.

A capped two-step SHA R1CS-F' run after this fix measured:

| Phase | Time |
|---|---:|
| preprocess R1CS-F' | `0.22s` |
| step 0 append (base) | `0.05s` |
| step 1 append (recursive) | `0.23s` |
| terminal `chain.finish()` | `0.40s` |
| `verify_uncompressed` | `0.09s` |
| total | `1.12s` |

Peak RSS for that run was about `2.2GB` (`2,242,150,400` bytes by
`/usr/bin/time -l`). This is now close to the near-term viability target for
the two-step SHA fixture, and the remaining time is dominated by terminal
finalization rather than the online recursive append.

The most recent terminal-fold slice made two protocol-neutral NC/FE
representation cuts:

1. Parallelize construction of the weighted SuperNeo matrix caches used by
   the carried-CE `Eval` term in `Π_CCS`.
2. Keep balanced lane-0 values such as `-1` in the compact NC digit-table
   representation. The previous compact table treated only `0` and `+1` as
   lane-0 and fell back to a dense `[K; D]` row table as soon as a Π_DEC child
   contained `-1`.

On the same two-step SHA run, terminal `Π_CCS` moved from about `407ms` before
these two cuts to about `184ms`. The `RowStreamState::build_weighted_matrix_caches`
substage moved from about `101ms` to about `58ms`; the `NcOracle::new` digit
tables moved from about `139ms` to about `19ms`; the NC sumcheck moved from
about `87ms` to about `27ms`. The remaining terminal `Π_CCS` hotspots are now
the weighted SuperNeo cache build (`~58ms`) and row-phase `y_eval` (`~48ms`).

The next terminal slice made the low-norm digit operations explicit in two
places:

1. Specialize the existing dense `Π_RLC` rotation multiply for the exact
   low-norm digits that the reduction already carries: `0`, `+1`, and `-1`.
   This preserves the same `rot(ρ)·Z` equation from SuperNeo §7.4, but skips
   zero lanes and replaces `coeff * ±1` with add/sub.
2. In the `Π_DEC` SuperNeo ring-form evaluator, multiply by digit blocks via
   monomial shifts when the child block is `0/±1`, falling back to full
   `Rq::mul` for any non-digit block. This is the local version of the
   paper's pay-per-bit recommendation: small coefficients should add rotations
   instead of paying for a dense ring product.

On the same capped two-step SHA run, terminal finalization measured about
`400ms`; total test time was about `1.12s`. The terminal split is now roughly:

| Terminal slice | Time |
|---|---:|
| `Π_CCS` | `181ms` |
| `Π_RLC` | `70ms` |
| `Π_DEC` | `140ms` |
| terminal total | `400ms` |

The `Π_CCS` row-phase `y_eval` slice is about `43ms` after the digit-block
helper is wired through the active-block paths. The weighted SuperNeo cache
build remains about `59ms`, and NC oracle/sumcheck remain about `31ms`/`28ms`.

The next micro-slice attacked duplicated work rather than changing any
equation:

1. `verify_uncompressed` batches final-running witness commitments with
   `AjtaiSModule::commit_many`, so the verifier no longer starts one seeded
   Ajtai stream per final CE claim.
2. `Π_DEC` carries the non-zero digit-plane flags produced by `split_b`; all
   zero children are constructed as zero CE claims instead of evaluating
   `M · 0(r)` over the full F' structure.
3. `SuperneoZBlocks` records whether the packed witness is all-zero so the
   ring-form evaluator can return zero rows immediately.

The best capped two-step SHA run observed after these cuts was:

| Phase | Time |
|---|---:|
| preprocess R1CS-F' | `0.22s` |
| step 0 append (base) | `0.05s` |
| step 1 append (recursive) | `0.21s` |
| terminal `chain.finish()` | `0.37s` |
| `verify_uncompressed` | `0.09s` |
| total incl. drops | `1.07s` |

That run reported peak RSS around `2.16GB`. Noisy repeat runs on the same
machine landed as high as `1.22s-1.27s`, mostly from slower
`OptimizedStructureCache::build`, terminal `Π_CCS`, and terminal `Π_DEC`.
So this is close, but **not** a stable sub-second result yet. The recurring
folding work excluding one-time preprocess is already below one second for
this two-step SHA fixture; the full test including preprocess is not.

A later capped repeat on the same worktree, after the zero-witness verifier
fast path, measured:

| Phase | Time |
|---|---:|
| preprocess R1CS-F' | `0.31s` |
| step 0 append (base) | `0.06s` |
| step 1 append (recursive) | `0.27s` |
| terminal `chain.finish()` | `0.50s` |
| `verify_uncompressed` | `0.12s` |
| total incl. drops | `1.42s` |

Max RSS for that run was about `2.2GB`. This did **not** reproduce a
100GB live-memory spike; the large `100GB+` number seen locally came from
accumulated `target/` build artifacts, mostly `target/debug/deps` and
`target/debug/incremental`.

After removing those debug/perf build artifacts (`target/` dropped from
`108GB` to `16GB`) and adding cached per-block zero flags for SuperNeo packed
witness blocks, a fresh capped repeat measured:

| Phase | Time |
|---|---:|
| preprocess R1CS-F' | `0.28s` |
| step 0 append (base) | `0.05s` |
| step 1 append (recursive) | `0.24s` |
| terminal `chain.finish()` | `0.45s` |
| `verify_uncompressed` | `0.11s` |
| total incl. drops | `1.27s` |

Max RSS was again about `2.2GB`. The zero-block flag slice removes repeated
all-zero scans of 54-coefficient packed witness blocks inside SuperNeo
evaluation loops. It is a constant-factor cleanup, not the architectural cut
needed for stable sub-second end-to-end runs.

A later terminal-fold cleanup removed per-proof weighted-matrix materialization
from `RowStreamState::build` and skipped `M · 0` evaluations in
`precompute_for_r`. The whole test remained noisy, but the inner terminal
Π_CCS path improved:

| Terminal Π_CCS subphase | Before | After |
|---|---:|---:|
| `RowStreamState::build` | `~92ms` | `~64ms` |
| `precompute_for_r: y_eval` | `~46-51ms` | `~43ms` |
| `optimized_prove` total | `~215ms` | `~184ms` |
| terminal `chain.finish()` | `~0.45-0.50s` | `~0.44s` |

The same run measured `1.30s` total including preprocess and drops, with max
RSS about `2.0GB`. This confirms the terminal SuperNeo evaluator still matters,
but the remaining gap is no longer a single obvious cache construction bug.

A follow-up cleanup also treats a zero carried CE linear combination as an
absent Eval term in `RowStreamState::build`. This avoids constructing the Eval
table and lazily avoids the row-input χ table when the paired Eval table is
provably zero. It does not change the verifier polynomial: the skipped
contribution is `χ_r(row) · γ^k · 0`.

The same row-oracle builder now reuses the packed `SuperneoZBlocks` already
constructed by `OptimizedOracle::new` for MCS row tables, and skips row scans
entirely when an MCS witness is all zero. This removes duplicate witness
packing and another zero-work row loop without changing `M_j · Z(row)`.
After this reuse, the decoded `z_mcs` vectors became dead storage and were
removed; `precompute_for_r` now uses the MCS gamma table length as the
authoritative MCS count.

The row-phase sumcheck degree bound is now cached once when compiling `f`
instead of being recomputed from every term on each row-round evaluation. The
base-field fast paths also avoid allocating a temporary copy of the requested
evaluation points.

All-zero MCS witnesses are represented compactly in the row oracle instead of
allocating one zero row-domain table per used CCS matrix. The contribution is
not dropped: the row polynomial receives the exact `f(0, ..., 0)` constant for
that MCS slot, so arbitrary CCS polynomials with constant terms remain sound.

The row-phase evaluator now also tracks the active non-zero support inside the
padded row domain. When `f(0, ..., 0) == 0`, rows beyond `structure.n` are
provably zero for both the MCS and Eval terms; after each row challenge that
support shrinks by `ceil(len / 2)`. The tables are still folded over the full
padded domain, but the sumcheck coefficient scan only visits the live prefix.
If `f(0, ..., 0) != 0`, padded all-zero MCS rows may contribute a real constant,
so the support remains the full padded domain.

The SuperNeo ring-linear-form blocks also cache whether their real and
imaginary coefficient forms are nonzero. This avoids rescanning the `D` lanes of
each form on every terminal CE evaluation. A repeat SHA run after that micro-cut
landed at `1.07s` total, so the current best remains the `1.03s` run below; the
change is retained as a small hot-loop cleanup, not as evidence of a new timing
breakthrough.

A capped SHA Bellpepper R1CS-F' repeat after these row-oracle cleanups measured:

| Phase | Time |
|---|---:|
| preprocess R1CS-F' | `0.22s` |
| step 0 append (base) | `0.05s` |
| step 1 append (recursive) | `0.20s` |
| terminal `chain.finish()` | `0.34s` |
| `verify_uncompressed` | `0.09s` |
| total incl. drops | `1.03s` |

The run reported `structure.n = 27,808`, `structure.m = 1,345,818`, and
`plan.limbs = 1,167,257`, with max RSS about `2.0GB`
(`1,994,326,016` bytes by `/usr/bin/time -l`). Recurring proof work excluding
one-time preprocess is below one second on this fixture; the full test
including preprocess is just above one second and still needs repeatability
before it can be called a stable end-to-end result.

The latest terminal split is roughly:

| Terminal slice | Time |
|---|---:|
| `Π_CCS` | `145ms` |
| `Π_RLC` | `70ms` |
| `Π_DEC` | `120ms` |
| terminal total | `340ms` |

The remaining terminal `Π_CCS` work is split between row-oracle construction
(`~47ms`) and row-phase `y_eval` (`~35ms`) on the final fold. These are real
SuperNeo terms, not verifier-cache artifacts.

The same test under `scripts/profile_for_ai.sh` with a short sample window
showed the remaining CPU samples concentrated in real protocol work, not
system waits:

| Hot symbol / stage | Interpretation |
|---|---|
| `Rq::mul_by_monomial` | SuperNeo ring-form shifting remains hot in terminal CE evaluation. |
| `weighted_projection_form_from_orig` | Weighted SuperNeo cache construction is still a major terminal `Π_CCS` cost. |
| Poseidon2 permutation assembly | The remaining `state_x_out` trace and transcript/digest work are still visible. |
| extension-field multiplication | Sumcheck and evaluation accumulation still pay real `K` arithmetic cost. |
| `NcOracle::accumulate_inner_b2_at` / digit-table build | The NC side improved, but remains non-negligible in terminal folds. |

That profile supports the same conclusion as the phase timers: the next
meaningful gains should reduce terminal row/evaluation work or the remaining
bit-backed `state_x_out` trace. More Rayon can help constant factors, but
does not change the dominant equations or source-image width.

So Option D changes the next bottleneck: the hot image is no longer dominated
by the recursive accumulator-parent hash, the old duplicate
`public_trace_update` chain, source-image NIFS payload columns, or the old
boundary-update hash. It is now almost entirely the remaining chain-state
Poseidon2 trace, `state_x_out`.

The removed trace hashed

```text
H(tag, child_count, parent.c_data_len, parent.c_data...)
```

where `parent.c_data` is the Π_RLC parent commitment data produced by NIFS.V.
That trace was expensive because the current low-norm F' image materializes a
bit-backed Poseidon2 execution trace for every in-circuit hash preimage.

The next bottleneck to attack is now the last bit-backed chain-state hash.
Parallelizing either the old 6M-column shape or the new 157k-column shape may
recover constant factors, but it does not plausibly move recursive folding to
the sub-second regime by itself.

## Paper constraints

HyperNova §6.3 Construction 2 requires the augmented function F' to:

1. check the prior fresh instance references the previous output hash,
2. run `NIFS.V` to update the running instance `U_i -> U_{i+1}`,
3. output a hash over the next chain state, including the updated running
   instance.

The optimization note in the same section says the verifier circuit inside F'
should use memory checking to read/write running instances instead of directly
passing all running instances through each step. In this codebase, the analogous
problem is not many program lanes (`ell` is currently one); it is that one
running accumulator contains a large commitment vector.

SuperNeo §7 defines the folded CE relation. For a terminal CE claim and witness
`Z`, the verifier must still establish the CE obligations:

```text
c = L(Z)
x = L_in(Z)
||Z||_∞ < b
y_j = M_j Z(r)
```

The accumulator handle is therefore allowed to be a compact routing value, but
it cannot become the sole authority for those obligations. Digests compress;
they do not prove.

## Current local ownership

Current native and in-circuit state advance use:

```text
acc_digest = accumulator_digest_from_parent_claim(k, parent)
           = H(tag, k, parent.c_data_len, parent.c_data...)
```

Relevant owners:

| Concept | Current owner |
|---|---|
| native accumulator digest | `crates/neo-fold-clean/src/paper/digest.rs` |
| in-circuit accumulator digest | `crates/neo-fold-clean/src/paper/reductions/accumulator_digest_circuit.rs` |
| F' plan / one-shot trace | `crates/neo-fold-clean/src/frontends/f_prime/recursive_plan.rs` |
| F' image layout | `crates/neo-fold-clean/src/frontends/f_prime/image.rs` |
| F' structure rows | `crates/neo-fold-clean/src/frontends/f_prime/structure.rs` |
| R1CS-F' shell | `crates/neo-fold-clean/src/paper/f_prime/r1cs.rs` |
| NIFS.V output wires | `crates/neo-fold-clean/src/paper/nifs/circuit.rs` |

The cheap static guardrail is:

```text
crates/neo-fold-clean/tests/system/fibonacci_f_prime_layout_budget.rs
```

It deliberately stops at `FPrimeImageLayout`; it does not build the full
structure, preprocess, prove, or verify.

## Exact cut point removed

The removed hot-path trace was not ambiguous:

```text
build_recursive_step_image_config(plan)
  -> poseidon_one_shot_preimage_lens[2]
  -> poseidon_transition_enforcements[one_shot_index = 2]
  -> UnifiedAccumulatorSelector { recursive_trace_index: 2, ... }
  -> state_out.new_acc_digest
  -> state_x_out preimage
```

The removed preimage lanes at the old one-shot index `2` were:

```text
pack(tag) || child_count || c_data_len || NifsPayloadLane(parent.c_data[0..])
```

That was the `5,406,336`-column trace. The current static test
`fibonacci_f_prime_layout_has_no_producer_side_accumulator_hash_trace` verifies
that canonical unified mode no longer emits any producer-side Poseidon
transition that absorbs `NifsPayloadLane(parent.c_data[..])`.

The later boundary-update cut removed the old one-shot trace:

```text
H(tag, prev_z_i, chunk_digest)
```

Canonical F' now sets `new_z_i = chunk_digest` and enforces that equality in
the structure. The remaining canonical one-shot trace is `state_x_out`.

## Rejected shortcuts

| Shortcut | Why it is rejected |
|---|---|
| Use `fold_digest` as `acc_digest` | `fold_digest` is the Π_CCS transcript/header digest. It is not currently a digest of the full outgoing running accumulator authority. |
| Hash the full CE claim in a bit-backed trace | Safer than `parent.c_data` only, but likely wider than the current trace because it includes `c`, `X`, `r`, `y_ring`, `m_in`, and `fold_digest`. |
| Replace the trace with the existing field-var Poseidon2 R1CS gadget | Under today's R1CS-F' encoding every R1CS variable is still committed as 64 low-norm bits. The static budget test measures `149,196` extra field variables for this accumulator hash, or `9,548,544` encoded bits — worse than the current `5,406,336`-column trace. |
| Remove `z_0` or `vk_fs` from `state_x_out` | These are part of HyperNova's folded public link. Base/terminal pins do not bind historical private step witnesses after folding, so dropping them would trade columns for soundness. |
| Use a fixed random linear projection of `parent.c_data` | A projection is not authority unless the challenge is verifier-derived and the relation proving it is specified. |
| Only optimize Rayon / CPU utilization | Helpful after the shape is fixed, but it does not remove the `5,406,336`-column trace. |
| Switch hash families casually | Protocol-binding hashes are Poseidon2-only unless explicitly approved. A hash change is a protocol decision, not a local optimization. |

## Target design

Introduce an explicit accumulator handle layer:

```text
AccumulatorHandle {
    claim_count,
    authority_handle,   // compact Poseidon2-bound handle
    memory_root_or_log, // or another authenticated read/write object
}
```

The hot F' path should carry only this compact handle and prove the transition:

```text
handle_in + fresh claim + NIFS proof -> handle_out
```

The large authority relation must move out of the hot image into a separate
proof/relation that proves:

```text
handle_in  opens to the running accumulator consumed by NIFS.V
handle_out opens to the running accumulator produced by NIFS.V
```

This mirrors HyperNova's externalized-memory optimization: F' reads/writes a
small authenticated running-instance handle, while a memory/checking relation
proves that the handle corresponds to the real running instance.

## Candidate implementations

### Option A — Merkleized accumulator memory

Store the running CE claim payload in an authenticated memory tree. F' proves
read and write paths for the accumulator cells touched by one step.

Pros:

- Directly follows HyperNova's memory-checking suggestion.
- Makes the cost depend on `O(log N)` hashes rather than hashing the full
  `parent.c_data` vector in every F' step.
- Clear ownership: F' owns handle transition; memory relation owns opening
  correctness.

Cons:

- Poseidon2 Merkle paths are still bit-backed in the current low-norm image.
- Needs a careful cell layout so the path does not become another large trace.
- Requires an append/update memory semantics for the evolving CE parent.

### Option B — Dedicated accumulator-handle relation

Define a separate relation that proves `H(parent.c_data...) = handle` or proves
an equivalent commitment opening, then fold that relation separately from F'.

Pros:

- Minimal conceptual change to the existing accumulator digest.
- Keeps F' hot path small if the side relation is not embedded into every F'
  source image.

Cons:

- If implemented as the same bit-backed Poseidon2 trace, it only moves cost,
  not removes it.
- Needs a clear composition theorem: the verifier must check the side relation
  together with the F' relation, otherwise the handle is just a digest.

### Option C — Change the accumulator representation

Represent the running accumulator with a compact commitment that supports cheap
in-circuit update/opening under SuperNeo's low-norm constraints.

Pros:

- Potentially the largest speedup.
- Could avoid bit-backed hash traces over huge vectors entirely.

Cons:

- Biggest protocol change.
- Must preserve SuperNeo CE soundness and on-chain verification constraints.
- Requires a new security argument and substantial tests.

### Option D — Delayed accumulator-handle binding

Do not compute `new_acc_digest = H(parent.c_data...)` inside the same F'
step. Instead, carry `new_acc_digest` as a state-out value and require the
next recursive step to prove:

```text
state_in.acc_digest == H(actual running consumed by NIFS.V)
```

The current decider already has this next-step check in the recursive F' R1CS
path, and the terminal fold has the same check for the last step:

```text
next step:    acc_digest_in == NIFS.V.running_acc_digest
terminal:     last.state_out.acc_digest == terminal_fold.running_acc_digest
```

The full CE authority is separately carried by `children == next.running`
continuity and by the terminal CE closure. That means the same-step
`new_acc_digest` hash may be redundant for the current decider relation.

Pros:

- Removes the exact `5,406,336`-column trace without introducing a new hash,
  Merkle tree, or commitment backend.
- Keeps Poseidon2-only protocol binding.
- Matches the existing direction of the code: incoming accumulator handles are
  recomputed from authoritative NIFS wires; outgoing handles are consumed by
  the next step or terminal fold.

Cons / proof obligations:

- Literal HyperNova Construction 2 says F' outputs a hash over `U_{i+1}` after
  running `NIFS.V`; this option makes the `U_{i+1}` handle locally
  delayed rather than locally recomputed.
- It is only sound if every non-base state-out accumulator handle is consumed
  by either a next recursive step or the terminal fold, and if CE continuity
  pins the producer's `children` to that consumer's `running`.
- A standalone one-step F' relation would be weaker. The decider/lifecycle
  relation, not the isolated F' gadget, would own the delayed binding.

This is the implemented near-term column cut. It still needs the following
attack tests before it should be treated as a finished protocol change:

- Tamper a recursive step's `state_out.acc_digest` and assert the next-step
  `acc_digest_in` binding rejects.
- Tamper the last step's `state_out.acc_digest` and assert the terminal fold
  binding rejects.
- Keep the same-step accumulator Poseidon trace removed and assert the tamper
  tests still fail.
- Keep unified-mode source-image NIFS payload columns removed and assert the
  actual NIFS verifier messages still drive the accumulator authority checks.
- The layout budget gate now proves the canonical recursive image drops from
  `6,029,252` columns to `156,676` columns.

Current evidence:

- `system_lifecycle_f_prime_link::lifecycle_recursive_step_rejects_tampered_acc_digest_in`
  rejects a bad consumed accumulator handle.
- `system_lifecycle_f_prime_link::lifecycle_recursive_step_rejects_tampered_acc_digest_in_even_if_prior_x_out_rebuilt`
  rebuilds the prior `x_out` boundary after tampering the consumed handle;
  the step still rejects, so the recursive-link bit boundary is not the only
  guard.
- `system_lifecycle_f_prime_link::lifecycle_recursive_step_rejects_tampered_acc_digest_out_without_matching_x_out`
  rejects an outgoing handle that is not reflected in `state_x_out`.
- `system_decider_r1cs::decider_terminal_fold_rejects_tampered_last_acc_digest`
  emits the terminal NIFS.V circuit against an honest last-step handle and
  then against a one-byte-tampered handle; the tampered circuit is unsatisfied.
- `system_fibonacci_f_prime_layout_budget::fibonacci_f_prime_layout_budget_confirms_recursive_accumulator_trace_removed`
  checks that unified delayed-handle mode reserves zero source-image NIFS
  payload columns.

## Completed near-term path

Option D was the smallest path that removed the dominant trace without
adding a new proof system or a new protocol-binding hash. It introduced
the explicit `AccumulatorHandle` ownership boundary and moved the
producer-side accumulator hash out of the canonical F' source image.

Option D gives the needed column cut using
constraints already present in the decider:

```text
producer.children == consumer.running
consumer.acc_digest_in == H(consumer.running)
terminal.running_acc_digest == last.state_out.acc_digest
```

If those three checks cover every produced non-empty accumulator handle, the
same-step `H(parent.c_data...)` trace is a performance tax rather than a
soundness requirement for the composed lifecycle relation.

The first implementation slice cut the 5.4M-column accumulator trace at the
static layout level. The second cut removed the 87,552-column boundary-update
trace. The third cut removed the 21,888-column direct `z_0` absorb from
`state_x_out`: `z_0` is still pinned and linked as state, and its authority is
absorbed transitively through `vk_fs_digest` because
`z_0 = initial_boundary_digest(structure_digest, public_input_len)`.

## Next substantial cuts

The paper re-read narrows the useful optimization space:

- HyperNova Construction 2 wants `F'_j` to verify the prior NIFS fold,
  update the running instance, and output the next hash-chain value. It
  also explicitly suggests externalized memory for running instances when
  the carried state is large.
- SuperNeo §7 says the terminal CE relation still has to prove
  `c = L(Z)`, `x = L_in(Z)`, `||Z||_∞ < b`, and
  `y_j = M_j Z(r)`. Those checks can be optimized or moved into a
  compact terminal proof, but they cannot be replaced by a digest or by a
  Rust-only shortcut for compressed/on-chain consumers.

Ranked by expected impact:

### 1. Typed low-norm app variables

The current generic R1CS-F' image stores each app assignment variable as
a canonical 64-bit low-norm lane. That is conservative and simple, but
it is wasteful for Boolean-heavy circuits like SHA-256: a variable that
is already constrained to `{0,1}` should not consume 64 committed source
coordinates.

For the SHA fixture, `plan.limbs = m * 64 + 1`; with about `24.6k` R1CS
variables, that alone accounts for about `1.58M` source-image columns.
A bit-typed R1CS-F' frontend could store known-Boolean variables as one
source coordinate and only fall back to 64-bit canonical lanes for true
field-valued variables.

Current conservative evidence: a syntactic R1CS analyzer that only trusts
explicit rows equivalent to `v * (1 - v) = 0` finds `6,232 / 24,625`
SHA variables. The SHA R1CS-F' fixture now opts into a verifier-owned
typed app-private layout derived from that analyzer. The static layout
guard reports:

```text
typed app_private bits: current=1576000 typed=1183384
```

This is a real cut because the SuperNeo/Ajtai path pays for committed
source coordinates, not abstract R1CS variables. It is still deliberately
conservative: it only narrows variables proven Boolean by explicit R1CS
rows. It is not the order-of-magnitude reduction we would get from assuming
every SHA-internal value is Boolean. A stronger cut needs frontend-provided
type metadata or a richer analysis of variables that are linear functions of
Boolean values.

This is the closest local analogue to the Jolt inspiration: do not
generate one wide low-norm encoding per abstract variable when the
underlying constraint system tells us a narrower representation is
authoritative. The SuperNeo commitment still pays per committed
coordinate, so this is not just row sharing; it must reduce witness
coordinates.

### 2. Pack app public inputs before `state_x_out`

R1CS-F' currently appends every `app_public_input_var_indices` entry to
the `state_x_out` Poseidon2 preimage as its own field lane. For SHA-256,
the public input is the one-slot constant plus 256 digest bits. Absorbing
those bit variables one-by-one makes the chain hash trace much wider
than necessary.

A better shape is:

```text
public input bits -> packed 64-bit lanes -> state_x_out preimage
```

with in-circuit linear packing rows tying the packed lanes to the actual
public input bits. For the SHA path, this should replace roughly 257
Poseidon2 preimage lanes with about 5 packed lanes. It preserves the
same authority (`state_x_out` still binds the public input), but cuts
many bit-backed Poseidon2 permutations.

Current status: implemented for the SHA Bellpepper R1CS-F' fixture as an
opt-in verifier-owned plan mode. The static layout guard reports:

```text
state_x_out lanes: packed=24 vs old_full=280
```

The compact chain-state prefix is 19 lanes after removing the direct
`z_0` absorb; `z_0` is verifier-derived from values already absorbed by
`vk_fs_digest`. The one constant plus 256 SHA digest bits now occupy 5
packed semantic-state lanes instead of 257 individual lanes. This is
deliberately not enabled for generic R1CS plans unless the verifier-owned
plan declares the public variables Boolean and the app circuit constrains
them to `{0,1}`.

### 3. Stream or compact terminal finalization

After the F' source-image cut, the SHA two-step run spends more time in
`chain.finish()` than in the recursive append. That terminal fold is a
real SuperNeo obligation, not a fake verifier step: it folds the trailing
`latest` into the running CE accumulator and must process the final
`K + k_rho` claims.

Near-term work should inspect `prove_final_fold` / `nifs::prove` /
`optimized_prove` for avoidable materialization and cloning, especially
large witness matrices and zero-heavy digit tables. Longer-term work is a
compact terminal CE proof for compressed/on-chain consumers, but the
uncompressed path still needs the direct SuperNeo terminal equations.

Current status: the first memory slice is implemented. `paper::nifs::prover`
now splits fresh instances by move, then threads borrowed witness matrices
into Π_RLC via the borrowed `rlc_with_commit_refs` path. That removes the
previous clone of `running.witnesses` and avoids cloning fresh `Z` matrices
just to build the K+k Π_RLC witness array. This is a residency improvement,
not a proof-equation change: Π_CCS/Π_RLC/Π_DEC still process the same claims
and witnesses under the same transcript.

Validation:

```text
cargo check -p neo-fold-clean --release --tests
cargo test -p neo-fold-clean --release --test reductions_nifs_v
cargo test -p neo-fold-clean --release --test system_lifecycle_f_prime_link
```

### 4. Smaller `state_x_out` hash without dropping authority

For tiny Fibonacci, the remaining 156k-column image is almost entirely
the `state_x_out` Poseidon2 trace. The unsafe shortcuts are already
rejected: do not drop `z_0`, `vk_fs`, semantic state, or accumulator
state from the folded public link. Any further cut has to preserve the
Construction-2 hash-chain authority.

Two plausible directions remain:

- reduce the number of absorbed lanes by packing fields that are already
  bit-level data (counters, public input bits, typed app state), or
- change the low-norm trace representation for Poseidon2 itself.

The second is a protocol/circuit-design project. Poseidon2's internal
state is not low-norm after each round, so we cannot simply "check it
stays low norm" after every operation. The current bit-backed trace pays
the cost precisely because arbitrary field-state transitions must be
represented with low-norm committed coordinates.

### 5. Paper-grounded next cuts after the 1.03s SHA run

Re-reading HyperNova §6.1/§6.3 and SuperNeo §2.3/§7 points to three
directions that are still aligned with the papers and large enough to matter.

1. **Externalize or handle the running accumulator more aggressively.**
   HyperNova §6.3 explicitly recommends verifiable read/write of running
   instances from externalized memory instead of passing the whole running
   instance through every `F'_j`. The local delayed-handle work removed the
   largest producer-side trace, but the remaining `state_x_out` trace still
   hashes a concrete accumulator handle in every step. A deeper version would
   make the recursive step prove a memory read/write for the running CE object
   and hash only the memory root plus the step-local delta. This is the
   largest plausible F' width cut, but it needs new in-circuit memory
   constraints and negative tests proving that the handle is not authority.

2. **Keep terminal DEC witnesses in a digit/packed representation longer.**
   SuperNeo §2.3 says pay-per-bit work should add rotations for small digits,
   not repeatedly materialize dense ring products. The current terminal
   `Π_DEC` already skips all-zero digit planes and uses digit-aware ring
   evaluation, but it still materializes `k_rho` full `Mat<F>` witnesses from
   `split_b`, then re-packs nonzero children into `SuperneoZBlocks` for child
   `y_ring` evaluation and later folds. A real next cut is to let
   `split_b` produce a compact digit-plane object that can feed:
   commitments, `X` projection, `y_ring`, `y_zcol`, and the next running
   witness surface without round-tripping through dense matrices. This is
   still exactly `Π_DEC`; it only changes witness representation.

3. **Make preprocessing a reusable verifier-owned artifact.** The SHA timing
   still spends about `0.22s` in R1CS-F' preprocessing, with
   `OptimizedStructureCache::build` doing Poseidon2 matrix digest and
   SuperNeo cache construction once per test. In real deployment this should
   be keyed by the verifier-owned `(params, structure_digest, plan)` and
   reused across folds/proofs for the same circuit. The protocol digest must
   remain Poseidon2-only and verifier-derived; the optimization is ownership
   and artifact lifetime, not trusting prover cache.

One tempting but rejected shortcut is truncating the `Π_CCS` NC
`eq_beta_m` table to live columns. Padded columns have zero digit values, but
the multilinear extension still needs boundary padded-subtree `eq` values in
later rounds. Dropping them without a derived replacement changes the
sumcheck polynomial, so this is not a safe local optimization.

## Success criteria

Near-term:

- A static layout test shows the F' image no longer includes the
  `5,406,336`-column recursive accumulator trace. **Current status: done.**
- Negative tests prove the delayed handle is not a self-consistent digest
  shortcut. **Current status: recursive consumer-side and terminal-fold
  consumed-handle tests pass; `system_lifecycle_f_prime_link` and the
  two-step Fibonacci wall-clock path pass.**
- Native and in-circuit F' agree on the handle transition. **Current status:
  compile and targeted parity tests pass; full protocol review still needed.**

Long-term:

- The recursive Fibonacci R1CS-F' append/fold path is measured under the real
  proving command at `<= 1s` per fold on the target machine. **Current local
  two-step fixture: `0.26s` for the recursive append.**
- Verification does not rely on prover-only caches.
- The design remains compatible with Poseidon2-only protocol binding paths.
