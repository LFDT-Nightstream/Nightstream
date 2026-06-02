# Folding 2s viability notes

Date: 2026-06-01
Scope: `neo-fold-clean` R1CS-F' folding performance and soundness constraints

Historical note: the filename still says "subsecond" because this note began
as a one-second push. The current product viability bar is **2s or less** for
the relevant folding path. One-second numbers below are stretch-target history,
not the merge bar.

## Bottom line

The current dirty worktree is no longer in the old "multi-second per F' fold"
state for the measured production-shaped R1CS-F' path. On this machine, the
SHA-256 packed-state serial-quad benchmark is comfortably below the current
2s bar even when verifier-owned setup is included.

Fresh 2026-06-01 release run, excluding compilation:

```text
cargo test -p neo-fold-clean --release --features perf-timers \
  --test system_sha256_bellpepper_batching \
  sha256_production_core_serial_quad_packed_state_four_transitions_perf_snapshot \
  -- --ignored --exact --nocapture

setup total                      0.525s
append total                     0.194s
finish                           0.290s
verify                           0.178s
online prove                     0.484s
online prove+verify              0.662s
setup+online total               1.200s
```

The previously observed 6M-column / minute-plus Fibonacci recursive-step
fixture is obsolete in the current tree. The current ignored Fibonacci
R1CS-F' fixture uses the compact delayed-handle shape and finishes below one
second. The gate should be a real folding benchmark that
proves/folds/verifies the statement shape we expect to use.

| Fixture | Setup | Append/prove | Finish | Verify | Total |
|---|---:|---:|---:|---:|---:|
| Fibonacci R1CS-F' two-step | `0.08s` | `0.03s base`, `0.16s recursive` | `0.11s` | `0.10s terminal` | `0.64s` |
| SHA-256 packed-state serial-quad | `0.52s` | `0.19s` | `0.29s` | `0.18s` | `1.20s` |

The important split is:

```text
SHA-256 serial-quad online prove        = append + finish = 0.484s
SHA-256 serial-quad online prove+verify = 0.662s
SHA-256 serial-quad setup+online        = 1.200s
```

This proves the current measured SHA folding path meets the 2s viability bar
with margin. It does **not** prove arbitrary statements, the old K=1 repeated
append benchmark, obsolete pre-compact Fibonacci shapes, or future
compressed/on-chain wrapping meet the same bar. Those need separate numbers.

Current pass/fail matrix for the 2s wall-clock target:

| Path | Current result | Status | What it proves |
|---|---:|---|---|
| Fibonacci R1CS-F' two-step | `~0.6s` cold | PASS | Tiny app + full R1CS-F' path is no longer multi-second. |
| Serial-K2 Fibonacci, four app transitions | `0.48s` cold | PASS | Paper-safe serial batching inside one fresh CCS instance cuts append count without weakening SuperNeo. |
| SHA-256 packed-state serial-quad, four transitions | `1.200s` latest cold, `0.662s` online+verify | PASS | This is the current production-shaped viability benchmark. |
| SHA-256 serial-quad, same anchored prepared key amortized | `0.639s/proof` | PASS | Reusing a verifier-owned prepared artifact for the same anchored statement removes setup from recurring proof cost. |
| SHA-256 ordered-independent batch-4 | `1.060s` cold | PASS | Independent-batch throughput is viable under 2s, but it proves a different public statement than serial SHA. |
| SHA-256 old K=1 five-step chain | `2.14s` cold | MISS/BORDERLINE | Repeating full append machinery per transition is still the wrong shape; use serial mini-chain chunks instead. |
| Old 6M-column Fibonacci compiler stress shape | superseded | REMOVED/OBSOLETE | The current fixture is the compact shape above; do not use old screenshots as current evidence. |

The matrix is deliberately split by statement semantics. "Serial SHA" and
"ordered independent SHA batch" are different public statements; faster numbers
from one must not be used to justify the other.

Current 2s gate commands:

```bash
cargo test -p neo-fold-clean --release \
  --test system_fibonacci_compiler_unified_structure \
  compiler_two_step_chain_builds_from_scratch_and_rejects_terminal_only \
  -- --ignored --exact --nocapture

cargo test -p neo-fold-clean --release --features perf-timers \
  --test system_sha256_bellpepper_batching \
  sha256_production_core_serial_quad_packed_state_four_transitions_perf_snapshot \
  -- --ignored --exact --nocapture
```

Treat compilation time separately. The gate is the test-reported proof path:
preprocess/setup, append, finish, and verify. Both commands currently pass
below 2s on this machine.

The next substantial improvements should be pursued only if we need more
margin or a larger statement shape. They should not start by weakening F'
state binding or terminal CE authority. The fresh profile still shows real
protocol kernels: Ajtai commitments, Poseidon2 permutations, SuperNeo
digit/ring evaluation, and preprocessing cache construction. The paper-grounded
levers are:

1. **Make preprocessing a reusable artifact.** HyperNova's verifier key already
   owns the structures; the implementation should not rebuild
   `OptimizedStructureCache`, matrix digests, or derived plan data per proof.
   On the serial-quad SHA snapshot this is the difference between
   `~0.49s` online and `~1.00s` setup+online.
2. **Reduce the remaining chain-state Poseidon2 trace.** For tiny Fibonacci,
   `state_x_out` is still almost the entire F' image. Only packing/verifier-
   derived fields are safe cuts; dropping HyperNova state fields without an
   equivalent check is not.
3. **Keep SuperNeo terminal witnesses in compact digit form longer.** SuperNeo
   says small digits should pay by rotation/addition, not dense ring products.
   Terminal `Π_DEC` and CE evaluation still show digit/ring work in profiles.
4. **Batch more work per F' chunk when the app semantics allow it.** The
   serial-quad SHA fixture proves four SHA transitions in one F' append and is
   the first snapshot that reaches the cold one-second boundary.
5. **Only then chase kernel micro-optimizations.** Ajtai and SuperNeo kernels
   remain hot, but they are now split across phases; shaving them is useful,
   not the main architectural unlock.

## Paper contract

HyperNova Construction 2 says the recursive F' link must bind:

- `vk_fs`
- the step counter `i`
- `z_0`
- the current state `z_i`
- the running accumulator `U_i`
- the program counter `pc_i`

The implementation may externalize `U_i`, but only with verifiable read/write
or another authoritative opening/check. A digest is compression, not authority.

SuperNeo Definition 13 says each terminal CE claim must satisfy:

- `c = L(z)`
- `x = L_in(z)`
- `||z||_infty < b`
- `y_j = M_j z(r)` for every CCS matrix

This codebase stores `ct` as a denormalized field, so both native and circuit
paths must also enforce `ct == constant_term(y_ring)`.

Any performance work that changes those obligations is not an optimization; it
is a protocol change.

## Current soundness posture

The current implementation keeps the important authority checks in the right
places:

- `state_x_out` native, circuit, and planner use the compact
  `F_PRIME_STATE_X_OUT_DOMAIN` preimage in parity.
- The Construction-2 accumulator handle is full-running:
  `AccumulatorHandle::from_running_parts(claims, parent_authority)`, not just
  parent `c.data`.
- The F' recursive step enforces
  `acc_digest_in == nifs_outputs.running_acc_digest`.
- The terminal fold enforces the last F' `acc_digest` equals the terminal
  NIFS input running handle.
- The terminal CE relation checks commit opening, X projection, low norm,
  y-ring evaluation, and ct consistency.
- Packed public-input mode now requires one-bit, Boolean-constrained variables.
- The R1CS conventional constant lane is pinned when typed/packed Boolean
  assumptions rely on it.

Targeted checks run on 2026-06-01:

```text
cargo test -p neo-fold-clean --release --test f_prime_digest_circuit
  12 passed

cargo test -p neo-fold-clean --release --test system_phase_1_3d_step_parity
  3 passed

cargo test -p neo-fold-clean --release --test system_lifecycle_f_prime_link
  10 passed

cargo test -p neo-fold-clean --release --test system_r1cs_preprocess
  15 passed

cargo test -p neo-reductions --release --test superneo_eval_equivalence
  14 passed

cargo test -p neo-reductions --release --test rlc_dec_k_gt1
  10 passed
```

These are targeted, not a full proof of the entire branch.

## Current shape evidence

Fibonacci R1CS-F':

```text
structure.n = 2327
structure.m = 134788
state_x_out Poseidon trace = 131328 columns
Poseidon share = 97.4%
```

SHA-256 production-core R1CS-F':

```text
serial-quad packed-state fixture:
structure.n = 107278
structure.m = 1417418
plan.limbs = 1085641
app constraints = 101866
app vars = 101139
```

For Fibonacci-like tiny apps, Poseidon trace dominates. For SHA-sized apps,
both app witness width and terminal SuperNeo work matter.

## Current profiler evidence

Fresh continuation run, still on 2026-06-01, with `perf-timers` enabled:

```text
cargo test -p neo-fold-clean --release --features perf-timers \
  --test system_sha256_bellpepper_batching \
  sha256_production_core_serial_quad_packed_state_four_transitions_perf_snapshot \
  -- --ignored --exact --nocapture

setup plan/structure     0.225s
prepare cache            0.262s
preprocess               0.011s
append total             0.130s
finish                   0.227s
verify                   0.109s
online prove             0.357s
online prove+verify      0.466s
setup+online total       0.974s
```

Prepared-key amortization on three same-anchor serial-quad proofs:

```text
one-time setup total       0.520s
per-proof online totals    0.489s, 0.468s, 0.490s
amortized setup+online     0.656s/proof
```

These numbers are the current best evidence for the near-term target:

- recurring folding for the measured SHA serial-quad shape is comfortably
  below one second when verifier-owned preprocessing is reused;
- cold single-proof setup+online can land below one second, but only with a
  thin margin and noticeable run-to-run noise;
- the remaining recurring cost is split across real protocol kernels:
  R1CS-F' append (`~130ms`), terminal NIFS finish (`~225ms`), and terminal
  verifier witness authority (`~110ms`).

Soundness interpretation: the prepared artifact is part of the verifier key
for the same structure and initial semantic-state anchor. It is not a
prover-provided cache and it is not reusable across arbitrary initial states.

Fresh release snapshot, same machine/worktree:

```text
cargo test -p neo-fold-clean --release \
  --test system_sha256_bellpepper_batching \
  sha256_production_core_serial_quad_packed_state_four_transitions_perf_snapshot \
  -- --ignored --exact --nocapture

setup plan/structure     0.225s
prepare cache            0.272s
preprocess               0.011s
append total             0.137s
finish                   0.229s
verify                   0.120s
online prove             0.365s
online prove+verify      0.485s
setup+online total       1.002s
```

Three immediate repeats of the same release snapshot show why this should be
treated as a borderline miss rather than a clean pass:

```text
setup+online total, ms: [1027.410, 981.576, 972.371]
min = 972.371 ms
median = 981.576 ms
max = 1027.410 ms
```

The path can dip under one second, but it does not yet have enough margin to
call the cold target reliably achieved.

Later same-day snapshot after the seeded-Ajtai dispatch fix and red-team test
additions, with the same command and `perf-timers` enabled:

```text
setup plan/structure     0.199s
prepare cache            0.220s
preprocess               0.009s
setup total              0.428s
append total             0.131s
finish                   0.218s
verify                   0.110s
online prove             0.349s
online prove+verify      0.459s
setup+online total       0.895s
```

This is the strongest current cold-path evidence, but not a reason to stop:
the margin is still not large, and the profile still shows real protocol
kernels rather than one obvious serial bug.

Fresh continuation snapshot after fusing the terminal witness shape/low-norm
scan in `verify_uncompressed`:

```text
cargo test -p neo-fold-clean --release --features perf-timers \
  --test system_sha256_bellpepper_batching \
  sha256_production_core_serial_quad_packed_state_four_transitions_perf_snapshot \
  -- --ignored --exact --nocapture

setup plan/structure     0.220s
prepare cache            0.263s
preprocess               0.011s
setup total              0.494s
append total             0.130s
finish                   0.224s
verify                   0.131s
online prove             0.353s
online prove+verify      0.485s
setup+online total       0.988s
```

The scan fusion is equation-preserving and keeps the verifier obligations
unchanged, but it did not create enough visible margin to call the cold target
done. The verifier-side timer for this run was:

```text
[verify] witness authority (shared-r parallel, claims=14):
  forms 0.03s, commit 0.06s, project 0.00s, norm 0.03s, ce 0.06s
```

Interpretation: verifier time is now dominated by the required terminal
authority checks themselves, not by an obviously duplicated Rust pass. The next
large improvement should reduce the amount of terminal work or make Ajtai /
SuperNeo kernels cheaper while preserving the same checks.

Profiling command:

```text
./scripts/profile_for_ai.sh neo-fold-clean \
  system_sha256_bellpepper_batching \
  sha256_production_core_serial_quad_packed_state_four_transitions_perf_snapshot \
  --ignored --features perf-timers 2
```

The profiling profile is slower (`1.59s` total for the same snapshot) because
it builds with debug info and is sampled for symbols; use it for hotspot
ordering, not release baselines.

Top hot functions in the SHA serial-quad profile:

1. Poseidon2 assembly permutations
2. `neo_ajtai::s_module::commit_signed_unit_row_many_chunk`
3. `rand_core::block::BlockRng::generate_and_set`
4. `rayon::slice::sort::recurse` / insertion sort
5. `neo_reductions::superneo_eval::digit::apply_monomial_pair_in_place`
6. `neo_ajtai::commit::accumulate_binary_mask_sparse`
7. `neo_ccs::sparse::CscMat::from_triplets`
8. `neo_reductions::superneo_eval::SuperneoMatrixCache::eval_mle_ring_with_blocks_split_chi_scratch`
9. `neo_reductions::common::compute_y_zcol_from_witness`

This is a good sign and a constraint. The hottest code is arithmetic inside
Ajtai commitment, Poseidon2, and SuperNeo CE evaluation, not a giant unused
loop. It also means "more Rayon" alone is unlikely to be the architectural
answer: the current one-second miss is split across verifier-key setup and
real online kernels, while online prove+verify is already around half a second
for this batched SHA fixture.

Arithmetic-only kernel changes kept so far:

- `superneo_eval::digit`: remove zero-coefficient branches inside the hot
  monomial application loops. Same ring indices and signs; just adds/subtracts
  zero instead of branching.
- `neo_ajtai::s_module`: specialize the signed-unit `delta == 1` column
  rotation with an in-place Φ₈₁ step. The seeded/materialized PP equivalence
  tests pass, so the commitment map and randomness schedule are unchanged.
- `neo_ajtai::s_module`: schedule large signed-unit `commit_many` as
  independent `(kappa row, PP chunk)` tasks instead of only `kappa` long row
  tasks. Each task uses the same deterministic chunk seed as before and the
  partial commitments are added in `Fq`, so this changes work scheduling, not
  the commitment relation.
- `neo_ajtai::s_module`: keep seeded sparse binary/signed-unit commit paths in
  front of dense materialized-PP multiplication even when the seeded PP has
  been loaded into the global registry. For the SHA serial-quad dimensions
  (`D=54`, `κ=18`, `m=26249`, `14` signed-unit claims), the sparse global path
  measured `~39ms` median before loading the PP and `~37ms` median after
  loading it, while an owned dense materialized module measured `~1.98s`
  median. The outputs are asserted equal; this is dispatch hygiene, not a
  protocol change. The same SHA serial-quad snapshot after this guard measured
  `0.965s` cold setup+online and `0.447s` online prove+verify.
- `neo_reductions::optimized_engine::rlc`: exploit the fact that Π_RLC
  challenges are validated rotation matrices. For wide witness matrices,
  `rho * Z` is computed as ring multiplication in `F[X]/Phi_81` per packed
  column instead of a generic dense `D x D` matrix-vector multiply. The new
  `rlc_with_commit_sampled_rotation_rhos_matches_public_z_mix` test compares
  the optimized path against the generic `left_mul_acc` reference on sampled
  non-diagonal rotation challenges.
- `neo_reductions::optimized_engine::rlc`: add a sparse RHS path for the same
  validated-rotation multiply. When a packed witness column has only a few
  nonzero digits, `rho * column` is computed as a small sum of monomial
  rotations of `rho`; dense columns still use the Toom/Karatsuba path. The
  `rlc_with_commit_sparse_rotation_rhs_matches_public_z_mix` test forces this
  branch and compares against generic `left_mul_acc`.

Two tempting Rayon tweaks were tested and rejected:

- Parallelizing signed-unit column-plan construction made the SHA snapshot
  slower (`0.24s/op -> 0.26s/op` recursive append), so it was reverted.
- Raising the b=2 NC column-phase parallel threshold from `8192` to `16384`
  was neutral, and `32768` was slower, so the original threshold was restored.
- Sampling only active Ajtai signed-unit columns is not attractive for the
  serial-quad SHA shape: the representative 14-claim workload has
  `25739 / 26249 = 98.1%` active columns, `0` empty 32-column batches, and
  `443 / 821` completely full batches. A tiny full-active branch was also
  tried in `commit_signed_unit_row_many_chunk`; it preserved outputs but
  benchmarked slightly worse/noisier (`~39ms` median versus the previous
  `~38ms` median), so it was reverted. The remaining Ajtai work is real
  arithmetic/randomness, not mostly wasted inactive-column sampling.

## Ranked next work

### 1. Preserve current baselines

Before changing more hot code, keep the current perf snapshots as regression
guards. The old 12s/25s Fibonacci numbers should not silently return.

Useful commands:

```bash
cargo test -p neo-fold-clean --release \
  --test system_fibonacci_compiler_unified_structure \
  compiler_two_step_chain_builds_from_scratch_and_rejects_terminal_only \
  -- --ignored --exact --nocapture

cargo test -p neo-fold-clean --release \
  --test system_sha256_bellpepper_ccs \
  sha256_production_core_bellpepper_ivc_chain_five_steps_perf_snapshot \
  -- --ignored --exact --nocapture
```

### 2. Optimize Ajtai signed-unit batching

Best target: `crates/neo-ajtai/src/s_module.rs`.

Current hot path:

- `commit_packed_signed_unit_column_bits_many`
- `commit_signed_unit_row_many`
- `commit_signed_unit_row_many_chunk`
- `accumulate_signed_unit_column_plan`

Safe optimization shape:

- keep the same seeded PP stream and column order;
- keep commitment arithmetic byte-for-byte equivalent to materialized PP;
- reduce repeated per-column mask/plan work;
- tune chunk-local planning and Rayon thresholds;
- add/keep equivalence tests against materialized PP.

Do not change the commitment map or the randomness schedule.

### 3. Optimize SuperNeo digit kernels

Best target: `crates/neo-reductions/src/superneo_eval/digit.rs`.

Current hot path:

- `accumulate_pair_by_digit_block`
- `apply_monomial_pair_in_place`

Safe optimization shape:

- precompute nonzero coefficient positions for ring-linear forms;
- skip zero coefficient lanes without changing ring multiplication semantics;
- keep equivalence tests against direct matrix-vector evaluation;
- test real-only and complex/imaginary block cases.

This is aligned with SuperNeo: it speeds up `M_bar z(r)` evaluation without
weakening the CE relation.

### 4. Use SuperNeo K-batching where the state semantics allow it

Best target: `crates/neo-fold-clean/src/frontends/r1cs_f_prime/lifecycle.rs`
and the stateful frontend contract.

The paper-level lever for reducing wall time is not only making one fold
faster; it is reducing how many folds are needed. SuperNeo/HyperNova support
folding a chunk of K fresh instances into the running accumulator. This code
already has a K-aware chunk path, but the SHA app-public/stateful path
intentionally rejects K>1 because one chunk currently has only one outgoing
semantic-state digest.

Current evidence:

- `R1csChainBuilder::append_chunk` is K-aware.
- Tests explicitly reject SHA K=2/K=4 chunks.
- Recursive SHA append at K=1 is about `0.26s/op`; four recursive appends plus
  finish/verify put the five-step snapshot around `1.69s`.

Paper-derived constraint:

- HyperNova multi-folding folds a vector of input instances into one running
  instance; it does not automatically make a serial mini-chain.
- SuperNeo's K fresh inputs in Π_CCS are K CCS claims rooted at the same
  current accumulator point. A K chunk is sound only if the public statement
  says exactly what those K inputs mean.
- For a state machine, "step j+1 consumes step j's output" is an application
  relation. It must be represented inside the chunk circuit if we want K>1
  to mean K serial iterations.

Sound next shape:

- keep K=1 for any stateful app-public path until the chunk has a
  verifier-owned aggregate semantic output;
- for stateless or privately-state-threaded circuits, measure K=2/K=4 chunks
  directly;
- if SHA needs K>1, design the aggregate semantic-state rule first, then add
  tests that show every intermediate app state is still constrained.

The viable designs are:

1. **Independent public-batch mode.** A K-chunk proves K independent app
   statements rooted at the same prior Construction-2 state. The public
   statement is a Poseidon2 digest of the ordered K public outputs. This is
   sound for "prove this batch of SHA preimages" but it is not a serial state
   machine.
2. **Serial mini-chain mode.** One F' chunk internally carries K app states
   and proves `state_{j+1} = F(state_j)` for every substep. The chunk's
   semantic output is the final substep state. This preserves IVC semantics,
   but it requires a different R1CS-F' image shape with K copies of the app
   witness or a folded in-chunk app verifier.
3. **External app-output log.** Keep the Construction-2 semantic state as the
   serial state machine output, and add a separate verifier-owned digest of all
   app-public outputs in the chunk. This can expose every SHA output without
   pretending the outputs are the state transition itself.

Recommendation: implement option 1 first as a separate API/name if the goal is
throughput for independent SHA statements. Use option 2 only if the application
really needs "iteration N feeds iteration N+1" semantics inside the chunk.
Do not retrofit either behavior behind the existing stateful path without
changing the public statement and tests.

### 4a. What the current code already supports

The current code-level split is useful:

- `R1csChainBuilder::append_chunk` and `compile_chunk` are K-aware.
- The lifecycle layer can fold K fresh CCS instances as one SuperNeo chunk.
- The decider already has a varying-size batch test (`[2, 3]`) proving the
  replay path handles "previous chunk K" and "current chunk K" separately.
- `verify_prior_fold` is K-bound through the chunk digest; a proof prepared
  for K=1 rejects when replayed as K=2.
- The SHA/R1CS-F' stateful path rejects K>1 via
  `StatefulChunkMustBeSerial`, because there is only one outgoing semantic
  digest for the whole chunk.

That last bullet is the soundness boundary. Removing the rejection would make
the benchmark faster but would lie about the statement: multiple SHA public
outputs would collapse into one state coordinate.

### 4b. Minimal sound implementation plan for true serial K chunks

The highest-leverage path to sub-second serial SHA/Fibonacci traces is a new
R1CS-F' chunk image whose application sub-circuit is itself K-wide:

```text
chunk input semantic state:    s_0

for j in 0..K:
  prove app assignment A_j satisfies the app R1CS
  prove semantic_in(A_j)  == s_j
  prove semantic_out(A_j) == s_{j+1}

chunk output semantic state:   s_K
state_x_out absorbs s_K
fresh CCS public input still encodes the prior F' x_out
```

Concrete code surface:

- Add a separate plan/compiler path rather than weakening
  `semantic_state_digests_for_inputs`.
- The new plan must represent K copies of the app assignment region or a
  repeated app verifier region inside one F' image.
- `semantic_state_digests_for_inputs` should keep rejecting K>1 for the
  existing single-image path.
- The new compiler must emit in-circuit Poseidon rows for every
  `semantic_in_j` and `semantic_out_j`, plus equality rows
  `semantic_out_j == semantic_in_{j+1}` for `j < K-1`.
- `R1csChainBuilder` can then deposit one fresh CCS instance per chunk
  (the K-wide chunk image), or K fresh instances only if each fresh image
  independently carries its own substep and the chunk-level semantic output is
  separately constrained. The former is easier to audit.
- The public statement should expose either only `s_K` (serial state machine)
  or a Poseidon2 log digest of all intermediate public outputs if users need
  every SHA output.

Required negative tests:

- K=2 serial chunk accepts when `state_out_0 == state_in_1`.
- Tampering only the intermediate state link rejects in the F' structure.
- Tampering the first substep's app public output changes the chunk output/log
  digest or rejects.
- Tampering the second substep's app public output changes the final semantic
  output or rejects.
- A K=1-prepared prior fold still rejects when replayed as K=2.
- Existing single-image SHA K=2/K=4 rejection tests remain in place until the
  new path has a distinct API/name.

Expected performance effect:

- K=2 should roughly replace two recursive appends with one larger append. It
  will not be exactly 2x because the app image grows, but it removes one
  terminal-facing NIFS cycle and one `prepare_next_fold`/compile cycle.
- K=4 is the likely route for "five SHA steps under one second" if the K-wide
  image remains much cheaper than four separate F' images.
- If the K-wide image's Poseidon/state rows grow linearly with K and dominate,
  the next lever becomes reducing the number of per-substep Poseidon traces,
  not changing the SuperNeo fold.

Why this is the likely route to <1s for the five-step SHA snapshot:

- At K=1, four recursive appends alone cost about `1.03s` in the fresh
  2026-06-01 snapshot.
- A sound K=2 serial mini-chain would reduce four recursive appends to roughly
  two chunk appends, even if each chunk is somewhat larger.
- A sound K=4 chunk would reduce the recursive append count further, but only
  if the chunk image proves all four internal app transitions and exposes the
  right public/semantic output.
- This is a structure change, not a thread-count tweak. It must come with
  negative tests proving that missing or tampered intermediate states are
  rejected.

Bad direction:

- enabling K>1 by hashing only the final public state while ignoring the
  intermediate app-public outputs;
- treating a chunk digest as authority for the missing intermediate state
  checks.

### 4c. Implementation slices for serial K chunks

This should not start by weakening `R1csChainBuilder::append_chunk`. That API
already does the right conservative thing for the existing single-image path.
The safer sequence is:

Current ownership map:

- `frontends/f_prime/image.rs` owns the source-image regions. Today there is
  one `app_private` region and one state-in/state-out envelope.
- `frontends/f_prime/recursive_plan.rs` owns the Poseidon preimage sources for
  one `state_x_out` and one semantic input/output pair.
- `frontends/r1cs_f_prime/structure.rs` appends exactly one copy of the app
  R1CS product rows to the shared F' shell.
- `frontends/r1cs_f_prime/compiler.rs::semantic_state_digests_for_inputs`
  rejects K>1 because it can derive only one semantic input/output digest from
  the current single-image shape.

That means true serial K support is a new image shape, not a flag on the
existing shape.

1. **Add a K=2 Fibonacci-only prototype image.**
   - New plan type, separate from the current `RecursiveStepImagePlan`.
   - Two copies of the tiny Fibonacci app assignment region.
   - In-circuit equality rows:
     `semantic_out_0 == semantic_in_1`.
   - Chunk output digest is `semantic_out_1`.
   - Keep the current `StatefulChunkMustBeSerial` rejection for the generic
     R1CS-F' compiler.

2. **Route the prototype as one fresh CCS instance.**
   - The SuperNeo fold sees one larger fresh CCS claim, not two independent
     app claims pretending to be a serial trace.
   - This is simpler to audit: one chunk image proves both app transitions and
     exposes one final semantic state.

3. **Add adversarial tests before measuring speed.**
   - Honest K=2 Fibonacci serial chunk verifies.
   - Flip only the intermediate link (`out_0 != in_1`) and assert the F'
     structure rejects.
   - Flip only the first app assignment while preserving the second and assert
     the final chunk state/log changes or the structure rejects.
   - Flip only the second app assignment and assert the final semantic state
     changes or the structure rejects.
   - Replay a prior fold prepared for K=1 as K=2 and assert rejection.

4. **Only then generalize to SHA.**
   - SHA has a much wider app region, so the K-wide image may grow enough that
     K=4 is not automatically a win.
   - Measure K=2 first. If K=2 is clearly below the extrapolated two-step
     K=1 cost, proceed to K=4.

5. **Keep public output semantics explicit.**
   - If the public statement is "final state after K serial transitions", then
     expose only `s_K`.
   - If users need every intermediate SHA digest, add a separate Poseidon2 log
     digest over `(out_0, ..., out_{K-1})` and bind that log inside the chunk.
   - Do not overload the Construction-2 semantic state to mean both "current
     serial state" and "bag of public outputs".

This gives us a clean acceptance gate for the first real sub-second attempt:

```text
K=2 serial Fibonacci:
  soundness tests pass
  per-two-transition wall time < two K=1 appends

K=2 serial SHA:
  same soundness tests adapted to SHA
  five-step snapshot < current 1.69s

K=4 serial SHA:
  only after K=2 proves the image-growth curve is favorable
  target: five-step snapshot < 1.0s
```

If the prototype fails because the K-wide image grows nearly linearly with K,
the next optimization target is not "more batching"; it is reducing the
per-substep Poseidon/state machinery inside that K-wide image.

### 4d. First serial-K prototype landed

There is now a deliberately small proof-of-concept for the paper-safe batching
shape:

```text
tests/system/r1cs_compiler_stateful.rs
  r1cs_stateful_serial_k2_fibonacci_step_satisfies_and_binds_intermediate_link
  r1cs_stateful_serial_k2_fibonacci_chain_verifies_two_chunks
```

It does **not** remove the existing `StatefulChunkMustBeSerial` guard. Instead
it builds one larger app R1CS whose public/semantic assignment contains:

```text
(a_0, b_0) -> (a_1, b_1) -> (a_2, b_2)
```

with app-R1CS rows enforcing:

```text
a_1 = b_0
b_1 = a_0 + b_0
a_2 = b_1
b_2 = a_1 + b_1
```

The F' semantic input is `(a_0,b_0)` and the F' semantic output is
`(a_2,b_2)`. This means one F' fold represents two serial Fibonacci
transitions without pretending that two independent SuperNeo fresh inputs are
one serial trace.

The direct structure test flips only the intermediate `a_1` witness bit and
checks that an appended app-R1CS row rejects. The lifecycle test appends two
serial-K2 chunks:

```text
(1,1) -> (2,3)
(2,3) -> (5,8)
```

and verifies both `verify_uncompressed_audit` and non-replay
`verify_uncompressed`. This is not yet the general SHA K-wide image, but it is
evidence that the safe route is implementable using the current proof stack:
make the app relation serial inside one fresh CCS instance, then fold that
larger instance normally.

Targeted validation:

```text
cargo test -p neo-fold-clean --release --test system_r1cs_compiler_stateful \
  r1cs_stateful_serial_k2_fibonacci -- --nocapture
  2 passed

cargo test -p neo-fold-clean --release --test system_r1cs_compiler_stateful
  17 passed
```

Follow-up perf snapshot for the same four app transitions:

```text
cargo test -p neo-fold-clean --release --test system_r1cs_compiler_stateful \
  r1cs_stateful_serial_k2_fibonacci_perf_snapshot_compares_four_transitions \
  -- --ignored --exact --nocapture

stage                 K=1 four chunks   serial-K2 two chunks
-------------------   -------------   --------------------
preprocess               102.441                101.045
append total             616.569                206.150
finish                   121.365                109.680
verify                    52.401                 50.953
total                    907.555                483.328

app transitions/s          4.41                   8.28
```

This is a tiny Fibonacci fixture, not SHA, so do not extrapolate the exact
speedup. The important evidence is structural: proving two serial app
transitions inside one fresh CCS instance cuts append work by roughly 3x for
this workload while preserving the same verifier path. That is the paper-safe
direction to try next on SHA.

### 4e. SHA ordered-pair batch experiment landed

There is also a separate SHA batching experiment:

```text
tests/system/sha256_bellpepper_batching.rs
  sha256_ordered_pair_r1cs_binds_both_public_digests
  sha256_ordered_batch_r1cs_binds_middle_public_digest
  sha256_production_core_ordered_pair_batch_four_statements_perf_snapshot
  sha256_production_core_ordered_batch_size_curve_four_statements_perf_snapshot
```

This is **not** a serial SHA state machine. It proves an ordered public batch
of two independent SHA preimage statements inside one app R1CS, with public
inputs:

```text
[1, digest_0_bits, digest_1_bits]
```

That distinction matters. The experiment is sound for "prove these ordered
SHA preimages" because both digests are public inputs to the larger app
relation. It does not claim that SHA output 0 feeds SHA input 1, and it does
not weaken the existing stateful `K > 1` rejection.

The non-ignored smoke tests flip digest bits in both the ordered-pair and
ordered-batch R1CS shapes and check that the app relation rejects, so the
batched public outputs are not timing-only fixture data.

Perf snapshot for four independent 3-byte SHA statements:

```text
cargo test -p neo-fold-clean --release \
  --test system_sha256_bellpepper_batching \
  sha256_production_core_ordered_pair_batch_four_statements_perf_snapshot \
  -- --ignored --exact --nocapture

stage                     four singles   two ordered-pairs
-----------------------   ------------   -----------------
bellpepper synth shared       23.062ms           23.062ms
preprocess                   145.247ms          218.992ms
append total                 844.466ms          309.831ms
finish                       219.374ms          260.319ms
verify                        98.511ms          124.736ms
total                       1464.081ms         1113.296ms

statements/s                    2.73               3.59
```

A fresh batch-size curve on the same four statements:

```text
cargo test -p neo-fold-clean --release \
  --test system_sha256_bellpepper_batching \
  sha256_production_core_ordered_batch_size_curve_four_statements_perf_snapshot \
  -- --ignored --exact --nocapture

batch size                 1 statement        2 statements       4 statements
chunks                              4                 2                  1
preprocess                   162.280ms         231.118ms          401.722ms
append total                 937.153ms         337.994ms          118.936ms
finish                       233.790ms         280.105ms          235.015ms
verify                       102.456ms         123.603ms          118.497ms
total                       1524.958ms        1090.499ms         1059.652ms
statements/s                    2.62              3.67               3.77

shape                      1 statement        2 statements       4 statements
app constraints                 24803             49606              99212
app vars                        24625             49249              98497
app public inputs                 257               513               1025
plan limbs                    270578            541154            1082306
F' rows n                      28839             53984             104274
F' vars m                     514803            807267            1392195
F' matrices t                      8                 8                  8
```

This is the strongest current evidence that larger app statements can reduce
append time without touching the folding verifier. It does **not** yet make
four independent SHA statements reliably sub-second on this machine. Batch
size 4 makes append very cheap, but it grows the F' variable count from
`492,915` to `1,370,307`; that shifts cost into preprocessing and terminal
work. For the original five-step SHA benchmark, the next shape is a serial
SHA mini-chain or an explicit app-output log digest, not silent K>1 in the
current stateful path. The next optimization target is therefore the terminal
and preprocessing cost of large app statements, not just the number of appends.

### 4f. Profile-backed bottleneck map for the batch curve

The latest `profile_for_ai` run on the ordered-batch curve:

```text
./scripts/profile_for_ai.sh neo-fold-clean \
  system_sha256_bellpepper_batching \
  sha256_production_core_ordered_batch_size_curve_four_statements_perf_snapshot \
  --ignored 20
```

reported the following top self-time buckets:

| Rank | Hotspot | What it means |
|---:|---|---|
| 1 | `neo_ajtai::s_module::commit_signed_unit_row_many_chunk` | Ajtai commitment remains the largest cryptographic kernel. |
| 2 | Rayon closure bodies in folding/oracle code | Work is parallelized, but split across several protocol kernels; more threads alone is not a design fix. |
| 3 | `rayon::slice::sort::recurse` / insertion sort | Structure/preprocessing and sparse-matrix construction are now visible for large app statements. |
| 4 | `rand_core::block::BlockRng::generate_and_set` | Seeded PP generation inside Ajtai commitment is a meaningful part of commit cost. |
| 5 | Poseidon2 assembly permutations | Larger batched F' images make the bit-backed Poseidon2 trace cost visible again. |
| 6 | `superneo_eval::digit::apply_monomial_pair_in_place` | CE evaluation arithmetic is still a real terminal/finalization cost. |
| 7 | `neo_ajtai::commit::accumulate_binary_mask_sparse` | Sparse binary commit accumulation is another Ajtai-side cost. |
| 8 | `CscMat::from_triplets` | Matrix construction/sorting is a preprocessing bottleneck candidate. |

This profile changes the story from "one bad function is slow" to "several
paper-mandated kernels are now in the same range." That matters for the
sub-second plan:

1. **Batching/serial mini-chains are still the highest leverage.** They reduce
   the number of times we pay the full Construction-2 append machinery. This
   is the HyperNova/SuperNeo-native lever, as long as the K-wide app statement
   actually proves the internal serial transitions or ordered public batch.
2. **Ajtai commit optimization is the best kernel target.** It affects append,
   finish, and verify paths, and the profile still puts it at the top.
   Safe work here means preserving the exact seeded PP stream and checking
   materialized-PP equivalence.
3. **Preprocessing/sparse construction becomes important for large batches.**
   Once batch size grows, `F' vars m` and plan limbs grow enough that sorting
   and triplet conversion are no longer noise. Avoid optimizing only the test
   harness; target shared structure-building paths when possible.
4. **Poseidon2 trace count is the source-image bottleneck, not "Poseidon2 is
   wrong."** Under `b = 2`, bit-backed Poseidon2 is expensive because it must
   prove low-norm bits. Replacing it with unconstrained full-field lanes would
   be unsound. The safe direction is reducing the number/width of hash traces
   or proving a different low-norm-friendly digest circuit with the same
   transcript/security role.
5. **Terminal CE relation is not optional.** SuperNeo Def. 13 requires the
   terminal opened witness to satisfy commit, X, low-norm, and `y_ring = Mz(r)`;
   this code also stores `ct`, so `ct == constant_term(y_ring)` must stay.
   Moving that work out of the direct circuit is a separate compact-decider
   design, not a free optimization.

Concrete next experiments, in order:

| Experiment | Expected upside | Soundness gate |
|---|---:|---|
| Serial K=2 SHA mini-chain as one fresh CCS instance | Reduces append count without pretending independent claims are serial | The app R1CS must feed transition `i`'s output directly into transition `i+1`, and cross-chunk semantic input/output must be anchored. |
| Ajtai commit chunk/PP planning cleanup | Moves the largest current kernel | Seeded vs materialized PP equivalence and commitment equality tests. |
| Sparse construction/preprocess profiling on batch size 4 | Targets the `sort`/`from_triplets` profile bucket | Native/circuit structure digest parity must remain unchanged. |
| Poseidon2 trace-count audit in F' image | Could reduce columns for tiny apps and state hashing | Every removed absorb source must be transitively verifier-owned or pinned by rows. |
| Typed-width reduction for SHA app variables | Reduces app/private/control columns | Width must be proven by R1CS/range rows, not inferred from one witness. |

### 4g. Serial SHA mini-chain result

I added a sound serial SHA experiment in
`system_sha256_bellpepper_batching`: one app R1CS proves two SHA transitions
by feeding the first SHA digest directly into the second SHA gadget. This is
not SuperNeo K>1 folding. It is a larger single fresh CCS instance that
contains two serial app transitions, so HyperNova still sees one ordinary
append and the app circuit owns the internal transition link.

Two state encodings were measured for a four-transition SHA chain. The
important split is setup vs online work:

- `plan/structure` is the benchmark's fixed-shape plan convergence and
  `FPrimeStructure` probe.
- `preprocess` is lifecycle preprocessing for a fixed `(R1CS, F' plan, params)`.
  It should be reusable across many proofs of the same statement shape.
- `setup total` is `plan/structure + preprocess`.
- `online prove` is `append_total + finish`.
- `online total` is `append_total + finish + verify`.

| Encoding | Public state shape | F' chunks for 4 SHA transitions | F' rows `n` | F' vars `m` | Online total | Setup+online total | Takeaway |
|---|---:|---:|---:|---:|---:|---:|---|
| bit-public state | 256 input bits + 256 output bits | 2 | 99,730 | 3,610,887 | 4.83s | 8.30s | Sound but terrible; explicit semantic input/output over 512 bit variables creates huge extra state traces. |
| 56-bit packed state, pair chunks | 5 input limbs + 5 output limbs | 2 | 56,136 | 853,190 | 0.81s | 1.25s | Sound and now below one second for online prove+verify; 56-bit limbs are injective in Goldilocks and avoid the bit-public blowup. |
| 56-bit packed state, quad chunk | 5 input limbs + 5 output limbs | 1 | 106,936 | 1,395,530 | 0.55s | 1.53s | One append is cheaper, but the larger F' shape makes setup dominate and overall wall-clock worse. |

The 56-bit choice is deliberate. Packing 64 SHA bits into one Goldilocks
field is not injective for every 64-bit chunk because
`p = 2^64 - 2^32 + 1 < 2^64`; a chunk and `chunk - p` can collide in field
arithmetic for some values. Packing at 56 bits follows the transcript helper's
"7 bytes per field limb" pattern and keeps the public state limb equalities
injective.

The same packed-state pair circuit with only one two-transition chunk is the
first measured production-core setup+online path below one second:

```text
SHA^2 packed-state serial pair:
  setup: plan/structure 168.943 ms
  structure builds              1
  setup: preprocess     354.476 ms
  setup total           523.419 ms
  append                130.146 ms
  finish                165.714 ms
  verify                 90.401 ms
  online prove          295.861 ms
  online prove+verify   386.262 ms
  setup+online total    921.975 ms
```

For four linked SHA transitions using two packed-pair chunks:

```text
SHA^4 via two packed serial pairs:
  setup: plan/structure 134.756 ms
  structure builds              1
  setup: preprocess     288.637 ms
  setup total           423.394 ms
  append total          392.894 ms
  finish                298.652 ms
  verify                120.207 ms
  online prove          691.546 ms
  online prove+verify   811.753 ms
  setup+online total   1245.013 ms
```

This result is both encouraging and sobering:

- sound serial batching can reduce append count without changing SuperNeo;
- state representation dominates: bit-level public state made the right idea
  slower by 6x;
- the packed serial pair lands below one second for setup+online `SHA^2`;
- the packed serial pair also lands below one second for online `SHA^4`;
- larger chunks are not automatically better: the serial quad does fewer
  appends but grows the F' shape enough to lose on setup+online wall-clock.
- seeding the CE-shape fixed-point loop from cheap row/column-domain
  estimates cut plan convergence from two full structure builds to one.

The next viable route is probably not "make chunks arbitrarily larger." It is
finding the sweet spot where serial app work, F' source-image width, and
terminal fold cost balance. For SHA today that sweet spot is closer to packed
pairs than packed quads. To get cold `SHA^4` under one second, the main target
is now setup/preprocess: after the fixed-point seeding, `plan/structure` is
`~135ms` and lifecycle `preprocess` is `~289ms`.

This also explains the CPU-utilization observations: the hot code already
uses Rayon in the large arithmetic kernels, but some phases are serial by
construction (test orchestration, preprocessing boundaries, terminal fold
sequencing), and some parallel regions have uneven chunk sizes. More Rayon
threads may smooth local kernels; it will not by itself eliminate the
preprocess/terminal costs that appear when F' grows past one million variables.

### 4h. Reusable preprocessing artifact

The current SHA packed-pair numbers make the next cold-start bottleneck
unambiguous:

```text
SHA^4 via two packed serial pairs:
  plan/structure        ~135ms
  lifecycle preprocess  ~289ms
  online prove+verify   ~812ms
  setup+online         ~1245ms
```

After splitting the verifier-owned cache build out of lifecycle
preprocessing, the same four-transition serial-pair SHA snapshot reports:

```text
SHA^4 via two packed serial pairs, prepared-key split:
  plan/structure             184.931 ms
  prepare cache              168.364 ms
  preprocess from prepared     9.562 ms
  setup total                362.856 ms
  append total               406.880 ms
  finish                     305.751 ms
  verify                     141.612 ms
  online prove               712.632 ms
  online prove+verify        854.244 ms
  prove with prepared key    863.806 ms
  setup+online total        1227.496 ms
```

`prepare cache` is verifier-key work. `preprocess from prepared` is the
remaining lightweight construction needed to turn that prepared structure
artifact into a `Preprocessing` value for a chosen Ajtai seed and params. The
online proof path is unchanged in meaning: `append total + finish + verify`.

A one-second `profile_for_ai.sh` sample of this prepared-path snapshot shows
the online work is still dominated by real proof kernels:

```text
p3_goldilocks Poseidon2 permutation asm        transcript/hash work
neo_ajtai::commit_signed_unit_row_many_chunk  Ajtai commitments
neo_ajtai::accumulate_binary_mask_sparse      bit/signed-unit commit kernel
rayon sort / CscMat::from_triplets            sparse matrix construction
superneo_eval::apply_monomial_pair_in_place   SuperNeo matrix eval kernel
```

This points away from "one forgotten serial loop" and toward two real levers:
reduce how many giant F' witnesses are committed/evaluated, or make the Ajtai
and SuperNeo evaluation kernels cheaper while preserving the same equations.
HyperNova's memory-check optimization is relevant here, but only if the
externalized running accumulator is verifiably read and written; a shorter
digest or handle alone is not a sound substitute for `U_i`.

The next kernel-level target is Ajtai signed-unit commitment. It is already
parallelized across `(kappa, chunk)` tasks for large seeded commitments, so
the likely gains are local: less per-column rotation work, less temporary
allocation, or better packed-mask handling. Any change there needs an
equivalence gate against materialized-PP commitment, because commitment
speedups are directly proof-binding.

I added an ignored micro-snapshot for that kernel:

```text
seeded_pp_signed_unit_commit_many_sha_fprime_perf_snapshot
  d=54
  kappa=16
  m=16384
  claims=14
  nonzero_cols=16063
  avg_nonzero_entries_per_claim=5897.14
  best_ms=22.141
  median_ms=24.405
  mean_ms=24.420
```

That is too small to explain the whole SHA/F' online cost by itself. The full
profile's Ajtai samples are likely many repeated commitment/opening calls plus
surrounding transcript/proof work, not one standalone `commit_many` invocation
that can be cut in half locally. This pushes the next search back toward
call-count reduction and shape reduction, not just optimizing this one kernel.

The paper framing says this setup work belongs to the verifier key, not to
every proof. HyperNova `K` computes the encoded structures and verifier keys
once; SuperNeo `K` fixes `(pp, s)`. Rebuilding sparse caches, SuperNeo eval
tables, matrix digests, and `vk_fs` for every witness is useful for a benchmark
but not the target product shape.

Current code now has the safe seam:

- `R1csFPrimeDerivedStructure` owns the validated `(plan, r1cs, structure)`
  tuple and prevents external callers from assembling a mismatched triple.
- `R1csFPrimePreparedStructure` adds the verifier-derived
  `OptimizedStructureCache` and `structure_digest` to that same opaque bundle.
- `preprocess_seeded_prepared_with_params` consumes the prepared artifact and
  installs the cache into `Preprocessing` without rebuilding it.
- `lifecycle::preprocess_with_test_log_and_optimized_cache` is crate-private,
  shape-checks the cache, and is only reachable from frontends that own the
  structure/cache construction path.

The safe product shape is a verifier-owned key artifact:

```rust
pub struct R1csFPrimePreparedStructure {
    plan: RecursiveStepImagePlan,
    r1cs: R1csShape,
    structure: Arc<FPrimeStructure>,
    anchors: R1csRowAnchors,
    public_input_len: usize,
    optimized_cache: OptimizedStructureCache,
    structure_digest: [F; 4],
}
```

Construction is the only place that builds the cache:

```rust
pub fn prepare_sparse_preprocessing_structure(
    r1cs: &SparseR1cs,
    plan: &RecursiveStepImagePlan,
) -> Result<R1csFPrimePreparedStructure, Error> {
    prepare_derived_structure(derive_sparse_preprocessing_structure(r1cs, plan)?)
}
```

Then preprocessing from the prepared artifact moves, rather than trusts, that
cache into lifecycle preprocessing:

```rust
pub fn preprocess_seeded_prepared_with_params(
    prepared: R1csFPrimePreparedStructure,
    params: Params,
    seed: u64,
) -> Result<R1csFPrimePreprocessing, Error> {
    // Validate Ajtai dimensions against prepared.structure.ccs.
    // Build vk from prepared.structure_digest.
    // Install prepared.optimized_cache without rebuilding it.
    // Install semantic mode + initial semantic anchor from prepared.plan.
}
```

Do **not** add an API shaped like:

```rust
preprocess_with_external_cache(structure, optimized_cache)
```

That would let a caller pass a cache/digest pair not derived from the
structure. Even if honest benchmarks use it correctly, it violates the local
security rule: digests and caches are compression, not authority. The cache
must be an internal field of a structure-owned artifact built by the verifier.

Validation landed:

- Prepared and non-prepared preprocessing produce identical
  `structure_digest`, `vk.digest()`, semantic mode, and initial semantic
  anchor for the same `(r1cs, plan, params, seed)`:
  `r1cs_preprocess_prepared_structure_matches_standard_preprocess`.
- `validate_cached_structure()` passes on prepared preprocessing.
- The prepared API still rejects plan/R1CS public-input binding mismatches:
  `r1cs_preprocess_prepare_rejects_public_input_mismatch`.
- No public constructor accepts `(Structure, OptimizedStructureCache)` as
  independent arguments.

Expected impact:

- Recurring proofs already avoid this cost if callers reuse `Preprocessing`.
- Product cold-start can move this verifier-key work outside the per-proof
  path. Benchmarks should report both "prepare key" and "prove with prepared
  key" rather than mixing them into one wall-clock number.
- This is the least risky path to push `SHA^4` setup+online toward one second:
  it changes ownership of verifier preprocessing work, not any proof equation.

Prepared-key amortization snapshot:

```text
cargo test -p neo-fold-clean --release \
  --test system_sha256_bellpepper_batching \
  sha256_production_core_serial_quad_prepared_key_amortization_snapshot \
  -- --ignored --exact --nocapture --test-threads=1

proofs with same anchor        3
SHA transitions/proof          4
setup total                  474.553 ms

proof    append     finish     verify     online
0       134.217    232.336    121.858    488.411 ms
1       144.647    218.177    114.895    477.719 ms
2       141.330    221.804    112.477    475.611 ms

online prove+verify total   1441.741 ms
amortized setup+online/proof 638.765 ms
```

Important caveat: the prepared R1CS-F' structure contains the verifier-owned
initial semantic-state anchor. Attempting to reuse one prepared key for
different initial SHA states failed at F' structure satisfaction, which is the
correct soundness behavior. Product reuse is therefore "many proofs for the
same anchored statement/key", not "one key for arbitrary initial states".

### 5. Continue Π_RLC arithmetic cleanup

Best target: `crates/neo-reductions/src/engines/optimized_engine/rlc.rs`.

Current status:

- `Z_mix` now uses ring multiplication for validated rotation challenges.
- SHA timer builds show `Z_mix` mostly around `0.01s-0.02s`, down from the
  previous `0.02s-0.06s` range.
- The helper currently duplicates a small generic Phi_81 multiplication kernel.

Safe next shape:

- move the generic Phi_81 multiply into `neo-math` only if another caller needs
  it;
- keep `validate_rhos_are_rotation_matrices` as the authority gate before any
  ring shortcut;
- preserve generic `left_mul_acc` equivalence tests on sampled rotation rhos;
- avoid replacing the RLC challenge type with an unconstrained matrix.

### 6. Keep Poseidon2 reductions conservative

Fibonacci is almost all `state_x_out` Poseidon trace. Reducing trace count is
high leverage, but dangerous.

Allowed reductions must satisfy one of:

- the field is already absorbed transitively by `vk_fs`;
- the field is pinned as a state wire elsewhere in the F' circuit;
- the field is a duplicate of another bound coordinate;
- a parity test proves native and circuit preimages match.

Do not swap hash families in protocol-binding paths without explicit approval.
Do not replace bit-backed Poseidon2 with unconstrained full-field lanes under
`b = 2`.

### 7. Reduce SHA app width only with verifier-owned evidence

The SHA diagnostic shows many non-Boolean variables are small in the honest
assignment, but assignment values are not authority. Width reductions are sound
only when the R1CS shape proves the range, or the frontend provides
verifier-owned range constraints.

Good directions:

- make the SHA/R1CS frontend expose more variables as explicit bits;
- add range constraints where variables are intended to be bytes/words;
- then let R1CS-F' use smaller typed widths.

Bad direction:

- infer width from a sample witness and bake it into the proof system.

## Validation checklist for future changes

Every performance change in these files should run at least:

```bash
cargo test -p neo-fold-clean --release --test f_prime_digest_circuit
cargo test -p neo-fold-clean --release --test system_phase_1_3d_step_parity
cargo test -p neo-fold-clean --release --test system_lifecycle_f_prime_link
cargo test -p neo-fold-clean --release --test system_r1cs_preprocess
cargo test -p neo-reductions --release --test superneo_eval_equivalence
```

Every command must stay under the 5-minute cap. Use the project rule's
`timeout = 300000ms` wrapper when running through tools.

## Decision

Do not reset or stash the whole branch just because older audits mention the
compact accumulator work. The scary `state_x_out` mismatch from the stale audit
is fixed in the current tree. The current work is worth salvaging, but further
optimization should stay below the protocol line: Ajtai and SuperNeo arithmetic
kernels are fair game; verifier authority is not.
