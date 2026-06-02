# Uncommitted Soundness Salvage Notes

Date: 2026-05-31

This note separates the current dirty-tree work into pieces worth keeping,
pieces that should be rebuilt from a clean branch tip, and pieces that should
not be carried forward without a fresh proof argument.

The goal is not to preserve every local edit. The goal is to get back to a
small, auditable path toward sub-second folding without violating the
HyperNova/SuperNeo verifier contract.

## Paper Contract

HyperNova Construction 2 is the load-bearing reference for the F' recursive
link:

- F' checks that the fresh instance's public remainder is
  `enc_inst(hash(vk_fs, i, z_0, z_i, U_i, pc_i))`.
- F' computes `U_{i+1}` by running `NIFS.V` on `U_i[pc_i]` and `u_i`.
- F' outputs `hash(vk_fs, i+1, z_0, z_{i+1}, U_{i+1}, pc_{i+1})`.
- Externalizing `U_i` is allowed only with a verifiable read/write memory
  argument. A short digest is compression, not authority by itself.

SuperNeo Definition 13 is the load-bearing reference for terminal CE claims:

- `c = L(z)`
- `x = L_in(z)`
- `||z||_infty < b`
- `y_j = M_j z(r)` for every CCS matrix `j`

The implementation stores `ct` separately even though the paper treats it as
the constant term of `y_ring`. Therefore the implementation must additionally
enforce `ct[j] == constant_term(y_ring[j])` wherever `ct` is consumed as a
claim field.

## Keep

These changes look conceptually correct and should be re-applied, preferably in
small commits from the current branch tip rather than preserved by dragging the
whole dirty tree forward.

### Semantic State Anchor

Keep the verifier-owned initial semantic-state anchor:

- `vk_fs_digest` absorbs the initial semantic-state digest.
- `verify_uncompressed` rejects a proof whose recorded initial semantic state
  does not match preprocessing.
- The base F' image constrains the base `state_in` semantic digest to the
  verifier-owned anchor when semantic state is enabled.
- Public/prep drift is blocked by plan validation and by narrowing setter
  visibility.

This is a real soundness fix. It prevents relabeling a chain as starting from
one app state while the base F' wires encode another.

### Terminal CE Closure

Keep the direct terminal CE checks as the current reference verifier path:

- Native `verify_uncompressed` must check commit opening, X projection,
  low-norm, y-ring evaluation, and `ct` consistency for each terminal running
  claim/witness pair.
- The decider circuit reference gadget must enforce the same obligations
  in-circuit when the circuit verifier is the consumer.
- `r.len()` should be pinned to the row-domain length used by the fold engine,
  including the min-one-round convention for degenerate `n = 1` fixtures.

This is also a real soundness fix. Without it, a terminal folded CE claim can be
self-consistent in transcript space while failing to open to a witness that
satisfies the folded CCS evaluation relation.

### Last-Step CE Continuity

Keep the continuity gate that links the last recursive F' step's NIFS children
to the terminal fold's running input in the steady-state last-step terminal
path. The full-history path already had the analogous check; both paths should
match.

This is in-circuit continuity, not a Rust-only preflight.

### Test Splits and Regression Tests

Keep the test split that moved the oversized `r1cs_compiler.rs` sections into
smaller focused integration-test files. Also keep the soundness regression
tests for:

- semantic-state anchor tampering,
- base semantic-lane mismatch,
- terminal y-ring mismatch,
- terminal `ct` mismatch,
- terminal `r` shape mismatch,
- last-step continuity.

The exact files may be reorganized again, but the coverage should survive.

### Fibonacci R1CS-F' Perf Baseline

Keep the committed-style performance baseline that records the current
R1CS-F' Fibonacci stage timings. It is useful as a regression marker while the
next performance pass is rebuilt from a sound baseline.

## Rebuild From Clean Tip

These ideas may be valid, but the current dirty-tree implementation mixes too
many concerns and should not be rescued wholesale.

### Full-Running Accumulator Handle

The current `AccumulatorHandle::from_running_parts(children, parent)` direction
addresses a real issue: HyperNova hashes `U_i`, not just `parent.c.data`.

However, the dirty implementation should be rebuilt carefully because it mixed:

- a new native digest family,
- a new in-circuit digest family,
- SplitNc transcript changes,
- F' state output changes,
- terminal decider changes,
- compiler fixture changes,
- and a partial rollback of `state_x_out`.

The rebuild should have one explicit invariant:

```text
state.acc_digest == H(full public running accumulator U)
```

where the digest binds every public CE field that affects the SuperNeo verifier
state: commitment, X, r, s_col, y_ring, ct, y_zcol, fold digest, and any
supported offset/aux fields.

If the clean pipeline forbids Pattern-A/aux fields, the verifier must reject
non-empty aux/offset fields before hashing rather than silently omit them.

### F' Producer-Side Output Binding

Do not keep the dirty-tree "delayed binding" version where the recursive F'
step allocates `acc_digest_out` as a caller-supplied witness and relies on the
next consumer to check it.

Construction 2 requires the current F' step to output a hash of the
`U_{i+1}` it actually computed. The recursive F' circuit should derive
`acc_digest_out` from `NIFS.V` outputs in the same step, then absorb that value
into `state_x_out`.

If a future performance optimization avoids a direct Poseidon trace over the
large accumulator, it needs a verifiable replacement in the same step: a memory
read/write proof, a compact opening proof, or another verifier equation that is
equivalent to "this handle opens to the computed `U_{i+1}`." A delayed next-step
check is not enough.

### `state_x_out` Preimage Reduction

The old committed branch tip absorbed the wider Construction-2 state:

```text
vk_fs_digest,
structure_digest,
chunk_count,
step_count,
z_0,
z_i,
pc,
semantic_acc,
construction2_acc,
public_trace
```

The dirty tree reduced this absorb to fewer fields for performance. That may
be defensible only after a field-by-field proof that each removed value is
already pinned by an equivalent verifier equation. Until then, restore the
wider preimage from the clean branch tip.

This is especially important because `state_x_out` is the value encoded into
the next fresh CCS public input. It is the public recursive link, so slimming
it is a protocol change.

## Do Not Carry Forward As-Is

### Commitment-Only Accumulator Handle

The older committed code already used a handle derived from the Π_RLC parent
commitment data only. That was cheaper, but it is not a faithful replacement
for HyperNova's `U_i` unless paired with a real memory/checking argument.

Do not use `parent.c.data` as the sole authority for `state_x_out` in the
non-replay verifier path.

### Documentation Claiming Security From Rust Alone

Avoid framing `verify_uncompressed` as "secure because Rust checks it." The
correct framing is that Rust executes the verifier equations directly in the
standalone verifier path. When the consumer is the decider R1CS, the same
obligations must be expressed as circuit constraints.

## Current F' Cost Model

The current production-shaped Fibonacci R1CS-F' fixture is not slow because of
the Fibonacci app. The app is tiny. The cost comes from materializing the
recursive verifier's low-norm source image.

Using the committed/sound full source-image shape:

| Region | Approx columns |
| --- | ---: |
| `boundary_update` Poseidon2 trace | 106,880 |
| `public_trace_update` Poseidon2 trace | 106,880 |
| base accumulator Poseidon2 trace | 85,504 |
| recursive accumulator Poseidon2 trace over 972 `c_data` fields | 5,279,872 |
| `state_x_out` Poseidon2 trace | 235,136 |
| all Poseidon2 traces | 5,814,272 |
| non-Poseidon image regions | about 164,000 |
| total source image | about 5,978,000 |

So about 97 percent of the source-image width is bit-backed Poseidon2 trace,
and the recursive accumulator trace dominates that. This explains why CPU
parallelism alone is not enough: it can only process the same large image
faster. A sub-second fold needs the image itself to get much smaller.

However, the naive soundness fix is also not the performance solution. Hashing
the full CE parent directly in the F' source image would be much larger than
the old `c_data`-only trace. The full claim includes commitment metadata,
`X`, `r`, `y_ring`, `ct`, `s_col`, `y_zcol`, fold digest, and supported offset
fields. Direct bit-backed Poseidon2 over that material would make the
recursive accumulator trace grow rather than shrink.

The unavoidable conclusion is:

```text
Do not directly hash a giant U_i inside F'.
Do not replace U_i with c_data-only authority.
Instead, prove/verifiably open a compact handle to U_i.
```

That is exactly the HyperNova memory-checking opening: externalize the running
instance only if F' verifies a read/write or opening argument for it.

## Paper-Compatible Performance Options

The clean design target for these options is
[`fprime-running-accumulator-opening-design.md`](fprime-running-accumulator-opening-design.md).

These are ordered by soundness risk and expected payoff.

### 1. Verifiable Memory / Opening for the Running Accumulator

This is the main path toward viability.

F' should carry a compact handle for `U_i`, but the step must also verify that
the handle opens to the full public running accumulator consumed by `NIFS.V`,
and that the next handle opens to the `U_{i+1}` produced by that same verifier.

Acceptable shapes include:

- a Merkle-style read/write proof over a canonical Poseidon2 leaf digest of
  each CE claim field;
- a Spartan/FRI-style opening proof for the accumulator representation;
- another Poseidon2-only proof system that lets the F' circuit verify the
  opening without materializing all of `U_i`.

This is a protocol feature, not an optimization patch. It needs a new public
contract, a native prover/verifier, an in-circuit verifier, and adversarial
tests proving that mutating `r`, `y_ring`, `ct`, `X`, or `c.data` breaks the
opening even if the prover recomputes every digest it controls.

### 2. Keep `state_x_out` Full Until a Field-by-Field Proof Exists

The dirty tree's reduced `state_x_out` preimage is a performance experiment.
It should not ship as-is.

Slimming `state_x_out` can be revisited only after each removed field has a
local verifier equation that is at least as strong as absorbing it into the
recursive public link. Until then, the native and circuit paths should keep the
same wider Construction-2 preimage.

### 3. Replace Generic Bit-Backed Poseidon2 Traces With a Dedicated Verifier

The bit-backed Poseidon2 trace is low-norm friendly but very wide. A dedicated
opening verifier can keep Poseidon2 as the protocol-binding hash while avoiding
"commit every internal bit of every permutation" as the representation of the
whole accumulator.

This is different from changing hash families. It keeps Poseidon2 on binding
paths, but changes what F' proves: a compact verifier equation instead of a
full bit-level execution trace over a giant preimage.

### 4. Hand-Optimize the F' Verifier Image Only After the Handle Is Sound

Jolt-style shared/virtual constraint inspiration is useful, but only after the
authority boundary is right. If the circuit still needs to prove a giant
bit-backed hash of `U_i`, shared layout tricks will not change the asymptotic
problem. They may help constants, but not enough to rescue a 6M-column image.

### 5. Parallelism Is Secondary

Rayon/thread tuning is worthwhile for the remaining prover work once the image
is small. It is not the first lever for sub-second folding because the present
dominant cost is structural, not merely underutilized CPU.

## Recommended Clean Restart

The current branch tip already contains several known-good fixes as committed
history. Do not restart from an old main commit unless there is a separate
reason to abandon the branch. Restart from the current branch `HEAD`, then
discard or replay the dirty working-tree experiments selectively.

1. Preserve the current dirty tree in a stash or throwaway branch.
2. Return to the current committed branch tip.
3. Confirm the known-good soundness tests still exist and pass from `HEAD`.
4. Add one focused accumulator-handle regression that fails under a
   commitment-only handle:
   mutate a non-commitment CE field such as `r`, `y_ring`, or `ct`, recompute
   any carried digest chain that the prover controls, and require the
   non-replay verifier/circuit path to reject.
5. Only then re-introduce the full-running handle.
6. Keep the first version intentionally slow if necessary. Recover performance
   after the proof boundary is sound.

## Clean Rebuild Gates

These gates are the minimum acceptance criteria before returning to the
sub-second folding work.

| Gate | Evidence |
| --- | --- |
| Semantic-state anchor is verifier-owned | `system_r1cs_compiler_stateful` covers public initial tamper, raw proof-state initial tamper, base-lane attack, and post-encoder base-lane mutation. |
| Terminal CE closure is complete | `system_lifecycle_finalization` covers y-ring mismatch, `ct` mismatch, `r` shape mismatch, and end-to-end `verify_uncompressed` binding behavior. |
| Decider circuit enforces the same terminal CE obligations | `system_decider_ce_relation_isolation` covers honest terminal pair, y-ring tamper, commitment tamper, X tamper, low-norm tamper, `ct` tamper, r-shape mismatch, and non-trivial `log_n >= 2` tensor ordering through the actual gadget. |
| Last-step terminal path matches full-history continuity | `system_ivc_invariants` and `system_decider_r1cs` should cover last-step terminal synthesis and decider continuity. |
| `state_x_out` preimage is not silently slimmed | `tests/f_prime/digest_circuit.rs` and `system_phase_1_mini_1_state_x_out` must agree with native `state_x_out_digest` and include the wider Construction-2 slots unless a new proof justifies removal. |
| F' producer binds output accumulator in the same step | Add a new regression before implementing the fix: tamper the recursive step's claimed `acc_digest_out` away from the digest of NIFS.V's actual output `U_{i+1}`. The recursive F' circuit must reject without relying on a later consumer step. |
| Non-replay verifier is not fooled by commitment-only accumulator handles | Add a new regression before implementing the fix: mutate a non-commitment field of the pre-final running accumulator (`r`, `y_ring`, or `ct`), recompute every prover-controlled digest/recorded state field, and require `verify_uncompressed` or the decider circuit path to reject. |

Run every test command with the hard 5-minute cap. Example shape:

```bash
CARGO_BUILD_JOBS=1 perl -e 'alarm shift; exec @ARGV' 300 \
  cargo test -p neo-fold-clean --release --test system_lifecycle_finalization
```

## Suggested Stash Command

Do not run this blindly if there are personal scratch files that should remain
visible. If the intent is to preserve everything dirty and restart from HEAD:

```bash
git stash push -u -m "salvage: mixed fprime accumulator performance experiments before sound rebuild"
```

After stashing, re-open this note from the stash or copy it somewhere outside
the worktree before resetting if it should guide the rebuild.
