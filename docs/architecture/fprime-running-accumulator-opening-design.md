# F' Running-Accumulator Opening Design

Date: 2026-05-31
Status: design target for a clean rebuild
Scope: `neo-fold-clean` F' recursive-step performance

## Goal

Shrink the recursive F' image without weakening HyperNova Construction 2.

The current production-shaped Fibonacci F' image is dominated by bit-backed
Poseidon2 traces. The recursive accumulator hash over the running accumulator
handle is the largest single contributor. Removing that trace gets close to
sub-second folding in the dirty experiment, but doing so naively breaks the
paper contract.

This document defines the sound target:

```text
F' may carry a compact accumulator handle only if it verifies that the handle
opens to the full running accumulator consumed by NIFS.V, and that the outgoing
handle opens to the running accumulator produced by NIFS.V.
```

## Paper Contract

HyperNova Construction 2 says the recursive link binds the full running
instance:

```text
u'_i = enc_inst(hash(vk_fs, i, z_0, z_i, U_i, pc_i))
U_{i+1}[pc_i] = NIFS.V(vk_fs[pc_i], U_i[pc_i], u_i, pi)
x_out = hash(vk_fs, i+1, z_0, z_{i+1}, U_{i+1}, pc_{i+1})
```

The same section allows the verifier circuit to use memory checking to
verifiably read/write running instances from externalized memory. That is the
only safe way to replace direct `U_i` hashing with a short handle.

SuperNeo Definition 13 fixes what one CE claim means:

```text
c = L(z)
x = L_in(z)
||z||_infty < b
y_j = M_j z(r) for every CCS matrix j
```

The implementation additionally stores `ct` as a denormalized field. Any handle
that claims to represent a CE claim must either bind `ct` and enforce
`ct == constant_term(y_ring)`, or reject the representation.

## Non-Goals

- Do not change hash families on protocol-binding paths. Use Poseidon2 unless
  explicitly approved otherwise.
- Do not shrink `state_x_out` by dropping fields without a field-by-field proof
  that each field is equivalently bound elsewhere.
- Do not use `parent.c.data` as authority for `U_i`.
- Do not rely on a later consumer step to repair a bad producer step. The
  current F' step must prove the handle it outputs.
- Do not frame native Rust checks as a substitute for circuit constraints when
  the consumer is the circuit verifier.

## Required Data Model

Introduce a narrow, explicit protocol type for the handle. Names are
illustrative; the important part is ownership.

```rust
struct RunningAccumulatorHandle {
    root: [F; 4],        // Poseidon2-bound commitment to the public U_i data
    count: u64,          // number of CE children represented
    mode: HandleMode,    // Empty, DirectSmall, MemoryRoot, ...
}
```

The handle is public recursive state. It is not authority by itself. It is
accepted only together with opening material verified by F'.

The canonical CE-claim leaf digest must bind every public field consumed by the
SuperNeo verifier:

- commitment metadata and `c.data`
- active `X`
- `r`
- `s_col`
- `y_ring`
- `ct`
- `y_zcol`
- `m_in`
- `fold_digest`
- supported offset/aux fields

If a field is not supported by the F' frontend, reject non-empty values before
hashing. Silent omission is not allowed.

## Subsecond Strategy

The paper-grounded route to `<= 1s` folding is not "process the same giant
F' image faster." It is to remove work from the hot F' image only when an
equivalent verifier-owned proof or circuit row keeps the same authority.

| Priority | Cut | Expected impact | Soundness gate |
|---|---|---:|---|
| 0 | Keep native `state_x_out`, the F' circuit gadget, and the F' planner on one byte-identical preimage. | Correctness blocker, not a speedup. | The native/circuit parity tests for `state_x_out` and the F' R1CS accept tests must pass. |
| 1 | Delayed small-accumulator handle: remove producer-side full accumulator hashing from the source image, then prove every produced handle is consumed by the next step or terminal fold. | Very high for the current Fibonacci/SHA shapes. | Tampering `acc_digest_in`, `acc_digest_out`, child CE fields, or terminal running must reject without relying on self-consistent re-digests. |
| 2 | Real running-accumulator opening: memory read/write, direct small mode, or a compact proof that opens the handle to the full CE data consumed by NIFS.V. | Highest long-term impact. | The handle is never authority by itself; F' verifies the opening or a downstream terminal proof verifies it. |
| 3 | Typed low-norm app variables and packed public inputs. | High for Boolean-heavy circuits. | Every packed bit must be R1CS-Boolean-constrained or structurally pinned before it contributes to `state_x_out`. |
| 4 | Keep terminal DEC/RLC witnesses in digit/packed form longer. | Medium; targets `chain.finish()`. | Same SuperNeo CE equations, same transcript challenges, no Rust-only terminal shortcut. |
| 5 | Reuse verifier-owned preprocessing artifacts across folds/proofs for the same circuit. | Medium for end-to-end wall time; low for online append. | Cache keys include params, structure digest, plan, and public-input shape; prover caches are never trusted. |
| 6 | More Rayon / CPU tuning. | Constant factor only. | Apply after shape cuts; parallel kernels must be deterministic and covered by serial-vs-parallel tests. |

The first gate matters because a compact digest migration is easy to do
halfway. If native `state_x_out` is wide while the circuit is compact, honest
recursive links become unsatisfiable. If both are compact but a dropped field is
not pinned elsewhere, the branch trades soundness for speed. Every compact slot
therefore needs an explicit owner: `vk_fs` owns structure/params, `step_count`
owns the iteration counter, `z_0` is verifier-derived through `vk_fs`, `z_i`
stays in the hash, the accumulator handle is consumed by NIFS/terminal rows,
and `pc` is absorbed directly even though this frontend currently has only one
`F'` program.

## Candidate Opening Schemes

### Option A: Poseidon2 Merkle Memory

Represent the running accumulator as a Poseidon2-authenticated tree over
canonical CE-claim leaves. F' verifies:

1. `handle_in.root` opens to the prior running CE claim(s) consumed by NIFS.V.
2. NIFS.V computes outgoing child claims.
3. `handle_out.root` is the authenticated update of the tree after replacing
   the relevant slot with those outgoing claims.

Pros:

- Closest to HyperNova's stated memory-checking suggestion.
- Simple threat model: every mutated CE field changes the leaf and breaks the
  path.
- Works with Poseidon2-only binding discipline.

Cons:

- Poseidon2 Merkle paths are still expensive if represented with today's
  bit-backed hash traces.
- Best for many program counters; with `ell = 1`, the tree depth may not be
  where the cost is. The leaf digest/opening remains the hard part.

### Option B: Direct Small-Accumulator Mode

For tiny fixed `k`, prove the full `U_i` binding directly, but avoid the
source-image one-shot trace. The R1CS verifier already has the CE fields as
wires for NIFS.V; the compact handle verifier should reuse those wires and
compute a Poseidon2 digest in the R1CS circuit, not splice a separate giant
source-image trace.

Pros:

- Smallest conceptual jump from current code.
- Good first milestone: same authority as direct hashing, fewer duplicate
  source-image regions.

Cons:

- Still pays Poseidon2 verifier cost somewhere.
- Must prove that the R1CS Poseidon representation remains low-norm-compatible
  under the F' source-image encoding.

### Option C: Spartan/FRI Opening for U

Use a compact proof that the CE fields consumed by NIFS.V open to the public
handle. F' verifies the proof instead of hashing/opening the full data.

Pros:

- Matches the paper's broader compression posture: Spartan with FRI-friendly
  fields.
- Potentially the largest asymptotic win.

Cons:

- Larger protocol addition.
- Needs its own native prover, verifier, circuit verifier, serialization, and
  adversarial test suite.

## Recommended Phasing

### Phase 0: Return to Sound Baseline

Start from the current committed branch tip, not the dirty tree.

Restore native/circuit parity for full `state_x_out`. Keep direct full
accumulator binding until a replacement opening exists. Run the existing
soundness suites with the 5-minute cap.

### Phase 1: Lock the Handle Contract With Failing Tests

Before implementing a compact handle, add tests that fail under the old
commitment-only or c-data-only authority:

- mutate `r` in the running CE claim and recompute prover-controlled digests;
  verifier/circuit must reject;
- mutate `y_ring`;
- mutate `ct`;
- mutate active `X`;
- mutate `c.data`;
- tamper the outgoing handle away from NIFS.V's actual output in the same step.

These tests must hit both native lifecycle verification and the in-circuit F'
recursive step where applicable.

### Phase 2: Implement Direct Small-Accumulator Mode

This is the least speculative performance cut.

Replace source-image accumulator Poseidon traces with a reusable circuit-level
handle verifier that consumes the same CE wires NIFS.V already consumes. The
first version may still be slower than the dirty experiment; it must be sound
and measurable.

Acceptance gates:

- native handle digest equals in-circuit handle digest;
- F' recursive step rejects bad `handle_in`;
- F' recursive step rejects bad `handle_out`;
- no source-image `c_data`-only authority remains;
- performance report includes source-image columns by region.

### Phase 3: Choose Memory or Proof Backend

Only after Phase 2, decide whether Merkle memory or Spartan/FRI opening is the
right larger cut.

Decision criteria:

- total F' rows/columns;
- online recursive append time;
- terminal finish time;
- verifier time;
- peak RSS;
- implementation size;
- ease of proving `U_i` equivalence field-by-field.

## What Success Looks Like

The near-term performance target is not "the dirty experiment is fast." It is:

```text
Recursive append is near 1s while:
  - state_x_out still binds HyperNova's recursive public link,
  - U_i authority is opened/proven, not assumed,
  - native and in-circuit verifier equations agree,
  - adversarial CE-field mutations reject.
```

If that first sound version is slower than 1s, the measurement is still useful:
it tells us exactly which verified opening component needs the next cut.
