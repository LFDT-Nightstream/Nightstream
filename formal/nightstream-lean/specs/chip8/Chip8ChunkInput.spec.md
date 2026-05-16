# Chip8ChunkInput Spec

## Purpose

- **What it is**: The theorem-facing owner for the simple-kernel semantic chunk
  input contract.
- **Key property**: `SimpleKernelChunkInput` states the exact chunk facts that
  belong to the kernel input boundary rather than to Stage 1 / Stage 2 /
  Stage 3 extraction: the chunk has exactly `N` semantic rows and its head
  semantic pre-state matches the public `initial_state`.
- **Protocol role**: This module keeps the simple-kernel input contract explicit
  so later trace theorems do not smuggle those facts in as ad hoc hypotheses.

## Target Formulas

The simple-kernel input boundary in
`./crates/neo-fold-prototype/specs/chip8-kernel.md` requires:

- `semantic_trace_rows` contains exactly the `N` semantic rows of the chunk
- the first semantic row agrees with `initial_state`
- `N >= 1`

This owner packages those facts as:

$$
\mathrm{SimpleKernelChunkInput}(init, N, trace)
$$

meaning:

- `0 < N`
- `|trace| = N`
- if `trace = first :: rest`, then `InitialStateMatches(init, first.pre)`

### Projection theorems

This owner exposes the direct theorem-facing consequences:

$$
\mathrm{SimpleKernelChunkInput}(init, N, trace)
\Longrightarrow
0 < N
$$

$$
\mathrm{SimpleKernelChunkInput}(init, N, trace)
\Longrightarrow
|trace| = N
$$

$$
\mathrm{SimpleKernelChunkInput}(init, N, trace)
\Longrightarrow
trace \neq []
$$

$$
\mathrm{SimpleKernelChunkInput}(init, N, first :: rest)
\Longrightarrow
\mathrm{InitialStateMatches}(init, first.pre)
$$

These theorems are intentionally small: this owner does not re-prove Stage-3
start-boundary or final-boundary laws, because those remain owned by
`Chip8ContinuityBridge` and `Chip8AuthenticatedTrace`.

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/assurance-strategy.md`
- Anchors:
  - simple-kernel input contract
  - explicit ownership of exact semantic-row count
  - explicit ownership of head initial-state agreement

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/ChunkInput.lean` | Simple-kernel semantic chunk input contract |
| `Nightstream/Chip8/ChunkInputInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Input | `SimpleKernelChunkInput` | def | Definitional | Bundles exact semantic-row count and head initial-state agreement for one simple-kernel chunk |
| Theorem | `semanticRows_pos_of_simpleKernelChunkInput` | theorem | Theorem-Target | The input contract implies `N > 0` |
| Theorem | `traceLength_of_simpleKernelChunkInput` | theorem | Theorem-Target | The input contract fixes the exact semantic-row count |
| Theorem | `trace_nonempty_of_simpleKernelChunkInput` | theorem | Theorem-Target | The input contract implies the chunk trace is non-empty |
| Theorem | `headInitialStateMatches_of_simpleKernelChunkInput` | theorem | Theorem-Target | The input contract fixes the head semantic pre-state to the public initial state |

## Proof Obligations

- This owner must remain purely about the simple-kernel input contract.
- Do not move Stage-3 start-boundary or final-boundary ownership here.
- Do not pretend these facts are derived from Stage-1 / Stage-2 / Stage-3
  openings alone when they actually belong to the kernel input boundary.

## Assumption Ledger

- This owner does not prove that a Rust artifact populates the contract
  correctly; that belongs to later Rust-refinement or audit layers.
- This owner does not prove the row-local semantic facts of the trace; that
  belongs to `Chip8AuthenticatedTrace`.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/ExecutionSemantics.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/AuthenticatedTrace.lean`
  - later trace-level digest and audit owners

## Acceptance Criteria

1. `lake build Nightstream.Chip8.ChunkInput` succeeds.
2. The simple-kernel chunk facts are owned by one explicit theorem surface.
3. No `sorry`.
