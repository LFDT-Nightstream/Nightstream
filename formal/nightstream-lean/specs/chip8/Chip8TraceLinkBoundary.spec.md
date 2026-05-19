# Chip8TraceLinkBoundary Spec

## Purpose

- **What it is**: The theorem-facing owner for the exact whole-trace
  adjacent-frame state-link contract used by CHIP-8 chunk closure.
- **Key property**: `TraceLinkBound(trace)` is the explicit fieldwise
  adjacent-frame contract over semantic post/pre states; it is auditor-friendly
  and theoremally equivalent to `ExecutionLinked(trace)`.
- **Protocol role**: This module keeps the remaining whole-trace state-link
  contract explicit above row-local Stage 1 / Stage 2 / Stage 3 extraction,
  without pretending that per-row evidence alone already proves adjacent-frame
  linking. Higher-level authenticated trace closure is expected to derive this
  named contract from explicit component-wise temporal consistency, not from
  Stage-3 continuity alone.

## Target Formulas

### Adjacent-frame state link

For two execution frames

$$
\mathrm{current} = (\mathrm{dec}_c,\mathrm{pre}_c,\mathrm{post}_c,z_c)
$$

and

$$
\mathrm{next} = (\mathrm{dec}_n,\mathrm{pre}_n,\mathrm{post}_n,z_n),
$$

define:

$$
\mathrm{AdjacentStateLink}(\mathrm{current}, \mathrm{next})
$$

to mean:

$$
\mathrm{post}_c.pc = \mathrm{pre}_n.pc
$$

$$
\mathrm{post}_c.i = \mathrm{pre}_n.i
$$

$$
\forall idx,\; \mathrm{post}_c.v(idx) = \mathrm{pre}_n.v(idx)
$$

$$
\forall addr,\; \mathrm{post}_c.ram(addr) = \mathrm{pre}_n.ram(addr).
$$

### Whole-trace link contract

Define:

$$
\mathrm{TraceLinkBound}(trace)
$$

recursively by:

- the empty trace is linked
- a singleton trace is linked
- `current :: next :: rest` is linked iff

$$
\mathrm{AdjacentStateLink}(\mathrm{current}, \mathrm{next})
\land
\mathrm{TraceLinkBound}(next :: rest).
$$

### Equality bridge

This owner must prove:

$$
\mathrm{AdjacentStateLink}(\mathrm{current}, \mathrm{next})
\Longrightarrow
\mathrm{current.post} = \mathrm{next.pre}
$$

and conversely:

$$
\mathrm{current.post} = \mathrm{next.pre}
\Longrightarrow
\mathrm{AdjacentStateLink}(\mathrm{current}, \mathrm{next}).
$$

### Trace equivalence

This owner must then prove:

$$
\mathrm{TraceLinkBound}(trace)
\Longrightarrow
\mathrm{ExecutionLinked}(trace)
$$

and:

$$
\mathrm{ExecutionLinked}(trace)
\Longrightarrow
\mathrm{TraceLinkBound}(trace).
$$

So the final equivalence theorem is:

$$
\mathrm{TraceLinkBound}(trace)
\Longleftrightarrow
\mathrm{ExecutionLinked}(trace).
$$

### Induction accessors

Downstream trace proofs also need:

$$
\mathrm{TraceLinkBound}(current :: next :: rest)
\Longrightarrow
\mathrm{AdjacentStateLink}(current, next)
$$

and:

$$
\mathrm{TraceLinkBound}(current :: next :: rest)
\Longrightarrow
\mathrm{TraceLinkBound}(next :: rest).
$$

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./docs/assurance-strategy.md`
- Anchors:
  - exact staged row evidence does not automatically imply whole-trace linking
  - the exact trace-link contract should be explicit and auditor-visible
  - higher-level kernel closure should derive a named whole-trace link
    contract from the explicit temporal `pc` / `I` / register / RAM surfaces
    rather than weaken it into a vague boundary-only assumption

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/TraceLinkBoundary.lean` | Whole-trace adjacent-frame state-link contract |
| `Nightstream/Chip8/TraceLinkBoundaryInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Boundary | `AdjacentStateLink` | def | Definitional | One adjacent frame pair agrees on the full machine-state handoff |
| Boundary | `TraceLinkBound` | def | Definitional | A whole trace satisfies adjacent-frame state linking |
| Theorem | `post_eq_pre_of_adjacentStateLink` | theorem | Theorem-Target | Fieldwise adjacent-state agreement implies exact post/pre equality |
| Theorem | `adjacentStateLink_of_post_eq_pre` | theorem | Theorem-Target | Exact post/pre equality implies the fieldwise adjacent-state contract |
| Theorem | `executionLinked_of_traceLinkBound` | theorem | Theorem-Target | The named whole-trace link contract implies `ExecutionLinked` |
| Theorem | `traceLinkBound_of_executionLinked` | theorem | Theorem-Target | Raw `ExecutionLinked` can be normalized into the named whole-trace contract |
| Theorem | `traceLinkBound_iff_executionLinked` | theorem | Theorem-Target | The two formulations are equivalent |
| Theorem | `headAdjacentStateLink_of_traceLinkBound` | theorem | Theorem-Target | The head link law can be projected from a nontrivial linked trace |
| Theorem | `tailTraceLinkBound_of_traceLinkBound` | theorem | Theorem-Target | The linked-tail contract can be projected from a nontrivial linked trace |

## Proof Obligations

- This owner must stay purely about whole-trace state linking; it must not
  absorb row-local routing, staged evidence extraction, or transcript
  discipline.
- It must not claim that Stage 3 continuity alone implies `TraceLinkBound`.
- It should remain the thin extensional wrapper above
  `Chip8TemporalConsistency`, not a second owner for temporal proof objects.
- It must expose an auditor-friendly contract rather than forcing downstream
  owners to consume only the opaque `ExecutionLinked` chain.

## Assumption Ledger

- This owner does not prove where the adjacent-frame link contract comes from;
  it only fixes the exact named theorem surface that higher layers must derive.
- This owner does not re-prove any Stage 1 / Stage 2 / Stage 3 theorem surface.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/ExecutionSemantics.lean`
- **Downstream consumers**:
  - `Chip8AuthenticatedTrace`
  - `Chip8KernelSoundness`
  - `Chip8MainLaneTraceBoundary`
  - later digest/audit/release-boundary owners that want a named whole-trace
    link contract

## Acceptance Criteria

1. `lake build Nightstream.Chip8.TraceLinkBoundary` succeeds.
2. The named whole-trace link contract is theoremally equivalent to
   `ExecutionLinked`.
3. No `sorry`.
