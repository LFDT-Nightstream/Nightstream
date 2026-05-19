# Chip8MainLaneTraceBoundary Spec

## Purpose

- **What it is**: The theorem-facing owner for the wider root/main-lane trace
  obligations used by consumers that want one combined routing-plus-link
  contract, even when the critical kernel proof path has already discharged the
  exact link theorem elsewhere.
- **Key property**: `MainLaneTraceBound(trace)` packages row-local routing
  soundness for every exported row together with the named whole-trace link
  contract across the trace.
- **Protocol role**: This module keeps the wider root/main-lane trace contract
  explicit so consumers do not hide those facts inside row-local staged
  evidence. Exact authenticated trace closure already discharges routing
  directly from `Chip8EvidenceCoverage`; this module remains the owner of the
  optional wider contract that packages routing and linking together.

## Target Formulas

### Per-frame routing

For one execution frame

$$
\mathrm{frame} = (\mathrm{dec}, \mathrm{pre}, \mathrm{post}, z),
$$

define:

$$
\mathrm{FrameRoutingBound}(frame) := \mathrm{chip8RoutingSound}(z).
$$

This is the exact row-local main-lane consequence still required by
`Chip8StepComposition`.

### Whole-trace routing and linking

Define:

$$
\mathrm{TraceRoutingBound}(trace)
$$

to mean that every frame in the trace satisfies `FrameRoutingBound`.

Define:

$$
\mathrm{MainLaneTraceBound}(trace)
:=
\mathrm{TraceRoutingBound}(trace)
\land
\mathrm{TraceLinkBound}(trace).
$$

This is the exact wider theorem-facing root/main-lane trace contract for
consumers that want routing and linking packaged together.

### Projection theorems

This owner exposes the direct consequences:

$$
\mathrm{MainLaneTraceBound}(trace)
\Longrightarrow
\mathrm{TraceRoutingBound}(trace)
$$

$$
\mathrm{MainLaneTraceBound}(trace)
\Longrightarrow
\mathrm{TraceLinkBound}(trace)
$$

and the list-local accessors needed by downstream trace induction:

$$
\mathrm{TraceRoutingBound}(frame :: rest)
\Longrightarrow
\mathrm{FrameRoutingBound}(frame)
$$

$$
\mathrm{TraceRoutingBound}(frame :: rest)
\Longrightarrow
\mathrm{TraceRoutingBound}(rest).
$$

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/assurance-strategy.md`
- Anchors:
  - root/main-lane local row obligations remain distinct from Stage-1 /
    Stage-2 / Stage-3 extraction
  - some downstream consumers may still want an explicit packaged
    routing-plus-link trace contract
  - no theorem should smuggle root/main-lane facts into staged evidence

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/MainLaneTraceBoundary.lean` | Root/main-lane trace boundary facts |
| `Nightstream/Chip8/MainLaneTraceBoundaryInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Routing | `FrameRoutingBound` | def | Definitional | One execution frame satisfies the root/main-lane routing law |
| Routing | `TraceRoutingBound` | def | Definitional | Every frame in the trace satisfies the routing law |
| Boundary | `MainLaneTraceBound` | def | Definitional | Packages trace-wide routing and the named whole-trace link contract |
| Theorem | `traceRouting_of_mainLaneTrace` | theorem | Theorem-Target | `MainLaneTraceBound` implies trace-wide routing |
| Theorem | `executionLinked_of_mainLaneTrace` | theorem | Theorem-Target | `MainLaneTraceBound` implies `ExecutionLinked` via `TraceLinkBound` |
| Theorem | `headFrameRouting_of_traceRouting` | theorem | Theorem-Target | Head routing follows from trace-wide routing |
| Theorem | `tailTraceRouting_of_traceRouting` | theorem | Theorem-Target | Tail routing follows from trace-wide routing |
| Theorem | `frameRouting_of_traceRouting` | theorem | Theorem-Target | Any member frame inherits routing from the trace-wide routing predicate |

## Proof Obligations

- This owner must not pretend that Stage-3 continuity alone proves
  `TraceLinkBound`.
- It must not hide `chip8RoutingSound` inside staged evidence bundles when a
  consumer explicitly asks for the wider root/main-lane trace contract.
- It must stay narrow: this is only the wider optional root/main-lane trace
  contract, not a new semantic composition owner.

## Assumption Ledger

- This owner does not prove the root main-lane CCS proof itself.
- This owner does not re-prove Stage-1 / Stage-2 / Stage-3 semantic extraction.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/ExecutionSemantics.lean`
  - `Nightstream/Chip8/Routing.lean`
- **Downstream consumers**:
  - later artifact/digest/audit owners that want the wider combined trace
    contract
  - later Rust-refinement or release-boundary owners that package routing and
    linking together

## Acceptance Criteria

1. `lake build Nightstream.Chip8.MainLaneTraceBoundary` succeeds.
2. The wider root/main-lane trace contract is explicit and auditor-visible.
3. No `sorry`.
