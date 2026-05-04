# Frog Priority Queue (EV + Bayes)

> Updated: 2026-02-14
> Rule: hardest first, but only through an explicit expected-value gate.

## Ranking Model

### EV Gate

`EV_frog = (Impact * BugPrior * RuntimeReach) / (ProofCost * ModelCost)`

- Impact (1-5): correctness/security consequence if wrong
- BugPrior (1-5): prior chance proof attempt exposes a real code defect
- RuntimeReach (1-5): how widely this path is used in real execution
- ProofCost (1-5): Lean proof difficulty
- ModelCost (1-5): cost to model required semantics faithfully

Work frogs with `EV_frog >= 2.0` first.

### Posterior Update

`odds_post = odds_prior * BF(signals)`

Signal Bayes factors (starting defaults):
- concrete interleaving counterexample: BF = 8.0
- missing linearization witness: BF = 4.0
- contradiction with passing targeted test: BF = 0.35
- purely tactic/declaration-order blocker: BF = 0.20

### Final Priority Score

`priority = EV_frog * P_bug_post * AssuranceGapMultiplier`

AssuranceGapMultiplier:
- 1.0 (A/B coverage already strong)
- 1.4 (needs tier C confidence)
- 1.8 (needs tier D/E confidence)

## Current Seed Queue

These are initial priors. Recompute after each closed loop.

| Frog ID | Surface | EV_frog | Initial P_bug | Multiplier | Priority | Notes |
|---|---|---:|---:|---:|---:|---|
| `inv.race.losers_drained` | `src/combinator/race.rs` | 3.75 | 0.62 | 1.8 | 4.19 | top frog: cancellation + drain closure |
| `notify.no_lost_wakeup` | `src/sync/notify.rs` | 3.33 | 0.70 | 1.8 | 4.20 | atomic + lock interleaving |
| `inv.obligation.global_zero_leak` | `src/obligation/` | 2.67 | 0.58 | 1.8 | 2.79 | global safety theorem |
| `semaphore.cascading_wakeup` | `src/sync/semaphore.rs` | 2.40 | 0.55 | 1.8 | 2.38 | batch permit fairness/liveness |
| `pool.capacity_linearizable` | `src/sync/pool.rs` | 2.50 | 0.48 | 1.8 | 2.16 | TOCTOU elimination |
| `inv.authority.no_ambient` | `src/cx/` | 2.08 | 0.35 | 1.4 | 1.02 | deep capability modeling |
| `inv.cancel.idempotence` | `src/cancel/` | 2.25 | 0.32 | 1.4 | 1.01 | likely model strengthening |
| `runtime.split_state_refinement` | `src/runtime/state.rs` | 2.00 | 0.45 | 1.8 | 1.62 | drift-sensitive refinement |
| `lock_order.deadlock_freedom` | `src/runtime/sharded_state.rs` | 1.60 | 0.40 | 1.8 | 1.15 | high value, expensive model |

Interpretation:
- Priority >= 3.0: immediate work
- 1.5 to 3.0: queue behind current frog
- < 1.5: only if dependency/unblocker

## Frog Card Template (Use Per Candidate)

```text
Frog ID:
Surface (Rust file + function):
Failure class:
Theorem target:
Assumptions / non-goals:
Impact (1-5):
BugPrior (1-5):
RuntimeReach (1-5):
ProofCost (1-5):
ModelCost (1-5):
EV_frog:
Posterior signals observed:
P_bug_post:
Assurance tier target (A-E):
Required artifact outputs:
Fallback if blocked:
```

## Operating Rules

1. Only one active frog at a time.
2. Do not pick a lower-ranked frog unless it unblocks the top frog.
3. Recompute queue after every route decision (code-first/model-first/etc.).
4. If a frog produces a real bug, increase BugPrior of neighboring surfaces.
5. If a frog repeatedly stalls on theorem scope, tighten assumptions before retrying.

## Completed Frogs Ledger

When a frog is done, append:

```text
[DATE] frog_id=...
route=code-first|model-first|harness-first|theorem-first
lean_commit=...
rust_commit=...
tests=...
artifact_hash=sha256:...
```
