# Conformance Procedure (Non-Mechanical, Artifact-Based)

> Goal: prove that theorem claims, Rust behavior, and runtime evidence agree. Passing Lean alone is insufficient.

## When This Procedure Is Mandatory

- any change to `formal/lean/Asupersync.lean`
- any Rust change touching theorem-covered surfaces
- any update to formal semantics assumptions
- before closing a frog loop

## The 7 Required Checks

For each theorem, all 7 must be answered.

### 1. Statement Parity

Does the Lean statement match Rust semantics (including failure modes)?

Checklist:
- [ ] preconditions match
- [ ] postconditions match
- [ ] explicit error/early-return behavior represented
- [ ] non-goals are explicit in theorem comments

### 2. State Representation Parity

Does Lean state faithfully represent the active Rust representation?

Key drift hotspots:
- flat `State` vs split `RegionTable` + `ObligationTable`
- pure `Nat` time vs production clock semantics
- map semantics vs slab/index reuse behavior

If abstraction is deliberate, document why it preserves theorem truth.

Practical mapping checklist:

| Lean field | Rust target | Notes |
|---|---|---|
| `State.tasks` | task table / slab | confirm index reuse semantics |
| `State.regions` | region table | confirm parent-child ownership updates |
| `State.obligations` | obligation table | confirm close gate behavior |
| `State.scheduler` | ready/timed/cancel lanes | confirm queue discipline |
| `State.time` | lab virtual time + prod clock | note divergence assumptions |

### 3. Transition Fidelity

For each used `Step` constructor:
- [ ] mapped Rust function(s) identified
- [ ] all branches traced (success/error/cancel fast paths)
- [ ] lock/atomic boundaries identified
- [ ] linearization point either explicit or justified absent

### 4. Concurrency Semantics Parity

If theorem depends on interleavings:
- [ ] interleaving model present (not implicit)
- [ ] atomic ordering assumptions documented
- [ ] lock ordering relevance checked
- [ ] waker/notification semantics accounted for if applicable

Minimum statement to attach:

```text
Concurrency assumption:
Interleaving model:
Linearization points:
Why abstraction is sound for this theorem:
```

### 5. Cancellation and Drain Parity

For cancel-related theorems:
- [ ] request -> drain -> finalize sequence reflected
- [ ] idempotence conditions represented
- [ ] loser-drain obligations represented
- [ ] region close implies quiescence still derivable

### 6. Runtime Evidence Parity

Convert proof pressure into executable evidence:
- [ ] minimized witness trace or lab seed exists
- [ ] regression test added or linked
- [ ] conformance tests exercised for mapped surface

Recommended witness artifacts:
- minimized lab-runtime seed
- textual interleaving trace
- expected-before/after state deltas

### 7. Drift Gate

Check change chronology:

```bash
git log --oneline -10 -- formal/lean/Asupersync.lean
git log --oneline -20 -- src/<mapped-file>.rs
```

If Rust semantics changed after theorem assumptions, re-open conformance.

## Route Decision Matrix

When checks fail, choose exactly one route:

- `code-first`: Rust violates intended property; fix code + add regression
- `model-first`: theorem/model mismatches real intended behavior; fix Lean
- `harness-first`: mapping/tests/fixtures stale; fix harness and rerun
- `theorem-first`: theorem overstates contract; restate with explicit scope

Record route in artifact output.

Escalation trigger:
- If route remains unresolved after two iterations, split theorem and open two separate frog cards (one per plausible route).

## Conformance Decision Contract (Required Output)

```text
theorem_id:
surface:
route:
statement_parity: pass|fail
state_parity: pass|fail
transition_parity: pass|fail
concurrency_parity: pass|fail
cancel_drain_parity: pass|fail
runtime_evidence: pass|fail
drift_gate: pass|fail
actions_taken:
retest_commands:
artifact_hash:
```

Recommended optional fields:

```text
linearization_points:
mapped_rust_functions:
witness_seed:
coverage_delta:
assurance_tier_after:
```

## Verification Commands

```bash
cd /data/projects/asupersync/formal/lean
lake build

cd /data/projects/asupersync
cargo test --test refinement_conformance -- --nocapture
cargo test --test refinement_conformance refinement_trace_equivalence -- --nocapture
cargo test --test lean_baseline_report
cargo test --test lean_invariant_theorem_test_link_map
cargo test --test lean_gap_risk_sequencing_plan
```

If code changed:

```bash
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Hard Rules

1. Do not mark a theorem "aligned" without a route decision and artifact hash.
2. Do not claim verification tier C+ without runtime evidence parity.
3. If concurrency parity is unknown, theorem cannot be used as a bug-closure claim.

4. If drift gate fails, previous alignment claim is stale until revalidated.
