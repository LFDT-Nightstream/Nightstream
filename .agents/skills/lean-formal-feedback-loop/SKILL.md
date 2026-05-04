---
name: lean-formal-feedback-loop
description: >-
  Run Lean-Rust proof feedback loops to find runtime bugs. Use when proving theorems,
  triaging hard proof blockers, checking conformance, or closing formal assurance gaps.
---

<!-- TOC: Quick Start | Critical Rules | Loop Checklist | History Mining | Frog Ranking | Routing | Elicitation | Conformance & Artifacts | Quality Gates | Troubleshooting | Anti-Patterns | References -->

# Lean Formal Feedback Loop

> Treat proof friction as evidence. Hard proof failures are high-signal indicators of Rust defects, model drift, or theorem-scope mismatch.
> This skill is an execution program, not a theorem-writing tutorial.

## Quick Start (90 Seconds)

```bash
cd /data/projects/asupersync

# 1) Load frog candidates from coverage
cat formal/lean/coverage/gap_risk_sequencing_plan.json | jq '.gaps[] | {id, priority_score, summary}'
cat formal/lean/coverage/invariant_theorem_test_link_map.json | \
  jq '.invariant_links[] | select(.theorem_witnesses | length == 0) | .invariant_id'

# 2) Verify cass index health (history mining is mandatory)
cass status --robot

# 3) Build Lean baseline
(cd formal/lean && lake build)
```

Pick one frog and close a full loop before touching another frog.

## Critical Rules (Non-Negotiable)

- Work one frog at a time; no parallel frogs.
- No theorem is "done" before 7-check conformance parity passes.
- No alignment claim without witness + regression + artifact hash.
- Treat proof friction as evidence, not tactic debt.
- Change one lever per iteration, then re-run proof + conformance.
- When uncertain, bias `code-first` (expected-loss asymmetry).

## Loop Checklist (Required Every Time)

- [ ] Intake: target subsystem, risk, current tier, target tier, fallback mode.
- [ ] Mine historical signals from `cass` (Step 0.5 below) and update `BugPrior`.
- [ ] Rank frogs with EV + posterior; pick top valid candidate (`EV_frog >= 2.0`).
- [ ] Fill reverse-round card (Surface, Failure Class, Math, Artifact).
- [ ] Attempt proof to first hard blocker; classify route.
- [ ] Extract executable witness (trace or lab seed) + regression candidate.
- [ ] Apply one route-specific change and rerun proof.
- [ ] Run 7-check conformance pass and quality gates.
- [ ] Emit proof-carrying artifact record.
- [ ] Recompute queue and stop or pick next frog.

## Step 0.5: Mine Project History (sc Requirement, Mandatory)

Mine session history before ranking frogs.

```bash
cd /data/projects/asupersync

# Health check + index readiness
cass status --robot

# If missing/stale:
# cass index --full

# Pull bug-family and proof-surface signals
cass search "notify lost wakeup" --robot --limit 12
cass search "semaphore cascading wakeup" --robot --limit 12
cass search "pool.rs TOCTOU can_create" --robot --limit 12
cass search "refinement_conformance" --robot --limit 12
cass search "race losers drained cancellation" --robot --limit 12
```

Required extraction from hits (minimum 3 records):
- `source_path` and `line_number`
- concrete bug/fix clue (commit id, test name, theorem id, or failure shape)
- whether the signal raises or lowers `BugPrior`

No frog starts until this extraction is recorded in notes/artifacts.

Historical anchors (seed priors):
- `aee9d1b`: notify lost-wakeup fix
- `e9eb3d5`: semaphore cascading-wakeup fix
- `a2e4c64`: pool TOCTOU fix

## Frog Ranking (Alien + Graveyard Math)

`EV_frog = (Impact * BugPrior * RuntimeReach) / (ProofCost * ModelCost)` (all 1-5)

- Impact: correctness consequence if wrong | BugPrior: chance proof exposes real code defect
- RuntimeReach: how hot the code path is | ProofCost: Lean difficulty | ModelCost: modeling overhead

Work frogs with `EV_frog >= 2.0` first.

Update bug probability from proof signals:

`odds_post = odds_prior * BayesFactor(signals)`

Starting Bayes-factor signals:
- constructive interleaving counterexample found: BF ~= 8.0
- contradiction with existing passing regression test: BF ~= 0.35
- stuck only on tactic normalization/declaration order: BF ~= 0.2
- stuck at linearization point existence: BF ~= 4.0
- historical match to previously fixed bug family from `cass`: BF ~= 2.0 to 3.5

Priority:

`priority = EV_frog * P_bug_post * AssuranceGapMultiplier`

AssuranceGapMultiplier:
- 1.0 for tier A/B already
- 1.4 for tier C target
- 1.8 for tier D/E target

Full scoring details and queue operations: `references/FROG-PRIORITY.md`

## Reverse-Round Card (Mandatory)

For each frog, fill all four fields:
1. Surface: which Rust subsystem/function boundary
2. Failure class: concrete failure mode family
3. Math: theorem family and proof strategy (use router below)
4. Artifact: executable witness (test seed, trace, mapping, proof hash)

No frog proceeds without all 4.

## Routing (Expected Loss, Not Vibes)

Classify first hard blocker:
- `code-first`: likely Rust defect
- `model-first`: Lean abstraction mismatch
- `harness-first`: stale tests/mappings/fixtures
- `theorem-first`: property too strong or mis-scoped

Use the asymmetric loss matrix:

```text
              | code-first | model-first | harness-first |
--------------+------------+-------------+---------------+
code_bug      |     0      |    100      |      80       |
model_issue   |    30      |      0      |      20       |
stale_harness |    20      |     15      |       0       |
```

Bias toward `code-first` under uncertainty.

Escalation rule:
- unresolved after two iterations -> split theorem and route separately.

Failure-class strategy router:
- `references/LEAN-PATTERNS.md` (Failure Class -> Technique Router)
- `references/FEEDBACK-EXAMPLES.md` (route heuristics and examples)

## Stuck-Proof Elicitation (Alien Artifact Mode)

When blocked:
- run the deep-math elicitation prompt from `references/FEEDBACK-EXAMPLES.md`
- emit a galaxy-brain diagnostic card from the same reference
- continue only after choosing a concrete discriminator action

## Conformance and Artifact Closure

Core closure rule:
- theorem not done until statement parity + transition parity + runtime evidence parity pass.

Run the mandatory 7-check pass:
- `references/CONFORMANCE-PROCEDURE.md`

Assurance ladder target:
- A: invariants + golden checksums
- B: property/fuzz tests with minimized counterexamples
- C: bounded model checking (loom/kani) where feasible
- D: protocol model checks (TLA+/PlusCal or equivalent)
- E: deductive proof (Lean theorem family)

For this skill, target C + E minimum on high-risk frogs.

Artifact contract, hashing, witness, and emission workflow:
- `references/PROOF-ARTIFACTS.md`
- No artifact record, no formal closure claim.

### Budgeted Mode + Fallback Trigger (Graveyard Pattern)

Each loop must declare budgets and an exhaustion action:
- proof iteration budget (default: 2 hard blockers before route split)
- witness search budget (default: 30 minutes targeted extraction)
- conformance rerun budget (default: full 7-check after each route change)

On budget exhaustion:
- split theorem scope (`theorem-first`) or
- switch route with explicit rationale and updated posterior

## Quality Gates

```bash
cd /data/projects/asupersync/formal/lean
lake build

cd /data/projects/asupersync
cargo test --test refinement_conformance -- --nocapture
cargo test --test lean_baseline_report
cargo test --test lean_invariant_theorem_test_link_map
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

If Rust changed, run module-targeted tests for touched areas as well.

Full conformance command set:
- `references/CONFORMANCE-PROCEDURE.md`

## Troubleshooting Quick Map

Use:
- `references/FEEDBACK-EXAMPLES.md` for symptom -> route mapping and stuck-proof cards
- `references/LEAN-PATTERNS.md` for theorem and modeling strategy patterns

## Anti-Patterns

- proving easy lemmas first to inflate theorem count
- declaring "verified" without tier + artifacts
- treating every proof failure as a tactic problem
- skipping witness extraction from stuck proofs
- updating Lean without rerunning conformance tests
- closing beads/issues without proof-carrying record
- changing multiple levers in one loop iteration
- claiming alignment without posterior/routing rationale

## Reference Router

| Need | Load First | Then |
|---|---|---|
| Frog ranking + queue updates | `references/FROG-PRIORITY.md` | this file `Frog Ranking` |
| Full 7-check parity process | `references/CONFORMANCE-PROCEDURE.md` | this file `Conformance and Artifact Closure` |
| Real failure-to-fix examples | `references/FEEDBACK-EXAMPLES.md` | this file Troubleshooting |
| Lean theorem/proof design patterns | `references/LEAN-PATTERNS.md` | this file router + elicitation |
| Artifact hashing and witness pipeline | `references/PROOF-ARTIFACTS.md` | this file artifact contract |

## Reference Index

- Frog ranking and live queue: `references/FROG-PRIORITY.md`
- Conformance protocol and routing matrix: `references/CONFORMANCE-PROCEDURE.md`
- Concrete proof-failure -> bug examples: `references/FEEDBACK-EXAMPLES.md`
- Lean modeling/proof patterns for this codebase: `references/LEAN-PATTERNS.md`
- Proof-carrying artifact spec and workflow: `references/PROOF-ARTIFACTS.md`
