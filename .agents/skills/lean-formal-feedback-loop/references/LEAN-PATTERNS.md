# Lean Patterns for Frog-First Proofs

> Focus: patterns that maximize bug-finding power, not theorem count.

## Setup

```bash
cd /data/projects/asupersync/formal/lean
lake build
```

Use these patterns only after identifying a specific frog and failure class.

## Failure Class -> Technique Router

Use this table when picking proof strategy from a blocker classification:

| Failure Class | Technique | Lean Encoding |
|---|---|---|
| Data race / lost update | Linearizability | `ConcurrentStep` inductive with `LinWitness` |
| State leak (obligations) | Linear logic / resource accounting | obligation lifecycle chain, ledger emptiness |
| Protocol violation (cancel) | Temporal logic / LTL | potential function + termination proof |
| Capability escape | Information flow | Cx-threaded step constructors |
| Deadlock | Lock ordering / partial order | antisymmetric acquisition order |
| Quiescence failure | Bisimulation | quiescent predicate decomposition |
| TOCTOU | Atomicity / serializability | transaction-like step bracketing |

Pick by failure class, not by what is easiest to prove locally.

## Pattern 1: Linearization Witness First

Before proving safety invariants on concurrent primitives, define explicit linearization witness:

```lean
structure LinWitness where
  opId : Nat
  linStep : Nat
  preState : State
  postState : State

def has_linearization (trace : List Step) (op : Operation) : Prop := ...
```

Why: many high-value bugs are exactly "no valid linearization point exists."

## Pattern 2: Counterexample-Carrying Development

When a proof blocks, extract a concrete trace hypothesis immediately:

```lean
def badTraceCandidate : List Step := [...]
theorem candidate_violates_property : violates badTraceCandidate targetProp := by
  ...
```

Then mirror in Rust lab-runtime tests. This is the fastest path to code-first decisions.

## Pattern 3: Stuttering Refinement for Split State

For flat-model vs split-table drift:

```lean
def refines (split : SplitState) (flat : State) : Prop := ...
theorem step_refinement :
  SplitStep s s' -> refines s f -> exists f', Step f f' /\ refines s' f' := by
  ...
```

Allow stuttering steps when one Rust transition spans multiple abstract steps.

## Pattern 4: Rely/Guarantee Skeleton for Interleavings

Use rely/guarantee for lock+atomic mixtures:

```lean
def Rely (s s' : SyncState) : Prop := ...
def Guarantee (s s' : SyncState) : Prop := ...

theorem op_preserves_under_rely :
  RelyStar s t -> OpStep t t' -> Guarantee t t' -> Inv t' := by
  ...
```

This avoids pretending the world is sequential where it is not.

## Pattern 5: Potential Functions With Debt Terms

Plain task-state potential is often too weak. Add debt terms:
- outstanding obligations
- undrained loser count
- pending finalizers

```lean
def cancelDebt (s : State) : Nat := ...
theorem drain_step_decreases : Step s a s' -> cancelDebt s' < cancelDebt s := by
  ...
```

## Pattern 6: Invariant Factoring + Recomposition

Split large invariants into independent lemmas (structural wellformedness, ownership uniqueness, obligation accounting, close/quiescence implications), then recompose. Keeps failures diagnostic.

## Pattern 7: Ghost Fields For Concurrency Proofs

Introduce ghost-only metadata (enqueue sequence numbers, lock acquisition epochs, notifier tokens) when needed. Document clearly as proof scaffolding with no runtime effect.

## Pattern 8: Progress Proofs Need Fairness Scope

Liveness claims must state fairness assumptions explicitly (`axiom weakFair`). No implicit fairness assumptions.

## Pattern 9: Theorem Scope Contract

Every theorem should include:
- assumptions
- non-goals
- mapped Rust surfaces

Template:

```lean
/-!
theorem: race_loser_drained_before_return
assumes: cancel signal propagation + weak fairness
non-goals: lock-level starvation under adversarial scheduler
maps-to: src/combinator/race.rs
-/
```

## Pattern 10: Proof Friction Classification

Tag each hard blocker: `linearization_gap`, `state_abstraction_gap`, `premise_gap`, `tactic_debt`. Only `tactic_debt` may be resolved without reopening frog route.

Combined blocker + Rust mapping record:

```text
blocker_id:
class: linearization_gap|state_abstraction_gap|premise_gap|tactic_debt
first_seen_in_theorem:
rust_surface_guess:
current_route_hypothesis:
next_discriminator:
lean_location:
expected_interleaving:
rust_test_target:
lab_seed:
```

## Tactic Guidance (Minimal)

Prefer: `cases` on step constructors, `simp only [...]`, local helper lemmas before global theorem, small arithmetic lemmas + `omega`, tiny frame lemmas first then reuse, one constructor case at a time for preservation proofs.

Avoid: giant unscoped `simp`, helper lemmas with hidden assumptions.

## Pattern 11: Candidate Linearization Enumeration

For lock+atomic designs, explicitly enumerate candidate linearization points and reject non-viable ones:

```text
candidate_lps:
- before_atomic_read
- after_atomic_read
- inside_locked_update
- before_wake_emit
viable_lp:
- inside_locked_update
reason:
- only point preserving waiters/counter consistency across interleavings
```

This prevents vague "should be linearizable" reasoning.

## Pattern 12: Two-Layer Invariant Strategy

Layer 1: local invariant per primitive (notify/semaphore/pool). Layer 2: global runtime invariant (quiescence/no-leak/cancel closure). Prove Layer 1 first, connect Layer 2 via bridge lemmas.

## Pattern 13: Drift-Resistant Theorem Naming

Prefer names encoding contract + scope: `notify_no_lost_wakeup_under_lock_coupling`, `race_loser_drained_before_return`. Avoid vague names like `notify_correct`.
