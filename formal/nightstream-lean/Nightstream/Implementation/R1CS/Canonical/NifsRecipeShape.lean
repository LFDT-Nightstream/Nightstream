import Nightstream.Implementation.R1CS.Core.Projection.Polynomial

/-!
Contract: why the `nifsVerify` recipe cannot have the shape `CallRecipe`
currently demands.

Owns: a kernel-checked witness that the projection-root event is not vacuous at
the exact operations F′ uses.

Does not own: the `nifsVerify` row program, the `CallRecipe` contract, or any
probability bound.

## The obstruction

`CallRecipe.activeSoundness` demands an unconditional conclusion — satisfying
the rows gives `callEval call inputs = some outputs`. Every recipe constructed
so far can meet that, because equality, affine maps, zero tests and Poseidon2
are all *exact* arithmetic.

`nifsVerify` is the first call whose soundness is statistical. The honest
statement, `FPrimeConcreteNifs.recursive_rows_nifsVerify_or_badRoot`, is a
disjunction: accepted rows give the decoded accumulator **or** expose
`BatchBadRoot`. That second disjunct is
`¬identity.Exact ∧ eval lhs beta = eval rhs beta` — distinct coefficient vectors
colliding at the sampled challenge. No row program can rule it out, because the
rows only check the evaluation at `beta`; that is the entire point of the
projection optimization.

So `activeSoundness` is dischargeable for `nifsVerify` only if the event is
vacuous. `badRoot_at_production_ops` shows it is not.

## What this witness does and does not establish

It is built at `ProjectionProgram.K.ops` — the real Goldilocks-quadratic
operations, not a surrogate algebra — so the disjunction cannot be collapsed by
any general argument about the arithmetic.

It is a *hand-built* identity, not one arising from `BatchIdentity
recursiveTraces` on a real proof. It therefore says nothing about whether F′ is
attackable, and it is not a production defect. It rules out exactly one thing:
discharging `activeSoundness` by proving the event impossible.

The consequence is a contract question, recorded as
`FPRIME-NIFSVERIFY-SOUNDNESS-SHAPE`: either `CallRecipe` gains a named-event
disjunct, or `nifsVerify` carries a no-bad-root premise. The premise route is
the one §3 forbids — no real consumer can construct it, since constructing it is
the whole difficulty.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.NifsRecipeShape

open Nightstream.SuperNeo.ProjectionCheck
open Nightstream.Implementation.R1CS.ProjectionProgram

/-- Two distinct degree-one coefficient vectors over the production extension:
`X` and `1`. They differ, and they agree at `beta = 1`. -/
def collidingIdentity : Identity K where
  lhs := [K.zero, K.one]
  rhs := [K.one, K.zero]
  beta := K.one
  maxDegree := 1

/-- The identity is well formed: equal fixed widths, within the degree bound.
Without this the witness would be rejected before reaching the root branch, and
would prove nothing about `BadRoot`. -/
theorem collidingIdentity_wellFormed : collidingIdentity.WellFormed := by
  decide

/-- The two vectors are genuinely different, so the collision is between
distinct polynomials rather than a restatement. -/
theorem collidingIdentity_not_exact : ¬ collidingIdentity.Exact := by
  decide

/-- They evaluate equally at the challenge. -/
theorem collidingIdentity_collides :
    eval K.ops collidingIdentity.lhs collidingIdentity.beta
      = eval K.ops collidingIdentity.rhs collidingIdentity.beta := by
  decide

/-- **The projection-root event is not vacuous at production operations.**

`X ≠ 1` as coefficient vectors, yet both evaluate to `1` at `beta = 1`. So an
accepted projection identity genuinely need not be exact, and no proof can
discharge `activeSoundness` for `nifsVerify` by collapsing the disjunction. -/
theorem badRoot_at_production_ops : BadRoot K.ops collidingIdentity where
  wellFormed := collidingIdentity_wellFormed
  notExact := collidingIdentity_not_exact
  collision := collidingIdentity_collides

/-- The witness is also `Accepted`, which is what makes it reach the interface
in question: this is a check the verifier passes, not one it rejects. -/
theorem collidingIdentity_accepted : Accepted K.ops collidingIdentity :=
  ⟨collidingIdentity_wellFormed, collidingIdentity_collides⟩

/-! ## The check is not trivially satisfied

`collidingIdentity_accepted` would be worthless if `Accepted` held of
everything. Perturbing one coefficient of the same fixture breaks acceptance,
so the check does discriminate and the witness above passes a real test. -/

def separatedIdentity : Identity K where
  lhs := [K.zero, K.one]
  rhs := [K.one, K.one]
  beta := K.one
  maxDegree := 1

theorem separatedIdentity_not_accepted : ¬ Accepted K.ops separatedIdentity := by
  decide

/-! ## The event must be bound to the call occurrence

The natural next step is a closed event family keyed on the call:

```lean
inductive CallEvent : Call -> (ColumnId -> Field) -> Prop where
  | nifsBatchBadRoot (identity : Identity K) (bad : BadRoot K.ops identity) :
      CallEvent .nifsVerify assignment
```

Closed, uninhabited for every call but one — and **still an escape hatch**, for a
reason that is easy to miss. `unbound_event_is_inhabited` is the proof: an
event carrying an *unquantified* identity is satisfiable for every assignment,
because `collidingIdentity` witnesses it outright. A `nifsVerify` recipe could
then discharge `activeSoundness` with `Or.inr` and prove nothing at all — the
same defect as the reverted `badEvent := fun _ => True`, one level down.

So the identity must be *bound*: it has to be the identity that this call
occurrence's rows actually checked, derived from the assignment by the row
program's own trace function.

That function does not exist yet. It is defined by the Lean-owned NIFS row
program, which means the event family cannot be defined before that program —
the reverse of the order the queue assumed. This module records why. -/

/-- **An unbound event is free.**  For any assignment whatsoever there is a bad
root, so an event that merely asserts "some identity is a bad root" is no
constraint at all.  The binding to the call occurrence is load-bearing, not
presentational. -/
theorem unbound_event_is_inhabited (assignment : Nat → Nat) :
    ∃ identity : Identity K, BadRoot K.ops identity :=
  ⟨collidingIdentity, badRoot_at_production_ops⟩

end Nightstream.Implementation.R1CS.Canonical.NifsRecipeShape
