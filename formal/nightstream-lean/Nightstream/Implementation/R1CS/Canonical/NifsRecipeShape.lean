import Nightstream.Implementation.R1CS.Core.Projection.Polynomial

/-!
Contract: separate exact operational NIFS verification from its
coefficient-exact semantic refinement.

Owns: a kernel-checked witness that the projection-root event is not vacuous at
the exact operations F′ uses.

Does not own: the `nifsVerify` row program, the `CallRecipe` contract, or any
probability bound.

## The two theorem layers

`CallRecipe.activeSoundness` is an exact operational statement: satisfying
rows must agree with the selected deterministic `callEval`. A projection
collision does not make that verifier nondeterministic; it is one input on
which the verifier accepts. A complete `nifsVerify` row program can and should
therefore satisfy the existing exact `CallRecipe` contract.

The named `BatchBadRoot` event belongs to the subsequent semantic refinement:
operational verifier acceptance implies the paper transition or the exact
identity checked at the sampled point was a bad root. No row program can
collapse that disjunction, because evaluation at `beta` is the optimization
being encoded.

## What this witness does and does not establish

It is built at `ProjectionProgram.K.ops` — the real Goldilocks-quadratic
operations, not a surrogate algebra — so the disjunction cannot be collapsed by
any general argument about the arithmetic.

It is a *hand-built* identity, not one arising from `BatchIdentity
recursiveTraces` on a real proof. It therefore says nothing about whether F′ is
attackable, and it is not a production defect. It rules out exactly one thing:
erasing the event from the verifier-to-paper refinement by proving it
impossible.
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
accepted projection identity genuinely need not be coefficient-exact. -/
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
because `collidingIdentity` witnesses it outright. An event-aware semantic
refinement theorem could then take its event branch for every execution and
prove nothing at all.

So the identity must be *bound*: it has to be the identity that this call
occurrence's rows actually checked, derived from the assignment by the row
program's own trace function.

`KTraceProgram.Occurrence` now supplies this binding for the selected public
PiRLC quotient subprogram, and `KTraceBadRootFixture` proves its exact event
branch is reachable from satisfying rows. A complete operational
`nifsVerify` recipe still needs the remaining PiCCS, PiDEC, transcript,
point-binding, and accumulator programs. Its separate semantic-refinement
theorem must bind this occurrence to that complete call rather than to the
public subtotal alone. -/

/-- **An unbound event is free.**  For any assignment whatsoever there is a bad
root, so an event that merely asserts "some identity is a bad root" is no
constraint at all.  The binding to the call occurrence is load-bearing, not
presentational. -/
theorem unbound_event_is_inhabited (assignment : Nat → Nat) :
    ∃ identity : Identity K, BadRoot K.ops identity :=
  ⟨collidingIdentity, badRoot_at_production_ops⟩

end Nightstream.Implementation.R1CS.Canonical.NifsRecipeShape
