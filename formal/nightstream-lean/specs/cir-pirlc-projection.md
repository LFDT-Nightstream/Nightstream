# CIR-PIRLC-PROJECTION

```text
property_id: CIR-PIRLC-PROJECTION
frog_id: cir.pirlc.projection_exact_or_bad_root
route: theorem-first
claim:
  A fixed-width bounded polynomial identity accepted at beta is either equal
  coefficient-wise or beta is a root of a nonzero error polynomial. The batch
  theorem returns all identities exact or a concrete member carrying BadRoot.

  The exact production helper rows make the disjunction observably necessary:
  one honest witness is coefficient-exact, while E(X)=X-7 with beta=7 passes
  the same 714 rows despite a false full ring-action mix.

  For every canonical assignment, satisfaction of those 714 exact exported
  rows now implies `BatchAccepted`; this is a quantified artifact theorem, not
  a golden-vector premise. The reusable census theorem lifts the semantic
  argument across any list of traces, including shared ladder/rho definitions.

  The exact recursive profile separately pins 31 production identities, 15
  pairs per identity, one 1,892-row shared block, and 59,396 identity rows.
  All 31 identity ranges have the same sparse program shape up to wire renaming.
surface:
  - engine/r1cs_circuit/ring_action.rs::enforce_ring_action_projection_batch
  - paper/nifs/circuit/pi_rlc/projection/** PiRLC projection schedule
failure_class:
  A one-point identity is treated as deterministic coefficient equality,
  omitting the Schwartz-Zippel/Fiat-Shamir bad-event branch.
math:
  - Fixed-width constant-first coefficient lists.
  - Horner evaluation under explicit operations.
  - Accepted implies Exact or BadRoot, pointwise and for a batch.
  - Exact Karatsuba, power-ladder, polynomial-evaluation, quotient-times-phi,
    and terminal equality rows imply Accepted.
artifact:
  assurance/pi-rlc-projection-boundary.json
lean_theorems:
  - Nightstream.Implementation.R1CS.PiRLCProjection.exactRows_imply_batchAccepted
  - Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.census_batchAccepted
  - Nightstream.SuperNeo.ProjectionCheck.accepted_implies_exact_or_badRoot
  - Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot
  - Nightstream.Assurance.FPrimeRecursiveCircuit.projectedChecks_local_sound_or_badRoot
evidence_state: artifact-checked
assumptions:
  - None for the deterministic exact-or-bad theorem.
  - Bounding BadRoot requires beta sampled after the bounded identity is bound.
non_goals:
  - A BadRoot probability bound.
  - Poseidon2 Fiat-Shamir indifferentiability.
  - SIS binding of the projection-preimage digest.
  - The remaining global PiRLC-range-to-`ProjectionTrace` extraction theorem.
  - PiRLC completeness.
frog_score:
  impact: 5
  bug_prior: 4
  runtime_reach: 5
  proof_cost: 5
  model_cost: 4
  ev_frog: 5.0
history_signals:
  - e2b4ffb introduced the projection schedule and raises theorem-drift prior.
  - Extensive native PiRLC mutation tests lower the prior of a native verifier
    omission but do not cover circuit-to-native equivalence.
  - The steady-recursive manifest attributes 1,836,082 rows to PiRLC and exposes the
    missing deterministic/probabilistic split.
```
