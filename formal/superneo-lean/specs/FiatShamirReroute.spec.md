# FiatShamirReroute

## Purpose

- **What it is**: A deterministic parent-authority Fiat-Shamir boundary for the
  protocol design in which the `Π_RLC` parent claim is the transcript authority
  and `Π_DEC` children are checked as downstream validation witnesses.
- **Key property**: The challenge schedule reads only the parent authority. Child
  payloads cannot change the challenge. At theorem level, the parent-authority
  reroute with DEC validation is equivalent to the `Π_RLC` weak parent statement,
  because `Π_DEC` validation is derived from that weak statement.
- **Protocol role**: This module isolates the proof obligation behind using the
  RLC parent digest as the Fiat-Shamir input while still validating the DEC
  children by recomposition/checking outside the transcript authority path.

## Target Formulas

- `ParentAuthorityFS.challenge(fs, parent) = squeezeChallenge(digestParent(parent))`
- `challenge_independent_of_children`:
  for fixed `fs` and `parent`, any two child payloads induce the same challenge.
- `rlcParentAuthorityStatement(ctx) = piRLCWeakStatement(ctx)`
- `decValidationStatement(ctx) = piDECKnowledgeStatement(ctx)`
- `rlcParentAuthorityWithDecValidation(ctx) =
  rlcParentAuthorityStatement(ctx) ∧ decValidationStatement(ctx)`
- `rlcParentAuthorityWithDecValidation_of_parent`:
  `rlcParentAuthorityStatement(ctx) → rlcParentAuthorityWithDecValidation(ctx)`
- `rlcParentAuthorityWithDecValidation_iff_parent`:
  `rlcParentAuthorityWithDecValidation(ctx) ↔ rlcParentAuthorityStatement(ctx)`
- `parentAuthorityVerifier_continuation`:
  accepted parent-authority executions run continuation checks at the
  parent-derived challenge.
- `rlcParentAuthorityVerifierAccepts`:
  a SuperNeo-specialized verifier shape with `ProtocolTargetContext` as the
  transcript-authoritative parent.
- `rlcParentAuthorityVerifier_sound_from_checked_children`:
  if concrete child validation implies the `Π_DEC` statement, accepted rerouted
  executions establish the rerouted parent-authority statement and continuation.
- `rlcParentAuthorityVerifier_sound_from_parent`:
  accepted rerouted executions are sound from the `Π_RLC` parent authority while
  still checking the child predicate.
- `rlcParentAuthorityVerifier_exact_iff`:
  the exact theorem-level no-payload reroute verifier is equivalent to
  `rlcParentAuthorityWithDecValidation` plus the parent-derived continuation.

## Module Mapping

| Lean file | Role |
|---|---|
| `SuperNeo/FiatShamirReroute.lean` | Deterministic reroute theorem implementation |
| `SuperNeo/FiatShamirRerouteInterface.lean` | Curated theorem-facing interface |

## Contract Surface

| Group | Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Schedule | `ParentAuthorityFS` | structure | Theorem-Target | Digest and challenge are functions of the parent authority |
| Schedule | `ParentAuthorityFS.challenge` | def | Definitional | Parent-derived challenge |
| Schedule | `ParentAuthorityFS.challenge_independent_of_children` | theorem | Theorem-Target | DEC children cannot affect parent-derived challenge |
| Statement | `rlcParentAuthorityStatement` | def | Theorem-Target | Alias of `piRLCWeakStatement` |
| Statement | `decValidationStatement` | def | Theorem-Target | Alias of `piDECKnowledgeStatement` |
| Statement | `rlcParentAuthorityWithDecValidation` | def | Theorem-Target | Parent authority plus DEC validation |
| Theorem | `rlcParentAuthorityWithDecValidation_of_parent` | theorem | Theorem-Target | RLC weak statement derives DEC validation |
| Theorem | `rlcParentAuthorityWithDecValidation_iff_parent` | theorem | Theorem-Target | Reroute statement equivalent to RLC parent authority |
| Verifier | `parentAuthorityVerifierAccepts` | def | Theorem-Target | Generic checked verifier shape |
| Verifier | `parentAuthorityVerifier_continuation` | theorem | Theorem-Target | Continuation sees the parent-derived challenge |
| Verifier | `rlcParentAuthorityVerifierAccepts` | def | Theorem-Target | SuperNeo-specialized parent/children verifier shape |
| Verifier | `rlcParentAuthorityVerifier_sound_from_checked_children` | theorem | Theorem-Target | Checked children plus parent authority imply rerouted soundness |
| Verifier | `rlcParentAuthorityVerifier_sound_from_parent` | theorem | Theorem-Target | Parent authority alone derives DEC validation through existing theorem surface |
| Verifier | `rlcParentAuthorityVerifier_exact_iff` | theorem | Theorem-Target | Exact theorem-level reroute equivalence |

## Assumption Ledger

- The module does not add a random-oracle axiom.
- The module does not assert collision resistance of any digest.
- The module assumes the challenge schedule is defined from the parent authority
  by construction.
- Soundness of `Π_RLC` and `Π_DEC` is imported from the existing theorem surfaces
  in `PiRLC` and `PiDEC`.
- Cryptographic Fiat-Shamir transform soundness remains part of the external
  security model; this module proves the deterministic dependency needed by the
  local protocol redesign.

## Dependency and Consumer Map

Upstream dependencies:
- `SuperNeo/PiDEC.lean`: provides `piDEC_of_weak`.
- `SuperNeo/PiRLC.lean`: provides `piRLCWeakStatement`.

Downstream consumers:
- F' low-norm encoding design can use this module to justify hashing the RLC
  parent claim as transcript authority while checking DEC children separately.
- Rust-refinement and artifact-validation layers can instantiate the generic
  parent/children verifier shape with concrete claim and recomposition checks.
- Concrete verifier implementations should instantiate `childrenValidateAgainstParent`
  with recomposition checks over the DEC children and prove that those checks imply
  `decValidationStatement` for the parent context.

## Quality Expectations

The reroute theorem must keep authority explicit: digests compress authority but
do not create it. The `Π_RLC` parent statement is the authority; `Π_DEC` children
are valid only when checked against that parent.

## Acceptance Criteria

- `lake build SuperNeo.FiatShamirReroute` succeeds.
- `lake build SuperNeo.FiatShamirRerouteInterface` succeeds.
- No `sorry`, `admit`, or extra axiom is introduced.
