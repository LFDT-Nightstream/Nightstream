# Chip8ArtifactAudit Spec

## Purpose

- **What it is**: The Lean-defined audit-checker contract over normalized
  staged execution digests for the final CHIP-8 kernel.
- **Key property**: `artifactAuditSound`: if the audit checker accepts a
  staged execution digest, then that digest satisfies the exact theorem-facing
  realization predicate and therefore carries the semantic facts needed by the
  composition theorem.
- **Protocol role**: This is the theorem-facing audit layer that checks a
  concrete digest instance against the Lean-defined contract. For any imported
  digest field that is protocol-binding, the contract must be compatible with
  the exact Lean-owned Goldilocks / serialization / Poseidon2 semantics of that
  field rather than treating it as an implementation-defined blob.

## Target Formulas

### Audit input

The audit checker consumes one normalized digest instance:

$$
d : \mathrm{StagedExecutionDigest}.
$$

It does not own cryptographic verification. It owns only the semantic audit of
the staged execution boundary after a digest has been produced.

When a staged digest carries protocol-binding hash or challenge fields, those
fields must enter this boundary through an exact Lean-owned import meaning. The
checker may stay separate from the lower-layer cryptographic verifier while
still requiring exact computational agreement on the imported values.

### Executable checker surfaces

Define executable or decidable checks for the digest components:

$$
\mathrm{checkDigestPublicSurface}(d)
$$

$$
\mathrm{checkStage1Surface}(d)
$$

$$
\mathrm{checkStage2Surface}(d)
$$

$$
\mathrm{checkStage3Surface}(d)
$$

$$
\mathrm{checkExecutionResultSurface}(d)
$$

and the bundled checker:

$$
\mathrm{checkStagedExecutionDigest}(d).
$$

The intended meaning is:

- the public check validates the exact theorem-facing public-input boundary
- the Stage-1 check validates the exact Stage-1 digest surface
- the Stage-2 check validates the exact Stage-2 digest surface
- the Stage-3 check validates the exact continuity / bridge digest surface
- the Stage-3 check validates the exact continuity / prepared-step /
  bridge-binding digest surface
- the result check validates the exact semantic-result digest surface

### Audit acceptance predicate

Define:

$$
\mathrm{ArtifactAuditAccepted}(d)
$$

to mean that the bundled checker accepts the digest.

Operational policy about when this predicate is enforced is outside this spec.
This owner defines only what checker acceptance means, not when a build system
must run it.

### Checker soundness

The primary theorem target is:

$$
\mathrm{ArtifactAuditAccepted}(d)
\Longrightarrow
\mathrm{StagedExecutionDigestBound}(d,\dots).
$$

This bridges executable checking to the theorem-facing digest realization
predicate.

### Semantic consequence

Because `StagedExecutionDigestBound` projects back to the exact semantic theorem
surfaces, the checker must also support the corollary:

$$
\mathrm{ArtifactAuditAccepted}(d)
\Longrightarrow
\mathrm{ExecutionResultSurface}(\dots)
$$

for the exact supported-kernel execution slice encoded by `d`, using the
already-owned normalization and projection theorems from
`Chip8StagedExecutionDigest` and `Chip8StepComposition`.

In particular, the checker must support the direct semantic corollary:

$$
\mathrm{ArtifactAuditAccepted}(d)
\Longrightarrow
\mathrm{ExecutionFrameBound}(rom,\sigma,\langle dec, pre, post, z \rangle).
$$

and:

$$
\mathrm{ArtifactAuditAccepted}(d)
\Longrightarrow
\mathrm{MicrostepCorrect}(\dots).
$$

### Audit completeness boundary

The checker is intentionally not a second cryptographic verifier. It is complete
only with respect to the normalized digest contract:

$$
\mathrm{ArtifactAuditAccepted}(d)
$$

means:

- the staged digest is internally coherent,
- each stage surface matches the theorem-owned boundary,
- the stages compose into the semantic result surface.

It does not mean that low-level PCS or transcript verification has been
re-proved inside this module.

It does mean that the imported digest fields must have one exact Lean-owned
meaning. In particular, any imported Poseidon2-derived field must be tied to
the canonical Goldilocks serialization and transcript semantics fixed elsewhere
in the formal stack.

## Paper Anchors

- **Sources**:
  - `./docs/assurance-strategy.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/superneo-paper`
- Anchors:
  - staged proof composition
  - commitment-before-challenge discipline
  - Stage-1 / Stage-2 / Stage-3 ownership split
  - bridge export and prepared-step linkage
  - release-time audit over one explicit digest boundary

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/ArtifactAudit.lean` | Executable/declarative audit checker and soundness theorems over staged execution digests |
| `Nightstream/Chip8/ArtifactAuditInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Checker | `checkDigestPublicSurface` | def | Definitional | Checks the exact public digest surface |
| Checker | `checkStage1Surface` | def | Definitional | Checks the exact Stage-1 digest surface |
| Checker | `checkStage2Surface` | def | Definitional | Checks the exact Stage-2 digest surface |
| Checker | `checkStage3Surface` | def | Definitional | Checks the exact Stage-3 digest surface |
| Checker | `checkExecutionResultSurface` | def | Definitional | Checks the exact result digest surface |
| Checker | `checkStagedExecutionDigest` | def | Definitional | Bundled audit checker over one digest instance |
| Acceptance | `ArtifactAuditAccepted` | def | Definitional | Audit acceptance predicate over one digest instance |
| Theorem | `artifactAuditSound` | theorem | Theorem-Target | Accepted digest instances satisfy the exact theorem-facing digest realization predicate |
| Theorem | `artifactAuditImpliesBridgeBinding` | theorem | Theorem-Target | Accepted digest instances recover the exact row-level bridge-binding audit object |
| Theorem | `artifactAuditImpliesExecutionResultSurface` | theorem | Theorem-Target | Accepted digest instances imply the exact supported-kernel execution-result surface |
| Theorem | `artifactAuditImpliesExecutionFrameBound` | theorem | Theorem-Target | Accepted digest instances imply the exact row-backed execution-frame surface |
| Theorem | `artifactAuditImpliesMicrostepCorrect` | theorem | Theorem-Target | Accepted digest instances imply the exact row-level semantic theorem |

## Proof Obligations

- The checker must be defined over the Lean-owned digest contract, not over a
  Rust-owned export format.
- Checker acceptance must imply the exact theorem-facing digest realization
  predicate.
- The checker must preserve the Stage-1 / Stage-2 / Stage-3 ownership split.
- The checker must not silently assume low-level cryptographic verification that
  is not already part of the imported boundary.
- A slice-scoped audit owner must not silently upgrade one accepted digest into
  a whole authenticated execution trace.
- The import meaning of any protocol-binding digest or challenge field must be
  exact enough to support deterministic Rust↔Lean equality checks and
  golden-vector conformance.

## Assumption Ledger

- Cryptographic verification and transcript checking are imported or external.
- Serialization from external artifact format into `StagedExecutionDigest` is
  external to this module.
- This owner checks the staged semantic boundary, not the whole prover stack.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/StagedExecutionDigest.lean`
  - `Nightstream/Chip8/StepComposition.lean`
  - `Nightstream/Chip8/ExecutionSemantics.lean`
- **Downstream consumers**:
  - release-qualification artifact audits
  - later Rust-refinement theorems for accepted staged digests

## Implementation Plan

1. Define the checker surfaces over the normalized digest contract.
2. Define `ArtifactAuditAccepted`.
3. Prove checker soundness into `StagedExecutionDigestBound`.
4. Prove the semantic consequence theorem into the execution theorem surface.

## Quality Expectations

- Keep the checker contract narrow and theorem-owned.
- Keep executable checks aligned with the exact theorem-facing digest boundary.
- Separate digest checking from external serialization and cryptographic
  verification.
- Keep imported protocol-binding fields exact and executable enough for
  near-`1:1` Rust↔Lean compatibility testing.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.ArtifactAudit` succeeds.
2. Accepted digest instances imply the exact theorem-facing digest realization
   predicate.
3. Accepted digest instances imply the exact supported-kernel semantic result.
4. No `sorry`.

## Out of Scope

- production runtime policy
- CI policy
- Rust feature policy
- low-level cryptographic verification
