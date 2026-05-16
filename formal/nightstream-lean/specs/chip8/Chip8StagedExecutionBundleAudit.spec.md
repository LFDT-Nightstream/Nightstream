# Chip8StagedExecutionBundleAudit Spec

## Purpose

- **What it is**: The theorem-facing chunk-level audit owner over the Lean-owned
  staged execution digest bundle.
- **Key property**: `stagedExecutionBundleAuditAccepted_of_bundle` accepts one
  normalized chunk bundle and `bundleAuditImpliesEntryBound` recovers the exact
  staged execution digest realization of every bundled entry.
- **Protocol role**: This owner is the chunk-level checker layer above
  `Chip8StagedExecutionDigestBundle`. It does not yet own external
  serialization/import; it owns what acceptance means for one Lean-defined
  chunk artifact. This is the theorem-facing target the later Rust↔Lean bundle
  parity lane should hit.

## Target Formulas

### Entry acceptance

For one bundled frame-digest entry `entry`, define:

$$
\mathrm{EntryAuditAccepted}(entry)
$$

to mean that the slice-scoped audit checker accepts the exact staged digest
 carried by that entry at the exact `(stepIdx, pre, post, dec, row)` surface
 carried by the same entry.

This owner must not weaken entry acceptance to digest-shape compatibility. Entry
acceptance must remain the exact `Chip8ArtifactAudit` predicate.

### Chunk-level checks

For one chunk bundle `bundle`, define:

$$
\mathrm{checkBundlePublicSurface}(bundle)
$$

to recover the exact theorem-facing kernel public-input contract from the
 bundled public surface.

Define:

$$
\mathrm{checkBundleEntries}(bundle)
$$

to require that every bundled entry satisfies `EntryAuditAccepted`.

Define:

$$
\mathrm{checkBundleOrder}(bundle)
$$

to require that projecting bundled entries back to frames recovers the exact
 chunk frame order.

Bundle acceptance is the bundled checker:

$$
\mathrm{StagedExecutionBundleAuditAccepted}(bundle)
$$

which simultaneously validates:

- the exact public-input boundary,
- exact per-entry digest acceptance,
- exact chunk ordering.

### Acceptance from normalized bundles

For any Lean-owned normalized bundle:

$$
\mathrm{stagedExecutionBundleAuditAccepted\_of\_bundle}(bundle)
$$

must hold.

This is not vacuous. It is the theorem that allows later layers to consume one
 chunk object and recover the exact per-entry theorem surfaces without unpacking
 the bundle constructor by hand.

### Entry-level soundness

From accepted chunk-level audit:

$$
\mathrm{bundleAuditImpliesEntryAccepted}(bundle, entry)
$$

must recover slice-scoped audit acceptance for each bundled entry, and:

$$
\mathrm{bundleAuditImpliesEntryBound}(bundle, entry)
$$

must recover the exact staged execution digest realization theorem for that
 entry.

As corollaries:

$$
\mathrm{bundleAuditImpliesEntryExecutionFrameBound}(bundle, entry)
$$

and

$$
\mathrm{bundleAuditImpliesEntryMicrostepCorrect}(bundle, entry)
$$

must recover the exact row-backed execution-frame theorem surface for every
 bundled entry.

### Length and chunk-input consequences

Bundle acceptance must preserve the canonical chunk order, so:

$$
\mathrm{bundleAuditLength\_eq}(bundle)
$$

must recover that bundled digest count equals exact frame count.

Under the simple-kernel chunk input contract:

$$
\mathrm{bundleAuditLength\_eq\_semanticRows}(bundle)
$$

must recover that bundled digest count equals the public semantic-row count.

## Paper Anchors

- **Sources**:
  - `./docs/assurance-strategy.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./formal/nightstream-lean/specs/chip8/Chip8StagedExecutionDigestBundle.spec.md`
  - `./formal/nightstream-lean/specs/chip8/Chip8ArtifactAudit.spec.md`
- Anchors:
  - Layer-3 release gating should consume one explicit Lean-defined chunk
    artifact
  - chunk-level acceptance must recover exact per-slice theorem surfaces
  - bundle order and semantic-row count remain soundness-carrying, not mere
    export convenience

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Kernel/StagedExecutionBundleAudit.lean` | Chunk-level audit acceptance and recovery theorems over one staged digest bundle |
| `Nightstream/Chip8/Kernel/StagedExecutionBundleAuditInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Checker | `EntryAuditAccepted` | def | Definitional | One bundled entry satisfies the exact slice-scoped audit predicate |
| Checker | `checkBundlePublicSurface` | def | Definitional | Bundle public surface recovers the exact kernel public-input contract |
| Checker | `checkBundleEntries` | def | Definitional | Every bundled entry satisfies `EntryAuditAccepted` |
| Checker | `checkBundleOrder` | def | Definitional | Bundle preserves exact frame order |
| Checker | `checkStagedExecutionDigestBundle` | def | Definitional | Bundled chunk-level audit checker |
| Acceptance | `StagedExecutionBundleAuditAccepted` | def | Definitional | Chunk-level acceptance predicate over one staged digest bundle |
| Theorem | `stagedExecutionBundleAuditAccepted_of_bundle` | theorem | Theorem-Target | Any Lean-owned normalized bundle is accepted |
| Theorem | `stagedExecutionBundleAuditAccepted_of_frames` | theorem | Theorem-Target | Any exact authenticated chunk bundle built from frames is accepted |
| Theorem | `bundleAuditImpliesEntryAccepted` | theorem | Theorem-Target | Accepted chunk audit recovers accepted slice audits for bundled entries |
| Theorem | `bundleAuditImpliesEntryBound` | theorem | Theorem-Target | Accepted chunk audit recovers exact staged digest realization for bundled entries |
| Theorem | `bundleAuditImpliesEntryExecutionFrameBound` | theorem | Theorem-Target | Accepted chunk audit recovers exact row-backed execution-frame surfaces |
| Theorem | `bundleAuditImpliesEntryMicrostepCorrect` | theorem | Theorem-Target | Accepted chunk audit recovers row-level semantic correctness |
| Theorem | `bundleAuditLength_eq` | theorem | Theorem-Target | Accepted chunk audit preserves exact frame-count equality |
| Theorem | `bundleAuditLength_eq_semanticRows` | theorem | Theorem-Target | Accepted chunk audit preserves public semantic-row count |

## Proof Obligations

- Do not redefine chunk acceptance in terms weaker than the slice-scoped audit
  predicate already owned by `Chip8ArtifactAudit`.
- Do not drop bundle ordering; chunk-level order is part of the accepted
  boundary.
- Do not silently upgrade this owner into an external serialization checker;
  imported Rust artifacts belong to a later layer above this one.
- Do not treat protocol-binding bundled fields as semantically opaque if they
  are later compared against Rust-generated artifacts.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Kernel/StagedExecutionDigestBundle.lean`
  - `Nightstream/Chip8/Kernel/ArtifactAudit.lean`
  - `Nightstream/Chip8/Trace/ChunkInput.lean`
- **Downstream consumers**:
  - future Rust↔Lean chunk-bundle parity lane
  - future external bundle import/schema checker
  - release gating over the final staged chunk artifact

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Kernel.StagedExecutionBundleAudit` succeeds.
2. Chunk-level acceptance recovers exact per-entry audit acceptance.
3. Chunk-level acceptance recovers exact per-entry theorem surfaces.
4. Chunk-level acceptance preserves exact frame order and semantic-row count.
5. No `sorry`.
