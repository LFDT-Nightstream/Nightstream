# Chip8BridgeBinding Spec

## Purpose

- **What it is**: The exact per-row audit/provenance owner tying an
  authenticated Stage-3 row-binding claim to the exported prepared step for
  that row.
- **Key property**: `exists_bridgeBindingWitness_of_exactEvidence`: exact
  authenticated staged evidence determines an explicit bridge-binding witness
  for the row currently being exported.
- **Protocol role**: This is the theorem-facing owner for the audit objects
  called out by the CHIP-8 kernel spec between Stage 3 row claims and exported
  bridge payloads. Its leaves must point not only to the direct row-binding
  claim but also to the verified PCS refinement path for that claim. It does
  not re-own Stage-1 / Stage-2 / Stage-3 semantic closure.

## Target Formulas

### Row-projection witness

The kernel audit trail needs an explicit authenticated row-projection witness.
This owner must expose that witness from exact semantic evidence:

$$
\mathrm{ExactSemanticEvidenceCovered}(\dots)
\Longrightarrow
\exists \Gamma_1,\ row,\ ref,\
\mathrm{RowProjectionWitness}(\Gamma_1, ref, row).
$$

This is a provenance object. It is not a new semantic theorem.

The stronger theorem-facing packaging is one explicit `ProjectedRowPath`
carrying:

- the exact accepted direct opening for the row-binding claim,
- the row/view projection witness,
- the semantic row-consistency proof,
- the exact authenticated `rowClaim`,
- and the theorem that this same accepted row-opening path binds the projected
  semantic row.

### Bridge-binding witness

Define:

$$
\mathrm{BridgeBindingWitness}(stepIdx, z, rowClaim, preparedStep)
$$

to mean the conjunction of:

- one explicit `AcceptedDirectOpening` whose direct claim is exactly
  `rowClaim.openingClaim`
- `rowClaim.rowIndex = stepIdx`
- `RowBound(rowClaim, z)`
- `PreparedStepBound(z, preparedStep)`

This owner packages exactly the explicit per-row audit object connecting the
authenticated row-binding claim to the exact prepared-step artifact supplied by
the caller. When downstream digest or audit owners already carry an exported
prepared-step object, bridge binding must target that exact artifact rather
than a locally recomputed placeholder.

On the simple-kernel boundary, this is the root-facing binding surface. The
root opening manifest remains empty there; root-side handoff is modeled through
the exact prepared-step export plus this bridge leaf, not through a parallel
root-opening schema.

Define the stronger theorem-facing bridge bundle:

$$
\mathrm{BridgeBindingBundle}(\Gamma_1, stepIdx, pre, post, dec, z, preparedStep)
$$

to package:

- one explicit authenticated row projection carrying the exact accepted
  row-opening path for that row
- one `RowConsistent(row, z, dec, pre, post, stepIdx)` proof tying that row
  projection to the semantic row `z`
- one `BridgeBindingWitness(stepIdx, z, rowClaim, preparedStep)`
- one explicit identity witnessing that the row-projection path and the bridge
  witness reuse the same accepted direct opening

This is the row-local object that proves the prepared-step artifact is bound to
the exact same accepted row-opening path used by semantic extraction. The
same-path identity is not merely prose: the theorem-facing bridge bundle must
carry one shared `AcceptedDirectOpening`, and both the row-projection path and
the prepared-step bridge witness must be parameterized by that exact accepted
opening rather than merely by the same `rowClaim`.

### Existence from authenticated evidence

The main theorem target is:

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\forall preparedStep,\
\mathrm{PreparedStepBound}(z, preparedStep)
\Longrightarrow
\exists rowClaim,\
\mathrm{BridgeBindingWitness}(stepIdx, z, rowClaim, preparedStep).
$$

and likewise for `ExactSemanticEvidenceCovered`.

The stronger audit-facing existence theorem is:

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\forall preparedStep,\
\mathrm{PreparedStepBound}(z, preparedStep)
\Longrightarrow
\exists \Gamma_1,\ bundle,\
\mathrm{BridgeBindingBundle}(\Gamma_1, stepIdx, pre, post, dec, z, preparedStep).
$$

### Projection theorems

From one realized bridge-binding witness, downstream users must be able to
recover:

$$
\mathrm{BridgeBindingWitness}(\dots)
\Longrightarrow
\mathrm{RowBound}(rowClaim, z)
$$

and:

$$
\mathrm{BridgeBindingWitness}(\dots)
\Longrightarrow
\mathrm{PreparedStepBound}(z, preparedStep).
$$

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./docs/assurance-strategy.md`
- Anchors:
  - Stage-3 row binding
  - exported prepared-step bridge payload
  - explicit row-projection / bridge-binding audit trail

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Kernel/BridgeBinding.lean` | Per-row projection and bridge-binding audit witnesses |
| `Nightstream/Chip8/Kernel/BridgeBindingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Witness | `BridgeBindingWitness` | structure | Definitional | Packages the exact row-claim and prepared-step audit object for one exported row |
| Witness | `ProjectedRowPath` | structure | Definitional | Packages the row projection together with the exact accepted row-opening path used by semantic extraction |
| Witness | `BridgeBindingBundle` | structure | Definitional | Packages the bridge witness together with the exact accepted row-opening path reused by the authenticated row projection |
| Theorem | `rowBound_of_bridgeBinding` | theorem | Theorem-Target | Recovers the authenticated row-binding theorem |
| Theorem | `preparedStepBound_of_bridgeBinding` | theorem | Theorem-Target | Recovers the exported prepared-step theorem |
| Theorem | `rowBound_of_bridgeBindingBundle` | theorem | Theorem-Target | Recovers the authenticated row-binding theorem from the stronger bridge bundle |
| Theorem | `preparedStepBound_of_bridgeBindingBundle` | theorem | Theorem-Target | Recovers the exported prepared-step theorem from the stronger bridge bundle |
| Theorem | `exists_rowProjection_of_semanticEvidence` | theorem | Theorem-Target | Authenticated semantic evidence exports the row-projection witness |
| Theorem | `exists_rowProjection_of_exactEvidence` | theorem | Theorem-Target | Exact authenticated evidence exports the row-projection witness |
| Theorem | `exists_bridgeBindingWitness_of_semanticEvidence` | theorem | Theorem-Target | Authenticated semantic evidence exports the bridge-binding witness |
| Theorem | `exists_bridgeBindingWitness_of_exactEvidence` | theorem | Theorem-Target | Exact authenticated evidence exports the bridge-binding witness |
| Theorem | `exists_bridgeBindingBundle_of_semanticEvidence` | theorem | Theorem-Target | Authenticated semantic evidence exports the stronger row-projection-linked bridge bundle |
| Theorem | `exists_bridgeBindingBundle_of_exactEvidence` | theorem | Theorem-Target | Exact authenticated evidence exports the stronger row-projection-linked bridge bundle |

## Proof Obligations

- Keep row projection and bridge binding separate; they are distinct audit
  objects in the kernel spec.
- Do not smuggle new semantic closure into this owner.
- The witness must tie directly to the authenticated `RowBound` and the
  caller-supplied `PreparedStepBound`.
- The existence theorems must be parametric in the actual prepared-step
  artifact supplied by downstream digest or audit owners; they must not force
  `mkPreparedStep(z)` as the theorem target.
- The stronger bridge bundle must include the same authenticated row-projection
  path used by semantic extraction, not merely an existentially recomputed row
  or a bundle that only reuses the same `rowClaim`.
- This owner must remain row-local; it must not silently upgrade one witness
  into a whole-trace theorem.

## Assumption Ledger

- Lower-layer PCS opening refinement and semantic evidence extraction are
  imported.
- The concrete operational policy for when these audit objects are checked is
  outside this module.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Stage2/EvidenceCoverage.lean`
  - `Nightstream/Chip8/Stage3/ContinuityBridge.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/Kernel/StagedExecutionDigest.lean`
  - `Nightstream/Chip8/Kernel/ArtifactAudit.lean`
  - later Rust/Lean artifact refinement work
