# Chip8KernelArtifactAudit Spec

## Purpose

- **What it is**: The Lean-defined audit-checker contract over one normalized
  authenticated CHIP-8 kernel chunk digest.
- **Key property**: `kernelArtifactAuditSound`: if the audit checker accepts a
  kernel digest, then that digest satisfies the exact theorem-facing
  `Chip8KernelExecutionDigest` realization predicate and therefore recovers the
  full kernel conclusion bundle.
- **Protocol role**: This is the theorem-facing release-audit layer above
  `Chip8KernelExecutionDigest`. It checks the kernel-level boundary. It does
  not re-own cryptographic verification.

## Target Formulas

### Audit input

The audit checker consumes one normalized kernel digest:

$$
d : \mathrm{KernelExecutionDigest}.
$$

It checks the staged kernel boundary after a digest has already been produced.

### Checker surfaces

Define checks for the grouped kernel surfaces:

$$
\mathrm{checkKernelTraceSurface}(d)
$$

$$
\mathrm{checkKernelExportSurface}(d)
$$

$$
\mathrm{checkKernelAuditSurface}(d)
$$

$$
\mathrm{checkKernelManifestSurface}(d)
$$

$$
\mathrm{checkKernelTranscriptSurface}(d)
$$

$$
\mathrm{checkKernelErrorSurface}(d)
$$

and the bundled checker:

$$
\mathrm{checkKernelExecutionDigest}(d).
$$

The intended meaning is:

- the trace check validates the authenticated chunk-trace surface together with
  the named authenticated temporal-support bundle and exact execution
  correctness, including the explicit temporal-consistency surface that feeds
  whole-trace linking
- the export check validates exact prepared-step export
- the audit check validates exact row-projection and bridge-binding summaries
- the manifest check validates exact `root0` commitment discipline
- the transcript check validates exact transcript-order consequences
- the error check validates negligible total soundness error

### Audit acceptance predicate

Define:

$$
\mathrm{KernelArtifactAuditAccepted}(d)
$$

to mean that the bundled kernel audit checker accepts the digest.

Operational policy about when this checker runs is outside this spec. This
owner defines only the mathematical meaning of acceptance.

### Checker soundness

The primary theorem target is:

$$
\mathrm{KernelArtifactAuditAccepted}(d)
\Longrightarrow
\mathrm{KernelExecutionDigestBound}(d,\dots).
$$

### Semantic consequence

Because the kernel digest projects back to the full kernel conclusion bundle,
the audit checker must also support:

$$
\mathrm{KernelArtifactAuditAccepted}(d)
\Longrightarrow
\mathrm{KernelSoundnessConclusion}(\dots).
$$

and direct corollaries to:

$$
\mathrm{AuthenticatedChunkTraceBound}(frames),
$$

$$
\mathrm{AuthenticatedTemporalSupportBound}(frames),
$$

$$
\mathrm{kernelPreparedSteps}(frames).length = meta.semanticRows
\;\land\;
\mathrm{PreparedStepTraceBound}(
  \mathrm{traceOf}(frames),
  \mathrm{kernelPreparedSteps}(frames)
),
$$

$$
\mathrm{rowProjectionSummary}(frames)
\;\land\;
\mathrm{bridgeBindingSummary}(frames),
$$

and:

$$
\mathrm{IsNegligible}(accounting.\varepsilon_{\mathrm{total}}).
$$

The audit checker must also expose the exact adjacent-state link theorem
directly:

$$
\mathrm{KernelArtifactAuditAccepted}(d)
\Longrightarrow
\mathrm{TraceLinkBound}(\mathrm{traceOf}(frames)).
$$

and its raw execution-linked form:

$$
\mathrm{KernelArtifactAuditAccepted}(d)
\Longrightarrow
\mathrm{ExecutionLinked}(\mathrm{traceOf}(frames)).
$$

The audit checker must also expose direct exact execution correctness:

$$
\mathrm{KernelArtifactAuditAccepted}(d)
\Longrightarrow
\mathrm{ExecutionCorrect}(rom,\sigma,init,\mathrm{traceOf}(frames)).
$$

### Audit completeness boundary

This checker is intentionally not a second cryptographic verifier. It is
complete only with respect to the Lean-defined kernel digest contract.

## Paper Anchors

- **Sources**:
  - `./docs/assurance-strategy.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
- Anchors:
  - release-time audit runs over a Lean-defined kernel digest
  - audit targets the actual kernel boundary, not only one row slice
  - accepted kernel digests recover the strong execution-trace theorem owned by
    the kernel boundary through the same temporal-consistency path

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/KernelArtifactAudit.lean` | Audit checker and soundness theorems over one normalized kernel digest |
| `Nightstream/Chip8/KernelArtifactAuditInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Checker | `checkKernelTraceSurface` | def | Definitional | Checks the authenticated chunk-trace surface together with the exact per-row Stage-2 temporal seed summary and the named authenticated temporal-support bundle (chunk-global Stage-2 temporal context plus Stage-3 `pc` bridge) |
| Checker | `checkKernelExportSurface` | def | Definitional | Checks exact prepared-step export |
| Checker | `checkKernelAuditSurface` | def | Definitional | Checks exact row-projection and bridge-binding summaries |
| Checker | `checkKernelManifestSurface'` | def | Definitional | Checks exact `root0` commitment discipline |
| Checker | `checkKernelTranscriptSurface` | def | Definitional | Checks exact transcript-order consequences |
| Checker | `checkKernelErrorSurface` | def | Definitional | Checks negligible total soundness error |
| Checker | `checkKernelExecutionDigest` | def | Definitional | Bundled audit checker over one kernel digest instance |
| Acceptance | `KernelArtifactAuditAccepted` | def | Definitional | Audit acceptance predicate over one kernel digest instance |
| Theorem | `kernelArtifactAuditSound` | theorem | Theorem-Target | Accepted kernel digests satisfy the exact theorem-facing kernel digest realization predicate |
| Theorem | `kernelArtifactAuditImpliesKernelSoundnessConclusion` | theorem | Theorem-Target | Accepted kernel digests imply the full kernel conclusion bundle |
| Theorem | `kernelArtifactAuditImpliesAuthenticatedChunkTrace` | theorem | Theorem-Target | Accepted kernel digests imply the authenticated chunk-trace surface |
| Theorem | `kernelArtifactAuditImpliesStage2TemporalSeeds` | theorem | Theorem-Target | Accepted kernel digests imply the exact per-row Stage-2 temporal seed summary |
| Theorem | `kernelArtifactAuditImpliesTemporalSupport` | theorem | Theorem-Target | Accepted kernel digests imply the named authenticated temporal-support bundle |
| Theorem | `kernelArtifactAuditImpliesAuthenticatedExecutionTrace` | theorem | Theorem-Target | Accepted kernel digests imply the exact authenticated execution-trace bundle used by the final soundness corollaries |
| Theorem | `kernelArtifactAuditImpliesPreparedStepExport` | theorem | Theorem-Target | Accepted kernel digests imply exact prepared-step export |
| Theorem | `kernelArtifactAuditImpliesRowProjectionSummary` | theorem | Theorem-Target | Accepted kernel digests imply the exact row-projection audit summary |
| Theorem | `kernelArtifactAuditImpliesBridgeBindingSummary` | theorem | Theorem-Target | Accepted kernel digests imply the exact bridge-binding audit summary |
| Theorem | `kernelArtifactAuditImpliesKernelClaimsFixedInRoot0` | theorem | Theorem-Target | Accepted kernel digests imply the exact `root0` commitment-fixing law |
| Theorem | `kernelArtifactAuditImpliesKernelRootCommitmentsDisjoint` | theorem | Theorem-Target | Accepted kernel digests imply the exact kernel/root commitment disjointness law |
| Theorem | `kernelArtifactAuditImpliesChallengeAfterPhase0` | theorem | Theorem-Target | Accepted kernel digests imply the challenge-after-phase-0 transcript law |
| Theorem | `kernelArtifactAuditImpliesStage1TerminalAfterPhase0` | theorem | Theorem-Target | Accepted kernel digests imply the Stage-1 terminal-point transcript law |
| Theorem | `kernelArtifactAuditImpliesStage2TerminalAfterPhase0` | theorem | Theorem-Target | Accepted kernel digests imply the Stage-2 terminal-point transcript law |
| Theorem | `kernelArtifactAuditImpliesRowBindingCoverage` | theorem | Theorem-Target | Accepted kernel digests imply exact row-binding coverage for semantic rows |
| Theorem | `kernelArtifactAuditImpliesEmitKernelOpeningClaimsLast` | theorem | Theorem-Target | Accepted kernel digests imply the final emit placement law |
| Theorem | `kernelArtifactAuditImpliesNegligibleTotal` | theorem | Theorem-Target | Accepted kernel digests imply negligible total soundness error |
| Theorem | `kernelArtifactAuditImpliesTraceLinkBound` | theorem | Theorem-Target | Accepted kernel digests imply the exact adjacent-state link contract directly |
| Theorem | `kernelArtifactAuditImpliesExecutionLinked` | theorem | Theorem-Target | Accepted kernel digests imply the raw execution-linked trace law directly |
| Theorem | `kernelArtifactAuditImpliesExecutionCorrect` | theorem | Theorem-Target | Accepted kernel digests imply exact chunk execution correctness directly at the audited kernel boundary |

## Proof Obligations

- The checker must be defined over the Lean-owned kernel digest contract, not a
  Rust-owned export format.
- Checker acceptance must imply the exact theorem-facing kernel digest
  realization predicate.
- The checker must land in the actual top-level kernel theorem surface, not
  only in a weaker slice-level result.
- It must not weaken the kernel boundary by treating `ExecutionCorrect` as an
  optional corollary outside audit acceptance.
- The audited trace surface must remain explicit enough for a human auditor to
  see that whole-trace linking came from temporal `pc` / register / RAM
  consistency, not from Stage-3 continuity alone.
