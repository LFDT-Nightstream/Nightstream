# Chip8KernelExecutionDigest Spec

## Purpose

- **What it is**: The theorem-facing normalized digest contract for one
  authenticated CHIP-8 kernel chunk.
- **Key property**: `kernelExecutionDigest_of_conclusion` and
  `kernelExecutionDigest_of_acceptance` normalize the exact top-level
  `Chip8KernelSoundness` bundle into one explicit kernel-level digest shared by
  differential testing and release-time audit.
- **Protocol role**: This owner sits above `Chip8KernelSoundness`. It does not
  define new semantic facts; it repackages the already-owned kernel conclusion
  into one auditor-friendly digest boundary.

## Target Formulas

### Kernel digest ownership

The kernel digest packages the exact top-level theorem surfaces already owned by
the kernel proof:

- named authenticated temporal support for strong trace linking
- exact chunk execution correctness
- authenticated chunk-trace closure
- exact prepared-step export
- exact row-projection and bridge-binding audit summaries
- exact `root0` commitment discipline
- exact transcript-order consequences
- negligible total soundness error

This owner does not weaken or strengthen those facts. It normalizes them.

### Grouped digest surfaces

Define grouped surfaces for one kernel chunk:

$$
\mathrm{KernelTraceSurface}(frames)
$$

to mean:

$$
\mathrm{AuthenticatedChunkTraceBound}(frames)
\land
\mathrm{AuthenticatedTemporalSupportBound}(frames)
\land
\mathrm{ExecutionCorrect}(rom,\sigma,init,\mathrm{traceOf}(frames))
$$

$$
\mathrm{KernelExportSurface}(frames)
$$

to mean:

$$
\mathrm{kernelPreparedSteps}(frames).length = meta.semanticRows
\;\land\;
\mathrm{PreparedStepTraceBound}(
  \mathrm{traceOf}(frames),
  \mathrm{kernelPreparedSteps}(frames)
)
$$

$$
\mathrm{KernelAuditSurface}(frames)
$$

to mean the exact row-projection summary and bridge-binding summary exported for
the authenticated frame list. The bridge-binding summary must quantify over the
actual prepared-step artifact paired with each exported row, not a locally
recomputed stand-in.

$$
\mathrm{KernelManifestSurface}(kernelManifest, rootManifest)
$$

to mean the exact `root0` commitment-fixing and kernel/root separation laws.

$$
\mathrm{KernelTranscriptSurface}(meta.semanticRows, events)
$$

to mean the exact challenge ordering, terminal-point ordering, row-binding
coverage, and final emit placement exported by the transcript owner.

$$
\mathrm{KernelErrorSurface}(accounting)
$$

to mean:

$$
\mathrm{IsNegligible}(accounting.\varepsilon_{\mathrm{total}})
$$

### Bundled digest

Define one normalized kernel digest:

$$
\mathrm{KernelExecutionDigest}.
$$

Define its realization predicate:

$$
\mathrm{KernelExecutionDigestBound}(d,\dots)
$$

to mean the conjunction of the grouped kernel surfaces above for one exact
kernel chunk.

### Normalization theorems

The primary normalization theorem is:

$$
\mathrm{KernelSoundnessConclusion}(\dots)
\Longrightarrow
\exists d,\; \mathrm{KernelExecutionDigestBound}(d,\dots).
$$

The acceptance-level corollary is:

$$
\mathrm{KernelSoundnessAccepted}(\dots)
\Longrightarrow
\exists d,\; \mathrm{KernelExecutionDigestBound}(d,\dots).
$$

This keeps the digest Lean-defined and tied to the exact top-level kernel
theorem surface rather than to an implementation-local export format.

### Projection theorems

From one realized kernel digest, downstream users must be able to recover the
exact kernel theorem surfaces:

$$
\mathrm{KernelExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{AuthenticatedChunkTraceBound}(frames)
$$

$$
\mathrm{KernelExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{stage2TemporalSeedSummary}(frames)
$$

$$
\mathrm{KernelExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{AuthenticatedTemporalSupportBound}(frames)
$$

$$
\mathrm{KernelExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{TraceLinkBound}(\mathrm{traceOf}(frames))
$$

$$
\mathrm{KernelExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{ExecutionLinked}(\mathrm{traceOf}(frames))
$$

$$
\mathrm{KernelExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{ExecutionCorrect}(rom,\sigma,init,\mathrm{traceOf}(frames))
$$

$$
\mathrm{KernelExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{kernelPreparedSteps}(frames).length = meta.semanticRows
\;\land\;
\mathrm{PreparedStepTraceBound}(
  \mathrm{traceOf}(frames),
  \mathrm{kernelPreparedSteps}(frames)
)
$$

$$
\mathrm{KernelExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{rowProjectionSummary}(frames)
\;\land\;
\mathrm{bridgeBindingSummary}(frames)
$$

$$
\mathrm{KernelExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{IsNegligible}(accounting.\varepsilon_{\mathrm{total}})
$$

and, in bundled form:

$$
\mathrm{KernelExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{KernelSoundnessConclusion}(\dots).
$$

## Paper Anchors

- **Sources**:
  - `./docs/assurance-strategy.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
- Anchors:
  - Lean defines the digest contract
  - the digest follows the actual top-level kernel theorem surface
  - strong trace correctness flows through explicit temporal consistency before
    whole-trace link normalization
  - release-time audit should target the kernel boundary, not only one row slice

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/KernelExecutionDigest.lean` | Normalized digest contract for one authenticated kernel chunk |
| `Nightstream/Chip8/KernelExecutionDigestInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Digest | `KernelTraceSurface` | def/structure | Definitional | Packages the authenticated chunk-trace surface together with the named authenticated temporal-support bundle (chunk-global Stage-2 temporal context plus Stage-3 `pc` bridge) |
| Digest | `KernelExportSurface` | def/structure | Definitional | Packages exact prepared-step export |
| Digest | `RowProjectionSummaryEntry` | def | Definitional | One row-local row-projection audit summary entry |
| Digest | `BridgeBindingSummaryEntry` | def | Definitional | One row-local bridge-binding audit summary entry |
| Digest | `KernelAuditSurface` | def/structure | Definitional | Packages exact row-projection and bridge-binding audit summaries |
| Digest | `KernelManifestSurface` | def/structure | Definitional | Packages exact `root0` commitment discipline |
| Digest | `KernelTranscriptSurface` | def/structure | Definitional | Packages exact transcript-order consequences |
| Digest | `KernelErrorSurface` | def/structure | Definitional | Packages negligible total soundness error |
| Bundle | `KernelExecutionDigest` | def/structure | Definitional | One explicit normalized kernel digest contract |
| Bundle | `KernelExecutionDigestBound` | def | Definitional | Exact theorem-facing realization predicate for one kernel digest instance |
| Theorem | `kernelExecutionDigest_of_conclusion` | theorem | Theorem-Target | One exact kernel conclusion bundle determines one realized kernel digest |
| Theorem | `kernelExecutionDigest_of_acceptance` | theorem | Theorem-Target | One accepted kernel boundary instance determines one realized kernel digest |
| Theorem | `authenticatedChunkTraceBound_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the authenticated chunk-trace surface |
| Theorem | `stage2TemporalSeeds_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the exact per-row Stage-2 temporal seed summary |
| Theorem | `temporalSupport_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the named authenticated temporal-support bundle |
| Theorem | `authenticatedExecutionTraceBound_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the exact authenticated execution-trace bundle used by the final soundness corollaries |
| Theorem | `traceLinkBound_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the exact adjacent-state link contract directly |
| Theorem | `executionLinked_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the raw execution-linked trace law directly |
| Theorem | `executionCorrect_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers exact chunk execution correctness |
| Theorem | `preparedStepExport_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers exact prepared-step export |
| Theorem | `rowProjectionSummary_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the exact row-projection audit summary |
| Theorem | `bridgeBindingSummary_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the exact bridge-binding audit summary |
| Theorem | `kernelClaimsFixedInRoot0_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the exact `root0` commitment-fixing law |
| Theorem | `kernelRootCommitmentsDisjoint_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the exact kernel/root commitment disjointness law |
| Theorem | `challengeAfterPhase0_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the challenge-after-phase-0 transcript law |
| Theorem | `stage1TerminalAfterPhase0_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the Stage-1 terminal-point transcript law |
| Theorem | `stage2TerminalAfterPhase0_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the Stage-2 terminal-point transcript law |
| Theorem | `rowBindingCoverage_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers exact row-binding coverage for semantic rows |
| Theorem | `emitKernelOpeningClaimsLast_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the final emit placement law |
| Theorem | `negligibleTotal_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers negligible total soundness error |
| Theorem | `kernelSoundnessConclusion_of_digest` | theorem | Theorem-Target | Realized kernel digest recovers the full kernel conclusion bundle |

## Proof Obligations

- This owner must normalize the exact current kernel theorem surface, not a
  weaker slice-level approximation.
- It must not weaken the strong kernel theorem surface by dropping exact
  `ExecutionCorrect` and retaining only a boundary-only chunk summary.
- The normalized trace surface must preserve the explicit temporal-consistency
  route to `ExecutionCorrect`, including the named authenticated temporal-
  support bundle, not compress it into a vague “linked trace” claim with no
  theorem-facing provenance.
- It must remain above `Chip8KernelSoundness` rather than re-owning any
  Stage 1 / Stage 2 / Stage 3 theorem.
