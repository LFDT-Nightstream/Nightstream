# Chip8StagedExecutionDigestBundle Spec

## Purpose

- **What it is**: The theorem-facing chunk-level bundle owner for normalized
  staged CHIP-8 execution digests.
- **Key property**: `stagedExecutionDigestBundle_of_frames` packages one
  theorem-facing public surface together with one normalized staged execution
  digest for each exact authenticated semantic frame in chunk order.
- **Protocol role**: This owner lifts the per-slice
  `Chip8StagedExecutionDigest` surface to the chunk level so Rust and Lean can
  compare one ordered chunk artifact instead of ad hoc single-step digests. The
  intended comparison target is exact value equality on the Lean-defined chunk
  artifact, not merely shape compatibility.

## Target Formulas

### Frame-local digest entries

For one exact authenticated semantic frame `frame`, define a theorem-facing
entry:

$$
\mathrm{FrameDigestEntry}(frame)
$$

to package:

- the exact frame itself,
- one normalized `StagedExecutionDigest` for that frame, and
- the proof that the digest realizes the exact theorem-facing Stage-1 /
  Stage-2 / Stage-3 / execution-result surfaces of that frame.

This owner does not define new row semantics. It only records the normalized
digest corresponding to one already-authenticated frame.

### Chunk-level bundle

For one ordered exact frame list `frames`, define:

$$
\mathrm{StagedExecutionDigestBundle}(frames)
$$

to package:

- one `DigestPublicSurface`,
- one ordered list of `FrameDigestEntry`,
- one exact ordering law stating that projecting the entry list back to frames
  recovers exactly `frames`.

The bundle is therefore not just a bag of digests. It is an order-sensitive
normal form over one exact chunk frame list.

If any bundled field is protocol-binding and digest-derived, its meaning must be
the exact canonical Goldilocks / serialization / Poseidon2 computation attached
to that field, not an opaque Rust-chosen byte string.

### Public-surface law

From any bundle:

$$
\mathrm{kernelPublicInputsBound\_of\_bundle}
$$

must recover the exact theorem-facing kernel public-input contract already owned
by `Chip8StagedExecutionDigest`.

### Length laws

From any bundle:

$$
\mathrm{bundleLength\_eq}
$$

must recover that the digest-entry count equals the exact frame count.

Given the simple-kernel chunk input contract:

$$
\mathrm{bundleLength\_eq\_semanticRows}
$$

must recover that the bundle length equals the public semantic-row count of the
chunk.

### Entry projection laws

From any frame entry:

$$
\mathrm{executionFrameBound\_of\_entry}
$$

must recover the exact `ExecutionFrameBound` surface already owned by the
per-slice digest owner.

The bundle owner therefore remains a normalization layer. It does not replace
or weaken the theorem-facing row-local surfaces already proved elsewhere.

### Normalization theorem

For one exact authenticated frame list whose semantic trace satisfies the
simple-kernel chunk input contract, the owner must expose:

$$
\mathrm{stagedExecutionDigestBundle\_of\_frames}
$$

producing one canonical chunk-level digest bundle.

This owner is intentionally below the future chunk-level audit checker. It owns
the normalized chunk artifact, not the later acceptance predicate over an
externally imported Rust artifact.

## Paper Anchors

- **Sources**:
  - `./docs/assurance-strategy.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./formal/nightstream-lean/specs/chip8/Chip8StagedExecutionDigest.spec.md`
- Anchors:
- Layer-2 digest comparison needs one exact chunk artifact
- Layer-3 checker needs one exact chunk-level theorem-facing import surface
- normalized digest bundles must remain derived from exact authenticated
  theorem surfaces rather than Rust-local export convenience
- chunk-level compatibility should support deterministic golden vectors and
  exact Rust↔Lean equality over the final release artifact

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Kernel/StagedExecutionDigestBundle.lean` | Chunk-level bundle of normalized staged execution digests |
| `Nightstream/Chip8/Kernel/StagedExecutionDigestBundleInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Structure | `FrameDigestEntry` | structure | Definitional | Packages one exact authenticated frame, one staged digest, and its exact realization proof |
| Structure | `StagedExecutionDigestBundle` | structure | Definitional | Packages one public digest surface plus one ordered digest entry per exact frame |
| Function | `frameDigestEntry_of_exactFrame` | def | Constructor | Builds one normalized digest entry from one exact authenticated frame |
| Function | `frameDigestEntries_of_frames` | def | Constructor | Builds the canonical digest-entry list for one exact frame list |
| Function | `stagedExecutionDigestBundle_of_frames` | def | Constructor | Builds the chunk-level bundle from one exact frame list plus the simple-kernel chunk-input contract |
| Theorem | `kernelPublicInputsBound_of_bundle` | theorem | Theorem-Target | Bundle public surface recovers the exact kernel public-input contract |
| Theorem | `bundleLength_eq` | theorem | Theorem-Target | Bundle entry count equals exact frame count |
| Theorem | `bundleLength_eq_semanticRows` | theorem | Theorem-Target | Bundle entry count equals the public semantic-row count |
| Theorem | `executionFrameBound_of_entry` | theorem | Theorem-Target | Any digest entry recovers the exact execution-frame theorem surface |

## Proof Obligations

- The bundle owner must stay a normalization layer over exact authenticated
  frame evidence.
- Do not weaken the per-slice staged digest surface to make bundling easier.
- Do not silently identify bundle order with list order without one explicit
  ordering law.
- Do not move later chunk-level audit acceptance into this owner; that belongs
  to a later bundle-audit checker layer.
- Do not leave bundled protocol-binding digest fields semantically opaque; they
  must admit one exact Lean-owned computational meaning if they are exported for
  Rust↔Lean comparison.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Kernel/StagedExecutionDigest.lean`
  - `Nightstream/Chip8/Trace/AuthenticatedTrace.lean`
  - `Nightstream/Chip8/Trace/ChunkInput.lean`
- **Downstream consumers**:
  - future chunk-level staged-digest audit checker
  - Rust↔Lean digest comparison harness
  - release gating over the final chunk artifact

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Kernel.StagedExecutionDigestBundle` succeeds.
2. The chunk-level bundle shape is Lean-owned.
3. The bundle recovers exact per-entry theorem surfaces, not weaker summaries.
4. No `sorry`.
