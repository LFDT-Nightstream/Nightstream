# Chip8StagedExecutionDigest Spec

## Purpose

- **What it is**: The theorem-facing normalized digest contract for one staged
  CHIP-8 kernel execution.
- **Key property**: `stagedExecutionDigest_of_exactEvidence`: exact
  authenticated public inputs, Stage-1/Stage-2/Stage-3 semantic evidence, and
  the resulting execution theorem determine one normalized staged execution
  digest.
- **Protocol role**: This is the single explicit comparison and audit boundary
  shared by Rust and Lean. Its shape is dictated by the theorem surfaces that
  the composition theorem actually consumes, not by implementation-local export
  convenience.

## Target Formulas

### Digest ownership

The digest owner packages the exact theorem-facing surfaces of the final kernel
into one explicit record.

The intended components are:

- one public-input surface
- one Stage-1 surface
- one Stage-2 surface
- one Stage-3 continuity / bridge surface
- one final semantic-result surface

This owner does not define new semantic facts. It normalizes the existing ones
into one comparison and audit contract.

### Public surface

Define:

$$
\mathrm{DigestPublicSurface}(d_{\mathrm{pub}}, publicInput, meta, romTable, init)
$$

to mean that the public portion of the digest determines exactly the bundled
public-input theorem surface:

$$
\mathrm{KernelPublicInputsBound}(publicInput, meta, romTable, init).
$$

### Stage-1 surface

Define:

$$
\mathrm{Stage1DigestSurface}(d_1, romTable, pre, dec, z)
$$

to mean that the Stage-1 digest component determines exactly the authenticated
Stage-1 theorem surfaces needed downstream:

$$
\mathrm{FetchDecodeBound}(romTable, pre.PC, dec)
$$

and

$$
\mathrm{LookupBound}(dec, pre, z).
$$

### Stage-2 surface

Define:

$$
\mathrm{Stage2DigestSurface}(d_2, pre, post, init, dec, z)
$$

to mean that the Stage-2 digest component determines exactly the row-binding
and memory-binding theorem surfaces:

$$
\mathrm{WitnessBinds}(pre, post, dec, z)
$$

and

$$
\mathrm{MemoryBound}(pre, post, init, dec, z).
$$

### Stage-3 surface

Define:

$$
\mathrm{Stage3DigestSurface}(d_3, z, prepared)
$$

to mean that the Stage-3 digest component determines exactly the continuity and
bridge theorem surfaces:

$$
\mathrm{ContinuityRowBound}(d_3, z)
$$

and, when the current row is exported,

$$
\mathrm{PreparedStepBound}(d_3, z, prepared).
$$

The Stage-3 digest component must also carry the explicit row-level bridge
audit object:

$$
\mathrm{BridgeBindingWitness}(stepIdx, z, rowClaim, prepared).
$$

### Final semantic-result surface

Define:

$$
\mathrm{ExecutionResultSurface}(d_r, pre, post, dec, z)
$$

to mean that the digest result component determines the exact semantic theorem
surface for the current supported-kernel row or chunk-local execution object.

At minimum, this owner must support packaging the execution theorem surfaces
already owned by `Chip8ExecutionSemantics`, `Chip8BurstSession`, and
`Chip8StepComposition`. For one exact execution slice, that includes the
row-backed `ExecutionFrameBound` surface consumed by chunk-level
`ExecutionCorrect`.

### Bundled digest

Define the normalized digest object:

$$
\mathrm{StagedExecutionDigest}
(d_{\mathrm{pub}}, d_1, d_2, d_3, d_r).
$$

Define its exact realization predicate:

$$
\mathrm{StagedExecutionDigestBound}
(d, publicInput, meta, romTable, init, pre, post, dec, z, prepared)
$$

to mean the conjunction of:

- `DigestPublicSurface`
- `Stage1DigestSurface`
- `Stage2DigestSurface`
- `Stage3DigestSurface`
- `ExecutionResultSurface`

for one exact supported-kernel execution slice.

### Normalization theorem

The main normalization theorem is:

$$
\mathrm{ExactSemanticEvidenceCovered}(\dots)
\land
\mathrm{MicrostepCorrect}(\dots)
\Longrightarrow
\exists d,\ \mathrm{StagedExecutionDigestBound}(d,\dots).
$$

This is the theorem that makes the digest Lean-defined rather than Rust-defined.
Because this owner is parameterized by one exact execution slice
`(stepIdx, pre, post, dec, z)`, it must not overclaim a whole authenticated
execution trace without additional trace/start-boundary data.

### Projection theorems

From one realized digest, downstream users must be able to recover the exact
existing theorem surfaces:

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{KernelPublicInputsBound}(publicInput, meta, romTable, init)
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{FetchDecodeBound}(romTable, pre.PC, dec)
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{LookupBound}(dec, pre, z)
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{WitnessBinds}(pre, post, dec, z)
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{MemoryBound}(pre, post, init, dec, z)
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{ContinuityRowBound}(d_3, z).
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{BridgeBindingWitness}(stepIdx, z, rowClaim, prepared).
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{ExecutionFrameBound}(rom,\sigma,\langle dec, pre, post, z \rangle).
$$

This ensures the digest is a faithful normal form of the existing theorem
surfaces, not a parallel informal interface.

## Paper Anchors

- **Sources**:
  - `./docs/assurance-strategy.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/superneo-paper`
- Anchors:
  - staged proof composition
  - authenticated public-input binding
  - Stage-1 / Stage-2 / Stage-3 theorem ownership
  - bridge export into root prepared steps
  - one explicit digest/evidence contract for comparison and audit

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/StagedExecutionDigest.lean` | Normalized digest contract and normalization/projection theorems for the final CHIP-8 kernel |
| `Nightstream/Chip8/StagedExecutionDigestInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Digest | `DigestPublicSurface` | def | Definitional | The public digest component carries the exact bundled public-input theorem surface |
| Digest | `Stage1DigestSurface` | def | Definitional | The Stage-1 digest component carries the exact Stage-1 theorem surfaces |
| Digest | `Stage2DigestSurface` | def | Definitional | The Stage-2 digest component carries the exact Stage-2 theorem surfaces |
| Digest | `Stage3DigestSurface` | def | Definitional | The Stage-3 digest component carries the exact continuity / bridge theorem surfaces |
| Digest | `ExecutionResultSurface` | def | Definitional | The result digest component carries the exact supported-kernel semantic result surface |
| Bundle | `StagedExecutionDigest` | def | Definitional | One explicit normalized digest contract shared by Rust and Lean |
| Bundle | `StagedExecutionDigestBound` | def | Definitional | Exact theorem-facing realization predicate for one digest instance |
| Theorem | `stagedExecutionDigest_of_exactEvidence` | theorem | Theorem-Target | Exact authenticated evidence and row-level semantic correctness determine one realized digest |
| Theorem | `kernelPublicInputsBound_of_digest` | theorem | Theorem-Target | Realized digest recovers the exact public-input theorem surface |
| Theorem | `fetchDecodeBound_of_digest` | theorem | Theorem-Target | Realized digest recovers the exact Stage-1 theorem surface |
| Theorem | `memoryBound_of_digest` | theorem | Theorem-Target | Realized digest recovers the exact Stage-2 theorem surface |
| Theorem | `continuityRowBound_of_digest` | theorem | Theorem-Target | Realized digest recovers the exact Stage-3 theorem surface |
| Theorem | `executionFrameBound_of_digest` | theorem | Theorem-Target | Realized digest recovers the exact row-backed execution-frame surface |
| Theorem | `executionResultSurface_of_digest` | theorem | Theorem-Target | Realized digest recovers the exact supported-kernel result surface |
| Theorem | `microstepCorrect_of_digest` | theorem | Theorem-Target | Realized digest recovers the exact row-level semantic theorem |

## Proof Obligations

- The digest shape must be dictated by theorem ownership, not by Rust export
  convenience.
- The digest must be a normalized boundary over exact public, Stage-1, Stage-2,
  Stage-3, and execution-result surfaces.
- Realized digests must project back to the exact existing theorem surfaces.
- A slice-scoped digest owner must not silently claim whole-trace
  `ExecutionCorrect` without separately owning the authenticated trace boundary.
- The digest owner must not silently weaken or collapse the distinction between
  Stage-1, Stage-2, and Stage-3 responsibilities.

## Assumption Ledger

- Generic Shout, Twist, and lower-layer PCS theorems are imported.
- The digest owner does not prove cryptographic acceptance on its own.
- Serialization of digest instances between Rust and Lean is external to this
  module.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/RomScheduleBinding.lean`
  - `Nightstream/Chip8/EvidenceCoverage.lean`
  - `Nightstream/Chip8/ExecutionSemantics.lean`
  - `Nightstream/Chip8/BurstSession.lean`
  - `Nightstream/Chip8/StepComposition.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/ArtifactAudit.lean`
  - Rust-vs-Lean differential testing over normalized digest instances
  - later Rust-refinement theorems over the final kernel proof object

## Implementation Plan

1. Define the normalized staged execution digest and its component surfaces.
2. Define `StagedExecutionDigestBound`.
3. Prove normalization from exact authenticated evidence to a realized digest.
4. Prove the projection theorems back to the exact theorem-owned surfaces.

## Quality Expectations

- Keep the digest owner normalized and ownership-specific.
- Keep the Stage-1 / Stage-2 / Stage-3 split explicit.
- Treat the digest as a theorem-owned contract, not a convenience log format.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.StagedExecutionDigest` succeeds.
2. The digest boundary is explicit and Lean-defined.
3. The normalization and projection theorems are explicit.
4. No `sorry`.

## Out of Scope

- Rust serializer implementations
- CI policy
- release policy
- cryptographic verifier acceptance
