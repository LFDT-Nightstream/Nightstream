# Chip8ReleaseBridge Spec

## Purpose

- **What it is**: The concrete CHIP-8 release-path bridge owner above the four
  individual extension-family modules.
- **What it is not**: It is not a generic Nightstream routing theorem and it
  does not restate the exact kernel soundness boundary.
- **Protocol role**: It packages the current CHIP-8 release families into the
  stage split that the Rust planner uses, and it packages the concrete Stage-1
  readonly-batch evidence and Stage-2 history evidence that a later staged
  bridge artifact will need.

## Target Formulas

Define the release stages:

$$
\mathrm{ReleaseStage}
:=
\{
\mathrm{ReadonlyBatch},
\mathrm{RegisterHistory},
\mathrm{RamHistory}
\}.
$$

Define the family-to-stage map:

$$
\mathrm{familyStage}(\mathrm{BytecodeFetch}) = \mathrm{ReadonlyBatch},
$$

$$
\mathrm{familyStage}(\mathrm{InstructionSemanticsLookup}) = \mathrm{ReadonlyBatch},
$$

$$
\mathrm{familyStage}(\mathrm{RegisterHistory}) = \mathrm{RegisterHistory},
$$

$$
\mathrm{familyStage}(\mathrm{RamHistory}) = \mathrm{RamHistory}.
$$

Define the canonical family inventory per stage:

$$
\mathrm{stageFamilies}(\mathrm{ReadonlyBatch})
=
[\mathrm{BytecodeFetch}, \mathrm{InstructionSemanticsLookup}],
$$

$$
\mathrm{stageFamilies}(\mathrm{RegisterHistory})
=
[\mathrm{RegisterHistory}],
$$

$$
\mathrm{stageFamilies}(\mathrm{RamHistory})
=
[\mathrm{RamHistory}].
$$

This stage inventory must agree exactly with the family map:

$$
f \in \mathrm{stageFamilies}(s)
\iff
\mathrm{familyStage}(f) = s.
$$

For one Stage-1 fetch/decode surface, define the concrete readonly-batch
bundle:

$$
\mathrm{ReadonlyBatchBundle}(rom, pc, dec, regX, regY, xIdx),
$$

whose fields are:

- one concrete fetched opcode satisfying
  `BytecodeFetchRecordBound(rom, pc, fetchOpcode)`,
- one concrete instruction-semantics record satisfying
  `InstructionSemanticsLookupRecordBound(dec, regX, regY, xIdx, lookupRecord)`.

The theorem-facing constructor target is:

$$
\mathrm{FetchDecodeBound}(rom, pc, dec)
\Longrightarrow
\mathrm{ReadonlyBatchBundleBound}(rom, pc, dec, regX, regY, xIdx).
$$

The bundle must preserve the concrete helper consequences already owned by the
individual Stage-1 family modules:

$$
\mathrm{opcodeAt}(rom, pc) = \mathrm{some}(\mathrm{fetchOpcode}),
$$

$$
dec.\mathrm{lookupKind} = \mathrm{NoLookup}
\Longrightarrow
\mathrm{lookupRecord.lookupOut} = 0,
$$

$$
dec.\mathrm{isMemOp} = 0
\Longrightarrow
\mathrm{lookupRecord.burstLast} = 0.
$$

For one exact authenticated frame list `frames`, define the concrete history
bundle:

$$
\mathrm{HistoryBundle}(frames),
$$

whose fields are:

- the exact `RegisterHistoryBundle(frames)`,
- the exact `RamHistoryBundle(frames)`.

The theorem-facing constructor target is:

$$
\mathrm{ExactTraceEvidence}(frames)
\Longrightarrow
\mathrm{HistoryBundleBound}(frames).
$$

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Stage ids | `ReleaseStage` | def | Definitional | Enumerates the exact release-path stages currently owned by the CHIP-8 bridge |
| Stage map | `familyStage` | def | Definitional | Fixes the exact `ExtensionFamily -> ReleaseStage` map aligned to the Rust planner |
| Stage inventory | `stageFamilies` | def | Definitional | Fixes the exact family inventory per release stage |
| Theorem | `mem_stageFamilies_iff` | theorem | Theorem-Target | Canonical stage inventories agree exactly with the family-to-stage map |
| Bundle | `ReadonlyBatchBundle` | structure | Definitional | Packages the concrete Stage-1 `BytecodeFetch` and `InstructionSemanticsLookup` evidence needed by the readonly batch |
| Bundle | `ReadonlyBatchBundleBound` | def | Definitional | Named proposition that the exact Stage-1 boundary admits that readonly-batch bundle |
| Constructor | `readonlyBatchBundle_of_fetchDecodeBound` | def | Theorem-Target | Exact Stage-1 fetch/decode evidence yields the concrete readonly-batch bundle |
| Theorem | `readonlyBatchBundleBound_of_fetchDecodeBound` | theorem | Theorem-Target | Proposition-level version of the same construction |
| Theorem | `readonlyBatchBundle_opcodeAt` | theorem | Theorem-Target | The bundle preserves the exact fetched opcode relation |
| Theorem | `readonlyBatchBundle_lookup_zero_of_noLookup` | theorem | Theorem-Target | The bundle preserves the exact zero-lookup consequence |
| Theorem | `readonlyBatchBundle_burst_zero_of_nonMem` | theorem | Theorem-Target | The bundle preserves the exact non-memory burst consequence |
| Bundle | `HistoryBundle` | structure | Definitional | Packages the exact `RegisterHistoryBundle` and `RamHistoryBundle` into one Stage-2 history owner |
| Bundle | `HistoryBundleBound` | def | Definitional | Named proposition that the exact authenticated trace admits that history bundle |
| Constructor | `historyBundle_of_exactTrace` | def | Theorem-Target | Exact authenticated trace evidence yields the concrete Stage-2 history bundle |
| Theorem | `historyBundleBound_of_exactTrace` | theorem | Theorem-Target | Proposition-level version of the same construction |

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/ExtensionFamily.lean`
  - `Nightstream/Chip8/Stage1/BytecodeFetchProjection.lean`
  - `Nightstream/Chip8/Stage1/InstructionSemanticsLookupProjection.lean`
  - `Nightstream/Chip8/Stage2/RegisterHistoryProjection.lean`
  - `Nightstream/Chip8/Stage2/RamHistoryProjection.lean`
- **Downstream consumers**:
  - the later staged bridge artifact replacing the compatibility export path
  - later Rust refinement for `stages/planner.rs` and `bridge/mod.rs`

## Proof Obligations

- The release-stage map must stay aligned with the Rust planner’s family
  staging.
- This owner must not weaken any of the individual family theorem surfaces; it
  may only package them.
- The readonly-batch bundle may assume only the exact `FetchDecodeBound`
  surface.
- The history bundle may assume only the exact `ExactTraceEvidence` surface.

## Paper Anchors

- **Source**:
  - `./docs/assurance-strategy.md`
  - `./crates/neo-fold-prototype/src/proof.rs`
  - `./crates/neo-fold-prototype/src/stages/planner.rs`
  - `./formal/nightstream-lean/specs/chip8/Chip8BytecodeFetchProjection.spec.md`
  - `./formal/nightstream-lean/specs/chip8/Chip8InstructionSemanticsLookupProjection.spec.md`
  - `./formal/nightstream-lean/specs/chip8/Chip8RegisterHistoryProjection.spec.md`
  - `./formal/nightstream-lean/specs/chip8/Chip8RamHistoryProjection.spec.md`

## Out of Scope

- generic Nightstream routing proofs
- the final staged bridge artifact
- compressed opening / final release proof packaging
