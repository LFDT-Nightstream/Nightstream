# Chip8InstructionSemanticsLookupProjection Spec

## Purpose

- **What it is**: The concrete Nightstream bridge owner for the CHIP-8
  `InstructionSemanticsLookup` extension family.
- **What it is not**: It is not a generic Shout proof and it does not restate
  the full Stage-1 decode surface.
- **Protocol role**: It packages the exact authenticated Stage-1 helper-lookup
  facts owned by the final kernel into one concrete emitted family id and
  proves how that family is routed by the Nightstream bridge.

## Target Formulas

Define the concrete family id:

$$
\mathrm{InstructionSemanticsLookupFamily}
:=
\mathrm{InstructionSemanticsLookup}.
$$

Define the exact Stage-1 helper record:

$$
\mathrm{InstructionSemanticsLookupRecord}
:=
(\mathrm{lookupOut}, \mathrm{burstLast}).
$$

Define the corresponding authenticated helper-record surface:

$$
\mathrm{InstructionSemanticsLookupRecordBound}
(dec, regX, regY, xIdx, rec)
$$

to mean:

- `AluLookupBound(dec, regX, regY, rec.lookupOut)`, and
- `BurstEqBound(dec, xIdx, rec.burstLast)`.

For every authenticated decode row and fixed lane-side helper inputs, the exact
helper record must exist and be unique:

$$
\mathrm{FetchDecodeBound}(rom, pc, dec)
\Longrightarrow
\forall regX, regY, xIdx,\;
\exists! rec,\;
\mathrm{InstructionSemanticsLookupRecordBound}(dec, regX, regY, xIdx, rec).
$$

The authenticated helper record must preserve the kernel's Stage-1 defaults:

$$
dec.lookupKind = \mathrm{NoLookup}
\land
\mathrm{InstructionSemanticsLookupRecordBound}(dec, regX, regY, xIdx, rec)
\Longrightarrow
rec.lookupOut = 0.
$$

$$
dec.isMemOp = 0
\land
\mathrm{InstructionSemanticsLookupRecordBound}(dec, regX, regY, xIdx, rec)
\Longrightarrow
rec.burstLast = 0.
$$

Define the emitted obligation family:

$$
\mathrm{InstructionSemanticsLookupProjection}(p)
:=
\mathrm{ShoutReadProjection}(\mathrm{InstructionSemanticsLookupFamily}, p).
$$

This family is a concrete `ShoutReadEval` family, so it never enters the
Nightstream main lane directly:

$$
\neg
\mathrm{MainLaneAdmissible}
(\Pi.f_{\mathrm{main}}, \Pi.p_{\mathrm{main}}, \mathrm{InstructionSemanticsLookupProjection}(p)).
$$

Its classification therefore follows the generic separate-fold support rule:

$$
\Pi.S(\mathrm{InstructionSemanticsLookupFamily}, \mathrm{ShoutReadEval}, p)
\Longrightarrow
\mathrm{decideFamily}(\Pi, \mathrm{InstructionSemanticsLookupProjection}(p)) = \mathrm{foldSeparate}.
$$

$$
\neg \Pi.S(\mathrm{InstructionSemanticsLookupFamily}, \mathrm{ShoutReadEval}, p)
\Longrightarrow
\mathrm{decideFamily}(\Pi, \mathrm{InstructionSemanticsLookupProjection}(p)) = \mathrm{exportFinal}.
$$

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Family id | `instructionSemanticsLookupFamily` | def | Definitional | Fixes the concrete family id to `ExtensionFamily.instructionSemanticsLookup` |
| Helper record | `InstructionSemanticsLookupRecord` | def | Definitional | Packages the Stage-1 helper lookup output and burst-equality output |
| Helper binding | `InstructionSemanticsLookupRecordBound` | def | Definitional | Re-exports the exact authenticated ALU/Eq4 Stage-1 helper facts as one concrete record surface |
| Projection | `instructionSemanticsLookupProjection` | def | Definitional | Concrete `ShoutReadProjection` for the instruction-semantics family |
| Theorem | `instructionSemanticsLookupRecord_existsUnique_of_fetchDecodeBound` | theorem | Theorem-Target | Authenticated Stage-1 decode plus fixed helper inputs determine one exact helper record |
| Theorem | `instructionSemanticsLookupRecord_lookup_zero_of_noLookup` | theorem | Theorem-Target | The authenticated helper record preserves the kernel's `NoLookup -> 0` rule |
| Theorem | `instructionSemanticsLookupRecord_burst_zero_of_nonMem` | theorem | Theorem-Target | The authenticated helper record preserves the kernel's non-memory burst default |
| Theorem | `instructionSemanticsLookupProjection_not_mainLane` | theorem | Theorem-Target | Concrete instruction-semantics family stays out of the main lane |
| Theorem | `instructionSemanticsLookupProjection_decide_eq_foldSeparate_of_supported` | theorem | Theorem-Target | Explicit family support routes instruction semantics to separate folding |
| Theorem | `instructionSemanticsLookupProjection_decide_eq_exportFinal_of_unsupported` | theorem | Theorem-Target | Without explicit support the family remains final |

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/ExtensionFamily.lean`
  - `Nightstream/Chip8/Stage1/FetchDecodeBinding.lean`
  - `Nightstream/ShardComposition.lean`
- **Downstream consumers**:
  - later release-path CHIP-8 family routing owners
  - later Rust refinement for the staged readonly helper family

## Proof Obligations

- The concrete family id must stay aligned with the Rust `ExtensionFamily`
  boundary.
- The instruction-semantics family must be routed as a `ShoutReadEval` family,
  not as a main-lane `CE` family.
- This owner may assume only the exact Stage-1 helper theorem surfaces already
  owned by `Chip8FetchDecodeBinding`.

## Paper Anchors

- **Source**:
  - `./docs/assurance-strategy.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./crates/neo-fold-prototype/src/proof.rs`
  - `./crates/neo-fold-prototype/src/stages/planner.rs`
  - `./formal/nightstream-lean/specs/chip8/Chip8FetchDecodeBinding.spec.md`

## Out of Scope

- generic Shout soundness proofs
- bytecode-fetch routing
- register/RAM history routing
