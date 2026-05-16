# Chip8BytecodeFetchProjection Spec

## Purpose

- **What it is**: The concrete Nightstream bridge owner for the CHIP-8
  `BytecodeFetch` extension family.
- **What it is not**: It is not a generic Shout proof and it does not restate
  the full Stage-1 decode boundary.
- **Protocol role**: It ties the exact Stage-1 fetch boundary to one concrete
  emitted family id and proves how that family is routed by the Nightstream
  bridge.

## Target Formulas

Define the concrete family id:

$$
\mathrm{BytecodeFetchFamily} := \mathrm{BytecodeFetch}.
$$

Define the concrete fetch record surface:

$$
\mathrm{BytecodeFetchRecordBound}(rom, pc, opcode)
\iff
\mathrm{opcodeAt}(rom, pc) = \mathrm{some}(opcode).
$$

The Stage-1 fetch boundary must imply existence of one concrete fetch record:

$$
\mathrm{FetchDecodeBound}(rom, pc, dec)
\Longrightarrow
\exists opcode,\;
\mathrm{BytecodeFetchRecordBound}(rom, pc, opcode).
$$

Define the emitted obligation family:

$$
\mathrm{BytecodeFetchProjection}(p)
:=
\mathrm{ShoutReadProjection}(\mathrm{BytecodeFetchFamily}, p).
$$

This family is a concrete `ShoutReadEval` family, so it never enters the
Nightstream main lane directly:

$$
\neg
\mathrm{MainLaneAdmissible}
(\Pi.f_{\mathrm{main}}, \Pi.p_{\mathrm{main}}, \mathrm{BytecodeFetchProjection}(p)).
$$

Its classification therefore follows the generic separate-fold support rule:

$$
\Pi.S(\mathrm{BytecodeFetchFamily}, \mathrm{ShoutReadEval}, p)
\Longrightarrow
\mathrm{decideFamily}(\Pi, \mathrm{BytecodeFetchProjection}(p)) = \mathrm{foldSeparate}.
$$

$$
\neg \Pi.S(\mathrm{BytecodeFetchFamily}, \mathrm{ShoutReadEval}, p)
\Longrightarrow
\mathrm{decideFamily}(\Pi, \mathrm{BytecodeFetchProjection}(p)) = \mathrm{exportFinal}.
$$

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Family id | `bytecodeFetchFamily` | def | Definitional | Fixes the concrete family id to `ExtensionFamily.bytecodeFetch` |
| Fetch record | `BytecodeFetchRecordBound` | def | Definitional | `opcodeAt` determines the concrete fetched opcode |
| Projection | `bytecodeFetchProjection` | def | Definitional | Concrete `ShoutReadProjection` for the bytecode-fetch family |
| Theorem | `bytecodeFetchRecordBound_of_fetchDecodeBound` | theorem | Theorem-Target | Exact Stage-1 fetch boundary determines one concrete bytecode-fetch record |
| Theorem | `bytecodeFetchProjection_not_mainLane` | theorem | Theorem-Target | Concrete bytecode-fetch family stays out of the main lane |
| Theorem | `bytecodeFetchProjection_decide_eq_foldSeparate_of_supported` | theorem | Theorem-Target | Explicit family support routes bytecode fetch to separate folding |
| Theorem | `bytecodeFetchProjection_decide_eq_exportFinal_of_unsupported` | theorem | Theorem-Target | Without explicit support the family remains final |

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/ExtensionFamily.lean`
  - `Nightstream/Chip8/Stage1/FetchDecodeBinding.lean`
  - `Nightstream/ShardComposition.lean`
- **Downstream consumers**:
  - later release-path CHIP-8 family routing owners
  - later Rust refinement for `families/bytecode_fetch.rs`

## Proof Obligations

- The concrete family id must stay aligned with the Rust `ExtensionFamily`
  boundary.
- The bytecode-fetch family must be routed as a `ShoutReadEval` family, not as
  a main-lane `CE` family.
- The only Stage-1 fact this owner may assume is the exact `FetchDecodeBound`
  theorem surface.

## Paper Anchors

- **Source**:
  - `./docs/assurance-strategy.md`
  - `./crates/neo-fold-prototype/src/families/bytecode_fetch.rs`
  - `./formal/nightstream-lean/specs/chip8/Chip8FetchDecodeBinding.spec.md`

## Out of Scope

- generic Shout soundness proofs
- instruction-semantics lookup routing
- register/RAM history routing
