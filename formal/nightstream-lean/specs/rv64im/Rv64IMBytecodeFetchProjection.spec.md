# Rv64IMBytecodeFetchProjection Spec

## Purpose

- **What it is**: The concrete Nightstream bridge owner for the RV64IM
  expanded-bytecode fetch family.
- **What it is not**: It is not the full decoded-row theorem and it does not
  restate expanded-bytecode successor semantics.
- **Protocol role**: It ties the Stage-1 fetch boundary to one concrete family
  id and proves how that family is classified by the Nightstream bridge.

## Target Formulas

Define the concrete family id:

$$
\mathrm{BytecodeFetchFamily} := \mathrm{fetch}.
$$

Define the concrete fetch record surface:

$$
\mathrm{BytecodeFetchRecordBound}(\mathrm{bytecodeTable}, \mathrm{expandedPc}, row)
\iff
\mathrm{bytecodeTable}(\mathrm{expandedPc}) = \mathrm{some}(row).
$$

The Stage-1 fetch/decode boundary must imply one concrete fetch record:

$$
\mathrm{FetchDecodeBound}
(
\mathrm{bytecodeTable},
\mathrm{expandedPc},
x0,
\mathrm{isArchitectural},
row
)
\Longrightarrow
\mathrm{BytecodeFetchRecordBound}
(\mathrm{bytecodeTable}, \mathrm{expandedPc}, row).
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

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - fetch channel
  - expanded bytecode as canonical execution object
  - bytecode read-only lookup against `C_bytecode_table`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Stage1/BytecodeFetchProjection.lean` | Concrete Stage-1 fetch-family projection owner |
| `Nightstream/Rv64IM/Stage1/BytecodeFetchProjectionInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Family id | `bytecodeFetchFamily` | def | Definitional | Fixes the concrete family id to `ExtensionFamily.fetch` |
| Fetch record | `BytecodeFetchRecordBound` | def | Definitional | The committed expanded-bytecode table determines the fetched row exactly |
| Projection | `bytecodeFetchProjection` | def | Definitional | Concrete `ShoutReadProjection` for the RV64IM bytecode-fetch family |
| Theorem | `bytecodeFetchRecordBound_of_fetchDecodeBound` | theorem | Theorem-Target | Accepted Stage-1 fetch/decode rows come from one concrete expanded-bytecode record |
| Theorem | `bytecodeFetchProjection_is_projectionFamily` | theorem | Theorem-Target | The emitted fetch obligations are a homogeneous `ShoutReadEval` family |
| Theorem | `bytecodeFetchProjection_not_mainLane` | theorem | Theorem-Target | RV64IM bytecode fetch never enters the main lane directly |
| Theorem | `bytecodeFetchProjection_decide_eq_foldSeparate_of_supported` | theorem | Theorem-Target | Supported RV64IM bytecode fetch folds separately |
| Theorem | `bytecodeFetchProjection_decide_eq_exportFinal_of_unsupported` | theorem | Theorem-Target | Unsupported RV64IM bytecode fetch remains final/exported |

## Proof Obligations

- The concrete family id must stay aligned with the RV64IM extension-family
  inventory.
- The bytecode-fetch family is a read-only lookup owner over the expanded
  bytecode, not a main-lane CE owner.
- The only Stage-1 semantic input assumed here is the exact fetch/decode
  boundary exported by `Rv64IMFetchDecodeBinding`.

## Out of Scope

- expanded-bytecode successor / entrypoint semantics
- decode-handoff exactness
- ALU / branch slot authentication
- register / RAM histories
