# Rv64IMRegisterHistoryProjection Spec

## Purpose

- **What it is**: The theorem-facing Stage-2 owner for RV64IM register-history projection.
- **What it is not**: It is not the full Twist algorithm and it does not own Stage-1 decode.
- **Protocol role**: It fixes the concrete register-history family, the row-local register read/write surface, and the sink-routing rules for inactive ports.

## Target Formulas

Define the concrete family id:

$$
\mathrm{RegisterHistoryFamily} := \mathrm{registerHistory}.
$$

Define one row-local register-history record:

$$
\mathrm{RegisterHistoryRow}
:=
(\mathrm{rs1Addr}, \mathrm{rs2Addr}, \mathrm{waAddr},
\mathrm{rs1Value}, \mathrm{rs2Value}, \mathrm{writeValue},
\mathrm{usesRs2}, \mathrm{writesRd}).
$$

Define the row-local sink-routing rule:

$$
\mathrm{RegisterHistoryRowBound}(\bot_{reg}, row)
$$

meaning:

- if `usesRs2 = 0`, then `rs2Addr = ⊥_reg`,
- if `writesRd = 0`, then `waAddr = ⊥_reg`.

Define the theorem-facing bundle:

$$
\mathrm{RegisterHistoryBundle}
$$

carrying the register timeline, the fixed register sink, the per-row projected
register-history records, and a `List.Forall` proof that every row satisfies the
row-local sink-routing rule.

Define:

$$
\mathrm{RegisterHistoryProjection}(p)
:=
\mathrm{TwistValProjection}(\mathrm{RegisterHistoryFamily}, p).
$$

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - register-file domain and ports
  - register-file lane linkage
  - address-correctness obligations for Stage 2

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Stage2/RegisterHistoryProjection.lean` | Register-history projection owner |
| `Nightstream/Rv64IM/Stage2/RegisterHistoryProjectionInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Family id | `registerHistoryFamily` | def | Definitional | Fixes the concrete Stage-2 register-history family |
| Row | `RegisterHistoryRow` | structure | Definitional | Packages one row-local register read/write surface |
| Boundary | `RegisterHistoryRowBound` | def | Definitional | Inactive rs2 and write ports route to the register sink |
| Bundle | `RegisterHistoryBundle` | structure | Definitional | Packages the register timeline and row-local Stage-2 register facts |
| Theorem | `registerHistoryRowBound_rs2Sink_of_not_usesRs2` | theorem | Theorem-Target | Inactive rs2 reads route to the register sink |
| Theorem | `registerHistoryRowBound_waSink_of_not_writesRd` | theorem | Theorem-Target | Inactive writes route to the register sink |
| Projection | `registerHistoryProjection` | def | Definitional | Concrete Twist projection for the register-history family |

## Out of Scope

- RAM history
- Stage-1 decode correctness
- Stage-3 continuity
