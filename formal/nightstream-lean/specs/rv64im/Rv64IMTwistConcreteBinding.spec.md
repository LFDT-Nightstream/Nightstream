# Rv64IMTwistConcreteBinding Spec

## Purpose

- **What it is**: The theorem-facing Stage-2 owner for the concrete RV64IM Twist linkage batch.
- **What it is not**: It is not the generic Twist soundness theorem and it does not own Stage-1 decoding.
- **Protocol role**: It fixes the exact register and RAM linkage equalities between lane-visible claims and authenticated Stage-2 read/write values, and it owns the concrete authenticated non-zero-init `Val` surfaces consumed by the RV64IM kernel.

## Target Formulas

Define the paired limb carrier:

$$
\mathrm{LimbPair} := (\mathrm{lo}, \mathrm{hi}).
$$

Define the register linkage surface:

- `RegisterLaneClaims`
- `RegisterTwistClaims`

and:

$$
\mathrm{RegisterLinkageBound}(lane, twist)
$$

meaning:

- `rvRs1 = rs1`,
- `rvRs2 = rs2`,
- if `writesRd = 1`, then `wvReg = rdNext`.

Define the RAM linkage surface:

- `RamLaneClaims`
- `RamTwistClaims`

and:

$$
\mathrm{RamLinkageBound}(lane, twist)
$$

meaning:

- if `isLoad = 1`, then `memVal = rvRamWord`,
- if `isStore = 1`, then `memVal = rs2` and `wvRamWord = memVal`,
- if neither load nor store is active, then `memVal = 0`.

Define the full Stage-2 linkage package:

$$
\mathrm{Stage2LinkageBound}(registers, registerTwist, ram, ramTwist)
$$

as the conjunction of the register and RAM linkage relations.

Define the concrete shifted register `Val` surface from an authenticated initial
register table `init_reg` and increment table `inc_reg`:

$$
\mathrm{RegisterShiftedTimeTable}(init\_reg, inc\_reg)
$$

and the corresponding verifier-facing random-point target:

$$
\mathrm{RegisterShiftedValEvaluationExpression}(init\_reg, inc\_reg, r_{addr}, r_{cycle}).
$$

The module must expose the exact non-zero-init identities:

$$
\widetilde{\mathrm{RegisterShiftedTimeTable}(init\_reg, inc\_reg)}(r_{addr}, r_{cycle})
=
\mathrm{RegisterShiftedValEvaluationExpression}(init\_reg, inc\_reg, r_{addr}, r_{cycle})
$$

and at bit points:

$$
\mathrm{RegisterShiftedValEvaluationExpression}(init\_reg, inc\_reg, \mathbf{a}, \mathbf{j})
=
\mathrm{RegisterShiftedTimeTable}(init\_reg, inc\_reg, a, j).
$$

Define the concrete shifted RAM `Val` surface from an authenticated initial RAM
table `init_ram` and increment table `inc_ram`:

$$
\mathrm{RamShiftedTimeTable}(init\_ram, inc\_ram)
$$

and:

$$
\mathrm{RamShiftedValEvaluationExpression}(init\_ram, inc\_ram, r_{addr}, r_{cycle}).
$$

The module must expose the exact non-zero-init identities:

$$
\widetilde{\mathrm{RamShiftedTimeTable}(init\_ram, inc\_ram)}(r_{addr}, r_{cycle})
=
\mathrm{RamShiftedValEvaluationExpression}(init\_ram, inc\_ram, r_{addr}, r_{cycle})
$$

and at bit points:

$$
\mathrm{RamShiftedValEvaluationExpression}(init\_ram, inc\_ram, \mathbf{a}, \mathbf{j})
=
\mathrm{RamShiftedTimeTable}(init\_ram, inc\_ram, a, j).
$$

For both families, the module must also expose the concrete read-check and
write-check consequences of `ReadWriteMemoryRelation` and
`IncrementRelation`, both for the zero-init carrier and for the shifted
non-zero-init carrier.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - Stage-2 linkage batch
  - RAM store payload binding
  - raw register read/write claims
  - authenticated non-zero-initialization `Val` identity

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Stage2/TwistConcreteBinding.lean` | Concrete Stage-2 linkage owner |
| `Nightstream/Rv64IM/Stage2/TwistConcreteBindingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Value | `LimbPair` | structure | Definitional | Packages the low/high limb carrier used by Stage 2 |
| Value | `zeroLimbPair` | def | Definitional | Canonical zero paired-limb value |
| Registers | `RegisterLaneClaims` | structure | Definitional | Packages lane-visible register linkage claims |
| Registers | `RegisterTwistClaims` | structure | Definitional | Packages authenticated Stage-2 register read/write values |
| Registers | `RegisterLinkageBound` | def | Definitional | Fixes the concrete register linkage equalities |
| Registers | `RegisterShiftedTimeTable` | def | Definitional | Concrete non-zero-init register `Val` surface |
| Registers | `RegisterShiftedVirtualValue` | def | Definitional | Concrete non-zero-init register random-cycle value |
| Registers | `RegisterShiftedValEvaluationExpression` | def | Definitional | Concrete non-zero-init register verifier-facing `Val` target |
| Registers | `registerReadCheckAtBitCycle_of_relation` | theorem | Theorem-Target | Concrete register read-check consequence at a bit cycle |
| Registers | `registerWriteCheckAtBitPoint_of_incrementRelation` | theorem | Theorem-Target | Concrete register write-check consequence at a bit point |
| Registers | `registerShiftedVirtualValue_at_bitCycle` | theorem | Theorem-Target | Shifted register virtual value agrees with the shifted table entry at a bit cycle |
| Registers | `registerShiftedValEvaluationExpression_eq_timeTableMLE` | theorem | Theorem-Target | Shifted register `Val` expression equals the shifted table MLE |
| Registers | `registerTimeTableMLE_shifted_at_bitAddress` | theorem | Theorem-Target | Shifted register table MLE at a bit address agrees with the shifted virtual value |
| Registers | `registerShiftedValEvaluationExpression_at_bitPoint` | theorem | Theorem-Target | Shifted register `Val` expression equals the shifted table entry at a bit point |
| Registers | `registerShiftedReadCheckAtBitCycle_of_relation` | theorem | Theorem-Target | Shifted register read-check consequence at a bit cycle |
| Registers | `registerShiftedWriteCheckAtBitPoint_of_incrementRelation` | theorem | Theorem-Target | Shifted register write-check consequence at a bit point |
| RAM | `RamLaneClaims` | structure | Definitional | Packages lane-visible RAM linkage claims |
| RAM | `RamTwistClaims` | structure | Definitional | Packages authenticated Stage-2 RAM read/write values |
| RAM | `RamLinkageBound` | def | Definitional | Fixes the concrete load/store/no-op RAM linkage equalities |
| RAM | `RamShiftedTimeTable` | def | Definitional | Concrete non-zero-init RAM `Val` surface |
| RAM | `RamShiftedVirtualValue` | def | Definitional | Concrete non-zero-init RAM random-cycle value |
| RAM | `RamShiftedValEvaluationExpression` | def | Definitional | Concrete non-zero-init RAM verifier-facing `Val` target |
| RAM | `ramReadCheckAtBitCycle_of_relation` | theorem | Theorem-Target | Concrete RAM read-check consequence at a bit cycle |
| RAM | `ramWriteCheckAtBitPoint_of_incrementRelation` | theorem | Theorem-Target | Concrete RAM write-check consequence at a bit point |
| RAM | `ramShiftedVirtualValue_at_bitCycle` | theorem | Theorem-Target | Shifted RAM virtual value agrees with the shifted table entry at a bit cycle |
| RAM | `ramShiftedValEvaluationExpression_eq_timeTableMLE` | theorem | Theorem-Target | Shifted RAM `Val` expression equals the shifted table MLE |
| RAM | `ramTimeTableMLE_shifted_at_bitAddress` | theorem | Theorem-Target | Shifted RAM table MLE at a bit address agrees with the shifted virtual value |
| RAM | `ramShiftedValEvaluationExpression_at_bitPoint` | theorem | Theorem-Target | Shifted RAM `Val` expression equals the shifted table entry at a bit point |
| RAM | `ramShiftedReadCheckAtBitCycle_of_relation` | theorem | Theorem-Target | Shifted RAM read-check consequence at a bit cycle |
| RAM | `ramShiftedWriteCheckAtBitPoint_of_incrementRelation` | theorem | Theorem-Target | Shifted RAM write-check consequence at a bit point |
| Batch | `Stage2LinkageBound` | def | Definitional | Combines register and RAM linkage into one Stage-2 batch surface |
| Package | `TwistConcreteBindingProofPackage` | structure | Definitional | Packages one accepted concrete Stage-2 linkage instance |

## Out of Scope

- generic Twist completeness/soundness
- Stage-1 bytecode and ALU authentication
- Stage-3 continuity
