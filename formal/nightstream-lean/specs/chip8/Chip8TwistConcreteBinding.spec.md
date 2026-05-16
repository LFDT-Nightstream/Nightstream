# Chip8TwistConcreteBinding Spec

## Purpose

- **What it is**: The concrete CHIP-8-local Twist instantiation owner for the
  register and RAM time tables.
- **Key property**: It proves the exact bit-cycle and bit-point identities for
  the concrete shifted register and RAM tables used by the CHIP-8 Stage-2
  design.
- **Protocol role**: This module sits below any authenticated-evidence
  extraction owner. It fixes what the intended concrete Twist surfaces mean; it
  does not prove that generic claim witnesses from the kernel proof already use
  these concrete surfaces.

## Target Formulas

For registers, with concrete shifted table
`RegisterShiftedTimeTable(init, inc)`, the module must expose the exact
identities:

$$
\mathrm{ReadWriteMemoryRelation}(val, addr, rv)
\land
\mathrm{ValidAddressColumns}(ra, addr)
\Longrightarrow
rv(j) =
\mathrm{rwReadCheckExpression}(ra, val, \mathrm{bitVec}(j)).
$$

$$
\mathrm{IncrementRelation}(val, wa, wv, inc)
\Longrightarrow
inc(a,j) =
\mathrm{writeCheckExpression}(
  \mathrm{bitAddress}(a),
  \mathrm{bitVec}(j),
  wa,
  wv,
  val
).
$$

and, for the non-zero-init shifted table,

$$
\mathrm{RegisterShiftedValEvaluationExpression}(init, inc,
  \mathrm{bitAddress}(a), \mathrm{bitVec}(j))
=
\mathrm{RegisterShiftedTimeTable}(init, inc, a, j).
$$

The RAM side must expose the exact analogous formulas for
`RamShiftedTimeTable(init, inc)`.

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./formal/twist-shout-lean/TwistShout/TwistCore.lean`
  - `./formal/twist-shout-lean/TwistShout/TwistValueEval.lean`
- Anchors:
  - Stage-2 value-over-time must be grounded in concrete Twist time tables
  - non-zero initialization is part of the concrete virtual-value identity
  - bit-point identities are the bridge from the generic polynomial language to
    concrete per-cycle memory values

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/TwistConcreteBinding.lean` | Concrete register/RAM Twist identities for CHIP-8 |
| `Nightstream/Chip8/TwistConcreteBindingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Concrete | `RegisterTimeTable` | abbrev | Definitional | Concrete Stage-2 register/I time-table type |
| Concrete | `RamTimeTable` | abbrev | Definitional | Concrete Stage-2 RAM time-table type |
| Theorem | `registerReadCheckAtBitCycle_of_relation` | theorem | Theorem-Target | Concrete register read relation implies the exact read-check identity at a bit cycle |
| Theorem | `registerWriteCheckAtBitPoint_of_incrementRelation` | theorem | Theorem-Target | Concrete register increment relation implies the exact write-check identity at a bit point |
| Theorem | `registerShiftedValEvaluationExpression_at_bitPoint` | theorem | Theorem-Target | Concrete shifted register `Val` expression equals the shifted register table entry |
| Theorem | `registerShiftedReadCheckAtBitCycle_of_relation` | theorem | Theorem-Target | Concrete shifted register read relation implies the exact read-check identity |
| Theorem | `registerShiftedWriteCheckAtBitPoint_of_incrementRelation` | theorem | Theorem-Target | Concrete shifted register increment relation implies the exact write-check identity |
| Theorem | `ramReadCheckAtBitCycle_of_relation` | theorem | Theorem-Target | Concrete RAM read relation implies the exact read-check identity at a bit cycle |
| Theorem | `ramWriteCheckAtBitPoint_of_incrementRelation` | theorem | Theorem-Target | Concrete RAM increment relation implies the exact write-check identity at a bit point |
| Theorem | `ramShiftedValEvaluationExpression_at_bitPoint` | theorem | Theorem-Target | Concrete shifted RAM `Val` expression equals the shifted RAM table entry |
| Theorem | `ramShiftedReadCheckAtBitCycle_of_relation` | theorem | Theorem-Target | Concrete shifted RAM read relation implies the exact read-check identity |
| Theorem | `ramShiftedWriteCheckAtBitPoint_of_incrementRelation` | theorem | Theorem-Target | Concrete shifted RAM increment relation implies the exact write-check identity |

## Proof Obligations

- This owner must remain concrete: it must use the actual CHIP-8-local shifted
  register and RAM tables from `Chip8WitnessMemoryBinding`.
- It must not hide the non-zero-init shift inside an opaque theorem statement.
- It must not claim that authenticated kernel evidence already instantiates the
  generic Stage-2 surfaces to these concrete tables; that is a separate
  theorem-facing obligation.

## Assumption Ledger

- This owner assumes the concrete Twist hypotheses it states explicitly:
  `ReadWriteMemoryRelation`, `IncrementRelation`, and valid-address-column
  premises over the concrete tables.
- This owner does not assume any cryptographic opening/authentication theorem.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Chip8WitnessMemoryBinding`
  - `NonZeroInitTwist`
  - `TwistShout.TwistCore`
  - `TwistShout.TwistValueEval`
- **Downstream consumers**:
  - the future concrete Stage-2 evidence-instantiation owner
  - `Chip8RegisterTimeline`
  - `Chip8RamTimeline`
  - `Chip8TemporalConsistency`
