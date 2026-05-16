# Rv64IMTrivialPredicateArithmetic Spec

## Purpose

- **What it is**: The theorem-facing contract for trivial Stage-1 predicates
  whose proof rule is direct arithmetic on already-opened low bits or bytes.
- **What it is not**: It is not a lookup-table interface and it does not own
  wide validity relations such as signed remainder checks.
- **Protocol role**: It fixes the exact arithmetic discharge boundary for
  natural-alignment predicates in the RV64IM expanded-bytecode kernel.

## Target Formulas

Define the natural alignment width family:

$$
\mathrm{AlignmentWidth} \in \{\mathrm{byte},\ \mathrm{halfword},\ \mathrm{word},\ \mathrm{doubleword}\}.
$$

with byte widths:

$$
\mathrm{bytes}(\mathrm{byte}) = 1,\ 
\mathrm{bytes}(\mathrm{halfword}) = 2,\ 
\mathrm{bytes}(\mathrm{word}) = 4,\ 
\mathrm{bytes}(\mathrm{doubleword}) = 8.
$$

Define the architectural predicate:

$$
\mathrm{NaturalAlignment}(w, addr)
:
\iff
addr \bmod \mathrm{bytes}(w) = 0.
$$

Define the arithmetic discharge predicate over the opened low-byte residue:

$$
\mathrm{ArithmeticAlignmentFromLowByte}(w, lowByte)
:
\iff
lowByte \bmod \mathrm{bytes}(w) = 0.
$$

The exact theorem target is:

$$
\mathrm{NaturalAlignment}(w, addr)
\iff
\mathrm{ArithmeticAlignmentFromLowByte}(w, addr \bmod 256).
$$

More generally, if a Stage-1 witness exposes `lowByte = addr mod 256`, then:

$$
\mathrm{NaturalAlignment}(w, addr)
\iff
\mathrm{ArithmeticAlignmentFromLowByte}(w, lowByte).
$$

So alignment is fully discharged from the opened low-byte residue with direct
arithmetic and no dedicated alignment-table family.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - trivial predicates are arithmetic when they depend only on opened low bits
  - `VAssertAligned`
  - Stage-2 memory-width handling and natural alignment

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Stage1/TrivialPredicateArithmetic.lean` | Arithmetic discharge of natural alignment from low-byte data |
| `Nightstream/Rv64IM/Stage1/TrivialPredicateArithmeticInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Widths | `AlignmentWidth` | inductive | Definitional | The exact natural-alignment width family |
| Widths | `AlignmentWidth.bytes` | def | Definitional | Width-to-byte-count map |
| Predicate | `NaturalAlignment` | def | Definitional | Architectural alignment predicate |
| Predicate | `ArithmeticAlignmentFromLowByte` | def | Definitional | Direct arithmetic discharge over low-byte residue |
| Theorem | `naturalAlignment_iff_arithmetic_from_lowByte` | theorem | Theorem-Target | `addr mod 256` is sufficient to decide alignment |
| Theorem | `naturalAlignment_iff_of_lowByte_eq_mod` | theorem | Theorem-Target | Any explicit low-byte witness equal to `addr mod 256` decides alignment exactly |
| Theorem | `naturalAlignment_of_arithmetic_from_lowByte` | theorem | Theorem-Target | Arithmetic discharge from a valid low-byte witness is sound |
| Theorem | `arithmetic_from_lowByte_of_naturalAlignment` | theorem | Theorem-Target | Natural alignment implies the arithmetic low-byte condition |

## Proof Obligations

- The alignment theorem is exact only for the natural widths owned by the
  kernel: 1, 2, 4, and 8 bytes.
- The discharge rule must remain arithmetic over already-opened residues; it
  does not introduce a new lookup-table theorem surface.
- The arithmetic discharge theorem is the semantic owner for Stage-1
  `VAssertAligned` rows and Stage-2 natural-alignment obligations.

## Out of Scope

- branch or ALU comparison predicates
- wide validity tables
- RAM address decomposition beyond the low-byte alignment consequence
