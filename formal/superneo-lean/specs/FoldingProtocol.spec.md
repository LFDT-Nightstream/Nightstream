# FoldingProtocol — Section 7 barrel

## Purpose

- **What it is**: Re-export barrel for Neo's folding scheme for CCS (Section 7):
  constraint system definitions, the three interactive reductions (Π_CCS, Π_RLC, Π_DEC),
  arithmetic obligations, and the final protocol theorem.
- **Paper section**: Section 7 (lines 447-596) + Appendix D proofs.
- **Scope**: Aggregation-only; no new definitions or theorems.

## Modules re-exported

| Module | Paper item |
|---|---|
| `ProofSystem.ConstraintSystem` | Definitions 11-13 (CCS structure, relations) |
| `ProtocolSection71Data` | Explicit protocol-side Definition-14 data owner |
| `ProtocolSection71Context` | Packaged compact Section 7.1 theorem-native owner |
| `ProtocolTargetData` | Explicit protocol-side Section 7.5 target-data owner |
| `ProofSystem.SumCheck` | Proof-system-level sum-check facade |
| `ProofSystem.Folding` | Proof-system folding wrappers (PiCCS/PiRLC/PiDEC) |
| `ProtocolRelations` | Section 7.1 relation predicates |
| `PiCCS` | Section 7.3, Lemma 3 (Π_CCS is strong) |
| `PiRLC` | Section 7.4, Lemma 4 (Π_RLC is weak) |
| `PiDEC` | Section 7.5, Theorem 7 (Π_DEC is a reduction of knowledge) |
| `ArithmeticBundle` | Bundled arithmetic prerequisites for protocol |
| `ArithmeticObligations` | Arithmetic side-conditions for protocol reduction |
| `ProtocolTarget` | Protocol-target bridge (Thm 3 + obligations) |
| `ProtocolMathTarget` | Protocol math-target from arithmetic bundle |
| `ProtocolTheorem` | Final theorem shape (completeness + knowledge-soundness) |
| `ProofSystem.Protocol` | Proof-system entrypoint (final theorem wiring) |

## Contract Surface

This is a barrel file. Its contract is: importing `SuperNeo.FoldingProtocol`
transitively provides the full Section 7 protocol from CCS definitions
through the final composition theorem.

## Proof Obligations

None (barrel file).
