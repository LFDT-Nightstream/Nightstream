# Primitives — Section 4 barrel

## Purpose

- **What it is**: Re-export barrel for all Section 4 (Preliminaries) modules:
  fields, rings, coefficient maps, norms, decomposition, `eq` polynomial,
  MLE, sum-check, polynomial helpers, interpolation, and concrete parameters.
- **Paper section**: Section 4 (lines 268-353) + Appendix B.2 (parameters) +
  Appendix C polynomial tools (Lemmas 5-6, interpolation).
- **Scope**: Aggregation-only; no new definitions or theorems.

## Modules re-exported

| Module | Paper item |
|---|---|
| `Goldilocks` | Goldilocks field implementation |
| `Field` | Definition 1 (fields, rings, dimensions) |
| `Dimensions` | Definition 1 (concrete `η`, `d`, shape helpers) |
| `Ring` | Definition 1 (ring `R_q`), Definition 2 (`ct`) |
| `CoeffMaps` | Definition 2 (`cf`, `cf⁻¹`) |
| `Norm` | Definition 3 (centered `l_∞` norm) |
| `Decomp` | `split_b` balanced decomposition |
| `EqPoly` | `eq(x,y)` polynomial (Section 4, line 274) |
| `MLE` | Multilinear extension identity |
| `SumCheck` | Definition 6 (sum-check protocol) |
| `PolyLemmas` | Lemma 5 (Schwartz-Zippel), Lemma 6 (eq-lifting) |
| `Interp` | Polynomial interpolation/evaluation |
| `Parameters` | Appendix B.2 concrete constants |

## Contract Surface

This is a barrel file. Its contract is: importing `SuperNeo.Primitives` transitively
provides all Section 4 definitions and theorems.

## Proof Obligations

None (barrel file).
