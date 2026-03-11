# EmbeddingTheory — Section 5 barrel

## Purpose

- **What it is**: Re-export barrel for all Section 5 (Embedding products with
  evaluation homomorphism) modules.
- **Paper section**: Section 5 (lines 354-401) + Definition 15 (Module Homomorphism,
  Appendix C line 741) + Remark 2.
- **Scope**: Aggregation-only; no new definitions or theorems.

## Modules re-exported

| Module | Paper item |
|---|---|
| `Embedding` | Definition 7 (Coefficient Embedding, lines 358-366) |
| `Thm3Core` | Theorem 3 (Inner Product Transform, line 368) |
| `BarLift` | Definition 8 (Lifting the Transform, line 376) |
| `MatrixTransform` | Theorem 4 (Matrix-Vector Product Transform, line 384) |
| `EvalLink` | Remark 2 (evaluation/`ct` linkage) |
| `ModuleHom` | Definition 15 (Module Homomorphism, line 741) |
| `EvalHom` | Theorem 5 (Evaluation Homomorphism, line 390) |

## Contract Surface

This is a barrel file. Its contract is: importing `SuperNeo.EmbeddingTheory`
transitively provides the full Section 5 embedding chain from Definition 7
through Theorem 5.

## Proof Obligations

None (barrel file).
