# SecurityModel — Section 6 + Appendix C security barrel

## Purpose

- **What it is**: Re-export barrel for the interactive-reduction security framework
  (Section 6), lattice/commitment primitives, and security assumptions from Appendix C.
- **Paper section**: Section 6 (lines 402-446) + Appendix C security items
  (Definitions 16-18, Theorems 2, 8-9).
- **Scope**: Aggregation-only; no new definitions or theorems.

## Modules re-exported

| Module | Paper item |
|---|---|
| `InteractiveReductions` | Definitions 9-10 (Weak/Strong reductions), Theorem 6 (Composition) |
| `ProofSystem.Types` | Proof-system facade types (Context, Claim, Witness) |
| `ProofSystem.Security` | Probability/error model (`ProbModel`, `ErrorModel`) |
| `ProofSystem.Negligible` | `ErrorFn`, `IsNegligible` |
| `ProofSystem.Lattice` | Definition 16 (MSIS), Definition 18 (Ajtai), Theorem 2 |
| `ProofSystem.LatticePaper` | Goldilocks Appendix B.2 lattice-parameter family used by the narrowed final-theorem path |
| `ProofSystem.LatticeReductions` | MSIS-to-Ajtai security reductions |
| `InvertibilityAxioms` | Theorem 8 (Low-norm invertibility) |
| `InvertibilityGoldilocks` | Concrete Goldilocks Theorem-8 proof at the Appendix B.2 floor |
| `SamplingSet` | Definition 17 (Strong sampling), Theorem 9 (Expansion factors) |

## Contract Surface

This is a barrel file. Its contract is: importing `SuperNeo.SecurityModel`
transitively provides the full security framework needed by the folding protocol.

## Proof Obligations

None (barrel file).
