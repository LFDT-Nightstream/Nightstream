# neo-ccs

CCS relation layer: structures, claims, row-wise checks, and the matrix/polynomial
machinery they need. `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`.

## Owns

- **`relations`** — the §4.1/§7.1 data model and checks:
  `CcsStructure` (matrices + `SparsePoly` f), `CcsClaim` / `CcsWitness`,
  `CeClaim` / `CeWitness`; `check_ccs_rowwise_zero` / `check_ccs_rowwise_relaxed`,
  `check_ccs_claim_opening`;
  `build_superneo_ring_forms` for the ring-lifted forms Π_CCS consumes.
- **`matrix` / `sparse`** — dense `Mat`, `CsrMatrix`, CSC (`CscMat`, `CcsMatrix`),
  `SparseCache`.
- **`poly`** — `SparsePoly` / `Term` for the CCS constraint polynomial.
- **`r1cs`** — `r1cs_to_ccs` / `sparse_r1cs_to_ccs` embedding
  (`f = X₀·X₁ − X₂`), used by the direct-CCS and Bellpepper frontends.
- **`utils`** — tensor points, `mat_vec_mul_ff/fk`, and the direct-sum combinators
  (`direct_sum_transcript_mixed` is the cancellation-resistant production variant).
- **`traits::SModuleHomomorphism`** — the commitment-scheme seam `neo-ajtai`
  implements.
- **`crypto` / `gadgets`** — Poseidon2-based primitives and circuit gadgets used by
  CCS-native circuits.

Executable behavior checks live under `crates/neo-ccs/tests/`. Protocol-critical
relation rules must cite the pinned paper, the active protocol contract, or the
active Lean model instead of a copied per-crate specification.

The selected `PaddedRowIdentity` CE membership check belongs to the fold
lifecycle verifier. `neo-ccs` does not accept alternate CE layouts.
