# neo-ccs
## What it is
Implementation of Customizable Constraint Systems (CCS) relations, structures, instances, and witnesses. Includes sum-check based proving for satisfiability.

NOTE: This crate handles all CCS-specific logic, including structures, direct checks, and proving via sum-check (e.g., ccs_sumcheck_prover/verifier with norm batching). Use neo-sumcheck for generic sum-check tools only—do not add CCS-specific code there. This keeps CCS self-contained for use in folding.

## How it is used in the paper
CCS generalizes R1CS/Plonkish/AIR (Section 1, Definition 16-18). Folding scheme targets CCS over small fields (Section 4). Satisfiability check ensures f(M z) = 0 (Definition 17). Proving via sum-check reduces to evaluation claims (Section 4.1, Π_CCS).

## What we have
- `CcsStructure`, `CcsInstance`, `CcsWitness` types.
- `check_satisfiability` and `check_relaxed_satisfiability` functions.
- Sum-check proving: `ccs_sumcheck_prover` and `ccs_sumcheck_verifier` with norm batching, ZK blinding, and extension fields.
- Multivariate constraint poly as closure.
- Public inputs handling and multilinear encoding.
- Tests for satisfiability (small/large), proving (valid/invalid norms/constraints), and ZK hiding.

## What we are missing
- Commitment to CCS instances (integrated with Ajtai).
- Relaxed CCS for folding (Section 5, Definition 19).
- Conversion from R1CS/Plonkish (as generalized in paper).
- Benchmarks for large constraints (e.g., 2^20).
- Full Fiat-Shamir transcript integration (partial Poseidon2).
