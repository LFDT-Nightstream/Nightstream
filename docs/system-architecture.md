# Neo Lattice System Architecture

This document provides a high-level overview of the Neo Lattice system architecture, showing the flow from high-level inputs through various stages of proof generation.

## System Flow Diagram

```
+-----------------------------+
| High-Level Input            |
| (e.g., RISC-V/WASM/Circuit) |
| + Public Inputs x           |
| + Private Witness w         |
+-----------------------------+
            |
            v (External: Not in Neo)
+-----------------------------+
| Arithmetization to MCS      |
| Instance (via CCS)          |
| - Define structure s        |
| - Decompose z to Z          |
| - Commit c = L(Z) via Π_Mat |
+-----------------------------+
            |
            v (Pair with k-1 running ME claims for recursion)
+-----------------------------+
| Stage 1: CCS Reduction      |
| via Π_CCS (with Sumcheck)   |
| - Encode Q polynomial       |
| - Sumcheck over hypercube   |
| - Output: k ME claims       |
| + Shout for lookup folding  |
|   (read-only memory)        |
| + Twist for read/write      |
|   memory folding            |
+-----------------------------+
            |
            v
+-----------------------------+
| Stage 2: Aggregation        |
| via Π_RLC                   |
| - Random challenges ρ_i     |
| - Combine to 1 high-norm ME |
+-----------------------------+
            |
            v
+-----------------------------+
| Stage 3: Decomposition      |
| via Π_DEC                   |
| - Split Z to k-1 low-norm   |
| - Output: k-1 ME claims     |
+-----------------------------+
            |
            v
+-----------------------------+ <---+
| Stage 4: Recursive Loop     |     |
| - Repeat Stages 1-3 with    | (Feedback for IVC/PCD)
|   new MCS + running ME      |
+-----------------------------+ ----+
            |
            v (After all steps folded)
+-----------------------------+
| Stage 5: (Super)Spartan     |
| Integration                 |
| - Reduce to multilinear     |
| - Commit via FRI            |
+-----------------------------+
            |
            v
+-----------------------------+
| Output: Succinct Proof      |
| - Transcript with commits,  |
|   evaluations, FRI openings |
+-----------------------------+
```

## Architecture Overview

The Neo Lattice system processes high-level computational inputs through a multi-stage pipeline:

### Input Stage
- **High-Level Input**: Accepts various computation formats (RISC-V, WASM, circuits)
- **Public Inputs (x)**: Publicly known values
- **Private Witness (w)**: Secret values that prove computation correctness

### Arithmetization Stage
- **MCS Instance Creation**: Converts input to Matrix Commitment Scheme instance via CCS
- **Structure Definition**: Defines the computational structure `s`
- **Decomposition**: Breaks down `z` into `Z`
- **Commitment**: Creates commitment `c = L(Z)` via `Π_Mat`

### Core Processing Stages

#### Stage 1: CCS Reduction
- Uses `Π_CCS` protocol with Sumcheck
- Encodes Q polynomial
- Performs sumcheck over hypercube
- Outputs k ME (Matrix Extension) claims
- Handles memory operations:
  - **Shout**: For lookup folding (read-only memory)
  - **Twist**: For read/write memory folding

#### Stage 2: Aggregation
- Uses `Π_RLC` (Random Linear Combination) protocol
- Applies random challenges `ρ_i`
- Combines multiple claims into single high-norm ME

#### Stage 3: Decomposition
- Uses `Π_DEC` protocol
- Splits high-norm Z into k-1 low-norm components
- Outputs k-1 ME claims for next iteration

#### Stage 4: Recursive Processing
- Implements IVC (Incremental Verifiable Computation) / PCD (Proof-Carrying Data)
- Repeats Stages 1-3 with new MCS and running ME claims
- Provides feedback loop for recursive proof construction

#### Stage 5: Final Integration
- **(Super)Spartan Integration**: Final reduction phase
- **Multilinear Reduction**: Converts to multilinear form
- **FRI Commitment**: Uses Fast Reed-Solomon Interactive Oracle Proofs

### Output
- **Succinct Proof**: Final compressed proof containing:
  - Transcript with all commitments
  - Polynomial evaluations
  - FRI opening proofs

## Key Protocols

- **Π_Mat**: Matrix commitment protocol
- **Π_CCS**: Customizable Constraint System protocol
- **Π_RLC**: Random Linear Combination protocol  
- **Π_DEC**: Decomposition protocol
- **FRI**: Fast Reed-Solomon Interactive Oracle Proofs

## Memory Handling

The system supports two types of memory operations:
- **Read-only memory**: Handled via "Shout" mechanism for lookup folding
- **Read/write memory**: Handled via "Twist" mechanism for memory folding

This architecture enables efficient recursive proof generation while maintaining succinctness and verifiability.
