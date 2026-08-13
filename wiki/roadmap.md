# Roadmap

## Implemented

- SuperNeo PiCCS, PiRLC, and PiDEC
- Optimized and PaperExact engines
- Recursive R1CS F' induction
- Nebula memory checking
- Direct CCS audit verification
- Terminal R1CS compilation
- WIP Spartan with Goldilocks, Poseidon2, and WHIR
- Metal device work on supported Apple builds
- Public backend choices for optimized CPU, PaperExact, CUDA, and Metal

## Required work

### CUDA

Implement the canonical one-joint NIFS kernel. It must match the optimized CPU
and PaperExact transcript, proof bytes, output claims, and terminal openings.
Until then, CUDA construction returns an explicit unavailable error.

### Terminal backend

Audit and optimize `wip-spartan`. Complete the deployment-facing verifier
boundary and measure the terminal relation on representative applications.

### Protocol assurance

Keep the Lean model aligned with the Rust relation, complete the remaining
correspondence proofs, review the concrete parameters, and obtain an
independent security audit.

### Frontends

Keep direct CCS as an explicit audit path. Add a new production frontend only
when it supplies an authoritative fixed relation and a terminal-induction
proof.
