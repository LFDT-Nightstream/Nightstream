# neo-fold-prototype

`neo-fold-prototype` owns the active Rust proving path for:

- shared SuperNeo and Construction-2 proof plumbing,
- shared Bellpepper/SuperNeo circuit gadgets,
- the first-class direct CCS/R1CS frontend,
- the first-class RV32IM frontend,
- the published proof boundary,
- Spartan/decider wrappers,
- and generic VM contract descriptions used by VM frontends.

## Current owner map

- `core`: shared SuperNeo, Construction-2, proof, opening, and run plumbing
- `circuit`: shared Bellpepper/SuperNeo gadgets
- `frontends/direct_ccs`: first-class arbitrary CCS/R1CS direct path
- `frontends/rv32im`: first-class RV32IM machine path
- `public_proof`: published proof boundary and per-frontend adapters
- `decider`: Spartan/decider wrappers
- `vm`: generic VM descriptions used by VM frontends
- `bin`: thin diagnostics/probe entrypoints

## Important constraints

- Rust structure should follow runtime ownership, not Lean file layout.
- The exact Rust↔Lean protocol boundary must remain stable.
- The audit path should stay narrow and out of the hot path by default.
- Public APIs should be curated; internal owners should be imported directly by
  tests and tooling when they are intentionally exercising internals.

## Planning docs

- [specs/neo-fold-prototype-rust-structure-plan.md](./specs/neo-fold-prototype-rust-structure-plan.md)
- [specs/riscv-kernel.md](./specs/riscv-kernel.md)
