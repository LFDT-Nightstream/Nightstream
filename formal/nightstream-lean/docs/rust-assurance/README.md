# Rust assurance with Lean

Lean checks committed Rust results against independent protocol models.

- [`Π_CCS` execution checking](pi-ccs-execution.md) explains the general
  receipt theorem.
- The production golden runs a fixed 1-row, 54-column R1CS through the normal
  `NIFS.P -> NIFS.V` path.
- Lean checks `Π_CCS`, `Π_RLC`, and `Π_DEC`, including all 56 Poseidon2
  permutations and the exact 54-of-64 `Π_RLC` sampler.
- Lean proves that one accepted combined receipt satisfies all three paper
  acceptance relations and the cross-phase output links.
- Rust regenerates the receipt, compares it byte for byte, and rejects
  mutations in each phase.

Run the normal drift check with:

```bash
cargo test -p neo-fold-clean --release \
  --test nifs_production_golden_receipts
```

This is artifact-checked assurance for the committed execution. It is also
Rust-conformant evidence that the production path emits the same artifact. It
is not a proof of the Rust implementation for every input.

Lean checks how the `Π_CCS` output digest enters the transcript. Rust still
owns the SIS recomputation of that digest in this artifact. Do not use this
receipt as a Lean-only proof of the complete Fiat-Shamir binding.
