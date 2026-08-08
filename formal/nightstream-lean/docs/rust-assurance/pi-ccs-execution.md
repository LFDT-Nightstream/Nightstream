For each Rust `Π_CCS` execution:

1. Rust exports the complete statement and serialized `Π_CCS` proof.
2. Lean parses the exact proof bytes.
3. Lean independently replays the paper verifier:
   - reconstructs `Q` and the target;
   - derives all transcript challenges;
   - checks all 24 SumCheck rounds;
   - checks the terminal evaluations and openings;
   - checks the complete `Π_CCS` output.
4. Lean proves:

```lean
checkReceipt expectedRelationId statement rustProof = true →
  PaperPiCCS.Accepts expectedRelationId statement rustProof
```

5. Rust tests that the receipt comes from the production code path.

This proves each recorded Rust computation. It does not require Lean to
reproduce the Rust prover’s randomness.

Proving the Rust implementation correct for every possible input would
additionally require a full refinement proof of the Rust code, which is a much
larger task.

## Implemented boundary

Lean checks the exact selected rectangular profile and binds the receipt to a
verifier-owned relation ID. Rust exports a receipt only after the compact
production verifier accepts it. Rust tests reject proof, output, public-digest,
and transcript mutations.

The 24-round selected-profile fixture uses a test relation ID. It does not
certify the full production matrix artifact, and its large closed replay stays
outside the default suite.

The separate production golden uses a small real R1CS and the normal NIFS
prover and verifier. Its 6-round `Π_CCS` receipt runs in the default Lean suite
with the linked `Π_RLC` and `Π_DEC` checks.
