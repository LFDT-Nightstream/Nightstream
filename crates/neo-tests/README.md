# neo-tests

Integration and end‑to‑end tests across crates. No library code here.

## Scenarios
- E2E fold cycle: Π_CCS → Π_RLC → Π_DEC round‑trip on a toy CCS; assert norms, commitments, eval recomposition.
- Global param gate: GL‑128 preset satisfies `(k+1)·T·(b−1) < B`; tampering triggers failure.
- Bridge: final ME(b,L) → Spartan2 proof; verify succeeds; any public IO tampering fails.
- Negative paths: failed invertibility in `neo-challenge`, opening mismatch, DEC recomposition mismatch.

## Notes
- Read `s` (extension degree), `η`, `d`, and `T` from presets; do not assume fixed numeric values in tests.
