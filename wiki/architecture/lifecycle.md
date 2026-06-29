# Lifecycle API

`crates/neo-fold-clean/src/lifecycle/` is the only public surface. Everything below it
is `paper/` (auditable protocol) or `engine/` (implementation). The module doc in
`lifecycle/mod.rs` is normative; this page summarizes it.

## Two paths, two verifier types

The type split makes the verifier-authority boundary structural: `Uncompressed`
carries only what the terminal verifier reads; `UncompressedAudit` additionally keeps
the per-step audit trail (step proofs, public batches, final-fold NIFS proof).

```text
Terminal-only IVC:
  preprocess(params, structure, public_input_len)        one-time, derives vk_fs
  prove(prep, batches)              → UncompressedAudit   runs NIFS per batch
  extend(prep, audit, batch)        → UncompressedAudit   one more F′ step
  finish_uncompressed(prep, audit)  → Uncompressed        flush trailing latest, DROP audit trail
  verify_uncompressed(prep, &proof) → Result<()>          terminal-fold re-run only

Audit / decider:
  finish_uncompressed_with_audit(prep, audit) → UncompressedAudit   flush, KEEP trail
  verify_uncompressed_audit(prep, &audit)     → Result<()>          linear-time chain replay
  build_decider_statement(prep, &audit)       → decider::Statement  feeds Spartan compression
  compress(prep, audit) → Compressed;  verify(prep, &c)             PR5 — currently Unsupported
```

### When each verifier applies

- **`verify_uncompressed`** (terminal-only): constant-time-in-history verification by
  re-running the terminal fold (HyperNova §6.3 + SuperNeo §7). Accepts only chains
  whose terminal fold starts from an empty running accumulator — i.e. **single-chunk**
  chains. Multi-chunk histories are rejected explicitly
  (`TerminalOnlyMultiChunkUnsupported`): the evidence binding earlier chunks' counters
  and boundary coordinates lives in per-step rows that `Uncompressed` drops by design.
- **`verify_uncompressed_audit`** (chain replay): walks every step. Catches tampers in
  audit-trail fields (`steps`, `public_batches`, `final_fold.nifs`) that the
  terminal-only verifier ignores by design. Required for multi-chunk chains until the
  compressed decider lands; also the right tool for diagnostics and red-team tests.
- **`verify`** (compressed): the Spartan2 path. The seam is wired —
  `compress` finalizes, replay-verifies, builds the decider statement, and calls
  `decider::prove` — but the PR5 decider itself returns `Unsupported` today.

Beyond replay, `verify_uncompressed` independently re-checks the recorded final
accumulator: witness shape, commitment match, public-input projection, low-norm bound,
and the CE relation `y_ring = mle(M_j·Z)(r)` against the *opened* witness — the
`acc_digest` alone is never authority (see
[Transcript & digests](../protocol/transcript-and-digests.md)).

## Batching: `FoldSchedule`

`lifecycle/schedule.rs` — caller UX, not a protocol primitive:

- `RowsPerStep(1)` (default) — one row per fold, lowest latency.
- `RowsPerStep(n)` — amortize per-fold cost; `n` rows fold per step (the fresh-K per
  fold; capped by the parameter `MAX_FRESH_K = 61`).
- `WholeRun` — everything in one fold step.

`partition<T>` is generic over the row type so each frontend batches its own shape.

## Preprocessing

`preprocess` derives the `VerifierKey` (vk_fs) from `(params, structure)` and caches
the optimized-engine structure analysis (`OptimizedStructureCache`). Frontends wrap it:
`direct_ccs::preprocess` reads the Ajtai setup from the verifier-owned global registry;
`preprocess_seeded` derives it deterministically from a seed (test/demo convenience —
prover-supplied setup is not acceptable in production).

## Error design

`lifecycle::Error` is deliberately loud: every rejection names the violated invariant
and its location (e.g.
`FinalAccumulatorLowNormViolation { index, row, col }`). The red-team suite
(`tests/system/lifecycle_redteam.rs`) asserts specific error variants, so error
messages are part of the tested contract.
