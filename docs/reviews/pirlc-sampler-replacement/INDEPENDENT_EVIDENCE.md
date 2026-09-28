# Independent execution evidence

Both independent Lean folds, 2→3 and 3→4, both fresh successors, and all
iteration-four terminal acceptance/rejection checks passed. The second fold
consumes the exact first Lean successor. Complete values and all 945,983 proof
bytes match in each fold. Every active canonical row passes; padding is accounted
structurally. Balanced false openings reach the exact late Eval_K/Eval_A errors.

The implementation cut is `8ad1acfb04c988f1cbe2eb138f36c4dca35c3392`.
All 1,736 files in the final execution snapshot match committed Git blobs, and
the package matches its committed LFS identity. Earlier checkpoints retain their
own source versions and failed attempts; they are not relabeled as executions
at the final Git HEAD. The first native comparator was reused with verified
original receipts; the second native comparator and both Lean folds were
produced afresh.

The [release](https://github.com/LFDT-Nightstream/Nightstream/releases/tag/pr124-independent-generation-20260927) carries four archives. Every uploaded archive was
downloaded afresh and checked against the original bytes. Parts are ordered by
their numeric suffix; concatenate them before decompressing. The uncompressed
roots are `step-2-to-3`, `step-3-to-4`, `native`, and `provenance-inputs`.
The first two contain complete producer/checker outputs and receipts. The native
archive contains original inputs and comparison outputs. The provenance archive
contains source snapshots and content-addressed source objects, compiler and
validation logs, retained failed attempts, source binding, and terminal states
and receipts. Large build caches are excluded.

| Archive | Compressed bytes | Combined SHA-256 | Delivery files |
|---|---:|---|---|
| first-fold | 5,998,524,478 | `9988fb5d15ab6b23b6386449b85109f44b8b7bd2cbd08a47986ce83dd2d92529` | [manifest](https://github.com/LFDT-Nightstream/Nightstream/releases/download/pr124-independent-generation-20260927/first-fold-evidence.tar.manifest.json) / [checksums](https://github.com/LFDT-Nightstream/Nightstream/releases/download/pr124-independent-generation-20260927/FIRST_FOLD_SHA256SUMS) |
| second-fold | 6,385,433,135 | `46aaca2c22d696492e2b160b6f9b079a24e99be1061097c5586388e12f1c5e74` | [manifest](https://github.com/LFDT-Nightstream/Nightstream/releases/download/pr124-independent-generation-20260927/second-fold-evidence.tar.manifest.json) / [checksums](https://github.com/LFDT-Nightstream/Nightstream/releases/download/pr124-independent-generation-20260927/SECOND_FOLD_SHA256SUMS) |
| native-comparison | 1,991,962,690 | `e1a9d1e58564d2541bafc9078aed81e076d842699ff59708792d82b0838433c6` | [manifest](https://github.com/LFDT-Nightstream/Nightstream/releases/download/pr124-independent-generation-20260927/native-comparison-inputs.tar.manifest.json) / [checksums](https://github.com/LFDT-Nightstream/Nightstream/releases/download/pr124-independent-generation-20260927/NATIVE_COMPARISON_SHA256SUMS) |
| provenance | 947,740,019 | `df4d9a1385af139bea7ce20512d01ce75eff7438c61129696389d222ff22e7d8` | [manifest](https://github.com/LFDT-Nightstream/Nightstream/releases/download/pr124-independent-generation-20260927/provenance-and-terminal.tar.manifest.json) / [checksums](https://github.com/LFDT-Nightstream/Nightstream/releases/download/pr124-independent-generation-20260927/PROVENANCE_SHA256SUMS) |

[INDEPENDENT_DELIVERY.json](INDEPENDENT_DELIVERY.json) provides each part's URL,
size and digest, fresh-download verification times and the committed-source
binding. These hashes establish transport and file custody. The actual proof,
witness, row and rejection comparisons establish the recorded conformance.

Use the [maintained workflow](../../../scripts/INDEPENDENT_GENERATION.md) for a
new execution. Receipts containing a different machine's paths or executable
identity are not reusable checkpoints. Build the pinned tools and use a fresh
run directory; do not execute historical scratch orchestration scripts from the
archive as the current driver.

[INDEPENDENT_EXECUTION.json](INDEPENDENT_EXECUTION.json) indexes the completed
checks and handoffs; [INDEPENDENT_VALIDATION.json](INDEPENDENT_VALIDATION.json)
records the final regressions and proof/identity validation. Archive manifests
and early logs retain their creation-time scope, including earlier incomplete
states. This final index records the completed run.

All ten [decomposition requests](REVIEW.md) remain unsigned and pending, and
independent conformance review remains required. The PR stays in draft. These
executions do not discharge Fiat–Shamir applicability, fixed-seed MSIS or
Poseidon2 collision assumptions, and they do not claim concrete total security.
