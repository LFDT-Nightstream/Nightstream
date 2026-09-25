# Selected actual NIFS fixture

These fixtures use the selected wide-sampler package, built on the complete
Lean proof checkpoint `a0f4b5b41`. The earlier quotient, shared-flag and PiCCS
checkpoints remain in Git. The profile remains Goldilocks, Poseidon2, `b = 2`,
`k_rho = 16`, and `B = 65536`.

| Measure | Previous selected package | Wide sampler |
|---|---:|---:|
| Committed coordinates | 149,293,044 | 137,341,872 |
| Logical rows | 3,588,191 | 3,248,956 |
| Normalized matrix entries | 2,857,409,270 | 2,607,606,765 |

The package identity is
`[14833867475356745945, 15229254417532271792, 2646223786752830247, 8497811010041712527]`.

The fresh staged C → R → D run is in
`/tmp/nightstream-wide-selection-FAJ41z`. C and R use the original source
witnesses. The parent commitment and canonical split were recomputed.
Active children `[0, 1, 2, 3, 4, 5]` share preparation in one checked opening
batch. Inactive children use the checked zero-opening branch. The normal
NIFS verifier accepts, and all 43 NIFS mutation cases reject.

`actual_result.json` contains complete observed phase values. `proof.bin`
contains the 945,983-byte native encoding. The independent Lean expectation
is `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-base-nifs-result-v1.json`.
The comparison checks the selected package, complete phase values, proof
bytes, children and transcript; all 55 PiDEC mutation cases reject.
The recursive caller fixture is
`formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json`.
Both Lean files were emitted from source proof inputs and child claims. The
native result did not supply their expected output.

| Native stage | Wall time | Peak RSS (KiB) |
|---|---:|---:|
| C proof, source preparation and verification | 118.76 s | 5,522,316 |
| R proof, source preparation and replay | 46.65 s | 3,596,020 |
| Parent commitment and canonical split check | 108.02 s | 3,596,792 |
| Six active child openings with shared preparation | 201.05 s | 5,538,624 |
| D assembly, complete verification and mutations | 114.92 s | 3,596,024 |

Every native invocation uses the project 300-second cap; Lean uses the
1,500-second cap. C proof construction alone takes 75.945 s after source
loading. These are absolute staged-run measurements. Stages repeat some
preparation, so their sum is not a single-call proving benchmark. There is
no controlled before/after speed or memory claim.

This is fixture conformance on an actual base source with zero prior running
claims. It does not prove universal Rust semantics or test a later fold with
nonzero prior running witnesses. The proof and final consumer coverage are
recorded in
`tools/recursive-constraint-minimizer/experiments/wide-sampler-integration.md`.

From the repository root:

```sh
timeout --signal=KILL 300 cargo test -p neo-fold-legacy --release \
  --test nifs_stage1_nifs selected_nifs_saved_actual_result_matches_lean \
  -- --exact --nocapture
```

These SHA-256 values identify saved bytes; they are not protocol authority.

| Artifact | SHA-256 |
|---|---|
| nightstream-fprime-stage1-poseidon2-hash-chain-v1.json | `552037b491125305a45821577584c43fade57caee83e470cf25810f3e6844762` |
| nightstream-fprime-stage1-base-step-fixture-v1.json | `a1b62f2c9bc4e6e7b74b2b2c21024dc7c0e9b351587ab27fcdd66e27ff480ee5` |
| actual_result.json | `a5f6557d51e5a83d0c0776cea089db892998479412246236bd3ed6eb635ab0e5` |
| proof.bin | `f08fb717c6640f0cac3dcb7389584c5b1daa0b43ef6816adf7b524effff84313` |
| nightstream-fprime-stage1-base-nifs-result-v1.json | `8f613bb8d4e6be3564bea2aa0e0e171efe5b31f5bbbdeff7a837970e7af7556d` |
| nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json | `0cb8e4d1848937c1fae322bbde93739cd47c2be4f1395f74c1e6a732f323df76` |

The archive `docs/reviews/nightstream-fprime-requirements/NATIVE_NIFS_EVIDENCE.zip`
and its `.md` record remain historical evidence for the 2026-09-12 source.
They do not contain or certify these fixtures.
