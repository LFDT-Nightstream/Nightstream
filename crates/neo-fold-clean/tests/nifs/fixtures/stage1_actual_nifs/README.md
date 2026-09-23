# Selected actual NIFS fixture

The fixture bytes use the reduced package at `41aa5ef63`: shared running
flag, shared PiCCS expressions and powers, and the compact application.
The complete staged C → R → D run is in
`/tmp/nightstream-selected-batch-iUMnNY/nifs`. The 27.13% quotient checkpoint
remains in Git at `6bcdbb7c` (proof) and `f42a6d53d` (integration).
The shared-flag checkpoint remains at `c9aa75b04`.

C and R use actual source witnesses. The parent commitment and all canonical
split witnesses were recomputed. Active children are `[0, 1, 2, 3, 4, 5]`. Child
openings share preparation within checked batches; inactive children use the
checked zero-opening branch. The normal NIFS verifier accepts and all
43 NIFS mutation cases reject.

`actual_result.json` holds complete observed phase values. `proof.bin` holds
the 945,983-byte native encoding. The independent Lean expectation is
`formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-base-nifs-result-v1.json`.
The comparison checks the pinned package, complete phase values, proof bytes,
children, transcript, and 55 PiDEC rejection cases. The independent
recursive caller fixture is
`formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json`.
Both Lean files are generated from the source inputs and child claims;
the native result is not used as their expected output.

This is fixture conformance. It does not establish universal Rust semantics
or a new execution with nonzero prior running claims. The published-fixture and Nightstream consumer checks pass; results are
recorded in `tools/recursive-constraint-minimizer/experiments/piccs-lowering-metrics.json`. Every native invocation uses the project 300-second cap; Lean uses
the 1,500-second cap. Prover benchmarks remain paused.

The Nightstream Goldilocks profile remains `b = 2`, `k_rho = 16`, `B = 65536`,
with Poseidon2 protocol binding. The package identity is `[10236859257084328720, 2333344895175328281, 11662339516011578647, 12470290288965558271]`.

| Measure | Shared-flag checkpoint | Selected layout |
|---|---:|---:|
| Committed coordinates | 172,217,934 | 149,293,044 |
| Logical rows | 4,147,335 | 3,588,191 |
| Matrix nonzeros | 2,968,490,185 | 2,857,409,270 |

Coordinates fall by 13.31%, logical rows by 13.48%, and matrix nonzeros by
3.74% from that checkpoint. Matrix nonzeros remain above the original
2,335,822,475 baseline. These counts do not establish an overall performance
improvement. The wide sampler is merged but remains unselected.

From the repository root:

```sh
timeout --signal=KILL 300 cargo test -p neo-fold-clean --release \
  --test nifs_stage1_nifs selected_nifs_saved_actual_result_matches_lean \
  -- --exact --nocapture
```

These SHA-256 values identify published bytes; they are not protocol authority.

| Artifact | SHA-256 |
|---|---|
| Selected package | `f648cb69b41f47616105c13b6666b3954ece21f85e21c52dae24cea16b494f42` |
| Base caller fixture | `3ca0d0694fa48364262f26acad4ec65d89e6349053d106ac34353618334f4fea` |
| actual_result.json | `6dad0b1efeb14e64b7b38b79f6209cdfd8c3e07c17cd0592441fb7113bea74c6` |
| proof.bin | `3a2d02ccb9a862f8c0a58a3f60afa557bd63998cd978b7b78d96064d05d931c1` |
| nightstream-fprime-stage1-base-nifs-result-v1.json | `56ded93358497fd6acaa976015fbbeccd32b63c93f39edac3567d0dd55cbc61f` |
| nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json | `308c852ad88f46f2d9efcd9f0c18d3215aa254bb883f9956e096be071ce0ed3a` |

The archive `docs/reviews/nightstream-fprime-requirements/NATIVE_NIFS_EVIDENCE.zip`
and its `.md` record remain historical evidence for the 2026-09-12 source.
They do not contain or certify the present fixture bytes. Regenerate affected
fixtures from Lean; do not replace independent expectations with Rust output.
