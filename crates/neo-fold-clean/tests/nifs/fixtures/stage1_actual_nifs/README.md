# Selected actual NIFS fixture

These outputs use the Lean-proved shared running-transition flag and the
86-row Poseidon retained template. The complete staged C → R → D run is in
`/tmp/nightstream-running-flag-sGbJD2/nifs`. The 27.13% quotient checkpoint remains in Git at
`6bcdbb7c` (proof) and `f42a6d53d` (integration). The shared-flag checkpoint
includes the selected Lean layout, Rust consumer, and these checked fixtures.

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
or a new execution with nonzero prior running claims. Final published-fixture
and Nightstream consumer checks are recorded in the constraint experiment
plan. Every native invocation uses the project 300-second cap; Lean uses
the 1,500-second cap. Prover benchmarks remain paused.

The Nightstream Goldilocks profile remains `b = 2`, `k_rho = 16`, `B = 65536`,
with Poseidon2 protocol binding. The package identity is `[1105382808279536156, 17493643376564198179, 11015230893293348642, 7958894448054009516]`.

| Measure | Quotient checkpoint | Selected layout |
|---|---:|---:|
| Committed coordinates | 184,359,564 | 172,217,934 |
| Logical rows | 4,703,127 | 4,147,335 |
| Matrix nonzeros | 3,001,571,645 | 2,968,490,185 |

Committed coordinates are 31.9327% below the original 253,011,276 baseline.
Matrix nonzeros remain above the original 2,335,822,475 baseline. These
counts do not establish an overall performance improvement.

From the repository root:

```sh
timeout --signal=KILL 300 cargo test -p neo-fold-clean --release \
  --test nifs_stage1_nifs selected_nifs_saved_actual_result_matches_lean \
  -- --exact --nocapture
```

These SHA-256 values identify published bytes; they are not protocol authority.

| Artifact | SHA-256 |
|---|---|
| Selected package | `fc3d8a8e798fde5ebabd388c64d5caa3d29116cfac1bb38a7142e888cdd66c6a` |
| Base caller fixture | `f2babc267f704a552048dcb395acb8d385e7c6ebe693ad306d4965e75295e152` |
| actual_result.json | `17ebc93d1fd5e9d70a30547f23ed14582e62a00e8edcbdd60051af978d576b23` |
| proof.bin | `486ea01ecbedad5583b3cd74aa5333c593d6553801da40472ef996f5fc86ab1e` |
| nightstream-fprime-stage1-base-nifs-result-v1.json | `0bd3f9b463275359e4de1580ea27a02a87d20ee0da211a8fa21e7e69a5b7985a` |
| nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json | `c1511bd60ed2722a1957edec3931e121d6bdba7576fd87e13fbe14e205eefc54` |

The archive `docs/reviews/nightstream-fprime-requirements/NATIVE_NIFS_EVIDENCE.zip`
and its `.md` record remain historical evidence for the 2026-09-12 source.
They do not contain or certify the present fixture bytes. Regenerate affected
fixtures from Lean; do not replace independent expectations with Rust output.
