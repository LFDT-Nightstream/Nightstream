# Selected actual NIFS fixture

These are the published outputs of the staged quotient-package base
C → R → D run in `/tmp/nightstream-quotient-nifs-6bcdbb7c`. The quotient
layout was introduced at `8b7c07d8`; the complete Lean producer proof is saved
at signed checkpoint `6bcdbb7c`. The Rust integration changes are still
uncommitted at this update.

C and R use actual source witnesses. The R parent commitment and all canonical
split witnesses were recomputed. Children 0 through 5 are active; their
openings were computed separately. The remaining ten children use the checked
zero-opening branch. The normal NIFS verifier accepts, and all 43 recorded
NIFS mutations are rejected.

`actual_result.json` holds the complete observed phase values. `proof.bin`
holds the 945,983-byte native encoding. The independent Lean expectation is
`formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-base-nifs-result-v1.json`.
The comparison checks the current pinned package, complete phase outputs,
wire encoding, final children and transcript, and the 55 D rejection cases.
The independent Lean recursive caller fixture is
`formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json`.
Both Lean files and the native result/proof are published. This establishes
the fixture's conformance, not universal Rust semantics or a new nonzero-running
NIFS execution. Final Nightstream tests and the candidate lifecycle benchmark
remain pending. No overall speed or memory improvement is claimed.

| Stage | Result | Seconds |
| --- | --- | ---: |
| PiCCS proving and verification | Passed | 195.254 |
| PiRLC parent | Passed | 76.306 |
| Parent replay, commitment and canonical split | Passed | 166.380 |
| Child 0 opening | Passed | 221.122 |
| Child 1 opening | Passed | 222.475 |
| Child 2 opening | Passed | 221.158 |
| Child 3 opening | Passed | 203.917 |
| Child 4 opening | Passed | 208.760 |
| Child 5 opening | Passed | 169.425 |
| NIFS assembly, verification and 43 mutations | Passed | 213.997 |
| Independent Lean base C/R/D result | Passed | 10.797 |
| Independent Lean recursive caller fixture | Passed | 11.906 |

`compare.log` records `complete_nifs_wire=passed bytes=945983`, 55
`pi_dec_mutation` rejections, `saved_actual_pi_dec=passed`, and
`actual_selected_nifs_Lean_comparison=passed`. Each native command used the
300-second cap; Lean commands used the 1,500-second cap. These are stage
execution times, not a matched lifecycle benchmark.

From the repository root:

```sh
timeout --signal=KILL 300 cargo test -p neo-fold-clean --release \
  --test nifs_stage1_nifs selected_nifs_saved_actual_result_matches_lean \
  -- --exact --nocapture
```

The following hashes were computed from the published files. They identify
fixture bytes; they are not protocol authority.

| Artifact | SHA-256 |
| --- | --- |
| Current package | `6216d1f62250a58d073ecf0a908bd074d3834957ec5be3361620bbdeb5a97642` |
| Base caller fixture | `27da134177d8ad9ca26501e25646a90195e159b5f37fb0985592bb8e004a5cbe` |
| `actual_result.json` | `9d7af1fd1b7723bcfc379c2b64ae966df8818b13d396028a9b7c25a6d3eebf1b` |
| `proof.bin` | `3979f992efaa9aaf5bba3e208912530d12f14285e092e07d4b8d6a15cab17dc8` |
| Independent Lean base NIFS result | `493ccaa50df287504b322fe06a5ec6927e2a315c9375d616b9e31ea43d18d59f` |
| Independent Lean recursive caller fixture | `35fc2442a6e68caf835a91af117c42176d82fbc7d50109451c811eb75167bb96` |

The package identity is `[11780343655336100175, 3739837894403928952,
5393028243801154131, 3026325569224679930]`. The Nightstream Goldilocks profile
remains `b = 2`, `k_rho = 16`, `B = 65536`, with Poseidon2 protocol binding.
The package has 184,359,564 committed coordinates, 4,703,127 logical rows,
and 3,001,571,645 matrix nonzeros. Coordinates decrease by 27.1339%; matrix
nonzeros increase by 28.5017% from the original baseline. Recompute through
the staged commands when affected production logic changes. Do not replace
the independent Lean expectation with Rust output.

The archive `docs/reviews/nightstream-fprime-requirements/NATIVE_NIFS_EVIDENCE.zip`
and its `.md` record remain historical evidence for the 2026-09-12 run based
on `1607d34fe12593b39cad6f5c5c1f8b1ce853f8ab`. They contain the old package,
fixture hashes, source snapshots and logs. They do not contain or certify the
new quotient fixture bytes listed above.
