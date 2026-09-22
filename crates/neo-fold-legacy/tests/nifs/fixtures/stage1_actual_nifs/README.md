# Selected actual NIFS fixture

These are the exact outputs of the capped, staged selected-base C → R → D
run completed on 2026-09-12 UTC. C and R use actual source witnesses; D
openings were computed from their canonical split witnesses. The normal NIFS
verifier accepts, and all 43 recorded NIFS mutations are rejected.

`actual_result.json` holds the complete observed phase values. `proof.bin`
holds the 945,983-byte native encoding. The independent Lean expectation is
`formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-base-nifs-result-v1.json`.
The comparison checks the current pinned package, complete phase outputs,
wire encoding, final children and transcript, and the 55 D rejection cases.
It establishes this fixture's conformance, not universal Rust semantics.

From the repository root:

```sh
timeout --signal=KILL 300 cargo test -p neo-fold-clean --release \
  --test nifs_stage1_nifs selected_nifs_saved_actual_result_matches_lean \
  -- --exact --nocapture
```

The source base is `1607d34fe12593b39cad6f5c5c1f8b1ce853f8ab` plus the source
snapshots in `docs/reviews/nightstream-fprime-requirements/NATIVE_NIFS_EVIDENCE.zip`.
That archive contains the stage recipe, computed openings, independent Lean
result, logs and exact source/file hashes. See `NATIVE_NIFS_EVIDENCE.md` in
the same review directory for scope and measured results.

| Artifact | SHA-256 |
| --- | --- |
| Current package | `043bce25083eb15c733a903f4df2958acbc31a59dbe17420cd084c8242bd4d1f` |
| Original base fixture | `2552988651f7dfbc37e59d0e48fdf9403564356bcf14984327d068518c0a2af7` |
| `actual_result.json` | `9f7dccf82567bef88e30d5b844ff0efaeed2ea62dc7ee963e4cd4470966ae834` |
| `proof.bin` | `9bad16146f8e1b35fdf463166b153975a94edb494aa595bd01c79c7453c027b6` |
| Independent Lean result | `f451dd6c32e00b40085d1e0dc86292dffa2c5d1a81d520ad4bfe120165ddf17d` |

The package identity is `[9705822157724451396, 520958727644325895,
9285622073986934000, 874020794279380938]`. The profile remains `b = 2`,
`k_rho = 16`. Recompute through the documented stages when affected production
logic changes. Do not replace the independent expectation with Rust output.
