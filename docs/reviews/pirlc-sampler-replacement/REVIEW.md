# Independent review requests

All four requests are **pending**. The implementation author prepared the
proposals and did not assess, approve, or sign them. Earlier accepted records
refer to earlier source and do not approve this change.

The reviewed source cut is `8dc2b67506f0d56759fa170d1dbf70779a0f47b6`. Its protocol implementation is unchanged
from `557f13ef6203dbb0a70d359ccad42dfcac654d34`, on which the final local Lean
build and audits passed. Subsequent changes add this review package and prepare
Git LFS in CI. The captured Lean, Rust, and proof-checker sources are unchanged.

| Obligation | Proposal and exact binding | Response template |
|---|---|---|
| `stage1-terminal-assignment` | [request](requests/3d766c00b1d8747f44039edac01c07c17a2b69860d4d912acd57f80c38ab99b0.json) | [pending response](response-templates/stage1-terminal-assignment.json) |
| `stage1-terminal-parent` | [request](requests/2bcf7793c34172e930f3775722acc748232f6d5ee06c912343719f5621d34100.json) | [pending response](response-templates/stage1-terminal-parent.json) |
| `hypernova-linear-security` | [request](requests/131fae680acaa61a22c5456772bfea0959d9334e5fdc929f42e96972d6720a46.json) | [pending response](response-templates/hypernova-linear-security.json) |
| `hypernova-terminal-false-acceptance` | [request](requests/a9e217b7642a7b3935bbacb8c738cb652f33343676888471366b81be7dc7c189.json) | [pending response](response-templates/hypernova-terminal-false-acceptance.json) |

[review-index.json](review-index.json) lists the exact request, policy, checker
and snapshot bindings through each request. [source-manifest.json](source-manifest.json)
is the deterministic source/artifact cut; its SHA-256 is
`b59d7185514e7e6d2ff1168a9d143b0aabb4c53bc9366a93817647343f72f4b5`. These file hashes establish custody,
not protocol soundness or independent acceptance.

The two compressed snapshot manifests retain every captured file identity,
including the dependency seed. They contain no compiled binaries or witness
payloads. The complete local captures are in
`/tmp/nightstream-pirlc-replacement/review-evidence/snapshots/`. That temporary
store is not in Git and is not guaranteed to exist on another computer.

A reviewer should use the source commit above, materialize its Git LFS artifacts,
and check the source manifest with
`python3.12 -B scripts/fprime_stage1_review_manifest.py check MANIFEST_PATH`.
Keep the review package outside that checkout when checking the older source
cut. The build used optimized Lean commit
`3019a32cb6f44782ff1e1210676099d683b8d3a8`, based on Lean 4.32.2;
all Lean commands must use `formal/nightstream-fprime/scripts/validate.sh`.

The graph requests also bind the exact local library seed. An independently
rebuilt seed can have different bytes; capture it and issue a new request
with the same proposal instead of treating the old snapshot as identical.
The frozen manifest permits a file-by-file comparison of the two cuts.
An approved review requires the existing independent checker process. Empty
response templates and successful author-run tests do not satisfy that process.

The review must assess the exact target, every premise, the argument, its
correspondence to the selected implementation, and its use by the parent.
In particular, the security requests retain concrete Poseidon2 Fiat–Shamir
applicability as an external assumption. The proved sampler transfer accounts
for the complete query budget; it does not establish that applicability or a
numerical total-security estimate.
