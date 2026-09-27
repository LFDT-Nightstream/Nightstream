# Independent review requests

All ten requests are **pending**. Four cover the selected terminal/security
targets. Six cover the restored replay-kernel registrations and retain their
existing review requirements. The implementation author prepared the proposals
and did not assess, approve, or sign them. Earlier accepted records do not
approve this source and policy.

The reviewed source cut is `726d2b8f3395d1b8c08c3dbc25d2cca6d9a7c5b7`. The Lean and Rust implementation is unchanged
from `0846e4df26d76d75152c9da7c4b97f6e20c2b9d6`; the follow-up repairs documentation
and evidence registration. [REVIEW_FIXES.md](REVIEW_FIXES.md) records the clean
stock build and exact emitter comparison. Subsequent commits only add this
review bundle.

| Obligation | Proposal and exact binding | Response template |
|---|---|---|
| `fresh-witness-kernel` | [request](requests/132cdbb42749a964dc3b181e1834d43ba968217e2be404e54089d9de215d9c6c.json) | [pending response](response-templates/fresh-witness-kernel.json) |
| `hypernova-linear-security` | [request](requests/b1ee9ec10987ba62c3879b8b7e6982bead595b508570c4bba86ab8359b27112b.json) | [pending response](response-templates/hypernova-linear-security.json) |
| `hypernova-terminal-false-acceptance` | [request](requests/c3f58e68640c8f50c5bf1b9f7db81eb0fe3842d1980ae02694691df370f9b6e3.json) | [pending response](response-templates/hypernova-terminal-false-acceptance.json) |
| `pidec-commitment-kernel` | [request](requests/dbb7e282b55126cd509a6f8904deda7f430eb1c486e76caaec82bd9ce9739461.json) | [pending response](response-templates/pidec-commitment-kernel.json) |
| `pidec-evaluation-kernel` | [request](requests/dde19f42daedea3c5ac45e8dbe7a0d433d7bfe5e63364b9ea06032521a452016.json) | [pending response](response-templates/pidec-evaluation-kernel.json) |
| `pidec-witness-kernel` | [request](requests/a4dc7c4789c1d8c15790b8e1b28f3f5aaa7fea4c4537477e42287f52bbb61c45.json) | [pending response](response-templates/pidec-witness-kernel.json) |
| `pirlc-witness-kernel` | [request](requests/ba6cf6e3acf950d62ecb765846942ede0de30c037b6309c93b324ee6d3ea9023.json) | [pending response](response-templates/pirlc-witness-kernel.json) |
| `recursive-loop-kernel` | [request](requests/600ea1591703ca80e6258a5a54454af7ab009f139c01736db6bfb21a72f5654b.json) | [pending response](response-templates/recursive-loop-kernel.json) |
| `stage1-terminal-assignment` | [request](requests/0d2a8664353bc26aab1e84de32f0613578bca91894ec7157e2f8dd2018beebef.json) | [pending response](response-templates/stage1-terminal-assignment.json) |
| `stage1-terminal-parent` | [request](requests/a0229f1496f6161fd5257a4153bd4aea1b2af85533edec701c55cf403d355fe0.json) | [pending response](response-templates/stage1-terminal-parent.json) |

[review-index.json](review-index.json) names each exact request, policy, checker
and snapshot binding. [source-manifest.json](source-manifest.json) binds the
source/artifact cut; its SHA-256 is
`bc6f5733358d54e746933c3f3a548ba05f71a49f2b3c33f96ab51da231aea460`. File hashes establish custody, not protocol
soundness or independent acceptance.

The three compressed snapshot manifests retain all captured file identities,
including the clean stock-built dependency seed. They contain no compiled
binaries or witness payloads. Complete local captures are in
`/tmp/nightstream-pr124-review-fixes/review-evidence/snapshots/`; this temporary
store is not in Git and might not exist on another computer.

To verify the source cut, use the commit above, materialize its Git LFS artifacts,
and run `python3.12 -B scripts/fprime_stage1_review_manifest.py check MANIFEST_PATH`.
Keep this bundle outside that checkout when checking the older source cut. Use
`leanprover/lean4:v4.32.2` through `formal/nightstream-fprime/scripts/validate.sh`.
A separately rebuilt library seed may have different bytes; capture it and issue
a new request instead of claiming that the existing snapshot matches it. The
frozen manifests permit a file-by-file comparison.

Approval requires the existing independent checker process. The review must
assess each exact target, every premise, the argument, its correspondence to the
selected implementation and its use by the parent. Blank response templates
and successful author-run tests do not constitute approval.

Concrete Poseidon2 Fiat–Shamir applicability remains an external assumption.
The sampler term covers the full translated query budget and does not provide
a numerical total-security estimate. The kernel requests do not replace the
open complete independent-generation coverage described in
[the replay map](../../../scripts/lean_graph/REPLAY_COVERAGE.md).
