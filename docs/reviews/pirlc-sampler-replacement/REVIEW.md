# Independent review requests

All ten requests are **pending**. Four cover the selected terminal/security
targets. Six cover the restored replay-kernel registrations and retain their
existing review requirements. The implementation author prepared the proposals
and did not assess, approve, or sign them. Earlier accepted records do not
approve this source and policy.

The reviewed source cut is `02e5674207bfe7c9d10f7c5a2f7d196357d45102`. The Lean and Rust implementation is unchanged
from `0846e4df26d76d75152c9da7c4b97f6e20c2b9d6`; the follow-up repairs documentation,
evidence registration and status reporting. [REVIEW_FIXES.md](REVIEW_FIXES.md) records the clean
stock build and exact emitter comparison. Subsequent commits only add this
review bundle.

| Obligation | Proposal and exact binding | Response template |
|---|---|---|
| `fresh-witness-kernel` | [request](requests/7adb31408f682452afc60e14e8865c09c710d433c58853c52f63c4968d83d1fa.json) | [pending response](response-templates/fresh-witness-kernel.json) |
| `hypernova-linear-security` | [request](requests/5bb9c1b093c1e68b1e836724a16bc1c7aae869bdaf9eedda65122879c66bb58b.json) | [pending response](response-templates/hypernova-linear-security.json) |
| `hypernova-terminal-false-acceptance` | [request](requests/dbe084b7925c6641ef200d2c226b48e1d7d57c472538d313163e6b9271033fbe.json) | [pending response](response-templates/hypernova-terminal-false-acceptance.json) |
| `pidec-commitment-kernel` | [request](requests/3ea8e375f28c30edebaa979d94248164c280fce8153d9bf34c2258f3bd605607.json) | [pending response](response-templates/pidec-commitment-kernel.json) |
| `pidec-evaluation-kernel` | [request](requests/f5a700908af377dec484b5dd062ead797bbcdc870a6f354b7d02233cad7b475f.json) | [pending response](response-templates/pidec-evaluation-kernel.json) |
| `pidec-witness-kernel` | [request](requests/b569a547bfe3ed16515cda44b0cc3622af3f442565326d7ab8b2ae9596dbc98b.json) | [pending response](response-templates/pidec-witness-kernel.json) |
| `pirlc-witness-kernel` | [request](requests/85196de0f2e1af748756ce8d1ac506af50f29faac6b0b31c5915ccf415179393.json) | [pending response](response-templates/pirlc-witness-kernel.json) |
| `recursive-loop-kernel` | [request](requests/9816fdf02d505b2a41bdc94b7c5c24dc44b5dfac7e3aad35e0cbc99ba42de347.json) | [pending response](response-templates/recursive-loop-kernel.json) |
| `stage1-terminal-assignment` | [request](requests/2ae57d4585031620f69760924580e90384e5150eb17db6c8ac8a3368f1c11682.json) | [pending response](response-templates/stage1-terminal-assignment.json) |
| `stage1-terminal-parent` | [request](requests/2842b789e6402a17aa193a74c29e655855e6b3d96419b9846434763c3d066ca1.json) | [pending response](response-templates/stage1-terminal-parent.json) |

[review-index.json](review-index.json) names each exact request, policy, checker
and snapshot binding. [source-manifest.json](source-manifest.json) binds the
source/artifact cut; its SHA-256 is
`d5901dd67c0a254d633d7d33f5881a103cce09156ff12c91ac4b9a4e47a42728`. File hashes establish custody, not protocol
soundness or independent acceptance.

The three compressed snapshot manifests retain all captured file identities,
including the clean stock-built dependency seed. They contain no compiled
binaries or witness payloads. Complete local captures are in
`/tmp/nightstream-pr124-status-label/review-evidence/snapshots/`; this temporary
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
