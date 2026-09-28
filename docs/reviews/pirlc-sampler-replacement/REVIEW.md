# Independent review requests

All ten requests are **pending**. Four cover the selected terminal/security
targets, and six cover the retained replay-kernel registrations. The author
prepared these proposals and did not assess, approve or sign them. Earlier
accepted records do not approve the current source and policy.

The source cut is `8ad1acfb04c988f1cbe2eb138f36c4dca35c3392`. This follow-up completes independent two-fold
generation, improves proved replay arithmetic and repairs the execution tooling.
The selected sampler, profile and production package remain unchanged. The
[report](REPORT.md), [execution index](INDEPENDENT_EXECUTION.json) and
[validation record](INDEPENDENT_VALIDATION.json) state the completed checks and
their limits. Subsequent delivery commits update documentation and this bundle.

| Obligation | Proposal and exact binding | Response template |
|---|---|---|
| `fresh-witness-kernel` | [request](requests/4448b3184a5dd2d362be4eabb3e810f2a7716a21099e108dac847fd60c5f77cb.json) | [pending response](response-templates/fresh-witness-kernel.json) |
| `hypernova-linear-security` | [request](requests/3cd9afe26b17761723e01a022c68979f40aa1d411eb98f10340332933af9d259.json) | [pending response](response-templates/hypernova-linear-security.json) |
| `hypernova-terminal-false-acceptance` | [request](requests/44672f0a0e2d454e337921092ebc90dd9a7cc5e4c9aa3ba712825bd984dfc91b.json) | [pending response](response-templates/hypernova-terminal-false-acceptance.json) |
| `pidec-commitment-kernel` | [request](requests/083e4b2624534f18d828cb99a18ce9ec07bbed2bc5af0a9bec4f6abb765b0242.json) | [pending response](response-templates/pidec-commitment-kernel.json) |
| `pidec-evaluation-kernel` | [request](requests/1182dbcc3c2ec8f9753b103db26df7bd2034ea690b613d1b7e8cba8065c73019.json) | [pending response](response-templates/pidec-evaluation-kernel.json) |
| `pidec-witness-kernel` | [request](requests/0e70b2d80ed0e81c59ee3a581610bb9f7f0902878641efc862f46942d59080dc.json) | [pending response](response-templates/pidec-witness-kernel.json) |
| `pirlc-witness-kernel` | [request](requests/50a869a96f84f119e8e1731fff8ec4be8bcd8f89a16184def5a392786aa43dfc.json) | [pending response](response-templates/pirlc-witness-kernel.json) |
| `recursive-loop-kernel` | [request](requests/a58c8afd568b232ca319520ce13d413f1c30c1b3618bc1fb62289366d7477d26.json) | [pending response](response-templates/recursive-loop-kernel.json) |
| `stage1-terminal-assignment` | [request](requests/b2a3a910cebca4376c1b1fe6ae4727b46e4f217e76b8e5c05db9e5f9d705dc3e.json) | [pending response](response-templates/stage1-terminal-assignment.json) |
| `stage1-terminal-parent` | [request](requests/838607bedb6aaaf8f6112cd9589854e80234d85c938182fc8d08a7be7e246b04.json) | [pending response](response-templates/stage1-terminal-parent.json) |

[review-index.json](review-index.json) binds each request to its exact target,
policy, checker and snapshot. The [source manifest](source-manifest.json) has
SHA-256 `67e59393454eca6b3910e88d81db4764adda91f65fed2c4e7de28fea9b7ef43c`. Hashes establish file custody, not
protocol soundness or independent acceptance.

The three compressed manifests retain captured source and stock-built dependency
identities. Full local captures are under
`/tmp/nightstream-independent-review-evidence/review-evidence/snapshots/`.
This temporary cache is not required for obtaining the committed source or the
[published execution evidence](INDEPENDENT_EVIDENCE.md).

To check the source cut, use the commit above, materialize its Git LFS artifacts,
and run `python3.12 -B scripts/fprime_stage1_review_manifest.py check MANIFEST_PATH`.
Keep this later bundle outside that checkout. Use stock Lean 4.32.2 through
`formal/nightstream-fprime/scripts/validate.sh`. A newly built dependency seed
may differ bytewise; capture it and issue a new request instead of claiming the
existing snapshot matches it.

Approval requires independent assessment of every exact statement, premise,
argument, implementation correspondence and parent use. Blank templates and
author-run tests are not approval. The completed conformance execution also
awaits its independent formula/coverage review. Concrete Poseidon2 Fiat–Shamir
applicability remains an external assumption; no numerical total-security claim
follows from these executions.
