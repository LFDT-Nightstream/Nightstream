# Stage 1 baseline closure

The goal remains active on `nico/f-prime-constraints-cuda-formal`. Keep the
selected `b = 2`, `k_rho = 16` profile and its package unchanged. The existing
lean-graph `stage1-baseline` entry names the remaining work; its registration
and scope are in `scripts/lean_graph/STAGE1_BASELINE.md`.

## Current checkpoint

| Result | Evidence | State |
| --- | --- | --- |
| Complete arbitrary-assignment and terminal context targets | `ActualPiDECOutput.selectedRowsAndPublic_imply_step`, literal `Stage1Assignment`, `Stage1TerminalAssignment` and `Stage1TerminalParent` targets | Checked; exact targets and axiom gates pass. |
| Symbolic selected-terminal false-acceptance bound | `HyperNovaFalseAcceptance.probability_linear_bound`, literal `HyperNovaTerminalFalseAcceptance`; `FALSE_ACCEPTANCE_GATES.json` | Checked at `8084c256`, with its explicit premises and independent review. |
| Public initial/base lifecycle | `Stage1Envelope::initial`, `package.extend`, `package.verify`; `STAGE1_PUBLIC_BASE_EVIDENCE.zip` | Full base assignment/reference commitment and terminal checks pass at `48f06fab`. |
| Actual nonzero-running C/R/D | `NONZERO_NIFS_GATES.json`, `NONZERO_NIFS_REVIEW.json` | Passed at `854327d9`: original witnesses, complete Lean phase results, all 945983 proof bytes and required mutations. |
| Complete nonzero 2-to-3 successor and terminal | `NONZERO_SUCCESSOR_GATES.json`, `NONZERO_SUCCESSOR_REVIEW.json` | Passed at `7310cf24`: complete assignment, all physical/logical rows, terminal acceptance and three rejection cases. |
| Public active extension | `PUBLIC_ACTIVE_VALIDATION.md`, `PUBLIC_ACTIVE_REVIEW.json` | Compiled and reviewed. The short rehashed-child rejection passes. Two full calls await specific longer allowances. |
| Evidence delivery and live map | `EVIDENCE_STORAGE.md`, release manifests; committed map at `c95ea627` | Archives are prepared outside Git. Upload/download verification and live publication remain blocked on access. |

The nonzero execution starts from the actual iteration-2 envelope, with six
nonzero running witnesses. C/R/D produces seven nonzero digits and retains
all sixteen children. Rust and Lean agree on the full phase results and
proof bytes. The complete successor matches every Lean caller word, all
fresh carrier coordinates, its zero tail and the independent commitment.
The raw check passes 29225729 physical rows and 6377559 logical rows.

The public terminal verifier accepts the independent iteration-3 endpoint.
Rehashed and recommitted Pad and matrix mutations fail their exact opening
checks. A changed private witness with a recomputed commitment fails fresh
CCS row zero. Preparation is separate from verification to retain the
300-second cap; no saved flag can replace the verifier.

## Remaining work

1. Run the two reviewed public active-extension cases, `parent-absent` and
   `parent-changed`, if the owner approves their specific allowances. Both
   use the actual iteration-2 witnesses and must return the complete checked
   iteration-3 result. Their expected data never enters `package.extend`.
   The requested 985 seconds per call is the rounded sum of 984.22 seconds
   of measured stages, not a formal runtime upper bound or a granted limit.
2. Authenticate the GitHub CLI, upload the prepared release assets in
   `LFDT-Nightstream/Nightstream`, verify remote bytes and replay downloaded
   inputs. Reports and SHA-256 sums stay in Git; new archives stay outside it.
3. Publish the map from committed inputs through its existing owner account.
   The retained Sites project still returns `project_not_found` to the
   current connection. No replacement site or branch was created.

Protected-checker and production/backend approvals remain external. A local
review or scoped execution does not grant them. No new backend, constraint
change, profile, cryptographic research or numerical deployment budget is
part of this goal.

## Security statement and validation

The selected full-opening terminal event means acceptance with no advice
list of the advertised length and forward application result. On the
original mixed law, the checked target proves:

`Pr[FalseAccept] ≤ Σ(j < depth) [h_j + a_j - g(Q_j,a_j) + deltaFS(Q_j) + weakLoss + testError + 17 * m_j]`.

Here `a_j` is the actual good-active visit mass, `h_j` the marked hash
collision mass, and `m_j` actual adaptive MSIS success on that same guarded
law. Depth, queries, guarded FS models, invertibility, declared clocks and
moments remain explicit. The approved FS/MSIS/hash assumptions are retained.
This is not a bare NIFS-Boolean or unconditional-sampler claim.

The current Rust all-target `--no-run` check passes, including the fixture
binary harness and the real integration target. Ordered Lean static, build
and axiom gates pass. The map passes 2186 source-location checks, 19 Python
tests and 12 JavaScript tests. Package bytes, pins and production code are
unchanged by the new fixture staging. Each passing native validation used the
300-second cap; each Lean call used the existing 1500-second cap. Memory is
reported as measured process RSS, with no new memory ceiling.

Earlier failed capped C/R and C attempts remain recorded in
`NONZERO_NIFS_GATES.json`. The combined frozen graph timeout is retained in
`FALSE_ACCEPTANCE_GATES.json`; the separate security gate passed on that
same frozen snapshot. These failures are not relabeled as passes. Earlier
audit and execution reports retain their original source cuts and scope.
