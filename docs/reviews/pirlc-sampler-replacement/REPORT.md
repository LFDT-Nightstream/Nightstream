# Canonical PiRLC sampler replacement

Status: implementation and local validation complete. Independent review is
pending; this report is not merge approval. The final Lean build and axiom audits,
fresh CPU/Lean comparisons, Rust regressions, and Metal runtime checks passed.

## Scope

Branch `nico/pirlc-sampler-replacement` starts at
`7f51e1010ce382d15206d4d1fabcb27d88754cfe` on
`nico/f-prime-constraints-cuda-formal`. The earlier PR123 work was checkpointed
separately at `5b574d781`; it was not merged into this implementation.

The goal is one maintained implementation. The new sampler replaces the old
key and circuit in place. This work does not import the `Wide/` layout overlay
or the later constraint-reduction experiments from PR123.

The profile remains Goldilocks, `b=2`, `k_rho=16`, `B=65536`, rank 22, 28 rounds,
and the same key seed, domain, zero-extension policy, and maximum 4,708,530
key columns. Protocol binding uses Poseidon2. No new Cargo feature or
configuration environment variable is introduced.

## Protocol and proof coverage

The sampler interprets four transcript fields as a base-p integer, reduces
modulo `5^54`, and decodes 54 coefficients in `{−2,…,2}`. The strong set and
norm bounds are unchanged. The six-modulus CRT gadget has 681 rows per
challenge and proves soundness and total witness construction.

The independently checkable sampler definitions and component proofs were
reused from `e1a7c96a6`. The production integration changes the existing
`ProductionKey`, transcript, PiRLC lifecycle, layout, witness exporter, and
matrix owners. The current package has 19 matrix blocks and no tag-5 mapped
blocks. The old First54 selection and rejection-window implementations are
removed. Generic arithmetic and native arithmetic regression tests remain.

The proof targets quantify over the current `PerApplicationFixedPoint`
package. That name now denotes the canonical wide-sampler implementation;
there is no separate old package behind the assignment targets.

`HyperNovaCompleteness.recursive_nifs` and
`HyperNovaAcceptedNext.recursive_extend` / `base_extend` retain honest-prover
and accepted-successor coverage. They no longer require sampler success.
Actual prior openings, valid application advice, and a nonwrapping counter
remain required. Arbitrary-assignment soundness is retained: actual rows imply
the verifier's challenges and complete step relation, without an assumed
canonical witness or source-custody premise.

The concrete Phi81 low-norm invertibility proof is installed and audited.
Production consumers supply it instead of carrying an unproved premise.
No new `sorry`, axiom, native decision shortcut, unsafe evaluation, or proof
limit override is used.

## Security accounting

The paper baseline is the supplied September 4 SuperNeo v1.2, especially
sections 6–7 and Appendix B, and original
[HyperNova](https://eprint.iacr.org/2023/573) Constructions 2 and 3. Local edited
notes are not substituted for the original paper. Completeness coverage and
soundness guarantees are assessed separately.

For independent uniform four-field draws, the sampler proves the exact
statistical distance `delta = r(N-r)/(N*p^4) < 2^-132`, where `N=5^54` and
`r=p^4 mod N`. Its cached block-oracle comparison includes adaptive requests,
repeated queries, and observation of the raw field lanes.

The security-accounting constructor, selected source bound, final history
targets, and their axiom audits pass. They expose
`sampleQueries(Q) * delta` in the selected source-return and final history
bounds. `FiatShamirModel.of_blockOracle` uses the proved statistical comparison
to construct the existing transfer interface at this combined error. The raw
and balanced experiment correspondences remain explicit FS applicability
obligations. This is not a concrete Poseidon2 model or an adversary translator.
The cumulative sampler charge is `delta * sum_j sampleQueries(Q_j)`; query
budgets include adversarial calls and replays, not only 17 verifier challenges.

Fixed-seed MSIS, applicable Fiat–Shamir transfer, and named Poseidon2 collision
assumptions remain explicit. No numerical total-security claim follows from
the sampler bound. Independent security acceptance is pending.

## Canonical artifacts

| Quantity | Current value |
|---|---:|
| Logical coordinates | 242,590,792 |
| Padded coordinates | 242,590,842 |
| Selected key columns | 4,492,423 |
| Logical rows | 6,064,606 |
| Physical rows | 28,275,820 |
| Physical columns | 28,418,945 |
| Private columns | 28,418,666 |
| Public columns | 278 |
| Matrix blocks | 19 |
| Assignment blocks | 26 |

The general reference application has 4 witness fields, 7,696 locals, and
7,700 rows. Manifest version 3 has one reference, 14 phase children, and
application-local block index 25. Rust rejects the retired manifest and
assignment formats. There is one production package emitter, sealed loader, application assembly
route, and witness executor. The crate blueprint remains a symlink to the
formal artifact.

| Identity | Four Goldilocks words |
|---|---|
| Structural | `[10399493082217691252,12446518666506690329,1813819713387457721,9406193360901034503]` |
| Package | `[2798282647818380236,1009842070586539119,14473839254191436803,6982549496945545207]` |
| Transcript context | `[11272520275878376113,16991288497101528960,13385730300888416615,14951557978917292322]` |
| Verification key | `[6548923502318024247,10758504829621102555,9025540937081052320,16349174509930321794]` |

The context and verification-key digests are distinct. Every identity is
recomputed from its authoritative package and setup inputs before pinning.
Digest agreement does not replace the row and value comparisons below.

## Completed validation

| Check | Result |
|---|---|
| Final full Lean production, test and axiom targets, and emitter at `557f13ef6` | Passed, 74.01 s (incremental) |
| Final static, identity, and complete declaration-export checks at the same commit | Passed, 16.36 / 67.78 / 202.57 s |
| Sampler accounting, selected source bound, exact history targets and affected audits | Passed, 4,134 jobs, 115 s |
| Exact emitted bytes after cached sampler and reference cleanup | Unchanged |
| Every physical A/B/C entry against independent expansion | Passed, 22.96 s |
| Every entry of all 14 logical matrices | Passed, 27.67 s |
| Matrix order, column, and coefficient mutations | Rejected, 72.22 s |
| Base physical rows and complete logical assignment | Passed, 34.18 s; 50 alignment zeros |
| Detached application | Rejected at logical row 6,064,593, 33.31 s |
| Nonzero assignment and recipe mutations | Passed, 126.00 s |
| Sparse commitment | All 1,188 coefficients and three support coordinates match |
| Shared manifest | Selected and identity applications both pass |
| Rust assembly | Seven tests pass, including full package value equality |
| Tooling tests | 96 graph tests run with one platform skip; 18 golden-script, 40 native-coordinator, five validation-wrapper, and three review-manifest tests pass |

The nonzero check covers 26 assignment blocks, 13 nonzero matrix slots,
zero-slot rejection, 256 digest-bit mutations, four digest-word mutations,
and changed Phi81, challenge-word, and output-digest recipes. The initial
serial mutation scan hit its 300-second cap and failed. Parallel scanning of
disjoint immutable row ranges preserves every check and the minimum failing
row; the complete rerun passed. No time cap was raised.

The optimized Lean compiler is the exact requested commit
`3019a32cb6f44782ff1e1210676099d683b8d3a8` from `nicarq/lean4-optimized`.
The repository toolchain remains Lean 4.32.2. All commands use `validate.sh`.
Package emission took 17–18 seconds. This is not a stock-versus-optimized
build-time comparison. Cached materialization of sampler coefficients is
proved pointwise equal to the original function; the base emitter then
completed in 22 seconds. The earlier uncached attempt was stopped incomplete.

## Fresh recursive execution

Both fresh optimized CPU folds passed every producer stage and constructed
their successors. Terminal acceptance at state 3 passed in 167.24 seconds.
The recomputed changed-witness case was rejected in 160.52 seconds. Balanced
opening preparation and verification ran separately: `Eval_K` rejection took
157.19 seconds and `Eval_A` rejection took 158.93 seconds. Both reached the
required `Running { index: 0, reason: ... }` check.

Both fresh Lean runs accepted the full C/R/D result and matched all 945,983
native proof bytes, every caller word, and every one of the 28,418,945 physical
field values. Native and Lean mutation groups also passed. The final handoff
compared the exact inputs consumed by Lean with the native outputs. Only then
were the saved fold fixtures and 19-file archive refreshed.

The first physical replay attempt exited with a shell error after writing its
output because its wrapper was edited while running. Its failed receipt is
preserved. The complete first Lean comparison was repeated successfully with
the stable wrapper; the second comparison also passed. The fresh verifier
calls took 18.25 and 18.24 seconds, and physical replay took 50.32 and 49.67
seconds after the dependency build.

The maintained PiCCS checker ran on both fresh folds. Each accepted the complete
result and rejected all 562 proof, 282 statement, and 843 output mutations.
The second fold also rejected all 56 nonzero-point mutations. Both child-handoff
checks recomputed commitments and public values from all 17 input sources and
matched every returned value. All 40 native Python tooling tests pass with the
new archive.

The larger two-link application passed its full CPU mutation check in 235.65
seconds. Its Metal recursive fold passed in 72.51 seconds. Metal accepted a
CPU-produced proof and matched the CPU rejection row after a false witness was
recommitted, in 68.15 seconds. Device activity was checked.

Rust tests ran during construction of implementation commit `557f13ef6`; the
final Lean checks ran on that committed source. Later changes affect only review
tooling and documentation. The local receipts and scope are summarized in
[VALIDATION.json](VALIDATION.json). These executed comparisons do not constitute
a universal proof of Rust semantics.

The final Rust checks pass for the maintained crate, matrix interpreter,
assignment transport, prepared-value bound, package loader, production binding,
sampler parity, parameters, commitment prefixes and batches, and standalone
minimizer. Workspace release checking includes all targets with Metal enabled.
The missing-sampler-batch regression rehashes the mutated package before
requiring structural coverage rejection. The capacity tests use the unchanged
maximum key rather than the selected reference prefix. Contract tooling has
50 passing tests; its historical evidence is not production approval.

## Retirement and evidence

The obsolete lifecycle, WASM crate, minimizer bridge, legacy GPU adapters,
and orphan Spartan crate are removed. The retained Metal key-expansion
helpers are unchanged in their owning shader. Old replay coordinators and
unsupported registrations are removed; maintained PiCCS and child-handoff
checks use the current library. Timeout cleanup covers nested process groups.

Current README, wiki, artifact, and conformance instructions describe the
maintained path. Historical receipts remain identified as evidence for their
original source revisions. No root PR handoff files are added. Rust CI checks
all PR bases, with pushes limited to main; Lean stays local as requested.

The prepared-package metadata bound is derived from the unchanged maximum
key, not the smaller selected prefix. It permits 292,329 application fields;
the maximum-case compiler regression counts 32,264,258 metadata nodes. See
[the derivation](../../../crates/nightstream/tests/evidence/prepared-fixed-source-bound.md).

Saved fold fixtures must come only from successful fresh two-fold execution
and Lean comparisons. The maintained archive contains 19 interface files and
excludes large private witnesses. Follow
[the conformance workflow](../../../scripts/GOLDEN_CONFORMANCE.md).
