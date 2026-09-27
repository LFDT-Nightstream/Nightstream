# PR #124 review follow-up

The three requested repairs are complete. The PR remains a draft; no independent
approval or complete independent proof-generation run is claimed.

## Stock Lean compatibility

A separate compiler benchmark had already completed a clean build of commit
`0846e4df26d76d75152c9da7c4b97f6e20c2b9d6` with the official
`leanprover/lean4:v4.32.2` release (compiler commit
`f3b06c705e6c85f5314019d5d3baab0fec5b580c`). Its project and all ten dependency
checkouts started with zero compiled objects; `lake --no-cache` disabled downloaded
package build artifacts. The build, through `validate.sh` with the 1,500-second
cap, completed 8,196 jobs in **1,029.01 seconds**. It included production targets,
all test/axiom targets, the canonical emitter and the relevant fixture/verifier
executables. Four additional stock security-audit runs also passed.

This follow-up verified all 1,252 project source files against the committed
source and the current checkout, checked the dependency revisions and clean
tracked sources, and parsed all six retained target/witness closures from the
stock build log. No Lean or Rust implementation source changes in this follow-up.
The actual stock emitter output was compared byte for byte with the shipped
artifact: **128,162,339 identical bytes**. That emitter run took 52.71 seconds.

[STOCK_LEAN_VALIDATION.json](STOCK_LEAN_VALIDATION.json) retains commands, toolchain
identity, dependency pins, source identities, timings and comparison results.
The complete compressed build log is retained alongside it. This adopts and
verifies an existing clean run; it does not claim a second full rebuild or an
independent compiler-correctness proof.

## Evidence coverage

All six retained kernel statements now have proof-only registrations, with their
existing audits, checked closures, metadata commands and review requirements.
The [coverage map](../../../scripts/lean_graph/REPLAY_COVERAGE.md) names each retired
obligation, the retained target, the maintained execution evidence and its limits.

`golden-conformance` registers the existing two-fold native/Lean workflow.
`golden-coordinator-contract` checks its coordinator regressions; complete
source-bound execution and review remain separate requirements. The registered
`independent-generation` gap has no closing gate. Native proof verification and
independent physical-witness reconstruction do not replace complete Lean proof
generation. No retained theorem was deleted or weakened to remove this gap.

The graph suite ran 99 tests with one platform skip; all other tests passed.
Both registered coordinator commands passed their exact completion checks
(18 script tests and 40 native-coordinator tests). The new test fails against
all six missing registrations in the prior policy and also checks that passed
contract tests cannot close the unexecuted generation obligation. CI runs it.
The new coordinator gate explicitly uses Python 3.12 to avoid the system's
older Python interpreter.

## README

The capacity paragraph now states `242,275,092 + 41 × (W + L)`, a maximum key
capacity of 254,260,620 scalar coordinates, and the key-derived limit
`W + L ≤ 292,329`, subject to the other circuit checks. The reference prefix
is distinct from that maximum. Arithmetic checks confirm that one extra field
exceeds capacity by two coordinates and that 7,700 fields give the shipped
242,590,792 logical coordinates. The deleted-crate timing sentence is corrected.

The generic `Option` interface, audited sampler laws, published implementation
history and PR #121 remain unchanged.

## Evidence status labels

Registry schema 2 declares only the assurance `tier`: `Compiler`, `Conformance`,
or `Production`. The report computes `status: Open` while any required evidence
or review is missing. It emits the owner-defined `*-closed` status only after
accepted closure. Authored status claims and the old registry schema are rejected.
No gates, targets, reviews, open requirements or closure conditions were relaxed.

The label regression fails on the prior report and passes after this change.
The graph suite runs 100 tests with one platform skip; all other tests pass.
The stronger independent two-fold generation goal remains unmet; these review
repairs do not close it.
