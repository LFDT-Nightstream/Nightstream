# Dependency tools for constraint reduction

Assessment date: 2026-09-22. Scope: existing tools and their use for the
current Lean proof work. This change registers one existing theorem for
dependency export. A focused export passed and gives the dependency paths
below. The broad registered refresh did not complete, so the full registered
graph is not current. No new graph tool or framework was installed. These
checks do not select a production package or establish a performance result.

Lean-graph can help us find reusable lemmas, inspect proof dependencies,
and track which evidence an edit invalidates. It cannot find a smaller
constraint system from a dependency graph alone. Use the existing repository
tool; a second graph framework is not needed for this task.

## What is available

| Tool | Local state | Useful scope |
|---|---|---|
| Repository lean-graph | `scripts/lean_graph/evidence.py`, Python standard library | Declaration dependencies, exact registered targets, evidence freshness |
| `importGraph` | Already pinned in `formal/nightstream-fprime/lake-manifest.json` through Mathlib | Module imports and import cycles |
| Public `patrik-cihal/lean-graph` | No `lean-graph` executable found on `PATH` | Optional interactive dependency viewer |
| Lean-graph MCP | No matching callable tool found in the session tool metadata | No available MCP interface |

The [repository guide](../../../scripts/lean_graph/README.md) defines
`requires`, `used-by`, and `path`. They inspect checked declaration exports.
`status`, `stale`, and `explain` report evidence state. `run` and `checkpoint`
can execute builds and other registered checks; they are separate actions.

The repository exporter records declaration types, definition values, proof
terms, direct references, and source/package provenance. Meaning and proof
freshness are separate. Incomplete or unavailable declaration evidence falls
back to conservative source checks. The local
[`#evidence_closed` check](../../../formal/nightstream-fprime/tests/EvidenceMetadata.lean)
requires a checked theorem of the exact registered target type and permits
only `propext`, `Classical.choice`, and `Quot.sound`. The choice of target and
its permitted premises still needs review.

The public [Lean Graph project](https://github.com/patrik-cihal/lean-graph)
is a different tool. Its current
[extractor](https://raw.githubusercontent.com/patrik-cihal/lean-graph/main/static/DependencyExtractor.lean)
walks declaration bodies to a supplied depth, omits instance dictionaries
by default, and can replace a failed dependency lookup with an empty list.
These choices can help presentation, but omitted edges cannot establish
that a declaration is unused or that a proof has complete coverage.

[importGraph](https://github.com/leanprover-community/import-graph) can help
place a helper without creating an import cycle and can find redundant
imports. It does not show which theorem uses a particular read-support
proof. There is no need to install it again.

## Apply the existing graph to the current proof

The new canonical producer proof now composes sorted-event provenance,
support for each event family, and full/direct execution agreement. Its
public results are in
[CanonicalDirectPhysicalExecution.lean](../../../formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/CanonicalDirectPhysicalExecution.lean):
`events_safe`, `execute_agree`, and `successful_assignment_eq`. The last
result proves equality of the complete retained assignment and output digest
when both executions succeed. It does not state that the complete physical
scratch arrays are equal.

The [graph driver](../../../scripts/lean_graph/ExportMetadata.lean) now exports
`CanonicalDirectPhysicalExecution.successful_assignment_eq`. The existing
[`declaration-metadata` gate](../../../scripts/lean_graph/obligations.json)
requires its completion marker. There is no new obligation or closure
claim. The graph configuration still has no explicit quotient-reduction
target.

The focused export now shows the producer composition. Further use of the
registered graph, after a complete refresh, is:

1. Refresh the export of `successful_assignment_eq`. Its proof includes
   `execute_agree` and `events_safe`, so this one root includes the producer
   composition.
2. Inspect paths from `StoredPhysicalPlan.Plan.events_induction`, the event-family support
   lemmas, and the direct product execution theorem to the registered
   result. This exposes the actual composition, including witness recipes
   and compact-row checks.
3. Use `used-by` and evidence freshness to identify affected registered
   targets after an optimization. Check the relevant soundness,
   constructive completeness, witness transport, extraction, and profile
   results. A target absent from the export remains outside this report.

These steps reuse the present graph and proof owners. They do not require
new compiler logic or a second coverage inventory. The existing compiler
coverage gate checks the authored declaration inventory; that inventory
does not itself prove that every semantic case is covered.

## Exact refresh and query path

For dependency inspection, the change is one import and one
`#evidence_export` in `scripts/lean_graph/ExportMetadata.lean`, plus the
matching completion pattern in the existing `declaration-metadata` gate:

```lean
import NightstreamFPrime.Export.Stage1.CanonicalDirectPhysicalExecution

#evidence_export NightstreamFPrime.Export.Stage1.CanonicalDirectPhysicalExecution.successful_assignment_eq
```

The module is also imported by the explicit library root
`formal/nightstream-fprime/NightstreamFPrime/Export.lean`. This makes the
normal library build produce its compiled module before metadata inspection.

No new obligation or wrapper theorem is needed to query this theorem. Its
existing proof is the authority for its stated result. This registration
does not close runtime performance or full production conformance.

The theorem quantifies over `StoredPhysicalPlan.Plan`, which carries an
erased event-provenance proof. The actual `StoredPhysicalPlan.ofSources`
constructor supplies that proof through `ArraySortMembership.mem_qsort`.
That constructor need not be a dependency of the universal theorem, so a
missing path from `mem_qsort` to this root would not show a missing proof.
Constructor inspection would require a separate export; it is not part of
this minimal registration.

The supported commands below are for a future complete registered refresh.
The current focused result does not make these CLI queries current. Run
from the repository root when the one build queue is free and the source
checkpoint is stable:

```sh
timeout --signal=KILL 1500 python3 scripts/lean_graph/evidence.py \
  --store /tmp/nightstream-evidence run declaration-metadata

timeout --signal=KILL 300 python3 scripts/lean_graph/evidence.py \
  --store /tmp/nightstream-evidence --json path \
  NightstreamFPrime.Export.Stage1.StoredPhysicalPlan.Plan.events_induction \
  NightstreamFPrime.Export.Stage1.CanonicalDirectPhysicalExecution.successful_assignment_eq

timeout --signal=KILL 300 python3 scripts/lean_graph/evidence.py \
  --store /tmp/nightstream-evidence --json requires \
  NightstreamFPrime.Export.Stage1.CanonicalDirectPhysicalExecution.execute_agree
```

The current store already has a registered active criterion. The refresh
uses the existing host build lock and calls `validate.sh build`,
`validate.sh build tests.EvidenceTargets`, then
`validate.sh file ../../scripts/lean_graph/ExportMetadata.lean` in sequence.
It captures source and uses its retained checker build cache; the first
build for a new source state can be cold. Each Lean child has the project
1,500-second cap. The outer cap uses the same policy limit. Read-only query
commands use the project's 300-second non-Lean test ceiling.

## Check performed and limits

A read-only `requires` query against `/tmp/nightstream-evidence` for
`NightstreamFPrime.Export.Stage1.PiRLCCombinationDirectWitness.fullPhase_agrees_directPhase`
returned exit code 1 with `no current complete declaration export`. No
current dependency path was obtained. The query log is
`/tmp/nightstream-tooling-graph-query.json`.

The first refresh then passed its isolated library build in 577 seconds and
its `EvidenceTargets` build in 53 seconds. Metadata inspection failed because
`CanonicalDirectPhysicalExecution.olean` was absent. The module had been
checked through the axiom/test imports, but was not imported by the explicit
library export root. The missing import was added to `Export.lean`.

The second refresh passed the isolated library build with 4,067 jobs in
869 seconds and `EvidenceTargets` with 3,909 jobs in 83 seconds. The broad
export was stopped after about 21 minutes overall, with six of eighteen
roots emitted and more than 500 MB of output. It ended with exit code 1;
the interrupted metadata could not be parsed. Its log is
`/tmp/nightstream-direct-graph-refresh-2.log`. This is not a complete
registered export or current CLI evidence. The source registration and
completion marker remain for later broad use.

The focused driver uses the existing exporter with only the producer root.
Save this exact text as `/tmp/nightstream-direct-producer-graph.lean`:

```lean
import tests.EvidenceMetadata
import NightstreamFPrime.Export.Stage1.CanonicalDirectPhysicalExecution

#evidence_export NightstreamFPrime.Export.Stage1.CanonicalDirectPhysicalExecution.successful_assignment_eq
```

From `formal/nightstream-fprime`, the focused command is:

```sh
timeout --signal=KILL 1500 bash scripts/validate.sh file \
  /tmp/nightstream-direct-producer-graph.lean \
  > /tmp/nightstream-direct-producer-graph.log 2>&1
```

It passed in 33 seconds. The log has 99,092,829 bytes, about 95 MiB. The
complete focused export gives these actual dependency paths, where all
names are under `NightstreamFPrime.Export.Stage1`:

```text
StoredPhysicalPlan.Plan.events_induction
  → CanonicalDirectPhysicalExecution.events_safe
  → CanonicalDirectPhysicalExecution.execute_agree
  → CanonicalDirectPhysicalExecution.successful_assignment_eq

StoredPiRLCCombination.shifted_agree
  → StoredDirectPhysicalExecution.compact_agree
  → StoredDirectPhysicalExecution.executeEvent_agree
  → CanonicalDirectPhysicalExecution.execute_agree
  → CanonicalDirectPhysicalExecution.successful_assignment_eq
```

The extracted paths and scope are saved in
`/tmp/nightstream-direct-producer-graph-paths.json`. They show the connection
of the actual event-provenance and product-replacement proofs to the complete
assignment result. They do not assert fresh evidence for other registered
roots. The isolated build times and export sizes above are validation costs;
a warm workspace build does not remove the observed cost of the broad run.
They are not prover times or constraint savings.

Graph node counts and edge counts are not constraint rows, committed
coordinates, matrix nonzeros, prover time, or memory. Cost formulas and
measurements must remain separate. A cost table can guide which declaration
to inspect, but the graph cannot prove that deleting a witness preserves
the relation. cvc5 or exact algebra tools can propose or reject a stated
candidate class; Lean must prove soundness and constructive completeness
for the accepted transformation. The
[Lean kernel](https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/)
checks proof terms. Dependency visualization does not replace that check
or establish Rust conformance and runtime improvement.

## Finite-field proof reconstruction follow-up, 2026-09-22

[FF_CVC5_Lean](https://github.com/NethermindEth/FF_CVC5_Lean) reconstructs
cvc5 finite-field proofs in Lean. Its repository was archived on September 9,
2026; its README reports that `grind` was faster on its current examples.
This does not establish a faster path for the present obligations. No new
proof dependency is needed for the already checked shared-flag equations.

The [cvc5 finite-field documentation](https://cvc5.github.io/docs/latest/theories/finite_field.html)
describes both the Gröbner-basis and split solvers. The present saved searches
use the Gröbner-basis solver and independent counterexample checks. Solver
results remain candidate evidence; the selected change has separate Lean
soundness, constructive completeness, and witness-map proofs.
