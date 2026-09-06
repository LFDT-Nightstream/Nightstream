# lean-graph — Lean dependencies and evidence

lean-graph records proof obligations, exports Lean declaration dependencies,
and checks validation evidence. Its current configuration covers the
pilot/PiCCS assignment-proof chain, including the actual public-boundary and
hash-observation target. It does not prove the missing full
Stage 1 decoded-step theorem. The approved owner goal and phase order still
apply. The graph contains existing Lean declarations; the tool does not write
proofs or infer new owner criteria from the paper.

The implementation uses Python 3.10 or later and POSIX process groups and
file locks. It introduces no Python package dependency or Rust feature.

## Local use

For incremental proof work, use the existing `validate.sh build` and `file`
commands shown below. The evidence CLI checks frozen source. Its first project
build is cold; subsequent matching checks can use retained checker builds.

Run from the repository root. Keep the evidence store outside the candidate
source directory. All tasks on this project should use the same agreed
diagnostic store through `--store`, so they can see and resume each other's
matching results. A shared store does not grant checker authority.
The following commands use local diagnostic mode:

```sh
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence explain piccs-assignment
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence checkpoint piccs-assignment
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence status
```

`checkpoint` selects the owner criterion, orders its prerequisites, and resumes
matching successful checks. A failed check stops that checkpoint. Its completed
predecessors remain available for the next attempt. Missing approval does not
block diagnostics. It keeps accepted closure open.
The selection belongs to that invocation; `checkpoint` does not overwrite a
shared `active.json`. Prefer explicit checkpoints when several tasks use the
store. `active` is a convenience for a single operator using `run`.

`run <gate>` explicitly reruns one gate. Set an active criterion first when
none is active. Prerequisites must already have matching results; use
`checkpoint` to run them in order. To request a graph export directly:

```sh
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence \
  active piccs-assignment --evidence 'Inspect the PiCCS closure dependencies.'
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence run declaration-metadata
```

Capture includes the complete declared dependency set for the selected gate or
checkpoint, including transitive prerequisites and mandatory identity inputs.
A Lean-only checkpoint does not walk the Rust source group. The retained bytes
live under `snapshots/<id>`. Execution uses separate copies. Result records and
logs remain under `runs/<run-id>`, outside the source identity. No command changes
Git state or existing file permissions.

Relative file symlinks are retained only when their targets are also captured.
Absolute, escaping, cyclic, and directory symlinks are rejected.

Use `--snapshot <id>` to inspect or check a retained candidate. Without this
option, `status` and `stale` inspect current source. An edit to current source
does not change a retained snapshot or erase its evidence.
Changes to the checker, policy, runtime, or required inputs can still make an
earlier result inapplicable, even when its source snapshot is unchanged.

```sh
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence \
  --snapshot SNAPSHOT_ID status
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence stale
```

Different dependency sets have different snapshot identifiers. A retained
Lean-only snapshot cannot supply an omitted Rust or conformance input.
`status` covers the full registered map; gates outside a selected snapshot can
show `not-captured`. Use `explain <obligation>` for a focused report. Reviews
must match the exact selected snapshot and their registered scope.

## Commands and reports

Place global options such as `--store`, `--inputs`, `--snapshot`, and `--json`
before the command. The command names below follow
`python3 scripts/lean_graph/evidence.py --store /path/to/evidence`.

| Command | Result |
|---|---|
| `checkpoint <obligation>` | Select the criterion, run prerequisites in order, and resume matching results. |
| `explain <obligation>` | Show the exact target, argument, checks, reviews, evidence links, and next command. |
| `review-request <obligation> --proposal <file>` | Capture the target's source and prepare a decomposition-review request. |
| `record-review <request-id> <response-file>` | Import a separate reviewer's decision; accepted mode requires an already authenticated envelope. |
| `run <gate>` | Rerun one registered gate; matching prerequisites must already exist. |
| `status` | Report execution, freshness, reviews, and phase closure for the registered map. |
| `stale` | List changed dependencies and rejected records for the selection. |
| `active <obligation> --evidence <check>` | Record the current criterion and intended closing check. |
| `requires <declaration>` | Show its type and direct meaning/proof dependencies. |
| `used-by <declaration>` | Show direct consumers and linked evidence in the exported graph. |
| `path <helper> <parent>` | Follow dependency edges from the helper toward the named parent. |

`active` grants no closure credit. `run` and `checkpoint` return zero when their
requested execution succeeds in the selected mode. Accepted closure can still
be open. `status`, `stale`, and `explain` return zero when they produce a report;
their exit code is not an acceptance decision. Failed checks and missing
execution prerequisites produce a nonzero exit from `run` or `checkpoint`.

Status reports separate `execution`, `freshness`, `checker`, and prerequisites.
An execution can be `passed`, `current`, and `diagnostic` while accepted closure
remains open. Required reviews appear separately under each obligation.
`explain <obligation>` shows the target, mathematical argument, missing inputs,
required checks, evidence links, and the exact next command. It does not propose
a command to replace a missing target registration or independent review.

With `--json`, `status` and `explain` return the report directly. `checkpoint`
returns `checkpoint` execution details and a separate `status` report. `run`
returns its execution record, including `outcome`, commands, and artifact
references. In a status report, each `gates` entry separates:

| Field | Meaning |
|---|---|
| `execution` | Result of the displayed run, such as `passed`, `failed`, or `not-run`. |
| `freshness` | `current`, `stale`, `missing`, or `not-captured` for the selected dependencies. |
| `freshness_basis` | `sources` for source groups or `declarations` for registered declaration keys. |
| `checker` | `approved`, `diagnostic`, or `none`; this identifies result provenance. |
| `prerequisites`, `missing_inputs` | Checks or captured dependencies still required. |
| `accepted` | Whether this gate and its prerequisites have applicable approved-checker results. |
| `record` | Path to the displayed execution record, if one exists. |
| `elapsed_seconds`, `timings_seconds` | Gate total and its measured preparation, command, cache, metadata, and verification components. |

Gate acceptance does not replace obligation reviews. Use `obligations[].closed`
and `phase_statuses` for closure decisions. A later failed attempt does not erase
an earlier matching successful result; the report can display that earlier run.

The initial obligation map deliberately keeps the full Stage 1 assignment
target, complete compiler theorem set, complete branch coverage, and
production binding open. Existing conditional theorems remain useful facts.
The new pilot/PiCCS registrations do not discharge these other obligations.
The report separates PiCCS from Stage 1. A full Stage 1 gap does not erase a
closed PiCCS criterion. Each phase still needs its complete approved gate set.

## Decomposition review

The assignment criteria require a separate decomposition review. The CLI
prepares a request and checks the submitted record. The reviewer supplies the
mathematical judgment; the CLI does not launch an agent or judge an argument.
The review is per meaningful obligation, not per tactic edit or helper count.

Write a proposal JSON file outside the source with these fields:

```json
{
  "author": "implementation task identifier",
  "statement": "LeanGraph.Targets.PiCCSPublicAssignment",
  "premises": [
    "The application has its required fit proof, with the stated Ajtai key and proof template.",
    "The digest has four words, the actual public projection equals encHash of that digest, and the selected rows hold."
  ],
  "argument": "Derive the one cell from the public marker. Apply the decoded PiCCS phase theorem, then compose the running-input equality and the two hash observations.",
  "parent_use": "Supply phase and hash observations to the final step proof. The selected context and complete typed NIFS input/output connections remain open.",
  "dependencies": [
    "NightstreamFPrime.Export.Stage1.ActualPiCCSInputs.selectedRowsAndPublic_imply_phaseAndHashes"
  ]
}
```

Use the exact statement or its registered declaration, all permitted premises,
the mathematical argument, named dependencies, and the intended parent use.
An empty premises list is valid for an unconditional claim. A missing list is
not an assertion that there are no premises.

```sh
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence \
  review-request piccs-public-assignment --proposal /candidate/proposal.json
```

The response includes the request identifier, captured snapshot, target, owner
criterion, proposal, and a `response_template`. Give that material and the
captured source to an independent reviewer. The reviewer must inspect the
actual Lean statement and fill every assessment with `pass` or `fail` plus a
reason: substantiveness, premises, argument, correspondence, and parent use.
They must supply their name and an ISO `reviewed_at` timestamp with an offset.
The aggregate outcome is `pass` only when all assessments pass.

```sh
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence \
  record-review REQUEST_ID /reviewer/response.json
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence \
  explain piccs-public-assignment
```

Local imports are diagnostic. A different author/reviewer name is required,
but local names do not authenticate reviewer independence. Accepted imports
require `--authority` and an envelope already signed by the controlled review
process. The importer never signs an untrusted response as an approval.
It preserves all other required target, formula, proof, and production reviews.

Requests are bound to their exact target, proposal, snapshot, policy, and
checker. Changed content cannot reuse the request identifier. A changed source
cut or target makes the earlier review stale; it remains evidence for the old
cut. A later rejection for the same target and cut revokes an earlier approval.
Equal-time conflicting decisions reject. Missing, failed, or stale decomposition
reviews stop accepted `run` and `checkpoint` execution for that criterion.
Local diagnostic checks remain available and cannot grant accepted closure.

`PiCCSPublicAssignment` derives the public one cell rather than assuming it.
Its conclusions are the phase predicate, prior-running agreement, fresh public
hash, and claimed output hash. It is still an intermediate target: the complete
Stage 1 decoded-step target and selected-context connection remain open.

## Ordinary commands and the shared lock

Use the same guard for development builds and fixture generators that run
outside a registered checkpoint:

```sh
python3 scripts/lean_graph/guard.py --kind lean --cwd formal/nightstream-fprime \
  -- bash scripts/validate.sh build tests.EvidenceTargets
python3 scripts/lean_graph/guard.py --kind rust -- \
  cargo test --release -p nightstream-fprime --test package_loader
```

Registered gates already acquire this lock; do not put the guard around an
evidence CLI checkpoint. The guard uses one host path, independent of `TMPDIR`
or the selected evidence store. It refuses a held lock and reports its owner
PID. It also checks for unmanaged Lean/Lake/Cargo/rustc processes before launch.
Every ordinary fixture command must participate; a lock cannot exclude an
unmanaged native executable that ignores it. No command stops another task.

The guard enforces the existing 1,500-second Lean and 300-second other-command
caps, Lean's `validate.sh` entrypoint, Rust release mode, and empty
`RUSTC_WRAPPER`. It terminates only its own process group on timeout or
interruption. Its output is a development diagnostic, not a registered gate.

## Timing and stale-evidence detail

Each new gate record includes the total `elapsed_seconds` and component
`timings_seconds`. These separate preparation, command execution, cache
restoration/capture/publication, metadata processing, input validation, and
snapshot verification. `other` accounts for remaining gate overhead. The
command component includes the command's completion checks. The CLI JSON also
reports snapshot capture/inspection and total invocation time separately.
Older records without these fields do not acquire inferred component timings.

A cached build time is not the full gate time, and a cold-to-cached comparison
is not a speedup over ordinary incremental Lean development. Compare identical
work and report both command and total times before attributing an improvement.

`stale` and `status` identify changed, added, and removed files from the retained
manifests. When a current complete declaration export exists, they also name
changed registered meaning/proof roots. Missing snapshot data is marked
unavailable. Without a current complete graph, the report uses file/group
changes. Neither case weakens freshness checks or mandatory identity reruns.

## Conformance inputs

Pass `--inputs /absolute/path/inputs.json`. Its keys are the input names in
`obligations.json`; values are absolute file or directory paths:

```json
{
  "package": "/candidate/package.json",
  "binding": "/candidate/binding.json",
  "setup": "/candidate/setup.json",
  "identity": "/candidate/expected-structural-identity.json",
  "expanded": "/candidate/expanded.json",
  "base_fixture": "/candidate/base-fixture.json",
  "phase_input": "/candidate/phase-input.json",
  "lean_result": "/candidate/lean-result.json",
  "opening_cache": "/candidate/opening-cache"
}
```

The identity file contains the four separately selected structural identity
words as a JSON array. The CLI never copies a production identity from a
candidate package. It captures all bytes in an input directory, including
the opening carrier and its metadata. Supply a directory with only the
required inputs. Missing inputs keep their gates open.

Inspect missing inputs before starting a conformance checkpoint. Replace the
input-manifest path below with the file that lists your selected inputs:

```sh
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence \
  --inputs /candidate/inputs.json explain piccs-conformance
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence \
  --inputs /candidate/inputs.json checkpoint piccs-conformance
```

Registered gates call the existing package checker, complete PiCCS result
checker, mutation implementations, and independent opening evaluator.
The opening gates cover `K`, `A0` through `A13`, the CCS rows, and commitment
rows `C0` through `C21`, as required by the selected profile. A valid zero
matrix evaluation is retained. No generic nonzero-value check is added.

`lean-input` regenerates the Lean result and compares it with the selected
result bytes. The phase-result and opening gates depend on that check.
Their required test names and completion messages must appear in the logs.
A successful process that selected no required test is a failed gate.

Lean gates also bind `library_seed`: external dependency sources, Git metadata,
and compiled modules. Lake needs that metadata to use the pinned packages.
Local diagnostic mode defaults to the candidate's `.lake/packages`. Accepted
runs require `libraries/` under the protected checker authority and reject a
candidate-selected replacement. Candidate project build products are never
used as a seed.

The seed is captured as immutable input. Lake works on a separate copy. The
runner uses filesystem clones on macOS and streaming copies elsewhere. A
timeout remains a failed gate; cache availability does not waive a check.

## Retained builds

Successful Lean `build` commands can retain the project `.lake/build` products.
The cache binds the complete gate source and input dependencies, toolchain
binaries, command, build options, environment, host thread count, policy, and
checker version. It never imports candidate project build products.

The runner freezes products immediately after the build and publishes them only
after the gate passes its input and completion checks. Each reuse verifies the
cache manifest and product bytes, copies them into an isolated execution tree,
and still runs the registered build command. Changed or unauthenticated products
cause a cold build. A failed or interrupted gate cannot publish a new cache.

Diagnostic caches live in the store's `builds/` directory. Accepted caches live
in the protected authority's `builds/` directory and require authenticated
manifests. A diagnostic cache cannot supply accepted build products. Rust test
results can be resumed by `checkpoint`; Rust build products are not cached by
this version.

## Acceptance authority

Local diagnostic results cannot close acceptance criteria. To use
`--authority /checker/lean-graph`, the operator must first provision a controlled
checker environment. Candidate code must not be able to read its record key,
write its policy or reviews, or change its installed tools. Candidate commands
can execute code; a different directory under the same unrestricted user is
not this boundary. This CLI does not provision a sandbox or a CI service.

The authority directory contains:

- `policy.json`: the independently reviewed obligation map.
- `approval.json`: `outcome`, `reviewer`, `policy`, and `checker`. The latter
  fields identify the approved map and installed checker implementation.
- `record.key`: a secret authentication key owned by the checker service.
- `libraries/`: the trusted external Lean dependency seed, matching the pinned
  manifest and toolchain. Candidate processes must not be able to change it.
- `reviews/*.json`: authenticated review records supplied by the review process.
- `builds/`: authenticated Lean build products managed by the checker.

`policy.digest(policy)` and `policy.checker_key()` expose the identifiers for
operator inspection. The CLI does not create an approval, a key, or a review
judgment. `record-review` imports a decision from the separate review process.
An approved policy cannot be replaced with a candidate `--policy` argument.
The Lean metadata, acceptance driver, boundary script, and axiom checker sources
must also match the installed checker. The proposal's exact targets require
review before acceptance use.

Each authenticated envelope has `record` and `authentication` fields. The
authentication is HMAC-SHA256 over canonical JSON: sorted keys, compact
separators, and one final newline. This is evidence metadata outside every
protocol binding path. It is not a Poseidon2 relation identity or a proof.
The protected key supplies record provenance; matching public hashes alone
do not authorize a result.

A review record contains `review` (the registered review name), `scope`,
`snapshot`, `policy`, `reviewer`, and `outcome`. Scope must match the approved
review definition. The snapshot identifies the complete captured source and
input manifest. A free-text commit reference does not satisfy this check.

The runner enforces the project caps: 1,500 seconds for each Lean command and
300 seconds for each Rust test or other test command. Lean uses `validate.sh`;
Rust uses release mode and an empty `RUSTC_WRAPPER`. Managed commands share a
host lock. Preflight detects active Lean, Lake, Cargo, and rustc processes.
The acceptance environment must exclude other unmanaged builds and test executables.
If preflight finds another build or test, wait for it to finish before retrying.
The CLI does not stop another task's process. Status and graph queries are read-only.

## Dependency metadata and validation

`run declaration-metadata` performs declaration inspection. Its authored Lean
entrypoint is `ExportMetadata.lean`; it runs through `validate.sh file` after a
source build.
The witness graphs include the exact pilot, PiCCS, and public-boundary target
declarations. The latter includes `ActualHashSlots` and `ActualPiCCSInputs`.

Lean exports actual declaration types, definition values, inductive data,
proof terms, dependencies, and resolved module origins. Expression encoding
preserves DAG sharing and removes bound variable names and source metadata.
The Python reader checks module provenance against captured dependency sources and the pinned
toolchain. Unknown provenance produces incomplete metadata.

The retained graph contains declaration fingerprints, dependency edges, rendered
types, and exact encoded type expressions. Target propositions include their
premises. Display text does not determine the semantic fingerprints.
The exact raw metadata log is retained with gzip compression. It is not copied
into the status report or into an ordinary build log.

The map can register `declaration_freshness` for one source group of a gate. It
names the exporter gate, exact roots, and either `meaning` or `proof` keys.
The initial registration covers the assignment-target check with both closure
witnesses and their proof dependencies. Checkpoints obtain a current export
when needed. If the relevant keys match, a source-group edit can preserve that
check's earlier result. Unchanged source can resume without another export.

Missing, stale, incomplete, changed, or unauthenticated export evidence cannot
narrow accepted freshness. Those cases use the complete source group. Matching
graph keys never erase other source, input, checker, runtime, or policy changes.
Every identity-dependent gate still binds the full package, binding, setup, and
selected identity inputs. Conformance gates retain conservative source checks
until their complete declaration dependencies are registered and reviewed.

Use concise queries after a checkpoint or declaration export:

```sh
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence \
  requires LeanGraph.Targets.PiCCSAssignment
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence \
  used-by NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase.selectedRowsZero_implies_phaseHolds
python3 scripts/lean_graph/evidence.py --store /tmp/nightstream-evidence \
  path NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase.selectedRowsZero_implies_phaseHolds \
  LeanGraph.Targets.piCCSAssignment
```

`requires` shows direct meaning and proof dependencies. `used-by` shows direct
consumers. `path` follows dependencies from a helper toward the named parent.
Queries show the selected declaration's type and linked evidence. They cover
the exported roots and stop at the pinned external-library boundary. An absent
path is not a mathematical rejection. Composition still requires its theorem.
Lean's rendered types can abbreviate proof terms with `⋯`. The declaration's
exact encoded type is available with `--json`; the source link and retained raw
export also preserve the target definition behind a proposition alias.

If a query reports no current complete export, run `declaration-metadata` for
the intended source or select a matching retained snapshot. A checkpoint that
resumes an existing target result need not generate another graph.

## Maintaining registrations

The [obligation map](obligations.json) is a reviewed expectation, not generated
proof progress. Update its draft as the proof develops:

1. Keep each owner criterion's intended meaning and permitted premises explicit.
   Record its target, mathematical argument, required gates, input dependencies,
   coverage cases, and reviews. Preserve mandatory identity-dependent checks.
2. Register propositions and closure witnesses in
   [EvidenceTargets.lean](../../formal/nightstream-fprime/tests/EvidenceTargets.lean).
   Connect their exact checks in
   [EvidenceAcceptance.lean](../../formal/nightstream-fprime/tests/EvidenceAcceptance.lean)
   and the gate's completion rules. Retain the axiom audits.
3. Select graph roots in [ExportMetadata.lean](ExportMetadata.lean). Add
   `declaration_freshness` only when the named roots cover the gate's required
   declaration dependencies. An unregistered theorem remains outside this workflow.
4. Run the tool tests and focused Lean checks below, then use `explain` and
   `checkpoint` for the affected criterion. Accepted use still requires review
   of the targets and protected policy; local edits do not change that policy.

Another Lean package needs project-specific source paths, gates, target
registrations, and module-origin validation. `--policy` alone does not port
the current Nightstream F′ configuration to a separate SuperNeo Lean project.

## Validation

```sh
timeout --signal=KILL 300 python3 -m unittest discover -s scripts/lean_graph/tests -v
cd formal/nightstream-fprime
timeout --signal=KILL 1500 bash scripts/validate.sh build tests.EvidenceContractChecks
timeout --signal=KILL 1500 bash scripts/validate.sh build tests.EvidenceTargets
timeout --signal=KILL 1500 bash scripts/validate.sh file tests/EvidenceAcceptance.lean
```

The Python process fixtures test evidence handling, including zero values and
invalid input. They are small arithmetic harness cases, not Nightstream
opening proofs. The Lean regression checks reject a conditional witness for
a stronger target and accept a witness that discharges the extra premise.
Actual protocol evidence comes from the registered existing conformance tools.
