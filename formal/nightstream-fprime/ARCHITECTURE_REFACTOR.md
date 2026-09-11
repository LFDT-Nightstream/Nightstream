# Physical compiler architecture goal

Requested on 2026-09-10. Branch: `nico/lean-architecture`.
Starting Lean source: `0b2710b3e0138900cfc1071ef14becd37e430a6d`.
Upstream requirements changes are merged at `8c5d11ca`.

The app goal tool rejected this new goal because the earlier NIFS goal is
unfinished. This file records the requested architecture work; it does not
mark the NIFS goal complete or approve its proposed cryptographic model.

## Contract

Give physical compilation one owner in Layout. Export consumes proved
package interfaces and owns encoding, emission and parity. Preserve the
logical circuit framework, SuperNeo v1.1 semantics, theorem assumptions,
executable operations and the production profile: Goldilocks, `b = 2`,
`k_rho = 16`, 17 ordered sources, 16 children, 14 matrices, 28 rounds and
Poseidon2 binding.

The result must reduce dependence on allocation internals. Directory moves,
shorter tactics and source-line totals do not establish this result.

Required results:

- Derive phase sizes, prefix starts and source ranges from their owners.
- Expose phase inputs, outputs, footprint, support and semantic theorems.
  Consumers must use those interfaces instead of unfolding allocation.
- Keep one authoritative definition for each necessary logical, physical
  and serialized representation. Derive additional views by explicit
  conversions with preservation proofs. Remove superseded descriptions
  after their consumers use the authoritative conversion.
- Place allocation, row order, compact plans, matrix programs and physical
  preservation under Layout. Enforce the resulting ownership boundaries.
- Keep the final soundness statements and their remaining assumptions easy
  to inspect. Record the gates for later constraint changes.

The [assurance surface](ASSURANCE_SURFACE.md) names the current final statements
and separates their completeness scope from their remaining premises.

## Work order and evidence

1. Establish the current dependency baseline and identity gate. Reuse the
   canonical binding emitter and current pins. Record import reachability
   separately from measured rebuilds. Run an allocation-change baseline.
2. Pilot the boundary on PiDEC. Add source-range and ordering theorems to
   the existing Layout owner. Remove allocation expansion from the Export
   substitution proof. Retain the actual NIFS matrix-entry consumer. Record
   build time, memory, edited proofs and unchanged identities.
3. Extend the successful interface and derived-geometry pattern to the
   other phase owners. Use measured dependencies to select each next change.
4. Consolidate physical compilation and remove redundant representations.
   Update imports, audits, requirement citations and graph obligations in
   the same migration checkpoint. Prepare concrete protected-document
   changes before any approval-dependent architecture migration.
5. Complete the change playbook and test the allocation boundary again.

For the allocation test, an added gadget witness column may require edits
to that gadget's own proofs, pinned values, parent bounds that name its size
and package counts. Unrelated phase proofs must remain valid through their
interfaces. The test records the actual dependent failures; it does not
require all parents to remain textually unchanged.

A refactor marked **bytes unchanged** must preserve structural and package
identity. A later constraint optimization must prove that every satisfying
assignment for its new rows satisfies the unchanged specification, without
stronger assumptions, and that every valid specification instance still has
a witness. A cvc5 result alone cannot discharge either obligation.

## Supporting experiments

Shared simplification lemmas, module-system hiding, production algebra
instances, directory changes and resource-override removal are supporting
work when a measured maintenance problem requires them. Pilot each repeated
proof pattern before a rollout. Do not assume that private proof bodies
prevent rebuilds. A production ring instance must not conflict with the
pointwise instance on coefficient functions; use a proved wrapper if needed.

No proof-size percentage, file-count reduction or rebuild target is assumed.
The baseline numbers in the supplied reviews must be measured again.

## Validation and limits

Use one Lean or Rust build queue. Run Lean through `scripts/validate.sh`
under the project's 1,500-second command cap. Non-Lean tests have the
300-second cap. Stop a declaration that exceeds the user's 60-second
elaboration limit; do not add resource overrides or evaluate artifact-sized
data in proofs. Preserve explicit axiom audits and DCO sign-offs.

Run static, build and axioms in order before checkpoints, plus identity and
the relevant conformance checks for the changed claim. Do not repeat closed
checks unless code changes or new evidence requires them.

Keep the frozen corpus and the user's original dirty checkout unchanged.
This goal does not change transcript semantics, authorize a new backend,
approve the pending Fiat-Shamir model, prove full history extraction, or
publish the requirements site.

## PiDEC checkpoints

The source-range pilot is recorded in [pidec-pilot.json](architecture/pidec-pilot.json).
The next checkpoint derives the PiDEC sizes and starts from their child owners;
[pidec-derived-sizes.json](architecture/pidec-derived-sizes.json) records its checks.
The actual NIFS matrix-entry consumer, library, axiom audit and canonical identity
gate pass. The allocation test remains a separate claim. Source import reachability
has not decreased.

## Shared matrix interpreter

`Layout/MatrixProgram` owns the existing operands, block interpreters, source
projection, schedule laws and declared-work functions. `Layout/R1CS/ColumnMap`
owns the column-map implementation; the existing compact-row names are aliases.
The corresponding `Export/MatrixProgram` modules contain only `Format` codecs.
The static boundary gate enforces that codec-only scope.

The [rename map](architecture/matrix-owner-renames.json) records this cut. It does
not mean that all Stage 1 physical geometry has moved. The
[constraint change checks](CONSTRAINT_CHANGE_CHECKS.md) give the required proof,
identity and consumer checks for later changes.

The extra-cell probe still reaches a fixed running-transition endpoint through
Spartan, plus the selected default Values check. Full allocation isolation remains
open; see [the probe record](architecture/pidec-allocation.json).

## Spartan map boundary

`Spartan` now owns the generic column map and padding operations. `SpartanRows`
owns their complete-prefix instance, with the existing public names and bodies.
PiDEC source support no longer imports `RunningTransitionLowering`. The
[boundary record](architecture/spartan-map-boundary.json) contains the checks and
source dependency measurement. The changed-allocation probe is a separate check.

The repeated probe passes PiDEC source ranges after this split. It still stops
at global value checks and the compact parent offset; the probe record names
those failures.

The [module experiment](architecture/module-hiding-pilot.json) confirms that
a converted gadget consumer can keep its build after a producer proof edit.
Legacy Layout consumers still rebuild. No production header conversion was made.
