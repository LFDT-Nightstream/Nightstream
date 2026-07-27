# CIR-OBLIGATION-TREE

```text
property_id: CIR-OBLIGATION-TREE
evidence_state: artifact-checked
milestone_status: M4 historical-artifact complete
shipping_status: M4 current-shipping open

claim:
  The generated manifest for the captured 4,193,134-row full-history F-prime
  artifact expands into one root and eight nested generated partitions. Its
  deepest ownership schedule has 61 leaves: 59 materialized half-open row
  ranges and two zero-cost organizational leaves.

  The 59 materialized ranges are duplicate-free, contiguous, and cover
  exactly [0, 4,193,134). Consequently every physical row has exactly one
  deepest owner, no owner contains a row outside the program, and the sum of
  the materialized leaf costs is exactly the generated total. Every parent
  interval is exactly covered by its immediate children and its row count is
  the exact sum of their row counts.

  Every leaf carries a typed route to its protocol obligation, Rust emitter
  module, exact generated stage path, and checked Lean artifact
  soundness/completeness evidence. Zero-cost leaves additionally carry their
  checked necessity/census theorem. Formula-only estimates have a separate
  type that cannot own a physical range.

scope:
  - Exact generated manifest schema 2.
  - Artifact SHA-256
    cccfb8ab2eff0583b8e8469ebd85047ff0b53ecdb22c577296e7ce2964a481ca.
  - 4,193,134 rows, 3,582,173 columns.
  - Plain/stateless [1,1] full-history execution with one recursive
    invocation, terminal fold, direct terminal CE, and the captured
    minimal-supported-bit carrier.

authority:
  - Generated manifest data is authoritative only for physical row intervals,
    counts, stage labels, and hashes.
  - CIR-SOUND and CIR-COMPLETE own the meaning of satisfying those exact rows.
  - Paper semantics remain independently owned by the frozen SuperNeo and
    HyperNova specifications. Neither stage labels nor the artifact define
    protocol validity.

assumptions:
  - Exactly the assumptions and named event branches already published by
    CIR-SOUND and CIR-COMPLETE for this artifact.
  - The artifact importer remains the existing trusted translation boundary.
  - Existing native-decision certificates expose Lean.trustCompiler in the
    artifact-level axiom reports; the obligation-tree source adds no new
    native decision.

non_goals:
  - Claiming equality with the current 270-row terminal-link owner. The
    captured terminal link has 257 rows, and the tree retains the existing
    kernel-checked inequality as evidence.
  - Claiming current Rust emission equals this stale snapshot. The current
    Rust manifest-regeneration test fails closed at
    `FPrimeFullHistoryProjectionRoles.lean` before reaching later comparisons.
  - Claiming the snapshot is the selected canonical typed IR program.
  - Stateful, Nebula, other schedules, multiple recursive invocations,
    alternate carriers, or parameterized circuit-family ownership.
  - Concrete Poseidon2, Goldilocks, Ajtai, probability, extraction, or M6
    security reduction.
  - Global minimality over arithmetizations.

failure_class:
  A physical gap, overlap, duplicate interval, reversed interval, row outside
  the root, parent/child cost mismatch, formula estimate mislabeled as emitted
  rows, zero-cost node assigned rows, missing cross-layer route, or stale
  terminal owner promoted as current production.

counterexample_or_witness:
  Focused regressions reject explicit gap, overlap, duplicate, and reversed
  small partitions. A zero-width interval is proved to own no row. The
  existing terminal-link drift theorem proves the captured 257-row list differs
  from the current 270-row owner. The bounded current-Rust manifest regression
  independently reports drift at the projection-role artifact; its generated
  `.expected` file is diagnostic only and is not retained.

lean_theorems:
  - FPrimeFullHistoryObligationTree.every_parent_cost_exact
  - FPrimeFullHistoryObligationTree.exact_hierarchy_shape
  - FPrimeFullHistoryObligationTree.materialized_leaf_ranges_cover
  - FPrimeFullHistoryObligationTree.materialized_leaf_cost
  - FPrimeFullHistoryObligationTree.zero_cost_nodes_exact
  - FPrimeFullHistoryObligationTree.exact_leaf_census
  - FPrimeFullHistoryObligationTree.materialized_leaf_ranges_nodup
  - FPrimeFullHistoryObligationTree.every_materialized_row_has_exactly_one_leaf
  - FPrimeFullHistoryObligationTree.no_row_outside_materialized_leaves
  - FPrimeFullHistoryObligationTree.every_lean_evidence_checked
  - FPrimeFullHistoryObligationTree.every_leaf_cross_layer_mapped
  - FPrimeFullHistoryObligationTree.obligation_tree_retains_terminal_drift

axiom_report:
  Exact parent costs use [propext]. Exact row ownership uses at most [propext,
  Classical.choice, Quot.sound]. Cross-layer evidence inherits [propext,
  Classical.choice, Lean.trustCompiler, Quot.sound] from the existing
  artifact-checked CIR-SOUND/CIR-COMPLETE path. Every headline theorem has a
  fail-closed #audit_axioms guard.

retest_commands:
  - cd formal/nightstream-lean &&
    LEAN_TIMEOUT_SECONDS=900
    LEAN_BUILD_TARGET=tests.Axioms.FPrimeFullHistoryObligationTree
    ./scripts/validate.sh build
  - cd formal/nightstream-lean &&
    LEAN_TIMEOUT_SECONDS=900
    LEAN_BUILD_TARGET=Nightstream.Assurance
    ./scripts/validate.sh build
  - cd formal/nightstream-lean &&
    LEAN_TIMEOUT_SECONDS=900 ./scripts/validate.sh axioms
  - cd formal/nightstream-lean &&
    LEAN_TIMEOUT_SECONDS=900 ./scripts/validate.sh check
```
