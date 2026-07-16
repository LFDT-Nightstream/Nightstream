# CIR-FPR-SOURCE-CENSUS

```text
property_id: CIR-FPR-SOURCE-CENSUS
claim:
  A separately generated fixed-F-prime base or recursive source census is an
  exact partition of the source-column universe into owner-path-preserving
  SourceSegments. Every segment carries one exhaustive SlotRole, and every
  declared per-role count equals the corresponding source-run subtotal.

  Consequently every in-range source column has one unique role, eligible
  ordinary fields and explicitly excluded fields form an exact count
  partition, and all declared role counts sum to sourceColumnCount without a
  second artifact-supplied total.

  For the prospective architecture that assigns one disjoint 41-coordinate
  word to every eligible ordinary field, eligibleCount * 41 is its conditional
  capacity floor. Any smaller budget fails that capacity requirement.

  The generic packed consumer parses generator-bounded comma-decimal chunks
  without a global flattened token list; the Lean parser itself does not
  enforce a chunk-size resource bound. It derives every source start from one running
  cursor and rejects malformed input, invalid metadata, zero-length or
  non-maximal adjacent runs, mismatched final declarations, a non-unique
  constant-one count, and an initial constant-one whose physical owner is not
  exactly `fprime.assignment.constant_one`. One proved scan invariant supplies
  both the exact partition and exact role census to SourceCensusArtifact.

  The committed fixed-F-prime artifacts pass that check for 7,124 base runs
  over 22,353 source columns and 282,733 recursive runs over 2,399,107 source
  columns. Their ordinary-private eligible counts are exactly 3,050 and
  154,747. Therefore separate branch-private candidates that reserve one
  disjoint 41-coordinate word per eligible field have conditional floors of
  125,050 and 6,344,627 coordinates, or 6,469,677 in total when the branches
  are separate. Under the recursive capacity premise alone, a recursive
  budget at most 1,000,000 is impossible. Selector sharing can remove the
  base/recursive additive overhead, but not the recursive subtotal.

  A production observer also replays the complete source-loop allocation
  cursor without changing allocation or constraint emission. Lean derives
  every ordinary-private word start from the checked source-role runs and the
  fixed role-width ABI; the generated placement artifact contains only the
  source-phase end and final encoded-column bound, never a supplied start per
  field. The exact base source phase is `[257, 125695)`, with ordinary
  endpoints `(source 1, start 257)` and `(source 22336, start 125654)`. The
  recursive source phase is `[257, 7830083)`, with ordinary endpoints
  `(source 1, start 257)` and `(source 2399090, start 7830042)`, inside final
  encoded width 8,137,378.
assumptions:
  - Format version 1 and the fixed eleven-role Rust/Lean ABI.
  - The source R1CS and its validated production trace are the fixed
    implementation inputs to this census. They are not semantic authority for
    deciding which F'/NIFS obligations are sufficient or necessary.
  - The supported profile is the base and one-fresh-recursive fixture generated
    by the named full-relation test.
  - Only ordinaryPrivateField is eligible; every other SlotRole is explicitly
    excluded by the fail-closed role classification.
  - Packed runs use the fixed eleven-role wire order and mixed-radix formula
    `(length * 11 + roleIndex) * stageCount + stageIndex`.
  - The fixed-F-prime source ABI has exactly one constant-one column, first in
    source order, with physical owner `fprime.assignment.constant_one`.
  - Rust-conformance requires the production source-trace generator and exact
    byte drift gate to pass against this committed generated module.
  - Source-loop allocation widths are fixed by the production ABI: 1 for a
    private Boolean, 41 for an ordinary-private or SIS-opening source, 95 for
    a canonical-u64 source, and 0 for every other source role. Public bits are
    allocated once in the prefix rather than again in the source loop.
non_goals:
  - A claim that a successful packed check alone establishes Rust conformance.
  - A complete deferred-allocation, encoded-coordinate, or CE-coordinate
    partition; the observer stops at the end of the production source loop.
  - A proof that an accepted 41-coordinate word equals one chosen centered
    encoding, or an efficiently invertible NIVC encoding.
  - A selector-composed fixed assignment.
  - A claim that base and recursive branches cannot share selector-inactive
    slots in one physical coordinate arena.
  - A global lower bound over alternative encodings, algebraic gates, rows, or
    proof-backed authority boundaries.
paper_sources:
  - HyperNova Construction 2, step `enc(F')`, supplies the encoding context.
  - The role ABI, source census, and prospective per-field-41 placement are
    implementation-specific and are not paper claims.
lean_theorems:
  - Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.sourceColumn_hasUniqueRole
  - Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.sourceColumn_hasExactEligibilityClass
  - Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.eligibleCount_eq_ordinaryRunSubtotal
  - Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.declaredRoleTotal_eq_sourceColumnCount
  - Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.sourceColumnCount_eq_eligibleCount_add_excludedCount
  - Nightstream.Implementation.R1CS.FPrimeFieldLayout.SourceCensusArtifact.budget_below_perField41_is_no_go
  - Nightstream.Implementation.R1CS.FPrimeFieldLayout.PackedSourceCensus.Data.check_sound
  - Nightstream.Implementation.R1CS.FPrimeFieldLayout.PackedSourceCensus.Data.toSourceCensusArtifact
  - Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.base_data_check
  - Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_data_check
  - Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.base_eligible_count
  - Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_eligible_count
  - Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.base_ordinaryRunSubtotal_count
  - Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_ordinaryRunSubtotal_count
  - Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.base_perField41_width_floor
  - Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_perField41_width_floor
  - Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.combined_perField41_width_floor
  - Nightstream.Implementation.R1CS.FPrimeRecursiveSourceRoleCensus.recursive_one_million_perField41_budget_is_no_go
  - Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement.segmentPlacementStart_some_iff
  - Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement.sameSegment_wordRun_before
  - Nightstream.Implementation.R1CS.FPrimeFieldLayout.OrdinaryPlacement.Metadata.check_sound
  - Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.base_data_check
  - Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.recursive_data_check
  - Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.base_sourcePhaseEnd
  - Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.recursive_sourcePhaseEnd
  - Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.base_firstPlacement
  - Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.base_lastPlacement
  - Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.recursive_firstPlacement
  - Nightstream.Implementation.R1CS.FPrimeRecursiveOrdinaryPlacement.recursive_lastPlacement
artifact:
  Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/Generated/
    FPrimeBranchSourceRoleManifestData.lean
    FPrimeBranchOrdinaryPlacementData.lean
circuit_or_encoding_artifacts:
  - The generated module contains format version 1, the eleven roles, ordered
    physical stage paths, and positive ordered runs for the exact base and
    recursive source snapshots.
  - The compact placement module adds only format version, source-phase end,
    and final encoded-column count for each branch. It contains no per-field
    placement list.
  - These are source-loop placement facts, not a complete encoded/CE layout;
    they grant no constraint, row-removal, CE, or lifecycle authority.
failure_class:
  A source column is missing, duplicated, assigned the wrong role or physical
  owner, or represented by malformed/stale packed data while aggregate totals
  still appear plausible; or the source-loop cursor, role width, phase end, or
  final encoded bound drifts from the generated summary.
counterexample_or_witness:
  - Aggregate-preserving equal-length role swap, rejected by pointwise replay.
  - Overlap, zero-length or dropped run, duplicate constant one, changed role
    total, changed canonical-u64 census, malformed version/chunk/owner path.
  - Changed placement source/start, dropped or duplicated placement, changed
    public prefix/phase end/final bound, and changed source role all fail the
    Rust placement audit.
  - A direct canonical-u64 source consumes 95 source-loop coordinates although
    its decoder-visible returned range contains 64 bits.
  - Honest witness: the committed exact base and recursive packed runs.
drift_gate:
  cargo test -p neo-fold-clean --release --test f_prime_full_relation -- --nocapture
rust_surfaces:
  - frontends/f_prime/gadget_native.rs
  - frontends/f_prime/gadget_native/source_schedule.rs
  - frontends/f_prime/gadget_native/source_manifest.rs
  - frontends/f_prime/gadget_native/source_allocation.rs
  - frontends/f_prime/gadget_native/coordinate_gates.rs
  - tests/f_prime/full_relation.rs
  - tests/f_prime/full_relation/source_role_manifest.rs
  - tests/gadgets/canonical_u64_trace.rs
axiom_report:
  - Generic schema: `[propext, Quot.sound]`; ordinary-run subtotal uses only
    `[propext]`.
  - Packed decoder: `[propext, Classical.choice, Quot.sound]`.
  - Generic placement lookup uses `[propext]`; ordered word geometry uses
    `[propext, Quot.sound]`; metadata soundness uses no axioms.
  - Concrete native-decide certificates and dependent floors:
    `[propext, Classical.choice, Lean.trustCompiler, Quot.sound]`.
proof_hash:
  Pending final gates. Record individual SHA-256 hashes for CenteredTernary,
  LayoutWidthFloor, SourceCensus, PackedSourceCensus, the concrete census,
  source-census tests, both axiom guards, the generated artifact, and each
  named Rust surface. The rendered module is the artifact; there is no
  separately serialized payload.
conformance_status:
  Artifact-checked and pending live Rust replay. Promote to rust-conformant only
  after the drift test reproduces the committed module byte-for-byte, the Lean
  gates pass, hashes are recorded, and an append-only evidence-ledger entry is
  added.
evidence_state: artifact-checked; rust-conformance pending final replay and ledger
retest_commands:
  - `/usr/bin/perl -e '$t=shift; alarm $t; exec @ARGV' 300 cargo test -p neo-fold-clean --release --test f_prime_full_relation -- --nocapture`
  - `cd formal/nightstream-lean && /usr/bin/perl -e '$t=shift; alarm $t; exec @ARGV' 900 lake env lean tests/FPrimeSourceCensus.lean`
  - `cd formal/nightstream-lean && ./scripts/validate.sh all`
remaining_bridges:
  - Prove how every accepted 41-coordinate ordinary word maps to the chosen
    centered word used by an exact, efficiently invertible NIVC encoding.
  - Separately prove the complete deferred/encoded/CE partitions, compiler
    binding, commitment authority, row semantics, and selector-composed
    noninterference obligations.
```
