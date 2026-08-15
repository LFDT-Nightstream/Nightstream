# Recursive constraint minimizer

This subproject uses cvc5 to search for redundant constraints in bounded slices
of the Nightstream recursive verifier. Lean remains the proof authority. Rust
remains the source of the emitted relation.

The tool does not edit a circuit. It does not mark a row as safe to remove. It
records a finite-field query and the raw cvc5 result so that later Rust and Lean
checks can accept or reject the candidate.

## Status

Implemented:

- a versioned JSON format for a normalized Goldilocks R1CS slice;
- exact export of compact CSC, seeded Phi81, and geometric Rust matrix terms;
- an independent evaluation check that compares every exported Nebula source
  row with the three Rust sparse matrices, for full and sampled exports;
- complete physical-stage family ownership for the fixed base,
  bootstrap-recursive, and recursive arms;
- a 20-entry SuperNeo and HyperNova obligation map whose row-family names are
  checked against the exact emitted Nebula census;
- an exact read-only audit of the two physical `NebulaFPrimeRelation` arms;
  bootstrap and steady recursion share the recursive arm;
- exact source-to-final selective row mapping, including full rewrite closure;
- exact projection of the final thirteen-port row terms for exported slices,
  including compact seeded Phi81 metadata without dense expansion;
- an exclusive eight-family ledger for the direct terminal R1CS;
- an exact 18-guard terminal verifier ledger: four context guards, eleven
  statement guards, and three proof-boundary guards;
- Lean artifact checks that retain those 18 guards and keep them disjoint from
  the polynomial family set available to cvc5;
- model-level Lean exactness and inclusion-minimality proofs for all 18 native
  guards, both by boundary and as one combined plan;
- a Rust-shaped Lean program for the actual verification order: context,
  expected public image, statement, Spartan verification, and public statement;
  these proofs do not replace Rust source refinement, expected-value
  refinement, digest security, Spartan, or WHIR soundness;
- exact terminal source rows and their compact column map into the padded
  Spartan relation;
- a shared exact combined terminal test fixture with 58,593 source rows and all
  eight reviewed terminal families;
- manifest-bound combined Spartan lifecycle entry points that consume the
  pre-final HyperNova running and latest pair;
- diagnostic identities for each complete compiler plan and projected slice;
- row and row-family selection;
- active-column SMT queries, which do not declare unrelated source columns;
- exact SMT-LIB `QF_FF` generation;
- bounded cvc5 execution with the `gb` or `split` finite-field solver and an
  independent host wall-clock limit;
- one complete source-identity and family-ownership computation per refinement
  run; later iterations recover only the selected Rust rows;
- a bounded Nebula lifecycle audit that retains the first accepted normalized
  source assignment for each branch used by a segment;
- a versioned saved-assignment format that recomputes the exact Nebula source
  identity and replays every source row before the assignment can enter a
  cvc5 refinement run;
- fail-closed handling of timeout, `unknown`, invalid output, and process
  errors;
- exhaustive generic fixed-point, Nebula F-prime, and terminal drivers that
  record one fail-closed search result for every reviewed polynomial family
  and bind each ledger to its exact source and final relation geometry;
- iterative `sat` refinement with replay through every row of the complete
  sparse Rust relation;
- the same bounded refinement and full-relation replay for the terminal R1CS;
- a scalar polynomial-combination certificate checker;
- Lean definitions and theorems for exact bound artifacts, redundancy
  certificates, and complete removal counterexamples;
- normalized-assignment Lean semantics, where the constant-one column is an
  input-domain condition, with soundness, completeness, exactness, and
  inclusion-minimality assembly theorems for a complete plan;
- complete relation exports for Lean authority on bounded fixed-point fixtures
  and the current terminal fixture;
- Lean emission for exact selective and terminal bound artifacts,
  Rust-rechecked scalar redundancy certificates, and full-relation-replayed
  removal counterexamples for both paths, with normalized companion theorems
  for final-plan assembly;
- JSON evidence that contains the exact query, row map, model or unsat core,
  and source artifact identity;
- local positive and negative control fixtures.

The current reduced Rust relation has one checked redundancy control.
`nifs.pi_rlc.verify.padding.y_ring` has 1,120 rows. All 1,120 rows have a
scalar certificate with coefficient one: 840 use a retained
`nifs.pi_ccs.padded_row.canonicality` row, and 280 use a retained
`nifs.pi_dec.verify` row. This is a source-relation certificate, not permission
to edit the circuit. The production candidate, application, plan, and
parameter profile are not selected. The profile files still set
`production_claim` to false and leave `selected_candidate` empty.

In the current reduced compiler plan, 840 candidate rows map to final retained
rows. The other 280 rows are source-to-empty linear definitions. This is not an
after-removal cost result. A regenerated compiler can assign linear-definition
ownership to a different duplicate row, so the final saving must be measured
after the proved removal is applied.

The bounded projector now carries exact seeded Phi81 metadata for selected
rows. Lean checks its row and column geometry, chunk counts, and seed bytes.
This closes bounded seeded-row slices without dense expansion. It does not yet
prove that the Rust and Lean seeded samplers produce the same coefficients.

The complete recursive bound artifact is not available yet. It has millions
of rows, so a row-by-row Lean list is not a viable authority format. The
production path still needs a run-compressed complete relation artifact and
the Rust-to-Lean compact-matrix refinement. This limit does not affect exact
source export, Rust replay, cvc5 queries, or the checked 1,120-row scalar
certificate.

The current reduced final relation has a conservative fixed-width raw wire
size of 647,108,852 bytes. Its thirteen ports contain 85,271,251 CSC pointers,
19,975,740 explicit CSC coefficients, 1,650,688 geometric-run records, and 36
seeded blocks. Column-pointer compression alone is not sufficient. The exact
complete format must also compress repeated explicit-term and geometric-run
patterns.

A greedy adjacent-column affine-run probe was also too large. It still needed
9,364,416 column records and 18,545,718 entry patterns. The geometric data
still needed 1,428,713 groups. The complete artifact must therefore use the
compiler-owned row and slot grammar. Generic compression of the final CSC
arrays is not a viable Lean authority format.

The existing affine source-decoder format is also too large for the complete
recursive interval. It reduces 4,480,463 source columns to 1,447,684 runs.
This decoder remains useful for bounded ranges, but it is not the complete
artifact format.

Current Nebula tests use a reduced one-step plan. Lean defines the `e1`, `e4`,
`e8`, and `e16` field-native candidate identities, but Rust does not yet
compile all four candidates or produce their required measurement matrix. The
generic bridge still uses a small one-product fixture.

The current Rust WASM frontend separately supports instruction batch sizes 1,
4, 8, and 16. These values resemble the Lean candidate factors, but no Lean
refinement theorem identifies the emitted Rust relations with `e1`, `e4`,
`e8`, or `e16`. This project must not use the same number as proof that the two
relations are equal.

cvc5 1.3.4 is installed as a GPL build with CoCoALib. The local positive
control returns `unsat` with the `gb` solver. The local negative control returns
`sat` with the `split` solver and gives the expected assignment. An `unsat`
result remains search evidence until Lean checks a universal certificate. A
`sat` result remains search evidence until the merged assignment passes
complete Rust replay and Lean checks the complete retained relation.

The exact reduced Nebula base control also runs against cvc5. Its first query
for `fprime.base.step.initial` returns `sat`. Complete Rust replay finds a
violated retained row, so the bounded one-iteration control is `Inconclusive`
and keeps the family.

The real recursive query for `nifs.pi_rlc.verify.padding.y_ring` returns
`unsat` in the installed cvc5 control. This agrees with the independent scalar
certificate for all 1,120 candidate rows. The cvc5 result does not authorize
removal.

The exact combined test-manifest terminal control runs on the smallest family,
`terminal.fresh.selected_relation`. cvc5 returns `sat`. Complete Rust replay
finds retained row 56,700, so the one-iteration control is `Inconclusive` and
keeps the family.

## Repository basis

This design follows three existing boundaries:

- HyperNova
  [Definition 12](../../docs/hypernova-paper/13_6_2_NIVC_Compatible_multi_folding_schemes.md)
  requires a compact recursive verifier and recursive-size closure.
- HyperNova
  [Construction 2](../../docs/hypernova-paper/14_6_3_A_compiler_from_NIVC_compatible_folding_schemes_to_NIVC.md)
  requires the cost of the complete augmented circuit. A source-row reduction
  is useful only if the recursive fixed point also becomes smaller.
- The local Lean
  [`CheckPlan`](../../formal/nightstream-lean/Nightstream/SuperNeo/CheckPlan.lean)
  defines proved redundancy, removal counterexamples, and inclusion-minimal
  soundness. The cvc5 results are inputs to these Lean checks.

The Rust exporter reads existing immutable audit views, such as the
[`R1CS snapshot`](../../crates/neo-fold-clean/src/engine/r1cs_circuit/relation.rs),
the
[`Nebula constraint-source audit`](../../crates/neo-fold-clean/src/frontends/nebula/f_prime/constraint_source_audit.rs),
and the
[`selective projected rows`](../../crates/neo-fold-clean/src/frontends/r1cs_f_prime/selective_projected_rows.rs).
These views do not give permission to remove constraints.

## Paper obligation review gate

The bridge records nine SuperNeo obligations and eleven HyperNova obligations.
Each entry names an exact base, recursive, or terminal row family, a terminal
native guard, a fixed-profile condition, or an open check. A mapped entry only
identifies its enforcement owner. It is not a semantic proof and does not
authorize removal of that owner.

The validator checks base and recursive row names against the exact emitted
Nebula family census. It checks terminal rows and guards against the reviewed
terminal vocabularies. The fixed-one program condition requires
`TRIVIAL_PC == 1`. This condition makes unchanged-slot copying vacuous. A
profile with more than one program must add and review the copy constraints.

The combined terminal linkage is mapped to `finish_combined_with_spartan` and
`verify_combined_spartan`. Both entry points use the manifest-owned combined
preprocessing object and reject a delayed-Nebula final-fold proof. The combined
relation needs the private witnesses in the pre-final running and latest pair.

One obligation is open:

- The production profile is not frozen, so its recursive fixed point has not
  been solved after an accepted removal batch.

## Redundancy query

For each R1CS row, define the residual equation

```text
(A_i · z) * (B_i · z) = C_i · z
```

For a candidate set `S`, the tool asks cvc5 to solve

```text
all rows outside S hold
and
at least one row in S does not hold
```

The query declares only columns reached by its current rows. For `sat`, the
parser merges those values into an exact satisfying lifecycle assignment. Rust
then evaluates all three sparse matrices for the complete source arm. If a
retained row fails, that row enters the next query. The iteration count and the
cvc5 process time are bounded.

The background assignment is used only to complete a `sat` model. It is not an
SMT assertion. An `unsat` result applies to the exact rows in that query, but a
family can be removed only when Lean checks a certificate for arbitrary field
assignments.

Both Lean result emitters require a complete source relation. The redundancy
emitter checks that every candidate and support row equals its row in that
complete relation. The counterexample emitter replays the full assignment over
the complete retained plan. Generated production theorems also require Lean's
`CoversFullRelation` premise, which checks the full source-row range and family
coverage. A bounded query slice is never the theorem target.

The result has this meaning:

| cvc5 result | Search conclusion | Required next check |
|---|---|---|
| `sat` | Counterexample candidate. The retained rows do not imply the candidate in this query. | Merge the active values into a satisfying lifecycle assignment, replay every retained Rust row, then check the complete model in Lean. |
| `unsat` | Redundancy candidate. The retained rows imply the candidate in this query. | Check a certificate in Lean before any row removal. |
| `unknown` | No conclusion. | Keep every candidate row. |
| Process or parse error | No conclusion. | Keep every candidate row and fix the tool path. |

Removing constraints preserves completeness when the witness layout and public
interface do not change. Soundness is the active risk. A row or column layout
change requires a new completeness check and a new recursive fixed point.

## Assurance boundary

The iterative path is:

```text
Rust relation and row ownership
        |
        v
bounded normalized R1CS slice
        |
        v
cvc5 implication query
   |             |
  sat           unsat
   |             |
Rust replay   small support or algebraic certificate
   |             |
Lean counterexample or Lean redundancy theorem
        |
        v
regenerated Rust artifact and recursive fixed point
        |
        v
Lean lifecycle and security composition
```

The cvc5 evidence is never authoritative. Each accepted removal must pass these
levels:

1. **Model-level:** Lean proves redundancy or records a removal counterexample
   for the complete typed input.
2. **Artifact-checked:** the proof consumes the exact source rows, complete
   family claim, source-to-final mapping, and projected final row terms for one
   named profile.
3. **Rust-conformant:** a drift check ties the Lean artifact to the Rust
   emitter.
4. **Security-reduced:** the closed recursive lifecycle reduces acceptance to
   the stated hash, transcript, and NIFS assumptions.

## Query scopes

Every input declares one scope:

- `local` for a self-contained algebraic gadget;
- `branch` for one complete base, bootstrap-recursive, recursive, or terminal
  relation;
- `lifecycle` for adjacent producer and consumer steps plus terminal closure.

A local query cannot authorize removal of a delayed obligation. The current F'
design delays some fresh-public and accumulator checks to the next consumer or
terminal verifier. Such candidates require a lifecycle slice.

## Meaning of minimum

The initial target is inclusion-minimality inside a declared row-family set:
each remaining family has a Lean-checked counterexample when it is removed.

This is different from two stronger claims:

- A cardinality minimum compares different combinations of retained rows.
- A global arithmetization minimum compares different circuit encodings.

The second claim needs a finite rewrite grammar. The cost must also include the
stabilized recursive result, not only the source row count. Nightstream's
current cost order is recurring rows, committed columns, public columns, and
auxiliary columns.

## Input format

An input contains normalized source R1CS rows. Coefficients are canonical
decimal Goldilocks residues. Terms are strictly ordered by column. Column zero
is usually the constant-one column, but the field is explicit.

```json
{
  "schema": "nightstream/r1cs-redundancy-problem/v3",
  "source": {
    "profile": "fprime-recursive-output-slice",
    "artifact_digest": "sha256:...",
    "scope": "branch",
    "total_rows": 1000
  },
  "field_modulus": "18446744069414584321",
  "column_count": 3,
  "constant_one_column": 0,
  "public_input_count": 1,
  "complete_families": ["output.copy"],
  "rows": [
    {
      "id": "output.copy.0",
      "source_index": 701,
      "family": "output.copy",
      "a": [{ "column": 1, "coefficient": "1" }],
      "b": [{ "column": 0, "coefficient": "1" }],
      "c": [{ "column": 2, "coefficient": "1" }]
    }
  ]
}
```

`source_index` is the row index in the complete source relation.
`public_input_count` is the exclusive end of the normalized public-column
prefix. The private columns are the remaining columns.
`complete_families` lists only families whose full owned row set is present.
A family query fails if its name is not in that list. The exporter must derive
`artifact_digest`, row IDs, families, completeness, and coefficients from the
exact Rust artifact. Caller labels alone are not authority.

The generic fixed-point and Nebula bridges return this problem together with a
selective binding. The binding records the physical arm, requested and
rewrite-closure source rows, retained row images, complete rewrites, emitted
rows, final dimensions, and exact projected row terms. SHA-256 values in this
tool are diagnostic only. They are not protocol hashes and do not replace exact
row equality.

The terminal bridge returns the same source problem with a terminal binding.
That binding records every selected source row, the exact source-to-Spartan
column permutation, the projected Spartan row, padded dimensions, and padding
ranges. It also carries the exact verifier-native guard ledger. Lean checks
that the guard names are unique, nonempty, retained by exact artifact
validation, and absent from the polynomial family set. These checks are
artifact-checked. They do not replace the separate semantic proof for each
guard.

## Saved Nebula source assignments

The bridge can save an expensive accepted source assignment for later cvc5
runs. The file is untrusted. It is not a lifecycle proof and its digest is not
an authority.

`bind_nebula_source_assignment` accepts an in-memory assignment only after a
complete source-arm replay. `CheckedNebulaSourceAssignment::to_json_vec` writes
the checked value. `load_nebula_source_assignment` then does all checks again
against a new exact Rust audit. It checks the expected profile, physical arm,
recomputed source digest, field modulus, dimensions, canonical field values,
constant-one column, and every source row.

The format is:

```json
{
  "schema": "nightstream/nebula-source-assignment/v1",
  "profile": "reviewed-profile-name",
  "source_arm": "recursive",
  "source_artifact_digest": "sha256:...",
  "field_modulus": "18446744069414584321",
  "source_rows": 1000,
  "source_columns": 2000,
  "public_input_count": 100,
  "constant_one_column": 0,
  "values": ["1", "0", "42"]
}
```

Each value is a canonical decimal Goldilocks residue. Bootstrap-recursive and
steady-recursive steps use the same `recursive` physical arm. Thus, one saved
bootstrap assignment can be loaded for either recursive lifecycle role. Only
the returned checked values can be used as cvc5 background assignments.

## Build and test

This is an independent Cargo workspace. It does not enter the protocol or
prover dependency graph.

```bash
cargo test --workspace --release \
  --manifest-path tools/recursive-constraint-minimizer/Cargo.toml
```

cvc5 must be a GPL build with finite-field support. The
[cvc5 finite-field documentation](https://cvc5.github.io/docs/latest/theories/finite_field.html)
states that this support uses the prime-field `QF_FF` logic and requires the
extended build with CoCoA. The runner uses cvc5's
[`tlimit-per` option](https://cvc5.github.io/docs/latest/options.html) to set a
solver limit for each query. The host limit is slightly larger so cvc5 can
print its result and model before termination.

On Apple Silicon, download the macOS ARM64 GPL archive from the
[official cvc5 releases](https://github.com/cvc5/cvc5/releases), unpack it, and
check the binary:

```bash
/path/to/cvc5/bin/cvc5 --version
```

Pass that path with `--solver /path/to/cvc5/bin/cvc5`. The standard `cvc5`
Python package is the BSD build and does not provide the finite-field solver.

## Emit a query

The included fixture has one bitness row and two copies of `x = 0`. Removing
one copy is the positive redundancy control.

```bash
cargo run --release \
  --manifest-path tools/recursive-constraint-minimizer/Cargo.toml -- \
  emit \
  --input tools/recursive-constraint-minimizer/examples/known-local.json \
  --remove-row zero_copy \
  --output target/constraint-minimizer/zero-copy.smt2
```

Run the bounded check and save all evidence:

```bash
cargo run --release \
  --manifest-path tools/recursive-constraint-minimizer/Cargo.toml -- \
  check \
  --input tools/recursive-constraint-minimizer/examples/known-local.json \
  --remove-row zero_copy \
  --evidence target/constraint-minimizer/zero-copy.json \
  --ff-solver gb \
  --timeout-ms 60000
```

Removing the complete `zero` family is the negative control. The assignment
`x = 1` satisfies bitness and violates both removed rows, so cvc5 must return
`sat`.

```bash
cargo run --release \
  --manifest-path tools/recursive-constraint-minimizer/Cargo.toml -- \
  check \
  --input tools/recursive-constraint-minimizer/examples/known-local.json \
  --remove-family zero \
  --evidence target/constraint-minimizer/zero-family.json \
  --ff-solver split \
  --timeout-ms 60000
```

## Next work

1. Prove that the Rust and Lean seeded-Phi81 samplers produce the same row
   coefficients. Add a compiler-grammar complete Rust-to-Lean relation
   artifact. Use exact data as authority; do not replace rows with a digest.
2. Select and freeze the production field-native candidate, application, plan,
   and parameter profile. Set `production_claim` only after this review.
3. Record one accepted bootstrap-recursive source assignment and replay it
   against the shared bootstrap and steady-recursive source arm. The reduced
   base assignment already passes exact source-arm replay. The two-step CPU
   capture exceeds the five-minute test limit. Metal device creation is not
   available in the current execution environment. Use an available
   accelerator to create the checked saved source-assignment artifact.
4. Revalidate the paper obligation map against the frozen family census. Run
   the exhaustive Nebula and terminal drivers. Retain every inconclusive
   family.
5. Emit and check Lean artifacts for each redundancy certificate and complete
   removal counterexample. Refine the Rust context and statement computations
   to the native guard models.
6. Apply only Lean-proved removal batches, regenerate the two physical Nebula
   arms and terminal relation, recompute the recursive fixed point, and repeat
   until no checked removal remains.
7. Instantiate the normalized Lean assembly theorems for the final classified
   plan. Prove it sound and inclusion-minimal, then report its
   stabilized rows, committed columns, public columns, and auxiliary columns.

The next controls are the audited Poseidon2 output-copy rows, selector-total
necessity, and terminal prior-link necessity.
