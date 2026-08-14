# Recursive constraint minimizer

This subproject uses cvc5 to search for redundant constraints in bounded slices
of the Nightstream recursive verifier. Lean remains the proof authority. Rust
remains the source of the emitted relation.

The tool does not edit a circuit. It does not mark a row as safe to remove. It
records a finite-field query and the raw cvc5 result so that later Rust and Lean
checks can accept or reject the candidate.

## Status

The first milestone is implemented:

- a versioned JSON format for a normalized Goldilocks R1CS slice;
- row and row-family selection;
- exact SMT-LIB `QF_FF` generation;
- bounded cvc5 execution with the `gb` or `split` finite-field solver;
- fail-closed handling of `unknown`, invalid output, and process errors;
- JSON evidence that contains the exact query, row map, model or unsat core,
  and source artifact identity;
- local positive and negative control fixtures.

The exact recursive-verifier exporter and the Lean certificate checker are the
next integration steps. Until those exist, an `unsat` result is useful search
evidence, not an artifact-checked removal certificate.

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

The Rust exporter will read the existing immutable audit views, such as the
[`R1CS F-prime snapshot`](../../crates/neo-fold-clean/src/frontends/r1cs_f_prime/lowering/snapshot.rs)
and the
[`selective projected rows`](../../crates/neo-fold-clean/src/frontends/r1cs_f_prime/selective_projected_rows.rs).
These views do not give permission to remove constraints.

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

The result has this meaning:

| cvc5 result | Search conclusion | Required next check |
|---|---|---|
| `sat` | Counterexample candidate. The retained rows do not imply the candidate in this query. | Replay the model in Rust. Use the existing Lean completeness theorem to prove a removal counterexample. |
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
2. **Artifact-checked:** the proof consumes the exact rows for one named
   profile.
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

An input contains normalized R1CS rows. Coefficients are canonical decimal
Goldilocks residues. Terms are strictly ordered by column. Column zero is
usually the constant-one column, but the field is explicit.

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

## Build and test

This is an independent Cargo workspace. It does not enter the protocol or
prover dependency graph.

```bash
cargo test --release \
  --manifest-path tools/recursive-constraint-minimizer/Cargo.toml
```

cvc5 must include finite-field support. The
[cvc5 finite-field documentation](https://cvc5.github.io/docs/latest/theories/finite_field.html)
states that this support uses the prime-field `QF_FF` logic and requires the
extended build with CoCoA. The runner uses cvc5's
[`tlimit-per` option](https://cvc5.github.io/docs/latest/options.html) to set a
wall-clock limit for each query.

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

## Planned integration

The next implementation slices are intentionally small:

1. Add a diagnostic exporter that reads `R1csSnapshot` plus exact row-family
   ownership and emits this JSON format.
2. Parse cvc5 finite-field models into canonical Goldilocks assignments and
   replay every `sat` result against the retained Rust rows.
3. Define a compact polynomial-combination certificate for simple `unsat`
   results and check it in Lean.
4. Import the checked result into `CheckPlan.Redundant` or
   `NecessaryForSoundness`.
5. Regenerate the relation, recompute the recursive fixed point, and repeat the
   search.

The first production pilots should use known controls: duplicate Boolean rows,
the audited Poseidon2 output-copy rows, selector-total necessity, and terminal
prior-link necessity. A new family should enter the search only after the tool
reproduces those results.
