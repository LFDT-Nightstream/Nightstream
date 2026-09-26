# Recursive constraint minimizer

This independent Cargo workspace searches for redundant constraints in supplied
Goldilocks R1CS slices. It emits finite-field queries, runs cvc5, and records
results and candidate certificates. It does not enter the prover dependency
graph or change a production circuit.

The legacy Rust relation bridge and its WASM and Nebula integration are removed.
The generic minimizer and the [current experiments](experiments/README.md)
remain. Historical bridge results do not establish conformance of the selected
Nightstream package. The maintained Lean project is `formal/nightstream-fprime`;
the deprecated Lean corpus is not a production proof authority.

## Query and evidence

For each row, the residual equation is `(A_i · z) * (B_i · z) = C_i · z`.
For a selected row or family, the query requires every retained row to hold
and at least one selected row to fail. It declares the columns used by those
rows. The input schema, canonical field values, column ranges, row identities,
and family selection are checked before query generation.

| Result | Meaning |
|---|---|
| `sat` | A candidate counterexample for the supplied query. Check the complete retained relation before using it. |
| `unsat` | A candidate redundancy result. A Lean proof over the intended complete relation is required before row removal. |
| `unknown`, timeout, or process error | No conclusion; retain the selected rows. |

A digest identifies recorded data. It does not prove row equality or soundness.
A local slice cannot establish a deferred lifecycle obligation. Changes to row
or column layout also require completeness and recursive relation checks.

## Build and test

Run from the repository root. The 300-second cap comes from the root `AGENTS.md`.

```sh
timeout --signal=KILL 300 cargo test --workspace --release \
  --manifest-path tools/recursive-constraint-minimizer/Cargo.toml
cargo build --release --manifest-path tools/recursive-constraint-minimizer/Cargo.toml
```

The executable is under this workspace's `target/release` directory. For
solver checks, supply a cvc5 build with finite-field support through `--solver`
when it is not on `PATH`. `--timeout-ms` selects the query limit and cannot
exceed 300000. The outer command cap also covers result handling.

## Local controls

The included fixture has a bitness row and two copies of `x = 0`. Removing
one copy is the positive redundancy control. Removing the `zero` family is
the negative control: `x = 1` satisfies bitness and violates that family.

Emit the positive control query:

```sh
timeout --signal=KILL 300 tools/recursive-constraint-minimizer/target/release/recursive-constraint-minimizer emit \
  --input tools/recursive-constraint-minimizer/examples/known-local.json \
  --remove-row zero_copy --output /tmp/zero-copy.smt2
```

Check it and keep the evidence:

```sh
timeout --signal=KILL 300 tools/recursive-constraint-minimizer/target/release/recursive-constraint-minimizer check \
  --input tools/recursive-constraint-minimizer/examples/known-local.json \
  --remove-row zero_copy --evidence /tmp/zero-copy.json --ff-solver gb
```

Use `--remove-family zero --ff-solver split` for the negative control. Neither
control is a test of the complete recursive verifier. The experiments document
their own candidate inputs and required proof and conformance checks.
