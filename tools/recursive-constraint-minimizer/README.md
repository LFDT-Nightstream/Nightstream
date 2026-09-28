# Recursive constraint minimizer

This independent Rust tool reads a normalized Goldilocks R1CS slice and asks
cvc5 whether selected rows follow from the retained rows. It records the
query, row map, solver result, and any returned model or unsat core.

The former Nightstream export bridge has been removed. This tool accepts an
explicit JSON problem; it has no dependency on the current prover or verifier.
The current Nightstream relation is owned by `formal/nightstream-fprime`.
Constraint removal still needs a proof against that relation and measurements
of the resulting recursive circuit.

The library provides row and family selection, active-column SMT queries,
finite-field model replay, scalar polynomial certificate checking, and bounded
solver execution. Timeouts, unknown results, malformed output, and process
errors produce an inconclusive result.

## Build and test

This is a separate Cargo workspace:

```sh
timeout --signal=KILL 300 cargo test --workspace --release \
  --manifest-path tools/recursive-constraint-minimizer/Cargo.toml
```

Queries require a cvc5 build with `QF_FF` support. Pass its executable with
`--solver /path/to/cvc5`. Each query has a solver limit and a host deadline;
`--timeout-ms` cannot exceed 300000.

## Example

The [local fixture](examples/known-local.json) contains one bitness row and two
copies of `x = 0`. Removing one copy is the positive redundancy control:

```sh
timeout --signal=KILL 300 cargo run --release \
  --manifest-path tools/recursive-constraint-minimizer/Cargo.toml -- \
  emit --input tools/recursive-constraint-minimizer/examples/known-local.json \
  --remove-row zero_copy --output target/constraint-minimizer/zero-copy.smt2
```

To execute the query and retain its evidence:

```sh
timeout --signal=KILL 300 cargo run --release \
  --manifest-path tools/recursive-constraint-minimizer/Cargo.toml -- \
  check --input tools/recursive-constraint-minimizer/examples/known-local.json \
  --remove-row zero_copy --evidence target/constraint-minimizer/zero-copy.json \
  --ff-solver gb --timeout-ms 60000
```

Removing the whole `zero` family is the negative control: `x = 1` satisfies
bitness and violates the removed rows. Use `--remove-family zero` and a separate
evidence path for that check.

The CLI and library share the problem validation in [problem.rs](src/problem.rs).
A solver result describes the supplied slice. It does not establish a complete
Nightstream refinement or authorize editing the circuit.
