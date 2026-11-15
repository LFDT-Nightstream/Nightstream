These fixtures are recorded Lean outputs. The Rust tests read them and need no
Lean installation. The Rust application builder does not read them, and
`Circuit::compile`, `load`, `prove`, `extend` and `verify` do not read them.

## Poseidon2 application

`poseidon2-hash-chain-v1-digest.json` holds the row count, 5,484, and a digest
of the rows of the Lean Poseidon2 hash chain
(`Lifecycle/Stage1/Poseidon2HashChainV1.lean`), in the variable order of the
Rust application (`EmitMain.applicationDigest`).

`tests/application/poseidon2.rs` reads it. The row test requires the Rust
Poseidon2 application to have the same rows, row for row. A second test
checks that the Rust witness satisfies the rows and that its output is the
native Poseidon2 hash. A third test checks that a changed input, output or
intermediate value fails the constraints. These checks cover this one
application. They are not a proof for all Rust applications.

## Nebula memory applications

`nebula-memory-v1-rows.json` and `nebula-memory-n1-rows.json` name the
assertion rows of the two Nebula memory applications as `[name, first, end]`,
counted from the first row of the application. Each name comes from the
SPEC.md section 7 check group that defines its rows (`Checks` in
`Lifecycle/Nebula/RowList.lean` and `MemoryCircuit.lean`). The SPEC.md
section 9 rejection tests (`tests/nebula/conformance.rs`) use them to name the
check that a faulty step fails.

`nebula-memory-v1-start.json` and `nebula-memory-n1-start.json` hold `D_init`
of the default program of each application (`initRootWords`, proved equal to
`(context plan).initRoot`). `nebula-memory-v1-other-start.json` holds `D_init`
of `FirstPlan.otherPlan`, a second program of the V1 geometry with a distinct
nonzero word in every ROM and RAM cell. `tests/nebula_memory.rs` compares the
`D_init` of the verifier with each.

`nebula-memory-v1-terminal.json` and `nebula-memory-n1-terminal.json` hold
SPEC.md Algorithm 8 terminal-check cases of each plan
(`NebulaMemoryTerminal.cases`). Each case has the statement, the final carry
words, and whether every statement integer that becomes a field word is below
`q`. It also has each `TerminalChecks` field as Lean decides it
(`terminalChecks_iff`) and the final state words. `terminal_checks_match_lean`
requires the terminal check of the verifier to agree on every case.

`emitBoundedDepth` makes the rows, start and terminal files again from
`formal/nightstream-fprime`, after `scripts/validate.sh executables`. The
output is the same bytes as the committed files:

```sh
for mode in rows start terminal; do
  for plan in nebula-v1 nebula-n1; do
    timeout --signal=KILL 1500 bash scripts/validate.sh lean-executable .lake/build/bin/emitBoundedDepth --application-$mode $plan ../../crates/nightstream/tests/fixtures/nebula-memory-${plan#nebula-}-$mode.json
  done
done
timeout --signal=KILL 1500 bash scripts/validate.sh lean-executable .lake/build/bin/emitBoundedDepth --application-start nebula-v1-other ../../crates/nightstream/tests/fixtures/nebula-memory-v1-other-start.json
```

`nebula-memory-v1-digest.json`, `nebula-memory-n1-digest.json` and
`nebula-memory-r1-digest.json` (a RAM smaller than the ROM, `ThirdPlan.lean`) hold the row
count and a digest of the rows of each memory application, in the variable
order of the Rust application (`EmitMain.applicationDigest`).
`tests/nebula/parity.rs` requires the Rust-built memory circuit to have the
same rows. Make the four digest files again from `formal/nightstream-fprime`,
after `scripts/validate.sh executables`:

```sh
timeout --signal=KILL 1500 bash scripts/validate.sh lean-executable .lake/build/bin/emitBoundedDepth --application-digest poseidon2-hash-chain-v1 ../../crates/nightstream/tests/fixtures/poseidon2-hash-chain-v1-digest.json
timeout --signal=KILL 1500 bash scripts/validate.sh lean-executable .lake/build/bin/emitBoundedDepth --application-digest nebula-v1 ../../crates/nightstream/tests/fixtures/nebula-memory-v1-digest.json
timeout --signal=KILL 1500 bash scripts/validate.sh lean-executable .lake/build/bin/emitBoundedDepth --application-digest nebula-n1 ../../crates/nightstream/tests/fixtures/nebula-memory-n1-digest.json
timeout --signal=KILL 1500 bash scripts/validate.sh lean-executable .lake/build/bin/emitBoundedDepth --application-digest nebula-r1 ../../crates/nightstream/tests/fixtures/nebula-memory-r1-digest.json
```
