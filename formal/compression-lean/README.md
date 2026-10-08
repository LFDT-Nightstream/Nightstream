# compression-lean

The fixed layer-0 part of the compression circuit, built from the F′ Lean
circuits without change.

- `Terminal/Data.lean`: the terminal package, the F′ Stage 1 rows up to PiRLC
  (prior-state hash, PiCCS, PiRLC sampler and combination). No new row.
- `Terminal/Soundness.lean`: if every row holds, PiCCS accepts, the PiRLC
  parent is the key's parent, the prior state hashes to the fresh input, and
  the sampler state is the transcript state after PiRLC.
- `Terminal/Outputs.lean`: the columns of the parent and of that state.
- `artifacts/compression-terminal-v1.json`: the emitted artifact. Rust pins
  its identifier (`TERMINAL_PACKAGE_IDENTIFIER` in
  `crates/nightstream/src/lifecycle/finish.rs`).

Build and emit (each Lean command under the 1,500 s cap). Copy the F′ build
cache into `.lake` first (`cp -cR ../nightstream-fprime/.lake/packages .lake/`)
so that Lake does not rebuild Mathlib.

```bash
timeout --signal=KILL 1500 lake build
timeout --signal=KILL 1500 lake build emitTerminal
timeout --signal=KILL 1500 .lake/build/bin/emitTerminal artifacts/compression-terminal-v1.json
```

`lake build` also runs the axiom audit (`CompressionLeanTests/Axioms.lean`).
