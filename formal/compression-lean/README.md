# compression-lean

The fixed layer-0 part of the compression circuit, built from the F′ Lean
circuits without change.

- `Terminal/Data.lean`: the terminal package, the F′ Stage 1 rows up to PiRLC
  (prior-state hash, PiCCS, PiRLC sampler and combination). No new row.
- `Terminal/Soundness.lean`: if every row holds, PiCCS accepts, the PiRLC
  parent is the key's parent, the prior state hashes to the fresh input, and
  the sampler state is the transcript state after PiRLC.
- `Terminal/Outputs.lean`: the columns of the parent and of that state.
- `artifacts/compression-terminal-v1.json`: the emitted artifact. The
  `nightstream` crate embeds it and pins its identifier (`IDENTIFIER` in
  `crates/nightstream/src/lifecycle/terminal/package.rs`).

## Layer 1

`Layer1/` proves the soundness of the layer-1 verifier
(`crates/neo-spartan/src/verifier.rs`) against an adaptive prover.

- Model: the verifier is a public-coin game (`Game.lean`). Each challenge
  names its bad set, from the earlier messages only. If every path that
  avoids the bad sets ends in the claim, a prover breaks the claim with
  probability at most the sum of the bad-set densities on the worst path.
- Adversary: the prover picks every message after it sees all earlier
  challenges. It commits to at most `L0` P0 and `L1` P1 candidate tables
  (the WHIR lists). Every bad set is a union over these candidates.
- Idealized openings: the verifier accepts the two WHIR openings when one
  P0 candidate and one P1 candidate agree with every opened value
  (`Soundness.lean`, `Opens`). The WHIR error of this step is a named term
  of the budget (`Budget.lean`), not proved here.
- Claim: `Cost.lean`, `layer1_sound`. An accepted transcript has a P0
  candidate that is a CE(B) witness (`Model.lean`, `Relation`), except with
  probability `layer1Error`. `Budget.lean` bounds it at production sizes.
- Fiat–Shamir: `Restoration.lean` lets the prover ask for challenges at any
  node, as often as it wants, and output one path that uses only answers it
  got. `FiatShamir.lean` bounds layer 1 there by `queries · layer1Error`.
  The step from Poseidon2 to this experiment (random oracle, injective
  transcript encoding) is assumed.
- Stage order (`Stages.lean`): early (norm, row lookup, slot scatter, GKR),
  linear (ring rows at `λ`, `ζ`, two sum-checks), honest fold (setup
  partials, `α`, setup queries), late (block scatter, GKR).

Build and emit (each Lean command under the 1,500 s cap). Copy the F′ build
cache into `.lake` first (`cp -cR ../nightstream-fprime/.lake/packages .lake/`)
so that Lake does not rebuild Mathlib.

```bash
timeout --signal=KILL 1500 lake build
timeout --signal=KILL 1500 lake build emitTerminal
timeout --signal=KILL 1500 .lake/build/bin/emitTerminal artifacts/compression-terminal-v1.json
```

`lake build` also runs the axiom audit (`CompressionLeanTests/Axioms.lean`).
