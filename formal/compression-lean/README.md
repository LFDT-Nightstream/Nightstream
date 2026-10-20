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
  challenges, and it names the CE(B) statement first. It commits to at most
  `L0` P0 and `L1` P1 candidate tables (the WHIR lists). Every bad set is a
  union over these candidates.
- Idealized openings: the verifier accepts the two WHIR openings when one
  P0 candidate and one P1 candidate agree with every opened value
  (`Soundness.lean`, `Opens`). The WHIR error of this step is a named term
  of the budget (`Budget.lean`), not proved here.
- Claim: `Cost.lean`, `layer1_sound`. An accepted transcript has a P0
  candidate that is a CE(B) witness (`Model.lean`, `Relation`), except with
  probability `layer1Error`. `Budget.lean` bounds it at Rust's split:
  `2^-t` for layer bits `t`, when each WHIR commitment's composed error, as
  Plonky3 charges it, is at most `2^-(t+1)`.
- Fiat–Shamir: `Restoration.lean` lets the prover ask for challenges at any
  node, as often as it wants, and output one path that uses only answers it
  got. `FiatShamir.lean` bounds layer 1 there by `calls · layer1Error`,
  which is at most `calls / 2^t` at Rust's split.
- Verifier formulas: the model takes the public tables at their
  definitions. `ClosedForms.lean` and `EvalK.lean` prove the `mle.rs`
  formulas equal to them. The statement carries the point `r`, and `χ_r` is
  `eq(r, ·)` in `K` (`Model.lean`). `chi` and `eval_k` run in `K ⊗ Ext`, and
  their two parts are the extensions of the re and im tables of `χ_r`.
  `CubicField.lean` proves that `x³ - x - 1` is irreducible over Goldilocks,
  so the challenge field is a field.
- Stage order (`Stages.lean`): early (norm, row lookup, slot scatter, GKR),
  linear (ring rows at `λ`, `ζ`, two sum-checks), honest fold (setup
  partials, `α`, setup queries), late (block scatter, GKR).

Layer 1 assumes, and does not prove:

- WHIR: an opening is accepted only when one listed candidate agrees with
  every opened value (`Opens`), except with that opening's error. The P0
  candidates are tables over the base field; candidates over `Ext` are not
  modeled. The errors come from Plonky3's Johnson-bound report (`pcs.rs`).
- Fiat–Shamir: Poseidon2 acts as a random oracle, and the transcript
  encoding is injective on prefixes. One permutation call can give `k`
  challenges, so `Q` calls and the `R` draws of one path are at most
  `k·Q + R` queries of `Restoration.lean`. WHIR's own Fiat–Shamir step is
  part of the WHIR error.
- Query indices: the model draws each query leaf as a uniform Goldilocks
  word mod the leaf count, as `sample_bits` does (`HonestStage.lean`).
  `layer1_budget` needs at most 35 setup variables (`query_half`).
  `Pcs::new` gives it: P1 stacks `6·2^(setupVars - 3)` cells, and Plonky3's
  WHIR refuses a first folded domain above `2^32`.
- Setup root: the Merkle root of the setup binds its leaves, so the opened
  setup values are the honest setup's (`HonestStage.lean`).
- Key structure: the verifier reads the matrix column and the slot lane
  tables as step functions of the key's run bounds and slot segments
  (`KeyStructure.lean`). The honest key describes its setup
  (`Structure.Describes`): run `k` of matrix `j` lies between that matrix's
  run bounds, and each slot has its segment's class and lane.
  `Tables::from_rows` (`matrix.rs`) builds the key so. `Structure::check`
  also enforces the order part (`Structure.Checked`) on every key.
- Transcription: the model is written by hand from the Rust verifier.
- Charges: Plonky3 composes a commitment's error as its WHIR terms plus each
  charged term times its candidate count. `rustDegrees` (`Budget.lean`) is
  `outer_terms` written in Lean, and `Pcs::new` enforces the two composed
  bounds that `layer1_budget` takes as premises.

Build and emit (each Lean command under the 1,500 s cap). Copy the F′ build
cache into `.lake` first (`cp -cR ../nightstream-fprime/.lake/packages .lake/`)
so that Lake does not rebuild Mathlib.

```bash
timeout --signal=KILL 1500 lake build
timeout --signal=KILL 1500 lake build emitTerminal
timeout --signal=KILL 1500 .lake/build/bin/emitTerminal artifacts/compression-terminal-v1.json
```

`lake build` also runs the axiom audit (`CompressionLeanTests/Axioms.lean`).
