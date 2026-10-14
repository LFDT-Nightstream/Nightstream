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
  formulas equal to them. `CubicField.lean` proves that `x³ - x - 1` is
  irreducible over Goldilocks, so the challenge field is a field.
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
- Query indices: `sample_bits` takes the low `setupVars - 2` bits of a
  uniform Goldilocks element. Each index is at most `1 + 2^(setupVars - 66)`
  times as likely as under the uniform law.
- Setup root: the Merkle root of the setup binds its leaves, so the opened
  setup values are the honest setup's (`HonestStage.lean`).
- χ: the statement carries the values of `χ_r` (`Model.lean`). `mle_chi`
  proves the `chi` formula over any commutative ring; that its `K ⊗ Ext`
  value splits into the model's re and im tables is not proved.
- Key order: the key's runs are sorted by matrix and its slots by segment,
  so the matrix column and the slot tables are step functions
  (`ClosedForms.lean`).
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
