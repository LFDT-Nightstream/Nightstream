# compression-lean

This project proves a soundness bound for Nightstream's compressed proof: the final verifier
(`CompressionKey::verify`, `crates/nightstream/src/lifecycle/finish.rs`) accepts a false statement
only with a small probability, under the assumptions that `TRUST.md` lists. Two of them carry the
most weight: the WHIR openings are idealized, and Fiat–Shamir inside the final program is assumed,
not bounded (`TRUST.md` §3). The claims are the theorems of `CompressionLean/Claims.lean`.

## Strategy

The compressed proof has three layers. The terminal package is F′'s Stage 1 rows up to PiRLC. The
layer-1 verifier (`crates/neo-spartan/src/verifier.rs`) checks a CE(B) claim about the PiRLC parent.
The shrink verifier (`crates/neo-spartan/src/shrink/mod.rs`) checks a CCS that encodes the final
program, which is the terminal package and the layer-1 verifier written as a circuit.

1. Each verifier is a public-coin game against an adaptive prover (`Game/`). Each challenge names
   its bad set from the earlier messages. If every path that avoids the bad sets ends in a true
   statement, a prover reaches a false one with probability at most the sum of the bad-set
   densities on the worst path. Under Fiat–Shamir, a state-restoration experiment lets the prover
   ask for challenges at any node; then each query costs at most the game's error.
2. The terminal package is F′'s rows without change, so its rows give F′'s meaning (`Terminal/`).
3. Layer 1 and the shrink layer each prove that an accepted transcript has a witness, except on a
   bad challenge (`Layer1/`, `Shrink/`). The WHIR openings are idealized; their error is not part
   of the bound.
4. The chain (`Final/`): the shrink verifier accepts, so a candidate satisfies the CCS of the final
   program. The encoding premise (`ProgramEncodes`) says that such a witness holds a terminal
   assignment and accepting layer-1 paths for its parent claim. Layer 1 then gives a witness of
   that claim, and under the honest setup the claim is F′'s (`ParentClaim`). Both layers run in
   one experiment, so one bad-set argument covers both. The chain covers a valid program witness
   only where its in-program challenges are the experiment's oracle answers (the read condition,
   `TRUST.md` §3).
5. The encoding premise is proved for the production final program (`Sim/Encodes.lean`). Lean
   lowers the program to the shrink run as Rust's recorder does (`Lowering/`). A witness of the
   run's CCS gives an accepting run of an executor, and the simulation (`Sim/`) shows that each part
   of the circuit computes the values of an executable replay of the layer-1 verifier
   (`Ports/Replay/`). The bridge (`Bridge/`) turns an accepted replay into a path of the layer-1
   game.
6. The claims take the Rust artifacts as premises: facts about the key, the setup, the runs and the
   package. The executables check these facts on the files that Rust writes, and a theorem per check
   proves that a pass gives the premise (`Checks/`).
7. Lean transcribes parts of Rust by hand (`Ports/`). `ports.json` pins each transcription to its
   Rust item, and the cross-checks compare the transcriptions with Rust on fixed inputs.

`TRUST.md` lists what the claims trust: the kernel, F′, the named assumptions, the premises and
the ports. `STATUS.md` lists each claim's axioms and hypotheses, and how each premise is met.

## Layout

A module imports only its own folder and the folders above it in this list
(`CompressionLeanTests/Layers.lean`).

```text
CompressionLean/
  Lib/             generic lemmas: lists, arrays, the Boolean cube, sum-checks, logUp, x³ - x - 1
  Fields.lean      Gl, Cubic and the challenge field
  Production.lean  the production constants of the layer-1 key, each named once
  Game/            the game model, paths, state restoration, two games in one experiment
  Terminal/        F′'s terminal package and its soundness edge
  Layer1/          the layer-1 verifier as a game: model, stages, cost
  Ports/           the hand transcriptions of Rust, definitions only, folders as the Rust modules
  Shrink/          the shrink verifier as a game, over Ports/Shrink/
  Lowering/        the proofs about the lowering of a program to a shrink run
  Spec/            the definitions that the claims name and no folder above owns
  Bridge/          an accepted replay is a path of the layer-1 game
  Final/           the chain of the layers, the budget at Rust's split, F′'s parent claim
  Checks/          a pass of an executable's check gives a premise
  Sim/             the simulation of the circuit programs; the encoding premise
  Claims.lean      the claims
CompressionLeanExe/     the executables, their readers and the Lean side of the Rust tests
CompressionLeanTests/   the build gates
artifacts/              the emitted terminal package
scripts/crosschecks.sh  the Rust–Lean cross-checks
```

Each folder's aggregator module (`CompressionLean/<Folder>.lean`) states what the folder owns and
what it does not own.

## Build

Run each Lean command under the 1,500 s cap. Copy the F′ build cache into `.lake` first
(`cp -cR ../nightstream-fprime/.lake/packages .lake/`), so that Lake does not rebuild Mathlib.

```bash
timeout --signal=KILL 1500 lake build
timeout --signal=KILL 1500 lake exe status > STATUS.md
timeout --signal=KILL 1500 .lake/build/bin/emitTerminal artifacts/compression-terminal-v1.json
```

`lake build` builds the library, the executables and the gates, with warnings as errors:

- the axiom audit (`Axioms.lean`): every theorem uses only `propext`, `Classical.choice` and
  `Quot.sound`;
- the ports gate (`Ports.lean`): every definition of `Ports/` has a `ports.json` entry or a reason
  in `notPorts`, and `Ports/` holds no theorem. It also fails when `STATUS.md` is not what
  `lake exe status` prints, and the page fails when a premise of a claim has no entry in
  `Inventory.lean`;
- the folder gate (`Layers.lean`);
- the reading gate (`Reading.lean`): a valid program witness has a reading given WHIR's openings
  and `MerkleBinds`, and a leaf meets `MerkleBinds` when its path reaches the root at no other
  index below the leaf count and it holds its own index's codeword;
- the vocabulary gate (`Vocabulary.lean`): a claim names only constants of `Spec/`, the model and
  port folders, F′, Mathlib or core.

The `nightstream` crate embeds `artifacts/compression-terminal-v1.json` and pins its identifier
(`IDENTIFIER` in `crates/nightstream/src/lifecycle/terminal/package.rs`). The CI test
`lean_ports` (`nightstream`) fails when a Rust item that `ports.json` names changes.

## Cross-checks

`scripts/crosschecks.sh` builds the executables and runs the ignored Rust tests that compare Rust
with Lean, one after another, each under its cap. Its comments say what each check compares, which
files it writes and reads, and how to update a digest pin.
