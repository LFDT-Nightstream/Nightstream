# What is trusted

Everything that, if wrong, could make a claim of this project false without a Lean error. The claims
are the theorems of `CompressionLean/Claims.lean`, and `CompressionLeanTests/Axioms.lean` pins their
axioms. `lake build` checks that every theorem of the library uses only `propext`, `Classical.choice`
and `Quot.sound`. The README states, per layer, what each part assumes; this file lists the layers in
one place.

## 1. The Lean kernel and the build cache

The toolchain in `lean-toolchain`. No proof may use `native_decide` (the axiom audit rejects
`Lean.ofReduceBool`), so no proof trusts the compiled evaluator. The executables of section 5 do: they
link the claims to Rust's artifacts. The `.olean` files of Mathlib and of
`nightstream-fprime` come from a build cache that `lake build` does not re-check. Re-checking the
environment with `lean4checker`, or building both from source, removes this layer.

## 2. `nightstream-fprime`

This project imports F′'s definitions and theorems: the terminal package and its rows, the Poseidon2
specification, the Ajtai setup and the paper algebra. A claim here is only as good as F′'s, and
inherits F′'s own trusted layers.

## 3. Named assumptions

The claims bound a probability. They take these as part of the model, not as Lean axioms:

- **WHIR.** An opening is accepted only when one listed candidate agrees with every opened value
  (`Opens`), except with that opening's error. The error figures come from Plonky3's Johnson-bound
  report in Rust (`pcs.rs`) and enter `Budget.lean` as given. No Lean proof checks them. This is the
  weakest numeric layer.
- **Fiat–Shamir.** Poseidon2 acts as a random oracle, and the transcript encoding is injective on
  prefixes (`Game/Restoration.lean`, `Layer1/FiatShamir.lean`).
- **The read condition.** Inside the final program, the hash acts as a random oracle and the
  openings follow the WHIR model (`Final/Soundness.lean`).
- **Merkle binding.** A setup leaf that reaches the key root and disagrees with the setup gives a
  Poseidon2 collision (`MerkleBinds`).

## 4. Premises

A claim that takes a premise says nothing about a key, a package or a setup that does not meet it.
`STATUS.md` lists every hypothesis of every claim and says, for each premise, how it is met.
`CompressionLeanTests/Inventory.lean` holds that list, and `lake build` fails when a premise of a
claim has no entry there.

## 5. The ports

Lean transcribes parts of Rust by hand. The transcriptions are of two kinds.

**Ports that can make a claim false.** The claims are about these objects. If one differs from its
Rust, a claim is about a verifier, a program, a file or a check that Rust does not have.

- The shrink verifier, which is the final verifier (`CompressionKey::verify`, `shrink/mod.rs`
  `verify`). Its checks as a game (`Shrink/Game.lean`). The shrink relation and its public value
  (`Shrink/Ccs.lean`) and the batching (`batch`, `Shrink/Rounds.lean`). The matrix value: the
  shape run's total (`Shrink/Evaluate.lean`) plus the closed form of `block_part`
  (`Shrink/BlockPart.lean`, `Shrink/Diagonal.lean`). The layout that these read, the CCS of a run
  and its counts (`Shrink/Layout.lean`, `Run.ccs`, `Run.counts`). The sum-check rounds and the eq
  weights that the game takes from layer 1 (`sumcheck`, `roundValues`, `eqAt`, `eqEval`).
- The circuit programs and their lowering (`Circuit/`), with the parts of `Replay/` that the
  programs read: the data types (`Replay/Data.lean`), the fields (`Replay/Field.lean`), the bar
  transform (`Replay.bar`) and the GKR shape functions (`layerCount`, `layerDegree`,
  `layerFactors`, `partFactors`, `activeTrees`).
- The terminal loader (`Package.load`).
- The setup build and the key expander (`Final/RustSetup.lean`).
- The Merkle hash that `MerkleBinds` reads (`Replay.pathOk`, `merkleRoot`, `compress`,
  `hashLeaf`). The Merkle binding of section 3 gives a Poseidon2 collision only if this is the
  hash that made the key root.
- The error charges (`Layer1.rustDegrees`, `Shrink.rustDegrees`). If one differs from
  `outer_terms` (`lib.rs`) or from `terms` (`shrink/mod.rs`), the premises `p0`, `p1` and `rust`
  are not the checks that `Pcs::new` makes.
- The readers of the files that the Rust tests write (`Layer1Input`, `readRuns` in
  `CompressionLeanExe/SetupCheck.lean`): transcriptions of their formats.

**Code that a cross-check compares with Rust but that cannot make a claim false.** The rest of
`Replay/`: the replay of the layer-1 verifier, with its transcript, stages, formulas, GKR and WHIR.
The claims read it only inside `Reads` (`Final/Encodes.lean`): the draws of a reading are the
replay's draws. The simulation (`Circuit/Sim*.lean`) proves that the circuit's checks give a run of
the replay, and `Bridge/` proves that a run of the replay gives a path of the layer-1 game. So a
claim holds whether or not the replay agrees with Rust's native verifier. `ports.json` lists the
replay, so a change to the Rust items that it transcribes still shows.

These checks compare a transcription with its Rust:

- Rust tests in CI pin digests that Lean computes from the lowered programs, so a change to the
  circuit's instruction stream fails CI.
- `ports.json` names the Rust item that each transcription follows, with a hash of its tokens. The
  CI test `lean_ports` fails when a Rust item changes, until someone re-reads the Lean declaration
  and updates the hash. It detects a change; it does not show that the transcription is right.
  `lake build` fails when a declaration of a port module (`Inventory.portModules`) cites Rust and
  has no entry.
- Ignored Rust tests compare the replay's draws and verdicts, and the programs' instructions, with
  Rust on fixed inputs. They run by hand, not in CI (`scripts/crosschecks.sh`).
- The executables (`CompressionLeanExe/`, which `lake build` builds with warnings as errors) check
  premises of two kinds. `loads` and `valid` are closed propositions about Lean values
  (`circuitPackage ()`). `terminalCheck` checks them by compiled evaluation, because the kernel does
  not evaluate a package of this size. They say nothing about a Rust file. `KeyFacts`, `WidthFits`
  and `setupFailure key pushes = none` are facts about the files that Rust writes. `lowerCheck`,
  `terminalCheck` and `setupCheck` read those files, and a Rust test requires each pass.
  `lowerCheck` also reads the toy key of `lowering_matches_lean`, which has neither production's
  constants nor an application's width, so it prints each result and does not refuse the key.
  `key_facts_of_check`, `width_fits_of_check`, `package_templates_of_check` and
  `setup_facts_of_check` prove that a pass gives the premise. That the executables ran on the
  production artifacts is the evidence of the Rust tests. `STATUS.md` names the executable and the
  test of each premise.

## 6. Scope

The final theorems hold for the relation of an application of any logical width. The terminal package
does not depend on the width: F′'s row builders take it as a parameter but do not use it, and `rfl`
lemmas (`arithmeticRows_at`, `permutationInvocations_at`) state this. `poseidon2_bound` covers the
production key: F′'s relation of the Poseidon2 hash-chain application, at 43,963,750 columns and
814,144 blocks. It takes the premises that `STATUS.md` lists. The claims describe the native verifier
and the circuit that Rust records; they say nothing about code paths that Lean does not model.

## Not trusted

- The claims: they are proved.
- The Rust: it is the object being modeled.
- `.lake/`: build output.
