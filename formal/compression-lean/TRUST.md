# What is trusted

Everything that, if wrong, could make a claim of this project false without a Lean error. The claims
are the theorems of `CompressionLean/Claims.lean`, and `CompressionLeanTests/Axioms.lean` pins their
axioms. `lake build` checks that every theorem of the library uses only `propext`, `Classical.choice`
and `Quot.sound`. This file lists every assumption; the README does not repeat them.

A claim means what the definitions that it names mean. `CompressionLeanTests/Vocabulary.lean`
keeps those in `Spec/`, the model and port folders (`Fields`, `Production`, `Game`, `Terminal`,
`Layer1`, `Ports`), F′, Mathlib and core, so a reader of the claims reads no proof.

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
  report in Rust (`pcs.rs`) and enter `Final/Budget.lean` and `Shrink/Budget.lean` as given. No
  Lean proof checks them. This is the weakest numeric layer. The candidates are tables over the base
  field; candidates over `Ext` are not modeled. WHIR's own Fiat–Shamir step is part of its error.
- **Fiat–Shamir.** Poseidon2 acts as a random oracle, and the transcript encoding is injective on
  prefixes (`Game/Restoration.lean`, `Game/Compose.lean`). One permutation call can give `k`
  challenges, so `Q` calls and the `R` draws of one path are at most `k·Q + R` queries of the
  experiment. The shrink transcript and the layer-1 transcript share no prefix, so each layer has
  its own log: the shrink sponge starts with its domain words, and the layer-1 sponge starts with
  the digest that the fold transcript hands off. The shape words fix the shrink statement length.
- **The read condition.** Inside the final program, the hash acts as a random oracle and the
  openings follow the WHIR model (`Final/Soundness.lean`).
- **Merkle binding.** A setup leaf that reaches the key root and disagrees with the setup gives a
  Poseidon2 collision (`MerkleBinds`).
- **Query indices.** The model draws each query leaf as a uniform Goldilocks word modulo the leaf
  count, as `sample_bits` does (`Layer1/HonestStage.lean`, `Layer1/Queries.lean`). The budget needs
  at most 35 setup variables (`query_half`), and the replay bridge at most 31. `Relation::new`
  refuses more than 31: the setup check draws a root of unity of order `2^(setupVars + 1)`, and
  Goldilocks has none past `2^32`.
- **Key structure.** The model reads the matrix column and slot lane tables as step functions of the
  key's run bounds and slot segments (`Layer1/KeyStructure.lean`). That the key describes its setup
  (`Structure.Describes`) and the order part that `Structure::check` enforces (`Structure.Checked`)
  are premises (section 4).
- **Recorder.** A shrink run is the sink calls that emit glue rows, and a row call has at most
  three products (`Trace::row` asserts it). The model has one block kind, as `KINDS` has.
- **Ring.** `Φ` is `Φ81 = X^54 + X^27 + 1`, and `bar` is F′'s closed-form bar transform
  (`nativeBarEntry`). Only the manual test `neo-math/tests/phi81_bar_lean_parity.rs` compares Rust's
  runtime bar matrix with Lean's.
- **Fields.** `TerminalClaim` and `final_restore_sound` are generic in the base field `F` and the
  challenge field `K`; the terminal's words enter `F` by `Nat.cast`. The claims fix `F = Gl` and
  `K = Cubic`. `lowerCheck` runs `lower` in Goldilocks. In another field the zero tests of constant
  folding can differ from Rust's.

## 4. Premises

A claim that takes a premise says nothing about a key, a package, a setup or runs that do not meet
it. `STATUS.md` lists every hypothesis of every claim and says, for each premise, how it is met: by a
Lean proof, by a check of Rust's files, by a Lean value or by Rust. No premise of `poseidon2_bound`
is open. `CompressionLeanTests/Inventory.lean` holds that list, and `lake build` fails when a premise
of a claim has no entry there.

## 5. The ports

Lean transcribes parts of Rust by hand. Every transcription is a definition in `Ports/`, whose
folders follow the Rust modules; the proofs about them are in the folders above. The
transcriptions are of two kinds.

**Ports that can make a claim false.** The claims are about these objects. If one differs from its
Rust, a claim is about a verifier, a program, a file or a check that Rust does not have.

- The shrink verifier, which is the final verifier (`CompressionKey::verify`, `shrink/mod.rs`
  `verify`). Its checks as a game (`Ports/Shrink/Game.lean`). The shrink CCS and its public value
  (`Ports/Shrink/Ccs.lean`), with the relation over it (`Ccs.Relation`, `Shrink/Ccs.lean`), and
  the batching (`batch`, `Ports/Shrink/Rounds.lean`). The matrix value: the shape run's total
  (`Ports/Shrink/Evaluate.lean`) plus the closed form of `block_part`
  (`Ports/Shrink/BlockPart.lean`, `Ports/Shrink/Diagonal.lean`). The layout that these read, the
  CCS of a run and its counts (`Ports/Shrink/Layout.lean`, `Run.ccs`, `Run.counts`). The sum-check
  rounds and the eq weights that the game takes from layer 1 (`Game.sumcheck`, `Lib.roundValues`,
  `Lib.eqAt`, `Lib.eqEval`). These four are the only `ports.json` entries outside `Ports/`
  (`portsOutside`): `Layer1/`, which sits below `Ports/`, reads them too.
- The circuit programs and their lowering (`Ports/Circuit/`), with the parts of `Ports/Replay/`
  that the programs read: the data types (`Ports/Replay/Data.lean`), the fields
  (`Ports/Replay/Field.lean`), the bar transform (`Replay.bar`) and the GKR shape functions
  (`layerCount`, `layerDegree`, `layerFactors`, `partFactors`, `activeTrees`).
- The terminal loader (`Terminal.Package.load`).
- The setup build and the key expander (`Ports/Setup.lean`). The build is Rust's slot key while
  there are at most `2^24` run classes and the blocks fit 32 bits; `from_rows` refuses wider keys
  (`slot key width`). The proof reads only `ReadsBack` of the port, and the Rust test
  `derived_structure_steps_read_back_the_runs` checks the same property of `from_rows` on a
  fixture. The Rust tests `lean_setup_vectors_match_rust_and_fips202` and
  `streamed_key_blocks_match_all_lanes_of_authoritative_setup_cases` compare the key expander with
  the vectors that Lean emits.
- The Merkle hash that `MerkleBinds` reads (`Replay.pathOk`, `merkleRoot`, `compress`,
  `hashLeaf`). The Merkle binding of section 3 gives a Poseidon2 collision only if this is the
  hash that made the key root.
- The error charges (`Layer1.rustDegrees`, `Shrink.rustDegrees`). If one differs from
  `outer_terms` (`lib.rs`) or from `terms` (`shrink/mod.rs`), the premises `p0`, `p1` and `rust`
  are not the checks that `Pcs::new` makes.
- The readers of the files that the Rust tests write (`Layer1Key`, `readRuns` in
  `CompressionLeanExe/Runs.lean`): transcriptions of their formats.

**Code that a cross-check compares with Rust but that cannot make a claim false.** The rest of
`Ports/Replay/`: the replay of the layer-1 verifier, with its transcript, stages, formulas, GKR and WHIR.
The claims read it only inside `Reads` (`Spec/Bound.lean`): the draws of a reading are the
replay's draws. The simulation (`Sim/`) proves that the circuit's checks give a run of
the replay, and `Bridge/` proves that a run of the replay gives a path of the layer-1 game. So a
claim holds whether or not the replay agrees with Rust's native verifier. `ports.json` lists the
replay, so a change to the Rust items that it transcribes still shows.

These checks compare a transcription with its Rust:

- Rust tests in CI pin digests that Lean computes from the lowered programs, so a change to the
  circuit's instruction stream fails CI.
- `ports.json` names the Rust item that each transcription follows, with a hash of its tokens. The
  CI test `lean_ports` fails when a Rust item changes, until someone re-reads the Lean declaration
  and updates the hash. It detects a change; it does not show that the transcription is right.
  `lake build` fails when a definition of `Ports/` has neither an entry nor a reason in
  `notPorts` (`CompressionLeanTests/Ports.lean`), when `Ports/` holds a theorem, and when an
  entry names a declaration outside `Ports/` that `portsOutside` does not list.
- Ignored Rust tests compare the replay's draws and verdicts, and the programs' instructions, with
  Rust on fixed inputs. They run by hand, not in CI (`scripts/crosschecks.sh`). The terminal checks
  run on the pinned artifact with one context; the instructions do not depend on the statement
  values, and the context words enter as constants.
- The Rust tests `relation_shape_has_the_lean_constants` (`neo-spartan`) and
  `parent_relation_arguments_are_the_lean_constants` (`nightstream`) pin the production constants of
  `Production.lean`.
- The executables (`CompressionLeanExe/`, which `lake build` builds with warnings as errors) check
  premises of two kinds. `loads` and `valid` are closed propositions about Lean values
  (`circuitPackage ()`). `terminalCheck` checks them by compiled evaluation, because the kernel does
  not evaluate a package of this size. They say nothing about a Rust file. `KeyFacts`, `WidthFits`,
  `setupFailure key pushes = none` and `runsFailure pushes = none` are facts about the files that
  Rust writes. `lowerCheck`, `terminalCheck`, `setupCheck` and `runsCheck` read those files, and a
  Rust test requires each pass. `setupCheck` and `runsCheck` read the runs with the same reader, so
  their passes are about the same runs. `runsCheck` checks the rows in parallel tasks
  (`Task.spawn`); in the logic a task is the value of its function, and the compiled evaluator runs
  it on a thread. `lowerCheck` also reads the toy key of `lowering_matches_lean`, which has neither
  production's constants nor an application's width, so it prints each result and does not refuse
  the key. `key_facts_of_check`, `width_fits_of_check`, `package_templates_of_check`,
  `setup_facts_of_check` and `runsGive_of_check` prove that a pass gives the premise. That the
  executables ran on the production artifacts is the evidence of the Rust tests. `STATUS.md` names
  the executable and the test of each premise.

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
