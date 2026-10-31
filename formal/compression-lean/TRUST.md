# What is trusted

Everything that, if wrong, could make a claim of this project false without a Lean error. The claims
are the theorems of `CompressionLean/Claims.lean`, and `CompressionLeanTests/Axioms.lean` pins their
axioms. `lake build` checks that every theorem of the library uses only `propext`, `Classical.choice`
and `Quot.sound`. This file lists every assumption; the README does not repeat them.

A claim means what the definitions that it names mean. `CompressionLeanTests/Vocabulary.lean`
keeps the constants that a claim's type names in `Spec/`, the model and port folders (`Fields`,
`Production`, `Game`, `Terminal`, `Layer1`, `Ports`), F′, Mathlib and core. It does not follow
their bodies. Through them a claim also reads generic definitions of `Lib/` (such as `mle` and
`univ`), `Ccs.Relation` in `Shrink/Ccs.lean`, and these definitions of `Lowering/`: the executor
and the row semantics in `Lowering/Exec.lean` (`Exec.start`, `Exec.step`, `Exec.run`, the state
`Exec.State`, `Bits` and `R1csRow.Holds`), the values of a row's terms and wires in
`Lowering/Rows.lean` (`termsValue`, `Run.wireValue`), the value of a form in `Lowering/Forms.lean`
(`Shrink.Form.evaluate`) and the tape of a witness in `Lowering/Sound.lean` (`tapeOf`). A reader of
the claims reads these too; they sit in files that also hold proofs.

## 1. The Lean kernel and the build cache

The toolchain in `lean-toolchain`. No proof may use `native_decide`: it adds an axiom, and the axiom
audit allows only `propext`, `Classical.choice` and `Quot.sound` in the library and the gates. So no
proof trusts the compiled evaluator. The executables of section 5 do: they
link the claims to Rust's artifacts. The `.olean` files of Mathlib and of
`nightstream-fprime` come from a build cache that `lake build` does not re-check. Re-checking the
environment with `lean4checker`, or building both from source, removes this layer.

## 2. `nightstream-fprime`

This project imports F′'s definitions and theorems: the terminal package and its rows, the Poseidon2
specification, the Ajtai setup and the paper algebra. A claim here is only as good as F′'s, and
inherits F′'s own trusted layers.

## 3. Named assumptions

The claims bound a probability. They take these as part of the model, not as Lean axioms:

- **WHIR.** The claims idealize the openings: an opening is accepted only when one listed
  candidate agrees with every opened value (`Layer1.Opens` in the layer-1 game, `Opens` in the
  shrink game, `WhirOpens` in a reading),
  with no error. So `calls / 2^t` does not include WHIR's error. `Final/Budget.lean` and
  `Shrink/Budget.lean` take a WHIR error term, and `final_restore_budget` sets it to zero. In
  Rust, `Pcs::new` charges WHIR's terms from Plonky3's Johnson-bound report (`pcs.rs`) together
  with the terms that the premises `p0`, `p1` and `rust` state, which Rust meets. No
  Lean proof checks the WHIR figures. This is the weakest numeric layer. The candidates are tables
  over the base field; candidates over `Ext` are not modeled. WHIR's own Fiat–Shamir step is part
  of its error.
- **Fiat–Shamir.** Poseidon2 acts as a random oracle, and the transcript encoding is injective on
  prefixes (`Game/Restoration.lean`, `Game/Compose.lean`). One permutation call can give `k`
  challenges, so `Q` calls and the `R` draws of one path are at most `k·Q + R` queries of the
  experiment. The shrink transcript and the layer-1 transcript share no prefix, so each layer has
  its own log: the shrink sponge starts with its domain words, and the layer-1 sponge starts with
  the digest that the fold transcript hands off. The shape words fix the shrink statement length.
- **The read condition and Fiat–Shamir inside the program.** The bounded event
  (`FinalRestoreBreak`) asks every valid program witness among the shrink candidates for a reading
  whose walk under the layer-1 log accepts. A reading's challenges are the Poseidon2 outputs that
  the program computes (`Reads`). The walk accepts a challenge only where the experiment's oracle
  returned it at that node, and the oracle's answers are uniform and independent of Poseidon2. So
  the condition is not the random-oracle model for the hash inside the program and does not follow
  from it: a prover that computes the challenges with Poseidon2, as the program does, meets it only
  where Poseidon2's outputs equal the oracle's answers. The claims bound a prover with no valid
  program witness (the shrink layer), and a valid witness only where its challenges are the
  oracle's answers. For a valid witness Lean proves `ProgramEncodes`: it holds an accepting run of
  the layer-1 verifier, with the program's Poseidon2 challenges, for the parent claim of its
  terminal assignment. That such a run implies a true claim is layer 1 under Fiat–Shamir with
  Poseidon2 inside the program, which the claims assume and do not bound. The openings inside the
  program follow the WHIR model (`WhirOpens`).
- **Merkle binding.** A setup leaf that reaches the key root at an index below the leaf count
  `2^(n-2)` holds the setup's codeword on that index's coset (`MerkleBinds`). The bound is part
  of the assumption: the path reads only the low `n - 2` bits of an index, so without it the
  assumption would make every setup oracle constant on each queried coset. Under the honest key
  root (`rel.key.root = honestRoot rel S`), `merkleBinds_of_root` (`Bridge/Merkle.lean`) proves
  it for leaves of the replay's shape, a path of `n - 2` four-word digests and `(κ + 4)·8` values,
  unless a leaf passes the path check with a `PathCollision`: two different sixteen-word inputs of
  the permutation that compress alike, or two different leaves of that length that hash alike, at
  inputs computed from the leaf. The shape matters: `compress` and `hashLeaf` take arrays of any
  length, so a padded digest or extra values would match with no collision. A proof's leaves have
  it: the reading decodes four-word digests (`decodeDigest`), and the replay's `readProof` fixes
  the path length and the value count. `honest_leaves_bind` shows that honest leaves pass at their
  own index. That the shipped key's root is `honestRoot` of the claims' setup is not proved
  (section 5).
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
folders follow the Rust modules, except the readers of the files that the Rust tests write
(`CompressionLeanExe/Runs.lean`); the proofs about them are in the folders above. The
transcriptions are of two kinds.

**Ports that can make a claim false.** The claims are about these objects. If one differs from its
Rust, a claim is about a verifier, a program, a file or a check that Rust does not have.

- The shrink verifier, which is the final verifier (`CompressionKey::verify`, `shrink/mod.rs`
  `verify`). Its checks as a game (`Ports/Shrink/Game.lean`), the public value that it computes
  (`Ccs.publicValue`, `Ccs.publicTable`, `Ports/Shrink/Ccs.lean`) and the batching (`batch`,
  `Ports/Shrink/Rounds.lean`). The matrix value: the shape run's total
  (`Ports/Shrink/Evaluate.lean`) plus the closed form of `block_part`
  (`Ports/Shrink/BlockPart.lean`, `Ports/Shrink/Diagonal.lean`). The layout that these read, and
  the sizes, statement words and counts of a run (`Ports/Shrink/Layout.lean`, `Run.ccs`,
  `Run.counts`). The sum-check rounds and the eq weights that the game takes from layer 1
  (`Game.sumcheck`, `Lib.roundValues`, `Lib.eqAt`, `Lib.eqEval`). These four and the two
  readers below are the only `ports.json` entries outside `Ports/` (`portsOutside`): `Layer1/`,
  which sits below `Ports/`, reads these four too. `Lib.eqAt` puts
  coordinate `t` with bit `t` of the index, as `eq_at` and `eq_table` of `neo-verified`
  (`algebra.rs`) do; `ports.json` names the proofs for both. The verifier runs them on `Native`;
  the prover's parallel `eq_table` (`field.rs`) has a test against them.
- The circuit programs and their lowering (`Ports/Circuit/`), with the parts of `Ports/Replay/`
  that the programs read: the data types (`Ports/Replay/Data.lean`), the fields
  (`Ports/Replay/Field.lean`, whose `Word` and `Ext` operations are proved equal to the field of
  `neo-verified` in `CompressionLeanRust/Field.lean`; the verifier's `Native` backend computes with
  that field, and the prover's `Fast` backend with Plonky3's), the bar transform (`Replay.bar`) and the GKR shape functions
  (`layerCount`, `layerDegree`, `layerFactors`, `partFactors`, `activeTrees`).
- The terminal loader (`Terminal.Package.load`). Its rows are F′'s row instantiation
  (`Producer.rows`); Rust pushes the rows of its templates (`expand_matrix`). The final program
  holds every package row in one `r1cs` instruction, so `final_program_matches_lean` compares each
  row and term, in order, on the pinned artifact, by hand; in CI, `final_program_digest_is_pinned`
  pins the run digest of Lean's program. For another package, no check compares the rows.
- The setup build and the key expander (`Ports/Setup.lean`). The build is Rust's slot key while
  there are at most `2^24` run classes and the blocks fit 32 bits; `from_rows` refuses wider keys
  (`slot key width`). The proof reads only `ReadsBack` of the port, and the Rust test
  `derived_structure_steps_read_back_the_runs` checks the same property of `from_rows` on a
  fixture. The Rust tests `lean_setup_vectors_match_rust_and_fips202` and
  `streamed_key_blocks_match_all_lanes_of_authoritative_setup_cases` compare the key expander with
  the vectors that Lean emits. The claims need more than `ReadsBack`: `MerkleBinds` compares an
  honest leaf with `poseidon2Setup`'s oracles, so if they are not the oracles under the key's root,
  no valid witness has a reading and the claims say nothing. On the production runs, `setupCheck`
  compares the run oracles: `production_setup_columns` (`neo-spartan`) requires the runs to derive
  the committed key and writes the four run oracles that its root covers after the key rows, and
  `columnsFailure_sound` proves that a pass makes them `poseidon2Setup`'s. The key rows are
  compared only through the expander vectors above. On the toy key,
  `lean_setup_root_matches_the_rust_key` (`neo-spartan`) writes Rust's setup oracle tables and the
  key root, and `setupRootCheck` requires Lean's `tableRoot` of those tables to be that root;
  `tableRoot_eq` proves that `tableRoot` of tables that hold `setupOracle` is `honestRoot`. So on
  the toy key, Rust's codeword encoding
  (`setup.rs::encode` against `univ`, `leafY` and the leaf order `o·8 + v`), leaf hash and tree are
  Lean's. No check compares the key rows' layout on the cube (`seedKey` against `oracle`), and at
  production size nothing shows that the key's root is `honestRoot`.
- The Merkle hash that `MerkleBinds` reads (`Replay.pathOk`, `merkleRoot`, `compress`,
  `hashLeaf`). The Merkle binding of section 3 gives a Poseidon2 collision only if this is the
  hash that made the key root. Its permutation `Replay.permute` is F′'s `permute64`, which F′
  proves is `Poseidon2.permute` over F′'s constants. The native verifier's `Native::permute` runs
  the permutation of `neo-verified`, which `CompressionLeanRust/Poseidon2.lean` proves equal to
  `Replay.permute`, constant tables included, so the native verifier's sponge and compression run
  the proved permutation too. `CompressionLeanRust/Native.lean` proves the Rust `hash_leaf`,
  `compress` and `merkle_root`, run on `Native`, equal to `Replay.hashLeaf`, `Replay.compress` and
  `Replay.merkleRoot`, and `path_ok` (`setup.rs`) equal to `Replay.pathOk`
  (`CompressionLeanRust/Setup.lean`). The prover, the
  leaf hashes in hints and the trace digest use
  `hash::permutation`, which draws its constants from `SEED`, and
  `verified_permutation_matches_plonky3` (`neo-spartan`) compares the two permutations.
  `exported_round_constants_match_fprime_lean_spec` (`neo-ccs`, `tests/poseidon2_round_constants.rs`)
  compares them with F′'s, and `exported_round_constants_rebuild_the_canonical_permutation` shows
  that they make the workspace permutation; CI runs both (the `neo-ccs` group of
  `scripts/ci_rust_tests.sh`). Also in CI,
  `poseidon2_matches_the_workspace_permutation` (`neo-spartan`) shows that layer 1's permutation
  (`hash::permutation`) is the workspace permutation on 64 states, and the digest pins compare
  outputs: `permutation_template_matches_lean` hashes with F′'s `Poseidon2.hash`, and the run
  digests hash with `Replay.hashLeaf`.
- The error charges (`Layer1.rustDegrees`, `Shrink.rustDegrees`). If one differs from
  `outer_terms` (`relation.rs`) or from `terms` (`shrink/mod.rs`), the premises `p0`, `p1` and `rust`
  are not the checks that `Pcs::new` makes. The integer degrees of both charges are Rust in
  `neo-verified` (`degrees.rs`), proved equal to their ports (`CompressionLeanRust/Degrees.lean`).
  `outer_terms` and `terms`, which turn each degree into an `f64` term, keep a token hash.
- The readers of the files that the Rust tests write, which transcribe their formats:
  `Layer1Key`'s JSON (`layer1_input` and `relation_json` write it), and `readRuns` and
  `readColumns` (`CompressionLeanExe/Runs.lean`; `setup_runs.rs` and `setup_columns.rs` write
  them). `ports.json` pins each reader to its writers.

**Code that a cross-check compares with Rust but that cannot make a claim false.** The rest of
`Ports/Replay/`: the replay of the layer-1 verifier, with its transcript, stages, formulas, GKR and WHIR.
The claims read it only inside `Reads` (`Spec/Bound.lean`): the draws of a reading are the
replay's draws. The simulation (`Sim/`) proves that the circuit's checks give a run of
the replay, and `Bridge/` proves that a run of the replay gives a path of the layer-1 game. So a
claim holds whether or not the replay agrees with Rust's native verifier. `ports.json` lists the
replay, so a change to the Rust items that it transcribes still shows.

The entries of the shrink CCS and its relation (`Run.ccs` entries, `Ccs.z`, `Ccs.Relation`) cannot
make a claim false either. The shrink game reads only the sizes and statement words of `Run.ccs`;
its checks use the verifier's matrix value, and the bad sets that read the entries change no check
and no draw. The entries and `z` enter the claims only in the read condition (`FinalRestoreBreak`).
`verifierMatrices_eq` proves that the entries give the matrix value that the verifier computes, and
`shrinkGame_avoids` proves that an accepted transcript has a candidate that satisfies
`Ccs.Relation` on `z`. So the relation is the one that the verifier checks, whatever entries Rust's
prover uses. `ports.json` lists them.

These checks compare a transcription with its Rust:

- Rust tests in CI pin digests that Lean computes from the lowered programs, so a change to the
  circuit's instruction stream fails CI.
- For a port of Rust in `crates/neo-verified` that has a proof, `ports.json` names two theorems
  in place of a hash. The other ports of `neo-verified` code (`Recorder`, `Native`, the block
  template, `Shrink.Track`, the Replay duplex and transcript, the `Replay.eqAt` family,
  `Replay.setupCheck`, `Replay.leafWeight`, `Replay.monomial`, and the terminal loader) keep a
  hash. Aeneas translates the crate (`scripts/aeneas_extract.sh`), and `CompressionLeanRust` proves
  the translation equal to the port. `Sound` has no premise: if the translated function returns,
  it returns the port's value. `Complete` gives a premise, an input that meets it, and a value on
  every input that meets it. `CompressionLeanRustTests` checks the shape of both. The `aeneas` CI
  job translates the crate again, fails when the committed translation differs, builds the proofs
  and rechecks them with `leanchecker --fresh`. This trusts Charon, Aeneas and the Aeneas models
  of the Rust items in `formal/aeneas/std-models.txt`. The run check
  (`crates/neo-verified/tests/run_check.rs`) runs the translation and each model against Rust on
  fixed inputs in CI.
- `ports.json` names the Rust item that each other transcription follows, with a hash of its tokens. The
  CI test `lean_ports` fails when a Rust item changes, until someone re-reads the Lean declaration
  and updates the hash. It detects a change; it does not show that the transcription is right.
  A Rust constant whose value a transcription writes as a number has its own entry, so a change
  to the value shows too. `lake build` fails when a definition of `Ports/` has neither an entry
  nor a reason in `notPorts` (`CompressionLeanTests/Ports.lean`), when `Ports/` holds a theorem,
  and when an entry names a declaration outside `Ports/` that `portsOutside` does not list. An
  instance that `deriving` makes needs neither: its type has the entry.
- Ignored Rust tests compare the replay's draws and verdicts, the programs' instructions, and the
  toy key's setup root with Rust on fixed inputs. They run by hand, not in CI
  (`scripts/crosschecks.sh`). The terminal checks run on the pinned artifact with one context; the
  instructions do not depend on the statement values, and the context words enter as constants.
- The Rust tests `relation_shape_has_the_lean_constants` (`neo-spartan`) and
  `parent_relation_arguments_are_the_lean_constants` (`nightstream`) pin the production constants of
  `Production.lean`. The first also pins `Replay.D = 54`: `ports.json` names the alias `D` of
  `ring.rs`, but the value is in an inline module of `neo-params` that `lean_ports` cannot name.
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
  the executable and the test of each premise. `setupRootCheck` backs no premise and cannot make a
  claim false: it shows, on the toy key, that Rust's root is `tableRoot` of Rust's own tables
  (read by `RootInput`), which is `honestRoot rel S` for any `(rel, S)` whose setup oracles are
  those tables (`tableRoot_eq`). No Lean setup is compared with the toy key's tables.

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
