# What is trusted

Everything that, if wrong, could make a claim of this project false without a Lean error. The claims
are the theorems of `CompressionLean/Claims.lean`, and `CompressionLeanTests/Axioms.lean` pins their
axioms. `lake build` checks that every theorem of the library uses only `propext`, `Classical.choice`
and `Quot.sound`. This file lists every assumption; the README does not repeat them.

A claim means what the definitions that it names mean. `CompressionLeanTests/Vocabulary.lean`
keeps the constants that a claim's type names in `Spec/`, the model and port folders (`Fields`,
`Production`, `Game`, `Layer1`, `Ports`), F′, Mathlib and core. It does not follow
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

This project imports F′'s definitions and theorems: the step and finish packages and their rows, the Poseidon2
specification, the Ajtai setup and the paper algebra. A claim here is only as good as F′'s, and
inherits F′'s own trusted layers.

## 3. Named assumptions

The claims bound a probability. They take these as part of the model, not as Lean axioms:

- **WHIR.** The claims idealize the openings: an opening is accepted only when one listed
  candidate agrees with every opened value (`Layer1.Opens` in the layer-1 game, `Opens` in the
  shrink game, `WhirOpens` in a reading),
  with no error. So `calls / 2^t` does not include WHIR's error. `Final/Budget.lean` and
  `Shrink/Budget.lean` take a WHIR error term, and `Final.OptionB.restore_budget` sets it to zero. In
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
  oracle's answers. For a valid witness Lean proves `Final.OptionB.Encodes`: for each slot, it holds
  an accepting run of the layer-1 verifier, with the program's Poseidon2 challenges, for the slot's
  claim in its finish columns. That such a run implies a true claim is layer 1 under Fiat–Shamir with
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
- **Fields.** `Final.OptionB.restore_sound` is generic in the base field `F` and the challenge
  field `K`. The claims fix `F = Gl` and `K = Cubic`. `lowerCheck` runs `lower` in Goldilocks. In another field the zero tests of constant
  folding can differ from Rust's.

## 4. Premises

A claim that takes a premise says nothing about a key, a package, a setup or runs that do not meet
it. `STATUS.md` lists every hypothesis of every claim and says, for each premise, how it is met: by a
Lean proof, by a check of Rust's files, by a Lean value or by Rust. No premise of
`option_b_poseidon2_bound` is open. `CompressionLeanTests/Inventory.lean` holds that list, and
`lake build` fails when a premise of a claim has no entry there.

## Option B

`option_b_bound` bounds option B's final verifier (`OptionB::verify_final`) at the layer-1 level.
Its claim is a terminal environment of the finish package's columns in which the final program's
checks hold (`Spec.OptionB.FinishTerminal`: the statement and state keys on their words, every row
of the package's pieces, and each Poseidon2 block as the permutation of its lanes), whose two next
slots' claim words are in their layer-1 relations (`Spec.OptionB.Holds`).

`option_b_parent_bound` gives the same bound for the claim in F′'s terms (`Spec.OptionB.ParentHolds`):
a refresh step input whose counter and states are the statement, with `FinishHolds`, and whose two
next slots hold under their circuits' relations (SuperNeo v1.2 Definition 21). It reads the finish
package as the one that F′'s emitter writes (`Spec.OptionB.finishPackage`).

`option_b_last_fold_bound` gives the same bound for the claim in the terms of F′'s IVC walk
(`Spec.OptionB.LastFold`). It is the claim of `option_b_parent_bound` with a nonzero iteration and
the finish checks under the setup's keys. The final program reads the key digests from its state
keys, and the premise `keyWords` makes the state keys the setup's digests.

`option_b_ivc_bound` joins the final verifier with F′'s IVC bound
(`Export.OptionB.IvcSecurity.probability_bound`) in one experiment. The experiment draws an outcome
of the final verifier's experiment (`Game.law`), then the coins of the finish stage's extractor,
then the coins of the reverse extractor's stages. The finish fold is the first stage of the walk.
An answer that extracts it makes the last fold's input a terminal that `Terminal.HoldsFor` accepts
(`FinishWalk.holdsFor_of_extracted`). Three premises are open: HyperNova errata Assumption 1 for
the folds with Poseidon2, the closure premise of the stages, and that the finish stage's adversary
and the chain's IVC adversary are admitted. Both adversaries choose the last fold with
`Classical.epsilon`. The claims prove that a last fold exists. They do not extract one from the
final proof. So "admitted" asks for knowledge soundness of the final verifier, which no claim
proves. The experiment's oracle is independent of Poseidon2 (section 3), while Assumption 1 is
about the folds' Poseidon2 transcripts.

`option_b_poseidon2_bound` meets every premise for the Poseidon2 hash chain. One logical width is
shared by the finish circuit's relation argument and both circuit keys: the key width
`StepRelation.keyWidth = 2^26`, which is at least each step circuit's logical width (the emitter
refuses a package that is wider). The emitter builds the circuits at the key width; the light,
refresh and finish packages are the same bytes as at the earlier census width `2^24`. Each circuit
key's relation is its step package's rows through the package's slot layout, widened to the key
width, then 150 S-box rows for each Poseidon2 block (`Spec.OptionB.keyRelation`). `runsCheck`
compares Rust's runs of each layer-1 relation with these matrices row by row
(`keyRunsFailure`, `Checks.OptionB.keyRunsFailure_sound`). Rust pushes one run for each term of a
row, so the check sorts a row's runs and merges equal runs first, which keeps each column's sum
(`sortedRuns_sum`). `setupCheck` checks each layer-1 setup over its key's own columns,
and `optionBCheck` checks that each key's blocks fit the key width.

The bridge from each slot's layer-1 relation to its CE claim pads. Rust's light relation commits
fewer blocks than the refresh relation, while F′'s option-B model gives both circuit keys one width.
`Final.parent_holds` reads a slot's witness as zero past the key's committed blocks, so `ParentShape`
asks only that the committed blocks fit the width, and `HonestSetup` fixes the Ajtai key only on
the committed blocks.

Three things are trusted, not checked. The emitter writes each package as `packageOf` of its
circuit's pieces, and the checks read the bytes it writes (the `rows`, `blocks`, `start`,
`packed` and `finishPacked` premises). Rust embeds those bytes and pins their identifier. The layer-1
key's root is the honest root of the claim's setup (section 5).

## 5. The ports

Lean transcribes parts of Rust by hand. Every transcription is a definition in `Ports/`, whose
folders follow the Rust modules, except the readers of the files that the Rust tests write
(`CompressionLeanExe/Runs.lean`); the proofs about them are in the folders above. The
transcriptions are of two kinds.

**Ports that can make a claim false.** The claims are about these objects. If one differs from its
Rust, a claim is about a verifier, a program, a file or a check that Rust does not have.

- The shrink verifier, which is the final verifier (`OptionB::verify_final`, `shrink/mod.rs`
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
  coordinate `t` with bit `t` of the index, as `eq_at` and `eq_table` (`field.rs`) do; `ports.json`
  pins it to both.
- The circuit programs and their lowering (`Ports/Circuit/`), with the parts of `Ports/Replay/`
  that the programs read: the data types (`Ports/Replay/Data.lean`), the fields
  (`Ports/Replay/Field.lean`), the bar transform (`Replay.bar`) and the GKR shape functions
  (`layerCount`, `layerDegree`, `layerFactors`, `partFactors`, `activeTrees`).
- Option B's final program (`Ports/OptionB/Program.lean`, `FinalProgram::run` in
  `lifecycle/option_b/final_program.rs`). `optionBCheck` decodes the finish package from its
  artifact (`StepPackage.Package.format`, which round-trips), builds Lean's program with both
  layer-1 keys, and requires every instruction of Rust's run, rows and terms in order
  (`option_b_final_program_matches_lean`, by hand). In CI, `option_b_prefix_digest_is_pinned` pins
  the run up to layer 1; `option_b_program_digest_is_pinned` pins the whole run by hand. Rust writes
  each row port in Lean's form (`side`), so the rows compare term by term.
- The setup build and the key expander (`Ports/Setup.lean`). The build is Rust's slot key while
  there are at most `2^24` run classes and the blocks fit 32 bits; `from_rows` refuses wider keys
  (`slot key width`). The proof reads only `ReadsBack` of the port, and the Rust test
  `derived_structure_steps_read_back_the_runs` checks the same property of `from_rows` on a
  fixture. The Rust tests `lean_setup_vectors_match_rust_and_fips202` and
  `streamed_key_blocks_match_all_lanes_of_authoritative_setup_cases` compare the key expander with
  the vectors that Lean emits. The claims need more than `ReadsBack`: `MerkleBinds` compares an
  honest leaf with `setupAt`'s oracles, so if they are not the oracles under the key's root,
  no valid witness has a reading and the claims say nothing. On each slot's runs, `setupCheck`
  compares the run oracles: `option_b_setup_columns` (`neo-spartan`) requires the runs to derive
  the slot's layer-1 key and writes the four run oracles that its root covers after the key rows,
  and `columnsFailure_sound` proves that a pass makes them `setupAt`'s. The key rows are
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
  proves is `Poseidon2.permute` over F′'s constants. Rust draws its constants from `SEED`.
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
  are not the checks that `Pcs::new` makes. `outer_terms` charges `ring::row_count(shape) - 1`
  for the ring-row batching, and `Layer1.rustLinear` charges `s.rows - 1`, the same sum of
  `kappa`, `matrices` and `publicBlocks`; `ports.json` pins `rustLinear` to `row_count` too.
- The readers of the files that the Rust tests write, which transcribe their formats:
  `Layer1Key`'s JSON (`layer1_input` and `relation_json` write it), and `readRuns` and
  `readColumns` (`CompressionLeanExe/Runs.lean`; `runs_file.rs` and `setup_columns.rs` write
  them). `ports.json` pins each reader to its writers.

**Code that a cross-check compares with Rust but that cannot make a claim false.** The rest of
`Ports/Replay/`: the replay of the layer-1 verifier, with its transcript, stages, formulas, GKR and WHIR.
The claims read it only inside `Reads` (`Spec/OptionBRuns.lean`): the draws of a reading are the
replay's draws. The simulation (`Sim/`) proves that the circuit's checks give a run of
the replay, and `Bridge/` proves that a run of the replay gives a path of the layer-1 game. So a
claim holds whether or not the replay agrees with Rust's native verifier. `ports.json` lists the
replay, so a change to the Rust items that it transcribes still shows.

The entries of the shrink CCS and its relation (`Run.ccs` entries, `Ccs.z`, `Ccs.Relation`) cannot
make a claim false either. The shrink game reads only the sizes and statement words of `Run.ccs`;
its checks use the verifier's matrix value, and the bad sets that read the entries change no check
and no draw. The entries and `z` enter the claims only in the read condition (`Spec.OptionB.RestoreBreak`).
`verifierMatrices_eq` proves that the entries give the matrix value that the verifier computes, and
`shrinkGame_avoids` proves that an accepted transcript has a candidate that satisfies
`Ccs.Relation` on `z`. So the relation is the one that the verifier checks, whatever entries Rust's
prover uses. `ports.json` lists them.

These checks compare a transcription with its Rust:

- Rust tests in CI pin digests that Lean computes from the lowered programs, so a change to the
  circuit's instruction stream fails CI.
- `ports.json` names the Rust item that each transcription follows, with a hash of its tokens. The
  CI test `lean_ports` fails when a Rust item changes, until someone re-reads the Lean declaration
  and updates the hash. It detects a change; it does not show that the transcription is right.
  A Rust constant whose value a transcription writes as a number has its own entry, so a change
  to the value shows too. `lake build` fails when a definition of `Ports/` has neither an entry
  nor a reason in `notPorts` (`CompressionLeanTests/Ports.lean`), when `Ports/` holds a theorem,
  and when an entry names a declaration outside `Ports/` that `portsOutside` does not list. An
  instance that `deriving` makes needs neither: its type has the entry.
- Ignored Rust tests compare the replay's draws and verdicts, the programs' instructions, and the
  toy key's setup root with Rust on fixed inputs. They run by hand, not in CI
  (`scripts/crosschecks.sh`). The final-program check runs on the pinned artifacts with one
  statement; the instructions do not depend on the statement values.
- The Rust tests `relation_shape_has_the_lean_constants` (`neo-spartan`) and
  `final_proof_verifies` (`nightstream`) pin the production constants of
  `Production.lean`. The first also pins `Replay.D = 54`: `ports.json` names the alias `D` of
  `ring.rs`, but the value is in an inline module of `neo-params` that `lean_ports` cannot name.
- The executables (`CompressionLeanExe/`, which `lake build` builds with warnings as errors) check
  premises of two kinds. `fit` is a closed proposition about a Lean value: the finish package that
  `optionBCheck` decodes from its artifact. `optionBCheck` checks it by compiled evaluation, because
  the kernel does not evaluate a package of this size. It says nothing about a Rust file.
  `KeyFacts`, `optionBFits`, `setupFailureAt` and `keyRunsFailure` are facts about the files that
  Rust writes. `lowerCheck`, `optionBCheck`, `setupCheck` and `runsCheck` read those files, and a
  Rust test requires each pass. `setupCheck` and `runsCheck` read the runs with the same reader, so
  their passes are about the same runs. `runsCheck` checks the rows in parallel tasks
  (`Task.spawn`); in the logic a task is the value of its function, and the compiled evaluator runs
  it on a thread. `lowerCheck` also reads the toy key of `lowering_matches_lean`, which has neither
  production's constants nor an application's width, so it prints each result and does not refuse
  the key. `key_facts_of_check`, `option_b_shape_of_check`, `option_b_setup_facts_of_check`,
  `option_b_runs_of_check` and `option_b_logical_of_check` prove that a pass gives the premise. That
  the executables ran on the option-B artifacts is the evidence of the Rust tests. `STATUS.md` names
  the executable and the test of each premise. `setupRootCheck` backs no premise and cannot make a
  claim false: it shows, on the toy key, that Rust's root is `tableRoot` of Rust's own tables
  (read by `RootInput`), which is `honestRoot rel S` for any `(rel, S)` whose setup oracles are
  those tables (`tableRoot_eq`). No Lean setup is compared with the toy key's tables.

## 6. Scope

`option_b_parent_bound` holds for circuit keys of any logical width. `option_b_poseidon2_bound`
covers F′'s light and refresh circuits of the Poseidon2 hash-chain application at the key width
`2^26`. It takes the premises that `STATUS.md` lists. The claims describe the native verifier and
the circuit that Rust records; they say nothing about code paths that Lean does not model.

## Not trusted

- The claims: they are proved.
- The Rust: it is the object being modeled.
- `.lake/`: build output.
