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

- Model: the verifier is a public-coin game (`Game/Game.lean`). Each challenge
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
  of the budget (`Final/Budget.lean`), not proved here.
- Claim: `Cost.lean`, `layer1_sound`. An accepted transcript has a P0
  candidate that is a CE(B) witness (`Model.lean`, `Relation`), except with
  probability `layer1Error`. `Final/Budget.lean` bounds it at Rust's split:
  `2^-t` for layer bits `t`, when each WHIR commitment's composed error, as
  Plonky3 charges it, is at most `2^-(t+1)`.
- Fiat–Shamir: `Game/Restoration.lean` lets the prover ask for challenges at any
  node, as often as it wants, and output one path that uses only answers it
  got. `FiatShamir.lean` bounds layer 1 there by `calls · layer1Error`,
  which is at most `calls / 2^t` at Rust's split.
- Verifier formulas: the model takes the public tables at their
  definitions. `ClosedForms.lean` and `EvalK.lean` prove the `mle.rs`
  formulas equal to them. The statement carries the point `r`, and `χ_r` is
  `eq(r, ·)` in `K` (`Model.lean`). `chi` and `eval_k` run in `K ⊗ Ext`, and
  their two parts are the extensions of the re and im tables of `χ_r`.
  `Lib/CubicField.lean` proves that `x³ - x - 1` is irreducible over Goldilocks,
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
  `k·Q + R` queries of `Game/Restoration.lean`. WHIR's own Fiat–Shamir step is
  part of the WHIR error.
- Query indices: the model draws each query leaf as a uniform Goldilocks
  word mod the leaf count, as `sample_bits` does (`HonestStage.lean`).
  `layer1_budget` needs at most 35 setup variables (`query_half`), and the
  replay bridge at most 31. `Relation::new` refuses more than 31: the setup
  check draws a root of unity of order `2^(setupVars + 1)`, and Goldilocks
  has none past `2^32`.
- Setup root: the Merkle root of the setup binds its leaves, so the opened
  setup values are the honest setup's (`HonestStage.lean`).
- Key structure: the verifier reads the matrix column and the slot lane
  tables as step functions of the key's run bounds and slot segments
  (`KeyStructure.lean`). The honest key describes its setup
  (`Structure.Describes`): run `k` of matrix `j` lies between that matrix's
  run bounds, and each slot has its segment's class and lane.
  `Tables::from_rows` (`matrix.rs`) builds the key so. `Structure::check`
  also enforces the order part (`Structure.Checked`) on every key.
- Transcription: the model is written by hand from the Rust verifier. A
  test, not a proof, checks the replay below against Rust. `Bridge/` proves
  that a run the replay accepts is a path of `layer1Game` that draws the
  replay's challenges (`replay_accepts_implies_path`). On a path that
  avoids every bad set, a P0 candidate is a CE(B) witness
  (`replay_accepts_sound`). The bridge assumes WHIR at the opening points
  that the replay computes as Rust does (`WhirOpens`). It assumes that
  each leaf of the proof whose path reaches the key root holds the setup
  codeword on its coset (`MerkleBinds`). A leaf that breaks this gives a
  hash collision. Its premises are the relation facts that the replay does
  not check (`RelationFacts`), such as at most 31 setup variables.
- Charges: Plonky3 composes a commitment's error as its WHIR terms plus each
  charged term times its candidate count. `rustDegrees` (`Ports/Layer1/Budget.lean`) is
  `outer_terms` written in Lean, and `Pcs::new` enforces the two composed
  bounds that `layer1_budget` takes as premises.

## Shrink

`Shrink/` proves the soundness of the shrink verifier
(`crates/neo-spartan/src/shrink/mod.rs`) against an adaptive prover. The
argument is SuperSpartan (ePrint 2023/552) for the CCS that `layout.rs`
defines. The definitions that transcribe the verifier are in `Ports/Shrink/`,
under the same file names; the proofs about them are in `Shrink/`.

- Model: the relation is a CCS (`Ccs.lean`). Its eight matrices are a list
  of entries over `2^m` rows, and `z = (w, 1, x)` has `2^(n+1)` columns. A
  row holds when `(X·z)^7 + Σ_i (A_i·z)·(B_i·z) - C·z = 0`. `Layout.lean`
  builds the CCS of a program's shape run as `layout.rs` places it: the
  Poseidon2 block rows and cells in their wide and narrow parts, then the
  glue rows and cells, then the constant one and the statement words. The
  verifier is a game in the model of `Game/Game.lean` (`Game.lean`): `τ`,
  an outer sum-check of `m` rounds of degree 8 from `0`, the claimed values
  `M̃_j·z(rx)`, `ρ`, an inner sum-check of `n + 1` rounds of degree 2 from
  `Σ_j ρ^j·v_j`, the opening, and the checks of `mod.rs`.
- Verifier formulas: the last check uses the value that `mod.rs` computes
  for `Σ_j ρ^j·M̃_j(rx, ry)`: the shape run's total (`evaluate.rs`) plus the
  closed form `block_part` and its carry walk `diagonal` (`layout.rs`).
  `Evaluate.lean`, `BlockPart.lean` and `Diagonal.lean` write them in Lean
  and prove the value equal to the sum over the CCS entries
  (`verifierMatrices_eq`).
- Adversary: the prover names the program first, because Rust absorbs its
  key and statement before the commitment. It picks every message after it
  sees all earlier challenges. It commits to at most `L` candidates for `w`
  (the WHIR list). Every bad set is a union over these candidates.
- Idealized opening: the verifier accepts the WHIR opening when one
  candidate agrees with the opened value (`Game.lean`, `Opens`). The WHIR
  error of this step is a named term of the budget (`Shrink/Budget.lean`), not
  proved here.
- Claim: `Cost.lean`, `shrink_sound`. An accepted transcript has a
  candidate `w` such that `z = (w, 1, x)` satisfies every row of the named
  program's CCS, except with probability `shrinkError`. Per candidate, the
  error counts `m` bad values for `τ`, `8m` for the outer rounds, `7` for
  `ρ` and `2(n + 1)` for the inner rounds, over `|Ext|`.
- Rust's charge: the count is exactly `L` times the degrees of `terms`
  (`Shrink/Budget.lean`, `shrinkCount_eq_rust`). So when `Pcs::new` accepts the
  configuration for the layer bits `t`, the WHIR error plus `shrinkError`
  is at most `2^-t` (`shrink_budget`).
- Fiat–Shamir: in the state-restoration experiment of `Game/Restoration.lean`,
  the shrink layer errs with probability at most `calls · shrinkError`
  (`shrink_restore_sound`). At Rust's charge this is at most `calls / 2^t`
  (`shrink_restore_budget`).

The shrink layer assumes, and does not prove:

- WHIR: an opening is accepted only when one listed candidate agrees with
  the opened value (`Opens`), except with the opening's error. The
  candidates are tables over the base field; candidates over `Ext` are not
  modeled. The errors come from Plonky3's Johnson-bound report (`pcs.rs`).
- Fiat–Shamir: as in layer 1. Poseidon2 acts as a random oracle, and the
  transcript encoding is injective on prefixes. The shape words fix the
  statement length.
- Recorder: a run is its sink calls that emit glue rows, and a row call
  has at most three products (`Trace::row` asserts it). Every glue wire
  names a cell of its run (`Run.Valid`; `lower_valid` proves it for a
  lowered run), and the template entries name template rows and cells
  (`Template.WellFormed`). The model has one block kind, as `KINDS` has.
- Scope: that this CCS encodes the layer-1 verifier is not proved here.
- Transcription: the model is written by hand from the Rust verifier.

## End to end

`Final/` chains the three layers into one claim about the final verifier
(`CompressionKey::verify`, `finish.rs`).

- Chain: the shrink verifier accepts statement `x`, so a candidate `w`
  satisfies the CCS of program `x`. By the encoding premise, `w` holds a
  terminal assignment for `x` and accepting layer-1 paths for its parent
  claim. Layer 1 then gives a P0 candidate that is a CE(B) witness of that
  claim. `circuitPackage_implies_terminal` gives the F′ meaning of the
  assignment.
- Claim (`Claim.lean`, `TerminalClaim`): some terminal assignment for `x`
  has a parent claim with a CE(B) witness. In that assignment every row
  holds, the public context columns hold the circuit's context, the prior
  state preimage ends with `x`, and the iteration is not zero
  (`TerminalFor`). Layer 1 reads the parent claim from the output columns
  (`claimOf`).
- F′ claim (`ParentClaim.lean`, `ParentClaim`): some terminal assignment
  for `x` holds a PiCCS proof that F′'s verifier accepts, and the PiRLC
  parent that the verifier computes has a witness of F′'s relation
  `Spec.CE.Holds` (SuperNeo v1.2 Definition 21). Under the honest setup,
  the terminal claim gives it (`parentClaim_of_terminalClaim`). `claimOf`
  reads the parent's statement (`claimOf_parent`), and a layer-1 witness of
  that statement is an F′ witness (`Parent.lean`, `parent_holds`): the Ajtai
  commitment, the public input, the norm below `B`, and the Pad and matrix
  evaluations at the parent's point.
- Model (`Game/Compose.lean`): both layers' games run in one experiment. In
  sequence, layer 1 runs and then the shrink layer (`pair`). Under
  Fiat–Shamir, one state-restoration experiment keeps one log per layer
  (`restorePair`). One bad-set argument covers both games, so the proof
  does not build a layer-1 prover from the shrink prover.
- Adversary: as in each layer. The shrink extraction can return any
  candidate that satisfies the program's CCS, so the read condition below
  covers every such candidate.
- Result (`Soundness.lean`): `final_sound` bounds the probability of an
  accepted false claim by `layer1Error + shrinkError`.
  `final_restore_sound` bounds it by `calls · max shrinkError layer1Error`
  for `calls` queries in total. At Rust's split, `final_budget` gives
  `2 / 2^t` with the WHIR errors, and `final_restore_budget` gives
  `calls / 2^t`. Rust takes the layer bits `t` one above the bits that the
  fold leaves, so `2 / 2^t` is the error that the fold leaves.
- F′ result (`ParentClaim.lean`): `final_sound_parent` and
  `final_restore_budget_parent` give the same bounds for the F′ claim. Their
  base field is Goldilocks, and their challenge field is
  `Cubic = Gl[x]/(x³ - x - 1)` with `p³` elements (`Fields.lean`). The
  production bound is `calls / 2^t` under Fiat–Shamir.
- Production result (`Claims.lean`): `production_bound` is
  `final_restore_budget_parent` for the production final program
  (`Sim/Encodes.lean`), where `programEncodes` gives the encoding. Its
  other premises are facts about the key (`KeyFacts`), the terminal package
  and the setup. It holds for the relation of an application of any logical
  width `lw`, with `ParentShape` at `lw`. The terminal package is the same at
  every width: F′'s row builders take the width but do not use it
  (`arithmeticRows_at`, `permutationInvocations_at`). `poseidon2_bound` is
  `production_bound` for F′'s relation of the Poseidon2 hash-chain
  application, the production key's, at 43,963,750 columns. It takes the
  key's block count (`WidthFits`, 814,144 blocks) in place of `ParentShape`,
  and the runs that Rust pushes, the Ajtai setup, `RunsGive` and the pass of
  `setupCheck` in place of the setup and its facts. The setup is
  `poseidon2Setup`, the one that `build` returns for the seed and the runs.
  `Claims.lean` holds the library's claims; `STATUS.md` lists each claim's
  premises and how each one is met.

The chain takes these premises:

- Encoding (`ProgramEncodes`): a witness of the final program's CCS
  holds a terminal assignment for its statement, and every reading of its
  layer-1 part is an accepting layer-1 path for the parent claim of that
  assignment. It stands for `FinalProgram::run`, recorded by `record.rs`
  and placed by `layout.rs`. `programEncodes` (`Sim/Encodes.lean`) proves
  it for the production final program, for a key with `KeyFacts`: the
  files of `Sim/` show that each stage of the circuit computes the replay's
  values, and the bridge turns the accepting replay into a layer-1 path.
- Readings: a reading of a witness takes its statement, proof words and
  challenges from the witness. At each commitment it takes a candidate list
  with a table that agrees with every value that the witness opens there.
  The model commits to candidate lists and the program to Merkle roots, so
  a reading is a relation, not a function.
- Read condition, part of the bounded event: each program witness among
  the shrink candidates has a reading that the layer-1 verifier ran. In
  sequence, the reading ends where layer 1 ended. Under Fiat–Shamir, its
  walk under the layer-1 log accepts. This is the random-oracle model for
  the hash inside the program and the WHIR model for the openings inside
  it.
- Oracles: the shrink transcript and the layer-1 transcript share no
  prefix, so each layer has its own log. The shrink sponge starts with its
  domain words. The layer-1 sponge starts with the digest that the fold
  transcript hands off.
- Honest setup (`Parent.lean`, `HonestSetup`), for the F′ claim: the
  layer-1 setup's key rows are the rows of F′'s Ajtai key, and its runs give
  every entry of every completed CCS matrix of F′'s relation, with row `i`
  in F′'s little-endian order. `Ports/Setup.lean` ports Rust's setup build
  (`build`: `key` and `oracle` in `lib.rs` over `Tables::from_rows` in
  `matrix.rs`). `build_honest` proves `HonestSetup` of its result,
  for the key of the seeded Ajtai setup (`AjtaiSetupV1.Setup.verifierKey`).
  Its parts:
  - Source (`RunsGive`): the runs that `PackageRows::visit_rows`
    pushes name rows below `2^28` and give F′'s matrices. They are the rows
    of F′'s package that F′'s Rust prover and verifier also read. Rust
    visits only rows below the row count, and `Relation::new` checks that
    count against `2^28`. `Checks/PoseidonRows.lean` evaluates the rows of the
    Poseidon2 blocks of F′'s matrix program, 84% of the rows, and proves
    that they give F′'s matrix entries (`matrixSource_entry`).
  - Port: Rust's `from_rows`, `key` and `oracle` are `build`. This is a
    reading of the code. The proof uses only `ReadsBack` of the port, and
    the Rust test `derived_structure_steps_read_back_the_runs` checks the
    same property of `from_rows` on a fixture. The port is Rust's slot key
    while there are at most `2^24` run classes and the blocks fit 32 bits;
    `from_rows` refuses wider keys (`slot key width`).
  - Key expander: Rust's `coefficient_block` at `PRODUCTION_SEED` is
    `AjtaiSetupV1.wideCoefficientNat` at `productionSeed`. The Rust tests
    `lean_setup_vectors_match_rust_and_fips202` and
    `streamed_key_blocks_match_all_lanes_of_authoritative_setup_cases`
    check it against the vectors that Lean emits
    (`nightstream-fprime-ajtai-setup-v1-parity.json`): the seed in full, and
    sample rows and blocks.
  `Structure.Describes` and `Structure.Checked` cover the verifier's copy of
  the run matrices, the slot lengths, ratios and lanes, and their order.
- Shape (`ParentShape`): `parentShape_of` gives it from Rust's sizes for
  `Relation::new`: `D = 54`, six lane variables, `κ = 22`, four matrices,
  five public blocks, the carrier's `⌈lw / 54⌉` blocks and the norm bound
  `H = 17·216·(b - 1) = 3672` (`compression_bound`). `H` is below
  `B = 2^16`, so layer 1's norm check gives F′'s. `Production.lean` names
  each of these constants once, from F′ where F′ has it, with its value. The
  Rust tests `relation_shape_has_the_lean_constants` (`neo-spartan`) and
  `parent_relation_arguments_are_the_lean_constants` (`nightstream`) pin
  these numbers. `KeyFacts` states `κ`, the matrices, the public blocks and
  `H` of a key (`ProductionConstants`); `WidthFits` states the blocks for one
  application.
- Ring: `Φ` is `Φ81 = X^54 + X^27 + 1`, and `bar` is F′'s closed-form bar
  transform (`nativeBarEntry`). Only the manual test
  `neo-math/tests/phi81_bar_lean_parity.rs` compares Rust's runtime bar
  matrix with Lean's.
- Fields: `TerminalClaim` and `final_sound` are generic in the base field
  `F` and the challenge field `K`. The terminal's words enter `F` by
  `Nat.cast`. The F′ results fix `F = Gl` and `K = Cubic`.
- Each layer's assumptions, as listed above.

## Lowering

`Ports/Circuit/` and `Lowering/` give `ProgramEncodes`. A program is Lean data, and
Lean lowers it to the shrink run as `record.rs` does.

- Model (`Ir.lean`): a program is a flat list of instructions over value
  handles (`Instr`). An instruction is one `Backend` call that `record.rs`
  lowers by itself. Handle `v` is the `v`-th value of the run. The program
  cannot read a value, so no instruction depends on one. `Build` writes the
  list.
- Lowering (`Record.lean`): `lower` is a port of `record.rs`. It merges terms
  in Rust's order, keeps a form of at most four terms, folds constants, and
  numbers the glue cells, blocks, statement words and rows as Rust does. It
  computes in a field `F`.
- Claim (`Lowering/Valid.lean`): `lower_valid`. Every glue row of `lower L stmt` names
  only its own glue cells, the cells of its blocks, the constant one and its
  statement words (`Run.Valid`), for a template of at least 182 cells.
- Claim (`Lowering/Sound.lean`): `lower_sound`. A witness of the CCS of `lower L stmt`
  with the Poseidon2 template gives an accepting run of the executor `Exec`
  (`Lowering/Exec.lean`) on the tape read out of the witness (`tapeOf`): every
  `assertZero` value is zero, every `bits` decomposition is boolean, sums to
  its value and is canonical at 64 bits, every `extInv` output is an inverse,
  every `r1cs` row holds, and every block computes Poseidon2.
- `select` lowers to `whenZero + bit·(whenOne - whenZero)`, as `record.rs`
  records it. `Native::select` branches. The two agree when `bit` is 0 or 1.
- Verifier (`Algebra.lean`, `Hash.lean`, `Fold.lean`, `Sumcheck.lean`,
  `Views.lean`, `Gkr.lean`, `Early.lean`, `Ring.lean`, `Mle.lean`,
  `Linear.lean`, `Honest.lean`, `Late.lean`, `Whir.lean`, `Openings.lean`,
  `Verifier.lean`): the layer-1 verifier written with `Build`, each function
  in the call order of its Rust counterpart (`algebra.rs`, `hash.rs`,
  `fold.rs`, `sumcheck.rs`, the proof views, `gkr.rs`, the early and late
  stages' parts of `norm.rs`, `mle.rs` and `matrix.rs`, `ring.rs`,
  `setup.rs`, `whir/mod.rs`, `verifier.rs`). The sizes come from the replay's
  relation (`Ports/Replay/Data.lean`); what the WHIR verifier reads of each `Pcs`
  beyond the replay's `WhirShape` is a `WhirKey`. `verify` is all of
  `verifier.rs`: the seed, the relation words and the statement words,
  then the early stage (norm root, GKR, matrix leaf checks, early values),
  the linear stage (`verifier.rs:170-206`), the honest fold
  (`verifier.rs:208-249`: setup partials, `α`, setup queries), the late
  stage (`verifier.rs:251-260`: late `β`, GKR, lane tables, block scatter
  leaf checks, late values), the query points (`verifier.rs:262-265`) and
  the two WHIR openings with their equality checks (`verifier.rs:266-315`).

The lowering assumes, and does not prove:

- Port: that `lower` equals `record.rs` on every program. The check below
  compares the two on nine runs.
- Verifier port: that Lean's programs write Rust's instructions for every
  relation. The check below compares them on the toy relation and on the
  production key's relation.
- Fields: the check runs `lower` in Goldilocks. In another field, the zero
  tests of constant folding can differ from Rust's.

Build and emit (each Lean command under the 1,500 s cap). Copy the F′ build
cache into `.lake` first (`cp -cR ../nightstream-fprime/.lake/packages .lake/`)
so that Lake does not rebuild Mathlib.

```bash
timeout --signal=KILL 1500 lake build
timeout --signal=KILL 1500 .lake/build/bin/emitTerminal artifacts/compression-terminal-v1.json
```

`lake build` also builds the executables (`CompressionLeanExe/`: the
checks, their readers and the Lean side of the Rust test programs) with
warnings as errors, and runs the axiom audit
(`CompressionLeanTests/Axioms.lean`), the ports gate
(`CompressionLeanTests/Ports.lean`) and the folder gate
(`CompressionLeanTests/Layers.lean`): a module imports only its own folder
and lower ones, in the order `Lib`, `Fields`, `Production`, `Game`,
`Terminal`, `Layer1`, `Ports`, `Shrink`, `Lowering`, `Bridge`, `Final`,
`Checks`, `Sim`, `Claims`.

Ports. `Ports/` holds every hand transcription of Rust and nothing else: its
folders follow the Rust modules (`Ports/Circuit/` for `circuit/` and the
verifier programs, `Ports/Shrink/` for `shrink/`, `Ports/Replay/`,
`Ports/Terminal/`, `Ports/Setup.lean`, `Ports/Layer1/Budget.lean`), and a
port's namespace is its path without `Ports/`. `ports.json` pairs each Lean
port of Rust with the Rust item that it transcribes and a hash of the item's
tokens. The ports gate fails on a theorem in `Ports/`, on a declaration
outside its path's namespace, and on a definition of `Ports/` with neither a
`ports.json` entry nor a reason in `notPorts`. The CI test `lean_ports`
(`nightstream`) fails when an item changes and names the Lean declaration to
re-read. `STATUS.md` lists the claims with their axioms and hypotheses, how each
premise is met, the ports and the digest pins. `lake exe status > STATUS.md`
writes it. `lake build` fails when it is out of date, and when a premise of a
claim has no entry in `Inventory.premises`.

`scripts/crosschecks.sh` runs the replay, lowering, production, terminal,
full and setup checks below, in that order.

Replay check. `Ports/Replay/` is an executable layer-1 verifier in the model's
stage order. It draws every challenge as Rust does and runs every check of
`verifier.rs` except WHIR's own: it replays each WHIR transcript and takes the
opened values as `Opens` does. Build it with
`timeout --signal=KILL 1500 lake build replayLayer1`, then run
`cargo test --release -p neo-spartan --lib lean_replay -- --ignored` under the
5-minute cap. The test proves a toy claim, writes the honest proof and
tampered variants to `target/lean-replay/layer1.json`, and requires the
replay to draw the same challenges and give the same verdict as Rust. When
Rust rejects inside WHIR, the replay must reject later.

Lowering check. Build it with
`timeout --signal=KILL 1500 lake build lowerCheck`, then run
`cargo test --release -p neo-spartan --lib lowering_matches_lean -- --ignored`
under the 5-minute cap. The test records nine programs through
`record.rs` and logs each `Backend` call as an instruction: `Edges`, which
takes every branch of `record.rs`, `Shared`, which calls every helper of the
verifier, `Early`, which runs the early stage's helpers on a structure that
the toy relation does not have, `Linear`, which runs the linear stage on the
toy proof from the duplex state after the early stage, `Honest`, which runs
the honest fold and the query points on the toy proof from the state before
`μ`, `Late`, which runs the late stage on the toy proof from the state after
the setup queries, `LaneSplit`, which builds the lane tables of run classes
that the toy relation does not have, `Openings`, which runs both WHIR
openings on the toy proof from the state after the query points, and the toy
layer-1 verifier of `layer1_verifier_shrinks`. The programs are in
`tests/internal/lowering_stages.rs`. The test
writes the statement, the IR and Rust's trace to
`target/lean-lowering/<program>.txt`, the verifier's relation and WHIR keys
to `layer1.json` and the inputs of `Linear`, `Honest`, `Late` and
`Openings` to `linear.json`, `honest.json`, `late.json` and
`openings.json`. `lowerCheck` lowers the IR and
requires the same glue cells, blocks, statement words and rows, side by side
and term by term. For every program but `Edges` it also writes Lean's
program and requires all of Rust's instructions in order. On the toy
relation the verifier is 651,165 instructions.

Production check. `production_layer1_matches_lean` (same crate, ignored)
reads the production compression key (`nightstream`'s committed
`tests/fixtures/compression/key.bin`),
records the verifier's shape run (zero statement, no proof) and checks it as
`layer1` above. It writes `production_layer1.txt` and
`production_layer1.json`. The verifier is 1,928,933 instructions and
1,280,147 rows; the test takes about 12 s and 1.7 GB.

Setup check. `production_runs_rebuild_the_layer1_key` (`nightstream`,
ignored) writes the runs that `PackageRows::visit_rows` pushes for the
production circuit to `target/lean-lowering/production_runs.bin` and
requires `Key::derive` of the file to be the committed key's layer-1 key. It
writes 38,533,993 runs, about 1 GB, and takes about 94 s and 9.7 GB. Build
the check with `timeout --signal=KILL 1500 lake build setupCheck`, then run
`production_setup_matches_lean` (same crate, ignored) under the 5-minute cap.
It runs `setupCheck` on that file and on the `production_layer1.json` of
`production_layer1_matches_lean` (run both first) and requires
`setup facts hold`. `setupFailure` (`Checks/SetupFacts.lean`) decides the
named facts of `setupChecks`, from which `setupFailure_sound` proves that
`build` returns `poseidon2Setup` for the runs, that it has `Setup.Fits` and
that the key's structure describes it. It reads the runs a fixed number of
times: the classes and the slot keys go through a hash set and a sort of the
distinct values (`sortedValues`), which is `sortedSet`. The test takes about
48 s and 5.5 GB.

Digest pins. CI runs no Lean, so Rust pins the run digest that `lowerCheck`
computes from Lean's lowered program (`runDigest`, `digest` on each `match`
line): Poseidon2's leaf sponge of the glue cell, block, statement word and
row counts, then every row's sides as term counts, wires and coefficients.
`toy_layer1_digest_is_pinned` and `production_layer1_digest_is_pinned` (both
CI, the second on the committed key) record the verifier's shape run in Rust and
require the pinned value, so a change to the program or to `record.rs` that
changes a row fails them. To update a pin, run its Lean check
(`lowering_matches_lean` or `production_layer1_matches_lean`), and only when
it passes copy the `digest` of its `layer1` line into `lowering.rs`.
`nightstream`'s `terminal_digest_is_pinned` and
`final_program_digest_is_pinned` (both CI) pin the final program the same way,
up to layer 1 and whole: run `terminal_program_matches_lean` or
`final_program_matches_lean`, and only when it passes copy the `digest` of its
`match` line into `terminal_ir.rs`.

## Terminal program

`Ports/Terminal/Program.lean` writes the final program up to layer 1 as a builder
program: the wrapper of `FinalProgram::run`
(`crates/nightstream/src/lifecycle/terminal/mod.rs`) and `Package::apply`
(`terminal/package.rs`). The layer-1 verifier is a parameter of
`finalProgram`. It receives the claim and the transcript state that the
package's output columns hold.

- Package (`Ports/Terminal/Package.lean`): `load` is a port of Rust's `load`. It
  takes F′'s row instantiation (`RowSemantics.lean`) of every producer in
  Rust's schedule order, writes each side as `finish_row` does, keeps the
  columns that some row names and gives each column its source. It runs on
  `circuitPackage ()`, not on the artifact.
- Check: `terminalCheck` requires the builder's instructions to be Rust's,
  one by one: every column source, every row and term of the system, and the
  claim and state handles. Build it with
  `timeout --signal=KILL 1500 lake build terminalCheck`, then run
  `cargo test --release -p nightstream --lib terminal_program_matches_lean -- --ignored`
  under the 5-minute cap. The test logs `FinalProgram::run` with a toy
  layer-1 relation whose shape the claim does not fit, so the log stops at
  `verify_with`. It writes `target/lean-lowering/terminal.txt` (about
  800 MB). The test takes about 130 s and 22 GB.
- Full check: with a second argument, the layer-1 key file, `terminalCheck`
  writes `finalProgram` with `verifyWith` (`Verifier.lean`) and requires
  every instruction. `final_program_matches_lean` (same crate, ignored) logs
  the shape run of `FinalProgram::run` on the production compression key and
  package and passes the `production_layer1.json` of
  `production_layer1_matches_lean` (run it first). The final program is
  1,970,210 instructions; the test takes about 140 s and 23 GB.

The terminal program assumes, and does not prove:

- Rows: that the system's rows hold, with column one at one, exactly when
  `CircuitPackage.RowsHold` holds. The rows are F′'s instantiation,
  reordered and normalized, so this is a Lean proof about `load`. The check
  shows only that Rust's rows are these rows.
- Scope: the checks run on the pinned artifact with one context. The
  instructions do not depend on the statement values, and the context words
  enter as constants.
