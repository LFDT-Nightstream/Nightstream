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
- Transcription: the model is written by hand from the Rust verifier. The
  replay below is checked against Rust; that its acceptance gives the
  model's conditions is not proved.
- Charges: Plonky3 composes a commitment's error as its WHIR terms plus each
  charged term times its candidate count. `rustDegrees` (`Budget.lean`) is
  `outer_terms` written in Lean, and `Pcs::new` enforces the two composed
  bounds that `layer1_budget` takes as premises.

## Shrink

`Shrink/` proves the soundness of the shrink verifier
(`crates/neo-spartan/src/shrink/mod.rs`) against an adaptive prover. The
argument is SuperSpartan (ePrint 2023/552) for the CCS that `layout.rs`
defines.

- Model: the relation is a CCS (`Ccs.lean`). Its eight matrices are a list
  of entries over `2^m` rows, and `z = (w, 1, x)` has `2^(n+1)` columns. A
  row holds when `(X·z)^7 + Σ_i (A_i·z)·(B_i·z) - C·z = 0`. `Layout.lean`
  builds the CCS of a program's shape run as `layout.rs` places it: the
  Poseidon2 block rows and cells in their wide and narrow parts, then the
  glue rows and cells, then the constant one and the statement words. The
  verifier is a game in the model of `Layer1/Game.lean` (`Game.lean`): `τ`,
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
  error of this step is a named term of the budget (`Budget.lean`), not
  proved here.
- Claim: `Cost.lean`, `shrink_sound`. An accepted transcript has a
  candidate `w` such that `z = (w, 1, x)` satisfies every row of the named
  program's CCS, except with probability `shrinkError`. Per candidate, the
  error counts `m` bad values for `τ`, `8m` for the outer rounds, `7` for
  `ρ` and `2(n + 1)` for the inner rounds, over `|Ext|`.
- Rust's charge: the count is exactly `L` times the degrees of `terms`
  (`Budget.lean`, `shrinkCount_eq_rust`). So when `Pcs::new` accepts the
  configuration for the layer bits `t`, the WHIR error plus `shrinkError`
  is at most `2^-t` (`shrink_budget`).
- Fiat–Shamir: in the state-restoration experiment of `Restoration.lean`,
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
- Model (`Compose.lean`): both layers' games run in one experiment. In
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

The chain takes these premises:

- Encoding (`ProgramEncodes`, open): a witness of the final program's CCS
  holds a terminal assignment for its statement, and every reading of its
  layer-1 part is an accepting layer-1 path for the parent claim of that
  assignment. It stands for `FinalProgram::run`, recorded by `record.rs`
  and placed by `layout.rs`. No Lean proof of it exists.
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
  in F′'s little-endian order. Rust builds the setup in `lib.rs` (`key`,
  `oracle`) from `Tables::from_rows` (`matrix.rs`). `Structure.Describes`
  and `Structure.Checked` cover the run matrices, the slot lengths, ratios
  and lanes, and their order. They do not cover the key rows, the run rows,
  the run coefficients or the slot blocks. That Rust's tables and seeded
  key are these is not proved here.
- Shape (`ParentShape`): `Relation::new` gets `D = 54`, six lane variables,
  `κ = 22`, four matrices, five public blocks and the carrier's blocks. Its
  norm bound `H = 17·216·(b - 1) = 3672` (`compression_bound`) is below
  `B = 2^16`, so layer 1's norm check gives F′'s.
- Ring: `Φ` is `Φ81 = X^54 + X^27 + 1`, and `bar` is F′'s closed-form bar
  transform (`nativeBarEntry`). Only the manual test
  `neo-math/tests/phi81_bar_lean_parity.rs` compares Rust's runtime bar
  matrix with Lean's.
- Fields: `TerminalClaim` and `final_sound` are generic in the base field
  `F` and the challenge field `K`. The terminal's words enter `F` by
  `Nat.cast`. The F′ results fix `F = Gl` and `K = Cubic`.
- Each layer's assumptions, as listed above.

## Lowering

`Circuit/` is the first step toward `ProgramEncodes`. A program is Lean
data, and Lean lowers it to the shrink run as `record.rs` does.

- Model (`Ir.lean`): a program is a flat list of instructions over value
  handles (`Instr`). An instruction is one `Backend` call that `record.rs`
  lowers by itself. Handle `v` is the `v`-th value of the run. The program
  cannot read a value, so no instruction depends on one. `Build` writes the
  list.
- Lowering (`Record.lean`): `lower` is a port of `record.rs`. It merges terms
  in Rust's order, keeps a form of at most four terms, folds constants, and
  numbers the glue cells, blocks, statement words and rows as Rust does. It
  computes in a field `F`.
- Claim (`Valid.lean`): `lower_valid`. Every glue row of `lower L stmt` names
  only its own glue cells, the cells of its blocks, the constant one and its
  statement words (`Run.Valid`), for a template of at least 182 cells.
- `select` lowers to `whenZero + bit·(whenOne - whenZero)`, as `record.rs`
  records it. `Native::select` branches. The two agree when `bit` is 0 or 1.
- Verifier (`Algebra.lean`, `Hash.lean`, `Fold.lean`, `Sumcheck.lean`,
  `Views.lean`, `Gkr.lean`, `Early.lean`, `Ring.lean`, `Mle.lean`,
  `Linear.lean`, `Honest.lean`, `Late.lean`, `Verifier.lean`): the layer-1
  verifier written with `Build`, each function in the call order of its Rust
  counterpart (`algebra.rs`, `hash.rs`, `fold.rs`, `sumcheck.rs`, the proof
  views, `gkr.rs`, the early and late stages' parts of `norm.rs`, `mle.rs`
  and `matrix.rs`, `ring.rs`, `setup.rs`, `verifier.rs`). The sizes come
  from the replay's relation (`Replay/Data.lean`). `verify` is ported up to
  `verifier.rs:249`: the seed, the relation words and the statement words,
  then the early stage (norm root, GKR, matrix leaf checks, early values),
  the linear stage (`verifier.rs:170-206`) and the honest fold
  (`verifier.rs:208-249`: setup partials, `α`, setup queries). `lateStage`
  is the late stage (`verifier.rs:251-260`: late `β`, GKR, lane tables,
  block scatter leaf checks, late values); `verify` does not call it yet,
  and `queryPoint` (`setup::query_point`) waits for its wiring.

The lowering assumes, and does not prove:

- Port: that `lower` equals `record.rs` on every program. The check below
  compares the two on eight runs.
- Verifier port: that Lean's programs write Rust's instructions for every
  relation. The check below compares them on one toy relation.
- Fields: the check runs `lower` in Goldilocks. In another field, the zero
  tests of constant folding can differ from Rust's.
- Encoding: that a witness of a lowered run's CCS gives an accepting run of
  the program.

Build and emit (each Lean command under the 1,500 s cap). Copy the F′ build
cache into `.lake` first (`cp -cR ../nightstream-fprime/.lake/packages .lake/`)
so that Lake does not rebuild Mathlib.

```bash
timeout --signal=KILL 1500 lake build
timeout --signal=KILL 1500 lake build emitTerminal
timeout --signal=KILL 1500 .lake/build/bin/emitTerminal artifacts/compression-terminal-v1.json
```

`lake build` also runs the axiom audit (`CompressionLeanTests/Axioms.lean`).

Replay check. `Replay/` is an executable layer-1 verifier in the model's
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
under the 5-minute cap. The test records eight programs through
`record.rs` and logs each `Backend` call as an instruction: `Edges`, which
takes every branch of `record.rs`, `Shared`, which calls every helper of the
verifier, `Early`, which runs the early stage's helpers on a structure that
the toy relation does not have, `Linear`, which runs the linear stage on the
toy proof from the duplex state after the early stage, `Honest`, which runs
the honest fold and the query points on the toy proof from the state before
`μ`, `Late`, which runs the late stage on the toy proof from the state after
the setup queries, `LaneSplit`, which builds the lane tables of run classes
that the toy relation does not have, and the toy layer-1 verifier of
`layer1_verifier_shrinks`. It
writes the statement, the IR and Rust's trace to
`target/lean-lowering/<program>.txt`, the verifier's relation to
`layer1.json` and the inputs of `Linear`, `Honest` and `Late` to
`linear.json`, `honest.json` and `late.json`. `lowerCheck` lowers the IR and
requires the same glue cells, blocks, statement words and rows, side by side
and term by term. For `Shared`, `Early`, `Linear`, `Honest`, `Late`,
`LaneSplit` and the verifier it also writes Lean's program and requires
Rust's instructions in order: all of the first six programs' instructions,
and the verifier's first ones, through `verifier.rs:249`.

## Terminal program

`Circuit/Terminal.lean` writes the final program up to layer 1 as a builder
program: the wrapper of `FinalProgram::run`
(`crates/nightstream/src/lifecycle/terminal/mod.rs`) and `Package::apply`
(`terminal/package.rs`). The layer-1 verifier is a parameter of
`finalProgram`. It receives the claim and the transcript state that the
package's output columns hold.

- Package (`Circuit/Package.lean`): `load` is a port of Rust's `load`. It
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
  800 MB).

The terminal program assumes, and does not prove:

- Rows: that the system's rows hold, with column one at one, exactly when
  `CircuitPackage.RowsHold` holds. The rows are F′'s instantiation,
  reordered and normalized, so this is a Lean proof about `load`. The check
  shows only that Rust's rows are these rows.
- Scope: the check runs on the pinned artifact with one context. The
  instructions do not depend on the statement values, and the context words
  enter as constants.
