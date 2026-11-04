# How to make the claim-critical Rust provable with Aeneas

Goal: each `ports.json` entry that TRUST.md section 5 marks as able to make a claim false
has a Lean theorem that ties it to Aeneas's translation of the Rust that runs. CI
rebuilds the translation and the theorems on every change. The reasons for this shape
are in [aeneas-design.md](aeneas-design.md).

Each phase ends in a check. Start a phase only when the check of the phase before it
passes. Every phase also checks that the proof bytes for a fixed input do not change.
Sizes are guesses. The anchors are the pilot's two proofs, of 229 and 175 Lean lines.

## Phase 0. Pin the tools and add the CI gate

1. Add `formal/aeneas/versions` with the Aeneas tag (`nightly-2026.10.30-de16f3c`), the
   Charon commit it pins (`373d840`), the Rust nightly (`nightly-2026-09-17`), the Lean
   version (`v4.35.0`), and the Charon sysroot choice. The pilot's Charon fell back to
   `--sysroot default` because `miri` was not installed. Also pin the Cargo profile and
   target that Charon compiles: overflow checks on, and a 64-bit target. Every machine
   must produce the same output.
2. Add `scripts/aeneas_extract.sh`. It runs `charon cargo --preset=aeneas` and
   `aeneas -backend lean -split-files` on `neo-verified`, then any post-processing step.
   It writes to `formal/compression-lean/CompressionLean/Extracted/`. It fails on:
   - an Aeneas `[Error]` or `[Warn]`, or a non-zero exit;
   - a generated `axiom` that is not on the allowlist in `formal/aeneas/axioms.txt`;
   - `opaque`, `sorry`, `admit`, `implemented_by`, `extern`, or `debug.skipKernelTC` in
     the extracted or proof files;
   - a `#[charon::...]` or `#[aeneas::...]` attribute in `neo-verified`;
   - an `*External.lean` file that the allowlist does not name. Aeneas writes these files
     once and does not write them again, so the `git diff` in step 3 cannot see an edit
     to them;
   - a clippy or grep failure for the code rules at the end of this file.

   Each post-processing step links its upstream issue, and each hand-written model it
   adds goes in TRUST.md. Before you write a post-processing step, read
   [Plonky3-lean](https://github.com/Plonky3/Plonky3-lean)'s `docs/extractor-issues.md`
   and `HaxExt.lean`. They list translator gaps and the Lean models that fill them, on
   the same Lean version.
3. Add a CI job. It installs the pinned Linux binaries and the nightly, runs the script,
   and fails on `git diff --exit-code` of `Extracted/`. Then it runs `lake build` on the
   extracted modules and the refinement theorems.
4. Measure the job time with the Mathlib cache. If it is too long, build only the
   targets that import `Extracted/`.
5. Add a test that pins the proof bytes for one fixed input. The retained vectors that
   `crates/nightstream/tests/check_golden_vectors.py` replays check the verifiers, not
   the prover's bytes.
6. Fix the theorem shape. Each proved `ports.json` entry has two theorems:
   - Soundness: `extracted x = ok y → y = copy x`, with no premise. The claims use only
     this one. A Rust panic in the verifier is a reject (`panic = "abort"` in release),
     so a `fail` in Lean is safe if it fails only where Rust panics.
   - Completeness: `H x → extracted x = ok (copy x)`. Each premise `H` has a "met by"
     row in `CompressionLeanTests/Inventory.lean`, and one input that satisfies it.

   Soundness alone holds for a function that always fails, so completeness is needed.
   Write the premise witnesses as ProtogaLattice's `tests/Meta/NonVacuity.lean` does.
   A Lean check confirms both shapes, including every output and every `&mut` result,
   and `CompressionLeanTests/Axioms.lean` pins their axioms. A theorem name alone is not
   evidence.

   Replace `formal/nightstream-fprime/tests/AxiomAudit.lean` with ProtogaLattice's
   `tests/Meta/AxiomAudit.lean`. It started as a copy of ours and adds a full-name check
   and a check for compiled-code overrides (`implemented_by`, `extern`) outside the
   toolchain and Mathlib. Ours has neither. Also port its `tests/Meta/AuditNegative.lean`,
   which shows that the audit rejects each bad case.
7. Lock the Lean version. CI fails when `formal/compression-lean/lean-toolchain` or
   `formal/nightstream-fprime/lean-toolchain` differs from the Lean version in
   `formal/aeneas/versions`. The proofs import the extracted code and both projects, so
   all three must use one Lean. A Lean upgrade moves the Aeneas pin in the same change and
   runs the extraction again. If no Aeneas release supports the new Lean, the upgrade
   waits. On each Aeneas or Charon pin change, run step 9 again and read the open bug
   issues for each model on its list.
8. Add a run check. The proofs trust Charon, Aeneas, and Aeneas's library models. A Rust
   test writes the outputs of each `neo-verified` function on three input sets:
   - about 200 inputs from a fixed seed;
   - edge values: 0, 1, `p - 1`, `p`, `2^63`, `u64::MAX`, empty slices, and
     non-canonical words;
   - the calls that one `verify` of a retained vector makes.

   The inputs reach every branch and every loop exit. A `lake exe` reader, like
   `CompressionLeanExe/Runs.lean`, runs the translation on the same inputs and compares.
   Where the copy is computable, it runs the copy too. A Lean failure must match a Rust
   panic. Do not use `native_decide` for this, because it adds an axiom. The pilot ran the
   translated `split` with `#eval`. It gave `[4, 0]`, `[4, 1]`, and `[512, 512]` for 4, 5,
   and 1000, and an `integerOverflow` failure for 0.
9. Pin and test the library models. The script lists each `core.*` and `alloc.*` constant
   that the extracted code uses. CI compares the list with `formal/aeneas/std-models.txt`,
   so a new model fails CI until someone adds it. `formal/aeneas/StdModels.lean` tests each
   listed model with `#guard` on edge inputs (0, 1, MAX, width - 1, width, an empty range,
   `i == len`) against values that a Rust test writes. A model must fail exactly where
   Rust panics. A model that fails where Rust returns a value puts that input outside the
   soundness theorem. At the pin, open issues report such edge errors in signed `%`
   ([#1275](https://github.com/AeneasVerif/aeneas/issues/1275)), subslices with
   `start == end` ([#1238](https://github.com/AeneasVerif/aeneas/pull/1238)), and
   `Vec::insert` at `len` ([#1288](https://github.com/AeneasVerif/aeneas/issues/1288)). Do
   not use a model that fails its tests, or fix it in post-processing with a link to the
   upstream fix.
10. Add a mutation check. A script plants one mutation in each proved function (a flipped
    comparison, a changed constant, a removed check), runs the extraction, and confirms
    that a theorem fails to build. Run it once per phase, not on every change.
11. Run `lean4checker` on the theorem modules in CI. TRUST.md section 1 names it as
    the way to re-check the cached `.olean` files, but nothing runs it today.

Check: CI fails on each of these: a hand edit to a generated file, a new `axiom`, a
broken theorem, a Lean version that differs from the pin, a changed output in the run
check, a new library model, and a broken code rule. Size: small to medium.

Status, 2026-10-30 (uncommitted, in worktree `aeneas-rust-lean-compiler-019889`):

- Done, and each check shown to fail on a planted break: steps 1 to 5 and 7 to 11. The
  gate needs real code, so Phase 0 moved `split` and `variables` (Phase 1) into
  `neo-verified` and proved both shapes for them.
- Where it differs from the steps above:
  - The translation is in `formal/compression-lean/NeoVerified/`, the theorems in
    `CompressionLeanRust/`, and the shape check of step 6 in `CompressionLeanRustTests.lean`.
    They are separate Lake libraries, so CI builds them without the rest of the project.
  - The run check runs with `lake env lean --run`, not a `lake exe`. An executable would
    compile Mathlib to native code. Its library-model cases replace `StdModels.lean`.
  - The pin of step 5 runs the prover on one thread. With more, the bytes change from run
    to run.
  - Step 11 runs `leanchecker --fresh` in the CI job, not each night. It ships with Lean 4.35
    and takes 97 seconds here.
- Open: the axiom audit port of step 6, and a first CI run on GitHub.

## Phase 1. Start `neo-verified` with integer code

1. Create `neo-verified` with no dependencies and
   `#![deny(clippy::arithmetic_side_effects)]`.
2. Move `Kind` with `rows`, `cells`, and `index`, `Wire`, and the integer parts of
   `shrink/layout.rs`: `Shape` counts, `Part`, `split`, `place`, `variables`, and the
   `Layout` positions. Keep `block_part` and `diagonal` in `neo-spartan` for now. Derive
   serde for `Shape` in `neo-spartan` with `#[serde(remote = ...)]`.
3. Rewrite `place` without the `impl Fn` argument and without `sort_by_key`. With one
   block kind, the order is fixed. Rewrite the arithmetic with `checked_*` or
   `wrapping_*`.
4. Split the error charges. `relation.rs::outer_terms` and `shrink/mod.rs::terms` compute
   each term's degree, then an `f64` log and a `p3-security` `SecurityTerm`. Aeneas has no
   floating-point model. Move the integer degree of each term, and `ring.rs::row_count`,
   into `neo-verified`. Keep the `f64` and `SecurityTerm` code in `neo-spartan`.
5. Port the pilot proofs to the Phase 0 shape. Prove the rest against
   `Ports/Shrink/Layout.lean`, `Layer1.rustDegrees`, `Shrink.rustDegrees`, and
   `Layer1.rustLinear`.
6. Give each proved `ports.json` entry a `"proof"` field that names its two theorems.
   The `lean_ports` test skips the token hash for those entries. Keep the token-hash
   entries for the `neo-spartan` code that calls proved functions. An entry that cannot
   have a proof gets `"proof": {"excluded": "axiom" | "boundary" | "mismatch", "issue":
   ...}`. Fix a mismatch in the Rust or in the copy, and add a run-check input that shows
   it. Do not hide a mismatch behind a premise.

Check: the theorems build in CI with only `propext`, `Classical.choice`, and
`Quot.sound`. Size: small to medium.

Status, 2026-10-31: done in two steps, 1a (layout) and 1b (degrees). 18 function ports have
Sound and Complete theorems. neo-verified keeps no `Layout` struct: each position is a pure
function of `Shape`, so Sound needs no premise about cached fields. Type ports and the `f64`
code keep token hashes. 50 source mutations each break a proof, and the proof bytes do not
change.

## Phase 2. Give `neo-verified` its own field

1. Write `Gl` and `Ext = Gl[w]/(w^3 - w - 1)` in safe scalar Rust with canonical values.
   Write only the operations that the verifier files use. Start `Gl` from ProtogaLattice's
   `pgl-arith/src/fq.rs`. Remove the `black_box` and its `#[cfg]`, the `i64` functions,
   `reduce_le256`, and the `Sum` impl. Replace the masks with the branches of
   `mulmod`, `addmod`, and `submod` in its `Exec/Rq/Native.lean`. The masks keep the
   timing independent of secret values. The verifier handles public data only, and the
   branches let the proofs in step 4 reuse that file's lemmas. Use its `e.rs`
   (`u^2 - 7`) as a template for `Ext`.
2. Convert at the `neo-spartan` boundary, as `field.rs` already converts between Plonky3
   versions. Serialization stays on Plonky3 types. Put the canonical check (`u64` to `Gl`,
   reject values `>= p`) in `neo-verified` and prove it.
3. Test each operation against Plonky3 0.8 on random inputs. Also test non-canonical
   Plonky3 words at the conversion, including the words that GPU writebacks store.
4. Prove each operation equal to the `ZMod p` and `AdjoinRoot (X^3 - X - 1)` operations.
   Reuse `mulmod_toNat`, `addmod_toNat`, `submod_toNat`, and their `_lt` lemmas from
   ProtogaLattice's `Exec/Rq/Native.lean`. They are stated over Lean's `UInt64`, so first
   prove a bridge from Aeneas's `U64`.

Check: the field theorems build with no field axiom, and the proof bytes for a fixed
input are unchanged. Size: medium.

## Phase 3. Move the circuit core and Poseidon2

1. Replace the `&dyn Fn` in `Backend::hint` with a closed set of hint kinds.
2. Make the `circuit/algebra.rs` helpers take `u64` words, not Plonky3 values.
3. Move `Backend`, `Native`, `Recorder`, `Sink`, `Form`, `R1cs`, and `Kind::entries` into
   `neo-verified`. Replace the `OnceLock` in `Kind::entries` with a function or a
   constant table.
4. Write a scalar Poseidon2 permutation in `neo-verified` with the round constants as a
   constant table. Start from ProtogaLattice's `pgl-hash/src/poseidon2.rs`. It has the same
   width, S-box, round counts, and internal diagonal. Replace its round constants with
   Nightstream's, and rewrite its `iter_mut`, `fold`, `map`, `chunks_exact_mut`, and
   `&mut dyn FnMut` as index loops. Use it in `Native::permute` in place of
   `hash::permutation()`. The
   existing tests `exported_round_constants_match_fprime_lean_spec` and
   `poseidon2_matches_the_workspace_permutation` must still pass. Keep the packed
   permutation for the prover, with a test against the scalar one.
5. Prove the permutation equal to `NightstreamFPrime.Spec.Poseidon2.permute`. Prove the
   `circuit/algebra.rs` and `circuit/hash.rs` helpers that section 5 lists against their
   copies, run with a Lean backend that is `Build`.

Check: the theorems build in CI, and the lowered-program digest pins still pass. Size:
large.

## Phase 4. Prove the verifier's field helpers

1. Move `shrink/mod.rs::eq_at`, `shrink/mod.rs::eq`, `shrink/mod.rs::replay`,
   `field.rs::eq_table`, and the `diagonal` walk into `neo-verified`. Write them with
   loops and the allowed iterator methods. Restructure `diagonal` without
   `continue 'bit`.
2. Keep the rayon `eq_table` for the prover, with a test against the verified one.
3. Prove them against `Lib.eqAt`, `Lib.eqEval`, `Lib.roundValues`, `Game.sumcheck`, and
   the `Ports/Shrink/Diagonal.lean` closed form.

Check: the theorems build in CI. Size: medium.

## Phase 5. Prove the shrink verifier

1. Design the theorem shape first. `Ports/Shrink/Game.lean` is a game with random draws,
   so "translation equals copy" cannot state it. Write a simulation statement with the
   transcript and the opening check as parameters. Before you move code, prove that:
   - the simulation statement and the existing game claim together give the current
     top-level claim;
   - a verifier that skips one check, such as the final opening check, or that accepts
     everything, does not satisfy the statement.

   Write one plain paragraph that says what the statement promises, and agree on it.
2. Move the checks of `shrink/mod.rs::verify` into `neo-verified`: the public value, the
   batching, the matrix value with `block_part`, and the sum-check rounds. Make them
   sequential. `shrink/mod.rs` has 10 rayon uses today. Replace `.map().sum()` and
   `.map().product()` with loops.
3. Measure verify time before and after.
4. Prove the simulation statement, and prove the function copies (`Ccs.publicValue`,
   `Ccs.publicTable`, `Ports/Shrink/Rounds.lean`, `Ports/Shrink/Evaluate.lean`,
   `Ports/Shrink/BlockPart.lean`). Prover-only copies, such as `outerTables`,
   `innerTables`, and `constraintTable`, are out of scope.

Check: every shrink-verifier entry in TRUST.md section 5 has a proof. Size: large, with
the theorem shape as the main unknown.

## Phase 6. Prove the setup check, the Merkle path, and the terminal loader

1. Move `setup.rs::check` and the Merkle helpers of `circuit/hash.rs` into
   `neo-verified`. Prove them against `Ports/Setup.lean`, `Replay.pathOk`, `merkleRoot`,
   `compress`, and `hashLeaf`.
2. Move the row logic of `nightstream/src/lifecycle/terminal/package.rs::load` and
   `nightstream-fprime/src/package/r1cs.rs::expand_matrix`. Prove it against
   `Terminal.Package.load` and `Terminal.Package.Producer.rows`.

Check: the remaining TRUST.md section 5 entries have proofs, except the Lean readers of
Rust-written files. Aeneas does not cover those. Size: medium to large.

## Phase 7. Decide on the remaining circuit programs

The 227 `Ports/Circuit` entries have digest pins on the lowered programs. After Phase 6,
decide whether to move the generic programs into `neo-verified` and prove each one against
its copy, run with a `Build` backend.

## Not planned

- Removing Plonky3 0.5.3 from the workspace. It touches about 240 files outside
  `neo-spartan`, and no section 5 copy needs it.
- Rewriting WHIR. The Plonky3 0.8 stack is about 130,000 lines, and TRUST.md treats WHIR
  as a named assumption.

## Follow the code rules in `neo-verified`

The pilot, the review, two papers, and the Aeneas and Charon bug lists found these rules.
Some breaks show as an Aeneas error, a warning, or a new `axiom`. Others translate with no
error, so the extraction script checks them with clippy and grep.

- No dependencies. No serde.
- No rayon, threads, `unsafe`, SIMD, `f32`, or `f64`.
- No signed integer types. Two open model bugs at the pin are in signed operations
  ([#1275](https://github.com/AeneasVerif/aeneas/issues/1275),
  [#1287](https://github.com/AeneasVerif/aeneas/issues/1287)).
- No plain `+`, `-`, `*`, `/`, `%`, `<<`, or `>>` on integers. Use `checked_*` with an
  explicit reject, or `wrapping_*`. Charon makes overflow a failure only when the MIR has
  overflow checks, and Cargo ignores this crate's profile when another workspace builds
  `nightstream`. So a plain `+` can fail in Lean and wrap in a deployed binary.
- No `impl Fn` or `&dyn Fn` parameters. No closures that change what they capture.
- No `return` inside a loop. A loop ends by its condition or by one `break`. No labeled
  `break` or `continue`.
- No `split_at_mut` with zipped `iter_mut`. Write through indices.
- Iterators: `enumerate`, `rev`, `take`, `zip`, and `collect` only. Write a loop for
  `map`, `sum`, `product`, and `any`.
- Derives: `Clone`, `Copy`, `PartialEq`, and `Eq` only.
- No `Drop` impls, `OnceLock`, lazy statics, or `HashMap`.
- No `debug_assert!`, `#[cfg]`, or `size_of`. Charon reads one build, and these change
  between builds or machines.
- No `expect` and no panic messages. Aeneas turns a string literal into a `native_decide`
  axiom ([#1372](https://github.com/AeneasVerif/aeneas/issues/1372)).
- Each item name is unique in the crate, and no parameter or field has the name of a
  module. Otherwise a call can resolve to the wrong item
  ([#1458](https://github.com/AeneasVerif/aeneas/issues/1458)).
- Standard functions that Aeneas does not model, such as `usize::trailing_zeros` and
  `next_power_of_two`, become axioms. Write them out, or add a model to TRUST.md.

## Open decisions

Each has a default. The plan uses the default unless you choose otherwise.

1. Crate name. Default: `neo-verified`.
2. Runner and time budget for the CI Lean job. Default: the self-hosted runner, with the
   budget set from the Phase 0 measurement.
3. A limit on verifier slowdown from going sequential. Default: measure and report in
   Phase 5, with no limit set in advance.
4. Whether Phase 7 happens. Default: decide after Phase 6.
5. Who does a Lean upgrade. Default: the person who changes the Lean version also moves
   the Aeneas pin and runs the extraction in the same change.