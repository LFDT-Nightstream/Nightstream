# Why the claim-critical Rust moves into one crate that Aeneas reads whole

This note explains the shape of the plan in [aeneas-plan.md](aeneas-plan.md). It is based on the
`nico/f-prime-constraints-cuda-formal` branch at `fe09c1510`, a throwaway pilot run on
2026-10-30, and an independent review of the first draft that ran its own experiments.

## The problem

`formal/compression-lean` proves its claims about Lean copies of Rust code. The 570
entries in `ports.json` tie each copy to a Rust item with a hash of its tokens.
[TRUST.md](../formal/compression-lean/TRUST.md) section 5 states the limit:
the `lean_ports` test "detects a change; it does not show that the transcription is
right." If a copy differs from its Rust, a claim is about a verifier that Rust does
not have.

Aeneas translates Rust into Lean. With it, the Lean side of each copy can come from
the Rust itself, and a theorem can state that the translation equals the copy.

## What the pilot and the review showed

The tools were Aeneas `nightly-2026.10.30-de16f3c` and Charon 0.1.301. Aeneas pins Lean
v4.35.0 and Mathlib v4.35.0. Every Lean dependency commit matches `compression-lean`.

- **The real crate failed.** Charon on `neo-spartan` from `shrink::layout` pulled in
  Plonky3 0.8 field traits, serde derives, and rayon. Aeneas logged about 550 internal
  errors and then crashed. Charon flags did not stop the pull.
- **A crate with no dependencies worked.** The integer layout functions extracted
  cleanly apart from two Aeneas bugs. Their translations are proved equal to
  `Ports/Shrink/Layout.lean` (229 Lean lines).
- **Field code worked behind a local type.** An index-loop `eq_table` over a local `Ext`
  is proved equal to `Lib.eqAt`, with no panic and termination when `2^len ≤ usize::MAX`
  (175 Lean lines). The rayon body and the `iter_mut` body both failed with an internal
  error at `table = next;`.
- **Only one extraction run joins crates.** Extracting a crate turns every other crate it
  uses into axioms. Extracting the other crate separately gives definitions that do not
  link. `charon --include` gives one output with bodies and no axioms.
- **Aeneas models few iterator methods.** `enumerate`, `rev`, `take`, `zip`, and
  `collect` translate. `map`, `sum`, and `any` become axioms, and `.copied().product()`
  crashes Aeneas. `&dyn Fn` fails ("Higher-ranked vtables are not supported"). Generic
  `B: Backend` and `impl Program` parameters translate.
- **Proofs surface edge cases.** `split(0)` panics in Rust debug builds, returns
  `[2^63, 2^63]` in release, and is `[1, 0]` in the copy. No production input hits it.

## What the papers and the bug lists showed

Charon contains a copy of the hax frontend, the part that reads rustc's data
(`charon/src/bin/charon-driver/hax/`, 23 files at the pinned commit). So bugs in hax,
Charon, and Aeneas can all matter. A review of three papers and the three bug lists found
the following.

- **Overflow follows the build.** Charon makes `+` a failure only where the MIR has an
  overflow check (`reconstruct_fallible_operations.rs`). Cargo ignores a member crate's
  profile when another workspace builds it. "Verification Facade"
  ([2026/670](https://eprint.iacr.org/2026/670)) shows the same gap in hax.
- **Library models had real bugs.** Fixed before the pin: `leading_zeros(0)`,
  `Range::next`, `rotate_right`, `min` and `max` on equal inputs, and others. Open at the
  pin: signed `%` ([#1275](https://github.com/AeneasVerif/aeneas/issues/1275)), subslices
  with `start == end` ([#1238](https://github.com/AeneasVerif/aeneas/pull/1238)),
  `Vec::insert` at `len` ([#1288](https://github.com/AeneasVerif/aeneas/issues/1288)), and
  a name that resolves to the wrong item
  ([#1458](https://github.com/AeneasVerif/aeneas/issues/1458)).
- **A checked theorem can promise nothing.** "Nobody Will Read the Lean"
  ([2026/2441](https://eprint.iacr.org/2026/2441)) shows a Lean security theorem that
  held for every adversary. "Verification Theatre"
  ([2026/192](https://eprint.iacr.org/2026/192)) found 4 of 13 libcrux bugs in the
  verified specs and proofs.
- **Tests find what proofs skip, and proofs find what tests skip.** hax runs its library
  models on generated tests ([2025/142](https://eprint.iacr.org/2025/142)). Jolt-QED's
  proofs found a bug that its differential tests missed
  ([2026/2157](https://eprint.iacr.org/2026/2157)).

No issue in the three papers is in the shared hax frontend. The paper issues are in the
hax engine, its F* backend, or libcrux's own code.

## What we reuse

ProtogaLattice (`dcSpark/ProtogaLattice`, read at `96b57a1c801b`) is a Goldilocks
lattice-folding project on the same Lean and Mathlib versions. These parts fit the plan:

| Part | Use | Phase |
|---|---|---|
| `pgl-arith/src/fq.rs` | The start of `Gl`. It has no dependencies, keeps values canonical, and uses explicit wrapping. | 2 |
| `formal/ProtogaLattice/Exec/Rq/Native.lean` | Lemmas that word-level `addmod`, `submod`, and `mulmod` compute mod `p`. | 2 |
| `pgl-arith/src/e.rs` | A template for `Ext`. It is `u^2 - 7`, Nightstream's K, not the cubic `Ext`. | 2 |
| `pgl-hash/src/poseidon2.rs` | The start of the scalar Poseidon2. Same width, S-box, round counts, and internal diagonal as `NightstreamFPrime.Spec.Poseidon2`. The round constants differ. | 3 |
| `formal/tests/Meta/AxiomAudit.lean`, `AuditNegative.lean` | A stricter axiom audit, first copied from ours, and tests that it rejects bad cases. | 0 |
| `formal/tests/Meta/NonVacuity.lean` | The pattern for premise witnesses. | 0, 5 |

ProtogaLattice is an internal dcSpark repository, and its `Cargo.toml` says Apache-2.0.
The owner approved copying these parts into public Nightstream on 2026-10-30. Each copied
file starts with a header that names the source repository, commit, and path, as
ProtogaLattice's `AxiomAudit.lean` does. The protocol crate, its Lean protocol proofs,
and the SHAKE transcript do not fit. Nightstream allows only Poseidon2 in proof paths.

[Plonky3-lean](https://github.com/Plonky3/Plonky3-lean) translates Plonky3 with hax's Lean
backend, which runs Charon and Aeneas, on Lean v4.35.0. Its `goldilocks` crate is partly
translated, and `field`, `poseidon2`, and `whir` are stubs. It gives no proof that this plan
can use today. Its list of translator issues and its Lean gap models are references for
Phase 0.

## Assumptions, tested

| Assumption | Necessity or convention | Kept? |
|---|---|---|
| The translated code must be the code that runs, or the code that the running code is tested against. | Necessity. A proof about a copy is a hand port again. | Kept |
| Plonky3 must leave the workspace. | Convention. Charon translates only what the verified crate reaches. The verifier files use only Plonky3 0.8 `Gl` and `Ext`. | Dropped. The verified crate owns its field, and `neo-spartan` converts at the boundary. |
| The verified code needs two crates, field and logic. | Convention. Two separate extractions do not link. | Dropped. One crate, `neo-verified`. |
| The circuit code can wait. | False. The shrink verifier runs `Recorder` and `Native`, and `Lib.roundValues` pins `shrink/mod.rs::replay`, which calls `algebra::interpolate`. | Dropped. The circuit core moves before the verifier. |
| Verified code must be single-threaded. | Necessity. Aeneas has no concurrency model. | Kept, for verifier code. Phase 5 measures the cost. |
| Proofs must replace the existing copies. | Convention. The copies are the specs that the current proofs use. | Dropped. Each translation is proved against its copy, and the copy stays. |
| "Translation equals copy" fits every copy. | False for `Ports/Shrink/Game.lean`, which is a game with random draws. | Kept for functions. The verifier needs a simulation theorem, designed first in Phase 5. |
| CI must build Lean. | Necessity. Today CI checks Lean artifact digests only. | Kept |
| Aeneas must be forked. | Convention. Every tool problem so far had a workaround. | Dropped. Pin upstream. |
| The formal projects can change their Lean version on their own. | False. The proofs import the extracted code, which builds only on the Lean that Aeneas pins. | Dropped. CI locks all three to one Lean. |
| Release overflow checks in our `Cargo.toml` make deployed code match the proofs. | False. Cargo ignores them when another workspace builds `nightstream`. | Dropped. The code uses `checked_*` or `wrapping_*`, and clippy denies plain arithmetic. |
| A theorem that names the translation and the copy ties them. | False. A soundness theorem alone holds for a function that always fails. | Dropped. Each entry has a soundness and a completeness theorem, and each premise has a witness. |
| Aeneas's library models match Rust. | False at edge inputs, at the pin and before it. | Dropped. CI lists the models used and tests each one at its edges. |

## The decisions

1. **One crate, `neo-verified`, holds all verified code.** It has no dependencies, so one
   Charon run reads all of it with no flags. It contains the field, the circuit core, the
   scalar Poseidon2, and the claim-critical verifier logic.
2. **`neo-verified` owns a field with `Gl` and `Ext` only.** It is safe scalar Rust with
   canonical values. Aeneas translates the arithmetic itself, so a proof against `ZMod p`
   and `AdjoinRoot (X^3 - X - 1)` takes the field out of the trusted base. Serialization
   stays in `neo-spartan` on Plonky3 types, so proof bytes do not change.
3. **The circuit core moves into `neo-verified`.** `Backend`, `Native`, `Recorder`, `Sink`,
   `Wire`, `Form`, `R1cs`, and `Kind` move. `Backend::hint` takes a closed set of hint
   kinds instead of `&dyn Fn`. `Kind::entries` loses its `OnceLock`.
4. **`neo-verified` has a scalar Poseidon2.** It replaces `hash::permutation()` behind
   `Native::permute`, with the round constants as a constant table. It is proved equal to
   `NightstreamFPrime.Spec.Poseidon2.permute`. The prover's packed Poseidon2 stays, with a
   test against the scalar one.
5. **The verifier runs the verified code. The prover keeps its fast paths.** A test checks
   each prover fast path against its sequential twin. Prover bugs cost completeness, not
   soundness.
6. **Plonky3 stays everywhere else.** WHIR (about 130,000 lines of Plonky3 0.8), the
   0.5.3 field layer, the transcript, and the GPU crates do not change. Removing them
   serves no section 5 copy.
7. **Each entry has a soundness and a completeness theorem, or a simulation for the
   game.** Soundness ("if the translation returns, it returns the copy's value") has no
   premise, and the claims use only it. Completeness ("on good inputs, it returns")
   stops a translation that always fails from passing. A Lean check confirms both shapes.
   A name alone is not evidence.
8. **The extraction is the gate.** CI runs Charon and Aeneas from pinned versions with a
   pinned sysroot and profile. It fails on any Aeneas error or warning, a non-zero exit,
   a generated `axiom` that is not on an allowlist, a new library model, or a broken code
   rule. It fails when the output differs from the committed Lean. It then builds the
   proofs. The authority is a recomputation from Rust plus the Lean kernel, not a stored
   hash.
9. **`neo-verified` spells out overflow.** Each integer operation is `checked_*` with an
   explicit reject, or `wrapping_*`. The Lean and every build of the Rust then agree,
   whatever profile builds it.

## Alternatives rejected

- **Keep the code in `neo-spartan` and steer Charon with flags.** The pilot tried this
  four times and failed each time.
- **Two crates, extracted separately.** The outputs do not link.
- **Remove all of Plonky3.** It costs the 0.5.3 codemod (about 240 files outside
  `neo-spartan`) and a WHIR rewrite for no change to any claim.
- **Verify word-for-word copies in a scratch crate.** The copy drifts like a hand port.
- **Fork Aeneas.** No problem so far needed a patch to the tool.
- **Use Plonky3-lean's models of Plonky3's field and Poseidon2.** They are stubs today, and
  a separate extraction does not link with ours.

## What stays trusted at the end

- Charon's and Aeneas's translation, and Aeneas's Lean models of the Rust standard
  library that `neo-verified` uses. CI lists the models used and tests each one at its
  edges. The run check runs the translated Lean and the Rust on the same seeded, edge,
  and real-verify inputs. A translation bug that changes an output on those inputs fails
  CI. A bug on other inputs does not. The mutation check shows that the theorems can
  fail.
- Every hand-written model that post-processing adds. Each one goes in TRUST.md. The
  goal is none.
- The conversion between Plonky3 0.8 values and `neo-verified` values in `neo-spartan`,
  which tests check.
- Prover fast paths, which tests check against the verified code.
- WHIR and everything else that TRUST.md lists and this plan does not touch.
