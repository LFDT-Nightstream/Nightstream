# Phase 5 design: the statement for the Rust shrink verifier

This note is step 1 of Phase 5 in [aeneas-plan.md](aeneas-plan.md). It fixes the theorem that
`shrink/mod.rs::verify` must meet after it moves into `neo-verified`. Before any code moves, it
proves that the theorem gives the production claim and that the theorem is not empty. No Rust
code moved.

## What the statement promises

The statement is about one Lean function: the type that Aeneas will give
`neo_verified::shrink::verify`. The verifier reads its Fiat–Shamir transcript and its WHIR opening
through a channel, and it reads the program's shape run through a second parameter. The statement
runs the verifier on a channel that records each call and answers each challenge with an arbitrary
function of the calls so far. `Sound` promises that if the verifier accepts, then four things are
true. The verifier made exactly the channel calls of `schedule`, in the order of the Rust code
today. The WHIR opening returned a value. The statement words are the program's public words. And
`shrinkGame`, the game that the claims bound, accepts the play whose messages are the proof's
messages and whose draws are the challenges that the verifier drew. `Sound` holds for every proof,
every statement, every challenge function, and every list of WHIR candidates, so no premise hides
an adversarial input. Its one premise is that the opening is sound, which is the WHIR assumption
that the claims make already. `Complete` promises that the verifier accepts every honest run, and
that an honest run exists. The statement does not promise that Aeneas translates the Rust
correctly, that the Plonky3 challenger and the WHIR verifier in `neo-spartan` act like the
recording channel, that the Rust program records the rows of the Lean program `shrinkRun`, or that
Poseidon2 is a random oracle. These stay in TRUST.md.

## The statement in Lean

The definitions are in `formal/compression-lean/CompressionLeanRust/Verifier.lean`. `VerifyFn` is
the translated `verify` with slices read as lists. `recorder oracle opening` is the recording
channel. `schedule` is the order of channel calls. `msgsOf` and `drawsOf` turn a run into a play of
`shrinkGame`, and `evalOf T s run` is the shape run of a Lean `Run` as `Ports/Shrink/Evaluate.lean`
computes it.

```lean
abbrev VerifyFn : Type 1 :=
  ∀ {σ : Type}, Channel σ → Evaluate → Counts → List field.Gl → Rounds → σ → Result (Bool × σ)

def Sound (verify : VerifyFn) (T : Template Gl) : Prop :=
  ∀ {I : Type} [Nonempty I] (prog : I → Run Gl) (s : Counts) (L : ℕ) (i : I) (stmt : List field.Gl) (π : Rounds)
    (cands : {l : List (ℕ → Gl) // l.length ≤ L}) (oracle : History → field.Ext)
    (opening : History → List Cubic → Option field.Ext) (h : History),
    WhirSound cands.1 opening →
    verify (recorder oracle opening) (evalOf T s (prog i)) s stmt π [] = ok (true, h) →
    ∃ d w, schedule (recorder oracle opening) (Layout.rowVars T s) (Layout.cellVars T s) stmt π [] = ok (d, h) ∧
      d.opened = some w ∧ (prog i).publics = stmt.map glOf ∧
      ∃ tr, Game.Reaches (shrinkGame (K := Cubic) T s prog L) (msgsOf i cands π (cubicOf w)) (drawsOf d) (some tr)

def Complete (verify : VerifyFn) (T : Template Gl) : Prop :=
  (∃ (prog : Unit → Run Gl) (s : Counts) (L : ℕ) (stmt : List field.Gl) (π : Rounds)
      (cands : {l : List (ℕ → Gl) // l.length ≤ L}) (oracle : History → field.Ext)
      (opening : History → List Cubic → Option field.Ext),
      WhirSound cands.1 opening ∧ Honest T prog s L () stmt π cands oracle opening) ∧
    ∀ {I : Type} [Nonempty I] (prog : I → Run Gl) (s : Counts) (L : ℕ) (i : I) (stmt : List field.Gl) (π : Rounds)
      (cands : {l : List (ℕ → Gl) // l.length ≤ L}) (oracle : History → field.Ext)
      (opening : History → List Cubic → Option field.Ext),
      Honest T prog s L i stmt π cands oracle opening →
      ∃ h, verify (recorder oracle opening) (evalOf T s (prog i)) s stmt π [] = ok (true, h)

def Simulates (verify : VerifyFn) (T : Template Gl) : Prop := Sound verify T ∧ Complete verify T
```

`Honest` asks that the sizes fit a 32-bit `usize` (`Small`), that the statement words are below
p and are the program's public words, that the rounds have the lengths that the shape fixes
(`RoundsFit`), that the opening returns a value, and that `shrinkGame` accepts the play.

Step 4 must prove `Simulates (adapt neo_verified.shrink.verify) poseidonTemplate`. `adapt` is glue
between the two types, and it may do only these things:

- turn each list into a slice, the Lean channel into Aeneas's `shrink.Channel` structure, and the
  `Evaluate` function into a `shrink.ShapeRun`;
- fail when a size does not fit the Rust integers, that is, when `Small` is false;
- map `Ok(())` to `true` and each `Err` to `false`.

It changes nothing else. It does not read the proof, the statement or the channel answers, and it
does not choose the verdict. The step-4 reviewer checks `adapt` against this list, so that the glue
cannot move the verdict.

## What is proved

All files are in `formal/compression-lean`. Every theorem uses only `propext`, `Classical.choice`,
and `Quot.sound`. `#check_rust_axioms` in `CompressionLeanRustTests.lean` checks this for the 2,065
theorems of `CompressionLeanRust`. A planted theorem that uses a new axiom fails it. A `#guard_msgs` pin in
`CompressionLeanRustClaims.lean` checks the claim below.

(a) The statement gives the production claim. `CompressionLeanRustClaims.poseidon2_bound_rust`
(`CompressionLeanRustClaims.lean`) takes `Sound verify poseidonTemplate` and the premises of
`CompressionLean.Claims.poseidon2_bound`. It gives the same bound, `calls / 2 ^ t`, in the same
restoration experiment, for the event `RustBreak`. In `RustBreak`, the prover's shrink outcome is
the leaf of a play where `verify` accepted the proof with the challenges that it drew, the claim of
the instance is false, and every valid WHIR candidate has an accepting layer-1 reading.
`RustBreak` contains the case where the prover's own play is such a run, so the bound covers it.
The generic step is `rustBreak_gameBreak` and `rust_bound` in `CompressionLeanRust/Verifier.lean`.
It needs only `Sound`, as the plan's Phase 0 rule says.

(b) The statement is not empty. `CompressionLeanRust/VerifierModels.lean` defines small model
verifiers and proves:

- `honest_simulates`: a model that makes every check of `verify` meets `Simulates`. So the
  statement can be met, and the proof plan of step 4 works on a model.
- `acceptsAll_not_sound`: a model that makes the channel calls and accepts everything is not
  `Sound`. It accepts a run whose outer sum-check fails.
- `skipsOpening_not_sound`: a model that accepts when the WHIR opening fails is not `Sound`.
- `skipsFinal_not_sound`: a model that skips the final check of the inner sum-check is not
  `Sound`. It accepts a run whose inner rounds end at 1 while the matrix value is 0.
- `skipsObserve_not_sound`: a model that makes every check but draws the batching challenge
  without observing the matrix values is not `Sound`. The game alone cannot see this
  Fiat–Shamir error. The `schedule` part of `Sound` sees it.
- `skipsPublics_not_sound`: a model that makes every check but does not check that the statement
  words are the program's public words (the conjunct `(ev …).2.2 = stmt.map glOf`) is not
  `Sound`. It accepts the statement `[0]` for a program with no public word.
- `skipsCounts_not_sound`: a model that makes every check but does not check that the program's
  counts are the shape's is not `Sound`. It accepts the empty shape for a program with one glue
  cell, and `shrinkGame` rejects that play.
- `rejectsAll_not_complete`: a model that rejects everything is `Sound` (`rejectsAll_sound`) but
  not `Complete`.

Each counterexample holds for every template `T`. The play lemmas that step 4 needs are in
`Verifier.lean`: `run_path` builds the play of a run, `run_outer_ok` shows that an accepted play
passed the outer check, and `shrinkGame_leaf` reads `Final` off an accepted play.

To check: `lake build CompressionLeanRust CompressionLeanRustTests CompressionLeanRustClaims` in
`formal/compression-lean`. The default `lake build --wfail`, the executables, `Ports.lean`, and
`SpecCensus.lean` also pass.

## The Rust signature that the move needs

```rust
pub struct Rounds {
    pub outer: Vec<Vec<Ext>>,
    pub values: [Ext; MATRICES],
    pub inner: Vec<Vec<Ext>>,
}

pub trait Channel {
    fn observe_words(&mut self, words: &[Gl]);
    fn commit(&mut self);
    fn observe(&mut self, values: &[Ext]);
    fn sample(&mut self) -> Ext;
    fn open(&mut self, point: &[Ext]) -> Option<Ext>;
}

pub struct Evaluation {
    pub total: Ext,
    pub counts: Shape,
    pub publics: Vec<Gl>,
}

pub trait ShapeRun {
    fn evaluate(&self, rx: &[Ext], ry: &[Ext], rho: Ext) -> Evaluation;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reject { Shape, Outer, Opening, Counts, Inner }

pub fn verify<C: Channel, P: ShapeRun>(
    shape: &Shape, statement: &[Gl], program: &P, rounds: &Rounds, channel: &mut C,
) -> Result<(), Reject>
```

`neo-spartan` keeps the transcript, WHIR, and the program. Before it calls `verify`, it observes
the domain, shape, profile, and key words, as `Shrink::start` does today. Its `Channel` sends
`observe_words` to `observe_slice`, `commit` to `Pcs::observe`, `observe` to
`observe_algebra_slice`, `sample` to `sample_algebra_element`, and `open` to `Pcs::verify`. So the
transcript and the proof bytes do not change. Its `ShapeRun` runs the program on
`Recorder<Evaluate>`. `Reject` has no strings, because Aeneas turns a string literal into an
axiom, and `neo-spartan` maps each variant to today's error message.

The pinned Charon (0.1.301) and Aeneas (nightly-2026.10.30-de16f3c) translate a scratch copy of
`neo-verified` with this signature and a sketch body. They report no error, and the axioms of the
output are exactly the axioms in `formal/aeneas/axioms.txt`. The sketch is not in the repository.
It was run again on 2026-10-31, outside the repository, with the flags of
`scripts/aeneas_extract.sh`. The sketch is a copy of `crates/neo-verified/src` at commit
`f4bbca1ac`, with a new `src/shrink.rs` (the signature above, and a body that makes the checks of
today's `verify` with `block_part` as a stub that returns zero) and `pub mod shrink;` in `lib.rs`.
Its `Cargo.toml` is standalone. The translation is:

```lean
def shrink.verify {C : Type _} {P : Type _} (ChannelInst : shrink.Channel C) (ShapeRunInst : shrink.ShapeRun P)
  (shape : layout.Shape) (statement : Slice field.Gl) (program : P) (rounds : shrink.Rounds) (channel : C) :
  Result ((core.result.Result Unit shrink.Reject) × C)
```

## Steps that remain

1. Step 2, the move. Size: medium, about 350 Rust lines in `neo-verified` and 100 in
   `neo-spartan`. Move `verify`'s checks, `block_part` from `shrink/layout.rs`, and the
   `Evaluate` sink from `shrink/evaluate.rs`. Rewrite their rayon, `map`, `sum`, and `product`
   calls as loops. Copy the first `n` inner draws with a loop, not `ry[..n]`, because the
   subslice model has an open edge bug (#1238). Each new standard-library model needs a run-check
   case. Risk: low to medium. The sketch translates, and `toy_proof_bytes_are_pinned` shows any
   change to the transcript.
2. Step 3, timing. Size: small. Measure `layer1_verifier_shrinks` verify time before and after.
   Risk: low. The checks are already scalar. The sequential `Evaluate` can cost time in the
   shape run.
3. Step 4, the proofs. Size: large, about 1,500 to 2,500 Lean lines, by comparison with Phase 4.
   Prove `Simulates` for the adapted translation along the path of `honest_sound` and
   `honest_complete`. Prove the ports `Ccs.publicValue`, `Ccs.publicTable`, `batch`,
   `Ports/Shrink/Evaluate.lean`, and `Ports/Shrink/BlockPart.lean`. Prove that
   `(shrinkRun key loaded context x).publics = List.ofFn x`. This fact serves the Fiat–Shamir
   prefix injectivity: the Rust transcript carries the statement words, and the game position
   carries the instance `x`, so the two must determine each other. `poseidon2_bound_rust` does not
   use this fact. Give `ports.json` a proof kind for the verifier and teach the shape check and the
   `lean_ports` test about it. Update TRUST.md section 5. Risk: medium. The theorem shape is no
   longer the unknown. The unknowns are loops that call the channel, the proof of `Evaluate`
   against `glueTotal`, and whether `Small` covers the `Complete` premises of the translated
   helpers.

## Decisions

- The program's shape run is a parameter (`ShapeRun`), as the transcript and WHIR are. The
  statement then has no premise about the program. The other way, `verify<P: Program>` with
  `run<B: Backend>`, would need a premise that the program's run equals `shrinkRun`. The cost of
  this choice is a few lines of glue in `neo-spartan` that the proof does not cover.
- The channel answers challenges with an arbitrary function of the recorded calls, not a fixed
  list. This makes `Sound` speak of the transcript order. `skipsObserve_not_sound` shows that the
  order matters.
- `Sound` and `Complete` are separate, as in Phase 0. The claim uses only `Sound`.
- `poseidon2_bound_rust` lives in its own library, `CompressionLeanRustClaims`. The aeneas CI
  job builds the 102 local modules that `CompressionLeanRust` imports today. `Verifier.lean` and
  `VerifierModels.lean` add 7. Importing `CompressionLean.Claims` would add 697 more (588 from
  F′). No CI job builds the claims today, so this claim is in the same place as the claims that
  it uses.

## Limits

- `Sound` speaks of runs on the recording channel. The Rust `verify` is generic over `C: Channel`
  and cannot inspect it, so its run on the real transcript makes the same calls with the real
  answers. Lean does not check this step. A Lean function that looks at its channel type with
  classical logic could meet `Simulates` and act differently elsewhere. The Aeneas translation of
  Rust cannot do that.
- `commit` carries no words, and the words that `neo-spartan` observes before it calls `verify`
  (the domain, shape, profile and key words) are not in the history at all. So `Sound` does not
  bind the challenges to the WHIR commitment, the key or the shape. That binding comes from the
  `Channel` implementation in `neo-spartan`, where `commit` calls `Pcs::observe`
  (`crates/neo-spartan/src/pcs.rs`) and the transcript starts as `Shrink::start` starts it, and
  from the Fiat–Shamir and WHIR assumptions in TRUST.md section 3. It does not come from `Sound`.
- `evalOf` reads the shape run from the Lean `Run`. That the Rust program records the same rows
  stays an assumption (TRUST.md section 5, the circuit programs).
- `Small` is a guess at the premises that the translated helpers need. Step 4 can tighten it.
