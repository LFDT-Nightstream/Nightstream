# Lean-built fixed parts of the compression circuit (design)

Date: 2026-10-08. Branch `claude/compression-spartan-whir` (PR #171), after M3 (9f2f04f1c).
Owner direction (2026-10-08): "split the Lean generated circuit and reuse that and only implement
in Rust what changes."

Process note: the architect workflow asks for `/pstack:how` and `/pstack:arena`. Both are
user-invoked only, so Phase A below is my own trace and Phase B compares three designs that I
wrote myself. The owner can run `/pstack:arena` on this file for a multi-model comparison.

## Grounding (Phase A)

1. **The F′ contract** (`FPRIME_LEAN_ARCHITECTURE_SPEC.md`): Lean defines the semantics, builds
   and proves the circuit, and emits the package that Rust loads. Rust executes; it does not
   define a second relation. §12 rejects "Lean certifies large artifacts that Rust created" and
   generated Lean modules that hold rows. Proof cost must not grow with artifact size. Lean is
   never on the proving path. Parity surface (§11): computable Lean definitions give test vectors.
2. **The F′ per-application split already exists.** Lean emits one pinned verifier blueprint with
   14 child row families (`pi_ccs_poseidon`, `pi_ccs_ordinary`, `pi_rlc_sampler_*`,
   `pi_rlc_combination`, `pilot_digest_binding`, ..., `application`). Rust
   (`crates/nightstream/src/assembly/`) relocates the exported rows and adds the application's
   rows. Lean builds the fixed part once; Rust adds what changes.
3. **F′ already has proved logical circuits for the whole layer-0 replay.**
   `Lifecycle/PiCCS/v1_1/Formal.lean` is the one PiCCS circuit (statement binding with the state
   hash and the canonical-split sign bits, absorption, alpha/gamma, the round transcript and
   sum-check chain, the Eval_K/Eval_A/CCS/norm terminals, the final identity, output binding).
   `Lifecycle/PiRLC` has the sampler (base-5 decoder: the DSL has `quotientFive` and
   `remainderFive` hints) and the combination (commitment, public input, Eval_K, Eval_A). Each is
   a `FormalCircuit` (`main`, `assumptions`, `spec`, `soundness`, `completeness`) over `Expr`
   inputs. `AssemblerSoundness.lean` says F′ does "not include the outer terminal verifier": that
   verifier is exactly the compression's layer 0.
4. **Row form.** F′ rows use the gate `a·b − c + x^7` (4 matrix slots). The shrink row is
   `X^7 + Σ_{i<3} A_i·B_i − C`, so every F′ row is a shrink row.
5. **Size.** F′'s PiCCS circuit is 1,076,114 rows in F′ form (theorem `totalRowCount_eq_fixed`);
   838,440 of them are output-binding absorption, mostly Poseidon2. The M3 shrink is 2^23 rows
   (33,132 Poseidon2 blocks, 629 ring products, 1,480,723 glue rows). Flat import of F′ rows
   would pass 2^23. Poseidon2 calls must stay calls, so that the shrink places them as blocks.
6. **What changes per application** (M3 measurement): the layer-1 verifier's loops (matrix runs,
   slots, segments, logUp tree depths, WHIR and setup sizes), the setup root, the CCS polynomial
   and the verifier context. Layer 0 has the fixed v1.1 sizes.
7. **Build.** `formal/nightstream-fprime` is about 305k lines. A built `.lake` can be cloned from a
   sibling worktree in seconds. Lean commands run under the 25-minute cap. Another session works
   in `formal/nightstream-fprime`.
8. **The M3 circuit core** (`neo-spartan/src/circuit/`): the program is written once over
   `Backend`. `Native` computes; `Recorder` emits rows. Block kinds (Poseidon2, ring product) are
   uniform templates with closed forms in the shrink layout. Everything else is glue rows that the
   verifier rebuilds with a shape run.

## Problem

The M3 shrink circuit is all Rust. Its soundness depends on Rust code that emits every row; a
missing constraint anywhere breaks it. The owner wants Lean to be the authority for the fixed
parts, as in F′, and Rust only for the parts that change per application. Three facts shape the
answer: F′ already proves the layer-0 replay circuits, the F′ contract forbids Lean certifying
Rust-made artifacts, and the layer-1 loops change per application and must stay Rust.

## Usage (caller's view)

The public API does not change (`finish_with_spartan`, `CompressionKey::verify`,
`FinalProof`). What changes is inside the final program:

```rust
// crates/nightstream/src/lifecycle/terminal/mod.rs, after the change
fn run<B: Backend>(&self, b: &mut B) -> Result<(), Error> {
    let statement = self.statement_words(b);              // 9 public words
    let proof = TerminalWords::read(b, self.proof)?;      // proof words, as now
    // Layer 0: one Lean-built template, proved against the Lean terminal spec.
    let out = b.apply(templates::terminal(), &[statement, proof.words()].concat())?;
    let claim = ClaimWords::from_terminal(&out);          // parent claim words
    let transcript = FoldTranscript::from_state(out.transcript());
    // Layer 1: Rust, per application, built from Lean-built primitives.
    neo_spartan::verify_with(b, self.relation, transcript, &claim, self.proof.map(|p| &p.layer1))
}
```

```rust
// crates/neo-spartan/src/circuit/record.rs, after the change: a primitive is a template call
fn ext_mul(&mut self, a: [Form; 3], b: [Form; 3]) -> [Form; 3] {
    self.call(templates::ext_mul(), &[a, b].concat()).try_into().expect("three outputs")
}
```

## Shape

**One new concept: a Lean-built template.** A template is a fixed list of shrink rows over its
own cells, with input cells, output cells, a witness program for the other cells, and nested
calls to other templates (the Poseidon2 permutation). Lean builds each template from a proved
`FormalCircuit` and emits it. Rust loads the emitted file and checks its pinned Poseidon2 digest.

**One new `Backend` method:** `apply(template, inputs) -> Result<outputs>`.
- `Native`: run the template's witness program, check every row, return the outputs. (It can
  also use native code for speed, then check the rows.)
- `Recorder`: substitute the input forms into the template rows (no copy rows; F′ subcircuits
  take `Expr` inputs in the same way), add cells for the witness, and record a nested Poseidon2
  call as a Poseidon2 block.

**Who owns what:**

| Part | Owner | Proof |
| --- | --- | --- |
| Layer-0 replay (statement, state hash, canonical split, PiCCS, PiRLC sampler and combination, layer-1 handoff) | Lean template `terminal`, built from F′ `PiCCS.v1_1.Formal` and the PiRLC circuits | `rows ⇒ Spec.Terminal` and completeness, in Lean |
| Poseidon2 block | Lean template from the F′ permutation gadget | Lean |
| Ring-product block | Lean template (new `FormalCircuit`) | Lean |
| Layer-1 primitives (bit split with canonical check, cubic `Ext` multiply and inverse, select, p3 duplex step, Merkle step) | Lean templates (new `FormalCircuit`s, cubic `Ext` is new in Lean) | Lean |
| Layer-1 loops and wiring (per application) | Rust (`neo-spartan/src/verifier.rs`, `whir/`, `matrix.rs`, ...) | Rust tests; differential vectors |
| Linear forms (sums of wires with constants) | Rust (`Recorder` forms) | Linear only; no nonlinear row is Rust-made after migration |
| Shrink layout, closed forms, argument | Rust | Lean proofs of the closed forms (carry walk, part placement) for all sizes |
| Security budget | Lean theorem; WHIR, SuperSpartan, Fiat-Shamir across layers as named assumptions | Lean |

**Trust boundary after the change:** every nonlinear row comes from a Lean template. Rust adds
only template calls, linear forms and per-application loop structure. The remaining Rust trust
is that the layer-1 wiring implements the layer-1 verifier; differential vectors check it (the
Rust `Native` layer-1 verifier against the existing native reference, and the terminal template
against Lean's computable spec on golden vectors).

**Deliberately not done:** Lean does not build the per-application layer-1 circuit, and Lean
never runs at setup or proving time.

**Interface depth:** callers see one method (`apply`) and a fixed set of named templates; the
template format, witness execution, nested calls and digest pin stay inside `circuit/`.

## Synthesis decision

Three structurally different designs:

- **A. Templates (chosen).** Lean builds fixed templates (the whole layer-0 replay and every
  nonlinear primitive); Rust calls them and keeps only per-application wiring. It follows the F′
  contract (Lean builds, Rust executes) and reuses F′'s proved PiCCS and PiRLC circuits.
- **B. Lean mirror proofs.** Rust keeps building rows; Lean re-implements the same row builders
  and proves them, and tests compare rows. Less plumbing, but Rust stays the builder and Lean
  proves a copy, which is the "second relation" the F′ contract forbids. Rejected.
- **C. Lean builds the whole circuit.** A compiled Lean executable builds the full shrink
  circuit, layer 1 included, for each application at setup. Most authority, but it needs Lean
  specifications for the layer-1 verifier and WHIR, and Lean on the setup path. The owner judged
  Lean generation per application too slow. Rejected.

Adapted from B: golden vectors at the template boundary (Rust `Native` against Lean's computable
spec). Adapted from C: the closed forms of the shrink layout are proved in Lean for all sizes.

## Tradeoffs accepted

- We accept a Lean-to-Rust template format (one more emitted artifact and decoder) in exchange
  for Lean authority over every nonlinear row.
- We accept that layer-1 wiring stays Rust in exchange for one shrink circuit per application
  without Lean at setup.
- We accept a larger circuit if F′'s layer-0 rows are bigger than the M3 Rust rows, in exchange
  for reusing proved circuits. The template format keeps Poseidon2 calls as blocks to limit it.
- We accept a separate Lean package in exchange for not editing F′ files that another session
  owns.

## Alternatives considered

- B and C above.
- Import F′'s already-exported blueprint children directly into the shrink (no new Lean). Their
  inputs are F′ layout columns (pilot, endpoint), and their proved results are stated only
  inside F′'s composition, so a terminal use needs a new Lean composition anyway. Rejected.

## Open questions and risks

1. Location: a new package `formal/nightstream-compression` that requires
   `formal/nightstream-fprime` (no edits to F′ files). Is that acceptable?
2. Can F′'s `PiCCS.v1_1.Formal` and PiRLC circuits be used outside F′ as is? Their
   `assumptions` (range and shape facts that other F′ children provide) must be discharged by the
   terminal composition. This is the first risk to retire.
3. Size: with Poseidon2 calls as blocks, does the shrink stay at 2^23 rows? If it goes to 2^24,
   the proof grows by about 200 bytes, prover sum-check time doubles, and the verifier time grows.
4. The cubic `Ext` arithmetic is new in Lean (F′ has only `K`).
5. Template witness programs: reuse F′'s exported witness-recipe format and executor (in
   `nightstream-fprime`), or a small format owned by `neo-spartan`?

## Next implementation step

Spike in `formal/nightstream-compression`: compose F′'s `PiCCS.v1_1.Formal` and the PiRLC
circuits into one `Terminal` `FormalCircuit`, state `Spec.Terminal`, and compute its row, cell and
Poseidon2-call counts, to retire risks 2 and 3 before any Rust change.

## Spike result (2026-10-08)

Built against a cloned F′ cache (same tree as `pr-122-review`; `lake build` of
`Lifecycle.PiCCS.v1_1.Formal` was current in 5.6 s). No code written yet.

- **Risk 2 retired for reuse:** the assumptions of every PiCCS child are wiring conditions only
  (`VarsBelow offset`: inputs refer to earlier cells). PiRLC's are of the same kind. So the F′
  PiCCS circuit (`soundness`, then `spec_implies_phaseHolds` to `Spec.Folding.PiCCS.Accepted`,
  the round point and the outgoing state) and the PiRLC circuit can be used outside F′ if their
  inputs are cells made earlier.
- **Risk 3 materialized:** F′ does not keep a Poseidon2 permutation as a sub-circuit. Each one is
  a fixed 1,096-cell recipe program written inline (`Gadgets/Poseidon2/Permutation/Owned.lean`,
  `Duplex/Formal.lean`). Sizes: PiCCS 1,076,114 rows (838,440 in output binding), PiRLC 83,725
  rows and 106,505 cells, prior state hash = one raw hash of 27,819 words (about 2,319
  permutations, about 2.54M cells). The terminal check needs about 3,300 permutations: about 3.6M
  cells inline, against about 0.63M as 192-cell shrink blocks today.
- **Consequence:** reuse as it is moves the shrink from 2^23 to 2^24 (estimate: proof about 92-95
  KB, shrink prover about 2x). The verifier stays near 270 ms only if the exporter turns F′'s
  inline permutation programs into a block kind; as flat rows it would pass 400 ms.
- **Options:** (1) reuse as it is, export the inline permutations as a third block kind;
  (2) first make the permutation an opaque sub-circuit in F′ (F′ change, other session's lane),
  then call the 182-cell block and stay at 2^23; (3) Lean builds only the primitives and blocks,
  layer 0 stays Rust with golden vectors from F′'s computable spec.


## Implementation status (2026-10-08, option 1 "reuse as is")

Done (local commit, not pushed):

- **Lean (`formal/compression-lean`).** The terminal package is the F′ Stage 1
  package restricted to the prior-state chain and pilot rows, PiCCS and the
  PiRLC sampler and combination rows. It adds no row, column or template.
  `circuitPackage_implies_terminal`: if every row holds, the production key
  accepts PiCCS, its PiRLC parent is the instance in the output columns, the
  prior state hashes to the fresh input, and the sampler state is
  `PiRlcSampler.Transcript.stateAt (PiCCS outgoing state) 17`. The output map
  (`Outputs.lean`) has one theorem per family that ties its columns to these
  values. 18 theorems; axioms only `propext`, `Classical.choice`, `Quot.sound`.
- **Artifact.** `[1, package, outputs]`, 39.6 MB, emitted in 10.6 s. Rust pins
  its identifier.
- **Rust loader** (`nightstream-fprime/src/package/terminal.rs`). Every check
  of a full package except physical coverage; the witness intervals must not
  overlap.
- **Differential check** (production fold, ignored test
  `terminal_package_accepts_the_last_fold`). The F′ witness program accepts
  the honest inputs of the last fold, so all 3,302 permutations, the 5.13M
  compact PiRLC rows and every assertion hold. The output columns equal the
  native parent (commitment, public input, point, `Eval_K`, `Eval_A`) and the
  native transcript state after PiRLC. Load 4.3 s, witness 0.14 s.

Correction to the spike estimate: F′ builds one ring product in 8,154 rows
(54 lane templates), not in the Formal circuit's count. The PiRLC combination
is 629 × 8,154 = 5,128,866 rows. The layer-0 rows are about 8.83M (prior chain
2.54M, 982 permutations 1.08M, compact 5.13M, ordinary rows about 0.09M).
With layer 1 the shrink still fits 2^24 by count, with a small margin.

Next:

1. `neo-spartan`: a fixed R1CS block over caller wires (`Backend` method);
   `Recorder` places its rows as glue and evaluates them in parallel in the
   verifier's shape run.
2. `nightstream`: the final program feeds the package inputs (proof words,
   statement-bound preimage words, the context as constants), takes the
   produced columns as one hint, applies the rows, and starts layer 1 from the
   output columns; delete `terminal/{state,pi_ccs,pi_rlc,k}.rs`.
3. Measure rows, cells, proof size, prover time and verify time at
   production. If verify passes 400 ms, give the F′ permutation a block kind
   with a closed form, then the compact lane templates.
