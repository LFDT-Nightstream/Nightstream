# Nightstream Lean formal-verification specification

- Status: normative active specification
- Version: 0.4
- Date: 2026-07-10

This file governs the structure, claims, evidence, and completion criteria of
the active formalization in `formal/nightstream-lean`. The README is only an
overview. Per-property specs may add detail, but they may not weaken this file.

## 1. Objective

The project exists to establish that the production verifier accepts only valid
Nightstream executions, except with probability bounded by explicit
cryptographic failure events.

Lean code is useful only when it contributes to this assurance chain:

```text
paper relation
    ↓ faithful formal statement
Lean protocol semantics
    ↓ proved preservation and soundness
generated R1CS semantics
    ↓ checked circuit correspondence
Rust prover/verifier behavior
    ↓ state, encoding, transition, and failure-path refinement
terminal acceptance
    ↓ reduction to validity or named bad event
valid recursive execution
```

The project is not measured by theorem count, lines of Lean, or absence of
`sorry`. It is measured by closed assurance properties with explicit source
mapping and conformance evidence.

## 2. Governing principles

1. Authority comes from checked witnesses, relations, transcripts, and
   verifier-owned configuration. A digest is only compression.
2. Claim truth and verifier acceptance are separate predicates.
3. One semantic relation is shared by the native model, circuit, and verifier.
4. Completeness, deterministic correctness, and cryptographic soundness are
   separate theorem families.
5. A Lean-model theorem is not a Rust-refinement theorem.
6. Every assumption names an irreducible mathematical or cryptographic
   boundary. Assumptions may not contain local protocol conclusions.
7. Deprecated modules are evidence to mine, never dependencies to import.
8. The current implementation is the concrete `pc = 1` direct-F' specialization.
   General multi-program NIVC is deferred until the concrete path is assured.

## 3. Sources of truth

The following sources must be reconciled rather than silently prioritized:

| Source | Location | Owns |
|---|---|---|
| SuperNeo paper | `docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md:3-147` | CCS, CE, PiCCS, PiRLC, PiDEC |
| HyperNova multi-folding | `docs/hypernova-paper/08_3_Multi_folding_schemes.md:3-30` | Abstract multi-fold interface and soundness shape |
| HyperNova compatibility | `docs/hypernova-paper/13_6_2_NIVC_Compatible_multi_folding_schemes.md:3-20` | Encoder and default-instance obligations |
| HyperNova Construction 2 | `docs/hypernova-paper/14_6_3_A_compiler_from_NIVC_compatible_folding_schemes_to_NIVC.md:3-65` | F' state, recursive link, compiler, verifier |
| Rust carrier state | `crates/neo-fold-clean/src/paper/construction2/state.rs` (`State`) | Active state representation |
| Rust transition | `crates/neo-fold-clean/src/paper/construction2/transition.rs` (`state_base_case_check`, `advance_state`) | Branch checks, state advance, `x_out` |
| Rust native F' | `crates/neo-fold-clean/src/paper/f_prime/native.rs` (`prove_with_semantic_state`, `verify`) | Prover/verifier control flow and failures |
| Rust recursive circuit | `crates/neo-fold-clean/src/paper/f_prime/r1cs.rs` (`enforce_f_prime_recursive_step_circuit`) | Enforced recursive-step constraints |

If the paper, Lean model, circuit, and Rust disagree, the work is classified as
one of: `code-first`, `model-first`, `harness-first`, or `theorem-first`. The
disagreement must be resolved or recorded as an explicit implementation
specialization. It may not be hidden by a bridge assumption.

## 4. Threat and authority model

The adversary controls:

- proof bytes and prover messages;
- claimed fresh CCS instances and witnesses;
- serialized public inputs not fixed by verifier configuration;
- ordering, duplication, omission, and replay of instances;
- self-consistent mutation followed by recomputation of non-authoritative
  digests;
- malformed base/active tags, counters, encodings, and field elements.

The verifier owns:

- verifier keys and protocol parameters;
- the CCS/R1CS structure and semantic-state mode;
- transcript domain separators and challenge derivation;
- canonical decoding rules;
- the terminal statement and expected public inputs.

The following are never authority by themselves:

- a digest or accumulator handle;
- a prover-supplied mode, flag, counter, or structure description;
- a self-consistent chain of hashes;
- a record field named `sound`, `valid`, `accepted`, or `authorized`.

Denial of service, timing leakage, allocator safety, and general Rust memory
safety are outside the initial Lean scope unless they change protocol logic.

## 5. End-to-end claim

The primary deterministic reduction target is represented by
`Nightstream.Assurance.VerifierReductionTarget`:

```lean
verify statement proof = true →
  ValidExecution step terminalValid initial final steps ∨
  BadEvent statement proof
```

`ValidExecution` must eventually mean all of the following:

- the trace starts from the advertised `z0` and initial semantic state;
- every application transition is valid;
- every F' base or recursive transition is valid;
- every fresh instance satisfies the concrete CCS relation;
- every running instance is exactly the output of the checked fold verifier;
- the recursive instance binds the prior `x_out` preimage;
- all counters, modes, structures, and program counters are verifier-authorized;
- the terminal accumulator satisfies the concrete CE relation;
- the terminal decider checks that CE relation;
- every public digest is recomputed from authoritative preimage data;
- the advertised final public state equals the trace endpoint.

The cryptographic closure theorem must separately bound `BadEvent`:

```text
Pr[BadEvent] ≤
  epsilon_sumcheck +
  epsilon_ajtai_binding +
  epsilon_poseidon2 +
  epsilon_fiat_shamir +
  epsilon_decider
```

No deterministic theorem may replace this probabilistic statement by defining
acceptance to contain claim truth.

## 6. Project structure and ownership

The project remains one Lake package with deep modules and a small root import.

```text
formal/nightstream-lean/
├── Nightstream/
│   ├── SuperNeo/          paper relations and folding reductions
│   ├── HyperNova/         generic multi-fold and Construction-2 semantics
│   ├── Protocol/          concrete SuperNeo + HyperNova specialization
│   ├── Implementation/    Rust, encoding, transcript, and R1CS models
│   └── Assurance/         reachability, terminal, and security theorems
├── tests/                 counterexamples, regressions, axiom checks
├── specs/                 this file and per-property proof contracts
└── assurance/             append-only proof and conformance evidence
```

| Layer | Owns | Must not own |
|---|---|---|
| `SuperNeo` | CCS/CE, SumCheck claim, PiCCS, PiRLC, PiDEC | Recursive state, Rust layout |
| `HyperNova` | Relation interface, multi-fold contract, compatibility, F' compiler semantics | Ajtai or SuperNeo-specific algebra |
| `Protocol` | `pc = 1` Nightstream instantiation and cross-paper composition | Rust serialization or R1CS builder internals |
| `Implementation` | Exact Rust-shaped state, encodings, transcripts, native and R1CS semantics | Cryptographic security conclusions |
| `Assurance` | Trace validity, terminal closure, reduction games, final theorem | New protocol algorithms |

`SuperNeo` and `HyperNova` are sibling foundations. `Protocol` may import both.
`Implementation` may import the paper model and concrete protocol. `Assurance`
may import all active layers. Imports in the opposite direction are forbidden.
Tests may import any active module. Nothing may import `formal/deprecated`.

An interface module is justified only by at least one real consumer. File paths
and namespaces should name the concept they own; there is no mirrored
`Spec/Impl/Security` hierarchy.

## 7. Required formal properties

Evidence states are defined in Section 8.

| ID | Property | Target | Source surface | State |
|---|---|---|---|---|
| `REL-CCS` | Commitment, public projection, norm, and CCS satisfaction define membership | `SuperNeo.CCS.Holds`, `Concrete.ccsMembership_iff` | SuperNeo Definition 12 | `model-proved` |
| `REL-CE` | Commitment, projection, norm, point-domain shape, and all matrix evaluations define membership | `SuperNeo.CE.Holds`, `Concrete.ceMembership_iff` | SuperNeo Definition 13 | `model-proved` |
| `REL-CONCRETE` | Goldilocks, quadratic/cyclotomic rings, centered norm, MLE, projection, and Ajtai operations instantiate the relation semantics | `SuperNeo.Concrete.relationSemantics` | Rust relations and paper definitions | `model-proved` |
| `PARAM-GLOBAL` | Verifier-owned global parameters own `q`, `b`, `k`, `B = b^k`, `K_max`, `T`, the Definition-14 inequality `(K+k)·T·(b−1) < B`, the norm stages (`b` fresh / `B` combined / `q/2` extraction-ambient), and the binding regime (`(2B,C)`-relaxed binding ← `MSIS` at `8TB`, Appendix B) | `SuperNeo.GlobalParams`, `Concrete.productionGlobalParams` | SuperNeo Definition 14, Theorem 2, Appendix B; Rust `Params::production`, `Params::max_fresh_count` | `model-proved` |
| `SUM-CLAIM` | SumCheck truth is the actual `T = sum Q`; acceptance contains only verifier checks, and the executable checker is equivalent to that logical predicate | `SumCheck.Claim.True`, `SumCheck.Accepted`, `SumCheck.check_eq_true_iff_accepted` | PiCCS SumCheck | `model-proved` |
| `SUM-SOUND` | False SumCheck acceptance reduces to a bounded-degree bad-challenge event | `SumCheck.false_acceptance_implies_bad_challenge` | Lund/Schwartz-Zippel boundary | `model-proved` |
| `FOLD-PICCS` | Valid `CCS^K x CE^k` inputs produce `CE(b)^(K+k)` outputs; the reduction is **strong** (Definition 10) | `PiCCS.product_complete`, `PiCCS.strong_extract_or_bad_challenge`, `PiCCS.repeated_outputs_same_phi` | SuperNeo PiCCS | `model-proved` |
| `FOLD-PIRLC` | Commitment, input, witness, and evaluations are the same random linear combination into `CE(B)`; the reduction is **weak** (Definition 9) with respect to the commitment projection `φ` shared with PiCCS's strength — standalone knowledge soundness is deliberately NOT the target, and extraction lands in the ambient `CE(q/2)` stage (D.5) | `PiRLC.complete`, `PiRLC.ExtractionOutcome`, `PiRLC.same_phi_extractions_unique_or_collision` | SuperNeo PiRLC, Lemma 4, Appendix D.5 | `model-proved` |
| `FOLD-PIDEC` | Low-norm `CE(b)^k` children recompose exactly to the `CE(B)` parent commitment, input, witness, and evaluations; independently a **reduction of knowledge** (Theorem 7) — the post-decomposition relation returns to the `b` stage, not the ambient bound | `PiDEC.complete`, `PiDEC.split_recompose_exact`, `PiDEC.reduce_knowledge` | SuperNeo PiDEC, Theorem 7 | `model-proved` |
| `FOLD-COMPOSE` | The strong PiCCS composes with the weak PiRLC over the shared `φ` (Theorem 6), then PiDEC, to implement the concrete multi-fold contract | `Composition.shared_phi`, `Composition.fold_knowledge_or_bad_event` | SuperNeo Theorem 6, folding scheme | `model-proved` |
| `FPR-ENVELOPE` | Base/active tag, counters, `pc`, immutable coordinates, and trace copy are coherent | `FPrime.Envelope.check_sound` | Rust state/transition helpers | `model-proved` |
| `FPR-BASE` | The base branch uses the default running instance and enforces the verifier-derived boundary/trace seeds, semantic mode, initial Nebula lane, empty accumulator, exact NoFold variant, nonempty output batch, and recomputed output. The installed batch's public link is a one-step-delayed consumer/terminal obligation, not a producer-step check | `Step.fPrimeBaseLocal_sound`, `Step.holds_iff_local_and_outgoing`, `Step.closeLocal` | HyperNova Construction 2, Rust base branch and terminal latest link | `model-proved` |
| `FPR-BASE-SPEC` | Rust's empty `RunningInstance` is a valid zero-arity specialization of HyperNova's default instance `u_⊥` — the specialization is a theorem, not an assumption | `Default.emptyRunning_realizes_default` | HyperNova Construction 2 step 3, Rust `RunningInstance::default` | `model-proved` |
| `FPR-COUNTER-REFINE` | The paper's single step index `i` refines to Rust's `(chunk_count, step_count)` pair under an explicit refinement relation (`chunk_count` counts F' invocations, `step_count` sums nonempty fresh-batch cardinalities); native overflow is rejected rather than wrapped | `CounterRefinement.counter_refinement` | HyperNova Construction 2, Rust `advance_state` | `model-proved` |
| `FPR-RECURSIVE` | The recursive branch pins prior authority, recomputes its running handle, checks the prior batch's recursive link, obtains the next running value from executable NIFS.V, checks semantic/Nebula advance, installs a nonempty new batch, advances state, and recomputes x_out. The newly installed batch is linked by its consumer or the terminal fold | `Step.fPrimeRecursiveLocal_sound`, `Step.holds_iff_local_and_outgoing`, `Step.closeLocal`, `Step.holds_advance_facts` | HyperNova Construction 2, Rust recursive branch and terminal latest link | `model-proved` |
| `FPR-HASH` | Equal canonical `x_out` outputs imply equality of every direct, verifier-derived, or equality-pinned authority coordinate (including the source Nebula lane), or an explicit outer-hash/inner-lane-digest collision | `XOut.xOut_binding_or_collision` | Rust `compute_x_out`, `state_x_out_digest_with_mode` | `model-proved` |
| `TRACE-VALID` | Retained accepted invocations yield exact-step rich-edge reachability, nonzero batch schedules, exact split-counter refinement, and a pinned final state | `FPrimeTrace.accepted_trace_sound`, `FPrimeTrace.accepted_trace_valid_execution` | HyperNova compiler | `model-proved` |
| `CIR-SOUND` | For the exact 4,076,614-row `FPrimeFullHistoryRows.fullRows` artifact, every canonical satisfying assignment yields a two-edge `ValidExecution` with direct terminal validity, or the recursive/terminal PiRLC projection exposes its named `BadRoot`. Scope is exactly plain/stateless `[1,1]`, one recursive invocation, terminal fold, direct terminal CE, and `minimal-supported-bit-carrier` | `Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_sound_or_bad` | Rust full-history audit synthesis, base/recursive circuits, terminal fold, and direct terminal CE | `artifact-checked` |
| `CIR-FPR-RECURSIVE-MANIFEST` | The exact 2,640,071-row plain/stateless steady-recursive profile is partitioned into eight contiguous top-level owners; its 2,572,208-row NIFS block is exactly partitioned into PiCCS, PiRLC, PiDEC, and point binding. The PiRLC projection census pins one 1,892-row shared block and 31 equal-shape 1,916-row identities with 15 pairs each. This ownership-only profile is distinct from the one-recursive-invocation `[1,1]` artifact closed by `CIR-SOUND` | `FPrimeRecursiveManifest.topLevel_covers_program`, `FPrimeRecursiveManifest.nifs_covers_block`, `FPrimeRecursiveManifest.projection_census_shape`, `FPrimeRecursiveCircuit.decodedChecks_sound` | `enforce_f_prime_recursive_step_circuit`, `enforce_nifs_v_circuit_with_transcript_inner` | `artifact-checked` |
| `CIR-PIRLC-PROJECTION` | Every canonical assignment satisfying the exact 714-row production helper implies the complete 107-coefficient `BatchAccepted` predicate. Every accepted bounded projection batch is then coefficient-wise exact or exposes a nonzero error polynomial vanishing at beta. Honest, bad-root, and row-forgery regressions exercise the universal theorem boundary; a reusable theorem lifts the semantics across a complete shared-definition census | `PiRLCProjection.exactRows_imply_batchAccepted`, `ProjectionTrace.census_batchAccepted`, `ProjectionCheck.batchAccepted_implies_exact_or_badRoot`, `FPrimeRecursiveCircuit.projectedChecks_local_sound_or_badRoot` | `enforce_ring_action_projection_batch` and the PiRLC beta schedule | `artifact-checked` |
| `CIR-U64CANON` | Satisfying the exact exported canonical-u64 gadget rows forces boolean bits that recompose over the integers to the decomposed element's canonical value | `canonicalU64_sound` | `decompose_var_to_u64_bits` generated rows | `artifact-checked` |
| `CIR-U64INC` | Satisfying the exact exported u64-increment rows forces the output word to equal the input word plus one over the integers and rejects wraparound | `u64Increment_sound` | `enforce_u64_increment` generated rows | `artifact-checked` |
| `CIR-U64ADD` | Satisfying the exact exported u64-add rows forces the output word to equal both input words' integer sum and rejects wraparound | `u64Add_sound` | `enforce_u64_add` generated rows | `artifact-checked` |
| `CIR-FPR-COUNTER` | Satisfying the exact production-used recursive F' counter block binds source words, fixes the batch cardinality, advances both counters over the integers, and rejects wraparound | `FPrimeCounterSound.fPrimeCounter_sound` | F' input-binding and recursive-counter generated rows | `artifact-checked` |
| `CIR-FPR-TERMINAL-LINK` | Satisfying the exact terminal-fold delayed-link rows fixes every trailing fresh affine-one slot and equates all 256 public bits to the last producer step's canonical `x_out` bits; empty/wrong-length shapes are rejected before emission | `FPrimeTerminalLinkSound.fPrimeTerminalLink_sound` | `engine::decider::enforce_terminal_latest_link` | `artifact-checked` |
| `CIR-FPR-STATE-LINK` | Satisfying the plain full-history state-link row program equates every verifier key/header lane, counter, boundary, program counter, semantic/accumulator digest lane, and public-trace lane across adjacent steps | `FPrimeStateLinkSound.fPrimeStateLink_sound` | `engine::decider::enforce_state_link` | `artifact-checked` |
| `CIR-FPR-BASE-PINS` | Satisfying the seeded plain base-state row program pins all 31 verifier-owned authority coordinates to preprocessing-derived constants | `FPrimeBaseStateSound.fPrimeBaseState_sound` | `engine::decider::enforce_base_state_constants` | `artifact-checked` |
| `CIR-FPR-BASE-PROGRAM` | The complete 12,498-row plain base-step artifact is a checked program with 10,900 deterministic definitions and 1,598 retained assertions; satisfaction fixes x_out and valid checked execution constructs a satisfying witness | `FPrimeBaseProgramSound.fPrimeBaseProgram_sound`, `fPrimeBaseProgram_xOut_unique`, `fPrimeBaseProgram_complete` | `enforce_f_prime_base_step_circuit` | `artifact-checked` |
| `CIR-POSEIDON2` | The exact 600-row production width-8 Goldilocks Poseidon2 permutation is deterministic and complete: satisfying assignments agree with its extracted SSA interpreter, equal inputs force equal outputs, and interpreting any canonical input constructs a satisfying witness | `Poseidon2PermutationSound.poseidon2Permutation_sound`, `poseidon2Permutation_outputs_unique`, `poseidon2Permutation_complete` | `r1cs_circuit::poseidon2` generated rows | `artifact-checked` |
| `CIR-FPR-CHUNK-BIND` | All 6,661 exact chunk-shape digest rows form a deterministic and complete straight-line program from constant-one/start-step inputs to the four public digest lanes; the final-four-row binding theorem remains separately available | `FPrimeChunkDigestSound.fPrimeChunkDigest_sound`, `fPrimeChunkDigest_claim_unique`, `fPrimeChunkDigest_complete`, `fPrimeChunkDigest_binding_sound` | `f_prime::digest_circuit::enforce_f_prime_chunk_public_digest_circuit` plus the F' branch equality rows | `artifact-checked` |
| `CIR-FPR-CE-CONTINUITY` | The exact one-claim continuity artifact directly equates all 1,297 PiDEC-child/PiCCS-running authority coordinates, including data omitted by compact accumulator digests | `FPrimeCeContinuitySound.fPrimeCeContinuity_sound` | `engine::decider::enforce_children_equal_running` | `artifact-checked` |
| `CIR-COMPLETE` | Every independent successful `CompilerWitness` for the exact fixed `CIR-SOUND` profile reassembles into an assignment satisfying all `FPrimeFullHistoryRows.fullRows`; the witness carries source/interpreter executions and direct semantic inputs, not `Satisfies` or a verifier conclusion | `Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_complete` | Rust witness generation and full-history audit synthesis | `artifact-checked` |
| `ENC-CANON` | Byte/field encodings are canonical, length-checked, injective on accepted values, and enforced by the exact production `enc_inst` rows | `Encoding.FPrime.encInst_injective`, `FPrimeEncodingSound.fPrimeEncoding_sound`, `FPrimeEncodingSound.accepted_public_bits_injective` | Rust digest decoding, serializers, and `enc_inst` | `artifact-checked` |
| `TERM-CE` | Direct terminal acceptance uses verifier-derived children and binds witness cardinality, public width, commitment, public projection, verifier-owned norm, evaluation-point shape, all ring evaluations, constant terms, and supported sidecars | `TerminalCE.terminalCE_sound`, `TerminalCE.terminalCE_complete`, `Rust.Terminal.success_refines_terminalCE`, `Rust.Terminal.invalid_has_named_rejection` | native terminal CE verifier | `rust-conformant` |
| `DEC-SOUND` | Decider acceptance implies the terminal relation or a named decider failure | `decider_reduce` | Spartan/decider boundary | `planned` |
| `RUST-REFINE` | Every currently supported native Rust verifier success/rejection path refines the Lean executable F' and terminal models; the compact entrypoint is pinned to its current fail-closed `Unsupported` contract | `Rust.FPrime.verify_eq_ok_iff_checkLocal`, `Rust.FPrime.success_with_outgoing_refines_step`, `Rust.FPrime.invalid_has_named_rejection`, `Rust.Terminal.success_refines_terminalCE`, `Rust.Terminal.invalid_has_named_rejection` | `native.rs`, uncompressed/audit lifecycle verifier, direct terminal verifier, compact fail-closed seam | `rust-conformant` |
| `VERIFY-REDUCE` | Verifier acceptance implies `ValidExecution` or `BadEvent` | `VerifierReductionTarget` realization | public verifier | `specified` |
| `BAD-BOUND` | The union of named bad events—including SumCheck collisions, PiRLC projection roots, projection-preimage binding collisions, sampling failures, relaxed-binding collisions, and hash collisions—is negligible under explicit assumptions | final security theorem | complete security boundary | `planned` |

The table is normative. A new protocol-critical theorem must either discharge an
existing ID or add a reviewed property here before implementation.

## 8. Evidence states

These labels have fixed meanings:

| State | Required evidence |
|---|---|
| `planned` | Property and source surface identified |
| `specified` | Lean predicate exists and matches a reviewed human specification |
| `model-proved` | Lean theorem builds, has an axiom report, negative witness where relevant, and proof hash |
| `artifact-checked` | The theorem covers the exact generated R1CS/encoding artifact rather than only a handwritten surrogate |
| `rust-conformant` | State, transitions, success and rejection paths, runtime regression, and drift gate pass |
| `security-reduced` | Acceptance reduces to validity or named bad events, with probability bounds and explicit assumptions |

`verified`, `sound`, `aligned`, or `complete` may not appear in status reporting
without naming the property ID and its evidence state.

## 9. Permitted assumptions and trusted base

The expected trusted or assumed boundaries are:

- the Lean kernel and selected standard-library foundations;
- explicitly identified mathematical facts not yet reconstructed locally;
- Module-SIS/Ajtai binding or relaxed binding security games;
- Poseidon2 collision resistance or random-oracle modeling where required;
- Fiat-Shamir transform assumptions;
- the final SNARK/decider soundness game;
- the Rust-to-Lean translation or artifact importer until it is itself verified.

At the M2 model boundary specifically:

- SumCheck receives an independent semantic polynomial path and proves false
  acceptance implies a sampled-root collision; it does not assume acceptance
  implies claim truth;
- PiRLC and PiDEC receive typed homomorphism, norm-growth, and exact
  split/recompose laws. Concrete production-algebra instantiation remains a
  later refinement obligation;
- Appendix-D.5 rewinding is represented by `WeakExtractor`, whose failure arm
  carries evidence of a fixed `SamplingBoundary.Failure` proposition, and by
  `UniquenessBridge`, whose failure result is a literal Definition-4 relaxed-
  binding collision;
- the deterministic composition theorem proves validity or those named events.
  It does not prove their probability bounds.

Cryptographic assumptions must be passed as typed parameters to the theorem that
uses them. They must state an adversary, experiment, event, and bound. A field
such as `accepted_implies_valid : Accepted -> Valid` is not a cryptographic
assumption; it is the local conclusion and is forbidden.

Every final theorem must publish its complete trusted computing base. Project
status must say which boundaries remain assumed.

## 10. Per-property proof contract

Before implementing a property, create or update a short spec containing:

```text
property_id:
claim:
assumptions:
non_goals:
paper_sources:
rust_surfaces:
circuit_or_encoding_artifacts:
failure_class:
counterexample_or_witness:
lean_theorems:
axiom_report:
proof_hash:
conformance_status:
retest_commands:
```

The claim must be understandable without reading the proof. Assumptions and
non-goals must appear in the Lean module contract header as well.

## 11. Proof and conformance workflow

Work one high-value property at a time:

1. Select one property ID and one concrete failure class.
2. Fill the per-property proof contract and source map.
3. Define the semantic relation before the checker or prover.
4. Construct a failing or forged witness for the missing check where possible.
5. Implement the smallest executable model that covers the mapped surface.
6. Prove completeness and/or soundness with the exact documented scope.
7. Run `#print axioms` and record a source hash.
8. Check statement, state, transition, encoding, and error-path parity.
9. Add or link a Rust/circuit regression before claiming conformance.
10. Append the result to `assurance/evidence-ledger.jsonl`.

If a proof blocks, classify it before editing:

- `code-first`: the implementation likely violates the intended property;
- `model-first`: the Lean representation is incomplete or wrong;
- `harness-first`: generated artifacts or tests are stale;
- `theorem-first`: the statement is false, too broad, or mis-scoped.

Proof difficulty is evidence. It is not automatically tactic debt.

## 12. Completion gates

A property may advance to `rust-conformant` only when all applicable checks pass:

1. Statement parity, including preconditions, postconditions, and failure modes.
2. State representation parity or a written proof that an abstraction preserves
   the property.
3. Transition parity for every relevant success and rejection branch.
4. Canonical encoding and generated-artifact parity.
5. A minimized counterexample, trace, or runtime regression.
6. Lean build, axiom report, and proof hash.
7. A drift check against the current mapped Rust and circuit sources.

Not-applicable checks must be justified, not silently omitted.

## 13. Required local gates

Every formal change runs:

```bash
cd formal/nightstream-lean
lake build
lake exe check
rg -n '\b(sorry|admit|axiom|unsafe)\b' Nightstream tests -g '*.lean'
```

Additionally:

- `tests/Axioms.lean` fails closed via `#guard_msgs`: the build breaks if any
  completed theorem's axiom report differs from the recorded expectation;
- `lake exe check` computes every printed result (envelope probes including the
  empty-step regression, Rust symbol anchors) and exits nonzero on failure —
  no unconditional success strings;
- active modules are checked for imports from `formal/deprecated`;
- proof hashes and conformance status are recorded in the evidence ledger;
- Rust or circuit commands are property-specific and belong in the per-property
  spec rather than a generic command list.

## 14. Deprecated-code policy

The legacy packages `formal/superneo-lean`, `formal/direct-ccs-fprime-lean`,
and `formal/twist-shout-lean` are read-only reference material. They may be
physically consolidated under `formal/deprecated` later, but their path does
not grant them authority.

- No active module may import a legacy package or `formal/deprecated`.
- No mass port is allowed.
- A reused definition or lemma is copied into the active ownership layer,
  restated against the active semantic types, and re-proved or revalidated.
- Old theorem names and layouts carry no compatibility requirement.
- Counterexamples and low-level algebra are preferred reuse candidates; wrapper
  surfaces and assumption bundles are not.

## 15. Roadmap

| Milestone | Exit condition |
|---|---|
| M0: assurance foundation | Canonical spec, active relation/state types, one model-proved property, evidence ledger |
| M0.5: vertical artifact slice | Versioned artifact schema, one exact production-used composed R1CS block, universal theorem, Rust/Lean adversarial vectors, and drift gate |
| M1: concrete relations | `REL-CCS`, `REL-CE`, and `REL-CONCRETE` at least `model-proved` |
| M2: SuperNeo fold | `SUM-*` and `FOLD-*` properties at least `model-proved` |
| M3: F' semantics | `FPR-BASE`, `FPR-RECURSIVE`, `FPR-HASH`, and trace induction at least `model-proved` |
| M4: circuit correspondence | `CIR-SOUND`, `CIR-COMPLETE`, and `ENC-CANON` at least `artifact-checked` |
| M5: implementation conformance | `RUST-REFINE` and direct `TERM-CE` `rust-conformant`; every unsupported public path is explicitly fail-closed |
| M6: end-to-end security | `DEC-SOUND`, `VERIFY-REDUCE`, and `BAD-BOUND` `security-reduced` |

M0, M0.5, M1, M2, M3, M4 for its advertised fixed profile, and M5 satisfy
their stated exit conditions. M2 is model-proved over
typed algebra, rewinding, arithmetization, sampling, and relaxed-binding
boundaries. M3 is model-proved over explicit executable hash, transcript-bound NIFS, application,
fresh-link, running-digest, chunk-digest, and Nebula semantics. Its theorem
scope includes true initialization, exact base/recursive local obligations,
the one-step-delayed consumer/terminal fresh-link closure, collision-explicit
`x_out` authority, and exact closed-trace induction. M2/M3 are not
artifact-checked or security-reduced. M4 has artifact-checked `CIR-SOUND` for
one exact profile: the 4,076,614-row plain/stateless `[1,1]` full-history
artifact with one recursive invocation, terminal fold, direct terminal CE, and
the minimal-supported-bit-carrier relation. Satisfaction of its exact
`fullRows` list yields a two-edge `ValidExecution` and direct terminal
validity, or one of the separately named recursive/terminal PiRLC root events.
The probability of either root event is an M6 obligation; it is not hidden
inside deterministic circuit correspondence. `CIR-COMPLETE` is also
artifact-checked for this profile: independent successful compiler executions
reassemble into satisfaction of every exact `fullRows` row. M4 therefore meets
its exit condition for the advertised fixed profile only. Stateful, Nebula,
other schedules, multiple recursive invocations, alternate carriers, and
parameterized circuit families are outside the claim. M5 independently closes
conformance for the supported
uncompressed/audit lifecycle and direct terminal CE verifier: universal Lean
success/rejection theorems, executable negative witnesses, real Rust replay,
and full-file drift hashes all pass. This does not broaden M4 beyond its fixed
profile. The compact Spartan decider remains explicitly `Unsupported`, so
`DEC-SOUND` and all M6 claims remain open; changing that Rust branch
automatically reopens M5 through the drift gate.

## 16. Change control

Any change to a mapped paper definition, Rust state field, transcript preimage,
circuit constraint, public encoding, or verifier branch reopens affected
property IDs. Their evidence state returns to the highest level still supported
by current artifacts.

Changes to this specification require the same review discipline as changes to
protocol-critical code. The property matrix and evidence ledger must remain
consistent.
