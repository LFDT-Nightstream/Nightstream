# Nightstream Lean formal-verification specification

- Status: normative bootstrap specification
- Version: 0.1
- Date: 2026-07-09

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
| Rust carrier state | `crates/neo-fold-clean/src/paper/construction2/state.rs:43-90` | Active state representation |
| Rust transition | `crates/neo-fold-clean/src/paper/construction2/transition.rs:82-207` | Branch checks, state advance, `x_out` |
| Rust native F' | `crates/neo-fold-clean/src/paper/f_prime/native.rs:127-446` | Prover/verifier control flow and failures |
| Rust recursive circuit | `crates/neo-fold-clean/src/paper/f_prime/r1cs.rs:812-1130` | Enforced recursive-step constraints |

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
| `REL-CCS` | Commitment, public projection, norm, and CCS satisfaction define membership | `SuperNeo.CCS.Holds` | SuperNeo Definition 12 | `specified` |
| `REL-CE` | Commitment, projection, norm, and all matrix evaluations define membership | `SuperNeo.CE.Holds` | SuperNeo Definition 13 | `specified` |
| `REL-CONCRETE` | Goldilocks, ring, norm, MLE, projection, and Ajtai operations instantiate the relation semantics | concrete semantics instance | Rust relations and paper definitions | `planned` |
| `PARAM-GLOBAL` | Verifier-owned global parameters own `b`, `k`, `B = b^k`, `K_max`, `T`, the Definition-14 inequality `(K+k)·T·(b−1) < B`, the norm stages (`b` fresh / `B` combined / `q/2` extraction-ambient), and the binding regime (`(2B,C)`-relaxed binding ← `MSIS` at `8TB`, Appendix B) | `SuperNeo.GlobalParams`, `SuperNeo.NormStage` | SuperNeo Definition 14, Theorem 2, Appendix B; Rust `Params::max_fresh_count` | `specified` |
| `SUM-CLAIM` | SumCheck truth is the actual `T = sum Q`; acceptance contains only verifier checks | `SumCheck.Claim.True`, `Transcript.Accepted` | PiCCS SumCheck | `planned` |
| `SUM-SOUND` | False SumCheck acceptance reduces to a bounded bad-challenge event | `sumcheck_sound` | Lund/Schwartz-Zippel boundary | `planned` |
| `FOLD-PICCS` | Valid `CCS^K x CE^k` inputs produce `CE(b)^(K+k)` outputs; the reduction is **strong** (Definition 10) | `piCCS_complete`, `piCCS_strong` | SuperNeo PiCCS | `planned` |
| `FOLD-PIRLC` | Commitment, input, witness, and evaluations are the same random linear combination into `CE(B)`; the reduction is **weak** (Definition 9) with respect to the commitment projection `φ` shared with PiCCS's strength — standalone knowledge soundness is deliberately NOT the target, and extraction lands in the ambient `CE(q/2)` stage (D.5) | `piRLC_complete`, `piRLC_weak` | SuperNeo PiRLC, Lemma 4, Appendix D.5 | `planned` |
| `FOLD-PIDEC` | Low-norm `CE(b)^k` children recompose exactly to the `CE(B)` parent commitment, input, witness, and evaluations; independently a **reduction of knowledge** (Theorem 7) — the post-decomposition relation returns to the `b` stage, not the ambient bound | `piDEC_complete`, `piDEC_knowledgeSound` | SuperNeo PiDEC, Theorem 7 | `planned` |
| `FOLD-COMPOSE` | The strong PiCCS composes with the weak PiRLC over the shared `φ` (Theorem 6), then PiDEC, to implement the concrete multi-fold contract | `strongWeakComposition`, `superNeoFold_correct` | SuperNeo Theorem 6, folding scheme | `planned` |
| `FPR-ENVELOPE` | Base/active tag, counters, `pc`, immutable coordinates, and trace copy are coherent | `FPrime.Envelope.check_sound` | Rust state/transition helpers | `model-proved` |
| `FPR-BASE` | The base branch uses the default running instance and enforces the true initial state | `fPrimeBase_sound` | HyperNova Construction 2, Rust base branch | `planned` |
| `FPR-BASE-SPEC` | Rust's empty `RunningInstance` is a valid zero-arity specialization of HyperNova's default instance `u_⊥` — the specialization is a theorem, not an assumption | `emptyRunning_realizes_default` | HyperNova Construction 2 step 3, Rust `RunningInstance::default` | `planned` |
| `FPR-COUNTER-REFINE` | The paper's single step index `i` refines to Rust's `(chunk_count, step_count)` pair under an explicit refinement relation (`chunk_count` counts F' invocations, `step_count` sums fresh batch cardinalities) | `counter_refinement` | HyperNova Construction 2, Rust `advance_state` | `planned` |
| `FPR-RECURSIVE` | The recursive branch checks prior `x_out`, runs NIFS.V, advances application state, and installs the next fresh instance | `fPrimeRecursive_sound` | HyperNova Construction 2, Rust recursive branch | `planned` |
| `FPR-HASH` | `x_out` binds every authority-bearing coordinate with canonical Poseidon2 domain separation | `xOut_binding_or_collision` | Rust `compute_x_out` | `planned` |
| `TRACE-VALID` | Repeated valid F' steps yield exact-step reachability | `Assurance.Reachable`, trace induction theorems | HyperNova compiler | `specified` |
| `CIR-SOUND` | Every satisfying generated F' R1CS assignment implies the same F' step relation | `fPrimeCircuit_sound` | Rust recursive circuit | `planned` |
| `CIR-COMPLETE` | Every valid supported F' step has a satisfying circuit witness | `fPrimeCircuit_complete` | Rust witness generation | `planned` |
| `ENC-CANON` | Byte/field encodings are canonical, length-checked, and injective on accepted values | encoding theorem family | Rust serializers and `enc_inst` | `planned` |
| `TERM-CE` | Terminal acceptance binds commitment, public projection, norm, ring evaluation, constant term, and child authority | `terminalCE_sound` | terminal CE verifier | `planned` |
| `DEC-SOUND` | Decider acceptance implies the terminal relation or a named decider failure | `decider_reduce` | Spartan/decider boundary | `planned` |
| `RUST-REFINE` | Native Rust success and rejection paths refine the Lean executable model | refinement theorem/artifact family | `native.rs`, lifecycle verifier | `planned` |
| `VERIFY-REDUCE` | Verifier acceptance implies `ValidExecution` or `BadEvent` | `VerifierReductionTarget` realization | public verifier | `specified` |
| `BAD-BOUND` | The union of named bad events is negligible under explicit assumptions | final security theorem | complete security boundary | `planned` |

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

`formal/deprecated` is read-only reference material.

- No active module may import it.
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
| M1: concrete relations | `REL-CCS`, `REL-CE`, and `REL-CONCRETE` at least `model-proved` |
| M2: SuperNeo fold | `SUM-*` and `FOLD-*` properties at least `model-proved` |
| M3: F' semantics | `FPR-BASE`, `FPR-RECURSIVE`, `FPR-HASH`, and trace induction at least `model-proved` |
| M4: circuit correspondence | `CIR-SOUND`, `CIR-COMPLETE`, and `ENC-CANON` at least `artifact-checked` |
| M5: implementation conformance | `RUST-REFINE` and terminal properties `rust-conformant` |
| M6: end-to-end security | `VERIFY-REDUCE` and `BAD-BOUND` `security-reduced` |

M0 is the current project state. No later milestone is currently claimed.

## 16. Change control

Any change to a mapped paper definition, Rust state field, transcript preimage,
circuit constraint, public encoding, or verifier branch reopens affected
property IDs. Their evidence state returns to the highest level still supported
by current artifacts.

Changes to this specification require the same review discipline as changes to
protocol-critical code. The property matrix and evidence ledger must remain
consistent.
