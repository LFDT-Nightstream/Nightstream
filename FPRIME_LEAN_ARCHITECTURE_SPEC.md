# Nightstream F′ Lean Architecture Specification

Status: proposed architecture contract.

Amendment 2026-10-31 (owner approval, option B, draft for review): §2, §3,
§5, §6, §9 and §10 change from one step circuit to two, a light circuit and a
refresh circuit. The Lean results in `NightstreamFPrime/Spec/Folding/Light`,
`Refresh`, `Schedule` and `Spec/HyperNova/Construction2/BoundedDepth*` are the
authority for the departures from SuperNeo v1.2.

This document defines the required result and its authority boundaries. It is
not an implementation plan.

## 1. Purpose

Nightstream needs a sound and auditable production F′ relation that the Rust
prover and verifier can rely on.

Lean must define the protocol relation, the logical circuit, the physical
layout, and the proofs that connect them. Lean must emit the canonical circuit
package that Rust uses. Rust must not define a second high-level F′ relation.

The architecture must also make constraint reduction reliable. Every row
family and physical column must have a clear semantic or layout owner. Lean
must make it possible to test proposed removals and column reuse with cvc5,
without making Rust's current constraint layout an authority.

The target is the smallest practical proved relation. This does not imply a
mathematical minimum unless Lean proves a lower bound.

## 2. Normative protocol sources

The implementation must preserve the obligations in these paper sections.

### HyperNova

- `docs/hypernova-paper/10_5_Non_uniform_incrementally_verifiable_computation.md`
- `docs/hypernova-paper/11_6_HyperNova_NIVC_from_multi_folding_schemes.md`
- `docs/hypernova-paper/12_6_1_Overview_of_HyperNova.md`
- `docs/hypernova-paper/13_6_2_NIVC_Compatible_multi_folding_schemes.md`
- `docs/hypernova-paper/14_6_3_A_compiler_from_NIVC_compatible_folding_schemes_to_NIVC.md`
- `docs/hypernova-paper/39_H_2_Proof_of_Lemma_3_Folding_CCS_NIVC_compatibility.md`
- `docs/hypernova-paper/40_H_3_Proof_of_Theorem_4_HyperNova.md`

### SuperNeo

- `docs/superneo-paper-v1_2/04_preliminaries.md`
- `docs/superneo-paper-v1_2/05_embeddings_and_evaluation_homomorphism.md`
- `docs/superneo-paper-v1_2/06_strong_and_weak_interactive_reductions.md`
- `docs/superneo-paper-v1_2/07_superneo_folding_scheme_for_ccs.md`
- `docs/superneo-paper-v1_2/08_concrete_parameters.md`
- Appendix B.1 through B.4 in
  `docs/superneo-paper-v1_2/11_appendix_B_deferred_theorems_and_proofs.md`

### Nebula

- `docs/nebula-paper/02_2-preliminaries.md`
- `docs/nebula-paper/03_3-commitment-carrying-ivc.md`
- `docs/nebula-paper/04_4-efficient-read-write-memory-in-ivc.md`
- `docs/nebula-paper/05_5-nivc-using-a-universal-switchboard-circuit.md`
- `docs/nebula-paper/09_appendix-a.md`
- `docs/nebula-paper/11_appendix-c.md`

### Cyclo

- Cyclo, ePrint 2026/359, Theorem 3 item 2 and Section 2.2: folding without
  a norm check on the accumulator, for a bounded number of folds, then a
  refresh. SuperNeo v1.2 Section 1.3 states that SuperNeo is compatible with
  this contribution.

Coral is reference material only. Nightstream can use its ideas to understand
memory soundness. This specification does not require a Coral implementation,
a Coral SNARK, or Coral's complete memory checker.

## 3. Fixed production profile

The production profile is:

- Goldilocks field;
- `b = 2`;
- `k_rho = 16`;
- `B = 2^16`;
- two step circuits, light and refresh, in HyperNova Construction 2 (NIVC),
  with one running accumulator for each circuit;
- each running accumulator carries its depth, the number of light folds since
  its last refresh, and the light circuit rejects a fold at depth 64 (the cap);
- light circuit: one fresh source and one running source, 2 PiRLC inputs in
  fresh-then-running order with coefficient 1 on the running source, PiCCS norm
  terms on the fresh source only, no PiDEC;
- refresh circuit: PiDEC of the input accumulator into 16 children, then 17
  PiRLC inputs in exact `K + k` order, no PiDEC of the output;
- 4 CCS matrices;
- `κ = 22` commitment rows, with relaxed binding at the chain norm
  `8 · T · (B + 64 · T · (b − 1)) = 137,134,080`;
- Poseidon2-only protocol binding.

The two step circuits are one profile. The production package must bind this
profile exactly. It must not contain a radix-four or alternate-profile path.

## 4. Authority model

The authoritative path is:

```text
paper obligations
→ concrete Lean protocol semantics
→ Lean logical circuit builders
→ proved physical layout and lowering
→ canonical Lean-emitted circuit package
→ Rust prover and verifier
```

Lean is the authority for semantics, constraints, layouts, and deterministic
proofs.

Rust is the authority for efficient execution of the emitted package. Rust
owns the generic proof engine, package loading, and optimized primitives. It
does not own the F′ relation or its circuit structure.

cvc5 is an untrusted search and attack tool. It can find counterexamples and
redundancy candidates. It cannot authorize a protocol claim or a removal.

A digest is compression and binding evidence. It is not semantic authority.
Every protocol-binding digest must use Poseidon2 and must be recomputed from
authoritative data.

## 5. Stage 1: HyperNova over SuperNeo

Stage 1 must be a complete usable HyperNova system over the SuperNeo folding
scheme.

It must define the complete base, recursive, and terminal F′ lifecycle. It
must include:

- complete prior-state and public-input binding;
- CCS and LCCCS multi-folding;
- the light path PiCCS → PiRLC and the refresh path PiDEC → PiCCS → PiRLC;
- exact PiRLC ordering, sampling, evaluation, and transcript binding;
- in the refresh circuit, PiDEC decomposition, low-norm extraction, and all
  16 children;
- the depth guard and the depth update of each running accumulator, bound into
  the state hash;
- commitment ownership and opening checks;
- accumulator and running-instance transitions;
- Poseidon2 transcript and `XOut` binding;
- explicit recursive-size closure for each step circuit;
- a Lean-proved joint domain no larger than `2^28` for each step circuit.

Stage 1 must not contain a free memory predicate or an assumed Nebula phase.
Its public layout must follow its concrete semantics. The current 32-field
`XOut` preimage is a Nightstream implementation choice, not a paper rule, and
is not automatically a Stage 1 requirement.

## 6. Stage 2: Nebula as a new phase

Stage 2 must add one concrete Nebula memory phase to the completed Stage 1
system. Both step circuits carry this phase.

The phase must own the commitment-carrying memory state, presence transitions,
memory-state transitions, arm selection, and terminal memory acceptance. It
must use one common public input and exactly one selected active arm.

Stage 2 must reuse the proved HyperNova and SuperNeo core. It must not duplicate
or replace the Stage 1 core proofs.

Stage 2 still requires a new Nebula logical builder and an extended composition
theorem. These additions must use the existing Stage 1 results as closed
inputs.

If Nebula changes the public layout, private geometry, recursive instance, or
terminal relation, the extended system needs a new composition theorem,
fixed-point theorem, and domain theorem. Pinned public fields alone do not
prove that the complete Nebula geometry is unchanged.

## 7. Logical constraints and physical columns

The logical circuit and the physical layout are separate proof objects.

The logical circuit expresses the protocol obligations without dependence on
Rust column numbers. The physical layout maps logical values to rows, columns,
and assignment positions. Lean must prove that this mapping preserves the
logical relation.

Every production row family and column must have:

- one semantic or layout owner;
- one lifecycle and phase scope;
- a clear public, private, transcript, or intermediate role;
- a proved connection to the final circuit relation.

Column sharing, projection, and reuse are valid only when Lean proves that they
preserve the logical relation. A column must not exist only because an earlier
Rust emitter created it.

## 8. cvc5 audit boundary

cvc5 supports proof-guided circuit reduction. It can search for false
assignments after a row removal, column removal, or column merge. It can also
identify candidate redundant constraints.

Lean controls every final decision:

- A cvc5 counterexample is evidence only after Lean checks the assignment
  against the reduced relation and the violated semantic obligation.
- A cvc5 `UNSAT` result is only a redundancy candidate. Removal requires a
  Lean proof or a Lean-checked certificate that the reduced builder still
  implies the complete target relation.
- An unknown or unchecked result does not permit removal.

Necessity is always relative to a stated relation and proposed reduction. The
project must not claim global column minimality without a proved lower bound.

## 9. Lean-emitted circuit package

Lean must emit one canonical production package that contains the complete
data required by the Rust proof engine for both step circuits. The package
includes:

- the fixed profile and schema;
- the public-input layout;
- the sparse CCS matrices and physical layout;
- the transcript schedule;
- the terminal layout;
- a canonical witness-generation program.

The package is builder output. It is not an independent proof object, and it
must not return to Lean as generated source for per-artifact certification.

Rust must load this package rather than rebuild the F′ circuit. Rust can use
generic optimized primitives to execute it. Any native witness acceleration
can produce only non-authoritative hints. The emitted rows must check every
result.

The verifier must bind a verifier-owned expected identity for each step
circuit. A different accepted package must reduce to a named Poseidon2
collision or binding failure.

## 10. Proof boundary

The architecture must export two separate results.

### Deterministic soundness

Any assignment that satisfies the Lean-built rows satisfies the complete
concrete F′ lifecycle relation.

This theorem has no cryptographic hardness assumptions. It is a structural
theorem about the builders and their composition. Its proof cost must not grow
with the size of one emitted artifact.

### Security composition

If the production verifier accepts, the F′ lifecycle is valid unless a named
cryptographic failure event occurred.

The named boundary includes Poseidon2 collision resistance, commitment
binding (relaxed binding at the chain norm), Fiat–Shamir and sampling security,
and the knowledge reductions of the light fold and the refresh fold (SuperNeo
PiCCS, PiRLC, and PiDEC with the departures of §3). Cryptographic assumptions
belong only in this result.

## 11. Rust boundary

Rust must use the exact circuit package that Lean emits. The production path
must not contain a second high-level relation, a test-only substitute, or an
independent circuit emitter.

Rust can own:

- strict package loading and decoding;
- generic witness-program execution;
- field, Poseidon2, commitment, polynomial, and transcript primitives;
- the generic prover and verifier;
- CPU and GPU acceleration.

The Lean compiler and emitter, Rust package decoder, primitive
implementations, and proof-system verifier are explicit trusted implementation
links. Tests are conformance evidence for these links; they are not universal
Lean proofs.

### Parity surface

Every Lean definition on the Rust parity surface must be computable, so that
Lean can execute it to produce test vectors and act as a differential oracle.
Where a proof needs a noncomputable form, the package must also provide a
computable form and a Lean theorem that the two agree. A `noncomputable`
definition with no computable counterpart is not on the parity surface and
cannot be claimed as Rust-conformant.

Values must match bit-for-bit at exactly these interfaces:

1. primitive outputs: field and extension-field arithmetic, ring operations,
   `split_b` decomposition, Poseidon2, commitment;
2. derived challenges for a given transcript, under the Lean-defined absorb
   schedule;
3. the folded running instance after each of PiCCS, PiRLC, and PiDEC, on the
   same proof, between the compiled Lean verifier and the Rust verifier;
4. the emitted circuit package, as loaded by Rust.

Prover internals are free. How Rust computes sum-check polynomials,
commitments, or witnesses, on CPU or GPU, is unconstrained as long as the
result passes the Lean-defined verifier semantics.

Lean execution is never on the proving path. It is used only for emission,
test vectors, and differential verification, and it needs only to be fast
enough for those.

## 12. Architecture guidelines

Keep one canonical route from the exported security claim to the semantic
relation, builders, physical layout, emitted package, and production verifier.

Avoid:

- asking Lean to certify large artifacts that Rust created;
- generated Lean modules that contain rows, matrices, or assignments;
- proof work whose cost grows with an emitted artifact;
- free semantic predicates or authority records at the production boundary;
- parallel production and test-only relations;
- duplicate assignment views or circuit schedules;
- digests used in place of semantic proofs;
- circuit structure inherited from Rust implementation history;
- generic protocol frameworks or profiles that production does not use;
- proof volume used as evidence of protocol completion.

The old Lean project can remain a reference corpus. It is not an authority for
the new circuit and must not pull its generated certification architecture into
the canonical proof path.

### Mechanical enforcement

The canonical F′ proof path must be an independent Lean package. Its build gate
must mechanically reject:

- imports from the old artifact-certification package;
- generated proof modules or embedded artifact data;
- kernel evaluation whose cost grows with emitted artifact size;
- implicit build globs;
- alternate production profiles.

These checks enforce the architecture. Prose rules alone are not sufficient.

## 13. Completion standard

### Per-phase completion

A phase is complete only when:

- its concrete lifecycle relation exists;
- its builder implies that relation;
- its rows are in the canonical emitted package;
- the real Rust prover produces a satisfying assignment for those rows;
- the real Rust verifier consumes that exact relation.

A test-only phase, unused proof, or alternate relation does not count.

### Stage completion

A stage is complete only when all parts form one production path:

- the complete concrete lifecycle relation exists;
- the logical builders imply that relation;
- the physical layout provably preserves the logical circuit;
- every production row family and column has a named owner;
- all accepted reductions have Lean-authoritative evidence;
- Lean emits the package used by the real Rust verifier;
- the recursive fixed point and joint-domain bound are Lean theorems;
- the exported theorems contain only the intended assumptions;
- the Rust production path creates and verifies a proof against that package.

A generated package, a cvc5 report, a local row theorem, or a test-only proof
run is not completion by itself.

## 14. Short form

Lean defines the protocol, builds and proves the logical circuit, proves its
physical column layout, and emits the only circuit package that Rust accepts.
Rust executes that package with a small generic proof engine and optimized
primitives. cvc5 attacks proposed reductions, but Lean decides which rows and
columns are sound to keep, merge, or remove.
