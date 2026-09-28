# Nightstream F′ — Stage 1 goal

Owner file. Do not modify this file without the owner's explicit approval.

## Authorization

This goal supersedes every earlier F′ goal, including any instruction to
continue inside `formal/nightstream-lean` or its dirty worktree.

You are authorized to:

- create a new, independent Lean package for the production F′ relation;
- treat `formal/nightstream-lean` as a frozen, read-only reference corpus;
- copy the smallest audited semantic definitions from it, with a provenance
  note (source path and commit) at the top of each copied file.

You are not authorized to:

- add, repair, build, or delete anything in `formal/nightstream-lean`;
- stage, commit, reset, stash, or discard the current dirty worktree;
- start Stage 2 (Nebula memory). Stage 2 needs a separate go-ahead.

## Contract

`FPRIME_LEAN_ARCHITECTURE_SPEC.md` is the architecture contract. Read it
completely before any work and again before any change to a relation,
authority boundary, layout, transcript, or exported theorem. Where this goal
and the spec differ, stop and report; do not choose.

Also read completely before work:

- `AGENTS.md`
- `formal/nightstream-lean/AGENTS.md` (its discipline rules apply to the new
  package; its file-layout rules do not)
- every paper section listed below
- `FPRIME_STAGE1_EXTERNAL_REVIEW.md` and
  `FPRIME_STAGE1_EXTERNAL_FABLE_REVIEW.md`, if they exist

## External review checkpoints

Other AIs write review feedback to these files at any time:

- `FPRIME_STAGE1_EXTERNAL_REVIEW.md`
- `FPRIME_STAGE1_EXTERNAL_FABLE_REVIEW.md`

During active work, read both files at 20 minutes past every hour (13:20,
14:20, and so on), before the next command or edit. If a file does not exist,
skip it and continue; its absence is not a blocker and must not stop the work.

At each checkpoint: state which findings apply to the active claim, address
those that are necessary under the MSW rule, and reject unrelated work in one
line each in the final report. A review is evidence, not authority: verify a
finding against the current code before acting on it, and never modify a
review file.

## Paper sections

Read these completely before implementation work. Re-read the relevant
section before changing a relation, authority boundary, transcript or digest
schedule, recursive-size claim, or exported theorem.

### HyperNova — NIVC, F′, compiler, size closure

- `docs/hypernova-paper/10_5_Non_uniform_incrementally_verifiable_computation.md`
- `docs/hypernova-paper/11_6_HyperNova_NIVC_from_multi_folding_schemes.md`
- `docs/hypernova-paper/12_6_1_Overview_of_HyperNova.md`
- `docs/hypernova-paper/13_6_2_NIVC_Compatible_multi_folding_schemes.md`
- `docs/hypernova-paper/14_6_3_A_compiler_from_NIVC_compatible_folding_schemes_to_NIVC.md`
- `docs/hypernova-paper/39_H_2_Proof_of_Lemma_3_Folding_CCS_NIVC_compatibility.md`
- `docs/hypernova-paper/40_H_3_Proof_of_Theorem_4_HyperNova.md`

### SuperNeo — PiCCS, PiRLC, PiDEC, composition, parameters

- `docs/superneo-paper-v1_1/04_preliminaries.md`
- `docs/superneo-paper-v1_1/05_embeddings_and_evaluation_homomorphism.md`
- `docs/superneo-paper-v1_1/06_strong_and_weak_interactive_reductions.md`
- `docs/superneo-paper-v1_1/07_superneo_folding_scheme_for_ccs.md`
- `docs/superneo-paper-v1_1/08_concrete_parameters.md`
- `docs/superneo-paper-v1_1/11_appendix_B_deferred_theorems_and_proofs.md`:
  - B.1 Proof of Composition Theorem
  - B.2 Proofs for PiCCS
  - B.3 Proofs for PiRLC
  - B.4 PiDEC is a Reduction of Knowledge

The Appendix B.2 parameters `k_rho = 14`, `B = 2^14` are reference values
only. The Nightstream profile is `k_rho = 16`; do not describe it as paper
exact.

SuperNeo v1.1 is the normative folding protocol. The v1.0 and v1.0-errata
PiCCS layouts are reference material only and must not remain on the
production Lean or Rust path.

Nebula (`docs/nebula-paper/`) and Coral (`docs/Coral.pdf.md`) are Stage 2
reading. Do not read them for Stage 1 work.

## Outcome

Stage 1 of the spec: a complete HyperNova F′ over SuperNeo folding, for the
fixed Nightstream Goldilocks profile (spec §3), delivered as:

1. a concrete Lean lifecycle relation (base, recursive, terminal) with no free
   predicates or authority records;
2. Lean logical builders for every phase, each proved to imply its relation
   slot, and one composition theorem;
3. a proved physical layout and lowering;
4. the deterministic soundness theorem and the security composition theorem
   (spec §10), in the package's axiom gate;
5. a canonical Lean-emitted circuit package (spec §9) that the production
   Rust lifecycle loads and uses as its only relation;
6. Lean theorems for the recursive fixed point and a joint domain no larger
   than `2^28`;
7. one executed production `prove → verify` run through a separately
   owner-approved backend on that package.

All seven, on one production path, with no test-only relation beside it.

Outcome item 7 is a separate cryptographic-production obligation. It cannot
establish semantic correctness, Lean–Rust conformance, matrix equality, or
assignment satisfaction. Spartan is not an approved backend for this goal.

## Package

Location: `formal/nightstream-fprime` (rename only with owner approval). The
package already exists and is work in progress: continue inside it, do not
create a second package.

Its build gate mechanically enforces the boundaries in the spec (§12,
mechanical enforcement): no imports from `formal/nightstream-lean`, no
generated proof modules or embedded artifact data, no `native_decide`, no
artifact-sized kernel evaluation, explicit build targets, one profile. Read
`formal/nightstream-fprime/AGENTS.md` for its layers, rules, and the
`scripts/validate.sh` phases (`static`, `build`, `axioms`, `file <path>`,
`all`).

## Current state

Items 1–3 of the order of work below are closed. The pilot and PiCCS Lean
compiler paths exist, but their implementation-conformance gates are open.
Do not describe either path as integrated or production-closed.

The current source, emitted artifact, verifier identity, and Rust tests must
come from one unchanged cut before any green package claim is made. At the
latest reviewed cut, the Lean source uses schema 6 while the stored artifact
and recorded identity use schema 5; the Rust package gates are therefore red.

- `Spec/`: Goldilocks/Φ₈₁ algebra, profile, Poseidon2 reference, sumcheck,
  exact v1.1 PiCCS with separate Pad and 14-matrix evaluation families,
  PiRLC/PiDEC verifiers, composed NIFS `Key`/`verify`, Φ₈₁ PiRLC/PiDEC
  algebras, HyperNova Construction 2, and Goldilocks primality.
- `Lifecycle/`: `Types` (slotCount 1, cubeVariables 28,
  productionShape, phase order), `PaperAlgebra` (Ajtai-commitment semantics
  and algebras),
  `Transcript` (Poseidon2 Fiat–Shamir and strong-set ρ sampler, membership
  proved), `XOut` (length-prefixed preimage, `stateHash`, `encHash`,
  `defaultRunning`), `ProductionKey` (`LogicalRelation`, `key`: the one
  concrete NIFS key, all law fields discharged), `Relation` (`setup`,
  `machine`, `StepHolds := FixedAugmentedTransition`, `TerminalHolds`).
- The pilot has one proved Lean path through `Circuit/`, Poseidon2 gadgets,
  lifecycle builders, physical lowering, and the Lean emitter. Its main
  evidence is `Pilot.phase_soundness`,
  `Pilot.builders_imply_hash_slots`,
  `Export.Stage1.Package.circuitPackage_implies_pilotSpec`, and
  `circuitPackage_implies_recursive_hash_slots`. Exact Rust matrix
  conformance and independent raw-assignment evaluation remain open. The
  pilot-only layout is
  12,574,138 rows, 12,659,030 private columns, 58 public columns, 12,659,089
  total columns, and joint domain 12,659,088.
- PiCCS v1.1 has twelve audit leaves, one phase assembler, logical
  soundness and completeness, physical preservation, package soundness,
  and package completeness. `PackageCompleteness.complete_piCcsRows`
  constructs all canonical PiCCS package rows from `PhaseHolds`.
- The current pilot + PiCCS source package carries the Lean-owned selective
  relation with 14 matrix tags, 74 terms, degree bound 9, and 28 rounds. Its
  proved layout has 27,893,668 rows, 28,007,520 private columns, 58 public
  columns, 28,007,579 total columns, and joint domain 28,007,578. Re-check
  and update these values from the proved layout on every identity-changing
  source cut.
- Rust `paper_exact` implements the direct SuperNeo v1.1 formulas. Rust
  `optimized` keeps the same separate `Eval_K` and `Eval_A` values and is
  byte-equivalent on the tested Rust-to-Rust surface. This is useful
  differential evidence. It is not Lean-to-Rust parity.
- The package runtime is currently a test bridge. The production lifecycle
  does not consume it. The current Lean parity vector uses zero PiCCS proof
  messages and zero `Eval_K`/`Eval_A`; it does not close nonzero PiCCS
  conformance.
- The production relation uses a 25-variable row cube and therefore 25 PiCCS
  rounds. The same `2^28` profile bounds the Stage 1 physical joint domain.

Known debts, to close and not to hide:

1. `ProductionKey.key` uses `piDecDecision := Classical.propDecidable`, so the
   verifier is noncomputable. The parity surface needs a computable
   `Decidable (PiDEC.PaperVerifier.Accepted …)`.
2. The scoped fixed-size proof overrides for ring degree 54 remain candidates
   for structural proofs. Do not hide them with artifact-sized evaluation.
3. The PiDEC verifier must add the v1.1 public-input `B`-norm rejection before
   the canonical 16-child split. An out-of-range split must fail. No accepted
   verifier path may silently use `fallbackDigit`.
4. The production Rust lifecycle does not yet consume the complete Lean
   package. Its native PiCCS circuit still uses `y_ring`/`ct` wires and a
   24-variable application relation. The v1.1 engines are migrated, but the
   application must load the verifier-owned package, build its assignment
   from a real fold, and retire the superseded native PiCCS circuit authority
   in the same integration slice.
5. Rust's final expanded `A/B/C` matrices have not been compared
   entry-for-entry with Lean's canonical physical rows.
6. No independent evaluator checks a raw Rust assignment directly against
   the canonical Lean rows without using witness generation or Rust's row
   expander.
7. No valid nonzero PiCCS proof has complete three-way parity between
   executable Lean, Rust `paper_exact`, and Rust `optimized`.
8. Rust `sample_rot_rhos_n` still uses 3-bit candidates and pads sampler
   shortfall. Before PiRLC conformance, replace it with the exact Lean 16-bit,
   bounded, fail-closed schedule.

Lessons recorded: keep executable specs materialized; do not use
tuple-pattern `let (a, b) := e` in definitions that later need `rfl`; check
every new file with `validate.sh file`, which passes
`-DautoImplicit=false`.

## Reuse from the frozen package

Three different things are needed for each of PiCCS, PiRLC, and PiDEC, and
they are in different states in `formal/nightstream-lean`:

1. **Verifier semantics** (what the verifier computes and checks). Exists,
   model-level, paper-shaped, no `sorry`, no axioms. Candidates for
   copy-and-audit: `Nightstream/SuperNeo/Folding/{PiCCS,PiRLC,PiDEC,Nifs}`,
   `SuperNeo/SumCheck`, `SuperNeo/InteractiveReduction`, `SuperNeo/Concrete`
   (ring, profile, decomposition algebra), `HyperNova/Construction2`,
   `Protocol/FPrime`. Copy only the definitions the new relation references.
   `Folding/PiCCS` is 29k lines for one verifier; audit it down, do not copy
   it whole. Remove radix-four and every reference to the streaming or
   artifact layers. Instantiate oracles the old relation left abstract
   (PiRLC response, opening maps). Fix the profile to `k_rho = 16`.
2. **Circuit builders** (gadgets that enforce those checks in CCS rows).
   Do not exist in any reusable form. All new work: sumcheck round, PiCCS
   final evaluation check, PiRLC ring combination over 17 inputs with
   challenges from the sampling set, PiDEC `split_b` recomputation and
   recombination, Poseidon2 transcript, norm check, selectors and state
   binding.
3. **Security reductions** (B.1–B.4 as reductions of knowledge). Partially
   present in `SuperNeo/InteractiveReduction` and `Folding`. Needed only for
   the security composition theorem. Copy what exists; state the rest as the
   explicit SuperNeo soundness assumption.

`Protocol/Nebula` is not copied in Stage 1.

## Package structure

The closest existing design is Verified-zkEVM `clean`. Use its shape; avoid
its accumulated variants. The package has these layers, each with one
responsibility and imports flowing downward only:

```text
Spec/        paper semantics: field, ring, Poseidon2, sumcheck, PiCCS/PiRLC/
             PiDEC verifiers, HyperNova F′ relation (copied + audited)
Circuit/     the DSL: expressions, variables by offset, operations
             (witness, assert, subcircuit), the circuit monad, one
             FormalCircuit record (main, assumptions, spec, soundness,
             completeness), the simp set that normalizes it
Gadgets/     one directory per gadget; each exports only its FormalCircuit;
             parents see a child's spec, never its operations
Lifecycle/   the concrete F′ phase order, carried state, public layout,
             XOut recomputation; composition of gadget specs into the
             relation slots
Layout/      physical lowering: logical values → rows/columns; the
             layout-preservation theorems
Export/      serializer, witness-program IR, relation identifier; the
             decode∘encode theorem; the `lake exe` emitter
Gate/        axiom audit with explicit imports; enforcement script
```

Rules taken from what worked and what hurt in `clean`, `zkLean`, `ArkLib`,
and `vr1cs_lean`:

- One `FormalCircuit` record. `clean` grew four flavors plus two elaboration
  classes; do not. Add a variant only when a gadget cannot be expressed.
- Subcircuits are opaque to parents. A parent proof that unfolds a child's
  operations is a bug.
- Symbolic variables by offset; no column numbers anywhere above `Layout/`.
- Proof-cost control is a curated simp set and per-gadget lemmas, not
  `decide`. `clean`'s `performance-problems.md` documents what happens
  otherwise: `whnf` into concrete values, kernel size cliffs in large
  completeness compositions. Read it before writing the DSL.
- The witness program is an exported IR interpreted by Rust (as in `clean`'s
  `WitnessIR`), not native closures; native hints are non-authoritative.
- Export is `#eval`/`lake exe` only; no proof evaluates an emitted circuit.
- Do not depend on ArkLib. Its sumcheck and Fiat–Shamir carry `sorry`, its
  verifier is noncomputable, and its dependency stack is heavy. Copy our own
  sumcheck semantics instead.
- Do not build a `zkLean`-style interpreter-as-semantics; the interpreter
  would be in the trusted base and proofs would unfold it.
- Do not keep parallel pipelines, archived plans, or marketing docs in the
  package (`vr1cs_lean`). One pipeline.
- `lakefile.toml`: explicit `roots`, no glob; `autoImplicit=false`;
  `linter.style.longFile = 1500`; a lint driver.

## Verifier decomposition and auditability

Implement the recursive verifier as an explicit tree of small audit units.
Do not put PiCCS, PiRLC, or PiDEC into one monolithic constraint builder. A
human reviewer must be able to select one mathematical obligation and inspect
its meaning, circuit interface, constraints, proofs, and physical owner
without reading the complete verifier.

A leaf is one independently understandable paper obligation, not one
arithmetic operation. Keep small helper expressions with their owner. Use one
generic indexed leaf when the same formula repeats: one sumcheck-round gadget
composed for 28 rounds, one PiRLC combination gadget composed for 17 inputs,
and one PiDEC child gadget composed for 16 children. Do not copy a leaf for
each index.

Before the first repeated use, prove one offset-safe indexed-composition
interface. Its theorems must transport each child specification under its
offset, prove the child variable ranges, and prove that composition adds no
rows except the child rows and necessary parent semantic constraints. A
boundary-only equality or `assertZero` row whose only purpose is a file or
subcircuit boundary is forbidden.

Each leaf must provide one compact audit packet:

- the exact SuperNeo v1.1 section, definition, or equation that owns it;
- a named Lean semantic predicate and the exact parent-relation conjunct that
  it discharges;
- explicit symbolic inputs and outputs, including their representations;
- one `FormalCircuit` whose constraints enforce that predicate;
- soundness and completeness through that `FormalCircuit`;
- a symbolic variable range or footprint theorem for `Layout/`;
- a short constraint-group summary that lets a reviewer map each asserted
  equation to the paper formula;
- for every transcript challenge, a derivation theorem that satisfying rows
  force the challenge variable to equal the Poseidon2 squeeze of the bound
  transcript state, whose state is itself bound to the absorbed statement and
  messages; and
- the focused `validate.sh file` evidence.

The PiCCS ownership tree must remain visible in the source. Existing shared
Duplex and SumCheck gadgets stay shared; do not duplicate them only to match
this outline.

```text
PiCCS
├── statement and prior-claim binding
├── transcript schedule
│   ├── statement absorption
│   ├── round-message absorption
│   └── verifier-derived challenges
├── sumcheck
│   ├── one generic round equation
│   └── the fixed 25-round chain
├── final evaluation
│   ├── Eval_K: the separate Pad family
│   ├── Eval_A: the separate 14-matrix family
│   └── the v1.1 final joint identity
├── output reduced claims
└── the complete PiCCS FormalCircuit
```

The v1.0 compression is forbidden at every level of this tree. Do not place
Pad at matrix zero, combine `Eval_K` and `Eval_A` into one `Eval`, or use an
unproved equivalence to the compressed relation. Preserve the v1.1 split in
the semantic predicate, transcript, builder, layout, package, and Rust parity
interface.

Replace the v1.0 authority in place. Do not create a sibling
`Spec/Folding/PiCCS/v1_1/` relation or keep an old verifier behind an adapter.
In the same slice that installs each v1.1 authority path, remove the
superseded v1.0-only modules, fields, and imports that no longer have a v1.1
paper obligation. This includes the Pad-as-matrix-zero `PrefixLayout` and
`identityFirstEntry` authority surfaces. The NIFS `Key`,
`ProductionKey.key`, `nifsVerifier`, and `StepHolds` must all consume the same
exact v1.1 `Accepted` predicate that the gadget proves. A v1.1 gadget beside a
v1.0 lifecycle verifier is a forbidden second relation.

Apply the same ownership rule to PiRLC and PiDEC. PiRLC must visibly separate
input binding, transcript and strong-set sampling, the 17-input combination,
commitment and evaluation combinations, and output binding. PiDEC must
visibly separate the public-input `B`-norm rejection, `split_b`, digit or
low-norm checks, recombination, the 16 output children, and output binding.
Derive the final leaves from the v1.1 formulas, not from the old Rust code.

The exact v1.1 `Accepted` predicates on the Rust parity surface must be
computably decidable. Before PiDEC or composed-NIFS parity can close, replace
`Classical.propDecidable` with a computable decision procedure and prove that
it agrees with the semantic predicate. The circuit and Rust verifier must not
target different decidability surfaces.

Children are opaque to parents. A parent may use a child's `FormalCircuit`,
specification, input/output interface, proofs, and symbolic footprint. It must
not unfold the child's operations. The parent owns and proves only the wiring:
shared values, phase order, transcript-state order, verifier-derived
challenges, public binding, and complete conjunct coverage. One composition
theorem must prove that the children and their wiring are sound and complete
for the exact phase relation. Correct leaves without this wiring theorem do
not close the phase.

File boundaries are audit boundaries, not new circuit boundaries. They must
not add rows, copy transcript states, materialize duplicate values, or create
alternate relations. `Layout/` must lower the composed symbolic interfaces,
derive child boundaries from their footprint theorems, and assign each row
family and physical column to one leaf or parent-wiring owner. `Export/` must
preserve that ownership in the canonical package; it must not restate child
layout constants.

`Layout/` must also own one cumulative footprint ledger for the current
canonical composition. Each completed leaf or phase extends an exact theorem
that the composed joint domain is no larger than `2^28`, and its report gives
the new row count, column count, joint domain, and delta from the preceding
composition. Use the proved composed layout, including any proved reuse; do
not substitute an unproved sum or invent a per-leaf quota. Include transcript
rows when the transcript leaf closes so a late transcript expansion cannot
hide until final composition.

At the start of each v1.1 phase, report the Rust migration surface that the
phase will replace. For PiCCS this includes the joint proof-message and
polynomial layout, sumcheck witness construction, folded-instance encoding,
transcript schedule, and fixtures that still follow v1.0. Exact Rust file
names can change; the owned behaviors cannot be deferred until the final
parity test. Replace a Rust-owned behavior only when the corresponding Lean
rows reach the canonical package, as required by the Rust section below.

The required audit trace for every accepted formula is:

```text
paper formula
→ Lean predicate
→ FormalCircuit constraints
→ soundness and completeness
→ parent composition and wiring
→ proved physical rows and columns
→ canonical package rows
→ exact equality with Rust's final expanded A/B/C matrices
→ independent raw-assignment row evaluation
→ the only production relation
```

A formula-level test, a correct leaf, or a clean local layout is partial
evidence only. The phase closes only when every leaf reaches the one canonical
production path under the per-phase completion standard.

## Implementation assurance gates

These gates add incremental assurance before full production closure. They do
not replace the architecture specification's final completion requirements.

Use these status terms:

- **Compiler-closed:** the exact Lean predicate, circuit, soundness,
  completeness, composition, layout, and package theorems exist.
- **Conformance-closed:** the compiler-closed phase also passes all Lean–Rust
  matrix, value, assignment, and mutation checks below.
- **Production-closed:** the production lifecycle uses the validated package
  relation exclusively, and a separately owner-approved verifier binds that
  package, its validated matrices, verification key, and public input.

Do not use `complete`, `integrated`, `production`, `sound`, or `parity`
without naming the applicable status and evidence.

### Per-phase conformance

Close PiCCS, then PiRLC, then PiDEC. Do not start the next phase before the
active phase is conformance-closed.

A phase is conformance-closed only when:

1. An independent review artifact maps every exact SuperNeo `v1_1` verifier
   conjunct to its Lean predicate, circuit, and theorem. It confirms that no
   `v1_0` Pad-as-matrix-zero compression remains. The report names the
   reviewer and reviewed source cut.
2. Lean proves circuit soundness, completeness, parent wiring, conjunct
   coverage, and physical layout preservation.
3. The final Rust-expanded `A/B/C` object, after every Rust transformation,
   matches Lean's canonical expansion entry-for-entry. Equality covers row
   order, variable indices, coefficients, the constant column, public-input
   mapping, and canonical field encoding.
4. Executable Lean, Rust `paper_exact`, and Rust `optimized` consume the same
   serialized valid nonzero input and proof and produce byte-identical
   complete phase results.
5. A separate evaluator takes the raw assignment after witness generation and
   checks every canonical Lean row. It must not call or reuse Rust's witness
   generator, row expander, or constraint evaluator.
6. Mutations to every authoritative proof, transcript, output, row, column,
   and public-input family cause failure.
7. Deterministic generated nonzero fixtures cover the named semantic branches.
   Required rejection cases include PiRLC sampler shortfall through an
   injected candidate stream, PiDEC parent-norm rejection, and malformed
   phase outputs.

Before PiDEC compiler work, add the strict parent `B = 2^16` norm rejection,
make out-of-range `split_b` fail instead of silently using `fallbackDigit`,
replace `Classical.propDecidable` with a computable decision, and prove that
the decision agrees with the semantic predicate.

A nonzero lifecycle state combined with zero proof messages, zero `Eval_K`,
or zero `Eval_A` does not qualify as a nonzero phase fixture.

Exact matrix comparison may stream outside the Lean kernel. A digest,
compressed-package round trip, row count, shape check, or nonzero-count check
is not exact matrix equality.

### Complete phase results

PiCCS comparison includes acceptance, `alpha`, `gamma`, every round challenge
and intermediate transcript state, `r'`, terminal identities, all 17 output
commitments and public inputs, all 17 separate `Eval_K` families, all
`17 × 14` separate `Eval_A` families, and the outgoing state.

PiRLC comparison includes acceptance, all `rho` values and membership results,
the indexed 17-input commitment, public-input, `Eval_K`, and `Eval_A`
combinations, the output claim, and the outgoing state.

PiDEC comparison includes acceptance, parent-norm rejection, all 16 digits
and range results, recombination, all commitment/public-input/`Eval_K`/
`Eval_A` relations, all 16 child claims, and the outgoing state.

### Incremental composition

Each new phase consumes the exact conformance-closed output of the previous
phase and reruns the cumulative prefix:

```text
PiCCS
PiCCS → PiRLC
PiCCS → PiRLC → PiDEC
```

The final Stage 1 integration must additionally prove all cross-phase wiring,
compare the complete Rust matrices with the complete Lean package, execute
one valid nonzero full fold through the independent row evaluator, remove or
make unreachable every native alternate relation, and prove the recursive
fixed point and complete `2^28` bound.

### Relation-identity changes

Every relation-identity change must mechanically run the exact expanded-matrix
comparison, independent raw-assignment evaluator, and applicable nonzero
parity gates before the verifier-owned identity can be re-pinned. The expected
identity must not be copied automatically from the artifact.

### Proof-backend boundary

Spartan is not authorized for this work. Do not run, modify, integrate, or
cite it as evidence.

A proof backend proves only the relation supplied to it. Backend acceptance
cannot establish semantic correctness, matrix equality, value parity,
assignment satisfaction, or production-path identity. Selection and audit of
a production backend require separate owner approval after the conformance
gates close.

## Order of work

Work on the first item without its required evidence. Do not start the next
item to avoid finishing the current one.

1. Package skeleton, enforcement script, axiom gate, bounded validate script
   with the 1,500 s Lean cap.
2. Fixed profile and minimal semantic types, copied or written.
3. Concrete lifecycle relation skeleton: phase order, carried state, public
   layout, `XOut` recomputation, with every phase slot named and concrete.
4. Reopen the pilot at the conformance boundary. Implement exact final-matrix
   comparison, the independent raw-assignment evaluator, nonzero pilot parity,
   and mutation coverage. Report pilot conformance closure.
5. Stop and request the owner's decision on the PiCCS statement-absorption
   schedule in `OPEN_ISSUES_LEAN_REFACTOR.md`. Do not pin a nonzero PiCCS
   fixture or relation identity before this decision. This goal does not
   authorize the implementation agent to choose the schedule.
6. Close complete nonzero PiCCS three-way parity, exact PiCCS matrix equality,
   independent PiCCS row evaluation, and PiCCS mutation coverage. Report
   PiCCS conformance closure before starting PiRLC.
7. Keep the existing PiRLC `InputBinding` leaf frozen in place. Do not delete,
   extend, or report it as active phase progress before PiCCS conformance
   closes.
8. Before other PiRLC work, migrate Rust to the exact Lean 16-bit, bounded,
   fail-closed `rho` sampler. Then build and conformance-close PiRLC.
9. Build and conformance-close PiDEC, then the accumulator, running-instance,
   application, and terminal phases, one at a time.
10. Build the Stage 1 assembler. Prove complete cross-phase wiring,
    deterministic soundness, the recursive fixed point, the complete `2^28`
    bound, and the security-composition theorem.
11. Put the validated package on the only production lifecycle path and
    remove every superseded native relation.
12. After separate owner approval of a production backend, execute the final
    production `prove → verify` obligation.

## Rust

Rust work in Stage 1 is limited to what the emitted package needs: a strict
loader, the witness-program interpreter, primitive conformance against the
Lean definitions, and wiring the real prover and verifier to the loaded
package. Do not rewrite, clean up, or delete the existing Rust F′ emitters
until a phase's rows come from the Lean package; then remove only that phase's
superseded emitter surface.

The production Rust folding path must implement SuperNeo v1.1. Update any
v1.0 PiCCS layout, folded-instance representation, transcript schedule, or
test fixture when the corresponding Lean v1.1 phase reaches the package.

Both Rust folding engines must migrate to this authority. The `paper_exact`
engine must implement the SuperNeo v1.1 formulas directly, with separate
`Eval_K` and `Eval_A` families and the paper's stated transcript and reduction
order. It must not retain the v1.0 Pad-as-matrix-zero compression behind an
adapter. The `optimized` engine may use a different internal algorithm, but
for the same authoritative inputs and proof it must produce byte-for-byte the
same parity-surface values as `paper_exact`. Focused differential tests must
cover each migrated phase before that phase is reported complete.

Rust conformance is measured at the parity surface in the spec (§11): Lean
definitions on that surface must be computable, and executed tests must show
bit-for-bit equality at the four interfaces — primitive outputs, derived
challenges, the folded instance after each of PiCCS/PiRLC/PiDEC, and the
loaded package. Each interface needs its test before the phase that uses it
is reported complete. Prover internals are free.

Rust conformance also requires two independent checks:

- compare the actual final expanded Rust matrices with Lean's canonical
  expansion entry-for-entry; and
- check the raw assignment against the canonical Lean rows with an evaluator
  that does not reuse witness generation or Rust's matrix expansion.

Rust-to-Rust differential tests remain required, but they cannot substitute
for a valid nonzero Lean-to-Rust phase comparison.

## Operating rules

- State one active acceptance criterion and its closing evidence before each
  command or edit. One claim at a time. After three coherent rounds without
  closure, stop and report the exact open item.
- Lean commands only through the package's bounded validate script;
  `timeout: 1500000`. Rust tests `--release`, `RUSTC_WRAPPER=""`,
  `timeout: 300000`. One Lean or Rust build process at a time.
- Never evaluate emitted data in the kernel. Proof cost must not grow with
  rows, columns, or schedule length.
- cvc5 (`/Users/nijaar/.local/bin/cvc5`) only within the spec's §8 boundary,
  only on single gadgets, only after the pilot closes.
- `cargo fmt --all` after Rust edits. Files below 1,500 lines. DCO sign-off on
  any commit the owner asks for. Simplified Technical English in reports.
- Do not modify `AGENTS.md`, either existing `AGENTS.md` file, the spec, or
  this file.

## Out of scope

- Nebula, Coral, memory, RAM/ROM (Stage 2).
- Reducing, refactoring, or building `formal/nightstream-lean`.
- Public lifecycle API design, benchmarks, GPU work beyond what the loaded
  package needs.
- Mathematical minimality, alternate profiles, radix-four, `k_rho = 14`.
- Generic protocol frameworks.

## Report

For each completed item: theorem names, package identity (Poseidon2 relation
identifier), focused Lean target, focused Rust test, and exact row, column,
and domain values where the item requires them.

For each compiler-closed, conformance-closed, or production-closed claim,
report the exact evidence for that status separately. Do not use a stronger
status because a weaker status is green.

For each open item: the exact obligation, the evidence obtained, and the
evidence still required. A partial milestone is not evidence that the stage
is done.

## Recursive verifier decomposition and auditability

This section supplements the existing goal, architecture specification,
`AGENTS.md` files, and validation rules. It does not replace them.

### Objective

Implement the recursive verifier as a clear tree of small, auditable circuit
gadgets.

Do not implement PiCCS, PiRLC, or PiDEC as one large block of constraints. A
human reviewer must be able to select one mathematical formula and inspect
only:

1. its semantic meaning;
2. its circuit inputs and outputs;
3. the constraints that enforce it;
4. its soundness and completeness proofs;
5. its layout and resource footprint.

The file structure is an audit structure. It must not create alternate
production relations or alternate verifier paths. All leaves compose into the
one production F′ circuit.

### SuperNeo version authority

Implement exact SuperNeo v1.1 semantics before implementing more PiCCS circuit
gadgets.

The old compressed v1.0 relation is not acceptable. In particular:

- Keep `Eval_K`, for the Pad evaluation family, separate.
- Keep `Eval_A`, for the CCS matrix evaluation family, separate.
- Do not treat Pad as matrix zero.
- Do not replace both families with one carried `Eval` value.
- Do not assume that the old compressed relation is equivalent to v1.1 without
  a proved equivalence theorem.
- Carry the v1.1 distinction through the semantic relation, transcript,
  circuit inputs, constraints, output claims, exported package, and Rust
  parity interface.

Lean is the semantic authority. Rust must later conform to the exact v1.1 Lean
relation.

For notation in new Lean namespaces and file names, prefer `v1_1` over `V11`
or other forms. Do not move or rename closed work only to apply this notation.

### Decomposition rule

A leaf gadget corresponds to one independently understandable mathematical
obligation.

Split an obligation when a reviewer can usefully verify it independently. Do
not split small helper expressions that have no independent mathematical
meaning.

Use one generic gadget when the same formula repeats:

- Use one generic sumcheck-round gadget and compose it for all rounds.
- Do not create 24 copied sumcheck-round files.
- Use indexed composition for the 17 PiRLC inputs.
- Use indexed composition for the 16 PiDEC decomposition components.
- Do not copy the same constraints into separate files.

The source tree shows the mathematical structure, but remains compact.

### Required audit packet for each leaf

Each leaf gadget contains or directly references:

- the exact SuperNeo v1.1 section, definition, or equation that it implements;
- a named Lean semantic predicate;
- explicit symbolic input and output types;
- the circuit constraints;
- a soundness theorem: satisfying the constraints implies the semantic
  predicate;
- a completeness theorem: valid semantic inputs permit a satisfying witness;
- a variable-bound or footprint theorem needed by layout;
- the exact mapping from this leaf to one part of the parent verifier
  predicate.

Use a short module comment similar to this:

```lean
/-!
Paper authority: SuperNeo v1.1, section/equation ...
Obligation: A short statement of the mathematical check.

Inputs:
- ...

Outputs:
- ...

Constraint groups:
- C1: ...
- C2: ...

Parent coverage:
- PiCCS.Accepted.<named obligation>
-/
```

Comments help the human audit, but comments are not proof. The composition
theorem mechanically shows that all required obligations are covered.

### Suggested logical PiCCS tree

Adapt names to the existing package. Do not move or rewrite closed work only
to match this example.

```text
PiCCS
├── Statement and prior-claim binding
├── Transcript schedule
│   ├── Statement absorption
│   ├── Round-message absorption
│   └── Challenge derivation
├── Sumcheck
│   ├── Generic round equation
│   └── Fixed round chain
├── Final evaluation
│   ├── Eval_K / Pad evaluation
│   ├── Eval_A / CCS matrix evaluations
│   └── v1.1 final joint identity
├── Output reduced claims
└── Complete PiCCS FormalCircuit
```

A possible file organization is:

```text
Spec/Folding/PiCCS/v1_1/
  Statement.lean
  Transcript.lean
  SumcheckRound.lean
  EvalK.lean
  EvalA.lean
  FinalIdentity.lean
  Accepted.lean

Gadgets/PiCCS/
  StatementBinding.lean
  TranscriptSchedule.lean
  SumcheckChain.lean
  EvalK.lean
  EvalA.lean
  FinalIdentity.lean
  OutputClaims.lean
  Formal.lean

Lifecycle/PiCCS/
  Builder.lean
  Soundness.lean

Layout/PiCCS/
  Lowering.lean
  Preservation.lean
```

Reuse the existing generic Duplex and SumCheck gadgets. Do not duplicate their
operations inside PiCCS.

### PiRLC and PiDEC

Apply the same method after PiCCS closes.

The PiRLC tree visibly separates:

- input-claim binding;
- transcript absorption;
- strong-set challenge sampling;
- proof that each sampled `ρ` belongs to the allowed set;
- the indexed combination of 17 inputs;
- commitment combination;
- evaluation combination;
- output-claim binding.

The PiDEC tree visibly separates:

- input binding;
- `split_b` digit construction;
- digit-range or low-norm checks;
- recombination;
- commitment and evaluation relations;
- the indexed construction of 16 output components;
- output-claim binding.

Derive the final leaf boundaries from the exact v1.1 paper formulas. Do not
derive them from the old Rust implementation.

### Parent and child boundaries

Each child exports its `FormalCircuit` and semantic contract.

A parent may use:

- the child specification;
- the child soundness and completeness theorems;
- the child input/output interface;
- the declared symbolic variable footprint.

A parent proof must not unfold the child's internal circuit operations. If the
parent must inspect those operations to prove correctness, the boundary is
wrong.

The parent owns the wiring between children. It proves:

- the same values flow between connected interfaces;
- transcript states occur in the exact required order;
- no challenge is supplied by the witness;
- public values bind to the correct internal values;
- `Eval_K` and `Eval_A` remain separate;
- no required verifier check is omitted or checked twice.

### Required assembly hierarchy

The decomposition has two logical assembly levels:

```text
leaf FormalCircuits
→ one FormalCircuit assembler for each protocol phase
→ one Stage 1 protocol assembler
→ proved physical layout
→ one emitted Stage 1 package
```

Protocol-specific leaves live under `Lifecycle/<phase>/v1_1/`. Reusable
arithmetic and transcript circuits remain under `Gadgets/`. Each phase has
exactly one assembler:

```text
Lifecycle/PiCCS/v1_1/Formal.lean
Lifecycle/PiRLC/v1_1/Formal.lean
Lifecycle/PiDEC/v1_1/Formal.lean
```

Each phase assembler exports one `FormalCircuit`, its soundness and
completeness theorems for the exact phase predicate, its exact symbolic
footprint, and a mechanical coverage theorem. It owns only child wiring. It
does not unfold a child's circuit operations.

The complete protocol assembly is:

```text
Lifecycle/Stage1/
  Interface.lean
  Formal.lean
  Soundness.lean
```

`Lifecycle/Stage1/Formal.lean` composes prior-state binding, the three phase
circuits, the application transition, output binding, and terminal checks. It
exports the only production logical circuit:

```lean
NightstreamFPrime.Lifecycle.Stage1.circuit : FormalCircuit
```

The Stage 1 parent proves exact cross-phase value flow, transcript-state
order, verifier-owned challenges, public binding, separate `Eval_K` and
`Eval_A`, and coverage without omission or duplication.

The physical hierarchy mirrors the logical hierarchy:

```text
Layout/PiCCS/v1_1/{Lowering,Preservation}.lean
Layout/PiRLC/v1_1/{Lowering,Preservation}.lean
Layout/PiDEC/v1_1/{Lowering,Preservation}.lean
Layout/Stage1/{Lowering,Ownership,Preservation}.lean
```

`Layout/Stage1/Lowering.lean` assigns each child one physical interval and
reuses shared symbolic values where possible. A file boundary does not add a
copy row.

One final assembler emits the proved Stage 1 circuit:

```text
Export/Stage1/
  WitnessProgram.lean
  Package.lean
  Emit.lean
```

Rust loads this one package. There is no separate pilot, phase-only, or
Rust-native production relation after the corresponding Stage 1 surface
closes.

Maintain `formal/nightstream-fprime/CONSTRAINT_TREE.md` as the concise audit
index. It shows the multi-level file tree, marks present and required files,
and maps each leaf and assembler to its mathematical constraint obligation.
Update it when a leaf or assembly level closes.

### Layout and efficiency

Use symbolic variables by offset above `Layout/`. Do not use physical column
numbers in semantic or gadget modules.

File boundaries must not add circuit rows automatically. Do not materialize
unnecessary copies at subcircuit boundaries. Pass symbolic values and
transcript states directly when possible.

Each leaf exposes enough footprint information for the physical layout proof.
The final circuit still lowers into one production circuit package.

### Per-leaf work order

Work on one leaf at a time:

1. Re-read the relevant architecture and paper section.
2. Identify one named conjunct of the exact verifier relation.
3. Define or confirm its semantic predicate.
4. Implement its logical builder.
5. Prove soundness.
6. Prove completeness.
7. Prove its symbolic variable footprint.
8. Run the focused package validation.
9. Report the formula, theorem names, and exact footprint.
10. Only then start the next leaf.

Do not report PiCCS complete until a composition theorem proves that the
complete PiCCS `FormalCircuit` is sound and complete for the exact v1.1
`Accepted` predicate.

### Phase completion evidence

For a complete protocol phase, provide:

- a coverage map from every conjunct of the semantic `Accepted` predicate to
  a leaf gadget or an explicit parent-wiring theorem;
- leaf soundness and completeness theorem names;
- the parent composition theorem;
- the physical layout-preservation theorem;
- exact row, column, and domain values;
- the focused Lean validation target;
- exact comparison of Lean's canonical expansion with Rust's final expanded
  `A/B/C` matrices;
- the independent raw-assignment row-evaluation target;
- the valid nonzero three-way parity target and complete compared outputs;
- the required mutation and rejection targets;
- confirmation that no v1.0 Pad-as-matrix-zero compression remains.

The decomposition is complete only when a human can trace:

```text
paper formula → Lean predicate → circuit constraints → composition → physical rows
```

without reading one large verifier implementation.
