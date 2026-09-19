# Nightstream crate goal

Status: implementation goal. This document records the agreed direction; it
does not record completed implementation or validation.

## Lemmas

These are fixed project requirements, not new mathematical theorem claims.
Every section below must respect them.

1. **Lean changes are export-only.** Keep `formal/nightstream-fprime` unchanged
   except for structured exports for Rust: export code, serialization,
   component metadata and export-specific checks that preserve the existing
   circuit. Do not modify the protocol implementation, circuit definitions,
   mathematical algorithms, physical layout rules, application proof
   interface or existing proofs.
2. **The generic assembler belongs in Rust.** Rust selects exported
   components, applies their supported parameters, connects them to
   Rust-defined application circuits and prepares the final circuit. Do not
   add a Lean application compiler or a new Lean assembly framework.
3. **Rust reuses the shared recursive verifier.** Its SuperNeo verification,
   transcript and state-binding constraints come from the existing Lean
   implementation through structured exports. Rust must not recreate these
   formulas by hand. Application circuits are Rust-owned; the independent
   Rust Poseidon2 application is the first golden-test subject.
4. **Users never need Lean.** Production builds, application circuit
   preparation, proving, verification and ordinary Rust tests must work with
   Lean absent. Lean artifact generation and export validation run only in
   the separate maintainer workflow. No hidden Lean call or Lean runtime is
   allowed in the user path.
5. **Proof gaps do not expand the scope.** Reuse existing Lean results only
   under their actual hypotheses. If a requirement needs Lean work beyond
   exports, report that limitation. Do not change the implementation, add a
   new non-export proof framework, weaken a guarantee or claim an unsupported
   result to make the goal appear complete.
6. **Rust work starts in a new crate.** Build the replacement in
   `crates/nightstream`. Keep `neo-fold-clean` intact as a reference and
   comparison baseline for this goal. Copy only the code the replacement
   needs; do not cut files out of the old crate, refactor it or continue
   development there. The new production dependency graph must not include
   `neo-fold-clean`.
7. **The Lean Poseidon2 export supplies the golden reference.** Independently
   implement the same `Poseidon2HashChainV1` application in Rust and assemble
   it with the shared Lean-exported recursive-verifier constraints. The
   assembled circuit and execution results must match the existing Lean
   export and recorded outputs for the same inputs, profile, key and
   transcript randomness. Comparing only the hash output is insufficient.
   The Rust application builder must not use the expected application rows
   to generate the circuit under test.

Here, `formal/nightstream-fprime` is the Lean project.
`crates/nightstream-fprime` is a separate Rust crate and is not frozen by the
export-only restriction.

## Outcome

Build a small Rust crate at `crates/nightstream` to replace the selected
production lifecycle in `neo-fold-clean`. Reuse the Lean work and existing
optimized arithmetic. Rust users must be able to define application circuits
without installing or running Lean.

A generic circuit assembler is a required deliverable. It must combine
supported Rust-defined applications with the Lean-generated recursive
verifier. Completing only the fixed Poseidon2 application does not complete
this goal.

The assembler is implemented in Rust. Lean exports reusable components in a
structured format that Rust can select, instantiate and connect. This goal
does not move application construction into Lean.

The ownership split is:

| Part | Authority and implementation |
| --- | --- |
| Shared recursive verifier | Existing Lean circuits and proofs define the SuperNeo PiCCS, PiRLC and PiDEC verifier, transcript checks and recursive state binding. Rust consumes structured exports of these components. |
| Application circuit | Users define its constraints in Rust. The existing Lean Poseidon2 application supplies the first golden reference; new user applications do not require a Lean definition. |
| Assembly | A Rust assembler selects exported components, substitutes supported parameters, allocates variables and connects the application. Lean exports component constraints and their interface/layout contracts. |
| Proving and terminal verification | Rust uses native arithmetic and optimized prover code. Terminal verification checks the expected state, all running claims and the latest fresh claim. |

Rust must not recreate the shared recursive verifier as another handwritten
circuit. Rust may independently implement an application circuit, even when
that application uses an operation such as Poseidon2 that also occurs inside
the shared verifier.

This distinction is deliberate: the first Rust Poseidon2 application is an
independent golden-test subject. Poseidon2 constraints inside the recursive
verifier continue to come from Lean.

## Fixed requirements

- Preserve the selected SuperNeo v1.2 behavior and HyperNova-style recursive
  lifecycle. The current selected baseline is recorded in
  [STAGE1_BASELINE.md](scripts/lean_graph/STAGE1_BASELINE.md).
- Preserve the existing Nightstream Goldilocks profile: `b = 2`,
  `k_rho = 16`, `B = 2^16`, one fresh claim and sixteen carried claims.
  Use Poseidon2 for protocol binding. This goal does not change the protocol,
  approve a new setup or select another profile.
- Preserve the selected package, key, transcript and outputs for the initial
  migration and golden comparison. Application-dependent changes require
  explicit binding and validation; do not silently reuse an unrelated key or
  verifier identity.
- No `lean`, `lake`, Lean runtime or Lean invocation from `build.rs` is
  allowed on the Rust user paths defined in the Lemmas.
- Generate and check Lean artifacts in a separate development workflow.
  Publish the required artifacts with the Rust library. Ordinary Rust tests
  consume saved references without invoking Lean.
- Reuse existing field, ring, commitment, transcript and reduction kernels.
  Use the existing shared crates directly. Where required code belongs to
  `neo-fold-clean`, copy only that code into the replacement and adapt it
  there. Leave the source code in the old crate intact.
- Preserve the completed Lean implementation, proofs and selected artifacts
  as the baseline. All Lean-side work is subject to the Lemmas.
- The new production dependency graph must not include `neo-fold-clean`.
  Development comparisons may run the unchanged old crate. New application,
  assembler and lifecycle work belongs in `nightstream`, not in the old crate.
- Keep fixture generators and evidence tools outside the production library
  dependency graph. Keep the public lifecycle interface small and direct.

## What is reusable, and what still needs work

The shared verifier is reusable protocol logic. This does not mean its final
matrices are identical for every application. Application dimensions can
change variable positions, witness encoding, active row counts, widths and
binding within the existing fixed profile. The selected padded domain remains
`2^28`, as defined by `Lifecycle.cubeVariables`; it does not make all active
dimensions constant. Export only the parameter ranges and layout rules
already supported by the existing implementation and contracts.

Use the existing compact row and witness formats where they fit. Generated
templates must contain the constraint formulas and their parameter rules.
A gadget name alone is not sufficient if Rust then reconstructs its formulas
by hand. Native implementations that compute witness values remain allowed;
the generated constraints must check their results.

The assembly contract must state and validate:

- the application's input state, private inputs and output state;
- variable allocation and the connections to the recursive verifier;
- the encoding needed to preserve the selected witness norm bounds;
- dimensions, public-input alignment and recursive size requirements;
- the application, profile and setup bound to the verifier's expected
  circuit identity.

The verifier must obtain its expected application from its own configuration,
not accept a prover-selected application or a self-consistent digest as
authority.

The current Lean application interface requires a Lean `Program` and its
proofs. It is not already a generic Rust application interface. Reuse its
results only where their hypotheses apply; a Rust-defined application does
not automatically satisfy that proof interface.

The first golden circuit checks one concrete assembly. The generic Rust
assembler must also be delivered, but tests of it must not be presented as a
universal Lean proof of Rust execution or arbitrary application semantics.

## Required structured Lean exports

Export the shared recursive verifier as a named component, with its required
child components and reusable gadgets separately identifiable where useful.
Rust must not find components by slicing an opaque application-specific file
at undocumented offsets.

Each exported component must describe:

- its stable identifier, format version, profile and dependencies;
- named input and output ports, their shapes and public/private roles;
- its actual constraint formulas or compact row templates;
- local variable allocation, row footprint and permitted relocation;
- supported parameters and the conditions under which they are valid;
- witness instructions and any inputs computed by native routines;
- the Lean definition and proof contracts that justify the component.

Use symbolic ports and local indices so Rust can connect components without
copying hidden constants from Lean source files. Preserve the existing
selected package as a reference against which the structured export is
checked. Export-specific checks may establish correspondence with the
unchanged circuit and layout; they must not introduce new circuit semantics
or layout rules.

Metadata must describe existing components faithfully. It must not invent new
parameter freedom, supported row forms or composition guarantees. If the
existing exportable component cannot meet an assembler requirement, apply
the limitation rule in the Lemmas.

Selecting components must not permit omission of required recursive-verifier
checks. The complete F′ assembly must retain the prescribed phase order,
transcript flow, state binding, application transition and terminal boundary.

## Required generic Rust circuit assembler

Rust owns the application interface and the assembly algorithm. It must:

- load the exported shared verifier and other needed components by their
  declared interfaces;
- accept application constraints and witness inputs from Rust users;
- instantiate supported parameters, allocate variables and connect the
  declared ports while preserving every component's constraints;
- combine the constraint and witness plans, validate their dimensions and
  encoding conditions, and bind the exact final circuit for verification.

Use the exported interface and layout contracts to justify these operations.
Reuse existing Lean composition and relocation proofs where they apply,
without changing or extending the Lean implementation beyond exports.
Identify remaining Rust implementation assumptions and proof-coverage gaps
explicitly. Users do not supply a Lean `Program` or a proof object to the
assembler.

New applications require only Rust preparation and the correct expected
circuit binding. No per-application Lean generation or hidden call to Lean is
allowed. This interface describes application constraints; it does not claim
to prove arbitrary Rust program execution or the author's intent.

Check the assembler against the Lean reference, including malformed component
metadata, invalid dimensions and changed public/private wiring. Demonstrate a
Rust-defined application other than the golden hash chain through the same
assembler with Lean absent. Keep these execution results distinct from the
existing component and composition proofs; this demonstration does not add
a universal Lean theorem about the Rust assembler.

Existing per-application Lean results are the starting point, including
[PerApplicationPackage](formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationPackage.lean)
and
[PerApplicationFixedPoint](formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationFixedPoint.lean).
These are read-only proof references. Apply their conclusions only when the
existing `Program`, size and other hypotheses are met. If the Rust interface
does not supply a hypothesis, record the uncovered claim instead of adding a
new Lean application framework. Their existence does not establish that the
Rust assembler is already complete.

## First milestone: independent Poseidon2 application golden test

Use the existing `Poseidon2HashChainV1` application as the reference:

```text
next_state = Poseidon2(domain_tag ++ previous_state ++ message)
```

Use its exact domain tag, four-word state, four-word message, field, hash
parameters and encoding. These values come from the existing
[Lean application](formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Stage1/Poseidon2HashChainV1.lean).

Build this application circuit independently through the new `nightstream`
application interface. Its builder must not read the expected application
rows or delegate to the Lean-generated application template. It still reuses
the Lean-generated shared recursive verifier. Run this milestone through the
new crate from the start.

The expected circuit is the complete Lean export: the Poseidon2 application
and shared recursive verifier. The actual circuit is the Rust-built
Poseidon2 application combined with the shared Lean-exported verifier through
the Rust assembler. The shared constraints are reused; the application
builder and assembly are what this comparison checks.

The golden check must:

1. Compare every application constraint, coefficient, variable index and
   public-input connection with the separately emitted Lean reference.
   Establish the same layout for this first comparison; a different optimized
   layout needs an explicit correspondence argument.
2. Check the assembled application and recursive-verifier boundary against
   the selected Lean package, including the final matrix representation used
   by the prover. Matching hash outputs, row counts or artifact digests alone
   is not sufficient.
3. Use the same inputs, profile, key and transcript randomness to compare
   execution results with the recorded Lean outputs. Check valid witnesses
   against the reference and reject changed outputs, inputs or intermediate
   values that violate the constraints. Include a deliberately changed
   constraint in the comparison test. The complete recursive comparison is
   specified in New crate integration below.
4. Build and run these Rust checks in an environment with Lean absent, using
   recorded reference artifacts. Record their source, generation command and
   supported dimensions.

Complete matrix equality establishes equality for this concrete circuit and
representation. It does not prove all behavior of the Rust builder or support
for arbitrary application circuits.

## New crate integration

Create `crates/nightstream` with its own public application and lifecycle
interfaces. Implement the generic assembler there. Copy only the selected
working state, folding composition and verification code that it needs from
`neo-fold-clean`, and adapt the copies to the new interfaces. Do not copy the
whole crate or add a wrapper around its old API.

Reuse suitable shared Rust crates directly, including the existing
`crates/nightstream-fprime` export loader and runtime where they fit the
structured export contract. A new lifecycle crate does not require a second
copy of each shared library.

Leave `neo-fold-clean` and its existing consumers intact during this work.
Use it as a comparison baseline; do not add features or repair or refactor it
for the replacement. Switching other consumers and deleting the old crate
are separate steps after validation of the replacement.

Run the Rust-defined golden application through the complete recursive path:
PiCCS, PiRLC, PiDEC, successor construction, use of that successor in the next
fold, and terminal acceptance and rejection checks. Compare the required
messages, transcript states, commitments, assignments and outputs with the
applicable existing reference evidence.

Preserve the separation between public claims and prover-held witnesses.
Checking the successor circuit does not replace terminal checks of the
remaining sixteen carried claims and latest fresh claim.

Copying a file with unchanged bytes does not automatically transfer execution
evidence to a new build. Record the path, dependency and toolchain changes and
run the affected checks. Reuse unchanged mathematical proofs and valid
reference outputs with explicit scope.

## Performance and validation

Prepare application-specific circuit data once and reuse it across steps.
Use compact representations and native kernels in the expensive computations.
Avoid expanding repeated structures merely to serialize or copy them again.

Measure preparation time, proving time, verification time and peak memory on
the same inputs and host as the baseline. Include loading and conversion
costs. Resolve measured regressions before claiming the replacement is faster.
This goal sets no unmeasured runtime or source-line target.

Use focused Rust tests and export validation during development. Rust
optimizations must preserve the exported constraints; changing the Lean
constraints is outside this goal. Full production-profile replay remains
integration evidence, not a required substitute for a proof after every local
edit. Required conformance gates still apply when their implementation or
circuit identity changes.

Follow [AGENTS.md](AGENTS.md) and the applicable subproject instructions,
including their command time limits and build queue rules.

## Completion and limits

Report these results separately:

- **Structured exports:** Rust can select the shared recursive verifier and
  needed components through explicit interfaces, with checked connections to
  their existing Lean constraints and proof contracts.
- **Golden circuit:** the independent Rust Poseidon2 application, assembled
  with the shared Lean-exported verifier constraints, matches the complete
  Lean reference circuit and its recorded execution results on the same
  inputs.
- **Crate integration:** `crates/nightstream` reuses the generated recursive
  verifier, completes the recursive and terminal checks, has no production
  dependency on the old crate, and builds and runs without Lean. Its own
  build and conformance checks pass; tests of the old crate alone do not
  establish completion. The old crate remains intact.
- **Generic assembler:** Rust accepts supported application descriptions and
  combines them with the exported recursive verifier without Lean. Rust checks
  the declared wire, dimension, encoding and binding conditions. Component
  and assembly conformance gates pass. State which connections are proved and
  which remain tested or trusted.

All of these results are required to complete this goal. The golden test is
the first milestone, not a substitute for the generic assembler. State the
supported circuit interface and bounds explicitly; do not claim support for
arbitrary Rust programs or circuits outside those bounds.

Completion also requires respecting the Lemmas. A result obtained by changing
the Lean implementation beyond structured exports does not satisfy this
goal. Report any unmet requirement or uncovered assurance claim explicitly.

Lean remains the authority for the shared verifier constraints and their
proofs. Rust decoding, assembly, arithmetic and terminal verification remain
explicit implementation boundaries to the extent not covered by existing
applicable proofs. This goal does not authorize a new Lean refinement project
to remove them. Comparison tests do not erase those boundaries or prove an
application author's intent.

This goal does not approve a production SNARK backend, start Nebula work,
unfreeze `formal/nightstream-lean`, or authorize deleting other consumers.
The existing mathematical architecture in
[FPRIME_LEAN_ARCHITECTURE_SPEC.md](FPRIME_LEAN_ARCHITECTURE_SPEC.md) remains the
reference for the shared verifier. This goal adds the Rust application
boundary, generic assembly connection and independent golden test; it does
not create a second F′ relation or mark earlier assurance work complete.
A Rust-defined application does not automatically meet the earlier
per-application Lean closure requirements. State that scope difference; do
not alter the Lean implementation to make the new path fit the earlier proof
interface or claim that its closure requirements have been met.
