# Nightstream F′ package instructions

These instructions apply to `formal/nightstream-fprime`. The architecture
contract is `/FPRIME_LEAN_ARCHITECTURE_SPEC.md`; the goal is
`/FPRIME_STAGE1_GOAL.md`. Both outrank this file.

## Purpose and proof scope

Prove SuperNeo v1.1 and its implementation links so that Lean can support
constraint reduction and validation of the complete Rust implementation.
The authority path is the protocol relation, logical circuit, physical
layout, emitted package and Rust consumer.

- Before adding work, name the protocol, implementation, parameter or
  optimization obligation it closes. Architecture work must address a
  demonstrated ownership, dependency or verification problem on this path.
- Established cryptographic security results can remain explicit assumptions
  with their exact model, parameters and applicability conditions. Prove
  additional supporting facts when a relied-on claim requires them. A full
  proof of each cryptographic primitive is not the project goal. Missing
  checks, value correspondences, encoding rules and parameter bounds cannot
  be replaced by cryptographic assumptions.
- For each probabilistic claim used by the implementation, state its event,
  experiment and per-call bound. Give the cumulative bound for the intended
  number of uses, with query count and recursion depth where relevant. Keep
  the use count as a parameter. Select concrete counts from deployment
  requirements, not illustrative examples. Keep sampler failure, verifier test error
  and extraction loss separate; an isolated bound is not total security.
- cvc5 proposes candidates. For a constraint optimization that preserves
  semantics, Lean must prove that the new constraints imply the unchanged
  specification without stronger assumptions, and that valid specification
  instances still have witnesses. Protocol changes require a revised
  specification and the affected correctness and security arguments.
- Connect soundness and completeness to the complete selected Rust path.
  Check the same relation, parameters, transcript, package and outputs, with
  valid-input and rejection evidence. State the scope of executed checks;
  passing examples are not universal proofs of Rust semantics.
- Extractor work uses its declared mathematical clock. Do not require a
  machine-runtime refinement unless a claim being delivered relies on it.

## Layers

Imports flow downward only. `scripts/check-boundaries.sh` enforces it.

```text
Spec       paper semantics (field, ring, Poseidon2, sumcheck, PiCCS/PiRLC/PiDEC
           verifiers, HyperNova F′ relation)
Circuit    the DSL: expressions, offset variables, operations, circuit monad,
           one FormalCircuit record, the circuit_norm simp set
Gadgets    one directory per gadget; exports only its FormalCircuit
Lifecycle  concrete F′ phase order, carried state, public layout, XOut
Layout     physical lowering and layout-preservation theorems
Export     serializer, witness IR, relation identifier, lake exe emitter
tests      axiom gate (explicit imports, `#audit_axioms` per theorem)
```

## Rules

- No import from `formal/nightstream-lean`. Copy the smallest audited
  definition and put a provenance comment (source path, commit) at the top.
- No generated modules, no embedded artifact data, no `native_decide`, no
  `sorry`/`admit`/`axiom`/`unsafe`, no file at or above 1,500 lines, no glob
  in `lakefile.toml`, one profile (`b = 2`, `k_rho = 16`).
- No `maxRecDepth`/`maxHeartbeats` override, except a per-declaration
  `set_option … in -- fixed-size: …` whose cost axis is a protocol constant
  (ring degree 54, Poseidon2 width) and never emitted data. Each such site is
  a recorded debt to replace by a structural proof. The boundary script
  rejects every other form.
- Subcircuits are opaque to parents: a parent proof that unfolds a child's
  operations is a bug. Proof cost must not grow with rows, columns, or
  schedule length.
- Every exported theorem is listed in `tests/Axioms.lean` with
  `#audit_axioms`. Allowed axioms: `propext`, `Classical.choice`,
  `Quot.sound`.
- Lean commands only through `scripts/validate.sh` (`static`, `build
  [target]`, `axioms`, `file <path>`, `all`), each under the 1,500 s cap.
  One Lean or Rust build process at a time.
- Before each command or edit: one active acceptance criterion and its
  closing evidence. Three rounds without closure: stop and report.

## Development speed

Fast feedback is a project requirement. Slow builds and generators multiply
the time for every proof, compiler, and conformance change and can make Stage 1
development impractical.

- Keep incremental Lean builds, emitters, fixture generators, row streamers,
  and parity generators as fast as the current design permits. Measure elapsed
  time before and after a change that can affect these paths, and report the
  measured result. Treat the 1,500 s cap as a failure ceiling, not as a target
  or an acceptable development-loop duration.
- Do not construct artifact-sized `List`, codec `Value`, token stream, or
  `String` values when the same result can be produced by a proved compact
  plan or an ordered stream. Delay closed executable data behind an explicit
  argument, keep proofs structural, and process large data in bounded blocks.
- Use the available CPU cores for independent immutable work in builds and
  generators. Derive concurrency from the Lean runtime or host hardware; do
  not hard-code an arbitrary worker count. Prepare independent blocks in
  parallel, then commit their bytes and identities in the canonical
  Lean-proved order.
- The one-build-process rule still applies. Parallel work must occur inside
  that one authorized build, test, emitter, or generator process. Do not start
  competing Lean or Rust processes to obtain parallelism.
- Speed work must not move semantic authority into Rust, weaken a proof, skip
  a gate, change canonical ordering, or make a digest authoritative. When a
  development path becomes materially slower, fix or redesign that path
  before extending it with another large phase.
