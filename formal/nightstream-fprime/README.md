# Nightstream F′ Lean package

This package defines SuperNeo v1.2 and the Nightstream F′ implementation.
The goal is to prove SuperNeo's protocol rules and implementation links,
validate the complete Rust implementation against them, and support later
protocol optimization and reductions in constraint count, including candidates
found with cvc5. Work must close an obligation on that path; architecture
changes must address a demonstrated ownership, dependency or verification
problem.
The authority path is the Lean relation, logical circuit, physical layout,
emitted package, and Rust consumer. See the [architecture contract](../../FPRIME_LEAN_ARCHITECTURE_SPEC.md)
and [Stage 1 goal](../../FPRIME_STAGE1_GOAL.md).

Lean is the source of truth for every result. [SECURITY_MODEL.md](SECURITY_MODEL.md)
states the adversaries that the results protect against, the Lean result for
each, and what you must trust. `tests/EvidenceTargets.lean` has the literal
statements of the registered targets, and `tests/Axioms*.lean` lists every
audited theorem.

## Phase map

The Stage 1 step circuit `F′` has eight children, in this order
(`Lifecycle/Stage1/Formal.lean`). Each module header states what the module
owns. Row and column counts are Lean theorems in
`Export/Stage1/Poseidon2HashChainV1Package.lean`: `structuralRowCount`
(logical rows), `logicalWidth` (logical coordinates), and
`physicalPackageRowCount` and `physicalPackageTotalColumnCount` (physical
package). Do not copy these numbers into a document.

| Child | Paper | Lean module | Checks |
|---|---|---|---|
| Prior state hash | HyperNova Construction 2 | `Lifecycle/PriorStateHash.lean` | The prior state hash of the running instance. |
| Output state hash | HyperNova Construction 2 | `Lifecycle/OutputHash.lean` | The public output digest of the next state. |
| Π_CCS | SuperNeo v1.2 Section 7.3 | `Lifecycle/PiCCS/v1_2/Formal.lean` | The complete Π_CCS reduction in its exact transcript order. |
| Π_RLC | SuperNeo v1.2 Section 7.4 | `Lifecycle/PiRLC/v1_2/Formal.lean` | The random linear combination of the sources. |
| Π_DEC | SuperNeo v1.2 Section 7.5 | `Lifecycle/PiDEC/v1_2/Formal.lean` | The split of the parent into the children, with their commitments and evaluations. |
| Running instance | HyperNova Construction 2 | `Lifecycle/Stage1/RunningTransition.lean` | The default running instance at the base step, else the Π_DEC output. |
| Application | Nightstream | `Lifecycle/Stage1/Application.lean` | The step of the verifier-selected application. |
| Next preimage | HyperNova Construction 2, step 5 | `Lifecycle/Stage1/NextPreimage.lean` | The next state-hash preimage keeps `z0` and increments `i`. |

Outside `F′`, `Lifecycle/Stage1/Terminal.lean` owns the terminal relation and
`Lifecycle/Stage1/VerificationKey.lean` the verification-key binding.
`Layout/Stage1/AssemblerSoundness.lean` composes the children's semantics,
`Layout/Stage1/PreservationClosure.lean` proves that the physical layout
preserves it, and `Export/Stage1/PerApplicationSoundness.lean` proves that the
emitted package rows imply the typed step.

## Credits

The `Circuit/` DSL (operations, opaque subcircuits, the `FormalCircuit`
record) and the exported witness IR follow the design of
[Clean](https://github.com/Verified-zkEVM/clean) by zkSecurity (MIT License).
The code is a fresh implementation. The headers of `Circuit/Basic.lean` and
`Circuit/StraightLine.lean` state the exact boundary.

## Proof scope

Prove the soundness and completeness of the concrete checks, exact transcript
and public-input binding, parameter bounds, and the links consumed by the
selected verifier. Keep Pad and the matrix-evaluation families separate.
The selected profile is Goldilocks, `b = 2`, `k_rho = 16`, `B = 65536`,
17 ordered sources and 16 children. Protocol binding uses Poseidon2.

The project does not require a full proof of every cryptographic primitive.
Established cryptographic security results can remain explicit assumptions.
The implementation must use their exact interfaces and satisfy their
hypotheses. A missing check, value equality, or parameter bound cannot become
a cryptographic assumption. [SECURITY_MODEL.md](SECURITY_MODEL.md) lists the
cryptographic premises in use.

Additional proofs are required when their absence would leave a protocol
check, a Rust difference, or a parameter error undetected. A cryptographic
primitive's full security proof and an extractor's machine-time analysis
are not prerequisites for a deterministic circuit-correctness claim.

## Probability and work

State each error event and the experiment in which its bound holds. For
repeated use, retain the number of folds and the adversary's query count as
parameters. Evaluate the cumulative bound at the intended deployment count;
illustrative counts are not requirements or defaults. Per-call bounds
conditional on the full preceding history give an aggregate bound
`min(1, sum of the per-call bounds)`; independence is not required. A uniform
per-call bound `epsilon` gives `min(1, n * epsilon)` for the union of those
events over `n` calls. Sampler abort probability, decoder distribution error,
verifier test error, and extraction loss are different terms. A union bound for one
term is not a complete NIFS or history-extraction security estimate. Do not
select a deployment count or recursion-depth limit from an example.

`PaperForkExtractionWork.Result.work` is a declared mathematical clock.
Its equations and bounds concern the returned counters and the explicit
driver charges. They do not establish Lean or Rust execution time. Use
“expected work under the declared clock model” for those results. Preserve
value correctness separately; a fast or small counter is not evidence that
an implementation computes the required value.

## Constraint reduction and Rust checks

cvc5 can propose removals, column reuse, and counterexamples. For an
optimization that preserves semantics, Lean must prove that the changed
constraints imply the unchanged specification without stronger assumptions
and that valid specification instances retain a satisfying witness. Removing
rows alone preserves completeness for the same witness layout; changing columns or witness
construction needs its own preservation proof. An unchecked `UNSAT` result
does not authorize a change. A protocol change needs a revised specification
and the affected correctness and security arguments.

| Change | Owner and proof obligation | Required evidence |
|---|---|---|
| Proof or compiler refactor with unchanged rows | Preserve the public statements, executable operations, row order and encoding. Layout owns the physical transformations; Export consumes them. | Static, library and axiom gates; the affected real consumer; `validate.sh identity` for the selected package. Record the checked commit. |
| Constraints inside a gadget with the same interface | Keep the gadget specification. Prove soundness, completeness, footprint and variable support. For row removal, prove that the retained rows imply the original specification; the old witness supplies completeness only after the placement and witness interface are checked. | The gadget and its real parent consumer; static, library and axiom gates; regenerate the selected package and run the affected Lean/Rust valid-input and rejection checks. |
| Witness footprint, column reuse or row order | The gadget owns its footprint. Phase owners derive starts and expose support and semantic theorems. Layout proves that relocation and lowering preserve the specification; package counts and selected value checks follow the derived geometry. | Affected size bounds and Values checks, selected package emission and identity re-pin, matrix and assignment parity, and affected mutation checks. List the exact executed coverage. |
| Transcript, digest format, challenge distribution or protocol parameters | Revise the semantic verifier and the affected security argument together. Protocol binding and Rust must use the same revision. The production decomposition and Poseidon2-only rules still apply. | An approved concrete protocol change, revised Lean statements and audits, updated per-call and cumulative probability assumptions where affected, a new package identity and the corresponding conformance evidence. |

`validate.sh identity` compares the freshly emitted canonical binding with the
current fixture and Rust pins. A changed identity in a refactor is a failed
check; updating the pins does not repair that refactor. Moving a file, passing
a parser, matching package hashes or passing selected Rust examples does not
by itself prove these obligations.

Rust validation must cover the complete selected path from package loading
and witness generation to proving and verification. Each result must state
which part of that path it establishes and any remaining assumptions or
implementation gaps. Rust conformance uses the same selected relation, key,
parameters, authenticated input, transcript, proof bytes, and complete phase outputs.
Retain malformed-input and output-mutation checks. Tests establish their
recorded execution scope; they are not universal proofs of Rust semantics.
Package or relation changes require the corresponding identity and consumer
checks. A digest alone is not protocol authority.

Run `scripts/validate.sh static`, `scripts/validate.sh build`, and
`scripts/validate.sh axioms`, in that order, before a proof checkpoint.
Follow [AGENTS.md](AGENTS.md) for bounded commands and the single build queue.
