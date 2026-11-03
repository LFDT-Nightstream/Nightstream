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
each, and what you must trust. `tests/Axioms*.lean` lists every audited
theorem.

## Step circuits

Option B has two step circuits (`Layout/OptionB/StepCircuits.lean`). Circuit 0
runs a light fold, and circuit 1 runs a refresh fold. Both run the same
application step. The state has two slots, one for the accumulator of each
circuit. Each module header states what the module owns. Do not copy row or
column counts into a document.

| Circuit | Lean layout | Gadgets, in order |
|---|---|---|
| Light | `Lifecycle/OptionB/LightStepLayout.lean` | The application, the depth guard on the depth word of the active slot, the prior state hash with its two slot hashes, the next state hash with its two slot hashes, the light Π_CCS and Π_RLC on the active slot, and the slot select. |
| Refresh | `Lifecycle/OptionB/RefreshStepLayout.lean` | The application, the prior state hash with its two slot hashes, the next state hash with its two slot hashes, Π_DEC on the active slot, the refresh Π_CCS on the 16 children, Π_RLC, and the slot select. There is no depth guard, and the new depth of the slot is 0. |
| Finish | `Lifecycle/OptionB/FinishStep.lean` | The last fold of a compressed proof. It is the refresh step circuit without the application and without the hashes of the next slots and the next state. |

`Lifecycle/OptionB/Step.lean` states the step relation (`stepHolds_iff`).
`LightStep.lean` and `RefreshStep.lean` give its two cases from the rows of
each circuit, and `StepCircuits.soundness` gives the case of either circuit.
`LightAssemblerCompleteness.complete` and
`RefreshAssemblerCompleteness.complete` give completeness.
`Lifecycle/OptionB/Terminal.lean` owns the terminal relation.
`Export/OptionB/StepRelation.lean` and `Export/OptionB/TerminalBridge.lean`
connect the exported relations to the step relation and the terminal.
The Lean results in `Spec/Folding/Light`, `Spec/Folding/Refresh`,
`Spec/Folding/Schedule.lean` and `Spec/HyperNova/Construction2/BoundedDepth*`
are the authority for the departures from SuperNeo v1.2.

`lake exe emitOptionB` (`Export/OptionB/EmitMain.lean`) writes the light and
refresh packages, `artifacts/nightstream-fprime-option-b-light-v1.json` and
`artifacts/nightstream-fprime-option-b-refresh-v1.json`.
`lake exe emitOptionB --finish` writes
`artifacts/nightstream-fprime-option-b-finish-v1.json`. Rust pins the
identities of these packages (`OPTION_B_PACKAGES_IDENTITY` and
`OPTION_B_FINISH_IDENTITY` in `crates/nightstream-fprime/src/option_b.rs`).

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
The selected profile is Nightstream Goldilocks, k_rho = 16, with `b = 2` and
`B = 65536`. A light fold has two sources, the fresh instance and the
accumulator. A refresh fold has 17 ordered sources, the fresh instance and the
16 Π_DEC children of the accumulator. Protocol binding uses Poseidon2.

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
| Proof or compiler refactor with unchanged rows | Preserve the public statements, executable operations, row order and encoding. Layout owns the physical transformations; Export consumes them. | Static, library and axiom gates; the affected real consumer; an unchanged package identity against the Rust pins. Record the checked commit. |
| Constraints inside a gadget with the same interface | Keep the gadget specification. Prove soundness, completeness, footprint and variable support. For row removal, prove that the retained rows imply the original specification; the old witness supplies completeness only after the placement and witness interface are checked. | The gadget and its real parent consumer; static, library and axiom gates; regenerate the selected package and run the affected Lean/Rust valid-input and rejection checks. |
| Witness footprint, column reuse or row order | The gadget owns its footprint. Phase owners derive starts and expose support and semantic theorems. Layout proves that relocation and lowering preserve the specification; package counts and selected value checks follow the derived geometry. | Affected size bounds and Values checks, selected package emission and identity re-pin, matrix and assignment parity, and affected mutation checks. List the exact executed coverage. |
| Transcript, digest format, challenge distribution or protocol parameters | Revise the semantic verifier and the affected security argument together. Protocol binding and Rust must use the same revision. The production decomposition and Poseidon2-only rules still apply. | An approved concrete protocol change, revised Lean statements and audits, updated per-call and cumulative probability assumptions where affected, a new package identity and the corresponding conformance evidence. |

Rust compares the identity of the emitted packages with its pins. A changed
identity in a refactor is a failed check. Updating the pins does not repair
that refactor. Moving a file, passing
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

Run `scripts/validate.sh static`, `scripts/validate.sh build`,
`scripts/validate.sh axioms`, and `scripts/validate.sh compression`, in that
order, before a proof checkpoint. The last one rebuilds `../compression-lean`,
which imports this package.
Follow [AGENTS.md](AGENTS.md) for bounded commands and the single build queue.
