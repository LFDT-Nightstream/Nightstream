# Nightstream F′ Lean package

This package defines SuperNeo v1.1 and the Nightstream F′ implementation.
The goal is to prove SuperNeo's protocol rules and implementation links,
validate the complete Rust implementation against them, and support later
protocol optimization and reductions in constraint count, including candidates
found with cvc5. Work must close an obligation on that path; architecture
changes must address a demonstrated ownership, dependency or verification
problem.
The authority path is the Lean relation, logical circuit, physical layout,
emitted package, and Rust consumer. See the [architecture contract](../../FPRIME_LEAN_ARCHITECTURE_SPEC.md)
and [Stage 1 goal](../../FPRIME_STAGE1_GOAL.md).

## Proof scope

Prove the soundness and completeness of the concrete checks, exact transcript
and public-input binding, parameter bounds, and the links consumed by the
selected verifier. Keep Pad and the 14 matrix-evaluation families separate.
The selected profile is Goldilocks, `b = 2`, `k_rho = 16`, `B = 65536`,
17 ordered sources and 16 children. Protocol binding uses Poseidon2.

The project does not require a full proof of every cryptographic primitive.
Established cryptographic security results can remain explicit assumptions.
The implementation must use their exact interfaces and satisfy their
hypotheses. A missing check, value equality, or parameter bound cannot become
a cryptographic assumption. The fixed-seed MSIS premise is recorded in the
[approved assumption](../../docs/reviews/nightstream-fprime-requirements/PUBLIC_SEED_MSIS_ASSUMPTION.md).
The [Fiat–Shamir model note](../../docs/reviews/nightstream-fprime-requirements/FIAT_SHAMIR_MODEL.md)
records the current transfer boundary; an additive Poseidon2 transcript does
not by itself establish a published overwrite-sponge theorem's hypotheses.

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
The frozen `formal/nightstream-lean` package is not a production dependency.
