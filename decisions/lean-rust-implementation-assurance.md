# Lean-to-Rust Implementation Assurance

**Status:** Accepted

## Problem

Nightstream needs strong evidence that optimized Rust enforces the intended
SuperNeo relation. Rust tests alone are circular when Rust constructs the
circuit, generates its witness, and judges both with its own evaluator.
Agreement between two Rust paths is useful, but both paths can share the same
mistake.

The project needs a precise implementation-assurance claim without presenting
that claim as complete protocol security.

## SuperNeo

SuperNeo v1.1 defines the interactive folding relations and their mathematical
security arguments. It does not certify a Lean model, circuit compiler,
physical layout, package decoder, Rust implementation, proof backend, or
deployment.

Those implementation links need separate evidence. The required boundary is
defined by the [F′ Lean architecture contract](../FPRIME_LEAN_ARCHITECTURE_SPEC.md)
and the [Stage 1 owner goal](../FPRIME_STAGE1_GOAL.md).

## Decision

Lean is the semantic authority for the fixed Nightstream SuperNeo v1.1
relation. Lean must connect the concrete verifier predicate to logical circuit
rows, prove that physical lowering preserves those rows, and emit the only
package that production can use. Rust owns efficient execution of that
package. Rust does not own a second high-level F′ relation.

Conformance requires all of the following on one unchanged package cut:

- the final Rust-expanded `A`, `B`, and `C` matrices equal the Lean-lowered
  rows entry for entry;
- an independent evaluator checks the raw Rust assignment against those Lean
  rows without using Rust's witness generator or row evaluator as its oracle;
- executable Lean, Rust PaperExact, and optimized Rust consume the same valid
  nonzero input and proof and produce the same complete verifier result;
- mutations cover every authoritative statement, transcript, proof, output,
  row, column, and public-input family; and
- the verifier pins the final package identity and production has no reachable
  alternate relation.

This gives stronger implementation assurance for three reasons. Lean makes the
intended computation explicit and proof-checked. Exact matrix equality checks
the relation for every possible assignment, rather than only the selected test
vectors. The independent assignment evaluator prevents the Rust compiler and
Rust evaluator from approving the same lowering error. PaperExact and optimized
Rust then provide separate executable checks of the verifier computation.

The supported claim is:

> Subject to the named trust boundaries, the Rust implementation conforms to
> the Lean-defined relation for the validated package.

This is not a complete security claim. The Lean definitions can still
misstate the paper, and cryptographic security still depends on Poseidon2,
commitment binding, Fiat–Shamir and sampling security, the SuperNeo knowledge
reduction, the proof backend, and production deployment. Deterministic circuit
soundness and cryptographic security composition therefore remain separate
results.
