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

SuperNeo v1.2 defines the interactive folding relations and their mathematical
security arguments. It does not certify a Lean model, circuit compiler,
physical layout, package decoder, Rust implementation, proof backend, or
deployment.

Those implementation links need separate evidence. The required boundary is
defined by the [security model](../formal/nightstream-fprime/SECURITY_MODEL.md).

## Decision

Lean is the semantic authority for the fixed Nightstream SuperNeo v1.2
relation. Lean connects the verifier predicate to the rows of each step
circuit and emits the only packages that production can use. Rust owns
efficient execution of those packages. Rust does not own a second high-level
F′ relation. A Rust application replaces only the application region of a
package ([SPEC.md](../SPEC.md) Algorithm 2, step 3).

The conformance evidence is the list of [SPEC.md](../SPEC.md) section 9. On
the Lean exports it has these parts:

- Rust's expansion of each package gives the row count and the row checksum
  of Lean;
- Rust's identity code, witness program and logical assignment give the
  values of Lean;
- the Rust gates, encodings and transcripts give the values of the Lean
  parity files;
- edited statements, proofs and keys fail ([SPEC.md](../SPEC.md), lifecycle
  rule 3); and
- the verifier pins the package identity, so production has no other
  relation that a prover can reach.

Each check compares Rust with a value that Lean emits. The comparison of the
rows covers the relation for each possible assignment, not only for the
selected test vectors.

The supported claim is:

> Subject to the named trust boundaries, the Rust implementation conforms to
> the Lean-defined relation for the validated package on the recorded inputs.

No theorem covers arbitrary Rust execution
([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), trust
item 5). This is not a complete security claim. The Lean definitions can
still misstate the paper, and cryptographic security still depends on the
assumptions and the premises of the security model. Deterministic circuit
soundness and cryptographic security composition stay separate results.
