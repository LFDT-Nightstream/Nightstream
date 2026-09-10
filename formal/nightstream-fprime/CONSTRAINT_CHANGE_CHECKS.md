# Constraint change checks

The invariant is the unchanged semantic specification. For an optimization,
every assignment that satisfies the new rows must satisfy that specification
under the same assumptions. Every valid specification instance must still
have a witness. cvc5 can propose a change; Lean must check both obligations.

| Change | Owner and proof obligation | Required evidence |
|---|---|---|
| Proof or compiler refactor with unchanged rows | Preserve the public statements, executable operations, row order and encoding. Layout owns the physical transformations; Export consumes them. | Static, library and axiom gates; the affected real consumer; `validate.sh identity` for the selected package. Record the checked commit. |
| Constraints inside a gadget with the same interface | Keep the gadget specification. Prove soundness, completeness, footprint and variable support. For row removal, prove that the retained rows imply the original specification; the old witness supplies completeness only after the placement and witness interface are checked. | The gadget and its real parent consumer; static, library and axiom gates; regenerate the selected package and run the affected Lean/Rust valid-input and rejection checks. |
| Witness footprint, column reuse or row order | The gadget owns its footprint. Phase owners derive starts and expose support and semantic theorems. Layout proves that relocation and lowering preserve the specification; package counts and selected value checks follow the derived geometry. | The allocation experiment, affected size bounds and Values checks, selected package emission and identity re-pin, matrix and assignment parity, and affected mutation checks. List the exact executed coverage. |
| Transcript, digest format, challenge distribution or protocol parameters | Revise the semantic verifier and the affected security argument together. Protocol binding and Rust must use the same revision. The production decomposition and Poseidon2-only rules still apply. | An approved concrete protocol change, revised Lean statements and audits, updated per-call and cumulative probability assumptions where affected, a new package identity and the corresponding conformance evidence. |

Use `scripts/validate.sh static`, `build` and `axioms` in that order for a
checkpoint. The one-build-queue rule and the project command caps apply.
`identity` compares the freshly emitted canonical binding with the current
fixture and Rust pins. A changed identity in a refactor is a failed check;
updating the pins does not repair that refactor.

The allocation experiment uses a separate clean worktree. It adds one unused
cell per PiDEC split invocation while preserving the constraints and semantic
specification. The changed gadget's proofs, selected numeric checks, parent
bounds that name its size and package counts may need updates. Unrelated
phase proofs must remain valid through their interfaces. Report every reached
failure and the point where the build stopped. A failure in a default-value
module can hide later failures; it does not establish full isolation.

Use the existing source, physical plan and serialized package representations.
A new view must derive from its owner by an explicit conversion. Preserve the
source-range, witness and semantic interfaces through the real consumer.
Moving a file, passing a parser, matching package hashes or passing selected
Rust examples alone does not prove all these obligations. The
[assurance surface](ASSURANCE_SURFACE.md) names the current final theorem
scope and remaining premises.
