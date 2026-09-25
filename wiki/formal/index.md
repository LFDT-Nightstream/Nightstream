# Formal (Lean)

`formal/nightstream-lean/` is the active assurance-first Lean 4 project. Its
own `README.md`, `AGENTS.md`, property specification, and evidence ledger
define the maintained proof surface and the exact meaning of model-level,
artifact-checked, Rust-conformant, and security-reduced claims.

The other packages under `formal/` are legacy reference material pending
deletion. They are not dependencies of the active project and are not
authoritative merely because a theorem kernel-checks there. Reusable algebra
lemmas or counterexamples must be copied selectively, restated with the active
types and assumptions, and validated through the active project's gates.

## How the formal work relates to the Rust code

- The Rust crates' specifications state implementation contracts. Independent
  paper semantics in `formal/nightstream-lean` define what must be proved; a
  generated artifact or historical circuit is never semantic authority.
- A Lean theorem authorizes a Rust/R1CS change only after the theorem is tied to
  the exact production data through the stated refinement and conformance gate.
- Read `formal/nightstream-lean/AGENTS.md` before changing the active project;
  use its bounded validation wrapper rather than invoking Lean directly.
