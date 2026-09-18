# AGENTS.md

## Formal Lean Subproject
- Use this 2-layer layout for each formalized component:
  - Human spec: `specs/<Name>.spec.md`
  - Lean implementation: `SuperNeo/<Section>/<Name>.lean`
- `<Name>Interface.lean` files exist only where they are a real consumed boundary:
  - a downstream package imports the module (e.g. `PiCCSInterface`, `PiRLCInterface`,
    `PiDECInterface`, `DecompInterface` consumed by `direct-ccs-fprime-lean`), or
  - the file declares boundary content of its own instead of re-exporting the implementation.
  Do not add per-module interface façades that merely `abbrev`/delegate to the
  implementation; they double the file count without adding a checked boundary.
- File-size cap exceptions (owner-confirmed): `SuperNeo/Primitives/Ring.lean`, `SuperNeo/EmbeddingTheory/Thm3Core.lean`, and `SuperNeo/Primitives/Decomp.lean` exceed the repo's 1,500-line cap as single cohesive proof developments over file-spanning `private` lemma substrates. Do not split them mechanically (that would force de-privatizing their helpers); revisit a split only when one of them is reopened for substantive work.
- Lean build discipline:
  - During iteration, build only the target module(s) you changed and their dependencies, not the whole package.
  - Prefer narrow commands such as `lake build SuperNeo.<Name>` while working.
  - If several Lean modules changed, build the narrowest affected theorem-facing targets that cover those edits.
  - Only once the Lean work is complete, run a full `lake build` to catch package-wide breakage before finishing.
- Closure standard (mandatory): **Paper-faithful proof-complete**.
  - A module is only considered complete when the exact mathematical construction/claim from
    `./SuperNeo.pdf.md` is proved in Lean at quantified theorem level.
  - Regression checks (`lake exe check`, generated vectors, booleans) are required but are never
    sufficient evidence for completion.
  - Interface-level or assumption-level closure (`Done (Boundary)`) is intermediate only.
  - Do not claim proof completion by redefining theorem-facing surfaces to be definitionally equal
    to the target expression while leaving the executable/paper construction unproved; prove the
    bridge theorem explicitly.
  - Any trusted assumption/axiom that remains must be explicit, minimal, and accompanied by a
    concrete closure plan in the module spec and README.
- Project-local skill for this workflow:
  - Path: `../../.codex/skills/superneo-lean-interface-spec/SKILL.md`
  - Purpose: create/update per-module specs (`specs/<Name>.spec.md`) and, only at
    consumed boundaries, `SuperNeo/<Section>/<Name>Interface.lean`.
  - Use when: standardizing specs or auditing assumptions/consumers against `./SuperNeo.pdf.md`.
- Keep the remaining interface files colocated with implementations (Objective-C style), not in a separate top-level folder.
- `*.spec.md` is the external/human-facing specification.
- Specs must be **stateless**: they describe the timeless mathematical target (what the module must achieve), never the current implementation progress. Do not use language like "currently proved", "not yet implemented", "scaffold", "pending", or "in progress" in specs. A spec should read identically whether the module is 0% or 100% complete.
- Avoid naming Lean boundary files as `*Spec.lean` or `*Contract.lean` to prevent confusion with prose specs and crypto terminology.
- Where an interface file exists, keep it thin: theorem/definition shapes and boundary assumptions only, no implementation details.
