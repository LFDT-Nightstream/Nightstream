# AGENTS.md

## Formal Lean Subproject
- Use this 3-layer layout for each formalized component:
  - Human spec: `specs/<Name>.spec.md`
  - Typed Lean interface: `SuperNeo/<Name>Interface.lean`
  - Lean implementation: `SuperNeo/<Name>.lean`
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
  - Purpose: create/update per-module Lean contract pairs
    (`SuperNeo/<Name>Interface.lean` + `specs/<Name>.spec.md`).
  - Use when: standardizing specs, adding missing interface/spec files, or
    auditing assumptions/consumers against `./SuperNeo.pdf.md`.
- Keep interface files colocated with implementations (Objective-C style), not in a separate top-level folder.
- `*.spec.md` is the external/human-facing specification; `*Interface.lean` is the machine-checked boundary.
- Specs must be **stateless**: they describe the timeless mathematical target (what the module must achieve), never the current implementation progress. Do not use language like "currently proved", "not yet implemented", "scaffold", "pending", or "in progress" in specs. A spec should read identically whether the module is 0% or 100% complete.
- Avoid naming Lean boundary files as `*Spec.lean` or `*Contract.lean` to prevent confusion with prose specs and crypto terminology.
- Interfaces should expose theorem/definition shapes and boundary assumptions clearly; implementations should satisfy or instantiate those interfaces.
- Prefer thin/stable interfaces and keep implementation details out of `*Interface.lean`.
