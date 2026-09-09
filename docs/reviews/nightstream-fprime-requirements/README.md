# Nightstream Stage 1 requirement hierarchy

This map starts from the local SuperNeo v1.1 and HyperNova sections selected by the owner goal. It then adds the Nightstream profile and the implementation links needed to realize those paper requirements in the current package.

- [Full nested list](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-requirements/MAP.md)
- [Open and conditional proof connections](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-requirements/OPEN_ITEMS.md)
- [Structured hierarchy](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-requirements/requirements.json)
- [Public website](https://nightstream-requirements.nicarq.chatgpt.site)
- [Standalone HTML page](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-requirements/site/index.html)

Reviewed code: `4fc02857c3aa4207f3290739cc62d794ba86f5f9`. The paper references identify the local document versions, including their amendments; this is not a claim that all wording appears in an unmodified external publication.

The website and standalone HTML include the [completed foundation update](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-requirements/FOUNDATION_CLOSURE.md), validated from working-tree changes on 2026-09-08. The original Markdown map and branch JSON files remain the review snapshot at the commit above.

## How to read the hierarchy

Each leaf is a specific operation, equation, range check, serialization rule, or proof connection. An indexed family is one requirement with its dimensions. For example, all 16 PiDEC commitment recompositions share the same requirement; they do not require 16 separate implementations.

| Axis | Meaning |
|---|---|
| Proof | The local Lean fact for this leaf. A characterization theorem, a conditional theorem, and a complete selected-assignment theorem have different scopes. |
| Link | Connection to the leaf's local consuming contract. A primitive may be locally connected while the complete phase or F′ result remains open. |
| Rust | Implementation or execution evidence. Code existence, a scoped test, a historical diagnostic, and approved phase closure are different results. |

`not_reviewed` means the map did not verify that coverage. It does not mean the operation is absent. `assumption` means the result is an explicit external premise, not a theorem silently treated as proved. `not_required` is scoped to that item; for example, an implementation test does not need a separate mathematical theorem of its own.

The map separates paper requirements, Nightstream profile choices, implementation requirements, external assumptions, and material outside the selected stage. The final group covers optional or inapplicable paper material so that it is not mistaken for missing Stage 1 work.

The dependency links are explanatory cross-references between requirements. They are not an automatically extracted Lean proof-dependency graph or an approved conformance manifest.

## Main groups

| ID | Group |
|---|---|
| F | Fields, rings, embeddings, multilinear algebra, norms, strong sets, commitments, setup, and profile conditions |
| T | Poseidon2, sponge/duplex operations, and canonical field words |
| C | PiCCS inputs, prover work, four polynomial families, sumcheck, terminal checks, output, and connections |
| R | PiRLC sampling, common inputs, ring combinations, bounds, output, and connections |
| D | PiDEC signed decomposition, public bound, commitments, evaluation recomposition, children, and connections |
| N | NIFS composition, non-interactivity, extraction, and explicit security boundaries |
| H | HyperNova compatibility, state, application, base/recursive execution, terminal verification, fixed point, and history composition |
| L | Circuit contracts, low-norm encoding, primitive rows, source lowering, matrix preservation, and export |
| P | Current artifacts, Rust execution, verifier authority, conformance evidence, and production acceptance |
| O | Paper constructions outside the selected Stage 1 |

## Current boundaries that matter

The current complete step theorem covers a canonical constructed assignment. The full accepted-assignment proof still needs exact sampler, parent, child/output, and selected-context connections. These are separate leaves from the proved phase equations.

The map also identifies lower-level qualifications that can be hidden by large group labels. Examples include the quadratic-extension field qualification, the selected strong-set cardinality connection, the external low-norm invertibility premise, and the admissible natural counter successor in the outer recursion argument. Their exact status and scope are in the leaf records. They are not reproduced attacks and do not automatically add work to the existing PiCCS conformance goal.

The local HyperNova paper text retains a conditional fixed-constant-step result and withdraws its original equal-half-padding CCS instantiation. Nightstream has separate concrete fixed-point and domain proofs. Both facts are recorded; neither replaces the missing full acceptance/security composition.

Paper reference parameters such as `k_rho = 14` are not production defaults. The selected Nightstream profile remains `b = 2`, `k_rho = 16`, `B = 65536`, rank 22, ring degree 54, one fresh input, 16 running inputs, 17 PiRLC sources, 16 PiDEC children, and 14 matrices. The internal 41-trit field encoding is a separate compiler choice.

## Review and evidence scope

The source work was divided into foundation, folding, HyperNova, and implementation branches. Each branch maps paper statements to actual definitions/theorems and code owners. No production source was changed and no conformance status was approved while making this hierarchy.

The earlier same-source review ran the Lean library and axiom gates and scoped native tests. Its records remain in [goal-split validation](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-goal-splits/VALIDATION.md) and [formal review validation](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/VALIDATION.md). No new protocol tests or Lean proofs were run for this map.

The current package payload and complete evidence archive are absent here. Historical diagnostic records are therefore marked as records rather than fresh approved conformance. A named legacy transform test points into the frozen project and can write there; it was not run or counted as current F′ evidence.

This hierarchy does not replace the existing goal, change its phase order, expand its permitted security assumptions, or authorize a backend. The compatible sequence remains pilot/PiCCS closure, then the same selected PiRLC/PiDEC/F′/production path. Shared primitive work stays on that path.

## Source coverage notes

- [Foundations](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-requirements/FOUNDATIONS_NOTES.md)
- [Folding](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-requirements/FOLDING_NOTES.md)
- [HyperNova](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-requirements/HYPERNOVA_NOTES.md)

The implementation branch is derived from architecture §§3–4 and §§7–13 and the reviewed compiler/export/runtime sources. `implementation.json` records its exact code references.

`build_map.py` assembles the reviewed JSON branches, validates IDs/parents/source locations, and produces the readable map and open-item view. It does not infer proof status from file names, run a prover, or grant completion. Counts describe this inventory, not a percentage complete.

The final map contains 365 indexed requirements and 12 scope exclusions. Validation checked all 1,596 source references and 2,000 local Markdown links for file and line validity. JavaScript syntax passed. The browser preview displayed all 453 group/leaf controls; nested expansion and dependency navigation worked. The expanded view had no horizontal overflow at 320px and 736px viewport widths, and the browser reported no warnings or errors. These checks validate the map and its display, not the protocol proofs.
