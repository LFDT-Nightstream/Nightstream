# Types.spec.md

## Purpose

- **What it is**: Generic proof-system type carriers `Context`, `Claim`, and `Witness` that parameterize public parameters, instance space, and witness space for interactive reductions.
- **Key property**: These structures form the ambient type-level carriers for relations \(\mathcal{R}_1, \mathcal{R}_2\) over \((\text{pp}, \mathbf{s}, u, w)\) tuples in Definitions 9–10.
- **Protocol role**: `ProofSystem` facade and protocol theorem modules use these types to state weak/strong interactive reduction interfaces and security compositions.

## Target Formulas

- \(\mathcal{U}_1\) (instance space of \(\mathcal{R}_1\)) ↔ `Claim`-indexed structures; \(\mathcal{U}_2\) ↔ `Claim`-indexed structures.
- \((\text{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}'_1\) ↔ `Context` × structure × `Claim` × `Witness` with relation predicate.
- Security parameter \(\lambda\) ↔ `Context.securityParam : Nat`.

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 9 (Weak Interactive Reductions), lines 404–416.
- Definition 10 (Strong Interactive Reductions), lines 418–436.
- Infrastructure for relations over \((\text{pp}, \mathbf{s}, u, w)\); no direct paper definition.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Public parameters / security param | `Context`, `Context.securityParam` | Definitional |
| Instance / claim | `Claim`, `Claim.id` | Definitional |
| Witness | `Witness`, `Witness.id` | Definitional |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Type carriers | `Context` | Structure with `securityParam : Nat` | Definitional |
| | `Claim` | Structure with `id : Nat` | Definitional |
| | `Witness` | Structure with `id : Nat` | Definitional |

## Proof Obligations and Closure Plan

All obligations closed. Types are definitional; no theorem-level proof obligations.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: None (no imports).
- **Consumers**:
  - `SuperNeo.ProofSystem.Security`: uses `Context`-like structure for probability models indexed by security parameter.
  - `SuperNeo.ProofSystem.Lattice`, `LatticeReductions`: depend on `ProofSystem` facade types for theorem-facing surfaces.
  - Protocol theorem modules: import `Types` for relation instance/witness typing.

## Implementation Plan

- Keep structures minimal; extend with relation-specific fields as protocol theorems require.
- No proof work; types remain definitional.

## Quality Expectations

- Structures stay lean; avoid over-specifying before protocol relations are formalized.
- Interface docstring references spec and paper anchors.

## Acceptance Criteria

- `lake build` succeeds.
- Spec contains explicit paper anchors with line ranges.
- Consumer map entries state what each consumer uses or imports.

## Out of Scope

- Concrete relation predicates; protocol-specific relation definitions.
- Probability or advantage semantics; those live in `Security` and `Negligible`.
