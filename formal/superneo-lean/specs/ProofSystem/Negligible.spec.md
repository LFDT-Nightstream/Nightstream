# Negligible.spec.md

## Purpose

- **What it is**: Error functions \(\varepsilon : \mathbb{N} \to \mathbb{Q}\) and the predicate \(\text{IsNegligible}(\varepsilon)\) in asymptotic inverse-polynomial form.
- **Key property**: \(\text{IsNegligible}(f) \leftrightarrow \forall c, \exists N, \forall n \ge N, f(n) \le \frac{1}{(n+1)^c}\).
- **Protocol role**: Section 6 security reductions bound adversary success by \(\epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda)\); `IsNegligible` formalizes the negligible error bound \(\varepsilon\) in Definitions 9–10.

## Target Formulas

- \(\text{negl}(\lambda)\) (paper) ↔ \(\exists \varepsilon : \text{ErrorFn}, \text{IsNegligible}(\varepsilon)\).
- \(\epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda)\) ↔ extractor success probability bounded below by advantage minus negligible term.
- \(\text{IsNegligible}(f) \leftrightarrow \forall _c, \exists N, \forall n \ge N, f(n) \le \frac{1}{(n+1)^c}\).

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 9 (Weak Interactive Reductions), lines 404–416 (negligible error bound \(\varepsilon\)).
- Definition 10 (Strong Interactive Reductions), lines 418–436.
- Negligible function concept used throughout Section 6 security reductions.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Error function | `ErrorFn` (= `Nat → Rat`) | Definitional |
| Inverse-poly bound | `invPolyBound` | Definitional |
| Negligible predicate | `IsNegligible` | Definitional |
| Zero is negligible | `isNegligible_zero` | Theorem-Target |
| Identically zero ⇒ negligible | `isNegligible_of_zero` | Theorem-Target |
| Iff characterization | `isNegligible_iff` | Theorem-Target |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Error functions | `ErrorFn` | `Nat → Rat` | Definitional |
| Negligible predicate | `IsNegligible` | \(\forall c, \exists N, \forall n \ge N, f(n) \le \frac{1}{(n+1)^c}\) | Definitional |
| Basic lemmas | `isNegligible_iff` | \(\text{IsNegligible}(f) \leftrightarrow \forall c, \exists N, \forall n \ge N, f(n) \le \frac{1}{(n+1)^c}\) | Theorem-Target |
| | `isNegligible_zero` | \(\text{IsNegligible}(\lambda n. 0)\) | Theorem-Target |
| | `isNegligible_of_zero` | \((\forall n, f(n) = 0) \to \text{IsNegligible}(f)\) | Theorem-Target |

## Proof Obligations and Closure Plan

All obligations closed. Negligible predicate and basic lemmas are proved.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: None (no imports).
- **Consumers**:
  - `SuperNeo.ProofSystem.Security`: uses `IsNegligible` and `ErrorFn` for `ErrorModel` negligible components.
  - `SuperNeo.ProofSystem.Lattice`: uses `IsNegligible` for `MSISHardnessAssumption`, `MSISHardnessBoundary`, `AjtaiBindingBoundary`, `AjtaiRelaxedBindingBoundary`.
  - `SuperNeo.ProofSystem.LatticeReductions`: depends on `IsNegligible` for advantage-bound implications.

## Implementation Plan

- Keep asymptotic inverse-polynomial negligible form as the default theorem-facing surface.
- Use this directly in `Security`, `Lattice`, and `ProtocolTheorem` error-accounting chains.

## Quality Expectations

- Spec documents asymptotic negligible behavior and the concrete bound shape.
- Interface exposes `ErrorFn`, `IsNegligible`, and key lemmas.

## Acceptance Criteria

- `lake build` succeeds.
- All theorems in module are proved (no axioms).
- Paper anchors include Definition 9 and line ranges.

## Out of Scope

- Tight asymptotic optimization (Big-O/Theta style APIs); current surface is theorem-usable and minimal.
- Full game-level probabilistic semantics; those live in reduction/protocol modules.
