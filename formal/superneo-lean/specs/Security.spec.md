# Security.spec.md

## Purpose

- **What it is**: Abstract probability model \(\Pr : \text{Prop} \to \mathbb{Q}\) with theorem-usable laws (`Pr False = 0`, monotonicity, union bound), and error accounting model decomposing total error into sum-check, MSIS, Schwartz–Zippel, binding, and relaxed-binding components.
- **Key property**: \(\varepsilon_{\text{total}} = \varepsilon_{\text{sumcheck}} + \varepsilon_{\text{MSIS}} + \varepsilon_{\text{SZ}} + \varepsilon_{\text{bind}} + \varepsilon_{\text{rk}}\) with each component negligible.
- **Protocol role**: Definitions 9–10 and Theorem 6 use probability bounds and negligible error terms; `ProbModel` and `ErrorModel` provide the typed surfaces for protocol theorem statements.

## Target Formulas

- \(\Pr[P] \in [0, 1]\) ↔ `ProbModel.prNonneg`, `ProbModel.prLeOne`.
- \(\Pr[\bot] = 0\) ↔ `ProbModel.prFalse`.
- \(P \Rightarrow Q \Rightarrow \Pr[P] \le \Pr[Q]\) ↔ `ProbModel.prMonotone`.
- \(\Pr[P \lor Q] \le \Pr[P] + \Pr[Q]\) ↔ `ProbModel.prUnionLeAdd`.
- \(\varepsilon_{\text{total}}(n) = \sum_i \varepsilon_i(n)\) ↔ `ErrorModel.epsTotal_decomp`.
- \(\text{IsNegligible}(\varepsilon_i)\) for each component ↔ `ErrorModel.negligibleSumcheck`, etc.

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 9 (Weak Interactive Reductions), lines 404–416.
- Definition 10 (Strong Interactive Reductions), lines 418–436.
- Security model structures underlying probability and error accounting in Section 6.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Probability measure | `ProbModel`, `ProbModel.Pr` | Definitional |
| Error decomposition | `ErrorModel` | Definitional |
| Zero-error model | `zeroErrorModel` | Definitional |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Probability | `ProbModel` | \(\Pr : \text{Prop} \to \mathbb{Q}\), \(0 \le \Pr(P) \le 1\), `Pr False = 0`, monotonicity, union bound | Definitional |
| Error accounting | `ErrorModel` | Decomposition + all components negligible | Definitional |
| Constructor | `ErrorModel.ofComponents` | Canonical constructor deriving `epsTotal` and `negligibleTotal` from the five component error surfaces | Definitional |
| Default | `zeroErrorModel` | Canonical zero-error model | Definitional |

## Proof Obligations and Closure Plan

All obligations closed. Structures are definitional; `ErrorModel.ofComponents` derives the consolidated error model and total negligibility from the five component boundaries, and `zeroErrorModel` is then an instance of that constructor.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- **Dependencies**: `SuperNeo.ProofSystem.Negligible` (uses `ErrorFn`, `IsNegligible`).
- **Consumers**:
  - `SuperNeo.ProofSystem.Lattice`: uses `ProbModel` for `MSISAdvantage`, `AjtaiBindingAdvantage`, `AjtaiRelaxedBindingAdvantage`; uses `ErrorModel` shape for protocol error composition.
  - `SuperNeo.ProofSystem.LatticeReductions`: uses `ProbModel` (e.g. `truthProb`) in reduction theorems.
  - Protocol theorem modules: depend on `ErrorModel` for total error bounds.

## Implementation Plan

- Keep `ProbModel` abstract; concrete instances (e.g. `truthProb`) live in reduction modules.
- `ErrorModel` serves as the protocol-facing error decomposition surface.

## Quality Expectations

- Spec states the decomposition equation and negligible requirements.
- Interface exposes `ProbModel`, `ErrorModel`, `zeroErrorModel`.

## Acceptance Criteria

- `lake build` succeeds.
- `zeroErrorModel` is well-formed with all negligible proofs.
- Paper anchors include Definitions 9–10 and line ranges.

## Out of Scope

- Concrete probability semantics (e.g. sampled games); abstract `Pr` suffices for theorem-facing layer.
- Full adversarial game definitions; those are in `Lattice` and reduction modules.
