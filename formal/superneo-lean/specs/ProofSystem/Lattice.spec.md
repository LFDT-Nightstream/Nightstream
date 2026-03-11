# Lattice.spec.md

## Purpose

- **What it is**: Ajtai commitment parameters, opening/collision structures, MSIS challenge/solution surfaces, and hardness/advantage boundaries for the Module-SIS and Ajtai commitment schemes.
- **Key property**: Theorem 2: Ajtai is \(B\)-binding (resp. \((B,\mathcal{C})\)-relaxed binding) assuming \(\text{MSIS}_{m,2B}^{\infty,\kappa,q}\) (resp. \(\text{MSIS}_{m,4TB}^{\infty,\kappa,q}\)) is hard.
- **Protocol role**: Provides the lattice-security surfaces consumed by `LatticeReductions` for MSIS-to-Ajtai binding reductions; protocol theorems assume MSIS hardness and derive Ajtai binding.

## Target Formulas

- \(\text{MSIS}_{m,B}^{\infty,\kappa,q}\): find \(z \neq 0\) with \(Mz = 0 \pmod q\), \(\|z\|_\infty < B\) ↔ `MSISBreakEvent`, `MSISSolution`.
- \(\text{Commit}(\text{pp}, z) = Mz\) ↔ `opensTo`: \(M \cdot z = c\) with \(\|z\|_\infty < B\).
- Relaxed: \(\Delta \cdot c = Mz\) ↔ `opensToRelaxed`.
- \(B\)-binding collision: \(z_1 \neq z_2\), \(Mz_1 = Mz_2 = c\), \(\|z_i\|_\infty < B\) ↔ `BindingCollision`.
- \((B,\mathcal{C})\)-relaxed: \(\Delta_1 z_2 \neq \Delta_2 z_1\), \(\Delta_i c = Mz_i\), \(\Delta_i \in \mathcal{C}-\mathcal{C}\) ↔ `RelaxedBindingCollision params C`.
- \(\text{Adv}_{\text{MSIS}} \le \varepsilon(n)\), \(\text{IsNegligible}(\varepsilon)\) ↔ `MSISAdvantageBound`, `MSISHardnessAssumption`.

## Paper Anchors

- **Source**: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 16 (MSIS), lines 743–744.
- Definition 18 (Ajtai commitment), lines 753–756.
- Theorem 2 (Properties), lines 319–321.
- Definition 4 (\(B\)-binding, \((B,\mathcal{C})\)-relaxed binding), lines 305–315.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---------------|-------------|--------|
| Ajtai params | `AjtaiParams` | Definitional |
| Commitment / opening | `Commitment`, `Opening` | Definitional |
| opensTo | `opensTo` | Definitional |
| opensToRelaxed | `opensToRelaxed` | Definitional |
| Binding collision | `BindingCollision` | Definitional |
| Relaxed binding collision | `RelaxedBindingCollision params C` | Definitional |
| MSIS challenge/solution | `MSISChallenge`, `MSISSolution` | Definitional |
| MSIS break event | `MSISBreakEvent` | Definitional |
| MSIS hardness | `MSISHardnessAssumption`, `MSISHardnessBoundary` | Definitional |
| Ajtai binding assumptions | `AjtaiBindingAssumption params`, `AjtaiRelaxedBindingAssumption params C` | Definitional |
| subRq/subVec/matVecMul/smulVec | `subRq`, `subVec`, `matVecMul`, `smulVec` | Definitional |
| subRq/subVec cancellation | `subRq_self`, `subVec_self` | Theorem-Target |
| Linearity/norm boundary package | moved to `LatticeReductions` | N/A in this module |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|-------|--------|-----------|--------|
| Parameters | `AjtaiParams` | \(\kappa, m, B, T\) | Definitional |
| Commitment relations | `opensTo`, `opensToRelaxed` | \(Mz = c\), \(Mz = \Delta \cdot c\) | Definitional |
| Collisions | `BindingCollision`, `RelaxedBindingCollision params C` | Paper-faithful distinctness and carrier-membership threading | Definitional |
| MSIS | `MSISBreakEvent`, `MSISSolution`, `MSISChallenge` | Homogeneous MSIS, \(Mz = 0\) | Definitional |
| Hardness | `MSISHardnessAssumption`, `MSISHardnessBoundary` | \(\exists \varepsilon, \text{IsNegligible}(\varepsilon) \wedge \text{Adv} \le \varepsilon\) | Definitional |
| Ajtai assumptions | `AjtaiBindingAssumption`, `AjtaiRelaxedBindingAssumption params C` | \(\exists \varepsilon, \text{IsNegligible}(\varepsilon) \wedge \text{Adv} \le \varepsilon\) | Definitional |
| Vector ops | `subRq_self`, `subVec_self` | \(\text{subRq}\,x\,x = 0\), \(\text{subVec}\,n\,v\,v = \text{zeroVec}\,n\) | Theorem-Target |

## Proof Obligations and Closure Plan

- **Theorem-Targets**: `subRq`, `subRq_self`, `subVec_self`, size lemmas, `matrixFlatLen_le_payloadLen`, `commitmentLen_le_payloadLen`, `msisNormBound_pos`, `ppMatrixFlat_size_of_wf`, `valueVec_size_of_wf`, `NormSound_mono`, `smulVec_size`, `matVecMul_size`.
- Extractor linearity/norm boundary obligations live in `LatticeReductions`.

## Assumption Ledger

No open boundary assumptions in this module. Reduction-layer assumptions are tracked in `specs/ProofSystem/LatticeReductions.spec.md`.

## Dependency and Consumer Map

- **Dependencies**: `SuperNeo.Ring`, `SuperNeo.Norm`, `SuperNeo.ProofSystem.Negligible`, `SuperNeo.ProofSystem.Security`.
- **Consumers**:
  - `SuperNeo.ProofSystem.LatticeReductions`: uses `BindingCollision`, `RelaxedBindingCollision`, `MSISBreakEvent`, `MSISHardnessAssumption`, `AjtaiBindingAssumption`, `AjtaiRelaxedBindingAssumption`, `subVec`, `matVecMul`, `smulVec`.

## Implementation Plan

- Keep this module definition-complete and assumption-minimal.
- `RelaxedBindingCollision` carries explicit `inDiff1 : samplingDiffSet C delta1` and `inDiff2 : samplingDiffSet C delta2` witnesses per Definition 4 (line 315). Norm bounds on deltas are derivable from C-C membership, not primary data.
- All relaxed-binding types (`AjtaiRelaxedBindingAssumption`, `AjtaiRelaxedBindingGame`, `AjtaiRelaxedBindingAdvantage`, `AjtaiRelaxedBindingAdvantageBound`, `AjtaiRelaxedBindingBoundary`) are parameterized by `C : SamplingCarrier`.
- Route extractor/ring-linearity assumptions through `LatticeReductions` only.

## Quality Expectations

- Spec states Theorem 2 and Definitions 16, 18 with line ranges.
- Interface exposes core structures and theorem-facing game/boundary symbols.

## Acceptance Criteria

- `lake build` succeeds.
- No open module-local assumptions are listed.
- Paper anchors include Definition 16, 18, Theorem 2 and line ranges.

## Out of Scope

- Concrete probabilistic game semantics (challenger/adversary sampling); abstract `breakAt` shells suffice for theorem-facing layer.
- Tight computational-soundness accounting across concrete samplers; theorem-facing bounds are modeled via `ProbModel` + negligible error terms.
