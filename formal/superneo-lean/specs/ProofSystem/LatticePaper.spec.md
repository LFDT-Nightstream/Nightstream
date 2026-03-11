# ProofSystem/LatticePaper

## Purpose

- **What it is**: The concrete Goldilocks Appendix B.2 lattice-parameter family and the active `paperCarrier` specialization of the MSIS-to-Ajtai reduction package.
- **Key property**: The active paper route reconstructs the Ajtai reduction package directly from the theorem-level MSIS hardness assumption at the Goldilocks Appendix B.2 parameters.
- **Protocol role**: Supplies the concrete Section 6.2 route consumed by the final protocol theorem on the active Goldilocks path.

## Target Formulas

- `goldilocksPaperAjtaiParams messageLength` fixes `κ = 18`, `B = 2^14`, `T = 216`, with message length explicit.
- `0 < messageLength → (goldilocksPaperAjtaiParams messageLength).SideConditions`.
- `MSISHardnessBoundary (goldilocksPaperAjtaiParams messageLength) → MSISToAjtaiReductions (goldilocksPaperAjtaiParams messageLength)`.
- `MSISHardnessAssumption (goldilocksPaperAjtaiParams messageLength) → MSISToAjtaiReductions (goldilocksPaperAjtaiParams messageLength)`.

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Appendix B.2 concrete parameter choices.
- Theorem 2 (Ajtai properties), lines 319-321.
- Definition 17 (strong sampling sets), lines 747-749.
- Definition 18 (Ajtai commitment), lines 753-756.

## Module Mapping

| Paper concept | Lean symbol | Role |
|---|---|---|
| Appendix B.2 lattice family | `goldilocksPaperAjtaiParams` | Theorem-Target |
| Concrete positivity/size side conditions | `goldilocksPaperAjtaiParams_sideConditions` | Theorem-Target |
| Active Goldilocks reduction package from MSIS boundary | `MSISToAjtaiReductions.ofGoldilocksPaperCarrierAndMSISBoundary` | Theorem-Target |
| Active Goldilocks reduction package from theorem-level hardness | `MSISToAjtaiReductions.ofGoldilocksPaperCarrierAndMSISHardness` | Theorem-Target |

## Contract Surface

| Group | Symbol | Guarantee | Role |
|---|---|---|---|
| Parameters | `goldilocksPaperAjtaiParams` | Fixes the Appendix B.2 constants and leaves message length explicit | Theorem-Target |
| Arithmetic | `goldilocksPaperAjtaiParams_kappa`, `goldilocksPaperAjtaiParams_msgLen`, `goldilocksPaperAjtaiParams_bindingNormBound`, `goldilocksPaperAjtaiParams_relaxedExpansion` | Concrete field equality facts for the parameter family | Theorem-Target |
| Side conditions | `goldilocksPaperAjtaiParams_relaxedExpansion_pos`, `goldilocksPaperAjtaiParams_three_d_le`, `goldilocksPaperAjtaiParams_sideConditions` | Concrete positivity / expansion hypotheses needed by Theorem 2 | Theorem-Target |
| Reductions | `MSISToAjtaiReductions.ofGoldilocksPaperCarrierAndMSISBoundary` | Reconstructs the concrete Goldilocks `paperCarrier` Ajtai package from an MSIS boundary | Theorem-Target |
| Reductions | `MSISToAjtaiReductions.ofGoldilocksPaperCarrierAndMSISHardness` | Reconstructs the concrete Goldilocks `paperCarrier` Ajtai package directly from theorem-level MSIS hardness | Theorem-Target |

## Assumption Ledger

- `MSISHardnessBoundary (goldilocksPaperAjtaiParams messageLength)` or `MSISHardnessAssumption (goldilocksPaperAjtaiParams messageLength)` is the intended Section 6.2 theorem-level security input.
- No extra carrier-side law bundle is exposed on this concrete Goldilocks route.

## Dependency and Consumer Map

- **Dependencies**: `ProofSystem/Lattice.lean`, `ProofSystem/LatticeReductionsDerived.lean`, `Parameters.lean`.
- **Consumers**:
  - `ProtocolTheorem.lean`
  - `ProofSystem/Protocol.lean`

## Quality Expectations

- The concrete parameter family stays Appendix-B.2-faithful.
- The concrete Goldilocks constructors remain thin wrappers over the proved reduction layer.

## Acceptance Criteria

- `lake build` succeeds.
- No `sorry`.

## Out of Scope

- Canonical in-repo choice of final theorem message length.
- Carrier-parametric Section 6.2 generalization beyond the active Goldilocks route.
