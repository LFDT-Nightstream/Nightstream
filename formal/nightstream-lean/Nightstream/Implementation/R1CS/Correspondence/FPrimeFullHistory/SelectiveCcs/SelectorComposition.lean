import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Semantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Complement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.PolynomialGating
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.RowPointGating
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Necessity
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.ArtifactRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.SelectorCoverageArtifact

/-!
Curated selector-composition surface for the executable selective CCS
compiler and the fixed two-arm complement compiler.

| Child | Mathematical ownership | Emits constraints? | Assurance tier |
|---|---|---|---|
| `Semantics` | indexed sum-to-one gating, soundness, completeness, branch-refinement interface | no | model-level |
| `Complement` | isolated fixed two-arm `s` / `1-s` selector convention | no | model-level |
| `PolynomialGating` | one selector-factor theorem for every arm-local row family | no | model-level |
| `RowPointGating` | physical selector matrix images imply factorization of the interpreted row residual | no | model-level |
| `Necessity` | omission countermodels and canonicalization classification | no | model-level |
| `ArtifactRefinement` | exact three selector rows, total row, and representative gated row | no | artifact-checked fixture |
| `SelectorCoverageArtifact` | compact all-row ledger/gate-class reconciliation and exact polynomial syntax for one fixture | no | artifact-checked fixture |

This component deliberately does not import paper F' semantics into its
algebra. A later refinement supplies exact residual-to-branch theorems for the
caller-owned base, bootstrap-recursive, and steady-recursive arm order.
-/
