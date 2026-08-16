import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Semantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Complement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.GroupedCommon
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.GroupedCommonArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.ScheduledGrouped
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.ScheduledGroupedArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.ScheduledLinkedOverlay
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.ScheduledLinkedOverlayArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.PolynomialGating
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.RowPointGating
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.Necessity
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.ArtifactRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.SelectorCoverageArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.RadixFourSelectorCoverageArtifact

/-!
Curated selector-composition surface for the executable selective CCS
compiler and the fixed two-arm complement compiler.

| Child | Mathematical ownership | Emits constraints? | Assurance tier |
|---|---|---|---|
| `Semantics` | indexed sum-to-one gating, soundness, completeness, branch-refinement interface | no | model-level |
| `Complement` | isolated fixed two-arm `s` / `1-s` selector convention | no | model-level |
| `GroupedCommon` | share one common row family per lifecycle group with checked group-sum and phase-activation links | no | model-level |
| `GroupedCommonArtifact` | exact Rust fixture rows for stored group sums and phase activation | no | artifact-checked fixture |
| `ScheduledGrouped` | share lifecycle and phase-kind row families under an exact arm schedule | no | model-level |
| `ScheduledGroupedArtifact` | exact total, group, activation, and cursor rows for the Rust schedule fixture | no | artifact-checked fixture |
| `ScheduledLinkedOverlay` | one schedule-selected private overlay with checked selector, activation, and decoded-field links | no | model-level |
| `ScheduledLinkedOverlayArtifact` | exact overlay equality, activation, radix-decoded field link, and padding rows for the Rust fixture | no | artifact-checked fixture |
| `PolynomialGating` | one selector-factor theorem for every arm-local row family | no | model-level |
| `RowPointGating` | physical selector matrix images imply factorization of the interpreted row residual | no | model-level |
| `Necessity` | omission countermodels and canonicalization classification | no | model-level |
| `ArtifactRefinement` | exact three selector rows, total row, and representative gated row | no | artifact-checked fixture |
| `SelectorCoverageArtifact` | compact all-row ledger/gate-class reconciliation and exact polynomial syntax for one fixture | no | artifact-checked fixture |
| `RadixFourSelectorCoverageArtifact` | complete selector-port reconciliation and exact polynomial syntax for the production-width radix-four candidate | no | Rust-conformant for `FPRIME-R4-SELECTOR-COVERAGE` |

This component deliberately does not import paper F' semantics into its
algebra. A later refinement supplies exact residual-to-branch theorems for the
caller-owned base, bootstrap-recursive, and steady-recursive arm order.
-/
