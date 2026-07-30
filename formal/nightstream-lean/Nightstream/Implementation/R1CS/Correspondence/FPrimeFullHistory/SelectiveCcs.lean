import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.RelationProfile
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.FixedPointShape
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningSplitNc
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningSplitNc.SelectedPhysicalRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningSplitNc.SelectedVerifierRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition

/-!
Curated arithmetic surface for the 13-port selective CCS relation.

| Child | Mathematical ownership | Emits constraints? | Assurance tier |
|---|---|---|---|
| `Polynomial` | exact ports, sparse syntax, named components, and omission witnesses | no | model-level |
| `Padding` | typed 270-carrier padding semantics, specialization, and necessity | no | model-level |
| `RelationProfile` | exact 13-matrix semantic shape, row-domain policy, and three-row mismatch | no | model-level |
| `FixedPointShape` | untrusted final-header equality, exact polynomial/public/alignment checks, and semantic profile construction | no | model-level schema/refinement |
| `CanonicalOpeningSplitNc` | typed opening-column ownership and Split-NC `b = 2` composition | no | model-level refinement |
| `CanonicalOpeningSplitNc.ArtifactRefinement` | exact generated 21 rows over the Split-NC-covered 41-digit/20-borrow layout | no | artifact-checked refinement |
| `CanonicalOpeningSplitNc.SelectedPhysicalRefinement` | Lean-owned 61-column layout and exact 21-row relocation | yes | model-level refinement |
| `CanonicalOpeningSplitNc.SelectedVerifierRefinement` | selected verifier rows imply canonicality or named security events | no | security-reduced |
| `Artifact` | compact wire schema, decoder, and matrix interpreter | no | model-level artifact schema |
| `SelectorComposition` | minimal sum-to-one/complement gating, semantic exactness, and omission witnesses | no | model-level |

The exact 13-port/66-term sparse polynomial and its typed public-padding
specialization are model-level closed here. The artifact child only defines a
validated compact-data interpreter. Production matrix rows and all remaining
row families stay fail-closed until generated-data equality and exact matrix
refinement are proved. The selector child is generic over the physical arm
count; the production mapping from base/bootstrap-recursive/steady-recursive
arms to the two paper constructors remains a separate branch-refinement
obligation.
-/
