import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding.Semantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding.Refinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding.ArtifactRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Padding.Necessity

/-!
Stable family surface for selective-CCS public-padding obligations.

Owns: the arithmetic-owner facade for normalized zero pins, exact polynomial
specialization, coefficient-based artifact classification, and obligation
necessity.

Does not own: a concrete generated carrier artifact, compiler layout, row
scheduling, full relation satisfaction, or row removal.

Emits constraints: no.

| Child | Mathematical ownership | Excluded boundary |
|---|---|---|
| `Semantics` | normalized typed zero pin and canonical completeness | sparse polynomial and physical rows |
| `Refinement` | exact 27-term polynomial specialization to `-(z0*zpad)` | concrete row coefficients |
| `ArtifactRefinement` | decoded coefficient shape and exact residual | generated values and multiplicity |
| `Necessity` | one countermodel for each omitted raw-input check | complete verifier minimality |

This parent is the arithmetic-owner axis. Compiler layout and row scheduling
remain separate and map here through their parent ownership tables.
-/
