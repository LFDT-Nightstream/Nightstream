import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Parameters
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Necessity

/-!
Independent Phi81 SplitNc semantics for SuperNeo `Pi_CCS`.

This component keeps the paper's square one-joint model intact and states the
production-motivated two-domain semantic obligations separately. It derives
fresh completion, carried assignment authority, coefficient matrices, FE
truth, and NC truth from explicit mathematical sources. It does not accept an
existing verifier or circuit as the definition of correctness.

Emits constraints: no.

| Protocol | Phase | Constraint family | Current result |
|---|---|---|---|
| `Pi_CCS` | parameters | row / carrier / candidate NC domains | semantic and arithmetization shapes are separated |
| `Pi_CCS` | source ownership | matrices / fresh / running | one connected source family; derived completion and coefficient images |
| `Pi_CCS` | FE semantics | CCS / carried evaluation | exact uncompressed residual equivalence |
| `Pi_CCS` | NC semantics | full-carrier strict norm | exact uncompressed diagonal-cubic equivalence |
| `Pi_CCS` | domain necessity | completed-carrier coverage | logical-width cube formally shown insufficient |
| production refinement | transcript / SumCheck / Rust / R1CS | open | no constraint-removal permission |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- Boundaries that remain open after the independent semantic layer. -/
inductive OpenBoundary where
  | productionRowDomain
  | productionCarrierDecoding
  | flatNcDomainCoverage
  | fePolynomialRefinement
  | feSumCheckSoundness
  | ncPolynomialRefinement
  | ncMixingSoundness
  | ncSumCheckSoundness
  | transcriptRefinement
  | outputAuthority
  | rustRefinement
  | r1csRefinement
deriving Repr, DecidableEq

/-- Diagnostic census only. Editing this list cannot discharge a boundary. -/
def openBoundaries : List OpenBoundary :=
  [.productionRowDomain, .productionCarrierDecoding, .flatNcDomainCoverage,
    .fePolynomialRefinement, .feSumCheckSoundness, .ncPolynomialRefinement,
    .ncMixingSoundness, .ncSumCheckSoundness, .transcriptRefinement,
    .outputAuthority, .rustRefinement, .r1csRefinement]

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc
