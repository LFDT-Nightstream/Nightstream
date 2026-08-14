import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CenteredDomainPackingArtifact

/-! Regression checks for the production centered-domain packing theorem. -/

namespace Tests.FPrimeFullHistorySelectiveCcsCenteredDomainPacking

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPacking
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CenteredDomainPackingArtifact

example : terms.length = 74 := term_count_exact

example : evaluate (centeredPairPoint 1 (1 : F) 0) = 0 :=
  (production_centeredPair_zero_iff 1 0).2 ⟨rfl, rfl⟩

example : evaluate (centeredPairPoint 1 (2 : F) 0) ≠ 0 := by
  intro packedZero
  have residuals := (production_centeredPair_zero_iff 2 0).1 packedZero
  exact (by decide : centeredUnitResidual (2 : F) ≠ 0) residuals.1

#check production_centeredPair_zero_iff
#check production_centeredTail_zero_iff
#check generated_pair_shape
#check generated_pair_zero_iff
#check generated_tail_shape
#check generated_tail_zero_iff

end Tests.FPrimeFullHistorySelectiveCcsCenteredDomainPacking
