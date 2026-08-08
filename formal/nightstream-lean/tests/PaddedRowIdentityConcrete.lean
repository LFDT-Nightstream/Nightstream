import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityConcreteCompleteness
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityArtifact

/-!
Focused theorem-surface regressions for the concrete `PaddedRowIdentity`
NIFS. These checks freeze the selected dimensions, transcript profile, codec
widths, concrete algebra, honest completeness, and conditional soundness
entrypoints before the Rust implementation starts.
-/

set_option autoImplicit false

namespace tests.PaddedRowIdentityConcrete

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCodec
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteAlgebra
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteComposition
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteCompleteness
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteExtraction
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteNifs
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityPoseidon2
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentitySamplerSecurity

#check logicalSourceHolds_iff_sourceValid
#check logicalSource_exists_verifiedTransition
#check evaluations_eq_paper
#check extractionAlgebra
#check extractionStrongSetUnits
#check existsFiniteReductionThroughPiDec
#check concreteFullOracleSoundness
#check boundedSampler_refines
#check shortfall_requires_eleven_rejections
#check samplerShortfall_probability_le_132_bits
#check publicWireFields_injective_on_admissible
#check proofWireFields_injective
#check ProductionMatrixRefinement

example : rowVariables = 24 := rfl
example : logicalRows = 14944219 := rfl
example : assignmentColumns = 11437038 := rfl
example : shape.matrixCount = 14 := rfl
example : shape.sourceCount = 15 := rfl
example : shape.carriedEvaluationCount = 10584 := rfl
example : verifierRows = 18 := rfl
example : relationShape.publicWidth = 270 := rfl
example : profileHeader.length = 9 := profileHeader_length
example : publicClaimsCodec.width = 39846 := publicClaimsCodec_width
example : proofCodec.width = 57936 := proofCodec_width
example : publicEnvelopeTag = 1001 := rfl
example : proofEnvelopeTag = 1002 := rfl
example : codecVersion = 1 := rfl
example : publicInputTag = 40 := rfl
example : protocolVersion = 1 := rfl
example : completeSamplerShortfallBound <= samplerSecurityTarget :=
  completeSamplerShortfallBound_le_target

end tests.PaddedRowIdentityConcrete
