import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement
import tests.Axioms.Support

/-! Fail-closed dependency gate for the generated opening-layout refinement. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement.artifactCoordinateNat_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement.artifactCoordinateNat_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement.splitNc_covers_generated_opening' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement.splitNc_covers_generated_opening

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement.splitNc_and_generatedLayoutCanonicalRows_encoded_lt_modulus' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement.splitNc_and_generatedLayoutCanonicalRows_encoded_lt_modulus

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement.splitNc_and_generatedArtifactRows_encoded_lt_modulus' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.CanonicalOpeningSplitNc.ArtifactRefinement.splitNc_and_generatedArtifactRows_encoded_lt_modulus
