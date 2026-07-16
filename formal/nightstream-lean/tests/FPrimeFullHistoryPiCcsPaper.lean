import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiCcs

/-! External checks for the explicit production-to-paper Π_CCS boundary. -/

namespace tests.FPrimeFullHistoryPiCcsPaper

open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs

example : recursiveArity.total = 1 := by rfl

example : terminalArity.total = 15 := by rfl

/-- The fresh and CE decoders retain their distinct public-input types. -/
example (assignment : Nat → Nat) (columns : FreshColumns) :
    (decodeFreshCcs assignment columns).publicInput =
      decodeFreshPublicInput assignment columns := by
  rfl

example (assignment : Nat → Nat) (columns : CeColumns) :
    (decodeCe assignment columns).publicInput =
      decodeCePublicInput assignment columns := by
  rfl

#check carrierCoverageArtifact_matches_fixedWidths
#check fixedWidth_pi_ccs_artifact_accepts_nc_false_carrier
#check fixedCarrierArtifact_exactProfile
#check fixedCarrierArtifact_protocolOutcomes
#check fixedCarrierArtifact_sameWitnessPair
#check fixedCarrierArtifact_tailDecodeViolatesSemanticTruth
#check fixedCarrier_pi_ccs_artifact_accepts_nc_false_carrier
#check fixedCarrier_nifs_artifact_accepts_nc_false_carrier
#check fixedCarrierArtifact_linkedRecursiveProfile
#check fixedCarrierArtifact_linkedTail_not_normBounded
#check fixedCarrier_recursive_f_prime_artifact_accepts_non_norm_carrier

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.distinguishedProjection_does_not_determine_packedInput' depends on axioms: [propext] -/
#guard_msgs in
#print axioms distinguishedProjection_does_not_determine_packedInput

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedWidth_pi_ccs_artifact_accepts_nc_false_carrier' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms fixedWidth_pi_ccs_artifact_accepts_nc_false_carrier

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedCarrier_pi_ccs_artifact_accepts_nc_false_carrier' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms fixedCarrier_pi_ccs_artifact_accepts_nc_false_carrier

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedCarrier_nifs_artifact_accepts_nc_false_carrier' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms fixedCarrier_nifs_artifact_accepts_nc_false_carrier

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs.fixedCarrier_recursive_f_prime_artifact_accepts_non_norm_carrier' depends on axioms: [propext] -/
#guard_msgs in
#print axioms fixedCarrier_recursive_f_prime_artifact_accepts_non_norm_carrier

end tests.FPrimeFullHistoryPiCcsPaper
