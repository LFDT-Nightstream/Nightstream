import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityNIVCCompatibility

/-!
Focused theorem-surface checks for the selected corrected HyperNova
Definition 12 boundary.
-/

set_option autoImplicit false

namespace tests.PaddedRowIdentityNIVCCompatibility

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityHyperNova
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNIVCCompatibility

#check parametersCodec_canonical
#check structureCodec_canonical
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCompilerDescription.fields_length
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCompilerDescription.structureFields_length
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCompilerDescription.matrices_eq_of_fields_eq
#check assignmentCodec_canonical
#check runningClaimCodec_canonical
#check freshClaimCodec_canonical
#check defaultAlgorithm_holds
#check canonicalZeroPadding
#check compilerLayout_holds
#check Nightstream.HyperNova.NIVCCompatibility.RecursiveSizeClosure.Holds
#check statementIdentifier_holds
#check statementIdentifier_matrices_eq_or_collision
#check statementCodec_encode_exact
#check statementCodec_encode_length
#check compactVerifier_holds
#check construction2Setup_initialTranscriptState
#check definition12_holds
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteNifs.verify_eq_compact
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityHyperNova.terminalHolds_iff_transition

#check (verifyRecursive :
  Nightstream.HyperNova.NIVCCompatibility.RecursiveVerifierKey
      VerifierProjection StatementId ->
    VerifierInput -> VerifierOutput)

example : assignmentColumns <= 2 ^ rowVariables :=
  assignmentColumns_covered

example : statementIdentifier.domainLabel = [statementDomain] := rfl

example : statementIdentifier.identifierWidth = 1 := rfl

namespace RuntimeIndependence

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteNifs
open Nightstream.SuperNeo.Concrete

abbrev RuntimeAjtaiKey :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNIVCCompatibility.AjtaiKey
abbrev RuntimeStructure :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNIVCCompatibility.DenseStructure
abbrev RuntimeRunning :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNIVCCompatibility.PublicRunning
abbrev RuntimeFresh :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNIVCCompatibility.PublicFresh
abbrev RuntimeProof :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityNIVCCompatibility.NifsProof

example
    (statementId : F)
    (leftKey rightKey : RuntimeAjtaiKey)
    (leftSystem rightSystem : RuntimeStructure)
    (running : RuntimeRunning)
    (fresh : RuntimeFresh) :
    (key statementId leftKey leftSystem).publicInputState running fresh =
      (key statementId rightKey rightSystem).publicInputState running fresh := by
  rw [key_publicInputState, key_publicInputState]

example
    (statementId : F)
    (leftKey rightKey : RuntimeAjtaiKey)
    (leftSystem rightSystem : RuntimeStructure)
    (running : RuntimeRunning)
    (fresh : RuntimeFresh) :
    ((key statementId leftKey leftSystem).statement running fresh).verifierInput
        (key statementId leftKey leftSystem).lift =
      ((key statementId rightKey rightSystem).statement running fresh).verifierInput
        (key statementId rightKey rightSystem).lift := by
  rw [key_verifierInput, key_verifierInput]

example
    (statementId : F)
    (ajtaiKey : RuntimeAjtaiKey)
    (system : RuntimeStructure)
    (running : RuntimeRunning)
    (fresh : RuntimeFresh)
    (proof : RuntimeProof) :
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify
        (key statementId ajtaiKey system) running fresh proof =
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.verify
        (compactKey statementId) running fresh proof :=
  verify_eq_compact statementId ajtaiKey system running fresh proof

end RuntimeIndependence

end tests.PaddedRowIdentityNIVCCompatibility
