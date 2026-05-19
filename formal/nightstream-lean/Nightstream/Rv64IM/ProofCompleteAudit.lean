import Nightstream.Rv64IM.AcceptedArtifactBackendRefinement
import Nightstream.Rv64IM.AcceptedArtifactChecks
import Nightstream.Rv64IM.AcceptedArtifactCompleteness
import Nightstream.Rv64IM.AcceptedArtifactConstructorAudit
import Nightstream.Rv64IM.AcceptedArtifactKernelDesignBridgeClosure
import Nightstream.Rv64IM.AcceptedArtifactRootExecutionClosure
import Nightstream.Rv64IM.AcceptedArtifactRootExecutionSemanticsClosure
import Nightstream.Rv64IM.Kernel.AcceptedProofCheckerBackendRefinement
import Nightstream.Rv64IM.Kernel.RequiredProofCompleteRustExportSurface
import Nightstream.Rv64IM.SideTerminalOpeningDigestBinding

/-!
Top-level RV64IM proof-completeness audit. This owner is intentionally harsher
than parity-only checks: it treats hidden exact-boundary witnesses and missing
backend refinement into `Π_CCS / Π_RLC / Π_DEC` as hard closure failures.
-/

namespace Nightstream.Rv64IM

inductive ProofCompleteStaticField where
  | chunkLayoutOwner
  | chunkedRootProofOwner
  | mainLaneTraceBoundaryCarriesSchedule
  | transcriptScheduleCarriesRootChunkEvents
  | rootExecutionSemanticsOwnerAboveExactTrace
  | requiredBackendPayloadSurfaceOwner
  | requiredRootExecutionSemanticsSurfaceOwner
  | requiredKernelDesignBridgeSurfaceOwner
  | requiredProofCompleteRustExportSurfaceOwner
  | rootExecutionSemanticsExportSurfacePresent
  | kernelDesignBridgeExportSurfacePresent
  | proofCompleteRustExportContractPresent
  | acceptedArtifactNoStoredExactBoundaryWitness
  | acceptedProofSoundnessNoStoredExactBoundaryWitness
  | acceptedProofCheckerNoStoredExactBoundaryWitness
  | publicProofBoundaryNoStoredAcceptedWitness
  | acceptedCheckerOwnsBoundaryConstruction
  | bridgeTheoremBindsAuthenticatedSelection
  | bridgeTheoremBindsRootExecution
  | bridgeTheoremBindsStageObligations
  | bridgeTheoremBindsKernelOpenings
  | backendPayloadSurfaceNotSummaryOnly
  | acceptedCheckerOwnsBackendRefinement
  | sideTerminalOpeningDigestCanonicallyBound
deriving DecidableEq, Repr

def proofCompleteStaticFieldName : ProofCompleteStaticField → String
  | .chunkLayoutOwner => "chunk_layout_owner"
  | .chunkedRootProofOwner => "chunked_root_proof_owner"
  | .mainLaneTraceBoundaryCarriesSchedule =>
      "main_lane_trace_boundary_carries_schedule"
  | .transcriptScheduleCarriesRootChunkEvents =>
      "transcript_schedule_carries_root_chunk_events"
  | .rootExecutionSemanticsOwnerAboveExactTrace =>
      "root_execution_semantics_owner_above_exact_trace"
  | .requiredBackendPayloadSurfaceOwner =>
      "required_backend_payload_surface_owner"
  | .requiredRootExecutionSemanticsSurfaceOwner =>
      "required_root_execution_semantics_surface_owner"
  | .requiredKernelDesignBridgeSurfaceOwner =>
      "required_kernel_design_bridge_surface_owner"
  | .requiredProofCompleteRustExportSurfaceOwner =>
      "required_proof_complete_rust_export_surface_owner"
  | .rootExecutionSemanticsExportSurfacePresent =>
      "root_execution_semantics_export_surface_present"
  | .kernelDesignBridgeExportSurfacePresent =>
      "kernel_design_bridge_export_surface_present"
  | .proofCompleteRustExportContractPresent =>
      "proof_complete_rust_export_contract_present"
  | .acceptedArtifactNoStoredExactBoundaryWitness =>
      "accepted_artifact_no_stored_exact_boundary_witness"
  | .acceptedProofSoundnessNoStoredExactBoundaryWitness =>
      "accepted_proof_soundness_no_stored_exact_boundary_witness"
  | .acceptedProofCheckerNoStoredExactBoundaryWitness =>
      "accepted_proof_checker_no_stored_exact_boundary_witness"
  | .publicProofBoundaryNoStoredAcceptedWitness =>
      "public_proof_boundary_no_stored_accepted_witness"
  | .acceptedCheckerOwnsBoundaryConstruction =>
      "accepted_checker_owns_boundary_construction"
  | .bridgeTheoremBindsAuthenticatedSelection =>
      "bridge_theorem_binds_authenticated_selection"
  | .bridgeTheoremBindsRootExecution =>
      "bridge_theorem_binds_root_execution"
  | .bridgeTheoremBindsStageObligations =>
      "bridge_theorem_binds_stage_obligations"
  | .bridgeTheoremBindsKernelOpenings =>
      "bridge_theorem_binds_kernel_openings"
  | .backendPayloadSurfaceNotSummaryOnly =>
      "backend_payload_surface_not_summary_only"
  | .acceptedCheckerOwnsBackendRefinement =>
      "accepted_checker_owns_backend_refinement"
  | .sideTerminalOpeningDigestCanonicallyBound =>
      "side_terminal_opening_digest_canonically_bound"

def requiredProofCompleteStaticFields : List ProofCompleteStaticField :=
  [ .chunkLayoutOwner
  , .chunkedRootProofOwner
  , .mainLaneTraceBoundaryCarriesSchedule
  , .transcriptScheduleCarriesRootChunkEvents
  , .rootExecutionSemanticsOwnerAboveExactTrace
  , .requiredBackendPayloadSurfaceOwner
  , .requiredRootExecutionSemanticsSurfaceOwner
  , .requiredKernelDesignBridgeSurfaceOwner
  , .requiredProofCompleteRustExportSurfaceOwner
  , .rootExecutionSemanticsExportSurfacePresent
  , .kernelDesignBridgeExportSurfacePresent
  , .proofCompleteRustExportContractPresent
  , .acceptedArtifactNoStoredExactBoundaryWitness
  , .acceptedProofSoundnessNoStoredExactBoundaryWitness
  , .acceptedProofCheckerNoStoredExactBoundaryWitness
  , .publicProofBoundaryNoStoredAcceptedWitness
  , .acceptedCheckerOwnsBoundaryConstruction
  , .bridgeTheoremBindsAuthenticatedSelection
  , .bridgeTheoremBindsRootExecution
  , .bridgeTheoremBindsStageObligations
  , .bridgeTheoremBindsKernelOpenings
  , .backendPayloadSurfaceNotSummaryOnly
  , .acceptedCheckerOwnsBackendRefinement
  , .sideTerminalOpeningDigestCanonicallyBound
  ]

def proofCompleteStaticFieldPresent : ProofCompleteStaticField → Bool
  | .chunkLayoutOwner => true
  | .chunkedRootProofOwner => true
  | .mainLaneTraceBoundaryCarriesSchedule => true
  | .transcriptScheduleCarriesRootChunkEvents => true
  | .rootExecutionSemanticsOwnerAboveExactTrace => true
  | .requiredBackendPayloadSurfaceOwner => true
  | .requiredRootExecutionSemanticsSurfaceOwner => true
  | .requiredKernelDesignBridgeSurfaceOwner => true
  | .requiredProofCompleteRustExportSurfaceOwner => true
  | .rootExecutionSemanticsExportSurfacePresent =>
      validGeneratedRv64imRequiredProofCompleteRootExecutionSemanticsSurfaceCases
  | .kernelDesignBridgeExportSurfacePresent =>
      validGeneratedRv64imRequiredProofCompleteKernelDesignBridgeSurfaceCases
  | .proofCompleteRustExportContractPresent =>
      validGeneratedRv64imRequiredProofCompleteRustExportCases
  | .acceptedArtifactNoStoredExactBoundaryWitness => true
  | .acceptedProofSoundnessNoStoredExactBoundaryWitness => true
  | .acceptedProofCheckerNoStoredExactBoundaryWitness => true
  | .publicProofBoundaryNoStoredAcceptedWitness => true
  | .acceptedCheckerOwnsBoundaryConstruction => true
  | .bridgeTheoremBindsAuthenticatedSelection => true
  | .bridgeTheoremBindsRootExecution => true
  | .bridgeTheoremBindsStageObligations => true
  | .bridgeTheoremBindsKernelOpenings => true
  | .backendPayloadSurfaceNotSummaryOnly =>
      validGeneratedRv64imRequiredProofCompleteBackendSurfaceCases
  | .acceptedCheckerOwnsBackendRefinement => true
  | .sideTerminalOpeningDigestCanonicallyBound =>
      validGeneratedRv64imSideTerminalOpeningDigestBindingCases

def proofCompleteStaticChecks : List (String × Bool) :=
  requiredProofCompleteStaticFields.map fun field =>
    (proofCompleteStaticFieldName field, proofCompleteStaticFieldPresent field)

def validRv64imProofCompleteStaticChecks : Bool :=
  proofCompleteStaticChecks.all Prod.snd

def proofCompleteStaticFailures : List String :=
  proofCompleteStaticChecks.filterMap fun (name, ok) =>
    if ok then none else some name

def proofCompleteCaseAccepted (artifact : Generated.AcceptedProofArtifactView) : Bool :=
  checkAcceptedArtifactCase artifact &&
    theoremCompleteAcceptedArtifact artifact &&
    exactTraceBoundaryConstructible artifact &&
    exactKernelBoundaryConstructible artifact &&
    requiredProofCompleteRustExportSurfacePresent artifact &&
    rootExecutionClosureAccepted artifact &&
    rootExecutionSemanticsClosureAccepted artifact &&
    kernelDesignBridgeClosureAccepted artifact &&
    backendRefinementAccepted artifact

private def appendUniqueStrings
    (acc : List String)
    (xs : List String) : List String :=
  xs.foldl (fun acc x => if x ∈ acc then acc else acc ++ [x]) acc

def proofCompleteRustExportBlockers
    (artifact : Generated.AcceptedProofArtifactView) : List String :=
  requiredProofCompleteRustExportBlockers artifact

def proofCompleteBackendRustExportBlockers
    (artifact : Generated.AcceptedProofArtifactView) : List String :=
  requiredProofCompleteBackendRustExportBlockers artifact

def proofCompleteRootExecutionSemanticsRustExportBlockers
    (artifact : Generated.AcceptedProofArtifactView) : List String :=
  requiredProofCompleteRootExecutionSemanticsRustExportBlockers artifact

def proofCompleteKernelDesignBridgeRustExportBlockers
    (artifact : Generated.AcceptedProofArtifactView) : List String :=
  requiredProofCompleteKernelDesignBridgeRustExportBlockers artifact

def uniqueProofCompleteRustExportBlockers : List String :=
  uniqueRequiredProofCompleteRustExportBlockers

def uniqueProofCompleteBackendRustExportBlockers : List String :=
  uniqueRequiredProofCompleteBackendRustExportBlockers

def uniqueProofCompleteRequiredBackendPayloadFields : List String :=
  uniqueRequiredProofCompleteBackendMissingFields

def uniqueProofCompleteRequiredRootExecutionSemanticsFields : List String :=
  uniqueRequiredProofCompleteRootExecutionSemanticsMissingFields

def uniqueProofCompleteRequiredKernelDesignBridgeFields : List String :=
  uniqueRequiredProofCompleteKernelDesignBridgeMissingFields

def uniqueProofCompleteRequiredRustExportFields : List String :=
  uniqueMissingRequiredProofCompleteRustExportFields

def validProofCompleteRequiredBackendPayloadSurface : Bool :=
  validGeneratedRv64imRequiredProofCompleteBackendSurfaceCases

def validProofCompleteRequiredRootExecutionSemanticsSurface : Bool :=
  validGeneratedRv64imRequiredProofCompleteRootExecutionSemanticsSurfaceCases

def validProofCompleteRequiredKernelDesignBridgeSurface : Bool :=
  validGeneratedRv64imRequiredProofCompleteKernelDesignBridgeSurfaceCases

def validProofCompleteRequiredRustExportSurface : Bool :=
  validGeneratedRv64imRequiredProofCompleteRustExportCases

def uniqueProofCompleteRootExecutionSemanticsRustExportBlockers : List String :=
  uniqueRequiredProofCompleteRootExecutionSemanticsRustExportBlockers

def uniqueProofCompleteKernelDesignBridgeRustExportBlockers : List String :=
  uniqueRequiredProofCompleteKernelDesignBridgeRustExportBlockers

def proofCompleteStaticFieldMissingRequiredFields :
    ProofCompleteStaticField → List String
  | .backendPayloadSurfaceNotSummaryOnly =>
      uniqueProofCompleteRequiredBackendPayloadFields
  | .rootExecutionSemanticsExportSurfacePresent =>
      uniqueProofCompleteRequiredRootExecutionSemanticsFields
  | .kernelDesignBridgeExportSurfacePresent =>
      uniqueProofCompleteRequiredKernelDesignBridgeFields
  | .proofCompleteRustExportContractPresent =>
      uniqueProofCompleteRequiredRustExportFields
  | _ => []

def proofCompleteStaticFieldRustExportBlockers :
    ProofCompleteStaticField → List String
  | .backendPayloadSurfaceNotSummaryOnly =>
      uniqueProofCompleteBackendRustExportBlockers
  | .rootExecutionSemanticsExportSurfacePresent =>
      uniqueProofCompleteRootExecutionSemanticsRustExportBlockers
  | .kernelDesignBridgeExportSurfacePresent =>
      uniqueProofCompleteKernelDesignBridgeRustExportBlockers
  | .proofCompleteRustExportContractPresent =>
      uniqueProofCompleteRustExportBlockers
  | .sideTerminalOpeningDigestCanonicallyBound =>
      uniqueSideTerminalOpeningDigestBindingBlockers
  | _ => []

structure ProofCompleteStaticFailureReport where
  field : String
  missingRequiredFields : List String
  rustExportBlockers : List String
deriving Repr

def proofCompleteStaticFailureReports : List ProofCompleteStaticFailureReport :=
  requiredProofCompleteStaticFields.filterMap fun field =>
    if proofCompleteStaticFieldPresent field then
      none
    else
      some
        { field := proofCompleteStaticFieldName field
        , missingRequiredFields :=
            proofCompleteStaticFieldMissingRequiredFields field
        , rustExportBlockers :=
            proofCompleteStaticFieldRustExportBlockers field
        }

def uniqueProofCompleteStaticFailureMissingRequiredFields : List String :=
  requiredProofCompleteStaticFields.foldl (fun acc field =>
    if proofCompleteStaticFieldPresent field then
      acc
    else
      appendUniqueStrings acc (proofCompleteStaticFieldMissingRequiredFields field))
    []

def uniqueProofCompleteStaticFailureRustExportBlockers : List String :=
  requiredProofCompleteStaticFields.foldl (fun acc field =>
    if proofCompleteStaticFieldPresent field then
      acc
    else
      appendUniqueStrings acc (proofCompleteStaticFieldRustExportBlockers field))
    []

def uniqueProofCompleteClosureBlockers : List String :=
  appendUniqueStrings proofCompleteStaticFailures uniqueProofCompleteRustExportBlockers

structure Rv64imProofCompleteReport where
  name : String
  parityAccepted : Bool
  completenessAccepted : Bool
  exactTraceAccepted : Bool
  exactKernelAccepted : Bool
  requiredBackendPayloadSurfaceAccepted : Bool
  requiredRootExecutionSemanticsSurfaceAccepted : Bool
  requiredKernelDesignBridgeSurfaceAccepted : Bool
  requiredRustExportSurfaceAccepted : Bool
  rootExecutionAccepted : Bool
  rootExecutionSemanticsAccepted : Bool
  kernelDesignBridgeAccepted : Bool
  backendRefinementAccepted : Bool
  missingKernelDesignBridgeFields : List String
  missingRootExecutionFields : List String
  missingRootExecutionSemanticsFields : List String
  missingBackendFields : List String
  missingCompletenessFields : List String
  missingTraceSlots : List String
  missingKernelSlots : List String
  missingRequiredBackendPayloadFields : List String
  missingRequiredRootExecutionSemanticsFields : List String
  missingRequiredKernelDesignBridgeFields : List String
  missingRequiredProofCompleteRustExportFields : List String
  backendRustExportBlockers : List String
  rootExecutionSemanticsRustExportBlockers : List String
  kernelDesignBridgeRustExportBlockers : List String
  rustExportBlockers : List String
deriving Repr

def rv64imProofCompleteChecks : List Bool :=
  Generated.AcceptedProofArtifacts.cases.map proofCompleteCaseAccepted

def validGeneratedRv64imProofCompleteCases : Bool :=
  Generated.AcceptedProofArtifacts.cases.all proofCompleteCaseAccepted

def rv64imProofCompleteReports : List Rv64imProofCompleteReport :=
  Generated.AcceptedProofArtifacts.cases.map fun artifact =>
    let requiredRustExportReport := requiredProofCompleteRustExportReport artifact
    { name := artifact.name
    , parityAccepted := checkAcceptedArtifactCase artifact
    , completenessAccepted := theoremCompleteAcceptedArtifact artifact
    , exactTraceAccepted := exactTraceBoundaryConstructible artifact
    , exactKernelAccepted := exactKernelBoundaryConstructible artifact
    , requiredBackendPayloadSurfaceAccepted :=
        requiredRustExportReport.backendSurfacePresent
    , requiredRootExecutionSemanticsSurfaceAccepted :=
        requiredRustExportReport.rootExecutionSemanticsSurfacePresent
    , requiredKernelDesignBridgeSurfaceAccepted :=
        requiredRustExportReport.kernelDesignBridgeSurfacePresent
    , requiredRustExportSurfaceAccepted :=
        requiredRustExportReport.aggregateSurfacePresent
    , rootExecutionAccepted := rootExecutionClosureAccepted artifact
    , rootExecutionSemanticsAccepted := rootExecutionSemanticsClosureAccepted artifact
    , kernelDesignBridgeAccepted := kernelDesignBridgeClosureAccepted artifact
    , backendRefinementAccepted := backendRefinementAccepted artifact
    , missingKernelDesignBridgeFields := missingKernelDesignBridgeClosureFields artifact
    , missingRootExecutionFields := missingRootExecutionClosureFields artifact
    , missingRootExecutionSemanticsFields := missingRootExecutionSemanticsClosureFields artifact
    , missingBackendFields := missingBackendRefinementFields artifact
    , missingCompletenessFields := missingAcceptedArtifactTheoremFields artifact
    , missingTraceSlots := missingExactTraceConstructorSlots artifact
    , missingKernelSlots := missingExactKernelConstructorSlots artifact
    , missingRequiredBackendPayloadFields :=
        requiredRustExportReport.missingBackendFields
    , missingRequiredRootExecutionSemanticsFields :=
        requiredRustExportReport.missingRootExecutionSemanticsFields
    , missingRequiredKernelDesignBridgeFields :=
        requiredRustExportReport.missingKernelDesignBridgeFields
    , missingRequiredProofCompleteRustExportFields :=
        requiredRustExportReport.missing
    , backendRustExportBlockers :=
        requiredRustExportReport.backendRustExportBlockers
    , rootExecutionSemanticsRustExportBlockers :=
        requiredRustExportReport.rootExecutionSemanticsRustExportBlockers
    , kernelDesignBridgeRustExportBlockers :=
        requiredRustExportReport.kernelDesignBridgeRustExportBlockers
    , rustExportBlockers := requiredRustExportReport.rustExportBlockers
    }

def validRv64imProofCompleteClosure : Bool :=
  validRv64imProofCompleteStaticChecks &&
    validGeneratedRv64imProofCompleteCases

end Nightstream.Rv64IM
