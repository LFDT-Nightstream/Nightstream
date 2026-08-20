import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyBinding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRecursiveVerifierKeyRowSound

/-!
Contract: recursive lifecycle binding for the exact production verifier-key rows.

Owns the adapter from accepted recursive source rows to the same verifier
digest and initial boundary used by the typed lifecycle configuration.

Does not own final selective-row inclusion, Poseidon2 collision resistance,
or any other lifecycle row family.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRecursiveVerifierKeyBinding

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleRecursiveVerifierKey
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyBinding
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRecursiveVerifierKeyProgramBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRecursiveVerifierKeyRowSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Protocol.FPrime

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen

/-- Deterministic agreement between the recursive verifier-key recipes and
the abstract lifecycle hashes. The context input object is shared with the
base artifact. -/
structure Poseidon2Compatible
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (encoding : ContextEncoding Params StructureDigest)
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount) : Prop where
  parametersExact : encoding.parameters configuration.context.params =
    productionParameters
  publicInputLengthExact : configuration.context.publicInputLength =
    productionParameters.publicInputLength
  verifierDigestExact : ∀ lane,
    computedPolicyDigest (contextInputs encoding configuration) lane =
      digestValues
        (XOut.verifierDigest configuration.hashSemantics configuration.context)
        lane
  initialBoundaryExact : ∀ lane,
    computedInitialBoundary (contextInputs encoding configuration) lane =
      digestValues
        (XOut.initialBoundary configuration.hashSemantics configuration.context)
        lane

/-- Exact recursive source rows bind the verifier digest and initial boundary
used by every post-base lifecycle invocation. -/
theorem rows_bind_lifecycle_digests
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (encoding : ContextEncoding Params StructureDigest)
    (compatible : Poseidon2Compatible encoding configuration)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (authority : InputColumnsBound assignment
      (contextInputs encoding configuration))
    (satisfied : StageSatisfied assignment) :
    (∀ lane,
      assignment
          (rawArtifact.policyDigestBinding.leftColumns.getD lane.val 0) =
        digestValues
          (XOut.verifierDigest configuration.hashSemantics
            configuration.context) lane) ∧
      (∀ lane,
        assignment
            (rawArtifact.initialBoundaryBinding.leftColumns.getD lane.val 0) =
          digestValues
            (XOut.initialBoundary configuration.hashSemantics
              configuration.context) lane) := by
  have outputs := stage_rows_imply_outputs assignment
    (contextInputs encoding configuration) canonical one authority satisfied
  constructor
  · intro lane
    exact (outputs.vkFsDigest lane).trans
      (compatible.verifierDigestExact lane)
  · intro lane
    exact (outputs.initialBoundary lane).trans
      (compatible.initialBoundaryExact lane)

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRecursiveVerifierKeyBinding
