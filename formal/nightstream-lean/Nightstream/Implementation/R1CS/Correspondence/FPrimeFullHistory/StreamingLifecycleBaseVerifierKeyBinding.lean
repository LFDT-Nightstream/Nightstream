import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRelation

/-!
Contract: base lifecycle binding for the exact production verifier-key rows.

Owns the typed encoding of every verifier-key preimage input and the adapter
from accepted source rows to the lifecycle verifier digest and initial
boundary.

Does not own final selective-row inclusion, Poseidon2 collision resistance,
or any other lifecycle row family.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyBinding

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKey
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyProgramBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyRowSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.VerifierKeyProgram
open Nightstream.Protocol.FPrime

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen

/-- Canonical raw-field encoders for the two abstract verifier-owned context
types. The parameter encoder also owns the Ajtai parameter digest. -/
structure ContextEncoding
    (Params : Type uParams) (StructureDigest : Type uStructure) where
  parameters : Params → Parameters
  structureDigest : StructureDigest → DigestFields
  ajtaiPpDigest : Params → DigestFields

/-- Complete raw-field input object derived from the typed lifecycle context. -/
def contextInputs
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (encoding : ContextEncoding Params StructureDigest)
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount) : Inputs where
  structureDigest := encoding.structureDigest
    configuration.context.structureDigest
  piCcsHeader := fun lane =>
    digestValues configuration.context.piCcsHeader lane
  ajtaiPpDigest := encoding.ajtaiPpDigest configuration.context.params
  initialSemanticStateDigest := fun lane =>
    digestValues configuration.context.initialSemanticState lane

/-- Deterministic agreement between the abstract lifecycle hashes and the
exact production Poseidon2 programs. All context inputs are explicit, so the
two digest equations are implementation compatibility claims, not authority. -/
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

/-- Exact base source rows bind the verifier digest and initial boundary used
by the lifecycle relation. Source columns are bound to the complete context
preimage; no carried digest is accepted as authority. -/
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

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyBinding
