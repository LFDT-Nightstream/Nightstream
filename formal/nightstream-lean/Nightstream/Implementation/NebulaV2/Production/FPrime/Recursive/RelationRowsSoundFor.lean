import Nightstream.Implementation.NebulaV2.Production.Carrier.NifsPublicCarrierFor
import Nightstream.Implementation.NebulaV2.Production.Carrier.FieldNativeCompactChainRowsFor
import Nightstream.Implementation.NebulaV2.Production.FPrime.Lifetime.PriorStateAuthorityRowsFor
import Nightstream.Implementation.NebulaV2.Production.NIFS.Core.NifsOutputRowsFor

/-!
Contract: one exponent-indexed HyperNova Construction-2 recursive relation
verifies and consumes one complete Nebula-on-SuperNeo fresh claim.

The theorem composes the complete full-claim carrier, PiCCS, PiRLC, PiDEC,
paper NIFS, checked memory batch, and prior-state hash at one explicit relation
exponent. It accepts no verifier result, sampler result, prior digest, parsed
carry, memory equality, or delayed transition as a premise.

Assurance tier: exponent-indexed section composition.

Does not own the application transition, successor rows, terminal invocation,
recursive-size closure, generated-artifact containment, cryptographic
reductions, external bytes, or Rust refinement.

Emits constraints: no; it composes named row-soundness theorems.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveRelationRowsSoundFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

abbrev ProtocolSchema
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductionFieldNativeFullClaim.protocolSchema
    (FullShape rowVariables logicalWidth publicFits)
    (ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)

abbrev ProtocolClaim
    (candidate : Id) (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductionBatchedFPrime.Claim candidate
    (ProtocolSchema rowVariables logicalWidth publicFits)
    Digest.Value (Challenges K) (ProductState.State K)

noncomputable def paperVerifier
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits) :
    ProductionBatchedFPrime.Verifier candidate
      (ProtocolSchema rowVariables logicalWidth publicFits)
      Digest.Value (Challenges K) (ProductState.State K) :=
  fun proof claim =>
    let selected := ProductionProductPiCcsTypedBridgeFor.paperKey candidate
      statementId config artifact
    let fresh := ProductNifsCodec.freshOfFor
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) claim.commitmentBundle claim.ccsPublic
    verify selected claim.recursiveState fresh proof =
      some (selected.output claim.recursiveState fresh proof)

noncomputable def boundWires
    (candidate : Id) (rowVariables : Nat)
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (carrier : ProductionFullClaimCarrierLayoutFor.Layout candidate rowVariables)
    (base : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables) :
    ProductionProductPiCcsTypedBridgeFor.Wires rowVariables :=
  ProductionFullClaimNifsPublicCarrierFor.bindPublicFields candidate
    rowVariables (FullShape rowVariables logicalWidth publicFits) 9 carrier base

structure RowsHold
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout)
    (nifsOutputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables)
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables)
    (compactManifest : SeedSchedule.Manifest)
    (compactLayout : ProductionFieldNativeCompactChainRowsFor.Layout)
    (assignment : Nat -> Nat) : Prop where
  piCcs : Satisfies
    (ProductPiCcsTranscriptRowsFor.rows
      (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
        config artifact value.recursiveState
        (ProductionFieldNativeFullClaim.freshOfValue
          (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
            publicFits).toShape value)
        wires)) assignment
  samplerTranscript : ProductPiRlcTranscriptRows.RowsHold
    (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
      config artifact value.recursiveState
      (ProductionFieldNativeFullClaim.freshOfValue
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits).toShape value)
      wires samplerBase) assignment
  samplerClassification : ProductPiRlcCandidateClassificationRows.RowsHold
    (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
      config artifact value.recursiveState
      (ProductionFieldNativeFullClaim.freshOfValue
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits).toShape value)
      wires samplerBase) assignment
  samplerSelector : ProductPiRlcFirstAcceptedBatchRows.RowsHold
    (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
      config artifact value.recursiveState
      (ProductionFieldNativeFullClaim.freshOfValue
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits).toShape value)
      wires samplerBase) assignment
  algebra : Satisfies (ProductPiRlcAlgebraRows.rows algebraLayout) assignment
  piDec : Satisfies (ProductPiDecRows.rows piDecLayout) assignment
  nifsOutputValid : nifsOutputLayout.Valid
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits) algebraLayout piDecLayout
  nifsOutput : Satisfies
    (ProductionProductNifsOutputRowsFor.rows nifsOutputLayout
      (ProductionProductNifsOutputRowsFor.verifierPoint candidate statementId
        config artifact value.recursiveState
        (ProductionFieldNativeFullClaim.freshOfValue
          (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
            publicFits).toShape value) wires)) assignment
  memory : Satisfies
    (ProductionMemoryCheckedBatchRows.rows
      priorAuthority.ccs.core.batch.frame.memory) assignment
  priorState : Satisfies
    (ProductionPaperPriorStateAuthorityRowsFor.rows priorAuthority statementId)
      assignment
  compactValid : compactLayout.Valid compactManifest
    priorAuthority.ccs.carrier priorAuthority.ccs.core.batch.frame.memory
  compact : Satisfies
    (ProductionFieldNativeCompactChainRowsFor.rows compactManifest compactLayout)
    assignment

structure Placement
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (baseWires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Prop where
  carrierValue : ProductionFullClaimCarrierLayoutFor.Placed
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits) priorAuthority.ccs.carrier assignment value
  memoryAliases : ProductionFullClaimCarrierLayoutFor.CheckedMemoryAliases
    priorAuthority.ccs.carrier priorAuthority.ccs.core.batch.frame.memory
  piCcsRemaining : ProductionFullClaimNifsPublicCarrierFor.RemainingPlacement
    candidate statementId config artifact value.recursiveState
    (ProductionFieldNativeFullClaim.freshOfValue
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits).toShape value)
    proof (boundWires (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate rowVariables
      priorAuthority.ccs.carrier baseWires) assignment
  nifs : ProductionProductNifsPaperRowsSoundFor.Placement candidate statementId
    config artifact value.recursiveState
    (ProductionFieldNativeFullClaim.freshOfValue
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits).toShape value)
    proof (boundWires (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate rowVariables
      priorAuthority.ccs.carrier baseWires) samplerBase algebraLayout
      piDecLayout assignment canonical

structure Result
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables)
    (assignment : Nat -> Nat)
    (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) where
  assignmentCanonical : forall column, assignment column < goldilocksP
  nifsOutputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables
  nifsOutputPlaced : ProductionProductNifsOutputRowsFor.Placed
    nifsOutputLayout assignment assignmentCanonical
    ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
      config artifact).output value.recursiveState
      (ProductionFieldNativeFullClaim.freshOfValue
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits).toShape value) proof)
  memoryResult : ProductionMemoryCheckedBatchRows.Result
    priorAuthority.ccs.core.batch.frame.memory assignment headers
  priorCarry : ProductionMemoryCarryRows.Sound
    priorAuthority.carry assignment headers
  priorState : ProductionSuccessorStateBinding.Value candidate
    (FullShape rowVariables logicalWidth publicFits)
  priorAuthorityResult : ProductionPaperPriorStateAuthorityRowsFor.Result
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits) priorAuthority assignment headers statementId priorPrefix value
    memoryResult priorCarry priorState
  verified : ProductionBatchedFPrime.Verified candidate
    (ProtocolSchema rowVariables logicalWidth publicFits)
    Digest.Value (Challenges K) (ProductState.State K)
    (paperVerifier candidate statementId config artifact)
  claimExact : verified.claim = value.toProtocolClaim
    (NifsProof := ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
  proofExact : verified.proof = proof
  memoryExact : memoryResult.suffixBatch = value.memory
  compactManifest : SeedSchedule.Manifest
  compactLayout : ProductionFieldNativeCompactChainRowsFor.Layout
  compactValid : compactLayout.Valid compactManifest
    priorAuthority.ccs.carrier
    priorAuthority.ccs.core.batch.frame.memory
  compactExact : ProductionFieldNativeCompactChainRowsFor.Result
    compactManifest priorAuthority.ccs.carrier
    priorAuthority.ccs.core.batch.frame.memory compactLayout assignment
    assignmentCanonical headers value
    (memoryResult.claim
      (ProductionFieldNativeCompactChainRowsFor.firstStep candidate))
  ccsFullMatches : ProductionMemoryBoundCcsPublic.FullMatches
    value.ccsPublic
      (ProductionSuccessorStateBinding.outputDigest statementId priorState)
      value.memory
  transition : ProductionBatchedFPrime.Transition
    (paperVerifier candidate statementId config artifact)
    MemoryProductBalanceRows.ConcreteBalanced
    (memoryResult.semantic 0) verified
    (memoryResult.semantic
      (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)))

theorem rows_imply_verified_exact_claim_and_memory_transition
    {Program : Type}
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      candidate rowVariables)
    (priorAuthorityValid : priorAuthority.Valid)
    (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (assignment : Nat -> Nat)
    (priorPrefixPlaced : ProductionPaperPriorStateAuthorityRowsFor.PrefixPlaced
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) priorAuthority assignment priorPrefix)
    (statement : ProductionStatement Program)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (valueCanonical : value.Canonical)
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (baseWires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout)
    (nifsOutputLayout : ProductionProductNifsOutputRowsFor.Layout rowVariables)
    (compactManifest : SeedSchedule.Manifest)
    (compactLayout : ProductionFieldNativeCompactChainRowsFor.Layout)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (prefixCanonical : ProductionFullClaimNifsPublicCarrierFor.PrefixCanonical
      candidate (FullShape rowVariables logicalWidth publicFits) 9)
    (headersPlaced : ProductionMemoryCheckedBatchRows.HeadersPlaced
      priorAuthority.ccs.core.batch.frame.memory assignment headers)
    (carryHeadersPlaced : MemoryCarryRows.HeadersPlaced
      priorAuthority.carry.carry assignment headers)
    (rowsHold : RowsHold candidate statementId config artifact value proof
      (boundWires (logicalWidth := logicalWidth)
        (publicFits := publicFits) candidate rowVariables
      priorAuthority.ccs.carrier baseWires)
      samplerBase algebraLayout piDecLayout nifsOutputLayout priorAuthority
      compactManifest compactLayout assignment)
    (placement : Placement candidate statementId config artifact priorAuthority
      value proof baseWires samplerBase algebraLayout piDecLayout assignment
      canonical) :
    let wires := boundWires (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate rowVariables
      priorAuthority.ccs.carrier baseWires
    let fresh := ProductionFieldNativeFullClaim.freshOfValue
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits).toShape value
    let sampleInput := ProductionProductPiRlcParentBridgeFor.samplerInput
      candidate statementId config artifact value.recursiveState fresh wires
        samplerBase
    exists _result : Result candidate statementId config artifact
        priorAuthority assignment headers priorPrefix value proof,
      _result.nifsOutputLayout = nifsOutputLayout /\
        _result.compactManifest = compactManifest /\
        ProductPoseidon2.samplerSucceeded
          (ProductPiRlcFirstAcceptedBatchSound.samplerState sampleInput
            assignment) = true /\
        piCcsCheck
          (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
            config artifact) value.recursiveState fresh proof = true /\
        piDecCheck
          (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
            config artifact) value.recursiveState fresh proof = true /\
        verify
          (ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
            config artifact) value.recursiveState fresh proof =
          some ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate
            statementId config artifact).output value.recursiveState fresh
              proof) := by
  dsimp only
  let wires := boundWires (logicalWidth := logicalWidth)
    (publicFits := publicFits) candidate rowVariables
    priorAuthority.ccs.carrier baseWires
  let fresh := ProductionFieldNativeFullClaim.freshOfValue
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits).toShape value
  have piCcsPlacement :=
    ProductionFullClaimNifsPublicCarrierFor.piCcsPlacement statementId config
      artifact canonical one prefixCanonical value placement.carrierValue proof
      baseWires placement.piCcsRemaining
  have nifsResult :=
    ProductionProductNifsPaperRowsSoundFor.rows_imply_exact_result candidate
      statementId config artifact value.recursiveState fresh proof wires
      samplerBase algebraLayout piDecLayout assignment canonical one
      piCcsPlacement rowsHold.piCcs rowsHold.samplerTranscript
      rowsHold.samplerClassification rowsHold.samplerSelector rowsHold.algebra
      rowsHold.piDec placement.nifs
  let nifsOutputSection : ProductionProductNifsOutputRowsFor.SectionRows
      candidate statementId config artifact value.recursiveState fresh wires
      algebraLayout piDecLayout assignment :=
    { layout := nifsOutputLayout
      valid := rowsHold.nifsOutputValid
      satisfied := rowsHold.nifsOutput }
  have nifsOutputPlaced :=
    ProductionProductNifsOutputRowsFor.section_rows_sound candidate statementId
      config artifact value.recursiveState fresh proof wires samplerBase
      algebraLayout piDecLayout assignment canonical one piCcsPlacement
      rowsHold.piCcs rowsHold.samplerTranscript rowsHold.samplerClassification
      rowsHold.samplerSelector rowsHold.algebra placement.nifs
      nifsOutputSection
  let memoryResult := ProductionMemoryCheckedBatchRows.derive
    priorAuthorityValid.ccsValid.memoryValid headers canonical one headersPlaced
      rowsHold.memory
  have memoryPlacement :=
    ProductionFullClaimCarrierLayoutFor.checkedMemoryPlacement
      placement.carrierValue placement.memoryAliases
  have memoryResultExact :=
    ProductionMemoryBatchCarrierBridge.rows_bind_and_consume_full_claim_memory
      priorAuthorityValid.ccsValid.memoryValid headers canonical one
      headersPlaced rowsHold.memory value valueCanonical
      memoryPlacement
  have compactExact := ProductionFieldNativeCompactChainRowsFor.exact
    rowsHold.compactValid canonical one headersPlaced placement.carrierValue
    memoryResult rowsHold.compact
  rcases
      ProductionPaperPriorStateAuthorityRowsFor.rows_imply_exact_prior_state_and_fullMatches
        (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
          publicFits) priorAuthorityValid canonical one headers statementId
        priorPrefix priorPrefixPlaced value placement.carrierValue
        carryHeadersPlaced memoryResult rowsHold.priorState with
    ⟨priorCarry, priorState, priorAuthorityResult⟩
  have ccsFullMatches : ProductionMemoryBoundCcsPublic.FullMatches
      value.ccsPublic
        (ProductionSuccessorStateBinding.outputDigest statementId priorState)
        value.memory := by
    rw [← memoryResultExact.1]
    exact priorAuthorityResult.ccsFullMatches
  let verified : ProductionBatchedFPrime.Verified candidate
      (ProtocolSchema rowVariables logicalWidth publicFits)
      Digest.Value (Challenges K) (ProductState.State K)
      (paperVerifier candidate statementId config artifact) :=
    { claim := value.toProtocolClaim
        (NifsProof := ProductionProductPiCcsTypedBridgeFor.ExactProof
          rowVariables)
      proof := proof
      accepted := by
        simpa [paperVerifier, fresh] using nifsResult.2.2.2 }
  let result : Result candidate statementId config artifact priorAuthority
      assignment headers priorPrefix value proof :=
    { assignmentCanonical := canonical
      nifsOutputLayout := nifsOutputLayout
      nifsOutputPlaced := nifsOutputPlaced
      memoryResult := memoryResult
      priorCarry := priorCarry
      priorState := priorState
      priorAuthorityResult := priorAuthorityResult
      verified := verified
      claimExact := rfl
      proofExact := rfl
      memoryExact := memoryResultExact.1
      compactManifest := compactManifest
      compactLayout := compactLayout
      compactValid := rowsHold.compactValid
      compactExact := compactExact
      ccsFullMatches := ccsFullMatches
      transition :=
        { consumes := by
            change ConsumesList MemoryProductBalanceRows.ConcreteBalanced
              (memoryResult.semantic 0) value.memory.suffixes
              (memoryResult.semantic
                (Fin.last
                  (ProductionMemoryCheckedBatchRows.StepCount candidate)))
            exact memoryResultExact.2 } }
  exact ⟨result, rfl, rfl, nifsResult⟩

end Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveRelationRowsSoundFor
