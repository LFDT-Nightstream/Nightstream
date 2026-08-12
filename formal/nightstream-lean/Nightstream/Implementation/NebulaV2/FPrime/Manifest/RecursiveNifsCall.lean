import Nightstream.Implementation.NebulaV2.FPrime.Claim.NifsCall
import Nightstream.Implementation.NebulaV2.NIFS.Running.ExactRows
import Nightstream.Implementation.NebulaV2.Memory.Transition.TransitionSound
import Nightstream.Implementation.NebulaV2.FPrime.Manifest.RecursiveSchema

/-!
Contract: selected NIFS call instantiated by one V2 recursive manifest schema.

Assurance tier: implementation schema.

Owns profile/key/manifest identity matching and derives the full-claim
row-inclusion certificate from the recursive artifact itself.

Does not derive `verifierAccepted` from the opaque NIFS row family. That exact
generated-row refinement remains a release obligation.

Emits constraints: through `RecursiveManifestSchema.Artifact.programRows`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.RecursiveManifestSchema
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2

local instance concreteKOne : One Nightstream.SuperNeo.Concrete.K :=
  ⟨Nightstream.SuperNeo.Concrete.K.one⟩

/-- All values for one selected verifier call. The manifest supplies the link
rows and their inclusion. The final generated NIFS refinement must supply the
Boolean verifier result. -/
structure Call {widths : CompilerWidths}
    (artifact : Artifact widths) (selected : SelectedVerifier widths)
    (assignment : Nat → Nat) where
  identity : artifact.MatchesSelected selected
  claim : FullClaimNifsReceipt.Claim selected
  proof : selected.Proof
  output : selected.Output
  input : FixedBits.Word widths.totalBits
  claimCanonical : (Value.ofProtocolClaim claim).Canonical
  canonicalAssignment : ∀ column, assignment column < goldilocksP
  one : assignment 0 = 1
  placed : FullClaimEnvelopeRows.Placed artifact.layouts.fullClaim assignment
    (Value.ofProtocolClaim claim) input
  verifierAccepted : selected.verify proof input output = true

namespace Call

/-- Exact incoming, intermediate, and outgoing carry parser inputs for one
nonterminal recursive call.
The bit placement and verifier-owned header placement are explicit row/link
boundaries. The incoming state-hash link is derived from mandatory rows below. -/
structure CarryBlocks
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) where
  headers : FPrime.ChainHeaders Digest.Value
  priorBlock : MemoryCarryParser.Block
  intermediateBlock : MemoryCarryParser.Block
  outgoingBlock : MemoryCarryParser.Block
  priorBitsPlaced :
    PublicBitBlock.Placed artifact.layouts.priorMemoryCarry.publicBits
      assignment priorBlock
  intermediateBitsPlaced :
    PublicBitBlock.Placed
      artifact.layouts.intermediateMemoryCarry.publicBits assignment
      intermediateBlock
  outgoingBitsPlaced :
    PublicBitBlock.Placed artifact.layouts.memoryCarry.publicBits assignment
      outgoingBlock
  priorHeadersPlaced :
    MemoryCarryRows.HeadersPlaced artifact.layouts.priorMemoryCarry.carry
      assignment headers
  intermediateHeadersPlaced :
    MemoryCarryRows.HeadersPlaced
      artifact.layouts.intermediateMemoryCarry.carry assignment headers
  outgoingHeadersPlaced :
    MemoryCarryRows.HeadersPlaced artifact.layouts.memoryCarry.carry
      assignment headers

def CarryBlocks.priorValue
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks) : MemoryCarryCodec.Value :=
  MemoryCarryParser.decodedValue carry.priorBlock

def CarryBlocks.intermediateValue
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks) : MemoryCarryCodec.Value :=
  MemoryCarryParser.decodedValue carry.intermediateBlock

def CarryBlocks.outgoingValue
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks) : MemoryCarryCodec.Value :=
  MemoryCarryParser.decodedValue carry.outgoingBlock

/-- The prior carry parser result is a conclusion of the mandatory rows. -/
theorem CarryBlocks.priorAccepted
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryParser.parse carry.headers carry.priorBlock =
      some carry.priorValue :=
  MemoryCarryPublicRows.rows_force_parse call.canonicalAssignment call.one
    carry.priorBitsPlaced carry.priorHeadersPlaced
    (artifact.priorMemoryCarry_satisfied satisfies)

/-- The intermediate carry parser result is a conclusion of the mandatory
rows. -/
theorem CarryBlocks.intermediateAccepted
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryParser.parse carry.headers carry.intermediateBlock =
      some carry.intermediateValue :=
  MemoryCarryPublicRows.rows_force_parse call.canonicalAssignment call.one
    carry.intermediateBitsPlaced carry.intermediateHeadersPlaced
    (artifact.intermediateMemoryCarry_satisfied satisfies)

/-- The outgoing carry parser result is a conclusion of the mandatory rows. -/
theorem CarryBlocks.outgoingAccepted
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryParser.parse carry.headers carry.outgoingBlock =
      some carry.outgoingValue :=
  MemoryCarryPublicRows.rows_force_parse call.canonicalAssignment call.one
    carry.outgoingBitsPlaced carry.outgoingHeadersPlaced
    (artifact.memoryCarry_satisfied satisfies)

def memoryBlock
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) : MemoryClaimParser.Block :=
  MemoryClaimParser.blockOfClaim call.claim.memory
    call.claimCanonical.memoryCanonical

/-- The memory validator reads the exact memory slice of the same complete
claim that is linked to the NIFS input. -/
theorem memoryBlockPlaced
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) :
    PublicBitBlock.Placed artifact.layouts.memoryClaim.publicBits assignment
      call.memoryBlock := by
  intro index indexBound
  let global : Fin widths.totalBits :=
    ⟨Section.memory.bitOffset widths + index, by
      have fits := Section.slice_fits widths .memory
      have memoryWidth : Section.memory.width widths =
          Nightstream.Protocol.NebulaV2.MemoryWireGeometry.stepPublicBits := rfl
      rw [memoryWidth] at fits
      omega⟩
  have source := (call.placed global).1
  change assignment
      (artifact.layouts.memoryClaim.publicBitStart + index) = _
  rw [artifact.layoutsValid.memoryClaimFromFullClaim]
  rw [show artifact.layouts.fullClaim.claimBitStart +
      Section.memory.bitOffset widths + index =
      artifact.layouts.fullClaim.claimBitStart + global.val by
    simp [global, Nat.add_assoc]]
  rw [source]
  change (Value.ofProtocolClaim call.claim).encode.get _ =
    (MemoryClaimCodec.encode call.claim.memory)[index]'_
  let envelope := Value.ofProtocolClaim call.claim
  calc
    envelope.encode.get _ =
        ((envelope.encode.drop (Section.memory.bitOffset widths)).take
          (Section.memory.width widths))[index]'(by
            have widthBound : index < Section.memory.width widths := by
              simpa [Section.width] using indexBound
            have remainingBound :
                index < envelope.encode.length -
                  Section.memory.bitOffset widths := by
              rw [envelope.encode_length]
              have fits := Section.slice_fits widths .memory
              omega
            simp only [List.length_take, List.length_drop]
            exact lt_min widthBound remainingBound) := by
      simp [global, envelope, Section.width]
    _ = (envelope.sectionBits .memory)[index]'(by
          simpa [envelope.sectionBits_length] using indexBound) := by
      simpa only [envelope.encode_slice .memory]
    _ = (MemoryClaimCodec.encode call.claim.memory)[index]'_ := rfl

def toCircuitCall
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) :
    FullClaimNifsCall.CircuitCall selected artifact.programRows assignment where
  claim := call.claim
  proof := call.proof
  output := call.output
  input := call.input
  claimCanonical := call.claimCanonical
  link := artifact.fullClaimCallSite assignment
    (Value.ofProtocolClaim call.claim) call.input call.canonicalAssignment
    call.one call.placed
  verifierAccepted := call.verifierAccepted

/-- The unique exact receipt obtained from this satisfying recursive call.
The definition keeps the selected proof, output, and complete claim envelope
owned by one object for global delayed-order composition. -/
def receiptOfRows
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    FullClaimNifsReceipt.Receipt selected :=
  call.toCircuitCall.toReceipt satisfies

@[simp] theorem receiptOfRows_claim
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    (call.receiptOfRows satisfies).claim = call.claim :=
  rfl

theorem selected_identity_is_exact_v2
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) :
    artifact.profile = Profile.v2 ∧ selected.profile = Profile.v2 ∧
      artifact.verifierKeyDigest = selected.verifierKeyDigest ∧
      artifact.relationManifestDigest = selected.relationManifestDigest := by
  exact ⟨artifact.profileExact,
    call.identity.profile.symm.trans artifact.profileExact,
    call.identity.verifierKeyDigest,
    call.identity.relationManifestDigest⟩

theorem satisfying_manifest_binds_exact_nifs_input
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    call.input = (Value.ofProtocolClaim call.claim).block :=
  call.toCircuitCall.input_is_exact_full_claim satisfies

/-- For the exact V2 verifier selection, a satisfying recursive manifest
links every one of the 83,210 decoded paper-running fields to its generated
canonical field column. The theorem has no arbitrary decoder or verifier
premise. -/
theorem exactPaperInputMatchesRunningRows
    {logicalWidth : Nat}
    {publicFits : 540 <=
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (expectedApplication :
      Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding.PublicImage)
    (verifierKeyDigest relationManifestDigest : Digest.Value)
    (statementId : ProductConcreteNifs.StatementId)
    (productConfig : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (relationArtifact : ProductConcreteNifs.RelationArtifact logicalWidth
      publicFits)
    {artifact : Artifact ProductFullClaimDecoder.widths}
    {assignment : Nat → Nat}
    (call : Call artifact
      (ProductExactNifsConfiguration.selected expectedApplication
        verifierKeyDigest relationManifestDigest statementId productConfig
        relationArtifact) assignment)
    (applicationExact :
      (Value.ofProtocolClaim call.claim).applicationPublic =
        ProductFullClaimDecoder.applicationWord expectedApplication)
    (memoryCarrierExact :
      MemoryBoundCcsPublic.MemoryMatches
        (Value.ofProtocolClaim call.claim).ccsPublic call.claim.memory)
    (satisfies : Satisfies artifact.programRows assignment) :
    ExactNifsRunningRows.InputMatches
      (ProductPaperAlgebra.fullShapeContract logicalWidth publicFits)
      expectedApplication
      artifact.layouts.runningClaim assignment
      (Value.ofProtocolClaim call.claim) := by
  have bitsPlaced := ExactNifsRunningRows.bitsPlaced_of_fullClaim
    artifact.layouts.fullClaim artifact.layouts.runningClaim
    artifact.layoutsValid.runningClaimFromFullClaim call.placed
  exact ExactNifsRunningRows.input_matches_rows
    (ProductPaperAlgebra.fullShapeContract logicalWidth publicFits)
    expectedApplication artifact.layouts.runningClaim assignment
    (Value.ofProtocolClaim call.claim) call.claimCanonical applicationExact
    memoryCarrierExact call.canonicalAssignment call.one bitsPlaced
    (artifact.runningClaim_satisfied satisfies)

/-- The complete memory validator is linked to the memory suffix of the same
full claim accepted by the selected verifier. -/
theorem memoryClaimColumnsMatch
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryClaimRows.ParsedColumnsMatch artifact.layouts.memoryClaim assignment
      call.claim.memory := by
  have holds := artifact.owner_satisfied satisfies .memoryClaimValidation
  change Satisfies
    (MemoryClaimRows.rows artifact.layouts.memoryClaim) assignment at holds
  exact MemoryClaimRows.parsed_columns_match call.canonicalAssignment call.one
    call.memoryBlockPlaced holds
    (MemoryClaimParser.parse_blockOfClaim call.claim.memory
      call.claimCanonical.memoryCanonical)

/-- The incoming carry block has one exact typed row interpretation. This
uses the mandatory 7,094-row validator but does not yet prove that the prior
state digest authenticates `priorBlock`; that is a separate Poseidon2 link. -/
theorem priorCarryColumnsMatch
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryPublicRows.ParsedColumnsMatch
      artifact.layouts.priorMemoryCarry assignment carry.headers
      carry.priorValue := by
  exact MemoryCarryPublicRows.rows_force_parsed_columns_match
    call.canonicalAssignment call.one carry.priorBitsPlaced
    carry.priorHeadersPlaced (artifact.priorMemoryCarry_satisfied satisfies)

/-- The outgoing carry block has one exact typed row interpretation. -/
theorem outgoingCarryColumnsMatch
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryPublicRows.ParsedColumnsMatch artifact.layouts.memoryCarry
      assignment carry.headers carry.outgoingValue := by
  exact MemoryCarryPublicRows.rows_force_parsed_columns_match
    call.canonicalAssignment call.one carry.outgoingBitsPlaced
    carry.outgoingHeadersPlaced (artifact.memoryCarry_satisfied satisfies)

/-- The checked-step transition's intermediate carry has one exact typed row
interpretation before any segment-boundary reopening occurs. -/
theorem intermediateCarryColumnsMatch
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryPublicRows.ParsedColumnsMatch
      artifact.layouts.intermediateMemoryCarry assignment carry.headers
      carry.intermediateValue := by
  exact MemoryCarryPublicRows.rows_force_parsed_columns_match
    call.canonicalAssignment call.one carry.intermediateBitsPlaced
    carry.intermediateHeadersPlaced
    (artifact.intermediateMemoryCarry_satisfied satisfies)

theorem priorStateClaimPlaced
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) :
    FullClaimEnvelopeRows.Placed artifact.layouts.priorStateLink.fullClaim
      assignment (Value.ofProtocolClaim call.claim) call.input := by
  rw [artifact.layoutsValid.priorStateLinkUsesFullClaim]
  exact call.placed

theorem priorStateCarryPlaced
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (carry : call.CarryBlocks) :
    PublicBitBlock.Placed
      artifact.layouts.priorStateLink.stateOutput.hash.carry.frame.packing.publicBits
      assignment carry.priorBlock := by
  rw [artifact.layoutsValid.priorStateLinkUsesPriorCarryBits]
  exact carry.priorBitsPlaced

/-- The same outgoing carry bits validated by the memory owner are the bits
absorbed by the mandatory outgoing state hash. -/
theorem outgoingStateCarryPlaced
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (carry : call.CarryBlocks) :
    PublicBitBlock.Placed
      artifact.layouts.stateOutput.hash.carry.frame.packing.publicBits
      assignment carry.outgoingBlock := by
  rw [← artifact.layoutsValid.carryBitsSharedWithStateOutput]
  exact carry.outgoingBitsPlaced

/-- The complete claim selected for NIFS has the exact 540-coordinate carrier
of the locally recomputed prior state output and its exact memory suffix. -/
theorem priorStateCcsPublicExact
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    PriorStateLinkRows.CcsPublicExact
      artifact.layoutsValid.priorStateLinkValid assignment
      (Value.ofProtocolClaim call.claim) call.canonicalAssignment := by
  exact PriorStateLinkRows.claimCcsPublicExact
    artifact.layoutsValid.priorStateLinkValid call.canonicalAssignment call.one
    call.priorStateClaimPlaced
    (by
      rw [artifact.layoutsValid.priorStateLinkUsesMemoryClaim]
      exact call.memoryClaimColumnsMatch satisfies)
    (artifact.priorStateLink_satisfied satisfies)

/-- The four encoded lanes are the fixed two-stage Poseidon2 hash of the same
incoming carry bits and typed prior non-memory payload. -/
theorem priorStateDigestExact
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    ∀ lane : Fin 4,
      (PriorStateLinkRows.outputDigest artifact.layouts.priorStateLink
        assignment call.canonicalAssignment lane).val =
      StateOutputPoseidonRows.pureDigest
        (StateOutputAuthorityRows.fullFrame
          (StateOutputAuthorityRows.payload
            artifact.layouts.priorStateLink.stateOutput.authority assignment)
          (MemoryCarryPoseidonRows.carryDigest carry.priorBlock)) lane.val := by
  exact PriorStateLinkRows.outputDigest_eq_typedPriorState
    artifact.layoutsValid.priorStateLinkValid call.canonicalAssignment call.one
    (call.priorStateCarryPlaced carry)
    (artifact.priorStateLink_satisfied satisfies)

/-- The manifest's mandatory close rows derive both product-balance equations
for the exact products-after value in the selected-verifier claim. -/
theorem closingProductsBalanced
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (phaseClosed :
      assignment
          (artifact.layouts.intermediateMemoryCarry.carry.fieldColumn .phase) =
        0)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryProductBalanceRows.ConcreteBalanced
      call.claim.memory.productsAfter := by
  have parsed := call.memoryClaimColumnsMatch satisfies
  have balanceParsed : MemoryClaimRows.ParsedColumnsMatch
      artifact.layouts.memoryBalance.claim assignment call.claim.memory := by
    rw [artifact.layoutsValid.memoryBalanceUsesMemoryClaim]
    exact parsed
  have balancePhase :
      assignment artifact.layouts.memoryBalance.closePhaseColumn = 0 := by
    rw [artifact.layoutsValid.memoryBalanceUsesIntermediatePhase]
    exact phaseClosed
  exact MemoryProductBalanceRows.parsed_claim_balanced_of_rows call.one
    balancePhase balanceParsed (artifact.memoryBalance_satisfied satisfies)

/-- A satisfying recursive artifact consumes the memory suffix of the exact
complete fresh claim accepted by the selected NIFS verifier. The conclusion
is the independent semantic `FPrime.Consumes` relation. Product balance is
derived only when the parsed intermediate phase is closed. -/
theorem consumesExactAcceptedMemoryClaim
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    FPrime.Consumes MemoryProductBalanceRows.ConcreteBalanced
      (MemoryCarryParser.semanticCarry carry.priorValue
        (MemoryCarryParser.parse_value_canonical
          (carry.priorAccepted satisfies)).stepIndex)
      call.claim.memory
      (MemoryCarryParser.semanticCarry carry.intermediateValue
        (MemoryCarryParser.parse_value_canonical
          (carry.intermediateAccepted satisfies)).stepIndex) := by
  have priorParsed := call.priorCarryColumnsMatch carry satisfies
  have claimParsed := call.memoryClaimColumnsMatch satisfies
  have intermediateParsed := call.intermediateCarryColumnsMatch carry satisfies
  have transitionPrior : MemoryCarryPublicRows.ParsedColumnsMatch
      artifact.layouts.memoryTransition.before assignment carry.headers
      carry.priorValue := by
    rw [artifact.layoutsValid.transitionUsesPriorCarry]
    exact priorParsed
  have transitionClaim : MemoryClaimRows.ParsedColumnsMatch
      artifact.layouts.memoryTransition.claim assignment call.claim.memory := by
    rw [artifact.layoutsValid.transitionUsesMemoryClaim]
    exact claimParsed
  have transitionIntermediate : MemoryCarryPublicRows.ParsedColumnsMatch
      artifact.layouts.memoryTransition.after assignment carry.headers
      carry.intermediateValue := by
    rw [artifact.layoutsValid.transitionUsesIntermediateCarry]
    exact intermediateParsed
  have balanceOnClose : carry.intermediateValue.phase = .closed →
      MemoryProductBalanceRows.ConcreteBalanced
        call.claim.memory.productsAfter := by
    intro phaseClosed
    apply call.closingProductsBalanced _ satisfies
    calc
      assignment
          (artifact.layouts.intermediateMemoryCarry.carry.fieldColumn .phase) =
          carry.intermediateValue.fieldValue .phase :=
        intermediateParsed.placed .phase
      _ = 0 := by
        simp [MemoryCarryCodec.Value.fieldValue, MemoryCarryCodec.phaseValue,
          phaseClosed]
  have transition := MemoryTransitionSound.consumes_of_rows
    call.canonicalAssignment call.one transitionPrior transitionClaim
    transitionIntermediate (artifact.exactMemoryTransition_satisfied satisfies)
    balanceOnClose
  simpa only [MemoryCarryParser.parse_value_canonical] using transition

/-- After the exact selected claim is consumed, the mandatory continuation
rows either copy the active intermediate carry or reopen a closed segment in
the same nonterminal invocation. -/
theorem continuesExactIntermediateCarry
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (carry : call.CarryBlocks)
    (authority : MemoryOpenSegment.Authority)
    (authorityPlaced : MemoryOpenSegmentSound.AuthorityPlaced
      artifact.layouts.segmentContinuation.opening assignment authority)
    (satisfies : Satisfies artifact.programRows assignment) :
    AugmentedLifecycle.Continues
      (fun closed precommit activeAccessCount =>
        MemoryOpenSegment.derive authority closed precommit activeAccessCount)
      carry.headers
      (MemoryCarryParser.semanticCarry carry.intermediateValue
        (MemoryCarryParser.parse_value_canonical
          (carry.intermediateAccepted satisfies)).stepIndex)
      (MemoryCarryParser.semanticCarry carry.outgoingValue
        (MemoryCarryParser.parse_value_canonical
          (carry.outgoingAccepted satisfies)).stepIndex) := by
  have intermediateParsed := call.intermediateCarryColumnsMatch carry satisfies
  have outgoingParsed := call.outgoingCarryColumnsMatch carry satisfies
  have continuationIntermediate : MemoryCarryPublicRows.ParsedColumnsMatch
      artifact.layouts.segmentContinuation.intermediate assignment
      carry.headers carry.intermediateValue := by
    rw [artifact.layoutsValid.segmentContinuationUsesIntermediateCarry]
    exact intermediateParsed
  have continuationOutgoing : MemoryCarryPublicRows.ParsedColumnsMatch
      artifact.layouts.segmentContinuation.outgoing assignment carry.headers
      carry.outgoingValue := by
    rw [artifact.layoutsValid.segmentContinuationUsesOutgoingCarry]
    exact outgoingParsed
  have continuationIntermediateBits : PublicBitBlock.Placed
      artifact.layouts.segmentContinuation.intermediate.publicBits assignment
      carry.intermediateBlock := by
    rw [artifact.layoutsValid.segmentContinuationUsesIntermediateCarry]
    exact carry.intermediateBitsPlaced
  have continuationOutgoingBits : PublicBitBlock.Placed
      artifact.layouts.segmentContinuation.outgoing.publicBits assignment
      carry.outgoingBlock := by
    rw [artifact.layoutsValid.segmentContinuationUsesOutgoingCarry]
    exact carry.outgoingBitsPlaced
  simpa only [MemoryCarryParser.parse_value_canonical] using
    (MemorySegmentContinuationRows.sound
      artifact.layoutsValid.segmentContinuationValid call.canonicalAssignment
      call.one continuationIntermediateBits continuationOutgoingBits
      (carry.intermediateAccepted satisfies) (carry.outgoingAccepted satisfies)
      continuationIntermediate
      continuationOutgoing authorityPlaced
      (artifact.segmentContinuation_satisfied satisfies))

/-- One recursive-step result: the selected complete claim is linked to the
prior state, and its exact memory suffix is consumed. NIFS row refinement and
production control-row refinement remain explicit outer obligations. -/
theorem priorStateLinkedAndConsumesExactAcceptedMemoryClaim
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    PriorStateLinkRows.CcsPublicExact
        artifact.layoutsValid.priorStateLinkValid assignment
        (Value.ofProtocolClaim call.claim) call.canonicalAssignment ∧
      (∀ lane : Fin 4,
        (PriorStateLinkRows.outputDigest artifact.layouts.priorStateLink
          assignment call.canonicalAssignment lane).val =
        StateOutputPoseidonRows.pureDigest
          (StateOutputAuthorityRows.fullFrame
            (StateOutputAuthorityRows.payload
              artifact.layouts.priorStateLink.stateOutput.authority assignment)
            (MemoryCarryPoseidonRows.carryDigest carry.priorBlock)) lane.val) ∧
      FPrime.Consumes MemoryProductBalanceRows.ConcreteBalanced
        (MemoryCarryParser.semanticCarry carry.priorValue
          (MemoryCarryParser.parse_value_canonical
            (carry.priorAccepted satisfies)).stepIndex)
        call.claim.memory
        (MemoryCarryParser.semanticCarry carry.intermediateValue
          (MemoryCarryParser.parse_value_canonical
            (carry.intermediateAccepted satisfies)).stepIndex) := by
  exact ⟨call.priorStateCcsPublicExact satisfies,
    call.priorStateDigestExact carry satisfies,
    call.consumesExactAcceptedMemoryClaim carry satisfies⟩

/-- Complete nonterminal memory result of one satisfying recursive manifest.
The exact NIFS-selected claim updates the prior carry into an intermediate
carry, and the same invocation derives the required active outgoing carry.
No claim-free boundary invocation is present. -/
theorem priorStateLinkedConsumesAndContinues
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (carry : call.CarryBlocks)
    (authority : MemoryOpenSegment.Authority)
    (authorityPlaced : MemoryOpenSegmentSound.AuthorityPlaced
      artifact.layouts.segmentContinuation.opening assignment authority)
    (satisfies : Satisfies artifact.programRows assignment) :
    PriorStateLinkRows.CcsPublicExact
        artifact.layoutsValid.priorStateLinkValid assignment
        (Value.ofProtocolClaim call.claim) call.canonicalAssignment ∧
      (∀ lane : Fin 4,
        (PriorStateLinkRows.outputDigest artifact.layouts.priorStateLink
          assignment call.canonicalAssignment lane).val =
        StateOutputPoseidonRows.pureDigest
          (StateOutputAuthorityRows.fullFrame
            (StateOutputAuthorityRows.payload
              artifact.layouts.priorStateLink.stateOutput.authority assignment)
            (MemoryCarryPoseidonRows.carryDigest carry.priorBlock)) lane.val) ∧
      FPrime.Consumes MemoryProductBalanceRows.ConcreteBalanced
        (MemoryCarryParser.semanticCarry carry.priorValue
          (MemoryCarryParser.parse_value_canonical
            (carry.priorAccepted satisfies)).stepIndex)
        call.claim.memory
        (MemoryCarryParser.semanticCarry carry.intermediateValue
          (MemoryCarryParser.parse_value_canonical
            (carry.intermediateAccepted satisfies)).stepIndex) ∧
      AugmentedLifecycle.Continues
        (fun closed precommit activeAccessCount =>
          MemoryOpenSegment.derive authority closed precommit
            activeAccessCount)
        carry.headers
        (MemoryCarryParser.semanticCarry carry.intermediateValue
          (MemoryCarryParser.parse_value_canonical
            (carry.intermediateAccepted satisfies)).stepIndex)
        (MemoryCarryParser.semanticCarry carry.outgoingValue
          (MemoryCarryParser.parse_value_canonical
            (carry.outgoingAccepted satisfies)).stepIndex) := by
  exact ⟨call.priorStateCcsPublicExact satisfies,
    call.priorStateDigestExact carry satisfies,
    call.consumesExactAcceptedMemoryClaim carry satisfies,
    call.continuesExactIntermediateCarry carry authority authorityPlaced
      satisfies⟩

/-- The typed carry headers are not caller authority. Satisfying rows derive
them from the fixed V2 header frames and the verifier-owned plan digest. -/
theorem chainHeadersExact
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    (∀ lane : Fin 4,
      (carry.headers.operations.lanes lane).val =
        CompactChainPoseidonRows.pureHash
          (.header .operations artifact.seedManifest.profile
            artifact.seedManifest.plan) lane.val) ∧
      (∀ lane : Fin 4,
        (carry.headers.memory.lanes lane).val =
          CompactChainPoseidonRows.pureHash
            (.header .memory artifact.seedManifest.profile
              artifact.seedManifest.plan) lane.val) := by
  have outputs := CompactChainHeaderRows.outputs_exact
    artifact.compactHeadersValid call.canonicalAssignment call.one
    (artifact.compactHeaders_satisfied satisfies)
  constructor
  · intro lane
    calc
      (carry.headers.operations.lanes lane).val =
          assignment
            (artifact.layouts.priorMemoryCarry.carry.headerColumn
              .operations lane) := by
        symm
        simpa [FPrime.ChainHeaders.roots, MemoryClaimCodec.rootValue] using
          carry.priorHeadersPlaced .operations lane
      _ = assignment
          (artifact.layouts.compactHeaders.operations.digestColumn lane) := by
        rw [artifact.layoutsValid.compactOperationsHeaderUsesPriorCarry]
      _ = CompactChainPoseidonRows.pureHash
          (.header .operations artifact.seedManifest.profile
            artifact.seedManifest.plan) lane.val :=
        outputs.1 lane
  · intro lane
    calc
      (carry.headers.memory.lanes lane).val =
          assignment
            (artifact.layouts.priorMemoryCarry.carry.headerColumn
              .initialSnapshot lane) := by
        symm
        simpa [FPrime.ChainHeaders.roots, MemoryClaimCodec.rootValue] using
          carry.priorHeadersPlaced .initialSnapshot lane
      _ = assignment
          (artifact.layouts.compactHeaders.memory.digestColumn lane) := by
        rw [artifact.layoutsValid.compactMemoryHeaderUsesPriorInitial]
      _ = CompactChainPoseidonRows.pureHash
          (.header .memory artifact.seedManifest.profile
            artifact.seedManifest.plan) lane.val :=
        outputs.2 lane

/-- The compact-chain bundle decoder reads the exact mandatory bundle slice
of the same complete claim selected by NIFS. -/
theorem compactBundleBitsPlaced
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) :
    CommitmentBundleFieldRows.BitsPlaced
      artifact.layouts.compactChain.bundleFields assignment
      call.claim.commitmentBundle := by
  intro index
  let sectionIndex : Fin (Section.commitmentBundle.width widths) :=
    ⟨index.val, by simpa [Section.width] using index.isLt⟩
  let global : Fin widths.totalBits :=
    ⟨Section.commitmentBundle.bitOffset widths + index.val, by
      have fits := Section.slice_fits widths .commitmentBundle
      have bundleWidth : Section.commitmentBundle.width widths =
          MemoryWireGeometry.mandatoryBundleBits := rfl
      rw [bundleWidth] at fits
      omega⟩
  have source := (call.placed global).1
  rw [artifact.layoutsValid.compactChainBundleFromFullClaim]
  rw [show artifact.layouts.fullClaim.claimBitStart +
      Section.commitmentBundle.bitOffset widths + index.val =
      artifact.layouts.fullClaim.claimBitStart + global.val by
    simp [global, Nat.add_assoc]]
  rw [source]
  simpa only [FullClaimEnvelopeRows.envelopeBit,
      BundleForwardingRows.bitAt, Value.sectionBits] using
    (Value.ofProtocolClaim call.claim).encode_get_section
      .commitmentBundle sectionIndex

/-- All three compact sequence roots are computed from the exact bundle and
the exact runtime step index in the NIFS-selected claim. -/
theorem compactChainRootsExact
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    CompactCheckedStepChainRows.LaneExact artifact.seedManifest .operations
        call.claim.memory.stepIndex artifact.layouts.compactChain.operations
        assignment call.canonicalAssignment
        (call.claim.commitmentBundle .operations)
        call.claim.memory.dSeenBefore.operations
        call.claim.memory.dSeenAfter.operations ∧
      CompactCheckedStepChainRows.LaneExact artifact.seedManifest .memory
        call.claim.memory.stepIndex
        artifact.layouts.compactChain.initialSnapshot assignment
        call.canonicalAssignment
        (call.claim.commitmentBundle .initialSnapshot)
        call.claim.memory.dSeenBefore.initialSnapshot
        call.claim.memory.dSeenAfter.initialSnapshot ∧
      CompactCheckedStepChainRows.LaneExact artifact.seedManifest .memory
        call.claim.memory.stepIndex
        artifact.layouts.compactChain.finalSnapshot assignment
        call.canonicalAssignment
        (call.claim.commitmentBundle .finalSnapshot)
        call.claim.memory.dSeenBefore.finalSnapshot
        call.claim.memory.dSeenAfter.finalSnapshot := by
  have parsed : MemoryClaimRows.ParsedColumnsMatch
      artifact.layouts.compactChain.memoryClaim assignment
      call.claim.memory := by
    rw [artifact.layoutsValid.compactChainUsesMemoryClaim]
    exact call.memoryClaimColumnsMatch satisfies
  exact CompactCheckedStepChainRows.all_lanes_exact
    artifact.compactChainValid call.canonicalAssignment call.one
    call.compactBundleBitsPlaced parsed
    (artifact.compactChain_satisfied satisfies)

/-- A typed output bundle interpretation for the forwarding columns. This
predicate contains no equality to the claim bundle. -/
def BundleOutputPlaced
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (output : CommitmentBundleCodec.Value) : Prop :=
  ∀ index : Fin MemoryWireGeometry.mandatoryBundleBits,
    assignment
        (artifact.layouts.bundleForwarding.outputStart + index.val) =
      BundleForwardingRows.bitAt output index

/-- The forwarding block input reads the exact mandatory bundle section of
the complete selected-verifier claim. -/
theorem bundleInputPlaced
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) :
    ∀ index : Fin MemoryWireGeometry.mandatoryBundleBits,
      assignment
          (artifact.layouts.bundleForwarding.inputStart + index.val) =
        BundleForwardingRows.bitAt call.claim.commitmentBundle index := by
  intro index
  let sectionIndex : Fin (Section.commitmentBundle.width widths) :=
    ⟨index.val, by simpa [Section.width] using index.isLt⟩
  let global : Fin widths.totalBits :=
    ⟨Section.commitmentBundle.bitOffset widths + index.val, by
      have fits := Section.slice_fits widths .commitmentBundle
      have bundleWidth : Section.commitmentBundle.width widths =
          MemoryWireGeometry.mandatoryBundleBits := rfl
      rw [bundleWidth] at fits
      omega⟩
  have source := (call.placed global).1
  rw [artifact.layoutsValid.bundleInputFromFullClaim]
  rw [show artifact.layouts.fullClaim.claimBitStart +
      Section.commitmentBundle.bitOffset widths + index.val =
      artifact.layouts.fullClaim.claimBitStart + global.val by
    simp [global, Nat.add_assoc]]
  rw [source]
  simpa only [FullClaimEnvelopeRows.envelopeBit,
      BundleForwardingRows.bitAt, Value.sectionBits] using
    (Value.ofProtocolClaim call.claim).encode_get_section
      .commitmentBundle sectionIndex

/-- Satisfying the manifest forwards the exact four-component commitment
bundle from the complete selected-verifier claim. -/
theorem forwardedBundleExact
    {widths : CompilerWidths} {artifact : Artifact widths}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (output : CommitmentBundleCodec.Value)
    (outputPlaced : call.BundleOutputPlaced output)
    (satisfies : Satisfies artifact.programRows assignment) :
    output = call.claim.commitmentBundle := by
  have placed : BundleForwardingRows.Placed
      artifact.layouts.bundleForwarding assignment
      call.claim.commitmentBundle output := by
    intro index
    exact ⟨call.bundleInputPlaced index, outputPlaced index⟩
  have holds := artifact.owner_satisfied satisfies .bundleForwarding
  change BundleForwardingRows.RowsHold
    artifact.layouts.bundleForwarding assignment at holds
  exact BundleForwardingRows.exact_bundle_forwarding call.canonicalAssignment
    call.one placed holds

end Call

end Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall
