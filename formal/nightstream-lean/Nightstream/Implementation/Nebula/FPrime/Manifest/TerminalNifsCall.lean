import Nightstream.Implementation.Nebula.FPrime.Claim.NifsCall
import Nightstream.Implementation.Nebula.NIFS.Running.ExactRows
import Nightstream.Implementation.Nebula.Memory.Transition.OpenSegmentSound
import Nightstream.Implementation.Nebula.Memory.Transition.TransitionSound
import Nightstream.Implementation.Nebula.FPrime.Manifest.TerminalSchema

/-!
Contract: selected trailing NIFS call instantiated by one V2 terminal manifest.

Assurance tier: implementation schema.

Owns the exact full-claim link, prior-state link, prior and intermediate carry
parsing, delayed memory transition, product balance, and row-derived terminal
closure. It constructs the selected full-claim receipt only after row
satisfaction proves that the verifier input is the complete claim envelope.

Does not derive NIFS acceptance from the opaque generated verifier rows, does
not prove accumulator folding, and does not prove the terminal CCS or public
result relation. Those remain explicit release obligations.

Emits constraints: through `TerminalManifestSchema.Artifact.programRows`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.TerminalManifestNifsCall

open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimNifsReceipt
open Nightstream.Implementation.Nebula.TerminalManifestSchema
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.SuperNeo.Concrete.Phi81Relation

local instance concreteKOne : One Nightstream.SuperNeo.Concrete.K :=
  ⟨Nightstream.SuperNeo.Concrete.K.one⟩

/-- All values for one selected terminal NIFS call. `verifierAccepted` is the
named generated-NIFS refinement boundary; the exact accepted input is derived
from mandatory rows below. -/
structure Call
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape)
    (selected : SelectedVerifier widths) (assignment : Nat → Nat) where
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

/-- Exact incoming and intermediate carry parser inputs. A terminal call has
no outgoing active carry block. -/
structure CarryBlocks
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) where
  headers : FPrime.ChainHeaders Digest.Value
  priorBlock : MemoryCarryParser.Block
  intermediateBlock : MemoryCarryParser.Block
  priorBitsPlaced :
    PublicBitBlock.Placed artifact.layouts.priorMemoryCarry.publicBits
      assignment priorBlock
  intermediateBitsPlaced :
    PublicBitBlock.Placed
      artifact.layouts.intermediateMemoryCarry.publicBits assignment
      intermediateBlock
  priorHeadersPlaced :
    MemoryCarryRows.HeadersPlaced artifact.layouts.priorMemoryCarry.carry
      assignment headers
  intermediateHeadersPlaced :
    MemoryCarryRows.HeadersPlaced
      artifact.layouts.intermediateMemoryCarry.carry assignment headers

def CarryBlocks.priorValue
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks) : MemoryCarryCodec.Value :=
  MemoryCarryParser.decodedValue carry.priorBlock

def CarryBlocks.intermediateValue
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks) : MemoryCarryCodec.Value :=
  MemoryCarryParser.decodedValue carry.intermediateBlock

/-- The terminal prior-carry parser result is forced by the mandatory rows. -/
theorem CarryBlocks.priorAccepted
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryParser.parse carry.headers carry.priorBlock =
      some carry.priorValue :=
  MemoryCarryPublicRows.rows_force_parse call.canonicalAssignment call.one
    carry.priorBitsPlaced carry.priorHeadersPlaced
    (artifact.priorCarry_satisfied satisfies)

/-- The terminal intermediate-carry parser result is forced by the mandatory
rows. -/
theorem CarryBlocks.intermediateAccepted
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    {call : Call artifact selected assignment}
    (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryParser.parse carry.headers carry.intermediateBlock =
      some carry.intermediateValue :=
  MemoryCarryPublicRows.rows_force_parse call.canonicalAssignment call.one
    carry.intermediateBitsPlaced carry.intermediateHeadersPlaced
    (artifact.intermediateCarry_satisfied satisfies)

def memoryBlock
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) : MemoryClaimParser.Block :=
  MemoryClaimParser.blockOfClaim call.claim.memory
    call.claimCanonical.memoryCanonical

/-- The memory validator reads the exact memory section of the complete claim
that is linked to the selected verifier input. -/
theorem memoryBlockPlaced
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) :
    PublicBitBlock.Placed artifact.layouts.memoryClaim.publicBits assignment
      call.memoryBlock := by
  intro index indexBound
  let global : Fin widths.totalBits :=
    ⟨Section.memory.bitOffset widths + index, by
      have fits := Section.slice_fits widths .memory
      have memoryWidth : Section.memory.width widths =
          MemoryWireGeometry.stepPublicBits := rfl
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
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
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

theorem selected_identity_is_exact_v2
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
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
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    call.input = (Value.ofProtocolClaim call.claim).block :=
  call.toCircuitCall.input_is_exact_full_claim satisfies

/-- For the exact V2 verifier selection, a satisfying terminal manifest links
every decoded trailing paper-running field to its generated canonical field
column. -/
theorem exactPaperInputMatchesRunningRows
    {logicalWidth : Nat}
    {publicFits : 540 <=
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    {operationsShape snapshotShape : Shape}
    (expectedApplication :
      Nightstream.Protocol.Nebula.WasmPublicStatementEncoding.PublicImage)
    (verifierKeyDigest relationManifestDigest : Digest.Value)
    (statementId : ProductConcreteNifs.StatementId)
    (productConfig : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (relationArtifact : ProductConcreteNifs.RelationArtifact logicalWidth
      publicFits)
    {artifact : Artifact ProductFullClaimDecoder.widths
      (ProductPaperAlgebra.FullShape logicalWidth publicFits) operationsShape
      snapshotShape}
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

/-- A selected receipt exists only after the exact full-claim link has been
checked. It cannot pair the verifier Boolean with another claim envelope. -/
def receiptOfRows
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    FullClaimNifsReceipt.Receipt selected where
  claim := call.claim
  proof := call.proof
  output := call.output
  accepted := by
    refine ⟨call.claimCanonical, ?_⟩
    rw [← call.satisfying_manifest_binds_exact_nifs_input satisfies]
    exact call.verifierAccepted

theorem memoryClaimColumnsMatch
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryClaimRows.ParsedColumnsMatch artifact.layouts.memoryClaim assignment
      call.claim.memory := by
  exact MemoryClaimRows.parsed_columns_match call.canonicalAssignment call.one
    call.memoryBlockPlaced (artifact.memoryClaim_satisfied satisfies)
    (MemoryClaimParser.parse_blockOfClaim call.claim.memory
      call.claimCanonical.memoryCanonical)

theorem priorCarryColumnsMatch
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryPublicRows.ParsedColumnsMatch
      artifact.layouts.priorMemoryCarry assignment carry.headers
      carry.priorValue := by
  exact MemoryCarryPublicRows.rows_force_parsed_columns_match
    call.canonicalAssignment call.one carry.priorBitsPlaced
    carry.priorHeadersPlaced (artifact.priorCarry_satisfied satisfies)

theorem intermediateCarryColumnsMatch
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    MemoryCarryPublicRows.ParsedColumnsMatch
      artifact.layouts.intermediateMemoryCarry assignment carry.headers
      carry.intermediateValue := by
  exact MemoryCarryPublicRows.rows_force_parsed_columns_match
    call.canonicalAssignment call.one carry.intermediateBitsPlaced
    carry.intermediateHeadersPlaced
    (artifact.intermediateCarry_satisfied satisfies)

/-- The terminal prior-state owner reads the exact complete trailing claim
selected by the terminal NIFS call. This placement is public because the
lifetime authority theorem must use this same claim, not a second envelope. -/
theorem priorStateClaimPlaced
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) :
    FullClaimEnvelopeRows.Placed artifact.layouts.priorStateLink.fullClaim
      assignment (Value.ofProtocolClaim call.claim) call.input := by
  rw [artifact.layoutsValid.priorStateLinkUsesFullClaim]
  exact call.placed

/-- The terminal prior-state hash reads the exact parsed incoming carry block
that the delayed memory transition consumes. -/
theorem priorStateCarryPlaced
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) (carry : call.CarryBlocks) :
    PublicBitBlock.Placed
      artifact.layouts.priorStateLink.stateOutput.hash.carry.frame.packing.publicBits
      assignment carry.priorBlock := by
  rw [artifact.layoutsValid.priorStateLinkUsesPriorCarryBits]
  exact carry.priorBitsPlaced

theorem priorStateCcsPublicExact
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
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

theorem priorStateDigestExact
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) (carry : call.CarryBlocks)
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

theorem closingProductsBalanced
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
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

/-- The terminal close row derives closure. There is no caller-supplied phase
premise and no continuation that can reopen the segment. -/
theorem intermediatePhaseClosed
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    carry.intermediateValue.phase = .closed := by
  have parsed := call.intermediateCarryColumnsMatch carry satisfies
  have terminalParsed : MemoryCarryPublicRows.ParsedColumnsMatch
      artifact.layouts.terminalClosed.carry assignment carry.headers
      carry.intermediateValue := by
    rw [artifact.layoutsValid.terminalClosedUsesIntermediateCarry]
    exact parsed
  exact TerminalClosedCarryRows.parsed_phase_closed call.canonicalAssignment
    call.one terminalParsed (artifact.terminalClosed_satisfied satisfies)

/-- The satisfying terminal artifact consumes the exact trailing claim to the
canonical closed carry. Product balance follows from the same rows. -/
theorem consumesExactAcceptedTrailingClaim
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    FPrime.Consumes MemoryProductBalanceRows.ConcreteBalanced
      (MemoryCarryParser.semanticCarry carry.priorValue
        (MemoryCarryParser.parse_value_canonical
          (carry.priorAccepted satisfies)).stepIndex)
      call.claim.memory
      (.closed (MemoryOpenSegmentSound.closedOfWire carry.intermediateValue)) := by
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
  have phaseClosed := call.intermediatePhaseClosed carry satisfies
  simpa only [MemoryCarryParser.parse_value_canonical,
    MemoryCarryParser.semanticCarry, phaseClosed,
    MemoryOpenSegmentSound.closedOfWire] using transition

/-- Exact selected full-claim transition used by the independent terminal
semantics. The receipt and the consumed memory suffix are one record. -/
theorem selectedTransitionToClosed
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) (carry : call.CarryBlocks)
    (satisfies : Satisfies artifact.programRows assignment) :
    FullClaimNifsReceipt.Transition selected
      MemoryProductBalanceRows.ConcreteBalanced
      (MemoryCarryParser.semanticCarry carry.priorValue
        (MemoryCarryParser.parse_value_canonical
          (carry.priorAccepted satisfies)).stepIndex)
      (call.receiptOfRows satisfies)
      (.closed (MemoryOpenSegmentSound.closedOfWire carry.intermediateValue)) :=
  { consumes := call.consumesExactAcceptedTrailingClaim carry satisfies }

/-- The terminal manifest establishes both the prior-state authority link and
the exact delayed transition to closed state. It does not assume an execution
conclusion. -/
theorem priorStateLinkedAndClosesExactAcceptedTrailingClaim
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {selected : SelectedVerifier widths} {assignment : Nat → Nat}
    (call : Call artifact selected assignment) (carry : call.CarryBlocks)
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
      FullClaimNifsReceipt.Transition selected
        MemoryProductBalanceRows.ConcreteBalanced
        (MemoryCarryParser.semanticCarry carry.priorValue
          (MemoryCarryParser.parse_value_canonical
            (carry.priorAccepted satisfies)).stepIndex)
        (call.receiptOfRows satisfies)
        (.closed
          (MemoryOpenSegmentSound.closedOfWire carry.intermediateValue)) := by
  exact ⟨call.priorStateCcsPublicExact satisfies,
    call.priorStateDigestExact carry satisfies,
    call.selectedTransitionToClosed carry satisfies⟩

end Call

end Nightstream.Implementation.Nebula.TerminalManifestNifsCall
