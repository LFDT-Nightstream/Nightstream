import Nightstream.Implementation.NebulaV2.FPrime.Claim.EnvelopeRows
import Nightstream.Implementation.NebulaV2.NIFS.Core.ExactConfiguration
import Nightstream.Implementation.NebulaV2.NIFS.Running.RunningFieldRows

/-!
Contract: exact generated-row bridge from a complete accepted V2 claim to the
paper-NIFS running input.

Assurance tier: implementation-to-protocol bridge.

Owns exact recursive-state section placement, strict canonical bit-to-field
rows, and equality between every paper-verifier input field and its generated
field column.

Does not own the paper-NIFS verifier arithmetic rows, setup digest
recomputation, cryptographic soundness, Rust conformance, or row capacity.

Emits constraints: through `ProductNifsRunningFieldRows.rows`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ExactNifsRunningRows

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.ProductFullClaimDecoder
open Nightstream.Implementation.NebulaV2.ProductNifsCodec
open Nightstream.Implementation.NebulaV2.ProductPaperNifsSelection
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
open Nightstream.SuperNeo.Concrete.Phi81Relation

/-- Exact semantic result of linking the selected paper input to generated
running-field columns. -/
def InputMatches
    {fullShape : Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (layout : ProductNifsRunningFieldRows.Layout)
    (assignment : Nat → Nat)
    (value : Value ProductFullClaimDecoder.widths) : Prop :=
  ∃ fields : ProductNifsRunningParser.Fields,
    ∃ fresh : ProductNifsCodec.Fresh fullShape,
      ProductFullClaimParser.decode contract expectedApplication value.block =
        some
          (ProductNifsRunningParser.runningOfFields contract fields, fresh) ∧
      ProductNifsFieldParser.parse value.recursiveState = some fields ∧
      ProductNifsRunningFieldRows.ParsedColumnsMatch layout assignment fields

/-- Exact row-derived paper input. The premises contain only fixed-envelope
well-formedness and generated row satisfaction. In particular, they contain
no paper verifier result. -/
theorem input_matches_rows
    {fullShape : Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (layout : ProductNifsRunningFieldRows.Layout)
    (assignment : Nat → Nat)
    (value : Value ProductFullClaimDecoder.widths)
    (canonical : value.Canonical)
    (applicationExact : value.applicationPublic =
      ProductFullClaimDecoder.applicationWord expectedApplication)
    (memoryCarrierExact :
      MemoryBoundCcsPublic.MemoryMatches value.ccsPublic value.memory)
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (bitsPlaced : ProductNifsRunningFieldRows.BitsPlaced layout assignment
      value.recursiveState)
    (rowsHold : Satisfies (ProductNifsRunningFieldRows.rows layout)
      assignment) :
    InputMatches contract expectedApplication layout assignment value := by
  rcases ProductNifsRunningFieldRows.parse_from_rows canonicalAssignment one
      bitsPlaced rowsHold with ⟨fields, fieldsParsed, columnsMatch⟩
  let running : ProductNifsCodec.Running fullShape :=
    ProductNifsRunningParser.runningOfFields contract fields
  have runningParsed :
      ProductNifsRunningParser.parse contract value.recursiveState =
        some running := by
    simp [ProductNifsRunningParser.parse, fieldsParsed, running]
  let wellFormed : ProductFullClaimParser.WellFormed contract
      expectedApplication value :=
    { canonical := canonical
      applicationExact := applicationExact
      memoryCarrierExact := memoryCarrierExact
      runningDecodes := ⟨running, runningParsed⟩ }
  let fresh := ProductFullClaimDecoder.freshOfValue contract value
  have decoded := ProductFullClaimParser.decode_block contract
    expectedApplication value wellFormed
  have runningExact :
      ProductFullClaimParser.runningOfValue contract value = running :=
    ProductFullClaimParser.runningOfValue_eq contract value runningParsed
  refine ⟨fields, fresh, ?_, fieldsParsed, columnsMatch⟩
  simpa [running, fresh, runningExact] using decoded

/-- Exact placement of the recursive-state section follows from placement of
the same complete claim and one manifest layout equality. -/
theorem bitsPlaced_of_fullClaim
    {assignment : Nat → Nat}
    {value : Value ProductFullClaimDecoder.widths}
    {input : FixedBits.Word ProductFullClaimDecoder.widths.totalBits}
    (fullLayout : FullClaimEnvelopeRows.Layout
      ProductFullClaimDecoder.widths)
    (runningLayout : ProductNifsRunningFieldRows.Layout)
    (linked : runningLayout.publicBitStart =
      fullLayout.claimBitStart +
        Section.recursiveState.bitOffset ProductFullClaimDecoder.widths)
    (placed : FullClaimEnvelopeRows.Placed fullLayout assignment value input) :
    ProductNifsRunningFieldRows.BitsPlaced runningLayout assignment
      value.recursiveState := by
  intro index
  let sectionIndex : Fin
      (Section.recursiveState.width ProductFullClaimDecoder.widths) :=
    ⟨index.val, by simpa [Section.width, ProductFullClaimDecoder.widths] using
      index.isLt⟩
  let global : Fin ProductFullClaimDecoder.widths.totalBits :=
    ⟨Section.recursiveState.bitOffset ProductFullClaimDecoder.widths +
        sectionIndex.val,
      by
        have fits := Section.slice_fits ProductFullClaimDecoder.widths
          Section.recursiveState
        have sectionBound := sectionIndex.isLt
        omega⟩
  calc
    assignment (runningLayout.publicBitStart + index.val) =
        assignment
          (fullLayout.claimBitStart + global.val) := by
      rw [linked]
      simp [global, sectionIndex, Nat.add_assoc]
    _ = FullClaimEnvelopeRows.envelopeBit value global :=
      (placed global).1
    _ = value.recursiveState.val.get
        ⟨index.val, by
          rw [value.recursiveState.property.1]
          exact index.isLt⟩ := by
      simpa [FullClaimEnvelopeRows.envelopeBit, global, sectionIndex] using
        value.encode_get_section Section.recursiveState sectionIndex

/-- The exact accepted paper input and every generated running-field column
come from one strict parser result. -/
theorem selected_input_matches_rows
    {logicalWidth : Nat}
    {publicFits : 540 <=
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
        logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (expectedApplication : PublicImage)
    (verifierKeyDigest relationManifestDigest : Digest.Value)
    (statementId : ProductConcreteNifs.StatementId)
    (productConfig : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (relationArtifact : ProductConcreteNifs.RelationArtifact logicalWidth
      publicFits)
    (layout : ProductNifsRunningFieldRows.Layout)
    (assignment : Nat → Nat)
    (claim : Claim
      (ProductExactNifsConfiguration.selected expectedApplication
        verifierKeyDigest relationManifestDigest statementId productConfig
        relationArtifact))
    (proof : Paper.Proof ProductNifsCodec.shape 9)
    (output : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (input : FixedBits.Word ProductFullClaimDecoder.widths.totalBits)
    (claimCanonical : (Value.ofProtocolClaim claim).Canonical)
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (verifierAccepted :
      (ProductExactNifsConfiguration.selected expectedApplication
          verifierKeyDigest relationManifestDigest statementId productConfig
          relationArtifact).verify
        proof input output = true)
    (inputExact : input = (Value.ofProtocolClaim claim).block)
    (bitsPlaced : ProductNifsRunningFieldRows.BitsPlaced layout assignment
      (Value.ofProtocolClaim claim).recursiveState)
    (rowsHold : Satisfies (ProductNifsRunningFieldRows.rows layout)
      assignment) :
    InputMatches
      (ProductPaperAlgebra.fullShapeContract logicalWidth publicFits)
      expectedApplication layout assignment
      (Value.ofProtocolClaim claim) := by
  have accepted : VerifyClaim
      (ProductExactNifsConfiguration.selected expectedApplication
        verifierKeyDigest relationManifestDigest statementId productConfig
        relationArtifact) (proof, output) claim := by
    refine ⟨claimCanonical, ?_⟩
    rw [← inputExact]
    exact verifierAccepted
  rcases ProductExactNifsConfiguration.accepted_input_has_exact_fields
      expectedApplication verifierKeyDigest relationManifestDigest statementId
      productConfig relationArtifact claim proof output accepted with
    ⟨fields, fresh, decoded, fieldsParsed⟩
  refine ⟨fields, fresh, decoded, fieldsParsed, ?_⟩
  exact ProductNifsRunningFieldRows.parsed_columns_match canonicalAssignment
    one bitsPlaced rowsHold fieldsParsed

end Nightstream.Implementation.NebulaV2.ExactNifsRunningRows
