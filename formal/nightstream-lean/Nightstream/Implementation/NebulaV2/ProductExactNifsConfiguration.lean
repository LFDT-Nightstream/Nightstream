import Nightstream.Implementation.NebulaV2.ProductConcreteNifs
import Nightstream.Implementation.NebulaV2.ProductFullClaimParser

/-!
Contract: the only V2 paper-NIFS configuration accepted by the integration.

Assurance tier: executable semantic verifier selection.

Owns the exact full-claim widths, the fixed one-fresh/fourteen-running product
shape, the strict complete-claim parser, and construction of the selected
paper verifier from the one concrete V2 key and two setup digests. A caller
cannot replace the transcript, relation algebra, commitment map, degree, or
challenge schedule.

Does not own generated verifier rows, verifier-key digest recomputation,
paper-NIFS cryptographic soundness, Rust conformance, or recursive-size
closure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductExactNifsConfiguration

open Nightstream.Implementation.NebulaV2.ProductFullClaimDecoder
open Nightstream.Implementation.NebulaV2.ProductNifsCodec
open Nightstream.Implementation.NebulaV2.ProductPaperNifsSelection
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Exact production configuration constructor. There is no decoder or paper
key argument. The constructor installs the strict complete V2 claim parser
and derives the only paper key from the concrete V2 inputs. -/
noncomputable def configuration
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (expectedApplication : PublicImage)
    (verifierKeyDigest relationManifestDigest : Digest.Value)
    (statementId : ProductConcreteNifs.StatementId)
    (productConfig : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (relationArtifact : ProductConcreteNifs.RelationArtifact logicalWidth
      publicFits) :
    Configuration ProductFullClaimDecoder.widths ProductConcreteNifs.State
      (ProductPaperAlgebra.FullShape logicalWidth publicFits)
      ProductNifsCodec.shape (Phi81CarrierLayout.carrierWidth logicalWidth)
      (Phi81ColumnLayout.blockCount
        (Phi81CarrierLayout.carrierWidth logicalWidth)) 9 where
  verifierKeyDigest := verifierKeyDigest
  relationManifestDigest := relationManifestDigest
  key := ProductConcreteNifs.key statementId productConfig relationArtifact
  decoder := ProductFullClaimParser.claimDecoder
    (ProductPaperAlgebra.fullShapeContract logicalWidth publicFits)
    expectedApplication
  samplerCheck := fun running fresh proof =>
    ProductPoseidon2.samplerSucceeded
      (((ProductConcreteNifs.key statementId productConfig relationArtifact
        ).piCcsExecution running fresh proof).outgoingState)

/-- The selected verifier for the exact V2 configuration. -/
noncomputable def selected
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (expectedApplication : PublicImage)
    (verifierKeyDigest relationManifestDigest : Digest.Value)
    (statementId : ProductConcreteNifs.StatementId)
    (productConfig : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (relationArtifact : ProductConcreteNifs.RelationArtifact logicalWidth
      publicFits) :=
  ProductPaperNifsSelection.selected
    (configuration expectedApplication verifierKeyDigest relationManifestDigest
      statementId productConfig relationArtifact)

theorem configuration_decoder_exact
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (expectedApplication : PublicImage)
    (verifierKeyDigest relationManifestDigest : Digest.Value)
    (statementId : ProductConcreteNifs.StatementId)
    (productConfig : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (relationArtifact : ProductConcreteNifs.RelationArtifact logicalWidth
      publicFits) :
    (configuration expectedApplication verifierKeyDigest relationManifestDigest
        statementId productConfig relationArtifact).decoder =
      ProductFullClaimParser.claimDecoder
        (ProductPaperAlgebra.fullShapeContract logicalWidth publicFits)
        expectedApplication := by
  rfl

/-- The configuration contains the one concrete V2 paper key. This equality
prevents an adapter from substituting a weaker transcript, algebra, shape, or
challenge schedule behind the V2 profile identity. -/
theorem configuration_key_exact
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (expectedApplication : PublicImage)
    (verifierKeyDigest relationManifestDigest : Digest.Value)
    (statementId : ProductConcreteNifs.StatementId)
    (productConfig : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (relationArtifact : ProductConcreteNifs.RelationArtifact logicalWidth
      publicFits) :
    (configuration expectedApplication verifierKeyDigest relationManifestDigest
        statementId productConfig relationArtifact).key =
      ProductConcreteNifs.key statementId productConfig relationArtifact := by
  rfl

/-- The exact configuration rejects every three-attempt sampler shortfall.
The gate starts from the complete post-PiCCS output state selected by the
same verifier key. -/
theorem configuration_samplerCheck_exact
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (expectedApplication : PublicImage)
    (verifierKeyDigest relationManifestDigest : Digest.Value)
    (statementId : ProductConcreteNifs.StatementId)
    (productConfig : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (relationArtifact : ProductConcreteNifs.RelationArtifact logicalWidth
      publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (proof : Paper.Proof ProductNifsCodec.shape 9) :
    (configuration expectedApplication verifierKeyDigest relationManifestDigest
        statementId productConfig relationArtifact).samplerCheck
        running fresh proof =
      ProductPoseidon2.samplerSucceeded
        (((ProductConcreteNifs.key statementId productConfig relationArtifact
          ).piCcsExecution running fresh proof).outgoingState) := by
  rfl

/-- Acceptance by the exact selected verifier exposes the one strict field
vector used to construct its paper running input. No caller supplies a parser
result, a transition, or a cryptographic soundness premise. -/
theorem accepted_input_has_exact_fields
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape :
      Nightstream.SuperNeo.Concrete.Phi81Relation.Shape}
    (expectedApplication : PublicImage)
    (verifierKeyDigest relationManifestDigest : Digest.Value)
    (statementId : ProductConcreteNifs.StatementId)
    (productConfig : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (relationArtifact : ProductConcreteNifs.RelationArtifact logicalWidth
      publicFits)
    (claim : FullClaimNifsReceipt.Claim
      (selected expectedApplication verifierKeyDigest relationManifestDigest
        statementId productConfig relationArtifact))
    (proof : Paper.Proof ProductNifsCodec.shape 9)
    (output : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (accepted : FullClaimNifsReceipt.VerifyClaim
      (selected expectedApplication verifierKeyDigest relationManifestDigest
        statementId productConfig relationArtifact) (proof, output) claim) :
    ∃ fields : ProductNifsRunningParser.Fields,
      ∃ fresh : ProductNifsCodec.Fresh
          (ProductPaperAlgebra.FullShape logicalWidth publicFits),
        ProductFullClaimParser.decode
            (ProductPaperAlgebra.fullShapeContract logicalWidth publicFits)
            expectedApplication
            (FullClaimEnvelope.Value.ofProtocolClaim claim).block =
          some
            (ProductNifsRunningParser.runningOfFields
              (ProductPaperAlgebra.fullShapeContract logicalWidth publicFits)
              fields,
              fresh) ∧
        ProductNifsFieldParser.parse
            (FullClaimEnvelope.Value.ofProtocolClaim claim).recursiveState =
          some fields := by
  let contract :=
    ProductPaperAlgebra.fullShapeContract logicalWidth publicFits
  let config := configuration expectedApplication verifierKeyDigest
    relationManifestDigest statementId productConfig relationArtifact
  rcases ProductPaperNifsSelection.verifyClaim_decodes_and_accepts_paper
      config claim proof output accepted with
    ⟨running, fresh, decoded, _samplerAccepted, _paperAccepted⟩
  change ProductFullClaimParser.decode contract expectedApplication
      (FullClaimEnvelope.Value.ofProtocolClaim claim).block =
        some (running, fresh) at decoded
  rcases ProductFullClaimParser.decode_success contract expectedApplication
      decoded with
    ⟨value, wellFormed, blockExact, runningParsed, _freshExact⟩
  have valueExact : value = FullClaimEnvelope.Value.ofProtocolClaim claim := by
    apply FullClaimEnvelope.Value.encode_injective_on_canonical
      wellFormed.canonical accepted.1
    exact congrArg Subtype.val blockExact
  subst value
  rcases ProductNifsRunningParser.parse_success_fields contract runningParsed
    with ⟨fields, fieldsParsed, runningExact⟩
  subst running
  exact ⟨fields, fresh, decoded, fieldsParsed⟩

end Nightstream.Implementation.NebulaV2.ProductExactNifsConfiguration
