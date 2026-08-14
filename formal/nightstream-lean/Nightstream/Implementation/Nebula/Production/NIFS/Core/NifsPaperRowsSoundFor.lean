import Nightstream.Implementation.Nebula.NIFS.PiDEC.TypedBridgeFor
import Nightstream.Implementation.Nebula.Production.NIFS.PiRLC.ParentBridgeFor

/-!
Contract: exact exponent-indexed section-row soundness for one production NIFS
call.

One relation exponent selects the complete PiCCS replay, post-PiCCS sampler,
PiRLC parent, PiDEC attempt, and verifier output. The final theorem derives
the executable paper verifier result from satisfied rows and physical
serialization links. No PiCCS, sampler, PiRLC, PiDEC, or verifier result is a
premise.

This theorem does not claim generated-artifact containment, byte-decoder
refinement, terminal verification, cryptographic security, or Rust
refinement.

Assurance tier: exponent-indexed section-row-to-paper refinement.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionProductNifsPaperRowsSoundFor

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Physical PiRLC-output-to-PiDEC-parent links and typed child
serialization. -/
structure PiDecLinks
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (attempt : ProductPiDecTypedBridgeFor.ExactAttempt rowVariables
      logicalWidth publicFits) : Prop where
  parentBundleColumn : forall component row lane,
    piDecLayout.parentBundle.column component row lane =
      algebraLayout.outputBundle component row lane
  parentEvaluationColumn : forall matrix coefficient limb,
    piDecLayout.parentEvaluation.column matrix coefficient limb =
      algebraLayout.outputEvaluation matrix limb coefficient
  childBundle : forall child,
    (attempt.messages child).commitment =
      ProductPiDecTypedBridgeFor.decodeBundle
        (piDecLayout.childBundle child) assignment canonical
  childEvaluation : forall child,
    (attempt.messages child).evaluations =
      #[ProductPiDecTypedBridgeFor.decodeEvaluation rowVariables
        (piDecLayout.childEvaluation child) assignment canonical]

/-- All non-row placement facts for one production call. -/
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
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat) (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Prop where
  piRlc : ProductionProductPiRlcParentBridgeFor.Placement candidate statementId
    config artifact running fresh proof wires samplerBase algebraLayout
    assignment canonical
  piDec : PiDecLinks algebraLayout piDecLayout assignment canonical
    ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId config
      artifact).piDecAttempt running fresh proof)

/-- PiDEC parent bundle columns decode to the exact PiRLC output bundle. -/
theorem parentBundle_decode_eq
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {algebraLayout : ProductPiRlcAlgebraRows.Layout}
    {piDecLayout : ProductPiDecRows.Layout} {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    {attempt : ProductPiDecTypedBridgeFor.ExactAttempt rowVariables logicalWidth
      publicFits}
    (links : PiDecLinks algebraLayout piDecLayout assignment canonical attempt) :
    ProductPiDecTypedBridgeFor.decodeBundle piDecLayout.parentBundle assignment
        canonical =
      ProductPiRlcAlgebraSoundFor.decodeOutputBundle algebraLayout assignment
        canonical := by
  funext component row lane
  simp only [ProductPiDecTypedBridge.decodeBundle,
    ProductPiRlcAlgebraSound.decodeOutputBundle,
    ProductPiRlcRingCombinationSound.wireField]
  rw [links.parentBundleColumn component row lane]

/-- PiDEC parent evaluation columns decode to the exact PiRLC output
evaluation at the same exponent. -/
theorem parentEvaluation_decode_eq
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {algebraLayout : ProductPiRlcAlgebraRows.Layout}
    {piDecLayout : ProductPiDecRows.Layout} {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    {attempt : ProductPiDecTypedBridgeFor.ExactAttempt rowVariables logicalWidth
      publicFits}
    (links : PiDecLinks algebraLayout piDecLayout assignment canonical attempt) :
    ProductPiDecTypedBridgeFor.decodeEvaluation rowVariables
        piDecLayout.parentEvaluation assignment canonical =
      ProductPiRlcAlgebraSoundFor.decodeOutputEvaluation rowVariables
        algebraLayout assignment canonical := by
  funext matrix coefficient
  change K.mk _ _ = K.mk _ _
  congr 1
  · simp only [ProductPiRlcAlgebraSound.decodeOutputEvaluationLimb,
      ProductPiRlcRingCombinationSound.wireField]
    rw [links.parentEvaluationColumn matrix coefficient ⟨0, by decide⟩]
    have limbEqual :
        (⟨0, by decide⟩ : ProductPiDecRows.ExtensionLimb) = 0 := by
      apply Fin.ext
      rfl
    rw [limbEqual]
  · simp only [ProductPiRlcAlgebraSound.decodeOutputEvaluationLimb,
      ProductPiRlcRingCombinationSound.wireField]
    rw [links.parentEvaluationColumn matrix coefficient ⟨1, by decide⟩]
    have limbEqual :
        (⟨1, by decide⟩ : ProductPiDecRows.ExtensionLimb) = 1 := by
      apply Fin.ext
      rfl
    rw [limbEqual]

/-- Row-derived PiRLC parent fields and physical links construct exact PiDEC
placement. -/
theorem piDecPlacement_of_parentFields
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {algebraLayout : ProductPiRlcAlgebraRows.Layout}
    {piDecLayout : ProductPiDecRows.Layout} {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    {attempt : ProductPiDecTypedBridgeFor.ExactAttempt rowVariables logicalWidth
      publicFits}
    (links : PiDecLinks algebraLayout piDecLayout assignment canonical attempt)
    (parentBundle :
      ProductPiRlcAlgebraSoundFor.decodeOutputBundle algebraLayout assignment
          canonical = attempt.parent.commitment)
    (parentEvaluation :
      #[ProductPiRlcAlgebraSoundFor.decodeOutputEvaluation rowVariables
          algebraLayout assignment canonical] = attempt.parent.evaluations) :
    ProductPiDecTypedBridgeFor.Placement piDecLayout assignment canonical
      attempt := by
  constructor
  · exact parentBundle.symm.trans (parentBundle_decode_eq links).symm
  · exact links.childBundle
  · exact parentEvaluation.symm.trans
      (congrArg (fun value => #[value])
        (parentEvaluation_decode_eq links).symm)
  · exact links.childEvaluation

/-- Exact production section rows imply the exact executable paper-NIFS
result at the same relation exponent. -/
theorem rows_imply_exact_result
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (samplerBase : Nat) (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (piCcsPlacement : ProductionProductPiCcsTypedBridgeFor.Placement candidate
      statementId config artifact running fresh proof wires assignment)
    (piCcsRows : Satisfies
      (ProductPiCcsTranscriptRowsFor.rows
        (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
          config artifact running fresh wires)) assignment)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (classificationRows : ProductPiRlcCandidateClassificationRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (selectorRows : ProductPiRlcFirstAcceptedBatchRows.RowsHold
      (ProductionProductPiRlcParentBridgeFor.samplerInput candidate statementId
        config artifact running fresh wires samplerBase) assignment)
    (algebraRows : Satisfies
      (ProductPiRlcAlgebraRows.rows algebraLayout) assignment)
    (piDecRows : Satisfies (ProductPiDecRows.rows piDecLayout) assignment)
    (placement : Placement candidate statementId config artifact running fresh
      proof wires samplerBase algebraLayout piDecLayout assignment canonical) :
    let selected := ProductionProductPiCcsTypedBridgeFor.paperKey candidate
      statementId config artifact
    let sampleInput := ProductionProductPiRlcParentBridgeFor.samplerInput
      candidate statementId config artifact running fresh wires samplerBase
    ProductPoseidon2.samplerSucceeded
          (ProductPiRlcFirstAcceptedBatchSound.samplerState sampleInput
            assignment) = true /\
      piCcsCheck selected running fresh proof = true /\
      piDecCheck selected running fresh proof = true /\
      verify selected running fresh proof =
        some (selected.output running fresh proof) := by
  dsimp only
  let selected := ProductionProductPiCcsTypedBridgeFor.paperKey candidate
    statementId config artifact
  let sampleInput := ProductionProductPiRlcParentBridgeFor.samplerInput candidate
    statementId config artifact running fresh wires samplerBase
  have samplerSucceeded : ProductPoseidon2.samplerSucceeded
      (ProductPiRlcFirstAcceptedBatchSound.samplerState sampleInput assignment) =
      true :=
    ProductPiRlcSamplerResponseSound.samplerSucceeded_eq_true sampleInput
      assignment canonical one transcriptRows classificationRows selectorRows
  have piCcsAccepted : piCcsCheck selected running fresh proof = true := by
    exact ProductionProductPiCcsTypedBridgeFor.rows_imply_piCcsCheck_true
      candidate statementId config artifact running fresh proof wires assignment
      canonical one piCcsPlacement piCcsRows
  have parentFields :=
    ProductionProductPiRlcParentBridgeFor.parentFields_of_rows candidate
      statementId config artifact running fresh proof wires samplerBase
      algebraLayout assignment canonical one piCcsPlacement piCcsRows
      transcriptRows classificationRows selectorRows algebraRows placement.piRlc
  have piDecPlacement : ProductPiDecTypedBridgeFor.Placement piDecLayout
      assignment canonical (selected.piDecAttempt running fresh proof) := by
    exact piDecPlacement_of_parentFields placement.piDec parentFields.1
      parentFields.2.2
  have piDecAccepted : piDecCheck selected running fresh proof = true := by
    apply (piDecCheck_eq_true_iff selected running fresh proof).2
    change PiDEC.PaperVerifier.Accepted
      (ProductPaperAlgebraFor.piDecAlgebra config)
      (ProductPaperAlgebraFor.evaluationArity config)
      (selected.piDecAttempt running fresh proof)
    exact ProductPiDecTypedBridgeFor.paperAccepted_of_rows_for_attempt config
      (selected.piDecAttempt running fresh proof) canonical one piDecRows
      piDecPlacement rfl rfl
  refine ⟨samplerSucceeded, piCcsAccepted, piDecAccepted, ?_⟩
  exact (verify_eq_some_iff selected running fresh proof
    (selected.output running fresh proof)).2
      ⟨piCcsAccepted, piDecAccepted, rfl⟩

end Nightstream.Implementation.Nebula.ProductionProductNifsPaperRowsSoundFor
