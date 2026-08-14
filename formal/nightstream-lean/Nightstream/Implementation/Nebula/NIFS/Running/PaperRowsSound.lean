import Nightstream.Implementation.Nebula.NIFS.PiDEC.TypedBridge
import Nightstream.Implementation.Nebula.NIFS.PiRLC.ParentBridge
import Nightstream.Implementation.Nebula.NIFS.PiRLC.SamplerResponseSound

/-!
Contract: exact section-row soundness for one V2 paper NIFS call.

This file connects the row-derived PiCCS result, the post-PiCCS full-field
sampler, all 110 PiRLC algebra families, and the 5,400 PiDEC coordinate rows.
The final theorem derives the exact executable paper verifier result.

The placement contains physical column identities, typed serialization links
for prover fields, and the existing PiCCS and PiRLC input placements. It does
not contain a sampler result, challenge, combined parent, recomposition
equation, acceptance Boolean, or NIFS verifier result.

This theorem is about the exact paper verifier sections. It does not claim
that a generated recursive artifact contains these sections, that Rust bytes
decode to these typed inputs, or that cryptographic bad events are negligible.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductNifsPaperRowsSound

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Physical PiRLC-output-to-PiDEC-parent links and typed serialization of
the fourteen prover-supplied PiDEC children.

The parent links are column identities. The child links state where the
typed proof fields are serialized. No field states a recomposition equation
or an acceptance result. -/
structure PiDecLinks
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (attempt : ProductPiDecTypedBridge.ExactAttempt logicalWidth publicFits) :
    Prop where
  parentBundleColumn : forall component row lane,
    piDecLayout.parentBundle.column component row lane =
      algebraLayout.outputBundle component row lane
  parentEvaluationColumn : forall matrix coefficient limb,
    piDecLayout.parentEvaluation.column matrix coefficient limb =
      algebraLayout.outputEvaluation matrix limb coefficient
  childBundle : forall child,
    (attempt.messages child).commitment =
      ProductPiDecTypedBridge.decodeBundle
        (piDecLayout.childBundle child) assignment canonical
  childEvaluation : forall child,
    (attempt.messages child).evaluations =
      #[ProductPiDecTypedBridge.decodeEvaluation
        (piDecLayout.childEvaluation child) assignment canonical]

/-- All non-row placement facts for one exact paper NIFS call. -/
structure Placement
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifs.StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifs.RelationArtifact logicalWidth publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (proof : ProductPiCcsTypedBridge.ExactProof)
    (wires : ProductPiCcsTypedBridge.Wires) (samplerBase : Nat)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Prop where
  piRlc : ProductPiRlcParentBridge.Placement statementId config artifact
    running fresh proof wires samplerBase algebraLayout assignment canonical
  piDec : PiDecLinks algebraLayout piDecLayout assignment canonical
    ((ProductConcreteNifs.key statementId config artifact).piDecAttempt
      running fresh proof)

/-- The PiDEC parent bundle columns decode to the exact PiRLC output bundle. -/
theorem parentBundle_decode_eq
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {algebraLayout : ProductPiRlcAlgebraRows.Layout}
    {piDecLayout : ProductPiDecRows.Layout}
    {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    {attempt : ProductPiDecTypedBridge.ExactAttempt logicalWidth publicFits}
    (links : PiDecLinks algebraLayout piDecLayout assignment canonical attempt) :
    ProductPiDecTypedBridge.decodeBundle piDecLayout.parentBundle assignment
        canonical =
      ProductPiRlcAlgebraSound.decodeOutputBundle algebraLayout assignment
        canonical := by
  funext component row lane
  simp only [ProductPiDecTypedBridge.decodeBundle,
    ProductPiRlcAlgebraSound.decodeOutputBundle,
    ProductPiRlcRingCombinationSound.wireField]
  rw [links.parentBundleColumn component row lane]

/-- The PiDEC parent evaluation columns decode to the exact PiRLC output
evaluation. -/
theorem parentEvaluation_decode_eq
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {algebraLayout : ProductPiRlcAlgebraRows.Layout}
    {piDecLayout : ProductPiDecRows.Layout}
    {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    {attempt : ProductPiDecTypedBridge.ExactAttempt logicalWidth publicFits}
    (links : PiDecLinks algebraLayout piDecLayout assignment canonical attempt) :
    ProductPiDecTypedBridge.decodeEvaluation piDecLayout.parentEvaluation
        assignment canonical =
      ProductPiRlcAlgebraSound.decodeOutputEvaluation algebraLayout assignment
        canonical := by
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

/-- Row-derived PiRLC parent fields and physical parent links construct the
exact PiDEC placement. -/
theorem piDecPlacement_of_parentFields
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {algebraLayout : ProductPiRlcAlgebraRows.Layout}
    {piDecLayout : ProductPiDecRows.Layout}
    {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    {attempt : ProductPiDecTypedBridge.ExactAttempt logicalWidth publicFits}
    (links : PiDecLinks algebraLayout piDecLayout assignment canonical attempt)
    (parentBundle :
      ProductPiRlcAlgebraSound.decodeOutputBundle algebraLayout assignment
          canonical = attempt.parent.commitment)
    (parentEvaluation :
      #[ProductPiRlcAlgebraSound.decodeOutputEvaluation algebraLayout assignment
          canonical] = attempt.parent.evaluations) :
    ProductPiDecTypedBridge.Placement piDecLayout assignment canonical
      attempt := by
  constructor
  · exact parentBundle.symm.trans (parentBundle_decode_eq links).symm
  · exact links.childBundle
  · exact parentEvaluation.symm.trans
      (congrArg (fun value => #[value])
        (parentEvaluation_decode_eq links).symm)
  · exact links.childEvaluation

/-- **Exact V2 paper-NIFS section-row soundness.**

All four results are derived from section-row satisfaction and placement.
The selected bounded sampler must succeed even though the abstract paper
verifier models a total strong-set response. The executable verifier then
returns only its verifier-computed running output. -/
theorem rows_imply_exact_result
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifs.StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifs.RelationArtifact logicalWidth publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (proof : ProductPiCcsTypedBridge.ExactProof)
    (wires : ProductPiCcsTypedBridge.Wires) (samplerBase : Nat)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (piDecLayout : ProductPiDecRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (piCcsPlacement : ProductPiCcsTypedBridge.Placement statementId config
      artifact running fresh proof wires assignment)
    (piCcsRows : Satisfies
      (ProductPiCcsTranscriptRows.rows
        (ProductPiCcsTypedBridge.rowInput statementId config artifact running
          fresh wires)) assignment)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold
      (ProductPiRlcParentBridge.samplerInput statementId config artifact
        running fresh wires samplerBase) assignment)
    (classificationRows : ProductPiRlcCandidateClassificationRows.RowsHold
      (ProductPiRlcParentBridge.samplerInput statementId config artifact
        running fresh wires samplerBase) assignment)
    (selectorRows : ProductPiRlcFirstAcceptedBatchRows.RowsHold
      (ProductPiRlcParentBridge.samplerInput statementId config artifact
        running fresh wires samplerBase) assignment)
    (algebraRows : Satisfies
      (ProductPiRlcAlgebraRows.rows algebraLayout) assignment)
    (piDecRows : Satisfies (ProductPiDecRows.rows piDecLayout) assignment)
    (placement : Placement statementId config artifact running fresh proof
      wires samplerBase algebraLayout piDecLayout assignment canonical) :
    let selected := ProductConcreteNifs.key statementId config artifact
    let sampleInput := ProductPiRlcParentBridge.samplerInput statementId config
      artifact running fresh wires samplerBase
    ProductPoseidon2.samplerSucceeded
          (ProductPiRlcFirstAcceptedBatchSound.samplerState sampleInput assignment) =
        true /\
      piCcsCheck selected running fresh proof = true /\
      piDecCheck selected running fresh proof = true /\
      verify selected running fresh proof =
        some (selected.output running fresh proof) := by
  dsimp only
  let selected := ProductConcreteNifs.key statementId config artifact
  let sampleInput := ProductPiRlcParentBridge.samplerInput statementId config
    artifact running fresh wires samplerBase
  have samplerSucceeded : ProductPoseidon2.samplerSucceeded
      (ProductPiRlcFirstAcceptedBatchSound.samplerState sampleInput assignment) =
      true :=
    ProductPiRlcSamplerResponseSound.samplerSucceeded_eq_true sampleInput
      assignment canonical one transcriptRows classificationRows selectorRows
  have piCcsAccepted : piCcsCheck selected running fresh proof = true := by
    exact ProductPiCcsTypedBridge.rows_imply_piCcsCheck_true statementId config
      artifact running fresh proof wires assignment canonical one piCcsPlacement
      piCcsRows
  have parentFields := ProductPiRlcParentBridge.parentFields_of_rows statementId
    config artifact running fresh proof wires samplerBase algebraLayout assignment
    canonical one piCcsPlacement piCcsRows transcriptRows classificationRows
    selectorRows algebraRows placement.piRlc
  have piDecPlacement : ProductPiDecTypedBridge.Placement piDecLayout assignment
      canonical (selected.piDecAttempt running fresh proof) := by
    exact piDecPlacement_of_parentFields placement.piDec parentFields.1
      parentFields.2.2
  have piDecAccepted : piDecCheck selected running fresh proof = true := by
    exact ProductPiDecTypedBridge.piDecCheck_true_of_rows statementId config
      artifact running fresh proof canonical one piDecRows piDecPlacement
  refine ⟨samplerSucceeded, piCcsAccepted, piDecAccepted, ?_⟩
  exact (verify_eq_some_iff selected running fresh proof
    (selected.output running fresh proof)).2
      ⟨piCcsAccepted, piDecAccepted, rfl⟩

end Nightstream.Implementation.Nebula.ProductNifsPaperRowsSound
