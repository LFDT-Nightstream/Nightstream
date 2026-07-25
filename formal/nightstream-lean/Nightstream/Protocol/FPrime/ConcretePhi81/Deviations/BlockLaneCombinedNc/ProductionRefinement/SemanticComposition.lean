import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.SemanticAttempt
import Nightstream.SuperNeo.Folding.Composition.ReferenceArithmetization
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth.Ambient
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement

/-!
Fixed-active knowledge composition for the production Split-NC prefix.

Assurance tier: model-level registered-deviation refinement.

Owns: the concrete join from production FE/delayed-NC acceptance to
`Pi_RLC`/`Pi_DEC` knowledge composition without the generic
`rewindArithmetization` callback. Every deterministic production failure is
retained as an exact FE, NC, or registered-deviation event.

Does not own: probability bounds, commitment security, extraction,
Fiat--Shamir, Poseidon2, Rust, R1CS, costs, or rows.

Emits constraints: no.

| Stage path | Owned equation | Authority |
|---|---|---|
| `fprime.piccs.production.semantic_composition` | production acceptance yields extraction or the exact generic/FE/NC/deviation event union | derived |
-/

set_option autoImplicit false
set_option maxRecDepth 2048

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.SemanticComposition

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

private abbrev sumcheckOps :=
  ConcreteCarrier.extensionOps.toOps.toSymbolic

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Exact failure surface added by the concrete production prefix. The generic
composition event remains unchanged and these branches retain the production
FE/NC constructor families and the registered delayed-state obligation. -/
def BadEvent
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (bindingOps : PiRLC.RelaxedBindingOps
      (SourceAssignment shape) (CommitmentValue verifierRows) RingF)
    (sampling : PiRLC.SamplingBoundary FixedActive.arity.total) : Prop :=
  let ccsAttempt := SemanticAttempt.attempt input certificate
  let execution := Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
    input.full certificate.materialize
  Composition.BadEvent (semantics input.full.key) productionGlobalParams
      bindingOps sampling ccsAttempt
      (execution.piRlcAttempt certificate.materialize).inputs ∨
    FeFailure input certificate ∨
    NcFailure input certificate ∨
    RegisteredDeviationObligation input certificate

/-- Production fixed-active composition with no retained
`rewindArithmetization` premise. The opening-derived carrier is not trusted as
an unchecked witness: its CE opening is proved from accepted paper truth and
source/output binding, then compared with extraction through the existing
commitment-uniqueness boundary. -/
theorem fold_extraction_or_named_failure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (input : AuthoritativeInput (shape := shape) (State := State)
      (publicRingColumns := publicRingColumns) (verifierRows := verifierRows)
      (publicFits := publicFits))
    (certificate : Certificate input)
    (accepted : ProductionVerifierAccepts input certificate)
    (bindingOps : PiRLC.RelaxedBindingOps
      (SourceAssignment shape) (CommitmentValue verifierRows) RingF)
    (sampling : PiRLC.SamplingBoundary FixedActive.arity.total)
    (finalAssignments :
      Fin productionGlobalParams.k -> SourceAssignment shape)
    (rlcAccepted : PiRLC.Accepted (rlcAlgebra input.full.key)
      ((Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive input.full
        certificate.materialize).piRlcAttempt certificate.materialize))
    (decAccepted : PiDEC.Accepted (decAlgebra input.full.key)
      ((Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive input.full
        certificate.materialize).piDecAttempt certificate.materialize))
    (finalValid : forall child,
      CE.Holds (semantics input.full.key) productionGlobalParams
        (((Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive input.full
          certificate.materialize).piDecAttempt
            certificate.materialize).children child)
        (finalAssignments child))
    (extractor : Composition.WeakExtractor (semantics input.full.key)
      productionGlobalParams (rlcAlgebra input.full.key)
      ((Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive input.full
        certificate.materialize).piRlcAttempt certificate.materialize)
      sampling)
    (uniqueness : PiRLC.UniquenessBridge (semantics input.full.key)
      productionGlobalParams bindingOps (n := FixedActive.arity.total)) :
    let ccsAttempt := SemanticAttempt.attempt input certificate
    Nonempty (Composition.ExtractedBatch (semantics input.full.key)
      productionGlobalParams ccsAttempt) ∨
      BadEvent input certificate bindingOps sampling := by
  let ccsAttempt := SemanticAttempt.attempt input certificate
  let execution := Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
    input.full certificate.materialize
  rcases accepted_implies_paper_or_algebraic_failure noZeroDivisors input
      certificate accepted with paper | fe | nc
  · by_cases pendingBound :
        ProductionPiCcs.PendingBound input.full input.data
    · have ccsAccepted : PiCCS.Accepted sumcheckOps ccsAttempt :=
        SemanticAttempt.accepted input certificate accepted
      have referenceArithmetization :
          PiCCS.Arithmetization (semantics input.full.key)
            productionGlobalParams sumcheckOps ccsAttempt
            (InputAuthority.productAssignments input.data
              input.full.alignment) :=
        SemanticAttempt.arithmetization input certificate accepted paper
          pendingBound
      have productHolds :
          ProductHolds publicRingColumns publicFits (commit input.full.key)
            ccsAttempt.outputs
            (InputAuthority.productAssignments input.data
              input.full.alignment) := by
        exact
          Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.OutputRefinement.materializedOutputsHold_of_yRingEq
            publicRingColumns publicFits (commit input.full.key) input.data
            input.full.alignment input.full.input
            (ProductionPiCcs.fePoint input.full
              certificate.materialize).row
            certificate.output production_norm_stages.1 paper
            input.sourceProduct_bound certificate.output_bound.1
      have outputFresh : forall source,
          (ccsAttempt.outputs source).stage = .fresh := by
        intro source
        exact OutputProduct.materialize_stage publicRingColumns publicFits
          input.full.alignment input.full.input
          (ProductionPiCcs.fePoint input.full
            certificate.materialize).row
          certificate.output source
      have freshLeAmbient :
          productionGlobalParams.b <= productionGlobalParams.q / 2 := by
        decide
      have ccsAmbient :
          PiRLC.AmbientOpenings (semantics input.full.key)
            productionGlobalParams ccsAttempt.outputs
            (InputAuthority.productAssignments input.data
              input.full.alignment) :=
        ProductTruth.ambientOpenings_of_productHolds publicRingColumns
          publicFits (commit input.full.key) ccsAttempt.outputs
          (InputAuthority.productAssignments input.data input.full.alignment)
          outputFresh freshLeAmbient productHolds
      have sameRlcInputs : forall source,
          (execution.piRlcAttempt certificate.materialize).inputs source =
            ccsAttempt.outputs source := by
        intro source
        rfl
      have referenceAmbient :
          PiRLC.AmbientOpenings (semantics input.full.key)
            productionGlobalParams
            (execution.piRlcAttempt certificate.materialize).inputs
            (InputAuthority.productAssignments input.data
              input.full.alignment) := by
        intro source
        rw [sameRlcInputs source]
        exact ccsAmbient source
      have result :=
        Composition.ReferenceArithmetization.fold_extraction_or_bad_event
          (semantics input.full.key) productionGlobalParams sumcheckOps
          (rlcAlgebra input.full.key) (decAlgebra input.full.key) bindingOps
          FixedActive.arity sampling ccsAttempt
          (execution.piRlcAttempt certificate.materialize)
          (execution.piDecAttempt certificate.materialize) finalAssignments
          (InputAuthority.productAssignments input.data input.full.alignment)
          (by decide) sameRlcInputs rfl ccsAccepted rlcAccepted decAccepted
          finalValid extractor uniqueness referenceAmbient
          referenceArithmetization
      rcases result with extracted | bad
      · exact Or.inl extracted
      · exact Or.inr (Or.inl bad)
    · cases pendingEq : input.full.pending with
      | none =>
          exfalso
          apply pendingBound
          simp [ProductionPiCcs.PendingBound, pendingEq]
      | some pending =>
          exact Or.inr (Or.inr (Or.inr (Or.inr
            (.delayedPackedYZcol pending pendingEq pendingBound))))
  · exact Or.inr (Or.inr (Or.inl fe))
  · exact Or.inr (Or.inr (Or.inr (Or.inl nc)))

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ProductionRefinement.SemanticComposition
