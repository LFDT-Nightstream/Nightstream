import NightstreamFPrime.Export.Stage1.ActualHashSlots
import NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase

/-!
Owns agreement between the typed PiCCS inputs and the actual prior hash.
The running claim uses the existing state decoder. The fresh public input
uses the pilot's shared forms. These connections add no rows or value premise.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiCCSInputs

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
  {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

/-- PiCCS reads the complete running claim from the same prior preimage
that the pilot hashes, for every assignment. -/
theorem evalRunning_eq_priorRunning
    (geometry : PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    PiCCS.v1_1.Formal.evalRunning
        (PiCCSInvocations.parentInterface relationLogicalWidth relationPublicFits)
        PiCCSInputs.phaseOffset
        (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv geometry assignment)) =
      (StateDecoder.preimage relationLogicalWidth relationPublicFits
        (ActualPreimageFraming.priorState geometry assignment)).running functionIndex := by
  change PiCCS.v1_1.StatementAbsorption.evalRunning
      (PiCCSInputs.runningExpr relationLogicalWidth relationPublicFits)
      (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv geometry assignment)) =
    StateDecoder.running relationLogicalWidth relationPublicFits
      (ActualPreimageFraming.priorState geometry assignment)
  rw [StateDecoder.evalRunning_eq_running]
  unfold StateDecoder.running
  apply congrArg (PiCCSInputs.decodedRunning relationLogicalWidth relationPublicFits)
  unfold StateDecoder.externalValues
  congr 1
  funext word
  exact ActualPreimageFraming.priorWord_eq geometry assignment word

/-- The typed fresh public input is the same public value as the prior hash
slot. The parent shares these forms without a copy constraint. -/
theorem evalFreshPublic_eq_priorPublic
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin productionShape.freshCount) :
    (PiCCS.v1_1.Formal.evalFresh
      (PiCCSInvocations.parentInterface relationLogicalWidth relationPublicFits)
      PiCCSInputs.phaseOffset
      (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry) assignment))).publicInputs source =
      ActualHashSlots.publicInput geometry assignment := by
  funext column
  exact (PilotDecodedEnvironment.priorPublic_agrees geometry assignment column).symm

/-- The selected rows bind each PiCCS fresh public input to the hash of the
same decoded prior state that supplies its running claim. -/
theorem selectedRowsZero_implies_freshPublicHash
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (one : assignment (ApplicationRetainedGeometry.oneColumn
      (PerApplicationFixedPoint.geometry application)) = 1)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero
      assignment)
    (source : Fin productionShape.freshCount) :
    (PiCCS.v1_1.Formal.evalFresh
      (PiCCSInvocations.parentInterface (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      PiCCSInputs.phaseOffset
      (Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
          (PerApplicationFixedPoint.geometry application)) assignment))).publicInputs source =
      encHash (publicFits := PerApplicationFixedPoint.publicFits application)
        (stateHash (publicFits := PerApplicationFixedPoint.publicFits application)
          (StateDecoder.preimage (PerApplicationFixedPoint.logicalWidth application)
            (PerApplicationFixedPoint.publicFits application)
            (ActualPreimageFraming.priorState
              (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
                (PerApplicationFixedPoint.geometry application)) assignment))) := by
  rw [evalFreshPublic_eq_priorPublic]
  exact (ActualHashSlots.selectedRowsZero_implies_hashSlots
    application fits assignment one accepted).1

/-- The actual public boundary and selected rows imply the phase and its
shared hash observations. The public marker supplies the one cell. These
facts retain one assignment and one decoded PiCCS environment throughout. -/
theorem selectedRowsAndPublic_imply_phaseAndHashes
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (template : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits)))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest) (fixed : digest.length = 4)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero
      assignment) :
    let geometry := PerApplicationFixedPoint.geometry application
    let relation := PerApplicationFixedPoint.relation application fits
    let interface := PiCCSInvocations.parentInterface
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
    let env := Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
    let prior := StateDecoder.preimage (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
      (ActualPreimageFraming.priorState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
    let next := ActualHashSlots.nextPreimage
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
      (ActualPreimageFraming.priorState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
      (ActualPreimageFraming.outputState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment)
    PiCCS.v1_1.Formal.PhaseHolds relation ajtai interface
        PiCCSInputs.phaseOffset env template ∧
      PiCCS.v1_1.Formal.evalRunning interface PiCCSInputs.phaseOffset env =
        prior.running functionIndex ∧
      (∀ source, (PiCCS.v1_1.Formal.evalFresh interface
          PiCCSInputs.phaseOffset env).publicInputs source =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application)
          (stateHash (publicFits := PerApplicationFixedPoint.publicFits application) prior)) ∧
      digest = stateHash (publicFits := PerApplicationFixedPoint.publicFits application)
        next := by
  let geometry := PerApplicationFixedPoint.geometry application
  have publicBound : RecursivePublicOutputPlan.publicInput geometry assignment =
      encHash (publicFits := RecursivePublicOutputPlan.carrierPublicFits geometry) digest := by
    rw [RecursivePublicOutputPlan.publicInput_eq_projectPublicInput]
    exact publicEqual
  have one := RecursivePublicOutputPlan.publicEqual_implies_one
    geometry assignment digest publicBound
  refine ⟨PiCCSDecodedPhase.selectedRowsZero_implies_phaseHolds
      application fits ajtai template assignment one accepted,
    evalRunning_eq_priorRunning
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment,
    ?_, ActualHashSlots.selectedRowsAndPublic_imply_outputHash
      application fits assignment digest fixed publicEqual accepted⟩
  intro source
  exact selectedRowsZero_implies_freshPublicHash
    application fits assignment one accepted source

end NightstreamFPrime.Export.Stage1.ActualPiCCSInputs
