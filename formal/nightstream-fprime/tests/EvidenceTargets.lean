import NightstreamFPrime.Export.Stage1.PilotDecodedPhase
import NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase
import NightstreamFPrime.Export.Stage1.ActualPiCCSInputs
import tests.EvidenceMetadata

/-! Exact assignment targets for the pilot and PiCCS evidence slice.
These targets do not claim the remaining full Stage 1 decoded-step theorem.
-/

namespace LeanGraph.Targets

open NightstreamFPrime
open Circuit Layout Layout.Stage1 Spec
open Export.Stage1 Lifecycle Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def PilotAssignment : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : Export.Stage1.PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (Export.Stage1.PerApplicationFixedPoint.logicalWidth application)),
    assignment (ApplicationRetainedGeometry.oneColumn
      (Export.Stage1.PerApplicationFixedPoint.geometry application)) = 1 →
    (Export.Stage1.PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment →
    Lifecycle.Pilot.SpecHolds PilotProduction.interface PilotProduction.witnessOffset
      (PilotSpartan.pullback (Export.Stage1.PilotDecodedEnvironment.env
        (Export.Stage1.DirectApplicationPrefixPlan.pilotOrdinaryGeometry
          (Export.Stage1.PerApplicationFixedPoint.geometry application)) assignment))

theorem pilotAssignment : PilotAssignment :=
  Export.Stage1.PilotDecodedPhase.selectedRowsZero_implies_specHolds

def PiCCSAssignment : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : Export.Stage1.PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : Lifecycle.PaperAlgebra.AjtaiKey
      (logicalWidth := Export.Stage1.PerApplicationFixedPoint.logicalWidth application)
      (publicFits := Export.Stage1.PerApplicationFixedPoint.publicFits application))
    (template : Lifecycle.Proof (Lifecycle.ProductionKey.degreeBound
      (Export.Stage1.PerApplicationFixedPoint.relation application fits)))
    (assignment : Assignment F (Export.Stage1.PerApplicationFixedPoint.logicalWidth application)),
    assignment (ApplicationRetainedGeometry.oneColumn
      (Export.Stage1.PerApplicationFixedPoint.geometry application)) = 1 →
    (Export.Stage1.PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment →
    Lifecycle.PiCCS.v1_1.Formal.PhaseHolds
      (Export.Stage1.PerApplicationFixedPoint.relation application fits) ajtai
      (PiCCSInvocations.parentInterface
        (Export.Stage1.PerApplicationFixedPoint.logicalWidth application)
        (Export.Stage1.PerApplicationFixedPoint.publicFits application))
      PiCCSInputs.phaseOffset
      (Spartan.pullback (Export.Stage1.PiCCSAssignmentSoundness.decodedEnv
        (Export.Stage1.DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
          (Export.Stage1.PerApplicationFixedPoint.geometry application)) assignment)) template

theorem piCCSAssignment : PiCCSAssignment :=
  Export.Stage1.PiCCSDecodedPhase.selectedRowsZero_implies_phaseHolds

def PiCCSPublicAssignment : Prop :=
  ∀ (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (template : Proof (ProductionKey.degreeBound
      (PerApplicationFixedPoint.relation application fits)))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest),
    digest.length = 4 →
    Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest →
    (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment →
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
      digest = stateHash (publicFits := PerApplicationFixedPoint.publicFits application) next

theorem piCCSPublicAssignment : PiCCSPublicAssignment :=
  ActualPiCCSInputs.selectedRowsAndPublic_imply_phaseAndHashes

#audit_axioms piCCSPublicAssignment

#audit_axioms pilotAssignment
#audit_axioms piCCSAssignment

end LeanGraph.Targets
