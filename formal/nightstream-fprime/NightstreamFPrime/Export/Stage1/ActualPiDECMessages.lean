import NightstreamFPrime.Export.Stage1.ActualPiDEC
import NightstreamFPrime.Export.Stage1.ActualPiDECParent
import NightstreamFPrime.Export.Stage1.ActualStep
import NightstreamFPrime.Layout.Stage1.AccumulatorSemantics

/-!
Owns the complete NIFS proof decoded from the actual selected assignment.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiDECMessages

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def sourceProof (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application)) :
    Proof (ProductionKey.degreeBound (PerApplicationFixedPoint.relation application fits)) :=
  AccumulatorInputs.proof (PerApplicationFixedPoint.relation application fits)
    (Spartan.pullback (ActualPiDEC.decodedEnv
      (ActualPiDEC.selectedGeometry application) assignment))

def proof (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application)) :
    Proof (ProductionKey.degreeBound (PerApplicationFixedPoint.relation application fits)) :=
  ActualStep.withDecodedPiCCS application fits assignment
    (sourceProof application fits assignment)

def attempt (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application)) :=
  PiDEC.v1_2.Semantics.inputAttempt (PerApplicationFixedPoint.relation application fits)
    (PiDECArithmetic.phaseInterface (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)) PiDECInputs.phaseOffset
    (Spartan.pullback (ActualPiDEC.decodedEnv
      (ActualPiDEC.selectedGeometry application) assignment))

theorem proof_commitment_eq
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (child : Fin productionShape.runningCount)
    (row : Fin productionProfile.commitmentWidth) (lane : Fin ringDegree) :
    (proof application fits assignment).piDecCommitments child row lane =
      (PiDECInputs.childCommitment (RunningTransitionInputs.childOfRunning child) row lane).eval
        (Spartan.pullback (ActualPiDEC.decodedEnv
          (ActualPiDEC.selectedGeometry application) assignment)) := by
  rfl

theorem proof_evalK_eq
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (child : Fin productionShape.runningCount)
    (coefficient : Fin productionShape.coefficientCount) :
    ((proof application fits assignment).piDecEvaluations child).pad coefficient =
      (PiDECInputs.childEvalK (RunningTransitionInputs.childOfRunning child) coefficient).eval
        (Spartan.pullback (ActualPiDEC.decodedEnv
          (ActualPiDEC.selectedGeometry application) assignment)) := by
  rfl

theorem proof_evalA_eq
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (child : Fin productionShape.runningCount) (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    ((proof application fits assignment).piDecEvaluations child).matrix matrix coefficient =
      (PiDECInputs.childEvalA (RunningTransitionInputs.childOfRunning child) matrix coefficient).eval
        (Spartan.pullback (ActualPiDEC.decodedEnv
          (ActualPiDEC.selectedGeometry application) assignment)) := by
  rfl

theorem attemptForParent_eq
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application)) :
    (ProductionKey.key (PerApplicationFixedPoint.relation application fits) ajtai).piDecAttemptForParent
        (proof application fits assignment)
          (attempt application fits assignment).parent =
      attempt application fits assignment := by
  let relation := PerApplicationFixedPoint.relation application fits
  let env := Spartan.pullback (ActualPiDEC.decodedEnv
    (ActualPiDEC.selectedGeometry application) assignment)
  have boundary :=
    (AccumulatorSemantics.piDecAttempt_eq_keyAttemptForParent relation ajtai env).symm
  rw [← AccumulatorSemantics.piDecParent_eq_piRlcOutput] at boundary
  simpa only [Nifs.PaperNonInteractive.Key.piDecAttemptForParent, proof,
    ActualStep.withDecodedPiCCS, PiCCS.v1_2.Formal.evalProof, sourceProof,
    attempt, PiDECArithmetic.phaseInterface, relation, env] using! boundary

theorem keyPiDecAttempt_eq_some_of_parent
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (running : Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (fresh : Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (parentEqual : (ProductionKey.key (PerApplicationFixedPoint.relation application fits) ajtai).parent
      running fresh (proof application fits assignment) =
        some (attempt application fits assignment).parent) :
    (ProductionKey.key (PerApplicationFixedPoint.relation application fits) ajtai).piDecAttempt
      running fresh (proof application fits assignment) =
        some (attempt application fits assignment) := by
  rw [Nifs.PaperNonInteractive.Key.piDecAttempt, parentEqual, Option.map_some,
    attemptForParent_eq]
  rfl

theorem selectedRowsAndPublic_imply_attempt
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (ajtai : AjtaiKey (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    let relation := PerApplicationFixedPoint.relation application fits
    let interface := PiCCSInvocations.parentInterface (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
    let env := Spartan.pullback (PiCCSAssignmentSoundness.decodedEnv
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
        (PerApplicationFixedPoint.geometry application)) assignment)
    (ProductionKey.key relation ajtai).piDecAttempt
        (PiCCS.v1_2.Formal.evalRunning interface PiCCSInputs.phaseOffset env)
        (PiCCS.v1_2.Formal.evalFresh interface PiCCSInputs.phaseOffset env)
        (proof application fits assignment) =
      some (attempt application fits assignment) := by
  have parentEqual := ActualPiDECParent.selectedRowsAndPublic_imply_parent
    application fits ajtai (sourceProof application fits assignment)
    assignment digest publicEqual accepted
  exact keyPiDecAttempt_eq_some_of_parent application fits ajtai assignment _ _ parentEqual

end NightstreamFPrime.Export.Stage1.ActualPiDECMessages
