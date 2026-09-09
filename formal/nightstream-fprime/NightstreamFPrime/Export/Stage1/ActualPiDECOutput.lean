import NightstreamFPrime.Export.Stage1.ActualPiDECMessages
import NightstreamFPrime.Export.Stage1.ActualPiDECCarriedValues

/-!
Owns the exact PiDEC check and carried output from arbitrary accepted selected
rows and actual public input. Both proof message families use their actual
decoders. The recursive branch carries the complete 16-child result; the base
branch uses the existing default-running contract.
-/

namespace NightstreamFPrime.Export.Stage1.ActualPiDECOutput

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable (application : Lifecycle.Stage1.Application.Program)
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
  (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment)

include digest publicEqual rows

/-- The actual PiDEC messages pass the production check over the exact
verifier-derived parent and actual prior running state. -/
theorem selectedRowsAndPublic_imply_check :
    Nifs.PaperNonInteractive.piDecCheck
      (ProductionKey.key (PerApplicationFixedPoint.relation application fits) ajtai)
      (StateDecoder.running (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)
        (ActualStep.priorState application assignment))
      (ActualStep.decodedFresh application assignment)
      (ActualPiDECMessages.proof application fits assignment) = true := by
  let relation := PerApplicationFixedPoint.relation application fits
  let env := Spartan.pullback (ActualPiDEC.decodedEnv
    (ActualPiDEC.selectedGeometry application) assignment)
  have phase := ActualPiDEC.selectedRowsAndPublic_imply_phaseHolds
    application fits ajtai assignment digest publicEqual rows
  have attempt := ActualPiDECMessages.selectedRowsAndPublic_imply_attempt
    application fits ajtai assignment digest publicEqual rows
  have checked := AccumulatorSemantics.piDecCheck_eq_true_of_attempt relation ajtai env
    _ _ (ActualPiDECMessages.proof application fits assignment) _ _ phase attempt
  have agreement := congrArg
    (fun running => Nifs.PaperNonInteractive.piDecCheck (ProductionKey.key relation ajtai)
      running (ActualStep.decodedFresh application assignment)
      (ActualPiDECMessages.proof application fits assignment))
    (ActualPiCCSInputs.evalRunning_eq_priorRunning
      (relationLogicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (relationPublicFits := PerApplicationFixedPoint.publicFits application)
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
        (PerApplicationFixedPoint.geometry application)) assignment)
  exact agreement.symm.trans checked

/-- The production key computes all sixteen typed children from the actual
PiDEC values. This identifies the complete value before branch selection. -/
theorem selectedRowsAndPublic_imply_decodedOutput :
    (ProductionKey.key (PerApplicationFixedPoint.relation application fits) ajtai).output
      (StateDecoder.running (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)
        (ActualStep.priorState application assignment))
      (ActualStep.decodedFresh application assignment)
      (ActualPiDECMessages.proof application fits assignment) =
      some (RunningTransitionInputs.piDecRunningOutput
        (PerApplicationFixedPoint.relation application fits)
        (Spartan.pullback (ActualPiDEC.decodedEnv
          (ActualPiDEC.selectedGeometry application) assignment))) := by
  let relation := PerApplicationFixedPoint.relation application fits
  let env := Spartan.pullback (ActualPiDEC.decodedEnv
    (ActualPiDEC.selectedGeometry application) assignment)
  have phase := ActualPiDEC.selectedRowsAndPublic_imply_phaseHolds
    application fits ajtai assignment digest publicEqual rows
  have attempt := ActualPiDECMessages.selectedRowsAndPublic_imply_attempt
    application fits ajtai assignment digest publicEqual rows
  have computed := AccumulatorSemantics.keyOutput_eq_some_of_attempt relation ajtai env
    _ _ (ActualPiDECMessages.proof application fits assignment) _ _
    (AccumulatorInputs.output relation env) phase attempt
    (AccumulatorSemantics.outputForAttempt_eq_accumulatorOutput relation ajtai env phase)
  have agreement := congrArg
    (fun running => (ProductionKey.key relation ajtai).output
      running (ActualStep.decodedFresh application assignment)
      (ActualPiDECMessages.proof application fits assignment))
    (ActualPiCCSInputs.evalRunning_eq_priorRunning
      (relationLogicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (relationPublicFits := PerApplicationFixedPoint.publicFits application)
      (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
        (PerApplicationFixedPoint.geometry application)) assignment)
  exact agreement.symm.trans computed

/-- On a recursive step, the key's complete output is exactly the running
value in the actual next-state preimage. -/
theorem selectedRowsAndPublic_imply_output
    (iterationNonzero : StateDecoder.iteration
      (ActualStep.priorState application assignment) ≠ 0) :
    (ProductionKey.key (PerApplicationFixedPoint.relation application fits) ajtai).output
      (StateDecoder.running (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)
        (ActualStep.priorState application assignment))
      (ActualStep.decodedFresh application assignment)
      (ActualPiDECMessages.proof application fits assignment) =
      some (StateDecoder.running (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application)
        (ActualStep.outputState application assignment)) := by
  let relation := PerApplicationFixedPoint.relation application fits
  let geometry := PerApplicationFixedPoint.geometry application
  let env := Spartan.pullback (ActualRunningTransition.decodedEnv
    (ActualRunningTransition.selectedGeometry application) assignment)
  have publicBound : RecursivePublicOutputPlan.publicInput geometry assignment =
      encHash (publicFits := RecursivePublicOutputPlan.carrierPublicFits geometry) digest := by
    rw [RecursivePublicOutputPlan.publicInput_eq_projectPublicInput]
    exact publicEqual
  have one := RecursivePublicOutputPlan.publicEqual_implies_one
    geometry assignment digest publicBound
  have specification := ActualRunningTransition.selectedRowsZero_implies_specHolds
    application fits assignment one rows
  have counter := ActualRunningTransition.selectedIteration_eq_prior application assignment
  have nonzero : Lifecycle.Stage1.RunningTransition.iterationValue
      (RunningTransitionInputs.interface (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      RunningTransitionInputs.phaseOffset env ≠ 0 := by
    intro zero
    apply iterationNonzero
    change (ActualStep.priorState application assignment 28).val = 0
    have fieldZero : ActualStep.priorState application assignment 28 = 0 :=
      counter.symm.trans zero
    exact congrArg (fun word : F => word.val) fieldZero
  have transition := RunningTransitionInputs.spec_typed_recursive_eq_piDecOutput
    relation specification nonzero
  have carried := (ActualRunningTransition.selectedOutputRunning_eq_running
    application assignment).symm.trans
      (transition.trans (ActualPiDECCarriedValues.selectedRunningOutput_eq
        application fits assignment))
  exact (selectedRowsAndPublic_imply_decodedOutput application fits ajtai assignment
    digest publicEqual rows).trans (congrArg some carried.symm)

/-- Every accepted selected assignment with the actual public digest gives
the complete typed step for its decoded prior context. No proof template,
honest-encoder, NIFS-acceptance, or child-output premise is required. The
canonical verifier-context binding retains its separate contract. -/
theorem selectedRowsAndPublic_imply_step (fixed : digest.length = 4) :
    StepHoldsFor (PerApplicationFixedPoint.relation application fits) ajtai
      (ActualStep.contextKey application assignment) application
      (ActualStep.input application fits assignment
        (ActualStep.decodedFresh application assignment)
        (ActualPiDECMessages.proof application fits assignment))
      (ActualStep.output application assignment digest) := by
  apply (ActualStep.selectedRowsAndPublic_step_iff_baseOrPiDec application fits ajtai
    assignment (ActualPiDECMessages.sourceProof application fits assignment)
    digest fixed publicEqual rows).mpr
  by_cases base : StateDecoder.iteration (ActualStep.priorState application assignment) = 0
  · exact Or.inl base
  · exact Or.inr ⟨selectedRowsAndPublic_imply_check application fits ajtai assignment
      digest publicEqual rows, selectedRowsAndPublic_imply_output application fits ajtai
      assignment digest publicEqual rows base⟩

end NightstreamFPrime.Export.Stage1.ActualPiDECOutput
