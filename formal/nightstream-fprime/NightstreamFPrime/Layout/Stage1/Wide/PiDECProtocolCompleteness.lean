import NightstreamFPrime.Layout.Stage1.Wide.PiRLCProtocolCompleteness
import NightstreamFPrime.Layout.Stage1.Wide.PiDECLoaded
import NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionSemantics

/-! Complete PiDEC from the accepted wide-key run and its constructed PiRLC
parent. The completed child family is the verifier's exact running output. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.PiDECProtocolCompleteness

open NightstreamFPrime.Circuit NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

theorem verifierInputs
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (accepted : Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai) running fresh proof = some result) :
    PiCCS.Accepted (PiRLC.Wide.Key.key relation ajtai) running fresh proof ∧
    ∃ challenges, (PiRLC.Wide.Key.key relation ajtai).piRlcChallenges running fresh proof = some challenges ∧
      PiDEC.PaperVerifier.Accepted (PiRLC.Wide.Key.key relation ajtai).piDecAlgebra
        (PiRLC.Wide.Key.key relation ajtai).piDecPublicInputSplit (PiRLC.Wide.Key.key relation ajtai).piDecEvaluationArity
        ((PiRLC.Wide.Key.key relation ajtai).piDecAttemptForParent proof
          ((PiRLC.Wide.Key.key relation ajtai).parentForChallenges running fresh proof challenges)) := by
  let key := PiRLC.Wide.Key.key relation ajtai
  obtain ⟨cCheck, dCheck, _⟩ := (Nifs.PaperNonInteractive.verify_eq_some_iff key running fresh proof result).mp accepted
  obtain ⟨attempt, attemptEq, checks⟩ := (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff key running fresh proof).mp dCheck
  change (key.parent running fresh proof).map (key.piDecAttemptForParent proof) = some attempt at attemptEq
  obtain ⟨parent, parentEq, attemptValue⟩ := Option.map_eq_some_iff.mp attemptEq
  change (key.piRlcChallenges running fresh proof).map (key.parentForChallenges running fresh proof) = some parent at parentEq
  obtain ⟨challenges, sampled, parentValue⟩ := Option.map_eq_some_iff.mp parentEq
  refine ⟨cCheck, challenges, sampled, ?_⟩
  rw [parentValue, attemptValue]
  exact checks

private theorem rValues_eq_of_agree (before after : Env)
    (agrees : ∀ index, index < PiRLCStarts.phaseFreshStart → after index = before index) :
    PiRLC.Wide.Semantics.evalChallenges (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
        PiRLCInputs.phaseOffset after =
      PiRLC.Wide.Semantics.evalChallenges (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits))
        PiRLCInputs.phaseOffset before ∧
    PiDECLoaded.parent relation after = PiDECLoaded.parent relation before := by
  constructor
  · funext source lane
    apply Expr.eval_eq_of_agree_below _ PiRLCStarts.phaseFreshStart after before _ agrees
    exact Expr.VarsBelow.mono _
      (PiRLC.Wide.ProjectedBatch.outputChallenge_below PiRLCInputs.phaseOffset (PiRLC.Wide.Semantics.sourceIndex source) lane)
      (by decide)
  · exact PiDECLoaded.parent_eq_of_agree relation after before agrees

private theorem running_ext
    (left right : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : left.point = right.point) (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs) (evaluations : left.evaluations = right.evaluations) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem running_of_children (env : Env) (proof : Proof (ProductionKey.degreeBound relation))
    (parent : PiDEC.v1_1.OutputBinding.Output logicalWidth publicFits)
    (outputs : PiDEC.v1_1.Semantics.output relation (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset env =
      PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai)
        ((PiRLC.Wide.Key.key relation ajtai).piDecAttemptForParent proof parent)) :
    RunningTransitionInputs.piDecRunningOutput relation env =
      (PiRLC.Wide.Key.key relation ajtai).outputForAttempt proof
        ((PiRLC.Wide.Key.key relation ajtai).piDecAttemptForParent proof parent)
        ((PiRLC.Wide.Key.key relation ajtai).piDecPublicInputSplit.split parent.publicInput) := by
  apply running_ext
  · exact congrArg (fun values => (values ⟨0, by decide⟩).point) outputs
  · funext source
    exact congrArg (fun values => (values (RunningTransitionInputs.childOfRunning source)).commitment) outputs
  · funext source
    exact congrArg (fun values => (values (RunningTransitionInputs.childOfRunning source)).publicInput) outputs
  · funext source
    exact congrArg (fun values => (values (RunningTransitionInputs.childOfRunning source)).evaluations.getD
      0 PaperAlgebra.evaluationZero) outputs

/-- Complete PiDEC after the PiRLC physical scratch interval. Acceptance and
the two PiRLC identities determine the exact checked child output. -/
theorem completePrefix_after_r
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof (ProductionKey.degreeBound relation))
    (result : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (accepted : Nifs.PaperNonInteractive.verify (PiRLC.Wide.Key.key relation ajtai) running fresh proof = some result)
    (initial : Env) (r : Sequence.Prefix initial PiRLCInputs.phaseOffset)
    (rOperations : r.operations = PiRLC.Wide.Formal.opsAt relation PiRLCInputs.interface PiRLCInputs.phaseOffset)
    (rSampled : (PiRLC.Wide.Key.key relation ajtai).piRlcChallenges running fresh proof =
      some (PiRLC.Wide.Semantics.evalChallenges
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset r.current))
    (rParent : PiDECLoaded.parent relation r.current =
      (PiRLC.Wide.Key.key relation ajtai).parentForChallenges running fresh proof
        (PiRLC.Wide.Semantics.evalChallenges
          (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset r.current))
    (afterR : Env)
    (preserved : ∀ index, index < PiRLCInputs.phaseOffset + localLength r.operations → afterR index = r.current index) :
    ∃ d : Sequence.Prefix (PiDECProofInputs.load afterR proof (PiDECLoaded.parent relation afterR).publicInput) PiDECInputs.phaseOffset,
      d.operations = PiDEC.v1_1.Formal.opsAt relation (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset ∧
      holdsFlat d.current r.operations ∧
      PiDEC.v1_1.Semantics.PhaseHolds relation ajtai (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset d.current ∧
      RunningTransitionInputs.piDecRunningOutput relation d.current = result := by
  have currentValues := rValues_eq_of_agree relation r.current afterR (by
    intro index below
    apply preserved index
    rw [rOperations, ← PiRLC.Wide.Formal.main_ops, PiRLC.Wide.Formal.localLength_eq]
    exact below)
  obtain ⟨_, challenges, sampled, checks⟩ := verifierInputs relation ajtai running fresh proof result accepted
  have afterSampled : (PiRLC.Wide.Key.key relation ajtai).piRlcChallenges running fresh proof =
      some (PiRLC.Wide.Semantics.evalChallenges
        (PiRLCInputs.interface (logicalWidth := logicalWidth) (publicFits := publicFits)) PiRLCInputs.phaseOffset afterR) := by
    rw [currentValues.1]
    exact rSampled
  have challengeEq := Option.some.inj (afterSampled.symm.trans sampled)
  have afterParent : PiDECLoaded.parent relation afterR =
      (PiRLC.Wide.Key.key relation ajtai).parentForChallenges running fresh proof challenges := by
    rw [currentValues.2, rParent, ← challengeEq, currentValues.1]
  let parent := PiDECLoaded.parent relation afterR
  let loaded := PiDECProofInputs.load afterR proof parent.publicInput
  have parentChecks : PiDEC.PaperVerifier.Accepted (PiRLC.Wide.Key.key relation ajtai).piDecAlgebra
      (PiRLC.Wide.Key.key relation ajtai).piDecPublicInputSplit (PiRLC.Wide.Key.key relation ajtai).piDecEvaluationArity
      ((PiRLC.Wide.Key.key relation ajtai).piDecAttemptForParent proof parent) := by
    dsimp only [parent]
    rw [afterParent]
    exact checks
  obtain ⟨d, dOperations⟩ := PiDEC.v1_1.Formal.completePrefix relation ajtai
    (PiDECInputs.interface logicalWidth publicFits) loaded PiDECInputs.phaseOffset
    (PiDECInputs.assumptions relation loaded) (PiDECLoaded.loaded_phase relation ajtai afterR proof parentChecks)
  have dRows : holds d.current (Circuit.ops (PiDEC.v1_1.Formal.main relation (PiDECInputs.interface logicalWidth publicFits)) PiDECInputs.phaseOffset) := by
    rw [PiDEC.v1_1.Formal.main_ops, ← dOperations]
    exact holdsFlat_implies_holds d.current d.operations d.rows
  have phase := PiDEC.v1_1.Semantics.spec_implies_phaseHolds relation ajtai _ _ _
    (PiDEC.v1_1.Formal.soundness relation _ _ _ (PiDECInputs.assumptions relation d.current) dRows)
  have beforeInputs : ∀ index, index < PiDECInputs.proofInputStart → d.current index = afterR index := by
    intro index below
    exact (d.agrees index (Or.inl (lt_of_lt_of_le below (Nat.le_add_right _ _)))).trans
      (PiDECProofInputs.load_agreesOutside afterR proof parent.publicInput index (Or.inl below))
  have rEnd : PiRLCInputs.phaseOffset + localLength r.operations ≤ PiDECInputs.proofInputStart := by
    rw [rOperations, ← PiRLC.Wide.Formal.main_ops, PiRLC.Wide.Formal.localLength_eq]
    exact PiDECInputs.parentEnd_le_proofInputStart
  have rRows : holdsFlat d.current r.operations := by
    intro expression member
    exact (expression.eval_eq_of_agree_below _ d.current r.current (r.scope expression member)
      (fun index below => (beforeInputs index (lt_of_lt_of_le below rEnd)).trans (preserved index below))).trans
      (r.rows expression member)
  have outputPreserved := PiDEC.v1_1.Semantics.output_eq_of_agree relation
    (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset loaded d.current
    (PiDECInputs.assumptions relation loaded) (fun index below => (d.agrees index (Or.inl below)).symm)
  have family := outputPreserved.symm.trans (PiDECLoaded.loaded_output relation ajtai afterR proof)
  have output := running_of_children relation ajtai d.current proof parent family
  have computed := Stage1.PiDECProtocolCompleteness.computed_output_eq relation (PiRLC.Wide.Key.key relation ajtai)
    running fresh proof result challenges sampled checks accepted
  dsimp only [parent] at output
  rw [afterParent] at output
  exact ⟨d, dOperations, rRows, phase, output.trans computed⟩

end NightstreamFPrime.Layout.Stage1.Wide.PiDECProtocolCompleteness
