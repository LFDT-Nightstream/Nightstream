import NightstreamFPrime.Layout.Stage1.AccumulatorInputs
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.VerifierView
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics

/-!
Owns the deterministic cross-phase semantics of the Stage 1 accumulator.

The proofs identify the existing zero-copy PiRLC and PiDEC views with the one
production NIFS verifier graph. This module emits no row and does not inspect
the canonical artifact.
-/

namespace NightstreamFPrime.Layout.Stage1.AccumulatorSemantics

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem inputInstance_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : NightstreamFPrime.Lifecycle.PiRLC.v1_1.InputBinding.InputInstance
      logicalWidth publicFits)
    (constraintSystem : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  simp_all

private theorem evaluationFamily_ext
    (left right : StrongReduction.EvaluationFamily K productionShape)
    (pad : left.pad = right.pad)
    (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

private theorem piDecChildMessage_ext
    (left right : PiDEC.PaperVerifier.ChildMessage
      PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (commitment : left.commitment = right.commitment)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

private theorem piDecAttempt_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.Attempt
      logicalWidth publicFits)
    (parent : left.parent = right.parent)
    (messages : left.messages = right.messages) : left = right := by
  cases left
  cases right
  simp_all

private theorem running_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (point : left.point = right.point)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

private theorem cubePoint_ext
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

private theorem piRlcPoint_eq_roundTranscript
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (env : Env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalPoint
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundPoint
          (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
          PiCCSInputs.phaseOffset) env =
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.evalRoundPoint
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptInterface
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.atOffset
            (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
            PiCCSInputs.phaseOffset))
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptOffset
          (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
          PiCCSInputs.phaseOffset) env := by
  have interfaceEq :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.atOffset
          (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
          PiCCSInputs.phaseOffset =
        AccumulatorInputs.piCcsInterface logicalWidth publicFits := by
    rfl
  have startEq :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptOffset
          (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
          PiCCSInputs.phaseOffset =
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptStart
          (AccumulatorInputs.piCcsInterface logicalWidth publicFits) := by
    rw [← NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundTranscriptStart_atOffset
      (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset, interfaceEq]
  apply cubePoint_ext
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalPoint
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.evalRoundPoint
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.roundPoint
  rw [interfaceEq, startEq]
  simp [canonicalFinIndices]

private theorem piCcsOutput_point
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env) (source : Fin Nifs.PaperProfile.arity.total) :
    ((ProductionKey.key relation ajtai).piCcsOutputs
      (AccumulatorInputs.running logicalWidth publicFits env)
      (AccumulatorInputs.fresh logicalWidth publicFits env)
      (AccumulatorInputs.proof relation env) source).point =
      ((ProductionKey.key relation ajtai).piCcsExecution
        (AccumulatorInputs.running logicalWidth publicFits env)
        (AccumulatorInputs.fresh logicalWidth publicFits env)
        (AccumulatorInputs.proof relation env)).coins.roundPoint := by
  rfl

/-- The PiRLC input family is exactly the production key's PiCCS output
family, including the key-derived round point and separate evaluation data. -/
theorem piRlcInputs_eq_keyOutputs
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env)
    (piCcsSpec : NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.SpecHolds
      relation (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env) :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.evalInputs relation
        (PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PiRLCInputs.phaseOffset env =
      (ProductionKey.key relation ajtai).piCcsOutputs
        (AccumulatorInputs.running logicalWidth publicFits env)
        (AccumulatorInputs.fresh logicalWidth publicFits env)
        (AccumulatorInputs.proof relation env) := by
  have pointEq :=
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.VerifierView.roundPoint_eq_key
      relation ajtai
      (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env (AccumulatorInputs.proof relation env)
      piCcsSpec
  rw [AccumulatorInputs.piCcsEvalProof_eq] at pointEq
  have directPointEq := piRlcPoint_eq_roundTranscript
    (logicalWidth := logicalWidth) (publicFits := publicFits) env
  have pointEq := directPointEq.trans pointEq
  funext source
  let joint : Fin productionShape.sourceCount :=
    Fin.cast (ProductionKey.key relation ajtai).total_eq_sourceCount source
  have sourceEq :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.sourceIndex source =
        joint := by
    apply Fin.ext
    rfl
  apply inputInstance_ext
  · rfl
  · change
      (fun row coefficient =>
        ((PiRLCInputs.sourceInput
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.sourceIndex source)
        ).commitment row coefficient).eval env) =
      Fin.addCases
        (AccumulatorInputs.fresh logicalWidth publicFits env).commitments
        (AccumulatorInputs.running logicalWidth publicFits env).commitments
        joint
    rw [PiRLCInputs.sourceInput_eq_canonical, sourceEq]
    refine Fin.addCases (fun fresh => ?_) (fun running => ?_) joint
    · funext row coefficient
      simp [PiRLCInputs.canonicalSourceInput, AccumulatorInputs.fresh,
        AccumulatorInputs.piCcsInterface, PiRLCInputs.piCcsInterface,
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalFresh,
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalFresh]
    · funext row coefficient
      simp [PiRLCInputs.canonicalSourceInput, AccumulatorInputs.running,
        AccumulatorInputs.piCcsInterface, PiRLCInputs.piCcsInterface,
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalRunning,
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalRunning]
  · change
      (fun column =>
        ((PiRLCInputs.sourceInput
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.sourceIndex source)
        ).publicInput column).eval env) =
      Fin.addCases
        (AccumulatorInputs.fresh logicalWidth publicFits env).publicInputs
        (AccumulatorInputs.running logicalWidth publicFits env).publicInputs
        joint
    rw [PiRLCInputs.sourceInput_eq_canonical, sourceEq]
    refine Fin.addCases (fun fresh => ?_) (fun running => ?_) joint
    · funext column
      simp [PiRLCInputs.canonicalSourceInput, AccumulatorInputs.fresh,
        AccumulatorInputs.piCcsInterface, PiRLCInputs.piCcsInterface,
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalFresh,
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalFresh]
    · funext column
      simp [PiRLCInputs.canonicalSourceInput, AccumulatorInputs.running,
        AccumulatorInputs.piCcsInterface, PiRLCInputs.piCcsInterface,
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalRunning,
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalRunning]
  · rw [piCcsOutput_point relation ajtai env source]
    exact pointEq
  · change
      #[NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
        (PiRLCInputs.sourceInput
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.sourceIndex source)
        ).evaluation env] =
      #[{
        pad := (AccumulatorInputs.proof relation env).piCcsOutput.padCoordinate
          joint
        matrix :=
          (AccumulatorInputs.proof relation env).piCcsOutput.matrixCoordinate
            joint }]
    rw [PiRLCInputs.sourceInput_eq_canonical, sourceEq]
    refine Fin.addCases (fun fresh => ?_) (fun running => ?_) joint
    · have injectionEq : UnifiedSources.freshSourceIndex fresh =
          Fin.castAdd productionShape.runningCount fresh := by
        apply Fin.ext
        rfl
      apply congrArg (fun value : StrongReduction.EvaluationFamily K
        productionShape =>
          (#[value] : Array (StrongReduction.EvaluationFamily K productionShape)))
      apply evaluationFamily_ext
      · funext coefficient
        simp [PiRLCInputs.canonicalSourceInput,
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation,
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalOutput,
          AccumulatorInputs.proof, PiRLCInputs.piCcsInterface,
          AccumulatorInputs.piCcsInterface, injectionEq]
      · funext matrix coefficient
        simp [PiRLCInputs.canonicalSourceInput,
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation,
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalOutput,
          AccumulatorInputs.proof, PiRLCInputs.piCcsInterface,
          AccumulatorInputs.piCcsInterface, injectionEq]
    · have injectionEq : UnifiedSources.runningSourceIndex running =
          Fin.natAdd productionShape.freshCount running := by
        apply Fin.ext
        rfl
      apply congrArg (fun value : StrongReduction.EvaluationFamily K
        productionShape =>
          (#[value] : Array (StrongReduction.EvaluationFamily K productionShape)))
      apply evaluationFamily_ext
      · funext coefficient
        simp [PiRLCInputs.canonicalSourceInput,
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation,
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalOutput,
          AccumulatorInputs.proof, PiRLCInputs.piCcsInterface,
          AccumulatorInputs.piCcsInterface, injectionEq]
      · funext matrix coefficient
        simp [PiRLCInputs.canonicalSourceInput,
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation,
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.evalOutput,
          AccumulatorInputs.proof, PiRLCInputs.piCcsInterface,
          AccumulatorInputs.piCcsInterface, injectionEq]
  · rfl

/-- The PiRLC sampler starts at the exact post-PiCCS transcript state and
therefore returns the production key's challenge vector. -/
theorem piRlcChallenges_eq_key
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env)
    (piCcsPhase : NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.PhaseHolds
      relation ajtai (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env (AccumulatorInputs.proof relation env))
    (piRlcPhase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCInputs.phaseOffset env) :
    (ProductionKey.key relation ajtai).piRlcChallenges
        (AccumulatorInputs.running logicalWidth publicFits env)
        (AccumulatorInputs.fresh logicalWidth publicFits env)
        (AccumulatorInputs.proof relation env) =
      some (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.evalChallenges
        (PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PiRLCInputs.phaseOffset env) := by
  have stateEq :
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.evalInitialState
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerInterface
            (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset
              (PiRLCInputs.interface
                (logicalWidth := logicalWidth) (publicFits := publicFits))
              PiRLCInputs.phaseOffset))
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset
            PiRLCInputs.phaseOffset) env =
        ((ProductionKey.key relation ajtai).piCcsExecution
          (AccumulatorInputs.running logicalWidth publicFits env)
          (AccumulatorInputs.fresh logicalWidth publicFits env)
          (AccumulatorInputs.proof relation env)).outgoingState := by
    calc
      _ = NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalState
          env (PiRLCInputs.piCcsOutputState
            (logicalWidth := logicalWidth) (publicFits := publicFits)) := by
            rfl
      _ = NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalState
          env
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.outputBindingFinalState
            relation (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
            PiCCSInputs.phaseOffset) := by
            rw [PiRLCInputs.piCcsOutputState_eq_parent relation]
            rfl
      _ = _ := piCcsPhase.outgoingState
  change
    NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.piRlcChallenges
        ((ProductionKey.key relation ajtai).piCcsExecution
          (AccumulatorInputs.running logicalWidth publicFits env)
          (AccumulatorInputs.fresh logicalWidth publicFits env)
          (AccumulatorInputs.proof relation env)).outgoingState
        Nifs.PaperProfile.arity.total = some _
  rw [← stateEq]
  exact piRlcPhase.response

/-- The constrained PiRLC output is exactly the production key's parent for
the constrained challenge vector. -/
theorem piRlcOutput_eq_keyParentForChallenges
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env)
    (piCcsSpec : NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.SpecHolds
      relation (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env)
    (piRlcPhase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCInputs.phaseOffset env) :
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.evalOutput relation
        (PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PiRLCInputs.phaseOffset env =
      (ProductionKey.key relation ajtai).parentForChallenges
        (AccumulatorInputs.running logicalWidth publicFits env)
        (AccumulatorInputs.fresh logicalWidth publicFits env)
        (AccumulatorInputs.proof relation env)
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.evalChallenges
          (PiRLCInputs.interface
            (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset env) := by
  have inputsEq := piRlcInputs_eq_keyOutputs relation ajtai env piCcsSpec
  let first : Fin Nifs.PaperProfile.arity.total := ⟨0, by decide⟩
  have inputPointEq := congrArg (fun inputs => (inputs first).point) inputsEq
  change
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.evalInputs relation
      (PiRLCInputs.interface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCInputs.phaseOffset env first).point =
    ((ProductionKey.key relation ajtai).piCcsOutputs
      (AccumulatorInputs.running logicalWidth publicFits env)
      (AccumulatorInputs.fresh logicalWidth publicFits env)
      (AccumulatorInputs.proof relation env) first).point at inputPointEq
  rw [piCcsOutput_point relation ajtai env first] at inputPointEq
  have outputPointEq :
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.evalOutput relation
        (PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PiRLCInputs.phaseOffset env).point =
      ((ProductionKey.key relation ajtai).piCcsExecution
        (AccumulatorInputs.running logicalWidth publicFits env)
        (AccumulatorInputs.fresh logicalWidth publicFits env)
        (AccumulatorInputs.proof relation env)).coins.roundPoint := by
    exact inputPointEq
  rw [NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.output_eq_combinedOutput
    relation ajtai
    (PiRLCInputs.interface
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    PiRLCInputs.phaseOffset env piRlcPhase]
  unfold Nifs.PaperNonInteractive.Key.parentForChallenges
  rw [inputsEq, outputPointEq]
  rfl

/-- The PiDEC parent view is the exact zero-copy PiRLC output instance. -/
theorem piDecParent_eq_piRlcOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    (NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.inputAttempt relation
      (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset
      env).parent =
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.evalOutput relation
        (PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PiRLCInputs.phaseOffset env := by
  rfl

/-- The constrained PiDEC input attempt is exactly the production key attempt
for the constrained PiRLC parent and the accumulator proof message. -/
theorem piDecAttempt_eq_keyAttemptForParent
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env) :
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.inputAttempt relation
        (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset
        env =
      (ProductionKey.key relation ajtai).piDecAttemptForParent
        (AccumulatorInputs.proof relation env)
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.evalOutput relation
          (PiRLCInputs.interface
            (logicalWidth := logicalWidth) (publicFits := publicFits))
          PiRLCInputs.phaseOffset env) := by
  apply piDecAttempt_ext
  · exact piDecParent_eq_piRlcOutput relation env
  · funext child
    let running : Fin productionShape.runningCount :=
      Fin.cast (ProductionKey.key relation ajtai).outputCount_eq child
    have childEq : RunningTransitionInputs.childOfRunning running = child := by
      apply Fin.ext
      rfl
    apply piDecChildMessage_ext
    · funext row coefficient
      simp [NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.inputAttempt,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.evalAttempt,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.evalMessage,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.inputBindingInterface,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.atOffset,
        PiDECInputs.interface, PiDECInputs.message,
        Nifs.PaperNonInteractive.Key.piDecAttemptForParent,
        AccumulatorInputs.proof,
        RunningTransitionInputs.piDecRunningOutput,
        RunningTransitionInputs.piDecInterface, childEq, running,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.output,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputBinding.evalOutput,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.outputBindingInterface,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.outputBindingOffset,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.recompositionOffset]
    · simp [NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.inputAttempt,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.evalAttempt,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.InputBinding.evalMessage,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.inputBindingInterface,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.atOffset,
        PiDECInputs.interface, PiDECInputs.message,
        Nifs.PaperNonInteractive.Key.piDecAttemptForParent,
        AccumulatorInputs.proof,
        RunningTransitionInputs.piDecRunningOutput,
        RunningTransitionInputs.piDecInterface, childEq, running,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.output,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputBinding.evalOutput,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.outputBindingInterface,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.outputBindingOffset,
        NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal.recompositionOffset]

/-- Successful PiRLC sampling makes the key's optional PiDEC attempt exactly
the constrained PiDEC input attempt. -/
theorem keyPiDecAttempt_eq_some
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env)
    (piCcsSpec : NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.SpecHolds
      relation (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env)
    (piCcsPhase : NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.PhaseHolds
      relation ajtai (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env (AccumulatorInputs.proof relation env))
    (piRlcPhase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCInputs.phaseOffset env) :
    (ProductionKey.key relation ajtai).piDecAttempt
        (AccumulatorInputs.running logicalWidth publicFits env)
        (AccumulatorInputs.fresh logicalWidth publicFits env)
        (AccumulatorInputs.proof relation env) =
      some (NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.inputAttempt
        relation (PiDECInputs.interface logicalWidth publicFits)
        PiDECInputs.phaseOffset env) := by
  have challengesEq := piRlcChallenges_eq_key relation ajtai env
    piCcsPhase piRlcPhase
  have parentEq := piRlcOutput_eq_keyParentForChallenges relation ajtai env
    piCcsSpec piRlcPhase
  have attemptEq := piDecAttempt_eq_keyAttemptForParent relation ajtai env
  unfold Nifs.PaperNonInteractive.Key.piDecAttempt
    Nifs.PaperNonInteractive.Key.parent
  rw [challengesEq]
  change some ((ProductionKey.key relation ajtai).piDecAttemptForParent
    (AccumulatorInputs.proof relation env)
    ((ProductionKey.key relation ajtai).parentForChallenges
      (AccumulatorInputs.running logicalWidth publicFits env)
      (AccumulatorInputs.fresh logicalWidth publicFits env)
      (AccumulatorInputs.proof relation env)
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.evalChallenges
        (PiRLCInputs.interface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        PiRLCInputs.phaseOffset env))) = some _
  rw [← parentEq, ← attemptEq]

/-- The constrained PiDEC phase makes the production key's executable
PiDEC check true. -/
theorem piDecCheck_eq_true
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env)
    (piCcsSpec : NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.SpecHolds
      relation (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env)
    (piCcsPhase : NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.PhaseHolds
      relation ajtai (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env (AccumulatorInputs.proof relation env))
    (piRlcPhase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCInputs.phaseOffset env)
    (piDecPhase : NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds
      relation ajtai (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset env) :
    Nifs.PaperNonInteractive.piDecCheck
        (ProductionKey.key relation ajtai)
        (AccumulatorInputs.running logicalWidth publicFits env)
        (AccumulatorInputs.fresh logicalWidth publicFits env)
        (AccumulatorInputs.proof relation env) = true := by
  have attemptEq := keyPiDecAttempt_eq_some relation ajtai env piCcsSpec
    piCcsPhase piRlcPhase
  have piDecSpec :=
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.phaseHolds_implies_spec
      relation ajtai (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset env piDecPhase
  have accepted := NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.accepted
    relation ajtai (PiDECInputs.interface logicalWidth publicFits)
    PiDECInputs.phaseOffset env piDecSpec
  exact (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff
    (ProductionKey.key relation ajtai)
    (AccumulatorInputs.running logicalWidth publicFits env)
    (AccumulatorInputs.fresh logicalWidth publicFits env)
    (AccumulatorInputs.proof relation env)).mpr
      ⟨_, attemptEq, accepted⟩

/-- The key output constructor over the constrained PiDEC attempt is exactly
the existing typed 16-child PiDEC running output. -/
theorem outputForAttempt_eq_accumulatorOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env)
    (piDecPhase : NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds
      relation ajtai (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset env) :
    (ProductionKey.key relation ajtai).outputForAttempt
        (AccumulatorInputs.proof relation env)
        (NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.inputAttempt relation
          (PiDECInputs.interface logicalWidth publicFits)
          PiDECInputs.phaseOffset env)
        ((ProductionKey.key relation ajtai).piDecPublicInputSplit.split
          (NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.inputAttempt relation
            (PiDECInputs.interface logicalWidth publicFits)
            PiDECInputs.phaseOffset env).parent.publicInput) =
      AccumulatorInputs.output relation env := by
  apply running_ext
  · rfl
  · funext runningIndex row coefficient
    let child : Fin productionGlobalParams.k :=
      Fin.cast (ProductionKey.key relation ajtai).runningCount_eq_outputCount
        runningIndex
    have childEq : RunningTransitionInputs.childOfRunning runningIndex = child := by
      apply Fin.ext
      rfl
    change
      ((NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.inputAttempt relation
        (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset
        env).messages child).commitment row coefficient =
      (NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.output relation
        (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset
        env (RunningTransitionInputs.childOfRunning runningIndex)).commitment
        row coefficient
    rw [childEq]
    rfl
  · funext runningIndex column
    let child : Fin productionGlobalParams.k :=
      Fin.cast (ProductionKey.key relation ajtai).runningCount_eq_outputCount
        runningIndex
    have childEq : RunningTransitionInputs.childOfRunning runningIndex = child := by
      apply Fin.ext
      rfl
    have publicEq := PiDEC.PaperVerifier.OutputAccepted.childPublicInput_eq
      piDecPhase child
    change
      (PaperAlgebra.publicInputSplit ajtai).split
          (NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.inputAttempt relation
            (PiDECInputs.interface logicalWidth publicFits)
            PiDECInputs.phaseOffset env).parent.publicInput child column =
        (NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.output relation
          (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset
          env (RunningTransitionInputs.childOfRunning runningIndex)).publicInput
          column
    rw [childEq]
    exact (congrFun publicEq column).symm
  · exact AccumulatorInputs.proof_piDecEvaluations relation env

/-- The production key computes exactly the existing typed PiDEC running
output. -/
theorem keyOutput_eq_some
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (env : Env)
    (piCcsSpec : NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.SpecHolds
      relation (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env)
    (piCcsPhase : NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.PhaseHolds
      relation ajtai (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env (AccumulatorInputs.proof relation env))
    (piRlcPhase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCInputs.phaseOffset env)
    (piDecPhase : NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds
      relation ajtai (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset env) :
    (ProductionKey.key relation ajtai).output
        (AccumulatorInputs.running logicalWidth publicFits env)
        (AccumulatorInputs.fresh logicalWidth publicFits env)
        (AccumulatorInputs.proof relation env) =
      some (AccumulatorInputs.output relation env) := by
  have attemptEq := keyPiDecAttempt_eq_some relation ajtai env piCcsSpec
    piCcsPhase piRlcPhase
  have piDecSpec :=
    NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.phaseHolds_implies_spec
      relation ajtai (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset env piDecPhase
  have accepted := NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.accepted
    relation ajtai (PiDECInputs.interface logicalWidth publicFits)
    PiDECInputs.phaseOffset env piDecSpec
  have result := (ProductionKey.key relation ajtai).output_eq_some_of_parentBounded
    (AccumulatorInputs.running logicalWidth publicFits env)
    (AccumulatorInputs.fresh logicalWidth publicFits env)
    (AccumulatorInputs.proof relation env)
    (NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.inputAttempt relation
      (PiDECInputs.interface logicalWidth publicFits) PiDECInputs.phaseOffset env)
    attemptEq accepted.parentBounded
  rw [outputForAttempt_eq_accumulatorOutput relation ajtai env piDecPhase]
    at result
  exact result

/-- The three constrained phase results compose into the exact deterministic
SuperNeo accumulator update. -/
theorem phases_imply_holds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (env : Env)
    (piCcsSpec : NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.SpecHolds
      relation (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env)
    (piCcsPhase : NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.PhaseHolds
      relation ajtai (AccumulatorInputs.piCcsInterface logicalWidth publicFits)
      PiCCSInputs.phaseOffset env (AccumulatorInputs.proof relation env))
    (piRlcPhase : NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics.PhaseHolds
      relation ajtai
      (PiRLCInputs.interface
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PiRLCInputs.phaseOffset env)
    (piDecPhase : NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds
      relation ajtai (PiDECInputs.interface logicalWidth publicFits)
      PiDECInputs.phaseOffset env) :
    NightstreamFPrime.Lifecycle.Stage1.Accumulator.Holds relation ajtai vk
      (AccumulatorInputs.running logicalWidth publicFits env)
      (AccumulatorInputs.fresh logicalWidth publicFits env)
      (AccumulatorInputs.proof relation env)
      (AccumulatorInputs.output relation env) := by
  apply (NightstreamFPrime.Lifecycle.Stage1.Accumulator.holds_iff_checks
    relation ajtai vk
    (AccumulatorInputs.running logicalWidth publicFits env)
    (AccumulatorInputs.fresh logicalWidth publicFits env)
    (AccumulatorInputs.proof relation env)
    (AccumulatorInputs.output relation env)).mpr
  exact ⟨piCcsPhase.accepted,
    piDecCheck_eq_true relation ajtai env piCcsSpec piCcsPhase piRlcPhase
      piDecPhase,
    keyOutput_eq_some relation ajtai env piCcsSpec piCcsPhase piRlcPhase
      piDecPhase⟩

end NightstreamFPrime.Layout.Stage1.AccumulatorSemantics
