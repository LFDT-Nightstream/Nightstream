import NightstreamFPrime.Lifecycle.PiDEC.v1_1.Formal

/-!
Owns the semantic closure of the logical PiDEC v1.1 assembler.

The six child specifications imply the exact operational
`PiDEC.PaperVerifier.OutputAccepted` predicate instantiated by
`ProductionKey.key`. Pad `Eval_K` and the 14-matrix `Eval_A` family remain
separate until they are assembled into the one typed evaluation record.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

private theorem evaluation_ext (left right : PaperAlgebra.Evaluation)
    (pad : left.pad = right.pad)
    (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

private theorem output_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : OutputBinding.Output logicalWidth publicFits)
    (constraintSystem : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  simp_all

private theorem childMessage_ext
    (left right : PiDEC.PaperVerifier.ChildMessage
      PaperAlgebra.Evaluation PaperAlgebra.Commitment)
    (commitment : left.commitment = right.commitment)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

private theorem attempt_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : InputBinding.Attempt logicalWidth publicFits)
    (parent : left.parent = right.parent)
    (messages : left.messages = right.messages) : left = right := by
  cases left
  cases right
  simp_all

private theorem point_ext (left right : PaperAlgebra.Point)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

def inputAttempt
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : InputBinding.Attempt logicalWidth publicFits :=
  InputBinding.evalAttempt relation
    (Formal.inputBindingInterface (Formal.atOffset interface offset))
    (Formal.inputBindingOffset offset) env

def output
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    Radix.ChildIndex → OutputBinding.Output logicalWidth publicFits :=
  OutputBinding.evalOutput relation
    (Formal.outputBindingInterface (Formal.atOffset interface offset))
    (Formal.outputBindingOffset offset) env

abbrev PhaseHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : Prop :=
  PiDEC.PaperVerifier.OutputAccepted
    (PaperAlgebra.piDecAlgebra ajtai)
    (PaperAlgebra.publicInputSplit ajtai)
    (PaperAlgebra.evaluationArity ajtai)
    (inputAttempt relation interface offset env).parent
    (output relation interface offset env)

private theorem evaluationFamily_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
        (interface.parent offset).evaluation env =
      PaperAlgebra.recomposeEvaluationFamily fun child =>
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
          (interface.message offset child).evaluation env := by
  have pad := EvalKRecomposition.parentCoverage
    (Formal.evalKInterface (Formal.atOffset interface offset))
    (Formal.evalKOffset offset) env specification.eval_K
  have matrix := EvalARecomposition.parentCoverage
    (Formal.evalAInterface (Formal.atOffset interface offset))
    (Formal.evalAOffset offset) env specification.eval_A
  apply evaluation_ext
  · simpa [NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation,
      PaperAlgebra.recomposeEvaluationFamily, EvalKRecomposition.evalParent,
      EvalKRecomposition.evalChildren, Formal.evalKInterface, Formal.atOffset,
      Formal.evalKOffset] using pad
  · simpa [NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation,
      PaperAlgebra.recomposeEvaluationFamily, EvalARecomposition.evalParent,
      EvalARecomposition.evalChildren, Formal.evalAInterface, Formal.atOffset,
      Formal.evalAOffset] using matrix

theorem accepted
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    PiDEC.PaperVerifier.Accepted
      (PaperAlgebra.piDecAlgebra ajtai)
      (PaperAlgebra.publicInputSplit ajtai)
      (PaperAlgebra.evaluationArity ajtai)
      (inputAttempt relation interface offset env) := by
  refine {
    parentBounded := ?_
    parentCombined := specification.inputBinding.parentCombined
    parentEvaluationSize :=
      InputBinding.accepted_parentEvaluationSize relation ajtai
        (Formal.inputBindingInterface (Formal.atOffset interface offset))
        (Formal.inputBindingOffset offset) env specification.inputBinding
    messageEvaluationSize :=
      InputBinding.accepted_messageEvaluationSize relation ajtai
        (Formal.inputBindingInterface (Formal.atOffset interface offset))
        (Formal.inputBindingOffset offset) env specification.inputBinding
    commitmentEquation := ?_
    evaluationEquation := ?_ }
  · have bounded := PublicInputSplit.parentBounded
      (Formal.publicInputInterface (Formal.atOffset interface offset))
      (Formal.publicInputOffset offset) env specification.publicInput
    simpa [inputAttempt, InputBinding.evalAttempt, InputBinding.evalParent,
      Formal.inputBindingInterface, Formal.publicInputInterface,
      Formal.atOffset, Formal.inputBindingOffset, Formal.publicInputOffset,
      PublicInputSplit.evalParent, PaperAlgebra.publicInputSplit] using bounded
  · have commitment := CommitmentRecomposition.parentCoverage
      (Formal.commitmentInterface (Formal.atOffset interface offset))
      (Formal.commitmentOffset offset) env specification.commitment
    simpa [inputAttempt, InputBinding.evalAttempt, InputBinding.evalParent,
      InputBinding.evalMessage, Formal.inputBindingInterface,
      Formal.commitmentInterface, Formal.atOffset, Formal.inputBindingOffset,
      Formal.commitmentOffset, CommitmentRecomposition.evalParent,
      CommitmentRecomposition.evalChildren, PaperAlgebra.piDecAlgebra] using
      commitment
  · have family := evaluationFamily_eq relation interface offset env specification
    have singleton := congrArg (fun value => #[value]) family
    simpa [inputAttempt, InputBinding.evalAttempt, InputBinding.evalParent,
      InputBinding.evalMessage, Formal.inputBindingInterface, Formal.atOffset,
      Formal.inputBindingOffset, PaperAlgebra.piDecAlgebra,
      PaperAlgebra.recomposeEvaluations] using singleton

private theorem attemptForOutput_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    PiDEC.PaperVerifier.attemptForOutput
        (inputAttempt relation interface offset env).parent
        (output relation interface offset env) =
      inputAttempt relation interface offset env := by
  rfl

private theorem outputComputed
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    PiDEC.PaperVerifier.children (PaperAlgebra.publicInputSplit ajtai)
        (PiDEC.PaperVerifier.attemptForOutput
          (inputAttempt relation interface offset env).parent
          (output relation interface offset env)) =
      output relation interface offset env := by
  rw [attemptForOutput_eq]
  have publicInputs := PublicInputSplit.children_eq_splitPublicInput
    (Formal.publicInputInterface (Formal.atOffset interface offset))
    (Formal.publicInputOffset offset) env specification.publicInput
  funext child
  apply output_ext
  · rfl
  · rfl
  · simpa [inputAttempt, output, InputBinding.evalAttempt,
      InputBinding.evalParent, Formal.inputBindingInterface,
      Formal.publicInputInterface, Formal.outputBindingInterface,
      Formal.atOffset, Formal.inputBindingOffset, Formal.publicInputOffset,
      Formal.outputBindingOffset, OutputBinding.evalOutput,
      PaperAlgebra.publicInputSplit] using (congrFun publicInputs child).symm
  · rfl
  · rfl
  · rfl

theorem spec_implies_phaseHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    PhaseHolds relation ajtai interface offset env := by
  refine {
    outputComputed := outputComputed relation ajtai interface offset env specification
    checks := ?_ }
  rw [attemptForOutput_eq]
  exact accepted relation ajtai interface offset env specification

private theorem phaseChecksAtInput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (phase : PhaseHolds relation ajtai interface offset env) :
    PiDEC.PaperVerifier.Accepted
      (PaperAlgebra.piDecAlgebra ajtai)
      (PaperAlgebra.publicInputSplit ajtai)
      (PaperAlgebra.evaluationArity ajtai)
      (inputAttempt relation interface offset env) := by
  rw [← attemptForOutput_eq]
  exact phase.checks

private theorem phaseEvaluationFamily_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (phase : PhaseHolds relation ajtai interface offset env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
        (interface.parent offset).evaluation env =
      PaperAlgebra.recomposeEvaluationFamily fun child =>
        NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
          (interface.message offset child).evaluation env := by
  have equation :=
    (phaseChecksAtInput relation ajtai interface offset env phase).evaluationEquation
  have first := congrArg
    (fun values : Array PaperAlgebra.Evaluation =>
      values.getD 0 PaperAlgebra.evaluationZero) equation
  simpa [inputAttempt, InputBinding.evalAttempt, InputBinding.evalParent,
    InputBinding.evalMessage, Formal.inputBindingInterface, Formal.atOffset,
    Formal.inputBindingOffset, PaperAlgebra.piDecAlgebra,
    PaperAlgebra.recomposeEvaluations] using first

theorem phaseHolds_implies_spec
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (phase : PhaseHolds relation ajtai interface offset env) :
    Formal.SpecHolds relation interface offset env := by
  let shared := Formal.atOffset interface offset
  have checks := phaseChecksAtInput relation ajtai interface offset env phase
  have family := phaseEvaluationFamily_eq relation ajtai interface offset env phase
  refine {
    inputBinding := ?_
    publicInput := ?_
    commitment := ?_
    eval_K := ?_
    eval_A := ?_
    outputBinding := ?_ }
  · exact {
      parentCombined := checks.parentCombined
      parentEvaluationSize := by
        simpa [PaperAlgebra.evaluationArity] using checks.parentEvaluationSize
      messageEvaluationSize := by
        intro child
        simpa [PaperAlgebra.evaluationArity] using
          checks.messageEvaluationSize child }
  · have bounded :
        NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.PublicInput.parentBounded
        (PublicInputSplit.evalParent
          (Formal.publicInputInterface shared)
          (Formal.publicInputOffset offset) env) := by
      simpa [shared, inputAttempt, InputBinding.evalAttempt,
        InputBinding.evalParent, Formal.inputBindingInterface,
        Formal.publicInputInterface, Formal.atOffset,
        Formal.inputBindingOffset, Formal.publicInputOffset,
        PublicInputSplit.evalParent, PaperAlgebra.publicInputSplit] using
        checks.parentBounded
    have children : PublicInputSplit.evalChildren
          (Formal.publicInputInterface shared)
          (Formal.publicInputOffset offset) env =
        NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput
          (PublicInputSplit.evalParent
            (Formal.publicInputInterface shared)
            (Formal.publicInputOffset offset) env) := by
      funext child coordinate
      have value := congrFun
        (PiDEC.PaperVerifier.OutputAccepted.childPublicInput_eq phase child)
        coordinate
      simpa [shared, output, inputAttempt, InputBinding.evalAttempt,
        InputBinding.evalParent, Formal.inputBindingInterface,
        Formal.publicInputInterface, Formal.outputBindingInterface,
        Formal.atOffset, Formal.inputBindingOffset, Formal.publicInputOffset,
        Formal.outputBindingOffset, PublicInputSplit.evalParent,
        PublicInputSplit.evalChildren, OutputBinding.evalOutput,
        PaperAlgebra.publicInputSplit,
        NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.PublicInput.splitPublicInput] using value
    exact PublicInputSplit.relationHolds_of_parentBounded_children_eq
      (Formal.publicInputInterface shared) (Formal.publicInputOffset offset)
      env bounded children
  · apply CommitmentRecomposition.specHolds_of_parentCoverage
    simpa [shared, inputAttempt, InputBinding.evalAttempt,
      InputBinding.evalParent, InputBinding.evalMessage,
      Formal.inputBindingInterface, Formal.commitmentInterface,
      Formal.atOffset, Formal.inputBindingOffset, Formal.commitmentOffset,
      CommitmentRecomposition.evalParent,
      CommitmentRecomposition.evalChildren,
      PaperAlgebra.piDecAlgebra] using checks.commitmentEquation
  · apply EvalKRecomposition.specHolds_of_parentCoverage
    have pad := congrArg
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction.EvaluationFamily.pad
      family
    simpa [shared,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation,
      PaperAlgebra.recomposeEvaluationFamily, Formal.evalKInterface,
      Formal.atOffset, Formal.evalKOffset, EvalKRecomposition.evalParent,
      EvalKRecomposition.evalChildren] using pad
  · apply EvalARecomposition.specHolds_of_parentCoverage
    have matrix := congrArg
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction.EvaluationFamily.matrix
      family
    simpa [shared,
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation,
      PaperAlgebra.recomposeEvaluationFamily, Formal.evalAInterface,
      Formal.atOffset, Formal.evalAOffset, EvalARecomposition.evalParent,
      EvalARecomposition.evalChildren] using matrix
  · exact ⟨fun _ => rfl, fun _ => rfl, fun _ => rfl, fun _ => rfl⟩

private theorem parentEvaluation_eq_of_agree
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {interface : Formal.Interface logicalWidth publicFits}
    {offset : Nat} {env : Env}
    (assumptions : Formal.Assumptions relation interface offset env)
    (left right : Env)
    (agrees : ∀ index, index < offset → left index = right index) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
        (interface.parent offset).evaluation left =
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
        (interface.parent offset).evaluation right := by
  apply evaluation_ext
  · funext coefficient
    exact KExpr.eval_eq_of_agree_below _ offset left right
      (assumptions.inputs.parentEval_K coefficient) agrees
  · funext matrix coefficient
    exact KExpr.eval_eq_of_agree_below _ offset left right
      (assumptions.inputs.parentEval_A matrix coefficient) agrees

private theorem messageEvaluation_eq_of_agree
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {interface : Formal.Interface logicalWidth publicFits}
    {offset : Nat} {env : Env}
    (assumptions : Formal.Assumptions relation interface offset env)
    (child : Radix.ChildIndex) (left right : Env)
    (agrees : ∀ index, index < offset → left index = right index) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
        (interface.message offset child).evaluation left =
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
        (interface.message offset child).evaluation right := by
  apply evaluation_ext
  · funext coefficient
    exact KExpr.eval_eq_of_agree_below _ offset left right
      (assumptions.inputs.messageEval_K child coefficient) agrees
  · funext matrix coefficient
    exact KExpr.eval_eq_of_agree_below _ offset left right
      (assumptions.inputs.messageEval_A child matrix coefficient) agrees

theorem inputAttempt_eq_of_agree
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env current : Env)
    (assumptions : Formal.Assumptions relation interface offset env)
    (agrees : ∀ index, index < offset → env index = current index) :
    inputAttempt relation interface offset env =
      inputAttempt relation interface offset current := by
  apply attempt_ext
  · apply output_ext
    · rfl
    · funext row lane
      exact Expr.eval_eq_of_agree_below _ offset env current
        (assumptions.inputs.parentCommitment row lane) agrees
    · funext coordinate
      exact Expr.eval_eq_of_agree_below _ offset env current
        (assumptions.inputs.parentPublicInput coordinate) agrees
    · apply point_ext
      change (List.ofFn fun coordinate =>
          (interface.point offset coordinate).eval env) =
        List.ofFn fun coordinate =>
          (interface.point offset coordinate).eval current
      apply congrArg List.ofFn
      funext coordinate
      exact KExpr.eval_eq_of_agree_below _ offset env current
        (assumptions.inputs.point coordinate) agrees
    · exact congrArg (fun value => #[value])
        (parentEvaluation_eq_of_agree assumptions env current agrees)
    · rfl
  · funext child
    apply childMessage_ext
    · funext row lane
      exact Expr.eval_eq_of_agree_below _ offset env current
        (assumptions.inputs.messageCommitment child row lane) agrees
    · exact congrArg (fun value => #[value])
        (messageEvaluation_eq_of_agree assumptions child env current agrees)

theorem output_eq_of_agree
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env current : Env)
    (assumptions : Formal.Assumptions relation interface offset env)
    (agrees : ∀ index, index < offset → env index = current index) :
    output relation interface offset env =
      output relation interface offset current := by
  funext child
  apply output_ext
  · rfl
  · funext row lane
    exact Expr.eval_eq_of_agree_below _ offset env current
      (assumptions.inputs.messageCommitment child row lane) agrees
  · funext coordinate
    exact Expr.eval_eq_of_agree_below _ offset env current
      (assumptions.inputs.digit child coordinate) agrees
  · apply point_ext
    change (List.ofFn fun coordinate =>
        (interface.point offset coordinate).eval env) =
      List.ofFn fun coordinate =>
        (interface.point offset coordinate).eval current
    apply congrArg List.ofFn
    funext coordinate
    exact KExpr.eval_eq_of_agree_below _ offset env current
      (assumptions.inputs.point coordinate) agrees
  · exact congrArg (fun value => #[value])
      (messageEvaluation_eq_of_agree assumptions child env current agrees)
  · rfl

theorem phaseHolds_of_agree
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env current : Env)
    (assumptions : Formal.Assumptions relation interface offset env)
    (agrees : ∀ index, index < offset → env index = current index)
    (phase : PhaseHolds relation ajtai interface offset env) :
    PhaseHolds relation ajtai interface offset current := by
  have attemptEq := inputAttempt_eq_of_agree relation interface offset
    env current assumptions agrees
  have outputEq := output_eq_of_agree relation interface offset
    env current assumptions agrees
  change PiDEC.PaperVerifier.OutputAccepted
    (PaperAlgebra.piDecAlgebra ajtai)
    (PaperAlgebra.publicInputSplit ajtai)
    (PaperAlgebra.evaluationArity ajtai)
    (inputAttempt relation interface offset current).parent
    (output relation interface offset current)
  rw [← attemptEq, ← outputEq]
  exact phase

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.Semantics
