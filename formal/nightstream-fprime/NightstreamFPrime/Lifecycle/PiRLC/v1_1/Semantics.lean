import NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal
import NightstreamFPrime.Spec.Folding.Nifs.PaperProfile

/-!
Owns the semantic result of the exact PiRLC v1.1 logical assembler.

The public attempt uses the production `K + k = 17` arity. Its inputs are
the input-binding values, its challenges are replayed sampler outputs, and
its output is the verifier-computed commitment, public input, separate Pad
evaluation, and separate 14-matrix evaluation family. This file emits no
rows and defines no second PiRLC relation.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem arityTotal_eq_sourceCount :
    Nifs.PaperProfile.arity.total = SamplerChain.sourceCount := by
  rw [Nifs.PaperProfile.arity_total, SamplerChain.sourceCount_eq]

def sourceIndex (source : Fin Nifs.PaperProfile.arity.total) :
    Fin SamplerChain.sourceCount :=
  Fin.cast arityTotal_eq_sourceCount source

def evalInputs
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    Fin Nifs.PaperProfile.arity.total →
      InputBinding.InputInstance logicalWidth publicFits :=
  fun source =>
    InputBinding.evalInputs relation
      (Formal.inputBindingInterface (Formal.atOffset interface offset))
      (Formal.inputBindingOffset offset) env (sourceIndex source)

def evalChallenges
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    Fin Nifs.PaperProfile.arity.total → RingF :=
  fun source =>
    SamplerChain.evalChallenges
      (Formal.samplerInterface (Formal.atOffset interface offset))
      (Formal.samplerOffset offset) env (sourceIndex source)

def evalOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    InputBinding.InputInstance logicalWidth publicFits :=
  OutputBinding.evalOutput relation
    (Formal.outputBindingInterface (Formal.atOffset interface offset) offset)
    (Formal.outputBindingOffset offset) env

theorem combinationChallenge_eval
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) (source : Fin 17)
    (lane : Fin ringDegree) :
    (SamplerChain.challengeExpr
        (Formal.samplerInterface (Formal.atOffset interface offset)) offset
        (Fin.cast CombinationFamily.sourceCount_eq.symm source) lane).eval env =
      evalChallenges interface offset env source lane := by
  exact SamplerChain.challengeExpr_eval
    (Formal.samplerInterface (Formal.atOffset interface offset)) offset env
    (Fin.cast CombinationFamily.sourceCount_eq.symm source) lane

theorem commitmentChallenges_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    CommitmentCombination.evalChallenges
        (Formal.commitmentInterface (Formal.atOffset interface offset))
        (Formal.commitmentOffset offset) env =
      evalChallenges interface offset env := by
  funext source lane
  exact combinationChallenge_eval interface offset env source lane

theorem publicInputChallenges_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    PublicInputCombination.evalChallenges
        (Formal.publicInputInterface (Formal.atOffset interface offset))
        (Formal.publicInputOffset offset) env =
      evalChallenges interface offset env := by
  funext source lane
  exact combinationChallenge_eval interface offset env source lane

theorem evalKChallenges_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    EvalKCombination.evalChallenges
        (Formal.evalKInterface (Formal.atOffset interface offset))
        (Formal.evalKOffset offset) env =
      evalChallenges interface offset env := by
  funext source lane
  exact combinationChallenge_eval interface offset env source lane

theorem evalAChallenges_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    EvalACombination.evalChallenges
        (Formal.evalAInterface (Formal.atOffset interface offset))
        (Formal.evalAOffset offset) env =
      evalChallenges interface offset env := by
  funext source lane
  exact combinationChallenge_eval interface offset env source lane

def evalInputFamily
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (source : Fin Nifs.PaperProfile.arity.total) : PaperAlgebra.Evaluation :=
  NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.evalEvaluation
    ((Formal.atOffset interface offset).input
      (Formal.inputBindingOffset offset) (sourceIndex source)).evaluation env

def evalOutputFamily
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : PaperAlgebra.Evaluation :=
  OutputBinding.evalEvaluation
    (Formal.outputBindingInterface (Formal.atOffset interface offset) offset)
    (Formal.outputBindingOffset offset) env

theorem inputEvaluations_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (source : Fin Nifs.PaperProfile.arity.total) :
    (evalInputs relation interface offset env source).evaluations =
      #[evalInputFamily interface offset env source] := by
  rfl

theorem outputEvaluations_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    (evalOutput relation interface offset env).evaluations =
      #[evalOutputFamily interface offset env] := by
  rfl

theorem commitmentInputs_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    CommitmentCombination.evalInputs
        (Formal.commitmentInterface (Formal.atOffset interface offset))
        (Formal.commitmentOffset offset) env =
      fun source => (evalInputs relation interface offset env source).commitment := by
  rfl

theorem publicInputInputs_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    PublicInputCombination.evalInputs
        (Formal.publicInputInterface (Formal.atOffset interface offset))
        (Formal.publicInputOffset offset) env =
      fun source => (evalInputs relation interface offset env source).publicInput := by
  rfl

theorem evalKInputs_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    EvalKCombination.evalInputs
        (Formal.evalKInterface (Formal.atOffset interface offset))
        (Formal.evalKOffset offset) env =
      fun source => (evalInputFamily interface offset env source).pad := by
  rfl

theorem evalAInputs_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    EvalACombination.evalInputs
        (Formal.evalAInterface (Formal.atOffset interface offset))
        (Formal.evalAOffset offset) env =
      fun source matrix =>
        (evalInputFamily interface offset env source).matrix matrix := by
  rfl

theorem commitmentOutput_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    (evalOutput relation interface offset env).commitment =
      CommitmentCombination.evalOutput
        (Formal.commitmentInterface (Formal.atOffset interface offset))
        (Formal.commitmentOffset offset) env := by
  rfl

theorem publicInputOutput_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    (evalOutput relation interface offset env).publicInput =
      PublicInputCombination.evalOutput
        (Formal.publicInputInterface (Formal.atOffset interface offset))
        (Formal.publicInputOffset offset) env := by
  rfl

theorem evalKOutput_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    (evalOutputFamily interface offset env).pad =
      EvalKCombination.evalOutput
        (Formal.evalKInterface (Formal.atOffset interface offset))
        (Formal.evalKOffset offset) env := by
  rfl

theorem evalAOutput_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    (evalOutputFamily interface offset env).matrix =
      EvalACombination.evalOutput
        (Formal.evalAInterface (Formal.atOffset interface offset))
        (Formal.evalAOffset offset) env := by
  rfl

private theorem evaluationFamily_ext
    (left right : PaperAlgebra.Evaluation)
    (pad : left.pad = right.pad)
    (matrix : left.matrix = right.matrix) : left = right := by
  cases left
  cases right
  simp_all

theorem outputFamily_eq_combine
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    evalOutputFamily interface offset env =
      PaperAlgebra.combineEvaluationFamily
        (evalChallenges interface offset env)
        (evalInputFamily interface offset env) := by
  apply evaluationFamily_ext
  · have coverage := EvalKCombination.parentCoverage
      (Formal.evalKInterface (Formal.atOffset interface offset))
      (Formal.evalKOffset offset) env specification.eval_K
    rw [evalKChallenges_eq interface offset env,
      evalKInputs_eq interface offset env] at coverage
    simpa [PaperAlgebra.combineEvaluationFamily] using
      (evalKOutput_eq interface offset env).trans coverage
  · have coverage := EvalACombination.parentCoverage
      (Formal.evalAInterface (Formal.atOffset interface offset))
      (Formal.evalAOffset offset) env specification.eval_A
    rw [evalAChallenges_eq interface offset env,
      evalAInputs_eq interface offset env] at coverage
    simpa [PaperAlgebra.combineEvaluationFamily] using
      (evalAOutput_eq interface offset env).trans coverage

theorem evaluationEquation
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    (evalOutput relation interface offset env).evaluations =
      PaperAlgebra.combineEvaluations
        (evalChallenges interface offset env)
        (fun source => (evalInputs relation interface offset env source).evaluations) := by
  rw [outputEvaluations_eq]
  have inputsEq :
      (fun source =>
        (evalInputs relation interface offset env source).evaluations) =
        fun source => #[evalInputFamily interface offset env source] := by
    funext source
    exact inputEvaluations_eq relation interface offset env source
  rw [inputsEq,
    PaperAlgebra.combineEvaluations_singletons (by decide :
      0 < Nifs.PaperProfile.arity.total),
    outputFamily_eq_combine relation interface offset env specification]

theorem commitmentEquation
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    (evalOutput relation interface offset env).commitment =
      (PaperAlgebra.piRlcAlgebra ajtai).combineCommitment
        (evalChallenges interface offset env)
        (fun source =>
          (evalInputs relation interface offset env source).commitment) := by
  have coverage := CommitmentCombination.parentCoverage
    (Formal.commitmentInterface (Formal.atOffset interface offset))
    (Formal.commitmentOffset offset) env specification.commitment
  rw [commitmentChallenges_eq interface offset env,
    commitmentInputs_eq relation interface offset env] at coverage
  simpa [PaperAlgebra.piRlcAlgebra] using
    (commitmentOutput_eq relation interface offset env).trans coverage

theorem publicInputEquation
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    (evalOutput relation interface offset env).publicInput =
      (PaperAlgebra.piRlcAlgebra ajtai).combinePublicInput
        (evalChallenges interface offset env)
        (fun source =>
          (evalInputs relation interface offset env source).publicInput) := by
  have coverage := PublicInputCombination.parentCoverage
    (Formal.publicInputInterface (Formal.atOffset interface offset))
    (Formal.publicInputOffset offset) env specification.publicInput
  rw [publicInputChallenges_eq interface offset env,
    publicInputInputs_eq relation interface offset env] at coverage
  simpa [PaperAlgebra.piRlcAlgebra] using
    (publicInputOutput_eq relation interface offset env).trans coverage

theorem inputBinding
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    PiRLC.v1_1.InputBinding.Holds
      (evalInputs relation interface offset env)
      (evalOutput relation interface offset env).constraintSystem
      (evalOutput relation interface offset env).point := by
  have binding := InputBinding.parentCoverage relation
    (Formal.inputBindingInterface (Formal.atOffset interface offset))
    (Formal.inputBindingOffset offset) env specification.inputBinding
  have output := OutputBinding.parentCoverage relation
    (Formal.outputBindingInterface (Formal.atOffset interface offset) offset)
    (Formal.outputBindingOffset offset) env specification.outputBinding
  refine {
    inputFresh := ?_
    sameStructure := ?_
    samePoint := ?_ }
  · intro source
    exact binding.inputFresh (sourceIndex source)
  · intro source
    exact (binding.sameStructure (sourceIndex source)).trans output.2.1.symm
  · intro source
    exact (binding.samePoint (sourceIndex source)).trans output.2.2.symm

theorem outputCombined
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    (evalOutput relation interface offset env).stage = .combined := by
  exact (OutputBinding.parentCoverage relation
    (Formal.outputBindingInterface (Formal.atOffset interface offset) offset)
    (Formal.outputBindingOffset offset) env specification.outputBinding).1

/-- The one public attempt checked by the canonical PiRLC relation. -/
def attempt
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) :
    PiRLC.Attempt (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment RingF
      productionGlobalParams Nifs.PaperProfile.arity where
  inputs := evalInputs relation interface offset env
  challenges := evalChallenges interface offset env
  output := evalOutput relation interface offset env

theorem spec_implies_equations
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    PiRLC.Equations (PaperAlgebra.piRlcAlgebra ajtai)
      (attempt relation interface offset env) := by
  refine PiRLC.v1_1.InputBinding.Holds.toEquations
    (algebra := PaperAlgebra.piRlcAlgebra ajtai)
    (attempt := attempt relation interface offset env) ?_ ?_ ?_ ?_ ?_
  · simpa [attempt] using inputBinding relation interface offset env specification
  · simpa [attempt] using outputCombined relation interface offset env specification
  · simpa [attempt] using
      commitmentEquation relation ajtai interface offset env specification
  · simpa [attempt] using
      publicInputEquation relation ajtai interface offset env specification
  · simpa [attempt, PaperAlgebra.piRlcAlgebra] using
      evaluationEquation relation interface offset env specification

/-- Complete deterministic meaning of one logical PiRLC phase. Successful
transcript replay remains explicit because membership alone does not prove
that the verifier derived the challenges. -/
structure PhaseHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env) : Prop where
  sampler : SamplerChain.RelationHolds
    (Formal.samplerInterface (Formal.atOffset interface offset))
    (Formal.samplerOffset offset) env
  accepted : PiRLC.Accepted (PaperAlgebra.piRlcAlgebra ajtai)
    (attempt relation interface offset env)

theorem PhaseHolds.response
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {interface : Formal.Interface logicalWidth publicFits}
    {offset : Nat} {env : Env}
    (phase : PhaseHolds relation ajtai interface offset env) :
    NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.piRlcChallenges
        (SamplerChain.evalInitialState
          (Formal.samplerInterface (Formal.atOffset interface offset))
          (Formal.samplerOffset offset) env)
        Nifs.PaperProfile.arity.total =
      some (evalChallenges interface offset env) := by
  simpa [evalChallenges, sourceIndex, arityTotal_eq_sourceCount] using
    phase.sampler.response

theorem PhaseHolds.outgoingState
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {interface : Formal.Interface logicalWidth publicFits}
    {offset : Nat} {env : Env}
    (phase : PhaseHolds relation ajtai interface offset env) :
    SamplerChain.evalFinalState
        (Formal.samplerInterface (Formal.atOffset interface offset))
        (Formal.samplerOffset offset) env =
      Nifs.NonInteractive.PiRlcSampler.stateAt
        NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.specification
        (SamplerChain.evalInitialState
          (Formal.samplerInterface (Formal.atOffset interface offset))
          (Formal.samplerOffset offset) env)
        Nifs.PaperProfile.arity.total := by
  simpa [arityTotal_eq_sourceCount] using phase.sampler.finalState

theorem challengesValid
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env) :
    ∀ source,
      (PaperAlgebra.piRlcAlgebra ajtai).challengeValid
        (evalChallenges interface offset env source) := by
  intro source
  have success := SamplerChain.sampleRingChallenge_eq
    (Formal.samplerInterface (Formal.atOffset interface offset))
    (Formal.samplerOffset offset) env specification.sampler.child
    (sourceIndex source)
  simpa [PaperAlgebra.piRlcAlgebra,
    Phi81Relation.PiRLCAlgebra.Challenge.challengeValid,
    evalChallenges] using
      NightstreamFPrime.Lifecycle.Transcript.PiRlcSampler.sampleRingChallenge_member
        success

/-- Mechanical coverage of the exact production PiRLC relation and its
transcript transition. No challenge or outgoing state is supplied as a
premise. -/
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
  have equations :=
    spec_implies_equations relation ajtai interface offset env specification
  refine {
    sampler := specification.sampler
    accepted := equations.withChallengesValid ?_
  }
  simpa [attempt] using
    challengesValid relation ajtai interface offset env specification

private theorem inputInstance_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : InputBinding.InputInstance logicalWidth publicFits)
    (constraintSystem : left.constraintSystem = right.constraintSystem)
    (commitment : left.commitment = right.commitment)
    (publicInput : left.publicInput = right.publicInput)
    (point : left.point = right.point)
    (evaluations : left.evaluations = right.evaluations)
    (stage : left.stage = right.stage) : left = right := by
  cases left
  cases right
  simp_all

theorem output_eq_combinedOutput
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (phase : PhaseHolds relation ajtai interface offset env) :
    evalOutput relation interface offset env =
      PiRLC.combinedOutput (PaperAlgebra.piRlcAlgebra ajtai)
        (evalOutput relation interface offset env).constraintSystem
        (evalOutput relation interface offset env).point
        (evalInputs relation interface offset env)
        (evalChallenges interface offset env) := by
  apply inputInstance_ext
  · rfl
  · exact phase.accepted.commitmentEquation
  · exact phase.accepted.publicInputEquation
  · rfl
  · exact phase.accepted.evaluationEquation
  · exact phase.accepted.outputCombined

/-- Honest input openings combine to an opening of the exact circuit output.
The combined witness is prover-side semantic data and emits no verifier row. -/
theorem combinedWitness_holds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Formal.Interface logicalWidth publicFits)
    (offset : Nat) (env : Env)
    (specification : Formal.SpecHolds relation interface offset env)
    (assignments : Fin Nifs.PaperProfile.arity.total →
      PaperAlgebra.Assignment
        (logicalWidth := logicalWidth) (publicFits := publicFits))
    (inputValid : ∀ source,
      CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (evalInputs relation interface offset env source)
        (assignments source))
    (pointValid : (PaperAlgebra.semantics ajtai).evaluationPointValid
      (evalOutput relation interface offset env).constraintSystem
      (evalOutput relation interface offset env).point) :
    CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (evalOutput relation interface offset env)
      (PiRLC.combinedWitness (PaperAlgebra.piRlcAlgebra ajtai)
        (evalChallenges interface offset env) assignments) := by
  have phase :=
    spec_implies_phaseHolds relation ajtai interface offset env specification
  have canonical := PiRLC.combinedOutput_holds
    (PaperAlgebra.semantics ajtai) productionGlobalParams
    (PaperAlgebra.piRlcAlgebra ajtai) Nifs.PaperProfile.arity
    (evalOutput relation interface offset env).constraintSystem
    (evalOutput relation interface offset env).point
    (evalInputs relation interface offset env)
    (evalChallenges interface offset env) assignments
    phase.accepted.inputFresh phase.accepted.sameStructure
    phase.accepted.samePoint phase.accepted.challengesValid inputValid pointValid
  rw [output_eq_combinedOutput relation ajtai interface offset env phase]
  exact canonical

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.Semantics
