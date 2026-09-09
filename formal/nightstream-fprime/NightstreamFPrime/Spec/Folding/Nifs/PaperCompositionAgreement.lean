import NightstreamFPrime.Spec.Folding.Nifs.InteractiveDistribution
import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakOutput
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkBinding

/-!
The two B.1 executions use independent verifier coins and independent suffix
endpoints from one original context. Their exact pair law retains the receipt
and endpoint, then maps them to the literal decoded output witness.

A disagreement between positive-mass weak returns exposes relaxed-binding
collision evidence for those actual endpoint returns and their common Phi.
The efficient collision-output reduction and its numerical hardness bound
remain the separate NIFS binding obligation.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionAgreement

open scoped BigOperators
attribute [local instance] Classical.propDecidable
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open PiCCS.PaperJoint PiCCS.PaperJoint.StrongReduction

section PairLaw

variable {State Endpoint : Type*} [Fintype Endpoint] [DecidableEq Endpoint]
  {shape : Shape} {columns width : Nat}
  (firstPhase : InteractivePrefix.Prover State shape width)
  (abortEndpoint : Endpoint)
  (suffixLaw : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape → State → PMF Endpoint)
  (consume : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape → State →
    Endpoint → Option (OutputWitness shape columns))

/-- The selected endpoint stays attached to the actual public receipt. -/
abbrev Observation (State Endpoint : Type*) (shape : Shape) :=
  Option ((Probe K shape × State) × Endpoint)

def outputOf : Observation State Endpoint shape → Option (Probe K shape × OutputWitness shape columns)
  | none => none
  | some (receipt, endpoint) =>
      (consume receipt.1.coins receipt.1.response.fullOutput receipt.2 endpoint).map
        fun witness => (receipt.1, witness)

/-- Only endpoints with positive probability need a pointwise event proof.
The aborted prefix is retained as its own observation. -/
def Supported : Observation State Endpoint shape → Prop
  | none => True
  | some (receipt, endpoint) =>
      0 < (suffixLaw receipt.1.coins receipt.1.response.fullOutput receipt.2 endpoint).toReal

noncomputable def endpointMean (value : Observation State Endpoint shape → ℝ)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) : ℝ :=
  match InteractivePrefix.run firstPhase alpha gamma point with
  | none => value none
  | some receipt =>
      ∑ endpoint,
        (suffixLaw receipt.1.coins receipt.1.response.fullOutput receipt.2 endpoint).toReal *
          value (some (receipt, endpoint))

/-- Both means resample their complete verifier stream and selected suffix
endpoint. The outer receipt or endpoint never supplies the inner draw. -/
noncomputable def pairMean (value : Observation State Endpoint shape →
    Observation State Endpoint shape → ℝ) : ℝ :=
  StrongProbability.verifierMean (endpointMean firstPhase suffixLaw fun left =>
    StrongProbability.verifierMean (endpointMean firstPhase suffixLaw (value left)))

private theorem sequentialMean_eq_endpointMean
    (value : Option (Probe K shape × OutputWitness shape columns) → ℝ) :
    InteractiveDistribution.sequentialMean firstPhase suffixLaw consume value =
      endpointMean firstPhase suffixLaw (fun observation => value (outputOf consume observation)) := by
  funext alpha gamma point
  cases returned : InteractivePrefix.run firstPhase alpha gamma point <;>
    simp only [InteractiveDistribution.sequentialMean, endpointMean, returned, outputOf]

/-- Apply the proved one-run law twice. The observable may depend jointly on
both actual outputs; no conditioning on witness agreement is used. -/
theorem paired_executionMean_eq
    (value : Option (Probe K shape × OutputWitness shape columns) →
      Option (Probe K shape × OutputWitness shape columns) → ℝ) :
    StrongProbability.executionMean
      (InteractiveDistribution.tapes firstPhase abortEndpoint suffixLaw)
      (InteractiveDistribution.coupled firstPhase consume)
      (fun left => StrongProbability.executionMean
        (InteractiveDistribution.tapes firstPhase abortEndpoint suffixLaw)
        (InteractiveDistribution.coupled firstPhase consume) (value left)) =
      pairMean firstPhase suffixLaw fun left right =>
        value (outputOf consume left) (outputOf consume right) := by
  simp_rw [InteractiveDistribution.executionMean_eq_sequentialMean,
    sequentialMean_eq_endpointMean]
  rfl

/-- The strong theorem's disagreement mass is exactly the sequential pair
mass of those decoded endpoint values. -/
theorem disagreementProbability_eq_pairMean
    {Commitment PublicInput : Type*} {blockCount : Nat}
    (maps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
    (statement : Statement K Commitment PublicInput shape columns blockCount ConcreteCarrier.baseOps) :
    StrongProbability.disagreementProbability
      (InteractiveDistribution.tapes firstPhase abortEndpoint suffixLaw)
      (InteractiveDistribution.coupled firstPhase consume) maps params statement =
      pairMean firstPhase suffixLaw fun left right =>
        if StrongProbability.SuccessfulDisagreement (width := width) maps params statement
          (outputOf consume left) (outputOf consume right) then (1 : ℝ) else 0 := by
  change StrongProbability.executionMean
    (InteractiveDistribution.tapes firstPhase abortEndpoint suffixLaw)
    (InteractiveDistribution.coupled firstPhase consume)
    (fun left => StrongProbability.executionMean
      (InteractiveDistribution.tapes firstPhase abortEndpoint suffixLaw)
      (InteractiveDistribution.coupled firstPhase consume)
      (fun right => if StrongProbability.SuccessfulDisagreement (width := width)
        maps params statement left right then (1 : ℝ) else 0)) = _
  exact paired_executionMean_eq firstPhase abortEndpoint suffixLaw consume _

private theorem endpointMean_mono
    (left right : Observation State Endpoint shape → ℝ)
    (implies : ∀ observation, Supported suffixLaw observation → left observation ≤ right observation)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K)
    (point : CubePoint K shape.cubeVariables) :
    endpointMean firstPhase suffixLaw left alpha gamma point ≤
      endpointMean firstPhase suffixLaw right alpha gamma point := by
  cases returned : InteractivePrefix.run firstPhase alpha gamma point with
  | none =>
      simpa only [endpointMean, returned] using implies none True.intro
  | some receipt =>
      simp only [endpointMean, returned]
      apply Finset.sum_le_sum
      intro endpoint _
      by_cases positive :
          0 < (suffixLaw receipt.1.coins receipt.1.response.fullOutput receipt.2 endpoint).toReal
      · exact mul_le_mul_of_nonneg_left (implies (some (receipt, endpoint)) positive)
          ENNReal.toReal_nonneg
      · have zero : (suffixLaw receipt.1.coins receipt.1.response.fullOutput receipt.2 endpoint).toReal = 0 :=
          le_antisymm (le_of_not_gt positive) ENNReal.toReal_nonneg
        simp only [zero, zero_mul, le_refl]

/-- A proved implication on actual positive endpoint pairs gives event
domination under that same independent pair law. Zero-mass terms vanish. -/
theorem pairMean_mono
    (left right : Observation State Endpoint shape → Observation State Endpoint shape → ℝ)
    (implies : ∀ first second, Supported suffixLaw first → Supported suffixLaw second →
      left first second ≤ right first second) :
    pairMean firstPhase suffixLaw left ≤ pairMean firstPhase suffixLaw right := by
  apply StrongProbability.verifierMean_mono
  intro alpha gamma point
  apply endpointMean_mono firstPhase suffixLaw
  intro first firstSupported
  apply StrongProbability.verifierMean_mono
  intro otherAlpha otherGamma otherPoint
  exact endpointMean_mono firstPhase suffixLaw (left first) (right first)
    (fun second secondSupported => implies first second firstSupported secondSupported)
    otherAlpha otherGamma otherPoint

private theorem endpointMean_const (constant : ℝ) :
    endpointMean firstPhase suffixLaw (fun _ => constant) = fun _ _ _ => constant := by
  funext alpha gamma point
  cases returned : InteractivePrefix.run firstPhase alpha gamma point with
  | none => simp only [endpointMean, returned]
  | some receipt =>
      have weights : ∑ endpoint,
          (suffixLaw receipt.1.coins receipt.1.response.fullOutput receipt.2 endpoint).toReal = 1 := by
        let law := suffixLaw receipt.1.coins receipt.1.response.fullOutput receipt.2
        rw [← ENNReal.toReal_sum (fun endpoint _ => law.apply_ne_top endpoint)]
        have total : ∑ endpoint, law endpoint = 1 := by
          simpa only [tsum_fintype] using law.tsum_coe
        rw [total, ENNReal.toReal_one]
      simp only [endpointMean, returned, ← Finset.sum_mul, weights, one_mul]

private theorem pairMean_const (constant : ℝ) :
    pairMean firstPhase suffixLaw (fun _ _ => constant) = constant := by
  simp only [pairMean, endpointMean_const, StrongProbability.verifierMean_const]

/-- Indicator pair means are probabilities. This supplies the bounded moment
needed when the final theorem averages over an arbitrary setup/context PMF. -/
theorem pairMean_range
    (value : Observation State Endpoint shape → Observation State Endpoint shape → ℝ)
    (range : ∀ left right, 0 ≤ value left right ∧ value left right ≤ 1) :
    0 ≤ pairMean firstPhase suffixLaw value ∧ pairMean firstPhase suffixLaw value ≤ 1 := by
  have lower := pairMean_mono firstPhase suffixLaw (fun _ _ => 0) value
    (fun left right _ _ => (range left right).1)
  have upper := pairMean_mono firstPhase suffixLaw value (fun _ _ => 1)
    (fun left right _ _ => (range left right).2)
  exact ⟨by simpa only [pairMean_const] using lower,
    by simpa only [pairMean_const] using upper⟩

end PairLaw

section WeakReturns

open PaperNonInteractive PaperStrongInterface
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw
open PiRLC.CoordinateOracle PiRLC.CoordinateCheckedCalls PiRLC.CoordinateTerminalLaw
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Primitives)

variable {Extension Commitment PublicInput Scalar KeyState : Type*}
  {shape : Shape} {columns blockCount width : Nat}
  (key : Key Extension Commitment PublicInput Scalar KeyState shape columns blockCount width)
  (running : Running Extension Commitment PublicInput shape)
  (fresh : Fresh Commitment PublicInput shape)
  (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
  (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
  [DecidableEq Scalar] [Fintype (Challenge key.piRlcAlgebra)]
  [Nonempty (Challenge key.piRlcAlgebra)] [Fintype (PaperLinearAlgebra.Assignment F columns)]
  (program : Primitives Scalar (PaperLinearAlgebra.Assignment F columns))
  (correct : PiRLC.PaperForkExtractionWork.Correct laws.ring laws.assignmentModule program)
  (ops : PiRLC.RelaxedBindingOps (PaperLinearAlgebra.Assignment F columns) Commitment Scalar)
  (compatible : PiRLC.PaperForkBinding.Compatible laws ops)

/-- The event records the actual endpoint lists, their disagreement, and the
collision consequence for the shared source commitment. It does not assert
that an unrelated collision exists somewhere in the setup. -/
def BindingEvent (left : Probe Extension shape)
    (leftEndpoint rightEndpoint : PaperWeakLaw.Endpoint (Fin key.arity.total)
      (Challenge key.piRlcAlgebra) (PaperLinearAlgebra.Assignment F columns)) : Prop :=
  ∃ leftValues rightValues,
    PaperWeakLaw.terminalValue program leftEndpoint = some leftValues ∧
    PaperWeakLaw.terminalValue program rightEndpoint = some rightValues ∧
    leftValues ≠ rightValues ∧
    ∃ coordinate, Nonempty (PiRLC.RelaxedBindingCollision key.piRlcSemantics key.params ops
      ((piRlcBatchForProbe key running fresh left).inputs coordinate).commitment)

include strongSet correct compatible in
/-- Different decoded witnesses from two positive weak endpoints imply the
binding event for these exact returned lists. Decode is a function, so equal
lists cannot create different witnesses. No injectivity assumption is needed. -/
theorem positive_decoded_disagreement_implies_binding
    (left right : Probe Extension shape)
    (leftOracle rightOracle : Oracle (Fin key.arity.total) (Challenge key.piRlcAlgebra)
      (PaperLinearAlgebra.Assignment F columns))
    (leftChecker rightChecker : Response (PaperLinearAlgebra.Assignment F columns) Scalar
      key.params key.arity → CheckResult)
    (leftCheckSpec : ∀ response, (leftChecker response).accepted = true ↔
      response.Success key.piRlcSemantics key.params key.piRlcAlgebra
        (piRlcBatchForProbe key running fresh left))
    (rightCheckSpec : ∀ response, (rightChecker response).accepted = true ↔
      response.Success key.piRlcSemantics key.params key.piRlcAlgebra
        (piRlcBatchForProbe key running fresh right))
    (leftEndpoint rightEndpoint : PaperWeakLaw.Endpoint (Fin key.arity.total)
      (Challenge key.piRlcAlgebra) (PaperLinearAlgebra.Assignment F columns))
    (leftPositive : 0 < (PaperWeakLaw.law leftOracle
      (oracleCheck key.piRlcAlgebra (fun response => (leftChecker response).accepted)) leftEndpoint).toReal)
    (rightPositive : 0 < (PaperWeakLaw.law rightOracle
      (oracleCheck key.piRlcAlgebra (fun response => (rightChecker response).accepted)) rightEndpoint).toReal)
    (leftWitness rightWitness : OutputWitness shape columns)
    (leftReturned : PaperWeakOutput.endpointWitness key program leftEndpoint = some leftWitness)
    (rightReturned : PaperWeakOutput.endpointWitness key program rightEndpoint = some rightWitness)
    (different : leftWitness ≠ rightWitness) :
    BindingEvent key running fresh program ops left leftEndpoint rightEndpoint := by
  cases leftEndpoint with
  | none =>
      simp [PaperWeakOutput.endpointWitness, PaperWeakLaw.terminalValue,
        PaperWeakLaw.terminalResult, PaperWeakOutput.decode] at leftReturned
  | some leftEndpoint =>
      rcases leftEndpoint with ⟨leftVector, leftInitial, leftOutputs⟩
      cases rightEndpoint with
      | none =>
          simp [PaperWeakOutput.endpointWitness, PaperWeakLaw.terminalValue,
            PaperWeakLaw.terminalResult, PaperWeakOutput.decode] at rightReturned
      | some rightEndpoint =>
          rcases rightEndpoint with ⟨rightVector, rightInitial, rightOutputs⟩
          cases leftResult : (PiRLC.CoordinateTerminalProgram.finish program
              leftVector leftInitial leftOutputs).value with
          | none =>
              simp [PaperWeakOutput.endpointWitness, PaperWeakLaw.terminalValue,
                PaperWeakLaw.terminalResult, leftResult, PaperWeakOutput.decode] at leftReturned
          | some leftValues =>
              cases rightResult : (PiRLC.CoordinateTerminalProgram.finish program
                  rightVector rightInitial rightOutputs).value with
              | none =>
                  simp [PaperWeakOutput.endpointWitness, PaperWeakLaw.terminalValue,
                    PaperWeakLaw.terminalResult, rightResult, PaperWeakOutput.decode] at rightReturned
              | some rightValues =>
                  have leftMass := (PaperWeakLaw.law_some_positive_iff leftOracle
                    (oracleCheck key.piRlcAlgebra (fun response => (leftChecker response).accepted))
                    leftVector leftInitial leftOutputs).mp leftPositive
                  have rightMass := (PaperWeakLaw.law_some_positive_iff rightOracle
                    (oracleCheck key.piRlcAlgebra (fun response => (rightChecker response).accepted))
                    rightVector rightInitial rightOutputs).mp rightPositive
                  have listsDifferent : leftValues ≠ rightValues := by
                    intro equal
                    have decoded := congrArg (fun values => (PaperWeakOutput.decode key (some values)).value) equal
                    have leftDecoded : (PaperWeakOutput.decode key (some leftValues)).value = some leftWitness := by
                      simpa only [PaperWeakOutput.endpointWitness, PaperWeakLaw.terminalValue,
                        PaperWeakLaw.terminalResult, leftResult] using leftReturned
                    have rightDecoded : (PaperWeakOutput.decode key (some rightValues)).value = some rightWitness := by
                      simpa only [PaperWeakOutput.endpointWitness, PaperWeakLaw.terminalValue,
                        PaperWeakLaw.terminalResult, rightResult] using rightReturned
                    exact different (Option.some.inj (leftDecoded.symm.trans (decoded.trans rightDecoded)))
                  obtain ⟨leftFork, _leftVector, _leftInitial, _leftOutputs, leftValuesEq, _leftValid⟩ :=
                    PiRLC.CoordinateExtraction.positive_return_implies_openings key.piRlcAlgebra
                      (piRlcBatchForProbe key running fresh left) laws strongSet leftOracle leftChecker
                      leftCheckSpec program correct leftVector leftInitial leftOutputs leftValues leftMass leftResult
                  obtain ⟨rightFork, _rightVector, _rightInitial, _rightOutputs, rightValuesEq, _rightValid⟩ :=
                    PiRLC.CoordinateExtraction.positive_return_implies_openings key.piRlcAlgebra
                      (piRlcBatchForProbe key running fresh right) laws strongSet rightOracle rightChecker
                      rightCheckSpec program correct rightVector rightInitial rightOutputs rightValues rightMass rightResult
                  rcases PiRLC.PaperForkBinding.two_forks_unique_or_collision laws ops compatible strongSet
                      (piRlcBatchForProbe key running fresh left) (piRlcBatchForProbe key running fresh right)
                      leftFork rightFork (piRlcBatchForProbe_same_phi key running fresh left right) with
                    same | collision
                  · exact False.elim (listsDifferent
                      (leftValuesEq.trans ((congrArg List.ofFn same).trans rightValuesEq.symm)))
                  · exact ⟨leftValues, rightValues, leftResult, rightResult, listsDifferent, collision⟩

end WeakReturns

section WeakPair

open PaperNonInteractive PaperStrongInterface
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw
open PiRLC.CoordinateOracle PiRLC.CoordinateCheckedCalls
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Primitives)

variable {Commitment PublicInput Scalar KeyState State : Type*}
  {shape : Shape} {columns blockCount width : Nat}
  (key : Key K Commitment PublicInput Scalar KeyState shape columns blockCount width)
  (running : Running K Commitment PublicInput shape)
  (fresh : Fresh Commitment PublicInput shape)
  [DecidableEq Scalar] [Fintype (Challenge key.piRlcAlgebra)]
  [Nonempty (Challenge key.piRlcAlgebra)] [Fintype (PaperLinearAlgebra.Assignment F columns)]

abbrev WeakEndpoint := PaperWeakLaw.Endpoint (Fin key.arity.total)
  (Challenge key.piRlcAlgebra) (PaperLinearAlgebra.Assignment F columns)

variable
  (oracle : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape → State →
    Oracle (Fin key.arity.total) (Challenge key.piRlcAlgebra) (PaperLinearAlgebra.Assignment F columns))
  (checker : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape → State →
    Response (PaperLinearAlgebra.Assignment F columns) Scalar key.params key.arity → CheckResult)
  (program : Primitives Scalar (PaperLinearAlgebra.Assignment F columns))

/-- The same charged weak oracle and public-output checker supply each
selected endpoint distribution. The family may depend on the captured state. -/
noncomputable def weakSuffixLaw (coins : PublicCoins K shape)
    (output : FullOutputCoordinates.FullOutput K shape) (state : State) : PMF (WeakEndpoint key) :=
  PaperWeakLaw.law (oracle coins output state)
    (oracleCheck key.piRlcAlgebra (fun response => (checker coins output state response).accepted))

/-- The coupled prover returns exactly the existing decoded terminal value. -/
def weakConsume (_coins : PublicCoins K shape)
    (_output : FullOutputCoordinates.FullOutput K shape) (_state : State)
    (endpoint : WeakEndpoint key) : Option (OutputWitness shape columns) :=
  PaperWeakOutput.endpointWitness key program endpoint

variable
  (ops : PiRLC.RelaxedBindingOps (PaperLinearAlgebra.Assignment F columns) Commitment Scalar)

/-- The event is attached to the actual two receipts and their sampled
endpoints. An aborted prefix cannot supply a collision event. -/
def observationBindingEvent : Observation State (WeakEndpoint key) shape →
    Observation State (WeakEndpoint key) shape → Prop
  | some (left, leftEndpoint), some (_, rightEndpoint) =>
      BindingEvent key running fresh program ops left.1 leftEndpoint rightEndpoint
  | _, _ => False

noncomputable def bindingProbability (firstPhase : InteractivePrefix.Prover State shape width) : ℝ :=
  pairMean firstPhase (weakSuffixLaw key oracle checker) fun left right =>
    if observationBindingEvent key running fresh program ops left right then 1 else 0

private theorem mapped_disagreement
    {OtherCommitment OtherPublicInput : Type*} {otherBlockCount : Nat}
    (maps : OpeningMaps OtherCommitment OtherPublicInput columns) (params : GlobalParams)
    (statement : Statement K OtherCommitment OtherPublicInput shape columns otherBlockCount ConcreteCarrier.baseOps)
    (leftProbe rightProbe : Probe K shape)
    (left right : Option (OutputWitness shape columns))
    (disagreement : StrongProbability.SuccessfulDisagreement (width := width) maps params statement
      (left.map fun witness => (leftProbe, witness))
      (right.map fun witness => (rightProbe, witness))) :
    ∃ leftWitness rightWitness, left = some leftWitness ∧ right = some rightWitness ∧
      leftWitness ≠ rightWitness := by
  cases left with
  | none =>
      rcases disagreement.1 with ⟨_, _, returned, _⟩
      cases returned
  | some leftWitness =>
      cases right with
      | none =>
          rcases disagreement.2.1 with ⟨_, _, returned, _⟩
          cases returned
      | some rightWitness =>
          rcases disagreement.2.2 with
            ⟨_, claimedLeft, _, claimedRight, leftEqual, rightEqual, different⟩
          simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at leftEqual rightEqual
          refine ⟨leftWitness, rightWitness, rfl, rfl, ?_⟩
          intro equal
          exact different (leftEqual.2.symm.trans (equal.trans rightEqual.2))

variable (laws : ExtractionAlgebra key.piRlcSemantics key.params key.piRlcAlgebra)
  (strongSet : StrongSetUnits laws.ring key.piRlcAlgebra.challengeValid)
  (correct : PiRLC.PaperForkExtractionWork.Correct laws.ring laws.assignmentModule program)
  (compatible : PiRLC.PaperForkBinding.Compatible laws ops)
  (checkSpec : ∀ (probe : Probe K shape) (state : State) response,
    (checker probe.coins probe.response.fullOutput state response).accepted = true ↔
      response.Success key.piRlcSemantics key.params key.piRlcAlgebra
        (piRlcBatchForProbe key running fresh probe))

include strongSet correct compatible checkSpec in
/-- Every positive sampled pair that exhibits successful decoded disagreement
has the tagged binding event for those same two terminal returns. -/
theorem successful_disagreement_implies_bindingEvent
    {OtherCommitment OtherPublicInput : Type*} {otherBlockCount : Nat}
    (maps : OpeningMaps OtherCommitment OtherPublicInput columns) (params : GlobalParams)
    (statement : Statement K OtherCommitment OtherPublicInput shape columns otherBlockCount ConcreteCarrier.baseOps)
    (left right : Observation State (WeakEndpoint key) shape)
    (leftSupported : Supported (weakSuffixLaw key oracle checker) left)
    (rightSupported : Supported (weakSuffixLaw key oracle checker) right)
    (disagreement : StrongProbability.SuccessfulDisagreement (width := width) maps params statement
      (outputOf (weakConsume key program) left) (outputOf (weakConsume key program) right)) :
    observationBindingEvent key running fresh program ops left right := by
  cases left with
  | none =>
      rcases disagreement.1 with ⟨_, _, returned, _⟩
      cases returned
  | some left =>
      rcases left with ⟨leftReceipt, leftEndpoint⟩
      cases right with
      | none =>
          rcases disagreement.2.1 with ⟨_, _, returned, _⟩
          cases returned
      | some right =>
          rcases right with ⟨rightReceipt, rightEndpoint⟩
          obtain ⟨leftWitness, rightWitness, leftReturned, rightReturned, different⟩ :=
            mapped_disagreement maps params statement leftReceipt.1 rightReceipt.1
              (PaperWeakOutput.endpointWitness key program leftEndpoint)
              (PaperWeakOutput.endpointWitness key program rightEndpoint) disagreement
          exact positive_decoded_disagreement_implies_binding key running fresh laws strongSet
            program correct ops compatible leftReceipt.1 rightReceipt.1
            (oracle leftReceipt.1.coins leftReceipt.1.response.fullOutput leftReceipt.2)
            (oracle rightReceipt.1.coins rightReceipt.1.response.fullOutput rightReceipt.2)
            (checker leftReceipt.1.coins leftReceipt.1.response.fullOutput leftReceipt.2)
            (checker rightReceipt.1.coins rightReceipt.1.response.fullOutput rightReceipt.2)
            (checkSpec leftReceipt.1 leftReceipt.2) (checkSpec rightReceipt.1 rightReceipt.2)
            leftEndpoint rightEndpoint leftSupported rightSupported leftWitness rightWitness
            leftReturned rightReturned different

include strongSet correct compatible checkSpec in
/-- The strong theorem's disagreement probability is bounded by a named
event of the same two independent sequential extractions. The numeric
cryptographic bound on this tagged event remains a separate premise. -/
theorem disagreementProbability_le_bindingProbability
    (firstPhase : InteractivePrefix.Prover State shape width)
    {OtherCommitment OtherPublicInput : Type*} {otherBlockCount : Nat}
    (maps : OpeningMaps OtherCommitment OtherPublicInput columns) (params : GlobalParams)
    (statement : Statement K OtherCommitment OtherPublicInput shape columns otherBlockCount ConcreteCarrier.baseOps) :
    StrongProbability.disagreementProbability
      (InteractiveDistribution.tapes firstPhase (none : WeakEndpoint key) (weakSuffixLaw key oracle checker))
      (InteractiveDistribution.coupled firstPhase (weakConsume key program)) maps params statement ≤
        bindingProbability key running fresh oracle checker program ops firstPhase := by
  rw [disagreementProbability_eq_pairMean]
  unfold bindingProbability
  apply pairMean_mono firstPhase (weakSuffixLaw key oracle checker)
  intro left right leftSupported rightSupported
  by_cases disagreement : StrongProbability.SuccessfulDisagreement (width := width) maps params statement
      (outputOf (weakConsume key program) left) (outputOf (weakConsume key program) right)
  · have collision := successful_disagreement_implies_bindingEvent key running fresh oracle checker
      program ops laws strongSet correct compatible checkSpec maps params statement left right
      leftSupported rightSupported disagreement
    simp only [if_pos disagreement, if_pos collision, le_refl]
  · simp only [if_neg disagreement]
    split_ifs <;> norm_num

/-- The binding-event mean has finite mass for outer context averaging. -/
theorem bindingProbability_range (firstPhase : InteractivePrefix.Prover State shape width) :
    0 ≤ bindingProbability key running fresh oracle checker program ops firstPhase ∧
      bindingProbability key running fresh oracle checker program ops firstPhase ≤ 1 := by
  apply pairMean_range firstPhase (weakSuffixLaw key oracle checker)
  intro left right
  split_ifs <;> norm_num

end WeakPair

end NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionAgreement
