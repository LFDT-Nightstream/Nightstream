import NightstreamFPrime.Lifecycle.Nifs.BindingBridge
import NightstreamFPrime.Spec.Folding.Nifs.BindingOutput
import NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionAgreement
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CostedWitnessProjection

/-!
The selected binding reduction emits integer vectors from actual weak
endpoints. Arithmetic uses the existing charged primitives. Coordinate reads
use the existing charged witness accessor; arbitrary functions have no assumed
unit access cost. Centered lifting counts fixed Goldilocks word operations.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.BindingReduction

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra
open _root_.NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtractionWork
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw
open PiRLC.CoordinateOracle PiRLC.CoordinateCheckedCalls
open StrongReduction ConcreteCarrier

/-- Projection, comparison, integer conversion and result return cost four
word operations. The negative branch also subtracts the fixed modulus. -/
def centered (word : F) : Result Int :=
  if word.val ≤ goldilocksModulus / 2 then
    ⟨(word.val : Int), 4⟩
  else ⟨(word.val : Int) - goldilocksModulus, 5⟩

theorem centered_value (word : F) :
    (centered word).value = ZMod.valMinAbs (n := goldilocksModulus) word := by
  rw [ZMod.valMinAbs_def_pos]
  change (centered word).value =
    if word.val ≤ goldilocksModulus / 2 then (word.val : Int) else (word.val : Int) - goldilocksModulus
  unfold centered
  split_ifs <;> rfl

theorem centered_work_le (word : F) : (centered word).work ≤ 5 := by
  unfold centered
  split <;> dsimp only <;> omega

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

abbrev Endpoint := PaperWeakLaw.Endpoint (Fin PaperProfile.arity.total)
  {scalar : RingF // Challenge.challengeValid scalar}
  (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- Materialize every integer coordinate of the candidate. The temporary
witness supplies only data to the already specified accessor. -/
def integerVector
    (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (assignment : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Result (List Int) :=
  collect fun column =>
    let word := access ⟨fun _ => assignment⟩ ⟨0, by decide⟩ column
    let lifted := centered word.value
    ⟨lifted.value, word.work + lifted.work⟩

theorem integerVector_value
    (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (correct : CostedWitnessProjection.Correct access)
    (assignment : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (integerVector access assignment).value =
      List.ofFn (fun column => ZMod.valMinAbs (n := goldilocksModulus) (assignment column)) := by
  rw [integerVector, collect_values]
  congr 1
  funext column
  simp only [centered_value]
  exact congrArg (ZMod.valMinAbs (n := goldilocksModulus)) (correct _ _ column)

theorem integerVector_work_le
    (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (accessBound : Nat) (bounded : CostedWitnessProjection.Bounded access accessBound)
    (assignment : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (integerVector access assignment).work ≤
      (FullShape logicalWidth publicFits).carrierWidth * (accessBound + 6) + 1 := by
  apply collect_work_le (accessBound + 5)
  intro column
  have read := bounded ⟨fun _ => assignment⟩ ⟨0, by decide⟩ column
  have lifted := centered_work_le (access ⟨fun _ => assignment⟩ ⟨0, by decide⟩ column).value
  change (access ⟨fun _ => assignment⟩ ⟨0, by decide⟩ column).work +
    (centered (access ⟨fun _ => assignment⟩ ⟨0, by decide⟩ column).value).work ≤ accessBound + 5
  omega

/-- Emit the raw integer vector for the supplied trial coordinate. Selecting
the coordinate uniformly belongs to the reduction experiment, not the NIFS
transcript. Missing endpoints return no vector. -/
def run
    (program : Primitives RingF
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
    (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (left right : Endpoint (logicalWidth := logicalWidth) (publicFits := publicFits))
    (coordinate : Fin PaperProfile.arity.total) : Result (Option (List Int)) :=
  let candidate := BindingOutput.candidate program left right coordinate
  match candidate.value with
  | none => ⟨none, candidate.work + 1⟩
  | some assignment =>
      let output := integerVector access assignment
      ⟨some output.value, candidate.work + output.work + 1⟩

theorem run_work_le
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Primitives RingF
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
    (bounds : PrimitiveBounds)
    (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds)
    (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (accessBound : Nat) (accessBounded : CostedWitnessProjection.Bounded access accessBound)
    (left right : Endpoint (logicalWidth := logicalWidth) (publicFits := publicFits))
    (coordinate : Fin PaperProfile.arity.total) :
    (run program access left right coordinate).work ≤ BindingOutput.crossWork bounds +
      (FullShape logicalWidth publicFits).carrierWidth * (accessBound + 6) + 8 := by
  have candidate := BindingOutput.candidate_work_le
    (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds bounded left right coordinate
  dsimp only [run]
  split
  · change (BindingOutput.candidate program left right coordinate).work + 1 ≤ _
    omega
  · rename_i assignment _
    have output := integerVector_work_le access accessBound accessBounded assignment
    change (BindingOutput.candidate program left right coordinate).work +
      (integerVector access assignment).work + 1 ≤ _
    omega

/-- Retain the two actual charged source-call results and emit one trial
vector. Both call clocks stay attached to their returned observations. The driver
charges the uniform-coordinate request in the existing interactive coin model. -/
def runPair {State : Type*}
    (program : Primitives RingF
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
    (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (left right : Result (PaperCompositionAgreement.Observation State
      (Endpoint (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape))
    (coordinate : Fin PaperProfile.arity.total) : Result (Option (List Int)) :=
  match left.value, right.value with
  | some (_, leftEndpoint), some (_, rightEndpoint) =>
      let output := run program access leftEndpoint rightEndpoint coordinate
      ⟨output.value, left.work + right.work + output.work + 4⟩
  | _, _ => ⟨none, left.work + right.work + 4⟩

/-- One uniform-coordinate request, two observation checks and the final
return add four steps to the proved integer-output work. Both call costs remain. -/
theorem runPair_work_le {State : Type*}
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Primitives RingF
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
    (bounds : PrimitiveBounds)
    (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds)
    (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (accessBound : Nat) (accessBounded : CostedWitnessProjection.Bounded access accessBound)
    (left right : Result (PaperCompositionAgreement.Observation State
      (Endpoint (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape))
    (coordinate : Fin PaperProfile.arity.total) :
    (runPair program access left right coordinate).work ≤ left.work + right.work +
      BindingOutput.crossWork bounds +
      (FullShape logicalWidth publicFits).carrierWidth * (accessBound + 6) + 12 := by
  dsimp only [runPair]
  split
  · rename_i _ leftEndpoint _ rightEndpoint _ _
    have output := run_work_le ajtai program bounds bounded access accessBound accessBounded
      leftEndpoint rightEndpoint coordinate
    change left.work + right.work + (run program access leftEndpoint rightEndpoint coordinate).work + 4 ≤ _
    omega
  · change left.work + right.work + 4 ≤ _
    omega

/-- The postprocessing clock depends only on the returned observations.
The actual left and right call clocks add without changing that work. -/
theorem runPair_work_eq {State : Type*}
    (program : Primitives RingF
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
    (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (left right : Result (PaperCompositionAgreement.Observation State
      (Endpoint (logicalWidth := logicalWidth) (publicFits := publicFits)) productionShape))
    (coordinate : Fin PaperProfile.arity.total) :
    (runPair program access left right coordinate).work = left.work + right.work +
      (runPair program access ⟨left.value, 0⟩ ⟨right.value, 0⟩ coordinate).work := by
  rcases left with ⟨left, leftWork⟩
  rcases right with ⟨right, rightWork⟩
  cases left <;> cases right <;> simp [runPair, Nat.add_assoc]

/-- Success means that the emitted integer list itself is a nonzero kernel
vector of the same Ajtai key, with the selected strict `8TB` bound. -/
def Succeeds
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (output : Option (List Int)) : Prop :=
  ∃ witness : Binding.ShortKernelVector ajtai productionGlobalParams.msisNormBound,
    output = some (List.ofFn witness.vector)

variable
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (correct : Correct (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (PaperExtractionAlgebra.extractionAlgebra ajtai).assignmentModule program)
  (strongSet : StrongSetUnits (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (ProductionKey.key relation ajtai).piRlcAlgebra.challengeValid)
  (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
  (accessCorrect : CostedWitnessProjection.Correct access)
  [DecidableEq RingF]
  [Fintype (PiRLC.CoordinateForkLaw.Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (PiRLC.CoordinateForkLaw.Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]

include correct strongSet accessCorrect in
/-- On the actual supported endpoint pair, the binding event yields a trial
coordinate where the executable reduction emits a short-kernel witness.
The existing existential collision is not used to choose the output data. -/
theorem bindingEvent_implies_success
    (left right : Probe K productionShape)
    (leftOracle rightOracle : Oracle (Fin PaperProfile.arity.total)
      (PiRLC.CoordinateForkLaw.Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
    (leftChecker rightChecker : Response
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
      RingF productionGlobalParams PaperProfile.arity → CheckResult)
    (leftCheckSpec : ∀ response, (leftChecker response).accepted = true ↔
      response.Success (ProductionKey.key relation ajtai).piRlcSemantics productionGlobalParams
        (ProductionKey.key relation ajtai).piRlcAlgebra
        (PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai) running fresh left))
    (rightCheckSpec : ∀ response, (rightChecker response).accepted = true ↔
      response.Success (ProductionKey.key relation ajtai).piRlcSemantics productionGlobalParams
        (ProductionKey.key relation ajtai).piRlcAlgebra
        (PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai) running fresh right))
    (leftEndpoint rightEndpoint : Endpoint (logicalWidth := logicalWidth) (publicFits := publicFits))
    (leftPositive : 0 < (PaperWeakLaw.law leftOracle
      (oracleCheck (ProductionKey.key relation ajtai).piRlcAlgebra
        (fun response => (leftChecker response).accepted)) leftEndpoint).toReal)
    (rightPositive : 0 < (PaperWeakLaw.law rightOracle
      (oracleCheck (ProductionKey.key relation ajtai).piRlcAlgebra
        (fun response => (rightChecker response).accepted)) rightEndpoint).toReal)
    (event : PaperCompositionAgreement.BindingEvent (ProductionKey.key relation ajtai)
      running fresh program (Binding.relaxedOps (shape := FullShape logicalWidth publicFits))
      left leftEndpoint rightEndpoint) :
    ∃ coordinate, Succeeds ajtai (run program access leftEndpoint rightEndpoint coordinate).value := by
  rcases event with ⟨leftValues, rightValues, leftReturned, rightReturned, differentValues, _collision⟩
  cases leftEndpoint with
  | none => simp [PaperWeakLaw.terminalValue, PaperWeakLaw.terminalResult] at leftReturned
  | some leftEndpoint =>
    rcases leftEndpoint with ⟨leftVector, leftInitial, leftOutputs⟩
    cases rightEndpoint with
    | none => simp [PaperWeakLaw.terminalValue, PaperWeakLaw.terminalResult] at rightReturned
    | some rightEndpoint =>
      rcases rightEndpoint with ⟨rightVector, rightInitial, rightOutputs⟩
      let key := ProductionKey.key relation ajtai
      let laws := PaperExtractionAlgebra.extractionAlgebra ajtai
      let leftBatch := PaperStrongInterface.piRlcBatchForProbe key running fresh left
      let rightBatch := PaperStrongInterface.piRlcBatchForProbe key running fresh right
      have leftMass := (PaperWeakLaw.law_some_positive_iff leftOracle
        (oracleCheck key.piRlcAlgebra (fun response => (leftChecker response).accepted))
        leftVector leftInitial leftOutputs).mp leftPositive
      have rightMass := (PaperWeakLaw.law_some_positive_iff rightOracle
        (oracleCheck key.piRlcAlgebra (fun response => (rightChecker response).accepted))
        rightVector rightInitial rightOutputs).mp rightPositive
      obtain ⟨leftFork, leftVectorEq, leftInitialEq, leftResponses, leftValuesEq, _leftValid⟩ :=
        PiRLC.CoordinateExtraction.positive_return_implies_openings key.piRlcAlgebra
          leftBatch laws strongSet leftOracle leftChecker leftCheckSpec program correct
          leftVector leftInitial leftOutputs leftValues leftMass leftReturned
      obtain ⟨rightFork, rightVectorEq, rightInitialEq, rightResponses, rightValuesEq, _rightValid⟩ :=
        PiRLC.CoordinateExtraction.positive_return_implies_openings key.piRlcAlgebra
          rightBatch laws strongSet rightOracle rightChecker rightCheckSpec program correct
          rightVector rightInitial rightOutputs rightValues rightMass rightReturned
      have different : extractedAssignment laws strongSet leftFork ≠
          extractedAssignment laws strongSet rightFork := by
        intro same
        exact differentValues (leftValuesEq.trans ((congrArg List.ofFn same).trans rightValuesEq.symm))
      have differentAt : ∃ coordinate, extractedAssignment laws strongSet leftFork coordinate ≠
          extractedAssignment laws strongSet rightFork coordinate :=
        Classical.byContradiction fun noDifference => different (funext fun coordinate =>
          Classical.byContradiction fun atCoordinate => noDifference ⟨coordinate, atCoordinate⟩)
      obtain ⟨coordinate, differentAt⟩ := differentAt
      let collision := PiRLC.PaperForkBinding.collisionAt laws
        (Binding.relaxedOps (shape := FullShape logicalWidth publicFits))
        (BindingBridge.compatible relation ajtai) strongSet leftBatch rightBatch leftFork rightFork
        (PaperStrongInterface.piRlcBatchForProbe_same_phi key running fresh left right)
        coordinate differentAt
      let witness := Binding.relaxedBindingCollision_to_shortKernel ajtai
        (leftBatch.inputs coordinate).commitment {
          delta₁ := collision.delta₁
          delta₂ := collision.delta₂
          opening₁ := collision.opening₁
          opening₂ := collision.opening₂
          delta₁Valid := collision.delta₁Valid
          delta₂Valid := collision.delta₂Valid
          firstEquation := collision.firstEquation
          secondEquation := collision.secondEquation
          firstNorm := collision.firstNorm
          secondNorm := collision.secondNorm
          crossDifferent := collision.crossDifferent }
      have leftBaseAt : leftFork.base.challenges coordinate = (leftVector coordinate).val :=
        congrFun leftVectorEq coordinate
      have rightBaseAt : rightFork.base.challenges coordinate = (rightVector coordinate).val :=
        congrFun rightVectorEq coordinate
      have leftForkAt : (leftFork.forks coordinate).challenges coordinate =
          (leftOutputs coordinate).1.val := by
        rw [(leftResponses coordinate).1, Function.update_self]
      have rightForkAt : (rightFork.forks coordinate).challenges coordinate =
          (rightOutputs coordinate).1.val := by
        rw [(rightResponses coordinate).1, Function.update_self]
      have candidateEq : (BindingOutput.candidate program
          (some (leftVector, leftInitial, leftOutputs))
          (some (rightVector, rightInitial, rightOutputs)) coordinate).value =
          some (Binding.difference
            (CarrierAction.act collision.delta₁ collision.opening₂)
            (CarrierAction.act collision.delta₂ collision.opening₁)) := by
        rw [← leftInitialEq, ← rightInitialEq]
        simp only [BindingOutput.candidate]
        rw [← (leftResponses coordinate).2, ← (rightResponses coordinate).2]
        change some (BindingOutput.crossDifference program _ _ _ _ _ _ _ _).value = _
        rw [BindingOutput.crossDifference_value laws.ring laws.assignmentModule program correct,
          ← leftBaseAt, ← rightBaseAt, ← leftForkAt, ← rightForkAt]
        rw [BindingBridge.subtraction_eq ajtai]
        rfl
      refine ⟨coordinate, witness, ?_⟩
      dsimp only [run]
      rw [candidateEq]
      change some (integerVector access _).value = _
      rw [integerVector_value access accessCorrect]
      rfl

end NightstreamFPrime.Lifecycle.Nifs.BindingReduction
