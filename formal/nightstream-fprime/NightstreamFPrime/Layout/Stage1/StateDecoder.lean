import Mathlib.Data.List.OfFn
import NightstreamFPrime.Layout.Stage1.RunningTransitionData
import NightstreamFPrime.Layout.Stage1.StateEncoding

/-!
Owns structural decoding of the fixed Stage 1 state-hash word array.

The decoder is a value view only. It does not select a package, application,
verification key, or transcript. Its right-inverse theorems require the exact
fixed-word conditions enforced by the PiCCS state-binding rows.
-/

namespace NightstreamFPrime.Layout.Stage1.StateDecoder

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- A bounded logical slice of an unbounded state-word view. -/
def slice (state : Nat → F) (start count : Nat) : List F :=
  List.ofFn fun index : Fin count => state (start + index.val)

@[simp] theorem slice_length (state : Nat → F) (start count : Nat) :
    (slice state start count).length = count := by
  simp [slice]

theorem slice_congr (left right : Nat → F) (start count : Nat)
    (equal : ∀ index : Fin count,
      left (start + index.val) = right (start + index.val)) :
    slice left start count = slice right start count := by
  unfold slice
  apply congrArg List.ofFn
  funext index
  exact equal index

/-- Adjacent state slices concatenate without inspecting their contents. -/
theorem slice_add (state : Nat → F) (start left right : Nat) :
    slice state start (left + right) =
      slice state start left ++ slice state (start + left) right := by
  unfold slice
  rw [← List.ofFn_fin_append]
  congr 1
  funext index
  refine Fin.addCases ?_ ?_ index
  · intro leftIndex
    simp [Fin.append]
  · intro rightIndex
    simp [Fin.append, Nat.add_assoc]

theorem slice_getD (state : Nat → F) (start count index : Nat)
    (bound : index < count) :
    (slice state start count).getD index 0 = state (start + index) := by
  unfold slice
  exact Lifecycle.PriorStateHash.ofFn_getD
    (fun position : Fin count => state (start + position.val))
    ⟨index, bound⟩ 0

/-- A repeated fixed-width interval is the ordered concatenation of its
subintervals. -/
theorem slice_mul (state : Nat → F) (start count width : Nat) :
    slice state start (count * width) =
      (List.finRange count).flatMap fun index =>
        slice state (start + index.val * width) width := by
  induction count generalizing start with
  | zero => simp [slice]
  | succ count inductionHypothesis =>
      rw [Nat.succ_mul, Nat.add_comm (count * width) width]
      rw [slice_add, inductionHypothesis]
      simp only [List.finRange_succ, List.flatMap_cons, Fin.val_zero,
        Nat.zero_mul, Nat.add_zero]
      rw [List.flatMap_map]
      apply congrArg (slice state start width ++ ·)
      apply List.flatMap_congr
      intro index _member
      apply congrArg (fun offset => slice state offset width)
      change start + width + index.val * width =
        start + (index.val + 1) * width
      rw [Nat.add_mul]
      simp only [Nat.one_mul]
      omega

def pair (state : Nat → F) (start : Nat) : K :=
  ⟨state start, state (start + 1)⟩

theorem serializeK_pair (state : Nat → F) (start : Nat) :
    serializeK (pair state start) = slice state start 2 := by
  apply List.ext_get
  · simp [slice, serializeK]
  · intro index leftBound rightBound
    have indexBound : index < 2 := by
      simpa [serializeK] using leftBound
    interval_cases index <;> simp [serializeK, pair, slice]

def ringValue (state : Nat → F) (start : Nat) : RingF :=
  fun coefficient => state (start + coefficient.val)

theorem serializeRingF_ringValue (state : Nat → F) (start : Nat) :
    serializeRingF (ringValue state start) = slice state start ringDegree := by
  unfold serializeRingF ringValue slice
  rw [List.ofFn_eq_map]

def commitment (state : Nat → F) (start : Nat) :
    PaperAlgebra.Commitment :=
  fun row coefficient =>
    state (start + row.val * ringDegree + coefficient.val)

theorem serializeCommitment_commitment (state : Nat → F) (start : Nat) :
    serializeCommitment (commitment state start) =
      slice state start (productionProfile.commitmentWidth * ringDegree) := by
  unfold serializeCommitment
  calc
    (List.finRange productionProfile.commitmentWidth).flatMap
          (fun row => serializeRingF (commitment state start row)) =
        (List.finRange productionProfile.commitmentWidth).flatMap
          (fun row => slice state (start + row.val * ringDegree) ringDegree) := by
      apply List.flatMap_congr
      intro row _member
      change serializeRingF
          (ringValue state (start + row.val * ringDegree)) = _
      exact serializeRingF_ringValue state _
    _ = slice state start
          (productionProfile.commitmentWidth * ringDegree) :=
      (slice_mul state start productionProfile.commitmentWidth ringDegree).symm

def publicInput
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) (start : Nat) :
    PaperAlgebra.PublicInput
      (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  fun column => state (start + column.val)

theorem serializePublicInput_publicInput
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) (start : Nat) :
    serializePublicInput (publicFits := publicFits)
        (publicInput logicalWidth publicFits state start) =
      slice state start (FullShape logicalWidth publicFits).publicWidth := by
  unfold serializePublicInput publicInput slice
  rw [List.ofFn_eq_map]

theorem serializePairs (state : Nat → F) (start count : Nat) :
    (List.finRange count).flatMap (fun index =>
        serializeK (pair state (start + index.val * 2))) =
      slice state start (count * 2) := by
  calc
    _ = (List.finRange count).flatMap (fun index =>
          slice state (start + index.val * 2) 2) := by
      apply List.flatMap_congr
      intro index _member
      exact serializeK_pair state _
    _ = slice state start (count * 2) :=
      (slice_mul state start count 2).symm

theorem serializePairRows (state : Nat → F)
    (start rowCount columnCount : Nat) :
    (List.finRange rowCount).flatMap (fun row =>
      (List.finRange columnCount).flatMap (fun column =>
        serializeK (pair state
          (start + row.val * (columnCount * 2) + column.val * 2)))) =
      slice state start (rowCount * (columnCount * 2)) := by
  calc
    _ = (List.finRange rowCount).flatMap (fun row =>
          slice state (start + row.val * (columnCount * 2))
            (columnCount * 2)) := by
      apply List.flatMap_congr
      intro row _member
      rw [← serializePairs state
        (start + row.val * (columnCount * 2)) columnCount]
    _ = slice state start (rowCount * (columnCount * 2)) :=
      (slice_mul state start rowCount (columnCount * 2)).symm

def point (state : Nat → F) (start : Nat) :
    CubePoint K cubeVariables where
  coordinates := List.ofFn fun coordinate : Fin cubeVariables =>
    pair state (start + coordinate.val * 2)
  dimension := by simp

theorem serializePoint_point (state : Nat → F) (start : Nat) :
    serializePoint (point state start) =
      slice state start (cubeVariables * 2) := by
  unfold serializePoint
  change (List.ofFn fun coordinate : Fin cubeVariables =>
    pair state (start + coordinate.val * 2)).flatMap serializeK = _
  rw [List.ofFn_eq_map]
  exact serializePairs state start cubeVariables

def evaluations (state : Nat → F) (start : Nat) :
    StrongReduction.EvaluationFamily K productionShape where
  pad := fun coefficient => pair state (start + coefficient.val * 2)
  matrix := fun matrix coefficient => pair state
    (start + productionShape.coefficientCount * 2 +
      matrix.val * (productionShape.coefficientCount * 2) +
      coefficient.val * 2)

theorem serializeEvaluations_evaluations (state : Nat → F) (start : Nat) :
    serializeEvaluations (evaluations state start) =
      slice state start
        ((productionShape.matrixCount + 1) *
          productionShape.coefficientCount * 2) := by
  unfold serializeEvaluations evaluations
  rw [serializePairs state start productionShape.coefficientCount]
  rw [serializePairRows state
    (start + productionShape.coefficientCount * 2)
    productionShape.matrixCount productionShape.coefficientCount]
  rw [← slice_add]
  apply congrArg (slice state start)
  norm_num [productionShape, productionProfile,
    Phi81MatrixSource.phi81Shape, ringDegree]

/-- View one state word array as the pilot's prior-preimage input. Other
pilot fields are irrelevant to running-instance decoding. -/
def externalValues (state : Nat → F) : PilotProduction.ExternalValues where
  priorPreimage := fun index => state index.val
  priorPublicInput := fun _ => 0
  outputPreimage := fun _ => 0
  outputDigest := fun _ => 0

def running
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  PiCCSInputs.decodedRunning logicalWidth publicFits (externalValues state)

def directRunning
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) where
  point := point state PiCCSInputs.runningPointStart
  commitments := fun source => commitment state
    (PiCCSInputs.runningCommitmentStart source.val)
  publicInputs := fun source => publicInput logicalWidth publicFits state
    (PiCCSInputs.runningPublicStart source.val)
  evaluations := fun source => evaluations state
    (PiCCSInputs.runningEvaluationStart source.val)

theorem directRunning_eq_running
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) :
    directRunning logicalWidth publicFits state =
      running logicalWidth publicFits state := by
  rfl

theorem evalRunning_eq_running
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Circuit.Env) :
    StatementAbsorption.evalRunning
        (PiCCSInputs.runningExpr logicalWidth publicFits) env =
      running logicalWidth publicFits
        (fun word => env (PilotProduction.priorPreimageStart + word)) := by
  unfold running PiCCSInputs.decodedRunning
    StatementAbsorption.evalRunning StatementAbsorption.evalPoint
    StatementAbsorption.evalEvaluation
  congr 1
  · funext source row coefficient
    simp [PiCCSInputs.runningExpr, PiCCSInputs.runningCommitment,
      PiCCSInputs.runningCommitmentIndex, externalValues,
      PilotProduction.priorPreimageStart]
  · funext source column
    simp [PiCCSInputs.runningExpr, PiCCSInputs.runningPublicInput,
      PiCCSInputs.runningPublicInputIndex, externalValues,
      PilotProduction.priorPreimageStart]
  · funext source
    congr 1
    · funext coefficient
      simp [PiCCSInputs.runningExpr, PiCCSInputs.runningEval_K,
        PiCCSInputs.runningEval_KIndex, PiCCSInputs.pairAt,
        Circuit.Quadratic.KExpr.eval, externalValues,
        PilotProduction.priorPreimageStart]
    · funext matrix coefficient
      simp [PiCCSInputs.runningExpr, PiCCSInputs.runningEval_A,
        PiCCSInputs.runningEval_AIndex, PiCCSInputs.pairAt,
        Circuit.Quadratic.KExpr.eval, externalValues,
        PilotProduction.priorPreimageStart]

private theorem cubePoint_ext
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
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

private theorem running_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : Running
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (pointEqual : left.point = right.point)
    (commitmentsEqual : left.commitments = right.commitments)
    (publicInputsEqual : left.publicInputs = right.publicInputs)
    (evaluationsEqual : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp only [
    NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Running.mk.injEq]
  exact ⟨pointEqual, commitmentsEqual, publicInputsEqual, evaluationsEqual⟩

theorem evalOutputRunning_eq_running
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Circuit.Env) :
    StatementAbsorption.evalRunning
        (RunningTransitionInputs.outputRunningExpr logicalWidth publicFits) env =
      running logicalWidth publicFits
        (fun word => env (PilotProduction.outputPreimageStart + word)) := by
  rw [← directRunning_eq_running logicalWidth publicFits
    (fun word => env (PilotProduction.outputPreimageStart + word))]
  apply running_ext
  · apply cubePoint_ext
    change List.ofFn (fun coordinate =>
        (RunningTransitionInputs.outputPoint coordinate).eval env) =
      List.ofFn (fun coordinate => pair
        (fun word => env (PilotProduction.outputPreimageStart + word))
        (PiCCSInputs.runningPointStart + coordinate.val * 2))
    apply congrArg List.ofFn
    funext coordinate
    simp [RunningTransitionInputs.outputPoint,
      RunningTransitionInputs.outputPairAt, pair,
      Circuit.Quadratic.KExpr.eval, RunningTransitionInputs.outputBase,
      Nat.add_assoc]
  · funext source row coefficient
    simp [StatementAbsorption.evalRunning,
      RunningTransitionInputs.outputRunningExpr,
      RunningTransitionInputs.outputCommitment, directRunning, commitment,
      RunningTransitionInputs.outputBase]
    apply congrArg env
    omega
  · funext source column
    simp [StatementAbsorption.evalRunning,
      RunningTransitionInputs.outputRunningExpr,
      RunningTransitionInputs.outputPublicInput, directRunning, publicInput,
      RunningTransitionInputs.outputBase]
    apply congrArg env
    omega
  · funext source
    apply evaluationFamily_ext
    · funext coefficient
      simp [StatementAbsorption.evalRunning,
        StatementAbsorption.evalEvaluation,
        RunningTransitionInputs.outputRunningExpr,
        RunningTransitionInputs.outputEval_K,
        RunningTransitionInputs.outputPairAt, directRunning, evaluations, pair,
        Circuit.Quadratic.KExpr.eval, RunningTransitionInputs.outputBase,
        Nat.add_assoc]
    · funext matrix coefficient
      simp [StatementAbsorption.evalRunning,
        StatementAbsorption.evalEvaluation,
        RunningTransitionInputs.outputRunningExpr,
        RunningTransitionInputs.outputEval_A,
        RunningTransitionInputs.outputPairAt, directRunning, evaluations, pair,
        Circuit.Quadratic.KExpr.eval, RunningTransitionInputs.outputBase,
        productionShape, Phi81MatrixSource.phi81Shape, ringDegree,
        Nat.add_assoc]

/-- The value-level form of the fixed-word rows. -/
def Canonical (state : Nat → F) : Prop :=
  ∀ word ∈ StateBinding.fixedWords, state word.index = word.value

theorem block_eq_slice (state : Nat → F) (start count : Nat)
    (payload : List F) (payloadEq : payload = slice state (start + 1) count)
    (header : state start = natWord count) :
    block payload = slice state start (count + 1) := by
  rw [payloadEq]
  unfold block
  rw [show count + 1 = 1 + count by omega, slice_add]
  simp [slice, header]

theorem canonical_pointHeader {state : Nat → F}
    (canonical : Canonical state) :
    state PiCCSInputs.priorRunningStart = natWord (cubeVariables * 2) := by
  apply canonical ⟨PiCCSInputs.priorRunningStart, natWord (cubeVariables * 2)⟩
  simp only [StateBinding.fixedWords, List.mem_append]
  apply Or.inl
  apply Or.inl
  apply Or.inr
  simp [PiCCSInputs.priorRunningStart, natWord]

theorem canonical_commitmentHeader {state : Nat → F}
    (canonical : Canonical state)
    (source : Fin productionShape.runningCount) :
    state (StateBinding.runningGroupStart source.val) = natWord 1188 := by
  apply canonical ⟨StateBinding.runningGroupStart source.val, natWord 1188⟩
  simp only [StateBinding.fixedWords, List.mem_append]
  apply Or.inl
  apply Or.inr
  rw [StateBinding.runningPrefixWords, List.mem_flatMap]
  refine ⟨source, by simp, ?_⟩
  simp [natWord]

theorem canonical_publicHeader {state : Nat → F}
    (canonical : Canonical state)
    (source : Fin productionShape.runningCount) :
    state (StateBinding.runningGroupStart source.val + 1189) = natWord 270 := by
  apply canonical
    ⟨StateBinding.runningGroupStart source.val + 1189, natWord 270⟩
  simp only [StateBinding.fixedWords, List.mem_append]
  apply Or.inl
  apply Or.inr
  rw [StateBinding.runningPrefixWords, List.mem_flatMap]
  refine ⟨source, by simp, ?_⟩
  simp [natWord]

theorem canonical_evaluationHeader {state : Nat → F}
    (canonical : Canonical state)
    (source : Fin productionShape.runningCount) :
    state (StateBinding.runningGroupStart source.val + 1460) = natWord 1620 := by
  apply canonical
    ⟨StateBinding.runningGroupStart source.val + 1460, natWord 1620⟩
  simp only [StateBinding.fixedWords, List.mem_append]
  apply Or.inl
  apply Or.inr
  rw [StateBinding.runningPrefixWords, List.mem_flatMap]
  refine ⟨source, by simp, ?_⟩
  simp [natWord]

theorem serializeRunning_commitmentPayload
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) (source : Fin productionShape.runningCount) :
    serializeCommitment
        ((directRunning logicalWidth publicFits state).commitments source) =
      slice state (StateBinding.runningGroupStart source.val + 1) 1188 := by
  change serializeCommitment
      (commitment state (PiCCSInputs.runningCommitmentStart source.val)) = _
  rw [serializeCommitment_commitment]
  apply congrArg₂ (slice state)
  · simp [PiCCSInputs.runningCommitmentStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      StateBinding.runningGroupStart,
      cubeVariables]
  · norm_num [productionProfile, ringDegree]

theorem serializeRunning_publicPayload
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) (source : Fin productionShape.runningCount) :
    serializePublicInput (publicFits := publicFits)
        ((directRunning logicalWidth publicFits state).publicInputs source) =
      slice state (StateBinding.runningGroupStart source.val + 1190) 270 := by
  change serializePublicInput (publicFits := publicFits)
      (publicInput logicalWidth publicFits state
        (PiCCSInputs.runningPublicStart source.val)) = _
  rw [serializePublicInput_publicInput]
  apply congrArg₂ (slice state)
  · simp [PiCCSInputs.runningPublicStart, PiCCSInputs.runningGroupStart,
      PiCCSInputs.runningGroupsStart, PiCCSInputs.priorRunningStart,
      PiCCSInputs.runningGroupWords,
      StateBinding.runningGroupStart, cubeVariables]
  · norm_num [FullShape, fullShape, Phi81Relation.Shape.publicWidth,
      publicRingColumns, ringDegree]

theorem serializeRunning_evaluationPayload
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) (source : Fin productionShape.runningCount) :
    serializeEvaluations
        ((directRunning logicalWidth publicFits state).evaluations source) =
      slice state (StateBinding.runningGroupStart source.val + 1461) 1620 := by
  change serializeEvaluations
      (evaluations state (PiCCSInputs.runningEvaluationStart source.val)) = _
  rw [serializeEvaluations_evaluations]
  apply congrArg₂ (slice state)
  · simp [PiCCSInputs.runningEvaluationStart,
      PiCCSInputs.runningGroupStart, PiCCSInputs.runningGroupsStart,
      PiCCSInputs.priorRunningStart, PiCCSInputs.runningGroupWords,
      StateBinding.runningGroupStart,
      cubeVariables]
  · norm_num [productionShape, productionProfile,
      Phi81MatrixSource.phi81Shape, ringDegree]

theorem serializeRunning_group
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    {state : Nat → F} (canonical : Canonical state)
    (source : Fin productionShape.runningCount) :
    block (serializeCommitment
        ((directRunning logicalWidth publicFits state).commitments source)) ++
      block (serializePublicInput (publicFits := publicFits)
        ((directRunning logicalWidth publicFits state).publicInputs source)) ++
      block (serializeEvaluations
        ((directRunning logicalWidth publicFits state).evaluations source)) =
      slice state (StateBinding.runningGroupStart source.val) 3081 := by
  have commitmentBlock : block (serializeCommitment
        ((directRunning logicalWidth publicFits state).commitments source)) =
      slice state (StateBinding.runningGroupStart source.val) 1189 := by
    exact block_eq_slice state (StateBinding.runningGroupStart source.val) 1188 _
      (serializeRunning_commitmentPayload logicalWidth publicFits state source)
      (canonical_commitmentHeader canonical source)
  have publicBlock : block (serializePublicInput (publicFits := publicFits)
        ((directRunning logicalWidth publicFits state).publicInputs source)) =
      slice state (StateBinding.runningGroupStart source.val + 1189) 271 := by
    exact block_eq_slice state
      (StateBinding.runningGroupStart source.val + 1189) 270 _
      (serializeRunning_publicPayload logicalWidth publicFits state source)
      (canonical_publicHeader canonical source)
  have evaluationBlock : block (serializeEvaluations
        ((directRunning logicalWidth publicFits state).evaluations source)) =
      slice state (StateBinding.runningGroupStart source.val + 1460) 1621 := by
    exact block_eq_slice state
      (StateBinding.runningGroupStart source.val + 1460) 1620 _
      (serializeRunning_evaluationPayload logicalWidth publicFits state source)
      (canonical_evaluationHeader canonical source)
  rw [commitmentBlock, publicBlock, evaluationBlock]
  calc
    slice state (StateBinding.runningGroupStart source.val) 1189 ++
          slice state (StateBinding.runningGroupStart source.val + 1189) 271 ++
          slice state (StateBinding.runningGroupStart source.val + 1460) 1621 =
        slice state (StateBinding.runningGroupStart source.val) (1189 + 271) ++
          slice state (StateBinding.runningGroupStart source.val + 1460)
            1621 := by
      rw [← slice_add]
    _ = slice state (StateBinding.runningGroupStart source.val)
          ((1189 + 271) + 1621) := by
      rw [← slice_add]
    _ = slice state (StateBinding.runningGroupStart source.val) 3081 := by
      norm_num

theorem serializeRunning_pointBlock
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    {state : Nat → F} (canonical : Canonical state) :
    block (serializePoint
        (directRunning logicalWidth publicFits state).point) =
      slice state PiCCSInputs.priorRunningStart 57 := by
  apply block_eq_slice state PiCCSInputs.priorRunningStart 56
  · change serializePoint (point state PiCCSInputs.runningPointStart) = _
    simpa [PiCCSInputs.runningPointStart, PiCCSInputs.priorRunningStart,
      cubeVariables] using serializePoint_point state
        PiCCSInputs.runningPointStart
  · simpa [cubeVariables] using canonical_pointHeader canonical

theorem serializeRunning_groups
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    {state : Nat → F} (canonical : Canonical state) :
    (List.finRange productionShape.runningCount).flatMap (fun source =>
      block (serializeCommitment
        ((directRunning logicalWidth publicFits state).commitments source)) ++
      block (serializePublicInput (publicFits := publicFits)
        ((directRunning logicalWidth publicFits state).publicInputs source)) ++
      block (serializeEvaluations
        ((directRunning logicalWidth publicFits state).evaluations source))) =
      slice state 96 (productionShape.runningCount * 3081) := by
  calc
    _ = (List.finRange productionShape.runningCount).flatMap (fun source =>
          slice state (StateBinding.runningGroupStart source.val) 3081) := by
      apply List.flatMap_congr
      intro source _member
      exact serializeRunning_group logicalWidth publicFits canonical source
    _ = (List.finRange productionShape.runningCount).flatMap (fun source =>
          slice state (96 + source.val * 3081) 3081) := by
      apply List.flatMap_congr
      intro source _member
      apply congrArg (fun start => slice state start 3081)
      simp [StateBinding.runningGroupStart, cubeVariables]
    _ = slice state 96 (productionShape.runningCount * 3081) :=
      (slice_mul state 96 productionShape.runningCount 3081).symm

/-- Every canonical raw running interval is the serialization of its unique
typed running-instance decode. -/
theorem serializeRunning_running
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    {state : Nat → F} (canonical : Canonical state) :
    serializeRunning (publicFits := publicFits)
        (running logicalWidth publicFits state) =
      slice state PiCCSInputs.priorRunningStart 49353 := by
  rw [← directRunning_eq_running logicalWidth publicFits state]
  unfold serializeRunning
  rw [serializeRunning_pointBlock logicalWidth publicFits canonical]
  rw [serializeRunning_groups logicalWidth publicFits canonical]
  calc
    slice state PiCCSInputs.priorRunningStart 57 ++
          slice state 96 (productionShape.runningCount * 3081) =
        slice state PiCCSInputs.priorRunningStart
          (57 + productionShape.runningCount * 3081) := by
      simpa [PiCCSInputs.priorRunningStart] using
        (slice_add state PiCCSInputs.priorRunningStart 57
          (productionShape.runningCount * 3081)).symm
    _ = slice state PiCCSInputs.priorRunningStart 49353 := by
      apply congrArg (slice state PiCCSInputs.priorRunningStart)
      norm_num [productionShape, productionProfile,
        Phi81MatrixSource.phi81Shape]

def keyDigest (state : Nat → F) : KeyDigest :=
  slice state StateBinding.contextWordStart PilotProduction.digestWords

def iteration (state : Nat → F) : Nat :=
  (state RunningTransitionInputs.iterationWordIndex).val

def initialState (state : Nat → F) : AppState :=
  slice state RunningTransitionInputs.initialStateWordStart
    Lifecycle.Stage1.Application.stateWordCount

def currentState (state : Nat → F) : AppState :=
  slice state RunningTransitionInputs.currentStateWordStart
    Lifecycle.Stage1.Application.stateWordCount

/-- Decode the exact Construction 2 state preimage. Stage 1 has one key,
one running slot, and the one-based program counter is fixed to one. -/
def preimage
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) :
    HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits) where
  verifierKeys := fun _ => keyDigest state
  iteration := iteration state
  z0 := initialState state
  current := currentState state
  running := fun _ => running logicalWidth publicFits state
  pc := 1

theorem natWord_val (value : F) : natWord value.val = value := by
  apply Fin.ext
  simp [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt value.isLt]

theorem natWord_val_add_one (value : F) :
    natWord (value.val + 1) = value + 1 := by
  apply Fin.ext
  simp [natWord, Poseidon2.ofNat, Fin.val_add, Nat.add_mod]

theorem canonical_tagWord {state : Nat → F}
    (canonical : Canonical state)
    (index : Fin stateDomainTag.length) :
    state index.val = stateDomainTag.getD index.val 0 := by
  apply canonical ⟨index.val, stateDomainTag.getD index.val 0⟩
  simp only [StateBinding.fixedWords, List.mem_append]
  apply Or.inl
  apply Or.inl
  apply Or.inl
  rw [StateBinding.tagWords, List.mem_map]
  exact ⟨index, by simp, rfl⟩

theorem stateDomainTag_eq_slice {state : Nat → F}
    (canonical : Canonical state) :
    stateDomainTag = slice state 0 stateDomainTag.length := by
  apply List.ext_get
  · simp
  · intro index leftBound rightBound
    let bounded : Fin stateDomainTag.length := ⟨index, leftBound⟩
    have tagValue := canonical_tagWord canonical bounded
    rw [List.getD_eq_get stateDomainTag 0 bounded] at tagValue
    have sliceValue := slice_getD state 0 stateDomainTag.length index rightBound
    rw [List.getD_eq_get (slice state 0 stateDomainTag.length) 0
      ⟨index, by simpa using rightBound⟩] at sliceValue
    calc
      stateDomainTag.get ⟨index, leftBound⟩ = state index := by
        simpa [bounded] using tagValue.symm
      _ = state (0 + index) := by simp
      _ = (slice state 0 stateDomainTag.length).get
          ⟨index, rightBound⟩ := by
        simpa using sliceValue.symm

theorem canonical_keyHeader {state : Nat → F}
    (canonical : Canonical state) : state 23 = natWord 4 := by
  apply canonical ⟨23, natWord 4⟩
  simp only [StateBinding.fixedWords, List.mem_append]
  apply Or.inl
  apply Or.inl
  apply Or.inr
  simp [natWord]

theorem canonical_initialHeader {state : Nat → F}
    (canonical : Canonical state) : state 29 = natWord 4 := by
  apply canonical ⟨29, natWord 4⟩
  simp only [StateBinding.fixedWords, List.mem_append]
  apply Or.inl
  apply Or.inl
  apply Or.inr
  simp [natWord]

theorem canonical_currentHeader {state : Nat → F}
    (canonical : Canonical state) : state 34 = natWord 4 := by
  apply canonical ⟨34, natWord 4⟩
  simp only [StateBinding.fixedWords, List.mem_append]
  apply Or.inl
  apply Or.inl
  apply Or.inr
  simp [natWord]

theorem canonical_pc {state : Nat → F}
    (canonical : Canonical state) : state 49392 = natWord 1 := by
  apply canonical ⟨49392, natWord 1⟩
  simp only [StateBinding.fixedWords, List.mem_append]
  apply Or.inr
  simp [natWord]

theorem keyBlock_eq_slice {state : Nat → F}
    (canonical : Canonical state) :
    block (keyDigest state) = slice state 23 5 := by
  apply block_eq_slice state 23 4
  · simp [keyDigest, StateBinding.contextWordStart,
      PilotProduction.digestWords, PilotValues.digestWords]
  · exact canonical_keyHeader canonical

theorem iterationWord_eq_slice (state : Nat → F) :
    [natWord (iteration state)] = slice state 28 1 := by
  apply List.ext_get
  · simp
  · intro index leftBound rightBound
    have indexZero : index = 0 := by simpa using leftBound
    subst index
    simp [iteration, RunningTransitionInputs.iterationWordIndex, slice,
      natWord_val]

theorem initialBlock_eq_slice {state : Nat → F}
    (canonical : Canonical state) :
    block (initialState state) = slice state 29 5 := by
  apply block_eq_slice state 29 4
  · simp [initialState, RunningTransitionInputs.initialStateWordStart,
      Lifecycle.Stage1.Application.stateWordCount]
  · exact canonical_initialHeader canonical

theorem currentBlock_eq_slice {state : Nat → F}
    (canonical : Canonical state) :
    block (currentState state) = slice state 34 5 := by
  apply block_eq_slice state 34 4
  · simp [currentState, RunningTransitionInputs.currentStateWordStart,
      Lifecycle.Stage1.Application.stateWordCount]
  · exact canonical_currentHeader canonical

theorem pcWord_eq_slice {state : Nat → F}
    (canonical : Canonical state) :
    [natWord 1] = slice state 49392 1 := by
  apply List.ext_get
  · simp
  · intro index leftBound rightBound
    have indexZero : index = 0 := by simpa using leftBound
    subst index
    simpa [slice] using (canonical_pc canonical).symm

/-- Every value array accepted by the fixed-word rows is exactly the
canonical serialization of its typed Construction 2 decode. -/
theorem serializePreimage_preimage
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    {state : Nat → F} (canonical : Canonical state) :
    serializePreimage (publicFits := publicFits)
        (preimage logicalWidth publicFits state) =
      List.ofFn fun index : Fin PilotProduction.stateHashWords =>
        state index.val := by
  unfold serializePreimage preimage
  change stateDomainTag ++ block (keyDigest state) ++
      [natWord (iteration state)] ++ block (initialState state) ++
      block (currentState state) ++
      serializeRunning (publicFits := publicFits)
        (running logicalWidth publicFits state) ++ [natWord 1] = _
  rw [stateDomainTag_eq_slice canonical, keyBlock_eq_slice canonical,
    iterationWord_eq_slice state, initialBlock_eq_slice canonical,
    currentBlock_eq_slice canonical,
    serializeRunning_running logicalWidth publicFits canonical,
    pcWord_eq_slice canonical]
  simp only [PiCCSInputs.priorRunningStart, stateDomainTag_length]
  simp only [List.append_assoc]
  rw [← slice_add state 39 49353 1]
  rw [← slice_add state 34 5 49354]
  rw [← slice_add state 29 5 49359]
  rw [← slice_add state 28 1 49364]
  rw [← slice_add state 23 5 49365]
  norm_num
  rw [← slice_add state 0 23 49370]
  unfold slice
  apply congrArg List.ofFn
  funext index
  simp

@[simp] theorem keyDigest_length (state : Nat → F) :
    (keyDigest state).length = PilotProduction.digestWords := by
  simp [keyDigest]

@[simp] theorem initialState_length (state : Nat → F) :
    (initialState state).length =
      Lifecycle.Stage1.Application.stateWordCount := by
  simp [initialState]

@[simp] theorem currentState_length (state : Nat → F) :
    (currentState state).length =
      Lifecycle.Stage1.Application.stateWordCount := by
  simp [currentState]

theorem preimage_fixed
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) :
    PilotProduction.FixedPreimage
      (preimage logicalWidth publicFits state) := by
  refine ⟨?_, ?_, ?_⟩ <;>
    simp [preimage, PilotProduction.digestWords, PilotValues.digestWords,
      Lifecycle.Stage1.Application.stateWordCount]

theorem preimage_wellFormed
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) :
    StateEncoding.WellFormed (preimage logicalWidth publicFits state) := by
  refine ⟨preimage_fixed logicalWidth publicFits state, ?_, rfl⟩
  exact (state RunningTransitionInputs.iterationWordIndex).isLt

/-- Canonical decoded words represent the complete prior hash preimage at
the pilot interface. The agreement is over the actual hashed word interval. -/
theorem priorRepresents
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Circuit.Env) (state : Nat → F)
    (canonical : Canonical state)
    (agrees : ∀ word : Fin PilotProduction.stateHashWords,
      env (PilotProduction.priorPreimageStart + word.val) = state word.val) :
    PriorStateHash.RepresentsPreimage PilotProduction.priorInterface
      PilotProduction.witnessOffset env (preimage logicalWidth publicFits state) := by
  unfold PriorStateHash.RepresentsPreimage
  rw [PilotProduction.priorInterface_preimage_apply]
  simp only [PilotProduction.priorPreimage,
    NightstreamFPrime.Gadgets.Poseidon2.Hash.evalList,
    PilotProduction.variableExprs, List.map_ofFn]
  rw [serializePreimage_preimage _ _ canonical]
  exact congrArg List.ofFn (funext agrees)

/-- Canonical decoded words represent the complete output hash preimage at
the pilot interface. No coordinate-encoding premise is needed. -/
theorem outputRepresents
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (env : Circuit.Env) (state : Nat → F)
    (canonical : Canonical state)
    (agrees : ∀ word : Fin PilotProduction.stateHashWords,
      env (PilotProduction.outputPreimageStart + word.val) = state word.val) :
    OutputHash.RepresentsPreimage PilotProduction.outputInterface
      (Pilot.outputOffset PilotProduction.interface PilotProduction.witnessOffset)
      env (preimage logicalWidth publicFits state) := by
  unfold OutputHash.RepresentsPreimage
  rw [PilotProduction.outputInterface_preimage_apply]
  simp only [PilotProduction.outputPreimage,
    NightstreamFPrime.Gadgets.Poseidon2.Hash.evalList,
    PilotProduction.variableExprs, List.map_ofFn]
  rw [serializePreimage_preimage _ _ canonical]
  exact congrArg List.ofFn (funext agrees)

end NightstreamFPrime.Layout.Stage1.StateDecoder
