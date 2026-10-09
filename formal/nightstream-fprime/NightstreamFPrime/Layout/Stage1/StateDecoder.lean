import Mathlib.Data.List.OfFn
import NightstreamFPrime.Layout.Stage1.RunningTransitionData
import NightstreamFPrime.Layout.Stage1.StateEncoding

/-!
Owns structural decoding of the fixed Stage 1 state block.

The decoder is a value view only. It reads the running fields and the tail in
place and takes every child public input as `split_b` of the unpacked parent.
Hence `serializeRunning ∘ running` is the identity on every word array, and
`running ∘ serializeRunning` is the identity on every canonical running
instance. It does not select a package, application, verification key, or
transcript.
-/

namespace NightstreamFPrime.Layout.Stage1.StateDecoder

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

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

def evaluations (state : Nat → F) (source : Nat) :
    StrongReduction.EvaluationFamily K productionShape where
  pad := fun coefficient =>
    pair state (PiCCSInputs.runningEvalKStart source + coefficient.val * 2)
  matrix := fun matrix coefficient => pair state
    (PiCCSInputs.runningEvalAStart source +
      matrix.val * (productionShape.coefficientCount * 2) + coefficient.val * 2)

def packedWords (state : Nat → F) : Fin packedParentWords → F :=
  fun word => state (StateBinding.packedWordStart + word.val)

/-- The running instance stored in one state block. Each child public input
is `split_b` of the unpacked parent coordinate. -/
def running
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    (state : Nat → F) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) where
  point := point state PiCCSInputs.runningPointStart
  commitments := fun source =>
    commitment state (PiCCSInputs.runningCommitmentStart source.val)
  publicInputs := fun source column =>
    Radix.splitScalar (unpackParent (packedWords state) column)
      (Fin.cast runningCount_eq_radixChildCount source)
  evaluations := fun source => evaluations state source.val

theorem unpackParent_packedColumn (packed : Fin packedParentWords → F)
    (word : Fin packedParentWords) (lane : Fin 3) :
    unpackParent (logicalWidth := logicalWidth) (publicFits := publicFits) packed
        (packedColumn word lane) =
      unpackWord (packed word) lane := by
  have laneBound := lane.isLt
  unfold unpackParent packedColumn
  congr 2 <;> (try apply Fin.ext) <;> dsimp only <;> omega

theorem parentPublic_running (state : Nat → F)
    (column : Fin (FullShape logicalWidth publicFits).publicWidth) :
    parentPublic (running logicalWidth publicFits state) column =
      unpackParent (packedWords state) column := by
  change Radix.recomposeScalar
      (Radix.splitScalar (unpackParent (packedWords state) column)) = _
  exact Radix.splitScalar_recompose _

theorem serializeParentPublic_running (state : Nat → F) :
    serializeParentPublic (running logicalWidth publicFits state) =
      slice state StateBinding.packedWordStart packedParentWords := by
  unfold serializeParentPublic slice
  rw [List.ofFn_eq_map]
  apply List.map_congr_left
  intro word _member
  rw [parentPublic_running, parentPublic_running, parentPublic_running,
    unpackParent_packedColumn, unpackParent_packedColumn,
    unpackParent_packedColumn, StateEncoding.packWord_unpackWord]
  rfl

/-- Every word array is the serialization of its decoded running instance. -/
theorem serializeRunning_running (state : Nat → F) :
    serializeRunning (publicFits := publicFits)
        (running logicalWidth publicFits state) =
      slice state PiCCSInputs.priorRunningStart 27794 := by
  have commitments :
      serializeCommitments (running logicalWidth publicFits state) =
        slice state 12 19008 := by
    unfold serializeCommitments
    calc
      _ = (List.finRange productionShape.runningCount).flatMap
          (fun source => slice state (12 + source.val * 1188) 1188) := by
        apply List.flatMap_congr
        intro source _member
        change serializeCommitment
            (commitment state (PiCCSInputs.runningCommitmentStart source.val)) = _
        rw [serializeCommitment_commitment]
        rfl
      _ = slice state 12 (productionShape.runningCount * 1188) :=
        (slice_mul state 12 productionShape.runningCount 1188).symm
  have evalKs :
      serializeEvalKs (running logicalWidth publicFits state) =
        slice state 19020 1728 := by
    unfold serializeEvalKs
    calc
      _ = (List.finRange productionShape.runningCount).flatMap
          (fun source => slice state (19020 + source.val * 108) 108) := by
        apply List.flatMap_congr
        intro source _member
        change (List.finRange productionShape.coefficientCount).flatMap
            (fun coefficient => serializeK (pair state
              (PiCCSInputs.runningEvalKStart source.val + coefficient.val * 2))) = _
        rw [serializePairs]
        rfl
      _ = slice state 19020 (productionShape.runningCount * 108) :=
        (slice_mul state 19020 productionShape.runningCount 108).symm
  have evalAs :
      serializeEvalAs (running logicalWidth publicFits state) =
        slice state 20748 6912 := by
    unfold serializeEvalAs
    calc
      _ = (List.finRange productionShape.runningCount).flatMap
          (fun source => slice state (20748 + source.val * 432) 432) := by
        apply List.flatMap_congr
        intro source _member
        change (List.finRange productionShape.matrixCount).flatMap (fun matrix =>
            (List.finRange productionShape.coefficientCount).flatMap
              (fun coefficient => serializeK (pair state
                (PiCCSInputs.runningEvalAStart source.val +
                  matrix.val * (productionShape.coefficientCount * 2) +
                  coefficient.val * 2)))) = _
        rw [serializePairRows]
        rfl
      _ = slice state 20748 (productionShape.runningCount * 432) :=
        (slice_mul state 20748 productionShape.runningCount 432).symm
  have parent := serializeParentPublic_running (logicalWidth := logicalWidth)
    (publicFits := publicFits) state
  have pointWords : serializePoint (running logicalWidth publicFits state).point =
      slice state 27660 56 :=
    serializePoint_point state PiCCSInputs.runningPointStart
  unfold serializeRunning serializeRunningFields
  rw [commitments, evalKs, evalAs, pointWords, parent]
  rw [show PiCCSInputs.priorRunningStart = 12 from rfl,
    show (27794 : Nat) = 19008 + 1728 + 6912 + 56 + 90 from rfl,
    slice_add state 12 (19008 + 1728 + 6912 + 56) 90,
    slice_add state 12 (19008 + 1728 + 6912) 56,
    slice_add state 12 (19008 + 1728) 6912,
    slice_add state 12 19008 1728]
  rfl

/-- A canonical running instance is the decode of its serialized words. -/
theorem running_eq_of_serialized {state : Nat → F}
    {value : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (canonical : Lifecycle.ChildrenCanonical value)
    (words : slice state PiCCSInputs.priorRunningStart 27794 =
      serializeRunning (publicFits := publicFits) value) :
    running logicalWidth publicFits state = value := by
  have encoded := (serializeRunning_running (logicalWidth := logicalWidth)
    (publicFits := publicFits) state).trans words
  unfold serializeRunning at encoded
  rcases List.append_inj encoded (by simp only [serializeRunningFields_length]) with
    ⟨fieldsEqual, parentEqual⟩
  rcases StateEncoding.serializeRunningFields_injective fieldsEqual with
    ⟨pointEqual, commitmentsEqual, evaluationsEqual⟩
  apply StateEncoding.running_ext pointEqual commitmentsEqual _ evaluationsEqual
  funext source column
  rcases StateEncoding.packedColumn_cover column with ⟨word, lane, rfl⟩
  have packedValue : packedWords state word =
      packWord (parentPublic value (packedColumn word 0))
        (parentPublic value (packedColumn word 1))
        (parentPublic value (packedColumn word 2)) := by
    have wordsEq := (serializeParentPublic_running (logicalWidth := logicalWidth)
      (publicFits := publicFits) state).symm.trans parentEqual
    have selected := congrArg (fun words => words.getD word.val 0) wordsEq
    have left : (slice state StateBinding.packedWordStart packedParentWords).getD
        word.val 0 = packedWords state word :=
      slice_getD state _ _ _ word.isLt
    have right : (serializeParentPublic value).getD word.val 0 =
        packWord (parentPublic value (packedColumn word 0))
          (parentPublic value (packedColumn word 1))
          (parentPublic value (packedColumn word 2)) :=
      finRange_map_getD _ word
    exact left.symm.trans (selected.trans right)
  have bounded (lane : Fin 3) := canonical.parentBounded (packedColumn word lane)
  rcases StateEncoding.unpackWord_packWord (bounded 0) (bounded 1) (bounded 2) with
    ⟨low, middle, high⟩
  have unpacked : unpackWord (packedWords state word) lane =
      parentPublic value (packedColumn word lane) := by
    rw [packedValue]
    fin_cases lane
    · exact low
    · exact middle
    · exact high
  change Radix.splitScalar (unpackParent (packedWords state) (packedColumn word lane))
      (Fin.cast runningCount_eq_radixChildCount source) =
    value.publicInputs source (packedColumn word lane)
  rw [unpackParent_packedColumn, unpacked, ← canonical.childDigits_eq]
  rfl

/-! ## Fixed words, tail, and preimage -/

/-- The value-level form of the fixed-word rows: the domain chunk. -/
def Canonical (state : Nat → F) : Prop :=
  ∀ word ∈ StateBinding.fixedWords, state word.index = word.value

theorem stateDomainChunk_eq_slice {state : Nat → F} (canonical : Canonical state) :
    stateDomainChunk = slice state 0 12 := by
  apply List.ext_getElem
  · simp [stateDomainChunk_length]
  · intro index leftBound _rightBound
    have fixed := canonical ⟨index, stateDomainChunk.getD index 0⟩ (by
      rw [StateBinding.fixedWords, List.mem_map]
      exact ⟨⟨index, leftBound⟩, List.mem_finRange _, rfl⟩)
    simp only [slice, List.getElem_ofFn, Nat.zero_add]
    rw [fixed, List.getD_eq_getElem _ _ leftBound]

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

theorem serializeTail_preimage (state : Nat → F) :
    serializeTail (preimage logicalWidth publicFits state) =
      slice state StateBinding.contextWordStart 13 := by
  have iterationWord : [natWord (iteration state)] = slice state 27810 1 := by
    simp [iteration, RunningTransitionInputs.iterationWordIndex, slice, natWord_val]
  unfold serializeTail
  change keyDigest state ++ [natWord (iteration state)] ++ initialState state ++
    currentState state = _
  rw [iterationWord, show (13 : Nat) = 4 + 1 + 4 + 4 from rfl,
    slice_add state _ (4 + 1 + 4) 4, slice_add state _ (4 + 1) 4,
    slice_add state _ 4 1]
  rfl

/-- Every value array accepted by the fixed-word rows is exactly the
canonical serialization of its decode. -/
theorem serializePreimage_preimage
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth)
    {state : Nat → F} (canonical : Canonical state) :
    serializePreimage (publicFits := publicFits)
        (preimage logicalWidth publicFits state) =
      List.ofFn fun index : Fin PilotProduction.stateHashWords =>
        state index.val := by
  unfold serializePreimage
  change stateDomainChunk ++
      serializeRunning (publicFits := publicFits)
        (running logicalWidth publicFits state) ++
      serializeTail (preimage logicalWidth publicFits state) = _
  rw [stateDomainChunk_eq_slice canonical, serializeRunning_running,
    serializeTail_preimage]
  have joined : slice state 0 12 ++ slice state PiCCSInputs.priorRunningStart 27794 ++
      slice state StateBinding.contextWordStart 13 = slice state 0 (12 + 27794 + 13) := by
    rw [slice_add state 0 (12 + 27794) 13, slice_add state 0 12 27794]
    rfl
  rw [joined]
  unfold slice
  simp only [Nat.zero_add]
  rfl

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

theorem iteration_lt (state : Nat → F) :
    (preimage logicalWidth publicFits state).iteration < goldilocksModulus :=
  (state RunningTransitionInputs.iterationWordIndex).isLt

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

/-! ## PiCCS and running-transition readback -/

private theorem cubePoint_ext {variableCount : Nat}
    {left right : CubePoint K variableCount}
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

private theorem evaluationFamily_ext
    {left right : StrongReduction.EvaluationFamily K productionShape}
    (pad : left.pad = right.pad) (matrix : left.matrix = right.matrix) :
    left = right := by
  cases left
  cases right
  simp_all

/-- The decode reads only the running words, all below the tail. -/
theorem running_congr {left right : Nat → F}
    (agree : ∀ word, word < StateBinding.contextWordStart → left word = right word) :
    running logicalWidth publicFits left = running logicalWidth publicFits right := by
  have bound : StateBinding.contextWordStart = 27806 := rfl
  apply StateEncoding.running_ext
  · apply cubePoint_ext
    change List.ofFn (fun coordinate : Fin cubeVariables =>
        pair left (PiCCSInputs.runningPointStart + coordinate.val * 2)) =
      List.ofFn (fun coordinate : Fin cubeVariables =>
        pair right (PiCCSInputs.runningPointStart + coordinate.val * 2))
    apply congrArg List.ofFn
    funext coordinate
    have coordinateBound : coordinate.val < 28 := coordinate.isLt
    unfold pair PiCCSInputs.runningPointStart PiCCSInputs.priorRunningStart
    rw [agree _ (by omega), agree _ (by omega)]
  · funext source row coefficient
    have sourceBound : source.val < 16 := source.isLt
    have rowBound : row.val < 22 := row.isLt
    have coefficientBound : coefficient.val < 54 := coefficient.isLt
    change left (PiCCSInputs.runningCommitmentStart source.val + row.val * 54 +
        coefficient.val) =
      right (PiCCSInputs.runningCommitmentStart source.val + row.val * 54 +
        coefficient.val)
    unfold PiCCSInputs.runningCommitmentStart PiCCSInputs.priorRunningStart
      PiCCSInputs.runningCommitmentWords
    exact agree _ (by omega)
  · have packed : packedWords left = packedWords right := by
      funext word
      have wordBound : word.val < 90 := word.isLt
      unfold packedWords StateBinding.packedWordStart
      exact agree _ (by omega)
    funext source column
    change Radix.splitScalar (unpackParent (packedWords left) column) _ =
      Radix.splitScalar (unpackParent (packedWords right) column) _
    rw [packed]
  · funext source
    have sourceBound : source.val < 16 := source.isLt
    apply evaluationFamily_ext
    · funext coefficient
      have coefficientBound : coefficient.val < 54 := coefficient.isLt
      change pair left (PiCCSInputs.runningEvalKStart source.val + coefficient.val * 2) =
        pair right (PiCCSInputs.runningEvalKStart source.val + coefficient.val * 2)
      unfold pair PiCCSInputs.runningEvalKStart PiCCSInputs.priorRunningStart
        PiCCSInputs.runningEvalKWords
      rw [agree _ (by omega), agree _ (by omega)]
    · funext matrix coefficient
      have matrixBound : matrix.val < 4 := matrix.isLt
      have coefficientBound : coefficient.val < 54 := coefficient.isLt
      change pair left (PiCCSInputs.runningEvalAStart source.val +
          matrix.val * (productionShape.coefficientCount * 2) + coefficient.val * 2) =
        pair right (PiCCSInputs.runningEvalAStart source.val +
          matrix.val * (productionShape.coefficientCount * 2) + coefficient.val * 2)
      have width : productionShape.coefficientCount * 2 = 108 := rfl
      rw [width]
      unfold pair PiCCSInputs.runningEvalAStart PiCCSInputs.priorRunningStart
        PiCCSInputs.runningEvalAWords
      rw [agree _ (by omega), agree _ (by omega)]

/-- PiCCS reads the decoded prior running instance whenever its child-split
rows hold: the region digits are `split_b` of the unpacked hashed parent. -/
theorem evalRunning_eq_running (env : Env)
    (split : StateBinding.ChildrenSplit
      (Formal.statementBindingInterface
        (Formal.atOffset (PiCCSInputs.interface logicalWidth publicFits)
          PiCCSInputs.phaseOffset)).state
      PiCCSInputs.phaseOffset env) :
    StatementAbsorption.evalRunning (PiCCSInputs.runningExpr logicalWidth publicFits) env =
      running logicalWidth publicFits
        (fun word => env (PilotProduction.priorPreimageStart + word)) := by
  apply StateEncoding.running_ext
  · apply cubePoint_ext
    change List.ofFn (fun coordinate =>
        (PiCCSInputs.runningPoint coordinate).eval env) =
      List.ofFn (fun coordinate => pair
        (fun word => env (PilotProduction.priorPreimageStart + word))
        (PiCCSInputs.runningPointStart + coordinate.val * 2))
    apply congrArg List.ofFn
    funext coordinate
    simp [PiCCSInputs.runningPoint, PiCCSInputs.pairAt, pair,
      Circuit.Quadratic.KExpr.eval, PilotProduction.priorPreimageStart]
  · funext source row coefficient
    simp [StatementAbsorption.evalRunning, PiCCSInputs.runningExpr,
      PiCCSInputs.runningCommitment, running, commitment,
      PilotProduction.priorPreimageStart]
  · funext source column
    rcases StateEncoding.packedColumn_cover column with ⟨word, lane, rfl⟩
    let digits := StateBinding.priorDigits
      (Formal.statementBindingInterface
        (Formal.atOffset (PiCCSInputs.interface logicalWidth publicFits)
          PiCCSInputs.phaseOffset)).state
      PiCCSInputs.phaseOffset env word
    have accepted (lane : Fin 3) : ∃ sign, Radix.UniformSignedDigits.Accepted
        (Radix.recomposeScalar (digits lane)) sign (digits lane) := by
      obtain ⟨sign, constraint⟩ := split.digits word lane
      exact ⟨sign, constraint, rfl⟩
    have bounded (lane : Fin 3) :
        centeredMagnitude (Radix.recomposeScalar (digits lane)) < 2 ^ 16 := by
      obtain ⟨_, accepted⟩ := accepted lane
      have parentBound := accepted.parentBounded
      rw [Radix.production_parameters.2.2] at parentBound
      exact parentBound
    rcases StateEncoding.unpackWord_packWord (bounded 0) (bounded 1) (bounded 2) with
      ⟨low, middle, high⟩
    have unpacked : unpackWord
        (packedWords (fun word => env (PilotProduction.priorPreimageStart + word)) word)
        lane = Radix.recomposeScalar (digits lane) := by
      rw [show packedWords (fun word => env (PilotProduction.priorPreimageStart + word)) word =
          packWord (Radix.recomposeScalar (digits 0)) (Radix.recomposeScalar (digits 1))
            (Radix.recomposeScalar (digits 2)) from split.packed word]
      fin_cases lane
      · exact low
      · exact middle
      · exact high
    show env (PiCCSInputs.runningPublicStart source.val + (packedColumn word lane).val) =
      Radix.splitScalar (unpackParent
          (packedWords (fun word => env (PilotProduction.priorPreimageStart + word)))
          (packedColumn word lane))
        (Fin.cast runningCount_eq_radixChildCount source)
    obtain ⟨_, acceptedLane⟩ := accepted lane
    rw [unpackParent_packedColumn, unpacked, ← acceptedLane.digits_eq_splitScalar]
    rfl
  · funext source
    apply evaluationFamily_ext
    · funext coefficient
      simp [StatementAbsorption.evalRunning, StatementAbsorption.evalEvaluation,
        PiCCSInputs.runningExpr, PiCCSInputs.runningEval_K, PiCCSInputs.pairAt,
        Circuit.Quadratic.KExpr.eval, running, evaluations, pair,
        PilotProduction.priorPreimageStart]
    · funext matrix coefficient
      simp [StatementAbsorption.evalRunning, StatementAbsorption.evalEvaluation,
        PiCCSInputs.runningExpr, PiCCSInputs.runningEval_A, PiCCSInputs.pairAt,
        Circuit.Quadratic.KExpr.eval, running, evaluations, pair,
        PilotProduction.priorPreimageStart, productionShape,
        Phi81MatrixSource.phi81Shape, ringDegree]

/-- The checked prior child digits make the decoded prior running instance
canonical. -/
theorem running_canonical (env : Env)
    (split : StateBinding.ChildrenSplit
      (Formal.statementBindingInterface
        (Formal.atOffset (PiCCSInputs.interface logicalWidth publicFits)
          PiCCSInputs.phaseOffset)).state
      PiCCSInputs.phaseOffset env) :
    Lifecycle.ChildrenCanonical (running logicalWidth publicFits
      (fun word => env (PilotProduction.priorPreimageStart + word))) := by
  rw [← evalRunning_eq_running env split]
  intro column
  rcases StateEncoding.packedColumn_cover column with ⟨word, lane, rfl⟩
  exact split.digits word lane

/-- The running-transition output words are the output block's running words. -/
theorem outputWords_eq_slice (env : Env) :
    Lifecycle.Stage1.RunningTransition.outputWords (RunningTransitionInputs.interface logicalWidth publicFits)
        RunningTransitionInputs.phaseOffset env =
      slice (fun word => env (PilotProduction.outputPreimageStart + word))
        PiCCSInputs.priorRunningStart 27794 := by
  unfold Lifecycle.Stage1.RunningTransition.outputWords slice
  apply congrArg List.ofFn
  funext index
  simp [RunningTransitionInputs.interface, RunningTransitionInputs.outputWord,
    RunningTransitionInputs.outputBase, Nat.add_assoc]

/-- Output words that serialize a canonical running instance decode to it. -/
theorem outputRunning_eq_of_serialized (env : Env)
    {value : Running (logicalWidth := logicalWidth) (publicFits := publicFits)}
    (canonical : Lifecycle.ChildrenCanonical value)
    (words : Lifecycle.Stage1.RunningTransition.outputWords
        (RunningTransitionInputs.interface logicalWidth publicFits)
        RunningTransitionInputs.phaseOffset env =
      serializeRunning (publicFits := publicFits) value) :
    running logicalWidth publicFits
        (fun word => env (PilotProduction.outputPreimageStart + word)) = value :=
  running_eq_of_serialized canonical ((outputWords_eq_slice env).symm.trans words)

end NightstreamFPrime.Layout.Stage1.StateDecoder
