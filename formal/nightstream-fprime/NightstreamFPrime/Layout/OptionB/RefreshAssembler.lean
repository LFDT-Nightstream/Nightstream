import NightstreamFPrime.Lifecycle.OptionB.RefreshStepSoundness
import NightstreamFPrime.Layout.OptionB.RefreshStepBounds
import NightstreamFPrime.Layout.OptionB.LightAssembler

/-! The refresh step's source layout (CW-3.6). Every shared word of `RefreshStepLayout.Interface`
is one variable in a source region that starts at `base`. The first groups are the light step's
(`LightAssembler`): the counters and selector (5), the key digests (8), the start, current and
next states (12), the application witness, the prior, active, inner and next slots
(7 × 2,055), the slot and output digests (20) and the fresh instance (1,458). Then come the 16
children's commitments (19,008) and evaluation families (8,640), their digit words (4,320), the
round messages (504) and the Π_CCS output of all 17 sources (9,180). The step's gadgets start at
`sharedEnd`. -/

namespace NightstreamFPrime.Layout.OptionB.RefreshAssembler

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Layout.OptionB.LightAssembler (kAt slotAt slotAt_below slotWordCount
  keysStart statesStart witnessStart slotsStart digestsStart freshStart)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Where each group of the refresh step's own shared words starts. -/
def childCommitmentsStart (base witnessWordCount : Nat) : Nat :=
  freshStart base witnessWordCount + 1458
def childEvaluationsStart (base witnessWordCount : Nat) : Nat :=
  childCommitmentsStart base witnessWordCount + 19008
def digitsStart (base witnessWordCount : Nat) : Nat :=
  childEvaluationsStart base witnessWordCount + 8640
def roundsStart (base witnessWordCount : Nat) : Nat := digitsStart base witnessWordCount + 4320
def outputStart (base witnessWordCount : Nat) : Nat := roundsStart base witnessWordCount + 504

/-- The first column after the shared words, where the step's gadgets start. -/
def sharedEnd (base witnessWordCount : Nat) : Nat := outputStart base witnessWordCount + 9180

theorem sharedEnd_eq (base witnessWordCount : Nat) :
    sharedEnd base witnessWordCount = base + witnessWordCount + 57540 := by
  simp only [sharedEnd, outputStart, roundsStart, digitsStart, childEvaluationsStart,
    childCommitmentsStart, freshStart, digestsStart, slotsStart, witnessStart, slotWordCount]
  omega

/-- The step's shared words as source columns from `base`. -/
def interface (base witnessWordCount : Nat) :
    RefreshStepLayout.Interface logicalWidth publicFits witnessWordCount where
  iteration := .var base
  inverse := .var (base + 1)
  recursive := .var (base + 2)
  priorPc := .var (base + 3)
  selector := .var (base + 4)
  verifierKeys := fun key lane => .var (keysStart base + 4 * key.val + lane.val)
  initialState := fun lane => .var (statesStart base + lane.val)
  currentState := fun lane => .var (statesStart base + 4 + lane.val)
  nextState := fun lane => .var (statesStart base + 8 + lane.val)
  witness := fun index => .var (witnessStart base + index.val)
  prior := fun slot => slotAt (slotsStart base witnessWordCount + slot.val * slotWordCount)
  active := slotAt (slotsStart base witnessWordCount + 2 * slotWordCount)
  inner := fun slot => slotAt (slotsStart base witnessWordCount + (3 + slot.val) * slotWordCount)
  next := fun slot => slotAt (slotsStart base witnessWordCount + (5 + slot.val) * slotWordCount)
  priorDigest := fun slot lane =>
    .var (digestsStart base witnessWordCount + 4 * slot.val + lane.val)
  nextDigest := fun slot lane =>
    .var (digestsStart base witnessWordCount + 8 + 4 * slot.val + lane.val)
  outputDigest := fun lane => .var (digestsStart base witnessWordCount + 16 + lane.val)
  fresh :=
    ⟨fun _ row lane => .var (freshStart base witnessWordCount + row.val * ringDegree + lane.val),
      fun _ column => .var (freshStart base witnessWordCount + 1188 + column.val)⟩
  childCommitment := fun child row lane =>
    .var (childCommitmentsStart base witnessWordCount + child.val * 1188 + row.val * ringDegree +
      lane.val)
  childEvaluation := fun child =>
    ⟨fun coefficient =>
        kAt (childEvaluationsStart base witnessWordCount + child.val * 540 + 2 * coefficient.val),
      fun matrix coefficient =>
        kAt (childEvaluationsStart base witnessWordCount + child.val * 540 + 108 +
          2 * (matrix.val * 54 + coefficient.val))⟩
  digit := fun child column => .var (digitsStart base witnessWordCount + child.val * 270 +
    column.val)
  round := fun roundIndex => ⟨fun coefficient =>
    kAt (roundsStart base witnessWordCount + 2 * (roundIndex.val * 9 + coefficient.val))⟩
  output := ⟨fun source coefficient =>
      kAt (outputStart base witnessWordCount + 2 * (source.val * 270 + coefficient.val)),
    fun source matrix coefficient =>
      kAt (outputStart base witnessWordCount +
        2 * (source.val * 270 + 54 + matrix.val * 54 + coefficient.val))⟩

private theorem kAt_below (column bound : Nat) (below : column + 1 < bound) :
    (kAt column).VarsBelow bound :=
  ⟨show column < bound by omega, show column + 1 < bound from below⟩

/-- Every shared word of the source layout lies below `sharedEnd`. -/
theorem inputsBelow (base witnessWordCount : Nat) :
    RefreshStepBounds.InputsBelow
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base witnessWordCount)
      (sharedEnd base witnessWordCount) := by
  have endEq := sharedEnd_eq base witnessWordCount
  have slotsEq : slotsStart base witnessWordCount = base + 25 + witnessWordCount := rfl
  have freshEq : freshStart base witnessWordCount = base + 25 + witnessWordCount + 14405 := by
    simp only [freshStart, digestsStart, slotsStart, witnessStart, slotWordCount]
  have activeWord := fun (word : Nat) (bounded : word < slotWordCount) =>
    (by rw [endEq, slotsEq] at *; simp only [slotWordCount] at *; omega :
      slotsStart base witnessWordCount + 2 * slotWordCount + word < sharedEnd base witnessWordCount)
  refine
    { iteration := show base < _ by rw [endEq]; omega
      priorPc := show base + 3 < _ by rw [endEq]; omega
      verifierKeys := fun key lane => by
        have keyBound := key.isLt
        have laneBound := lane.isLt
        show keysStart base + 4 * key.val + lane.val < _
        rw [endEq]
        simp only [keysStart]
        omega
      initialState := fun lane => by
        have laneBound := lane.isLt
        show statesStart base + lane.val < _
        rw [endEq]
        simp only [statesStart]
        omega
      currentState := fun lane => by
        have laneBound := lane.isLt
        show statesStart base + 4 + lane.val < _
        rw [endEq]
        simp only [statesStart]
        omega
      nextState := fun lane => by
        have laneBound := lane.isLt
        show statesStart base + 8 + lane.val < _
        rw [endEq]
        simp only [statesStart]
        omega
      witness := fun index => by
        have indexBound := index.isLt
        show witnessStart base + index.val < _
        rw [endEq]
        simp only [witnessStart]
        omega
      prior := fun slot => slotAt_below _ _ (by
        have slotBound := slot.isLt
        rw [endEq, slotsEq]
        simp only [slotWordCount]
        omega)
      next := fun slot => slotAt_below _ _ (by
        have slotBound := slot.isLt
        rw [endEq, slotsEq]
        simp only [slotWordCount]
        omega)
      activePoint := fun coordinate => kAt_below _ _ (by
        have coordinateBound := coordinate.isLt
        change coordinate.val < 28 at coordinateBound
        have := activeWord (1728 + 2 * coordinate.val + 1) (by simp only [slotWordCount]; omega)
        omega)
      activePad := fun coefficient => kAt_below _ _ (by
        have coefficientBound := coefficient.isLt
        change coefficient.val < 54 at coefficientBound
        have := activeWord (1188 + 2 * coefficient.val + 1) (by simp only [slotWordCount]; omega)
        omega)
      activeMatrix := fun matrix coefficient => kAt_below _ _ (by
        have matrixBound := matrix.isLt
        have coefficientBound := coefficient.isLt
        change matrix.val < 4 at matrixBound
        change coefficient.val < 54 at coefficientBound
        have := activeWord (1296 + 2 * (matrix.val * 54 + coefficient.val) + 1)
          (by simp only [slotWordCount]; omega)
        simp only [ringDegree]
        omega)
      activeCommitment := fun row lane => by
        have rowBound := row.isLt
        have laneBound := lane.isLt
        change row.val < 22 at rowBound
        change lane.val < 54 at laneBound
        have := activeWord (row.val * 54 + lane.val) (by simp only [slotWordCount]; omega)
        show _ < _
        simp only [ringDegree]
        omega
      activePublicInput := fun column => by
        have columnBound := column.isLt
        change column.val < 270 at columnBound
        have := activeWord (1784 + column.val) (by simp only [slotWordCount]; omega)
        show _ < _
        omega
      priorDigest := fun slot lane => by
        have slotBound := slot.isLt
        have laneBound := lane.isLt
        show digestsStart base witnessWordCount + 4 * slot.val + lane.val < _
        rw [endEq]
        simp only [digestsStart, slotsStart, witnessStart, slotWordCount]
        omega
      nextDigest := fun slot lane => by
        have slotBound := slot.isLt
        have laneBound := lane.isLt
        show digestsStart base witnessWordCount + 8 + 4 * slot.val + lane.val < _
        rw [endEq]
        simp only [digestsStart, slotsStart, witnessStart, slotWordCount]
        omega
      outputDigest := fun lane => by
        have laneBound := lane.isLt
        show digestsStart base witnessWordCount + 16 + lane.val < _
        rw [endEq]
        simp only [digestsStart, slotsStart, witnessStart, slotWordCount]
        omega
      freshCommitment := fun _ row lane => by
        have rowBound := row.isLt
        have laneBound := lane.isLt
        change row.val < 22 at rowBound
        change lane.val < 54 at laneBound
        show freshStart base witnessWordCount + row.val * ringDegree + lane.val < _
        rw [endEq, freshEq]
        simp only [ringDegree]
        omega
      freshPublicInput := fun _ column => by
        have columnBound := column.isLt
        change column.val < 270 at columnBound
        show freshStart base witnessWordCount + 1188 + column.val < _
        rw [endEq, freshEq]
        omega
      childCommitment := fun child row lane => by
        have childBound := child.isLt
        have rowBound := row.isLt
        have laneBound := lane.isLt
        change child.val < 16 at childBound
        change row.val < 22 at rowBound
        change lane.val < 54 at laneBound
        show childCommitmentsStart base witnessWordCount + child.val * 1188 +
          row.val * ringDegree + lane.val < _
        rw [endEq]
        simp only [childCommitmentsStart, freshEq, ringDegree]
        omega
      childPad := fun child coefficient => kAt_below _ _ (by
        have childBound := child.isLt
        have coefficientBound := coefficient.isLt
        change child.val < 16 at childBound
        change coefficient.val < 54 at coefficientBound
        rw [endEq]
        simp only [childEvaluationsStart, childCommitmentsStart, freshEq]
        omega)
      childMatrix := fun child matrix coefficient => kAt_below _ _ (by
        have childBound := child.isLt
        have matrixBound := matrix.isLt
        have coefficientBound := coefficient.isLt
        change child.val < 16 at childBound
        change matrix.val < 4 at matrixBound
        change coefficient.val < 54 at coefficientBound
        rw [endEq]
        simp only [childEvaluationsStart, childCommitmentsStart, freshEq]
        omega)
      digit := fun child column => by
        have childBound := child.isLt
        have columnBound := column.isLt
        change child.val < 16 at childBound
        change column.val < 270 at columnBound
        show digitsStart base witnessWordCount + child.val * 270 + column.val < _
        rw [endEq]
        simp only [digitsStart, childEvaluationsStart, childCommitmentsStart, freshEq]
        omega
      round := fun roundIndex coefficient => kAt_below _ _ (by
        have roundBound := roundIndex.isLt
        have coefficientBound := coefficient.isLt
        change roundIndex.val < 28 at roundBound
        rw [endEq]
        simp only [roundsStart, digitsStart, childEvaluationsStart, childCommitmentsStart,
          freshEq]
        omega)
      outputPad := fun source coefficient => kAt_below _ _ (by
        have sourceBound := source.isLt
        have coefficientBound := coefficient.isLt
        change source.val < 17 at sourceBound
        change coefficient.val < 54 at coefficientBound
        rw [endEq]
        simp only [outputStart, roundsStart, digitsStart, childEvaluationsStart,
          childCommitmentsStart, freshEq]
        omega)
      outputMatrix := fun source matrix coefficient => kAt_below _ _ (by
        have sourceBound := source.isLt
        have matrixBound := matrix.isLt
        have coefficientBound := coefficient.isLt
        change source.val < 17 at sourceBound
        change matrix.val < 4 at matrixBound
        change coefficient.val < 54 at coefficientBound
        rw [endEq]
        simp only [outputStart, roundsStart, digitsStart, childEvaluationsStart,
          childCommitmentsStart, freshEq]
        omega) }

/-- Soundness of the refresh step circuit at the source layout: its gadgets placed at
`sharedEnd`, with rows that hold and shared words that carry the step's input, output and
refresh proof, give the refresh case of `stepHolds_iff` at the production cap. No input-scoping
hypothesis remains. -/
theorem rows_implies_stepHolds (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (base : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (refresh : BoundedDepth.circuitStep circuit = .refresh)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness →
      control state witness = circuit)
    (env : Env)
    (rows : holds env (Circuit.ops (RefreshStepLayout.main program relation circuit
      (interface base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)))
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (represents : RefreshStepSoundness.StepRepresents program
      (interface base program.witnessWordCount) ajtai circuitKeys circuit input output proof env) :
    StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control circuit input
      output :=
  RefreshStepSoundness.rows_implies_stepHolds program (interface base program.witnessWordCount)
    ajtai circuitKeys step control circuit refresh stepEq validControl relation
    (sharedEnd base program.witnessWordCount) env
    (RefreshStepBounds.assumptions program relation circuit
      (interface base program.witnessWordCount) (sharedEnd base program.witnessWordCount)
      (inputsBelow base program.witnessWordCount) env)
    rows input output proof represents

end NightstreamFPrime.Layout.OptionB.RefreshAssembler
