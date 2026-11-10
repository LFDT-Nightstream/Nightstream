import NightstreamFPrime.Lifecycle.BoundedDepth.LightStepSoundness
import NightstreamFPrime.Layout.BoundedDepth.LightStepBounds

/-! The light step's source layout (assembler A1). Every shared word of
`LightStepLayout.Interface` is one variable in a source region that starts at `base`: the counters
and selector (5), the key digests (8), the start, current and next states (12), the application
witness, seven slots (7 × 2,055, each in the order of `SlotExpr.words`: the prior, active, inner
and next slots), the slot and output digests (20), the fresh instance (1,458), the round messages
(504), the Π_CCS output of both sources (1,080), and the new, inner and active digests (16). The
light rows read the active slot's words only; the prior, inner and next slots keep the shared-word
prefix equal to the refresh step's, whose finish circuit reads them. The step's gadgets start at
`sharedEnd`. -/

namespace NightstreamFPrime.Layout.BoundedDepth.LightAssembler

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.BoundedDepth.SlotWords
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Gadgets.Poseidon2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Two adjacent columns as a `K` value. -/
def kAt (column : Nat) : KExpr := ⟨.var column, .var (column + 1)⟩

/-- One slot's 2,055 words from column `start`, in the order of `SlotExpr.words`: the commitment
(1,188), `Eval_K` (108), `Eval_A` (432), the point (56), the public input (270) and the depth. -/
def slotAt (start : Nat) : SlotExpr logicalWidth publicFits where
  commitment := fun row lane => .var (start + row.val * ringDegree + lane.val)
  evaluation :=
    ⟨fun coefficient => kAt (start + 1188 + 2 * coefficient.val),
      fun matrix coefficient =>
        kAt (start + 1296 + 2 * (matrix.val * ringDegree + coefficient.val))⟩
  point := fun coordinate => kAt (start + 1728 + 2 * coordinate.val)
  publicInput := fun column => .var (start + 1784 + column.val)
  depth := .var (start + 2054)

def slotWordCount : Nat := 2055

/-- Where each group of shared words starts. -/
def keysStart (base : Nat) : Nat := base + 5
def statesStart (base : Nat) : Nat := base + 13
def witnessStart (base : Nat) : Nat := base + 25
def slotsStart (base witnessWordCount : Nat) : Nat := witnessStart base + witnessWordCount
def digestsStart (base witnessWordCount : Nat) : Nat :=
  slotsStart base witnessWordCount + 7 * slotWordCount
def freshStart (base witnessWordCount : Nat) : Nat := digestsStart base witnessWordCount + 20
def roundsStart (base witnessWordCount : Nat) : Nat := freshStart base witnessWordCount + 1458
def outputStart (base witnessWordCount : Nat) : Nat := roundsStart base witnessWordCount + 504
/-- The digest select's 16 words: the new slot's digest, each slot's inner digest, then the
active slot's digest. -/
def digestSelectStart (base witnessWordCount : Nat) : Nat :=
  outputStart base witnessWordCount + 1080

/-- The first column after the shared words, where the step's gadgets start. -/
def sharedEnd (base witnessWordCount : Nat) : Nat := digestSelectStart base witnessWordCount + 16

theorem sharedEnd_eq (base witnessWordCount : Nat) :
    sharedEnd base witnessWordCount = base + witnessWordCount + 17488 := by
  simp only [sharedEnd, digestSelectStart, outputStart, roundsStart, freshStart, digestsStart,
    slotsStart, witnessStart, slotWordCount]
  omega

/-- The step's shared words as source columns from `base`. -/
def interface (base witnessWordCount : Nat) :
    LightStepLayout.Interface logicalWidth publicFits witnessWordCount where
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
  priorDigest := fun slot lane =>
    .var (digestsStart base witnessWordCount + 4 * slot.val + lane.val)
  nextDigest := fun slot lane =>
    .var (digestsStart base witnessWordCount + 8 + 4 * slot.val + lane.val)
  outputDigest := fun lane => .var (digestsStart base witnessWordCount + 16 + lane.val)
  newDigest := fun lane => .var (digestSelectStart base witnessWordCount + lane.val)
  innerDigest := fun slot lane =>
    .var (digestSelectStart base witnessWordCount + 4 + 4 * slot.val + lane.val)
  activeDigest := fun lane => .var (digestSelectStart base witnessWordCount + 12 + lane.val)
  fresh :=
    ⟨fun _ row lane => .var (freshStart base witnessWordCount + row.val * ringDegree + lane.val),
      fun _ column => .var (freshStart base witnessWordCount + 1188 + column.val)⟩
  round := fun roundIndex => ⟨fun coefficient =>
    kAt (roundsStart base witnessWordCount + 2 * (roundIndex.val * 9 + coefficient.val))⟩
  output := ⟨fun source coefficient =>
      kAt (outputStart base witnessWordCount + 2 * (source.val * 270 + coefficient.val)),
    fun source matrix coefficient =>
      kAt (outputStart base witnessWordCount +
        2 * (source.val * 270 + 54 + matrix.val * 54 + coefficient.val))⟩

/-- A slot placed at `start` uses the columns `start` to `start + 2,054`. -/
theorem slotAt_below (start bound : Nat) (le : start + slotWordCount ≤ bound) :
    ∀ word ∈ (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits) start).words,
      word.VarsBelow bound := by
  intro word member
  simp only [SlotExpr.words, slotAt, kAt, StatementAbsorption.serializeCommitmentExpr,
    StatementAbsorption.serializeEvalKExpr, StatementAbsorption.serializeEvalAExpr,
    StatementAbsorption.serializePointExpr, StatementAbsorption.serializePublicInputExpr,
    StatementAbsorption.serializeKExpr, List.mem_append, List.mem_flatMap, List.mem_map,
    List.mem_finRange, List.mem_cons, List.not_mem_nil, or_false,
    true_and] at member
  simp only [slotWordCount] at le
  rcases member with ((((commitment | pad | matrix) | point) | publicWords) | depth)
  · obtain ⟨row, lane, rfl⟩ := commitment
    have rowBound := row.isLt
    have laneBound := lane.isLt
    change row.val < 22 at rowBound
    change lane.val < 54 at laneBound
    show start + row.val * ringDegree + lane.val < bound
    change start + row.val * 54 + lane.val < bound
    omega
  · obtain ⟨coefficient, rfl | rfl⟩ := pad <;>
    · have bound' := coefficient.isLt
      change coefficient.val < 54 at bound'
      show _ < bound
      omega
  · obtain ⟨matrix, coefficient, rfl | rfl⟩ := matrix <;>
    · have matrixBound := matrix.isLt
      have coefficientBound := coefficient.isLt
      change matrix.val < 4 at matrixBound
      change coefficient.val < 54 at coefficientBound
      show _ < bound
      simp only [ringDegree]
      omega
  · obtain ⟨coordinate, rfl | rfl⟩ := point <;>
    · have coordinateBound := coordinate.isLt
      change coordinate.val < 28 at coordinateBound
      show _ < bound
      omega
  · obtain ⟨column, rfl⟩ := publicWords
    have columnBound := column.isLt
    change column.val < 270 at columnBound
    show start + 1784 + column.val < bound
    omega
  · subst depth
    show start + 2054 < bound
    omega

private theorem kAt_below (column bound : Nat) (below : column + 1 < bound) :
    (kAt column).VarsBelow bound :=
  ⟨show column < bound by omega, show column + 1 < bound from below⟩

/-- Every shared word of the source layout lies below `sharedEnd`. -/
theorem inputsBelow (base witnessWordCount : Nat) :
    LightStepBounds.InputsBelow
      (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base witnessWordCount)
      (sharedEnd base witnessWordCount) := by
  have endEq := sharedEnd_eq base witnessWordCount
  have slotsEq : slotsStart base witnessWordCount = base + 25 + witnessWordCount := rfl
  have activeStart : slotsStart base witnessWordCount + 2 * slotWordCount + slotWordCount ≤
      sharedEnd base witnessWordCount := by
    rw [endEq, slotsEq]
    simp only [slotWordCount]
    omega
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
      activeDepth := by
        have := activeWord 2054 (by simp only [slotWordCount]; omega)
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
      newDigest := fun lane => by
        have laneBound := lane.isLt
        show digestSelectStart base witnessWordCount + lane.val < _
        rw [endEq]
        simp only [digestSelectStart, outputStart, roundsStart, freshStart, digestsStart,
          slotsStart, witnessStart, slotWordCount]
        omega
      innerDigest := fun slot lane => by
        have slotBound := slot.isLt
        have laneBound := lane.isLt
        show digestSelectStart base witnessWordCount + 4 + 4 * slot.val + lane.val < _
        rw [endEq]
        simp only [digestSelectStart, outputStart, roundsStart, freshStart, digestsStart,
          slotsStart, witnessStart, slotWordCount]
        omega
      activeDigest := fun lane => by
        have laneBound := lane.isLt
        show digestSelectStart base witnessWordCount + 12 + lane.val < _
        rw [endEq]
        simp only [digestSelectStart, outputStart, roundsStart, freshStart, digestsStart,
          slotsStart, witnessStart, slotWordCount]
        omega
      freshCommitment := fun _ row lane => by
        have rowBound := row.isLt
        have laneBound := lane.isLt
        change row.val < 22 at rowBound
        change lane.val < 54 at laneBound
        show freshStart base witnessWordCount + row.val * ringDegree + lane.val < _
        rw [endEq]
        simp only [freshStart, digestsStart, slotsStart, witnessStart, slotWordCount, ringDegree]
        omega
      freshPublicInput := fun _ column => by
        have columnBound := column.isLt
        change column.val < 270 at columnBound
        show freshStart base witnessWordCount + 1188 + column.val < _
        rw [endEq]
        simp only [freshStart, digestsStart, slotsStart, witnessStart, slotWordCount]
        omega
      round := fun roundIndex coefficient => kAt_below _ _ (by
        have roundBound := roundIndex.isLt
        have coefficientBound := coefficient.isLt
        change roundIndex.val < 28 at roundBound
        rw [endEq]
        simp only [roundsStart, freshStart, digestsStart, slotsStart, witnessStart, slotWordCount]
        omega)
      outputPad := fun source coefficient => kAt_below _ _ (by
        have sourceBound := source.isLt
        have coefficientBound := coefficient.isLt
        change source.val < 2 at sourceBound
        change coefficient.val < 54 at coefficientBound
        rw [endEq]
        simp only [outputStart, roundsStart, freshStart, digestsStart, slotsStart, witnessStart,
          slotWordCount]
        omega)
      outputMatrix := fun source matrix coefficient => kAt_below _ _ (by
        have sourceBound := source.isLt
        have matrixBound := matrix.isLt
        have coefficientBound := coefficient.isLt
        change source.val < 2 at sourceBound
        change matrix.val < 4 at matrixBound
        change coefficient.val < 54 at coefficientBound
        rw [endEq]
        simp only [outputStart, roundsStart, freshStart, digestsStart, slotsStart, witnessStart,
          slotWordCount]
        omega) }

/-- Soundness of the light step circuit at the source layout: its gadgets placed at `sharedEnd`,
with rows that hold and shared words that carry the step's input, output and light proof, give the
light case of `stepHolds_iff` at the production cap. No input-scoping hypothesis remains. -/
theorem rows_implies_stepHolds (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (base : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (light : Spec.HyperNova.Construction2.BoundedDepth.circuitStep circuit = .light)
    (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness →
      control state witness = circuit)
    (env : Env)
    (rows : holds env (Circuit.ops (LightStepLayout.main program relation circuit
      (interface base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)))
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape 8)
    (represents : LightStepSoundness.StepRepresents program
      (interface base program.witnessWordCount) ajtai circuitKeys circuit input output proof env)
    (next : output.runningNext = DigestSelect.nextSlots input.iteration input.running
      (LightStepLayout.selectedSlot program
        (interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          program.witnessWordCount) env)
      ((LightStepLayout.newSlot program relation circuit (interface base program.witnessWordCount)
        (sharedEnd base program.witnessWordCount)).value env)) :
    StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control circuit input
      output :=
  LightStepSoundness.rows_implies_stepHolds program (interface base program.witnessWordCount)
    ajtai circuitKeys step control circuit light stepEq validControl relation
    (sharedEnd base program.witnessWordCount) env
    (LightStepBounds.assumptions program relation circuit
      (interface base program.witnessWordCount) (sharedEnd base program.witnessWordCount)
      (inputsBelow base program.witnessWordCount) env)
    rows input output proof represents next

/-- The rows of the light step circuit at the source layout give the output digest as the hash of
the next state's words, and after the base case the other slot's next digest word is its prior
digest word (`LightStepSoundness.rows_nextWords`). -/
theorem rows_nextWords (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (base : Nat)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness →
      control state witness = circuit)
    (env : Env)
    (rows : holds env (Circuit.ops (LightStepLayout.main program relation circuit
      (interface base program.witnessWordCount)) (sharedEnd base program.witnessWordCount))) :
    let shared := interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
      program.witnessWordCount
    digestValue shared.outputDigest env =
        Poseidon2.hash (Hash.evalList env (LightStepLayout.statePreimageExpr program shared
          (shared.iteration + 1) shared.nextState (Expr.const (natWord (circuit.val + 1)))
          shared.nextDigest)) ∧
      (shared.iteration.eval env ≠ 0 → ∀ slot,
        slot ≠ LightStepLayout.selectedSlot program shared env →
        digestValue (shared.nextDigest slot) env =
          digestValue (shared.priorDigest slot) env) :=
  LightStepSoundness.rows_nextWords program (interface base program.witnessWordCount) step
    control circuit stepEq validControl relation (sharedEnd base program.witnessWordCount) env
    (LightStepBounds.assumptions program relation circuit
      (interface base program.witnessWordCount) (sharedEnd base program.witnessWordCount)
      (inputsBelow base program.witnessWordCount) env)
    rows

end NightstreamFPrime.Layout.BoundedDepth.LightAssembler
