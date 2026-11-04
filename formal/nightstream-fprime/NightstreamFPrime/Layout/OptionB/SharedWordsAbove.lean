import NightstreamFPrime.Layout.OptionB.RefreshAssembler

/-! The shared words of both step circuits lie at or above `base` (`LightAssembler.interface`,
`RefreshAssembler.interface`), so what they carry (`StepRepresents`) does not depend on the columns
below `base`. The exported relation puts its public input there (`StepRelation.base`).
`withFront` replaces the columns below `base`. -/

namespace NightstreamFPrime.Layout.OptionB.SharedWordsAbove

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout.OptionB.LightAssembler (kAt slotAt slotWordCount keysStart
  statesStart witnessStart slotsStart digestsStart freshStart)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- `env` with `front` below `base`. -/
def withFront (front env : Env) (base : Nat) : Env := fun column =>
  if column < base then front column else env column

theorem withFront_below {front env : Env} {base column : Nat} (below : column < base) :
    withFront front env base column = front column := by
  simp [withFront, below]

theorem withFront_above {front env : Env} {base column : Nat} (above : base ≤ column) :
    withFront front env base column = env column := by
  simp [withFront, Nat.not_lt.mpr above]

/-- `target` carries `honest`'s shared words, the columns from `base` to `sharedEnd`. -/
def Shared (honest target : Env) (base sharedEnd : Nat) : Prop :=
  ∀ column, base ≤ column → column < sharedEnd → target column = honest column

/-- With `honest` below `base`, a target that carries `honest`'s shared words agrees with
`honest` below `sharedEnd`. -/
theorem withFront_agree {honest target : Env} {base sharedEnd : Nat}
    (shared : Shared honest target base sharedEnd) :
    ∀ column, column < sharedEnd → withFront honest target base column = honest column := by
  intro column below
  by_cases low : column < base
  · exact withFront_below low
  · exact (withFront_above (Nat.le_of_not_gt low)).trans
      (shared column (Nat.le_of_not_gt low) below)

/-- A slot placed at `start` reads only columns from `start`. -/
theorem slotAt_above (start : Nat) :
    ∀ word ∈ (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits) start).words,
      word.VarsSatisfy (start ≤ ·) := by
  intro word member
  simp only [SlotExpr.words, slotAt, kAt, StatementAbsorption.serializeCommitmentExpr,
    StatementAbsorption.serializeEvalKExpr, StatementAbsorption.serializeEvalAExpr,
    StatementAbsorption.serializePointExpr, StatementAbsorption.serializePublicInputExpr,
    StatementAbsorption.serializeKExpr, List.mem_append, List.mem_flatMap, List.mem_map,
    List.mem_finRange, List.mem_cons, List.not_mem_nil, or_false,
    true_and] at member
  rcases member with ((((commitment | pad | matrix) | point) | publicWords) | depth)
  · obtain ⟨row, lane, rfl⟩ := commitment
    show start ≤ start + row.val * ringDegree + lane.val
    omega
  · obtain ⟨coefficient, rfl | rfl⟩ := pad
    · show start ≤ start + 1188 + 2 * coefficient.val
      omega
    · show start ≤ start + 1188 + 2 * coefficient.val + 1
      omega
  · obtain ⟨matrix, coefficient, rfl | rfl⟩ := matrix
    · show start ≤ start + 1296 + 2 * (matrix.val * ringDegree + coefficient.val)
      omega
    · show start ≤ start + 1296 + 2 * (matrix.val * ringDegree + coefficient.val) + 1
      omega
  · obtain ⟨coordinate, rfl | rfl⟩ := point
    · show start ≤ start + 1728 + 2 * coordinate.val
      omega
    · show start ≤ start + 1728 + 2 * coordinate.val + 1
      omega
  · obtain ⟨column, rfl⟩ := publicWords
    show start ≤ start + 1784 + column.val
    omega
  · subst depth
    show start ≤ start + 2054
    omega

private theorem starts (base witnessWordCount : Nat) :
    base ≤ keysStart base ∧ base ≤ statesStart base ∧ base ≤ witnessStart base ∧
      base ≤ slotsStart base witnessWordCount ∧ base ≤ digestsStart base witnessWordCount ∧
      base ≤ freshStart base witnessWordCount := by
  simp only [keysStart, statesStart, witnessStart, slotsStart, digestsStart, freshStart]
  omega

section Above

variable {base : Nat} {left right : Env}
  (agree : ∀ column, base ≤ column → left column = right column)
include agree

theorem kAt_eq {column : Nat} (above : base ≤ column) :
    (kAt column).eval left = (kAt column).eval right := by
  simp only [KExpr.eval, kAt, Expr.eval_var, agree column above, agree (column + 1) (by omega)]

theorem words_eq {words : List Expr} (support : ∀ word ∈ words, word.VarsSatisfy (base ≤ ·)) :
    Hash.evalList left words = Hash.evalList right words := by
  unfold Hash.evalList
  exact List.map_congr_left fun word member =>
    word.eval_eq_of_agree_satisfy _ left right (support word member) agree

theorem slotAt_words_eq {start : Nat} (above : base ≤ start) :
    Hash.evalList left (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits)
        start).words =
      Hash.evalList right (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits)
        start).words :=
  words_eq agree fun word member =>
    (slotAt_above start word member).mono _ fun _ low => le_trans above low

theorem slotAt_value_eq {start : Nat} (above : base ≤ start) :
    (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits) start).value right =
      (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits) start).value left :=
  SlotExpr.value_eq_of_words _ right _ ((slotAt start).depth.eval left).isLt
    ((slotAt_words_eq agree above).symm.trans (SlotExpr.words_eval _ left))

/-- The light step that the shared words carry does not depend on the columns below `base`. -/
theorem light_stepRepresents (program : Stage1.Application.Program)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape 8)
    (represents : LightStepSoundness.StepRepresents program
      (LightAssembler.interface base program.witnessWordCount) ajtai circuitKeys circuit input
      output proof left) :
    LightStepSoundness.StepRepresents program
      (LightAssembler.interface base program.witnessWordCount) ajtai circuitKeys circuit input
      output proof right := by
  obtain ⟨keys, states, witnessAbove, slots, digests, fresh⟩ := starts base program.witnessWordCount
  have var : ∀ column, base ≤ column → (Expr.var column).eval right = (Expr.var column).eval left :=
    fun column above => (agree column above).symm
  have slotWords : ∀ start, base ≤ start →
      Hash.evalList right (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits)
        start).words = Hash.evalList left (slotAt start).words :=
    fun start above => (slotAt_words_eq agree above).symm
  have freshEq : StatementAbsorption.evalFresh
      (LightAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount).fresh right =
      StatementAbsorption.evalFresh (LightAssembler.interface base program.witnessWordCount).fresh
        left := by
    unfold StatementAbsorption.evalFresh
    congr 1
    · funext origin row lane
      exact var _ (by omega)
    · funext origin column
      exact var _ (by omega)
  have rounds : base ≤ LightAssembler.roundsStart base program.witnessWordCount ∧
      base ≤ LightAssembler.outputStart base program.witnessWordCount := by
    simp only [LightAssembler.outputStart, LightAssembler.roundsStart]
    omega
  obtain ⟨roundsAbove, outputAbove⟩ := rounds
  have roundEq : ∀ roundIndex,
      ((LightAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount).round roundIndex).semanticPolynomial right =
      ((LightAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount).round roundIndex).semanticPolynomial left := by
    intro roundIndex
    unfold RoundTranscript.Message.semanticPolynomial Gadgets.SumCheck.FixedChain.Round.semanticPolynomial
      Gadgets.SumCheck.FixedChain.Round.coefficients
    congr 1
    apply List.map_congr_left
    intro value member
    obtain ⟨coefficient, rfl⟩ := List.mem_ofFn.mp member
    exact (kAt_eq agree (by omega)).symm
  exact {
    iteration := (var base (le_refl _)).trans represents.iteration
    iterationBound := represents.iterationBound
    priorPc := (var (base + 3) (by omega)).trans represents.priorPc
    priorPcBound := represents.priorPcBound
    verifierKeys := fun key =>
      (congrArg List.ofFn (funext fun lane => var _ (by omega))).trans (represents.verifierKeys key)
    initialState :=
      (congrArg List.ofFn (funext fun lane => var _ (by omega))).trans represents.initialState
    currentState :=
      (congrArg List.ofFn (funext fun lane => var _ (by omega))).trans represents.currentState
    nextState :=
      (congrArg List.ofFn (funext fun lane => var _ (by omega))).trans represents.nextState
    witness :=
      (congrArg List.ofFn (funext fun index => var _ (by omega))).trans represents.witness
    priorSlots := fun slot =>
      (slotWords _ (by simp only [slotWordCount]; omega)).trans (represents.priorSlots slot)
    priorDepth := represents.priorDepth
    nextSlots := fun slot =>
      (slotWords _ (by simp only [slotWordCount]; omega)).trans (represents.nextSlots slot)
    nextDepth := represents.nextDepth
    pcNext := represents.pcNext
    outputDigest := represents.outputDigest.trans
      (congrArg List.ofFn (funext fun lane => (var _ (by omega)).symm))
    fresh := freshEq.trans represents.fresh
    rounds := fun roundIndex => (roundEq roundIndex).trans (represents.rounds roundIndex)
    outputPad := fun source coefficient =>
      (kAt_eq agree (by omega)).symm.trans (represents.outputPad source coefficient)
    outputMatrix := fun source matrix coefficient =>
      (kAt_eq agree (by omega)).symm.trans (represents.outputMatrix source matrix coefficient)
    proof := represents.proof }

/-- The refresh step that the shared words carry does not depend on the columns below `base`. -/
theorem refresh_stepRepresents (program : Stage1.Application.Program)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (represents : RefreshStepSoundness.StepRepresents program
      (RefreshAssembler.interface base program.witnessWordCount) ajtai circuitKeys circuit input
      output proof left) :
    RefreshStepSoundness.StepRepresents program
      (RefreshAssembler.interface base program.witnessWordCount) ajtai circuitKeys circuit input
      output proof right := by
  obtain ⟨keys, states, witnessAbove, slots, digests, fresh⟩ := starts base program.witnessWordCount
  have var : ∀ column, base ≤ column → (Expr.var column).eval right = (Expr.var column).eval left :=
    fun column above => (agree column above).symm
  have slotWords : ∀ start, base ≤ start →
      Hash.evalList right (slotAt (logicalWidth := logicalWidth) (publicFits := publicFits)
        start).words = Hash.evalList left (slotAt start).words :=
    fun start above => (slotAt_words_eq agree above).symm
  have children : base ≤ RefreshAssembler.childCommitmentsStart base program.witnessWordCount ∧
      base ≤ RefreshAssembler.childEvaluationsStart base program.witnessWordCount ∧
      base ≤ RefreshAssembler.roundsStart base program.witnessWordCount ∧
      base ≤ RefreshAssembler.outputStart base program.witnessWordCount := by
    simp only [RefreshAssembler.childCommitmentsStart, RefreshAssembler.childEvaluationsStart,
      RefreshAssembler.roundsStart, RefreshAssembler.outputStart, RefreshAssembler.digitsStart]
    omega
  obtain ⟨commitmentsAbove, evaluationsAbove, roundsAbove, outputAbove⟩ := children
  have freshEq : StatementAbsorption.evalFresh
      (RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount).fresh right =
      StatementAbsorption.evalFresh (RefreshAssembler.interface base program.witnessWordCount).fresh
        left := by
    unfold StatementAbsorption.evalFresh
    congr 1
    · funext origin row lane
      exact var _ (by omega)
    · funext origin column
      exact var _ (by omega)
  have evaluationEq : ∀ child, StatementAbsorption.evalEvaluation
      ((RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount).childEvaluation child) right =
      StatementAbsorption.evalEvaluation
        ((RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
          program.witnessWordCount).childEvaluation child) left := by
    intro child
    unfold StatementAbsorption.evalEvaluation
    congr 1
    · funext coefficient
      exact (kAt_eq agree (by omega)).symm
    · funext matrix coefficient
      exact (kAt_eq agree (by omega)).symm
  have roundEq : ∀ roundIndex,
      ((RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount).round roundIndex).semanticPolynomial right =
      ((RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount).round roundIndex).semanticPolynomial left := by
    intro roundIndex
    unfold RoundTranscript.Message.semanticPolynomial Gadgets.SumCheck.FixedChain.Round.semanticPolynomial
      Gadgets.SumCheck.FixedChain.Round.coefficients
    congr 1
    apply List.map_congr_left
    intro value member
    obtain ⟨coefficient, rfl⟩ := List.mem_ofFn.mp member
    exact (kAt_eq agree (by omega)).symm
  exact {
    iteration := (var base (le_refl _)).trans represents.iteration
    iterationBound := represents.iterationBound
    priorPc := (var (base + 3) (by omega)).trans represents.priorPc
    priorPcBound := represents.priorPcBound
    verifierKeys := fun key =>
      (congrArg List.ofFn (funext fun lane => var _ (by omega))).trans (represents.verifierKeys key)
    initialState :=
      (congrArg List.ofFn (funext fun lane => var _ (by omega))).trans represents.initialState
    currentState :=
      (congrArg List.ofFn (funext fun lane => var _ (by omega))).trans represents.currentState
    nextState :=
      (congrArg List.ofFn (funext fun lane => var _ (by omega))).trans represents.nextState
    witness :=
      (congrArg List.ofFn (funext fun index => var _ (by omega))).trans represents.witness
    priorSlots := fun slot =>
      (slotWords _ (by simp only [slotWordCount]; omega)).trans (represents.priorSlots slot)
    priorDepth := represents.priorDepth
    nextSlots := fun slot =>
      (slotWords _ (by simp only [slotWordCount]; omega)).trans (represents.nextSlots slot)
    nextDepth := represents.nextDepth
    pcNext := represents.pcNext
    outputDigest := represents.outputDigest.trans
      (congrArg List.ofFn (funext fun lane => (var _ (by omega)).symm))
    fresh := freshEq.trans represents.fresh
    childCommitments := fun child row lane =>
      (var _ (by omega)).trans (represents.childCommitments child row lane)
    childEvaluations := fun child => (evaluationEq child).trans (represents.childEvaluations child)
    rounds := fun roundIndex => (roundEq roundIndex).trans (represents.rounds roundIndex)
    outputPad := fun source coefficient =>
      (kAt_eq agree (by omega)).symm.trans (represents.outputPad source coefficient)
    outputMatrix := fun source matrix coefficient =>
      (kAt_eq agree (by omega)).symm.trans (represents.outputMatrix source matrix coefficient)
    proof := represents.proof }

end Above

end NightstreamFPrime.Layout.OptionB.SharedWordsAbove
