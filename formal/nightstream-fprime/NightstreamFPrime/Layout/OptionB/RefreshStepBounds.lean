import NightstreamFPrime.Layout.OptionB.RefreshPiCCSBounds
import NightstreamFPrime.Layout.OptionB.LightStepBounds
import NightstreamFPrime.Lifecycle.OptionB.RefreshStepLayout

/-! Input scoping of the refresh step layout (CW-3.6): when the step's shared words lie below the
parent offset, `RefreshStepLayout.Assumptions` holds (`assumptions`). Today's Π_DEC reads the
active slot, the children and the digit words; the refresh Π_CCS layout gets its own scoping
(`RefreshPiCCSBounds`); today's Π_RLC sampler starts from the Π_CCS output binding's final state,
and its sources read the fresh instance, the children, the digit words and the Π_CCS output. -/

namespace NightstreamFPrime.Layout.OptionB.RefreshStepBounds

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Lifecycle.OptionB.RefreshStepLayout

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The shared words that the step's gadgets read lie below the parent offset. -/
structure InputsBelow {witnessWordCount : Nat}
    (interface : RefreshStepLayout.Interface logicalWidth publicFits witnessWordCount)
    (offset : Nat) : Prop where
  iteration : interface.iteration.VarsBelow offset
  priorPc : interface.priorPc.VarsBelow offset
  verifierKeys : ∀ key lane, (interface.verifierKeys key lane).VarsBelow offset
  initialState : ∀ lane, (interface.initialState lane).VarsBelow offset
  currentState : ∀ lane, (interface.currentState lane).VarsBelow offset
  nextState : ∀ lane, (interface.nextState lane).VarsBelow offset
  witness : ∀ index, (interface.witness index).VarsBelow offset
  prior : ∀ slot, ∀ word ∈ (interface.prior slot).words, word.VarsBelow offset
  next : ∀ slot, ∀ word ∈ (interface.next slot).words, word.VarsBelow offset
  activePoint : ∀ coordinate, (interface.active.point coordinate).VarsBelow offset
  activePad : ∀ coefficient, (interface.active.evaluation.eval_K coefficient).VarsBelow offset
  activeMatrix : ∀ matrix coefficient,
    (interface.active.evaluation.eval_A matrix coefficient).VarsBelow offset
  activeCommitment : ∀ row lane, (interface.active.commitment row lane).VarsBelow offset
  activePublicInput : ∀ column, (interface.active.publicInput column).VarsBelow offset
  priorDigest : ∀ slot lane, (interface.priorDigest slot lane).VarsBelow offset
  nextDigest : ∀ slot lane, (interface.nextDigest slot lane).VarsBelow offset
  outputDigest : ∀ lane, (interface.outputDigest lane).VarsBelow offset
  freshCommitment : ∀ source row lane,
    (interface.fresh.commitment source row lane).VarsBelow offset
  freshPublicInput : ∀ source column, (interface.fresh.publicInput source column).VarsBelow offset
  childCommitment : ∀ child row lane, (interface.childCommitment child row lane).VarsBelow offset
  childPad : ∀ child coefficient,
    ((interface.childEvaluation child).eval_K coefficient).VarsBelow offset
  childMatrix : ∀ child matrix coefficient,
    ((interface.childEvaluation child).eval_A matrix coefficient).VarsBelow offset
  digit : ∀ child column, (interface.digit child column).VarsBelow offset
  round : ∀ roundIndex coefficient,
    ((interface.round roundIndex).coefficient coefficient).VarsBelow offset
  outputPad : ∀ source coefficient,
    (interface.output.padCoordinate source coefficient).VarsBelow offset
  outputMatrix : ∀ source matrix coefficient,
    (interface.output.matrixCoordinate source matrix coefficient).VarsBelow offset

variable (program : Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (circuit : Fin 2)
  (interface : RefreshStepLayout.Interface logicalWidth publicFits program.witnessWordCount)

theorem statePreimage_below (iteration : Expr) (current : Fin 4 → Expr) (pc : Expr)
    (digests : Fin 2 → Fin 4 → Expr) (bound : Nat)
    (keys : ∀ key lane, (interface.verifierKeys key lane).VarsBelow bound)
    (initial : ∀ lane, (interface.initialState lane).VarsBelow bound)
    (iterationBelow : iteration.VarsBelow bound)
    (currentBelow : ∀ lane, (current lane).VarsBelow bound) (pcBelow : pc.VarsBelow bound)
    (digestBelow : ∀ slot lane, (digests slot lane).VarsBelow bound) :
    ∀ word ∈ statePreimageExpr program interface iteration current pc digests,
      word.VarsBelow bound := by
  intro word member
  simp only [statePreimageExpr, List.mem_append, List.mem_map, List.mem_ofFn,
    List.mem_singleton] at member
  rcases member with ((((((((⟨_, _, rfl⟩ | ⟨lane, rfl⟩) | ⟨lane, rfl⟩) | rfl) |
    ⟨lane, rfl⟩) | ⟨lane, rfl⟩) | rfl) | ⟨lane, rfl⟩) | ⟨lane, rfl⟩)
  · trivial
  · exact keys 0 lane
  · exact keys 1 lane
  · exact iterationBelow
  · exact initial lane
  · exact currentBelow lane
  · exact pcBelow
  · exact digestBelow 0 lane
  · exact digestBelow 1 lane

variable (offset : Nat) (below : InputsBelow interface offset) (env : Env)

include below

omit below in
/-- The gadgets before Π_DEC start at or after the parent offset. -/
theorem dec_le : offset ≤ decOffset program circuit interface offset := by
  have le := fun (child : FormalCircuit) (start : Nat) =>
    (Nat.le_add_right start _ : start ≤ Formal.nextOffset child start)
  have priorZeroLe : offset ≤ priorSlotZeroOffset program interface offset := le _ _
  have priorOneLe : offset ≤ priorSlotOneOffset program interface offset :=
    Nat.le_trans priorZeroLe (le _ _)
  have priorStateLe : offset ≤ priorStateOffset program interface offset :=
    Nat.le_trans priorOneLe (le _ _)
  have nextZeroLe : offset ≤ nextSlotZeroOffset program interface offset :=
    Nat.le_trans priorStateLe (le _ _)
  have nextOneLe : offset ≤ nextSlotOneOffset program interface offset :=
    Nat.le_trans nextZeroLe (le _ _)
  have nextStateLe : offset ≤ nextStateOffset program interface offset :=
    Nat.le_trans nextOneLe (le _ _)
  exact Nat.le_trans nextStateLe (le _ _)

omit below in
theorem piccs_le : offset ≤ piccsOffset program circuit interface offset := by
  have decLe := dec_le program circuit interface offset
  unfold piccsOffset Lifecycle.PiDEC.v1_2.Formal.finalOffset
    Lifecycle.PiDEC.v1_2.Formal.recompositionOffset
  omega

omit below in
/-- The Π_RLC offset is the end of the Π_CCS output binding. Proved by unfolding, so the kernel
never evaluates the output binding's length. -/
theorem rlcOffset_eq (offset : Nat) :
    rlcOffset program relation circuit interface offset =
      RefreshPiCCSLayout.outputOffset relation (piccsInterface program interface)
          (piccsOffset program circuit interface offset) +
        localLength (Circuit.ops (OutputBinding.circuit (RefreshPiCCSLayout.outputInterface
          (piccsInterface program interface) (piccsOffset program circuit interface offset))).main
          (RefreshPiCCSLayout.outputOffset relation (piccsInterface program interface)
            (piccsOffset program circuit interface offset))) := by
  unfold rlcOffset RefreshPiCCSLayout.finalOffset Formal.nextOffset Formal.childLength
    RefreshPiCCSLayout.outputCircuit
  rfl

omit below in
theorem piccs_le_rlc (offset : Nat) :
    piccsOffset program circuit interface offset ≤
      rlcOffset program relation circuit interface offset := by
  have outputLe : piccsOffset program circuit interface offset ≤
      RefreshPiCCSLayout.outputOffset relation (piccsInterface program interface)
        (piccsOffset program circuit interface offset) :=
    Nat.le_trans (RefreshPiCCSBounds.challenge_le _ _) (Nat.le_trans
      (RefreshPiCCSBounds.rounds_le _ _) (Nat.le_trans (RefreshPiCCSBounds.initial_le _ _)
        (Nat.le_trans (RefreshPiCCSBounds.sumcheck_le _ _) (Nat.le_trans
          (RefreshPiCCSBounds.evalK_le _ _) (Nat.le_trans (RefreshPiCCSBounds.evalA_le _ _)
            (Nat.le_trans (RefreshPiCCSBounds.ccs_le _ _) (Nat.le_trans
              (RefreshPiCCSBounds.norm_le relation _ _) (Nat.le_trans
                (RefreshPiCCSBounds.finalIdentity_le relation _ _)
                (RefreshPiCCSBounds.output_le relation _ _)))))))))
  exact Nat.le_trans outputLe (Nat.le_add_right _ _)

omit below in
theorem rlc_le : offset ≤ rlcOffset program relation circuit interface offset :=
  Nat.le_trans (piccs_le program circuit interface offset)
    (piccs_le_rlc program relation circuit interface offset)

/-- The refresh Π_CCS layout's inputs lie below its offset. -/
theorem piccsBelow : RefreshPiCCSBounds.InputsBelow (piccsInterface program interface)
    (piccsOffset program circuit interface offset) := by
  have piccsLe := piccs_le program circuit interface offset
  exact
    ⟨fun coordinate => KExpr.varsBelow_mono _ (below.activePoint coordinate) piccsLe,
      fun child row lane => Expr.VarsBelow.mono _ (below.childCommitment child row lane) piccsLe,
      fun child coefficient => KExpr.varsBelow_mono _ (below.childPad child coefficient) piccsLe,
      fun child matrix coefficient =>
        KExpr.varsBelow_mono _ (below.childMatrix child matrix coefficient) piccsLe,
      fun source row lane => Expr.VarsBelow.mono _ (below.freshCommitment source row lane) piccsLe,
      fun source column => Expr.VarsBelow.mono _ (below.freshPublicInput source column) piccsLe,
      fun roundIndex coefficient => KExpr.varsBelow_mono _ (below.round roundIndex coefficient)
        piccsLe,
      fun source coefficient => KExpr.varsBelow_mono _ (below.outputPad source coefficient) piccsLe,
      fun source matrix coefficient =>
        KExpr.varsBelow_mono _ (below.outputMatrix source matrix coefficient) piccsLe⟩

/-- Today's Π_DEC reads only shared words, which lie below its offset. -/
theorem decAssumptions :
    Lifecycle.PiDEC.v1_2.Formal.Assumptions relation (decInterface program interface)
      (decOffset program circuit interface offset) env := by
  have decLe := dec_le program circuit interface offset
  exact ⟨{
    point := fun coordinate => KExpr.varsBelow_mono _ (below.activePoint coordinate) decLe
    parentCommitment := fun row lane =>
      Expr.VarsBelow.mono _ (below.activeCommitment row lane) decLe
    parentPublicInput := fun column =>
      Expr.VarsBelow.mono _ (below.activePublicInput column) decLe
    parentEval_K := fun coefficient =>
      KExpr.varsBelow_mono _ (below.activePad coefficient) decLe
    parentEval_A := fun matrix coefficient =>
      KExpr.varsBelow_mono _ (below.activeMatrix matrix coefficient) decLe
    messageCommitment := fun child row lane =>
      Expr.VarsBelow.mono _ (below.childCommitment (childOf child) row lane) decLe
    messageEval_K := fun child coefficient =>
      KExpr.varsBelow_mono _ (below.childPad (childOf child) coefficient) decLe
    messageEval_A := fun child matrix coefficient =>
      KExpr.varsBelow_mono _ (below.childMatrix (childOf child) matrix coefficient) decLe
    digit := fun child column =>
      Expr.VarsBelow.mono _ (below.digit (childOf child) column) decLe }⟩

theorem sourceCommitment_below (source : Fin productionShape.sourceCount)
    (row : Fin productionProfile.commitmentWidth) (lane : Fin ringDegree) :
    (((rlcInterface program relation circuit interface offset).input
      (rlcOffset program relation circuit interface offset) source).commitment row lane).VarsBelow
        (rlcOffset program relation circuit interface offset) := by
  have rlcLe := rlc_le program relation circuit interface offset
  refine Fin.addCases (fun index => ?_) (fun child => ?_) source
  · simp only [rlcInterface, Fin.addCases_left]
    exact Expr.VarsBelow.mono _ (below.freshCommitment index row lane) rlcLe
  · simp only [rlcInterface, Fin.addCases_right]
    exact Expr.VarsBelow.mono _ (below.childCommitment child row lane) rlcLe

theorem sourcePublicInput_below (source : Fin productionShape.sourceCount)
    (column : Fin (PaperAlgebra.FullShape logicalWidth publicFits).publicWidth) :
    (((rlcInterface program relation circuit interface offset).input
      (rlcOffset program relation circuit interface offset) source).publicInput column).VarsBelow
        (rlcOffset program relation circuit interface offset) := by
  have rlcLe := rlc_le program relation circuit interface offset
  refine Fin.addCases (fun index => ?_) (fun child => ?_) source
  · simp only [rlcInterface, Fin.addCases_left]
    exact Expr.VarsBelow.mono _ (below.freshPublicInput index column) rlcLe
  · simp only [rlcInterface, Fin.addCases_right]
    exact Expr.VarsBelow.mono _ (below.digit child column) rlcLe

theorem sourcePad_below (source : Fin productionShape.sourceCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (((rlcInterface program relation circuit interface offset).input
      (rlcOffset program relation circuit interface offset) source).evaluation.eval_K
        coefficient).VarsBelow (rlcOffset program relation circuit interface offset) :=
  KExpr.varsBelow_mono _ (below.outputPad source coefficient)
    (rlc_le program relation circuit interface offset)

theorem sourceMatrix_below (source : Fin productionShape.sourceCount)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (((rlcInterface program relation circuit interface offset).input
      (rlcOffset program relation circuit interface offset) source).evaluation.eval_A matrix
        coefficient).VarsBelow (rlcOffset program relation circuit interface offset) :=
  KExpr.varsBelow_mono _ (below.outputMatrix source matrix coefficient)
    (rlc_le program relation circuit interface offset)

/-- The Π_RLC sampler starts from the Π_CCS output binding's final state, which lies below the
Π_RLC offset. -/
theorem samplerStart_below (lane : Fin Spec.Poseidon2.width) :
    ((rlcInterface program relation circuit interface offset).initialState
      (rlcOffset program relation circuit interface offset) lane).VarsBelow
        (rlcOffset program relation circuit interface offset) := by
  have bound := OutputBinding.finalState_varsBelow
    (RefreshPiCCSLayout.outputInterface (piccsInterface program interface)
      (piccsOffset program circuit interface offset))
    (RefreshPiCCSLayout.outputOffset relation (piccsInterface program interface)
      (piccsOffset program circuit interface offset)) (fun _ => 0)
    (RefreshPiCCSBounds.output relation (piccsInterface program interface)
      (piccsOffset program circuit interface offset)
      (piccsBelow program circuit interface offset below) (fun _ => 0)) lane
  rw [← rlcOffset_eq program relation circuit interface offset] at bound
  exact bound

omit below in
/-- Today's Π_RLC input scoping at any offset `start`: its sampler state and its sources lie below
`start`. The offset is a variable here, so the kernel never evaluates a layout length. -/
theorem rlcAssumptions_of (rlc : Lifecycle.PiRLC.v1_2.Formal.Interface logicalWidth publicFits)
    (start : Nat)
    (sampler : ∀ lane, (rlc.initialState start lane).VarsBelow start)
    (commitment : ∀ source row lane,
      ((rlc.input start source).commitment row lane).VarsBelow start)
    (publicInput : ∀ source column, ((rlc.input start source).publicInput column).VarsBelow start)
    (pad : ∀ source coefficient,
      ((rlc.input start source).evaluation.eval_K coefficient).VarsBelow start)
    (matrix : ∀ source matrix coefficient,
      ((rlc.input start source).evaluation.eval_A matrix coefficient).VarsBelow start) :
    Lifecycle.PiRLC.v1_2.Formal.Assumptions relation rlc start env := by
  have commitmentLe : start ≤ Lifecycle.PiRLC.v1_2.Formal.commitmentOffset start := by
    unfold Lifecycle.PiRLC.v1_2.Formal.commitmentOffset Lifecycle.PiRLC.v1_2.Formal.samplerOffset
    omega
  have publicInputLe : Lifecycle.PiRLC.v1_2.Formal.commitmentOffset start ≤
      Lifecycle.PiRLC.v1_2.Formal.publicInputOffset start := by
    unfold Lifecycle.PiRLC.v1_2.Formal.publicInputOffset
    omega
  have evalKLe : Lifecycle.PiRLC.v1_2.Formal.publicInputOffset start ≤
      Lifecycle.PiRLC.v1_2.Formal.evalKOffset start := by
    unfold Lifecycle.PiRLC.v1_2.Formal.evalKOffset
    omega
  have evalALe : Lifecycle.PiRLC.v1_2.Formal.evalKOffset start ≤
      Lifecycle.PiRLC.v1_2.Formal.evalAOffset start := by
    unfold Lifecycle.PiRLC.v1_2.Formal.evalAOffset
    omega
  have challengeBelow := fun source lane =>
    Lifecycle.PiRLC.v1_2.SamplerChain.outputChallenge_below start source lane
  refine {
    sampler := sampler
    commitment := ⟨fun source lane => challengeBelow source lane, fun source row lane _ =>
      Expr.VarsBelow.mono _ (commitment source row lane) commitmentLe⟩
    publicInput := ⟨fun source lane => Expr.VarsBelow.mono _ (challengeBelow source lane)
        publicInputLe, fun source block lane _ =>
      Expr.VarsBelow.mono _ (publicInput source
        (Lifecycle.PiRLC.v1_2.PublicInputCombination.publicColumn block lane))
        (Nat.le_trans commitmentLe publicInputLe)⟩
    eval_K := ⟨fun source lane => Expr.VarsBelow.mono _ (challengeBelow source lane)
        (Nat.le_trans publicInputLe evalKLe), fun source block lane cell => ?_⟩
    eval_A := ⟨fun source lane => Expr.VarsBelow.mono _ (challengeBelow source lane)
        (Nat.le_trans publicInputLe (Nat.le_trans evalKLe evalALe)),
      fun source matrixIndex lane cell => ?_⟩ }
  · have below := KExpr.varsBelow_mono _
      (pad source (Lifecycle.PiRLC.v1_2.EvalKCombination.coefficient lane))
      (Nat.le_trans commitmentLe (Nat.le_trans publicInputLe evalKLe))
    fin_cases cell
    · exact below.1
    · exact below.2
  · have below := KExpr.varsBelow_mono _
      (matrix source matrixIndex (Lifecycle.PiRLC.v1_2.EvalKCombination.coefficient lane))
      (Nat.le_trans commitmentLe (Nat.le_trans publicInputLe (Nat.le_trans evalKLe evalALe)))
    fin_cases cell
    · exact below.1
    · exact below.2

/-- Today's Π_RLC reads the Π_CCS output binding's final state and shared words, which lie below
its offset. -/
theorem rlcAssumptions :
    Lifecycle.PiRLC.v1_2.Formal.Assumptions relation
      (rlcInterface program relation circuit interface offset)
      (rlcOffset program relation circuit interface offset) env :=
  rlcAssumptions_of relation env (rlcInterface program relation circuit interface offset)
    (rlcOffset program relation circuit interface offset)
    (samplerStart_below program relation circuit interface offset below)
    (sourceCommitment_below program relation circuit interface offset below)
    (sourcePublicInput_below program relation circuit interface offset below)
    (sourcePad_below program relation circuit interface offset below)
    (sourceMatrix_below program relation circuit interface offset below)

/-- Input scoping of the refresh step layout. -/
theorem assumptions :
    RefreshStepLayout.Assumptions program relation circuit interface offset env := by
  have le := fun (child : FormalCircuit) (start : Nat) =>
    (Nat.le_add_right start _ : start ≤ Formal.nextOffset child start)
  have priorZeroLe : offset ≤ priorSlotZeroOffset program interface offset := le _ _
  have priorOneLe : offset ≤ priorSlotOneOffset program interface offset :=
    Nat.le_trans priorZeroLe (le _ _)
  have priorStateLe : offset ≤ priorStateOffset program interface offset :=
    Nat.le_trans priorOneLe (le _ _)
  have nextZeroLe : offset ≤ nextSlotZeroOffset program interface offset :=
    Nat.le_trans priorStateLe (le _ _)
  have nextOneLe : offset ≤ nextSlotOneOffset program interface offset :=
    Nat.le_trans nextZeroLe (le _ _)
  have nextStateLe : offset ≤ nextStateOffset program interface offset :=
    Nat.le_trans nextOneLe (le _ _)
  have keysAt := fun (bound : Nat) (le : offset ≤ bound) key lane =>
    Expr.VarsBelow.mono _ (below.verifierKeys key lane) le
  have initialAt := fun (bound : Nat) (le : offset ≤ bound) lane =>
    Expr.VarsBelow.mono _ (below.initialState lane) le
  exact {
    application := Stage1.Application.Program.assumptions program
      (applicationInterface program interface) offset env
      ⟨below.currentState, below.witness, below.nextState⟩
    priorSlotZero := ⟨LightStepBounds.slotPreimage_below _ _ fun word member =>
        Expr.VarsBelow.mono _ (below.prior 0 word member) priorZeroLe,
      fun lane => Expr.VarsBelow.mono _ (below.priorDigest 0 lane) priorZeroLe⟩
    priorSlotOne := ⟨LightStepBounds.slotPreimage_below _ _ fun word member =>
        Expr.VarsBelow.mono _ (below.prior 1 word member) priorOneLe,
      fun lane => Expr.VarsBelow.mono _ (below.priorDigest 1 lane) priorOneLe⟩
    priorState := ⟨statePreimage_below program interface _ _ _ _ _
        (keysAt _ priorStateLe) (initialAt _ priorStateLe)
        (Expr.VarsBelow.mono _ below.iteration priorStateLe)
        (fun lane => Expr.VarsBelow.mono _ (below.currentState lane) priorStateLe)
        (Expr.VarsBelow.mono _ below.priorPc priorStateLe)
        (fun slot lane => Expr.VarsBelow.mono _ (below.priorDigest slot lane) priorStateLe),
      fun column => Expr.VarsBelow.mono _ (below.freshPublicInput _ column) priorStateLe⟩
    nextSlotZero := ⟨LightStepBounds.slotPreimage_below _ _ fun word member =>
        Expr.VarsBelow.mono _ (below.next 0 word member) nextZeroLe,
      fun lane => Expr.VarsBelow.mono _ (below.nextDigest 0 lane) nextZeroLe⟩
    nextSlotOne := ⟨LightStepBounds.slotPreimage_below _ _ fun word member =>
        Expr.VarsBelow.mono _ (below.next 1 word member) nextOneLe,
      fun lane => Expr.VarsBelow.mono _ (below.nextDigest 1 lane) nextOneLe⟩
    nextState := ⟨statePreimage_below program interface _ _ _ _ _
        (keysAt _ nextStateLe) (initialAt _ nextStateLe)
        (Expr.VarsBelow.add _ _ _ (Expr.VarsBelow.mono _ below.iteration nextStateLe) trivial)
        (fun lane => Expr.VarsBelow.mono _ (below.nextState lane) nextStateLe) trivial
        (fun slot lane => Expr.VarsBelow.mono _ (below.nextDigest slot lane) nextStateLe),
      fun lane => Expr.VarsBelow.mono _ (below.outputDigest lane) nextStateLe⟩
    dec := decAssumptions program relation circuit interface offset below env
    piccs := RefreshPiCCSBounds.assumptions relation (piccsInterface program interface)
      (piccsOffset program circuit interface offset)
      (piccsBelow program circuit interface offset below) env
    rlc := rlcAssumptions program relation circuit interface offset below env }

end NightstreamFPrime.Layout.OptionB.RefreshStepBounds
