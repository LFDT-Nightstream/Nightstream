import NightstreamFPrime.Export.OptionB.StepCompleteness
import NightstreamFPrime.Export.OptionB.TerminalBridge
import NightstreamFPrime.Lifecycle.OptionB.FoldCompleteness
import NightstreamFPrime.Lifecycle.OptionB.BaseFold
import NightstreamFPrime.Lifecycle.PilotZeroRunning
import NightstreamFPrime.Layout.OptionB.LightAssemblerCompleteness
import NightstreamFPrime.Layout.OptionB.RefreshAssemblerCompleteness

/-! One honest option-B IVC step (CW-10.4), the option-B form of HyperNova Construction 2's
completeness. Take a proof that the terminal accepts at iteration `i`, whose slot witnesses stay
within the accumulator bound of their depth (`Honest`, CW-10.5), and a valid application step.
The honest prover folds the fresh instance into the slot that the program counter selects
(`FoldCompleteness`), runs the step circuit (`LightAssemblerCompleteness.complete`), and
commits to the exported relation's honest assignment (`StepCompleteness.light_complete`). The
terminal accepts the new proof at iteration `i + 1`, and the new slots stay within their bounds
(`light_extend`). Each slot witness and the fresh witness are zero on every ring block from the
logical width of their circuit's package (`Complete.packageWidth`, `BlockSupport.Supported`): the
fresh witness is zero past that width (`StepCompleteness.light_complete`), and the folds keep
zero blocks (`FoldCompleteness`).

The keys must carry the exported relations at the packages' widened slot layouts, and the
packages must pass the emitter's completeness checks (`Complete`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.StepExtend

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.OptionB.BlockSupport
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.OptionB.StepRelation
open NightstreamFPrime.Export.OptionB.TerminalBridge
open NightstreamFPrime.Export.OptionB.SlotCompleteness
open NightstreamFPrime.Export.OptionB.StepCompleteness

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)}
  {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
  {step : AppState → AppWitness → AppState}
  {cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}

/-- The setup's keys carry the exported relations at the step packages' slot layouts widened to
the key width, and both packages pass the emitter's completeness checks. -/
structure Complete (exports : Exports relation ajtai circuitKeys step cubeFits) where
  lightBatches : List WitnessBatch
  lightWide : ProductionAssignment.logicalWidth (SlotLayout.runSlots exports.lightWidth
    (StepPackage.slotRuns publicColumns (lightPieces exports.program relation ajtai)
      exports.lightFresh lightBatches)) ≤ logicalWidth
  lightLayout : exports.lightLayout = WideLayout.widen (SlotLayout.layout _) lightWide
  lightPacked : packable publicColumns (lightPieces exports.program relation ajtai) = true
  lightBelow : (lightPieces exports.program relation ajtai).all (pieceBelow exports.lightFresh) =
    true
  lightPublicBelow : publicColumns ≤ exports.lightFresh
  lightFree : publicFree (flatConstraints (Circuit.ops
    (LightAssemblerCircuit.stepCircuit exports.program base relation 0 ajtai).main
    (LightAssembler.sharedEnd base exports.program.witnessWordCount))) = true
  refreshBatches : List WitnessBatch
  refreshWide : ProductionAssignment.logicalWidth (SlotLayout.runSlots exports.refreshWidth
    (StepPackage.slotRuns publicColumns (refreshPieces exports.program relation ajtai)
      exports.refreshFresh refreshBatches)) ≤ logicalWidth
  refreshLayout : exports.refreshLayout = WideLayout.widen (SlotLayout.layout _) refreshWide
  refreshPacked : packable publicColumns (refreshPieces exports.program relation ajtai) = true
  refreshBelow : (refreshPieces exports.program relation ajtai).all
    (pieceBelow exports.refreshFresh) = true
  refreshPublicBelow : publicColumns ≤ exports.refreshFresh
  refreshFree : publicFree (flatConstraints (Circuit.ops
    (RefreshAssemblerCircuit.stepCircuit exports.program base relation 1 ajtai).main
    (RefreshAssembler.sharedEnd base exports.program.witnessWordCount))) = true

/-- The logical width of circuit `circuit`'s package: the columns of its slot layout before the
key widens them, as the emitter prints it (`EmitMain.emitPackage`). Slot `circuit` accumulates
the circuit's fresh instances (SPEC.md section 2.5). -/
def Complete.packageWidth {exports : Exports relation ajtai circuitKeys step cubeFits}
    (complete : Complete exports) (circuit : Fin 2) : Nat :=
  if circuit.val = 0 then
    ProductionAssignment.logicalWidth (SlotLayout.runSlots exports.lightWidth
      (StepPackage.slotRuns publicColumns (lightPieces exports.program relation ajtai)
        exports.lightFresh complete.lightBatches))
  else
    ProductionAssignment.logicalWidth (SlotLayout.runSlots exports.refreshWidth
      (StepPackage.slotRuns publicColumns (refreshPieces exports.program relation ajtai)
        exports.refreshFresh complete.refreshBatches))

theorem Complete.packageWidth_zero {exports : Exports relation ajtai circuitKeys step cubeFits}
    (complete : Complete exports) :
    complete.packageWidth 0 = ProductionAssignment.logicalWidth (SlotLayout.runSlots
      exports.lightWidth (StepPackage.slotRuns publicColumns
        (lightPieces exports.program relation ajtai) exports.lightFresh complete.lightBatches)) :=
  rfl

theorem Complete.packageWidth_one {exports : Exports relation ajtai circuitKeys step cubeFits}
    (complete : Complete exports) :
    complete.packageWidth 1 = ProductionAssignment.logicalWidth (SlotLayout.runSlots
      exports.refreshWidth (StepPackage.slotRuns publicColumns
        (refreshPieces exports.program relation ajtai) exports.refreshFresh
        complete.refreshBatches)) :=
  rfl

/-- An honest proof's slots and fresh instance (CW-10.5, CW-10.6). Every slot's depth is at most
the light cap; its witness lies within the accumulator bound of its depth and is zero on every
ring block from its circuit's package width. The fresh witness is zero on every ring block from
the package width of the circuit that ran last. -/
structure Honest (packageWidths : Fin 2 → Nat)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2) : Prop where
  depth : ∀ slot, (payload.running slot).2 ≤ Schedule.productionLightFoldCap
  norm : ∀ slot, (PaperAlgebra.semantics ajtai).normBounded
    (productionGlobalParams.accumulatorBound (payload.running slot).2 + 1)
    (payload.runningWitness slot)
  support : ∀ slot,
    Supported (logicalWidth := logicalWidth) (packageWidths slot) (payload.runningWitness slot)
  freshSupport : ∀ pcValid : InRange 2 payload.pc,
    Supported (logicalWidth := logicalWidth) (packageWidths (selectedIndex pcValid))
      payload.freshWitness

theorem accumulatorBound_succ (depth : Nat) :
    productionGlobalParams.accumulatorBound depth + 1 +
        productionGlobalParams.expansionT * (productionGlobalParams.b - 1) =
      productionGlobalParams.accumulatorBound (depth + 1) + 1 := by
  simp only [GlobalParams.accumulatorBound]
  ring

theorem bound_le_bigB {depth : Nat} (atMost : depth ≤ Schedule.productionLightFoldCap) :
    productionGlobalParams.accumulatorBound depth + 1 ≤ productionGlobalParams.bigB :=
  GlobalParams.production_accumulatorBound_lt_bigB (by
    simp only [Schedule.productionLightFoldCap] at atMost
    omega)

theorem normBounded_mono {witness : PaperAlgebra.Assignment (logicalWidth := logicalWidth)
      (publicFits := publicFits)} {small large : Nat}
    (bounded : (PaperAlgebra.semantics ajtai).normBounded small witness) (le : small ≤ large) :
    (PaperAlgebra.semantics ajtai).normBounded large witness :=
  fun column => lt_of_lt_of_le (bounded column) le

/-- One honest light step: from a recursive proof that the terminal accepts and whose slots are
honest, a valid application step whose circuit is the light circuit gives a recursive proof that
the terminal accepts at the next iteration, with honest slots. -/
theorem light_extend (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (holds : Terminal.HoldsFor cap ajtai circuitKeys step control statement (.recursive payload))
    (honest : Honest (ajtai := ajtai) complete.packageWidth payload)
    (witness : AppWitness) (valid : exports.program.valid statement.zi witness)
    (lightNext : control statement.zi witness = 0)
    (room : ∀ pcValid : InRange 2 payload.pc,
      (payload.running (selectedIndex pcValid)).2 < Schedule.productionLightFoldCap)
    (iterationNext : statement.iteration + 1 < goldilocksModulus)
    (nextLength : (step statement.zi witness).length = Stage1.Application.stateWordCount)
    (witnessLength : witness.length = exports.program.witnessWordCount) :
    ∃ next, Terminal.HoldsFor cap ajtai circuitKeys step control
        ⟨statement.iteration + 1, statement.z0, step statement.zi witness⟩ (.recursive next) ∧
      Honest (ajtai := ajtai) complete.packageWidth next := by
  obtain ⟨statementValid, pcValid, positive, freshPublic, slots, freshHolds⟩ :=
    (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control statement payload).mp holds
  let selected := selectedIndex pcValid
  have normWithin := honest.norm selected
  have depthLt := room pcValid
  have running : Light.Rlc.Holds (PaperAlgebra.semantics ajtai)
      (productionGlobalParams.accumulatorBound (payload.running selected).2 + 1)
      (toClaim ((circuitKeys selected).system ajtai) (payload.running selected).1)
      (payload.runningWitness selected) :=
    ⟨⟨(slots selected).1.1, (slots selected).1.2.1, normWithin⟩, (slots selected).2.1,
      (slots selected).2.2⟩
  obtain ⟨proof, folded, foldedWitness, accepted, foldedHolds, foldedSupported⟩ :=
    FoldCompleteness.light_folds_of_holds ajtai (circuitKeys selected) (payload.running selected).1
      payload.fresh _ (payload.runningWitness selected) payload.freshWitness running freshHolds
  let input : StepInput logicalWidth publicFits :=
    { iteration := statement.iteration, z0 := statement.z0, zi := statement.zi,
      running := payload.running, fresh := payload.fresh, priorPc := payload.pc,
      witness := witness, nifsProof := .light proof }
  let runningNext := Function.update payload.running selected
    (folded, (payload.running selected).2 + 1)
  let draft : StepOutput logicalWidth publicFits :=
    { zNext := step statement.zi witness, runningNext := runningNext, pcNext := 0, x := [] }
  let output : StepOutput logicalWidth publicFits :=
    { draft with x := stateHash (toState (Paper.nextHashPreimage
        ((setup Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit 0) input draft)) }
  have stepHolds : StepHolds Schedule.productionLightFoldCap ajtai circuitKeys step control 0
      input output := by
    apply (stepHolds_iff _ _ _ _ _ _ _ _).mpr
    refine ⟨lightNext, rfl, rfl, rfl, Or.inr ⟨pcValid, positive, freshPublic, fun _ => depthLt,
      ?_, ?_, ?_⟩⟩
    · show OptionBFolds.folds .light (keys (circuitKeys selected).relation ajtai)
          (toClaim ((circuitKeys selected).system ajtai) (payload.running selected).1) payload.fresh
          (.light proof) = some (toClaim ((circuitKeys selected).system ajtai)
            (output.runningNext selected).1)
      rw [show output.runningNext selected = (folded, (payload.running selected).2 + 1) from
        Function.update_self _ _ _]
      exact accepted
    · show (output.runningNext selected).2 = Schedule.nextDepth (payload.running selected).2 .light
      rw [show output.runningNext selected = (folded, (payload.running selected).2 + 1) from
        Function.update_self _ _ _]
      rfl
    · intro slot other
      exact Function.update_of_ne other _ _
  have small : (Schedule.productionLightFoldCap : Nat) < goldilocksModulus := by decide
  have pcBound : payload.pc < goldilocksModulus := by
    have := pcValid.2
    have : (2 : Nat) < goldilocksModulus := by decide
    omega
  have nextSelected : runningNext selected = (folded, (payload.running selected).2 + 1) :=
    Function.update_self _ _ _
  have nextOther : ∀ slot, slot ≠ selected → runningNext slot = payload.running slot :=
    fun slot other => Function.update_of_ne other _ _
  have nextDepthLe : ∀ slot, (runningNext slot).2 ≤ Schedule.productionLightFoldCap := by
    intro slot
    by_cases same : slot = selected
    · rw [same, nextSelected]
      exact depthLt
    · rw [nextOther slot same]
      exact honest.depth slot
  obtain ⟨env, rows, represents⟩ := LightAssemblerCompleteness.complete exports.program relation
    base ajtai circuitKeys step control 0 rfl exports.programStep input output proof selected
    folded stepHolds valid
    (fun _ => by
      show payload.pc = (payload.pc - 1) + 1
      have := pcValid.1
      omega)
    freshPublic (fun _ => rfl) accepted depthLt statementValid.1 pcBound
    (fun slot => lt_of_le_of_lt (honest.depth slot) small)
    (fun slot => lt_of_le_of_lt (nextDepthLe slot) small) keysShaped statementValid.2.1
    statementValid.2.2 nextLength witnessLength
  obtain ⟨assignment, freshNext, freshZero⟩ := light_complete exports.program relation ajtai
    circuitKeys input output proof env rows represents complete.lightFree exports.lightFresh
    exports.lightWidth exports.lightFits complete.lightPacked complete.lightBelow
    complete.lightPublicBelow complete.lightBatches complete.lightWide cubeFits
  have keyEq : (circuitKeys 0).relation = (CircuitRelation.plan publicColumns _ exports.lightFresh
      exports.lightWidth exports.lightFits
      (WideLayout.widen (SlotLayout.layout _) complete.lightWide)).logicalRelation
        (publicFits := publicFits) cubeFits := by
    rw [exports.lightKey, complete.lightLayout]
  have foldedWithin : Light.Rlc.Holds (PaperAlgebra.semantics ajtai)
      (productionGlobalParams.accumulatorBound ((payload.running selected).2 + 1) + 1)
      (toClaim ((circuitKeys selected).system ajtai) folded) foldedWitness := by
    rw [← accumulatorBound_succ]
    exact foldedHolds
  let next : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2 :=
    { running := runningNext
      runningWitness := Function.update payload.runningWitness selected foldedWitness
      fresh := { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai
                   (Phi81CarrierLayout.extendAssignment 0 assignment)
                 publicInputs := fun _ => encHash (publicFits := publicFits) output.x }
      freshWitness := Phi81CarrierLayout.extendAssignment 0 assignment
      pc := 1 }
  have pcOne : InRange 2 1 := ⟨le_refl 1, by decide⟩
  refine ⟨next, (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control _ next).mpr
    ⟨⟨iterationNext, statementValid.2.1, nextLength⟩, pcOne, Nat.succ_pos _, rfl, ?_, ?_⟩, ?_⟩
  · intro slot
    by_cases same : slot = selected
    · rw [same]
      show CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (toClaim ((circuitKeys selected).system ajtai) (runningNext selected).1)
        (Function.update payload.runningWitness selected foldedWitness selected)
      rw [nextSelected, Function.update_self]
      exact ⟨⟨foldedWithin.1.1, foldedWithin.1.2.1,
        normBounded_mono foldedWithin.1.2.2 (bound_le_bigB depthLt)⟩, foldedWithin.2.1,
        foldedWithin.2.2⟩
    · show CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (toClaim ((circuitKeys slot).system ajtai) (runningNext slot).1)
        (Function.update payload.runningWitness selected foldedWitness slot)
      rw [nextOther slot same, Function.update_of_ne same]
      exact slots slot
  · show CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (freshStatement (circuitKeys 0).relation next.fresh) next.freshWitness
    rw [keyEq]
    exact freshNext
  · refine ⟨nextDepthLe, fun slot => ?_, fun slot => ?_, fun pcNext => ?_⟩
    · show (PaperAlgebra.semantics ajtai).normBounded
        (productionGlobalParams.accumulatorBound (runningNext slot).2 + 1)
        (Function.update payload.runningWitness selected foldedWitness slot)
      by_cases same : slot = selected
      · rw [same, nextSelected, Function.update_self]
        exact foldedWithin.1.2.2
      · rw [nextOther slot same, Function.update_of_ne same]
        exact honest.norm slot
    · show Supported (complete.packageWidth slot)
        (Function.update payload.runningWitness selected foldedWitness slot)
      by_cases same : slot = selected
      · rw [same, Function.update_self]
        exact foldedSupported _ (honest.support selected) (honest.freshSupport pcValid)
      · rw [Function.update_of_ne same]
        exact honest.support slot
    · rw [show selectedIndex pcNext = (0 : Fin 2) from Fin.ext rfl, Complete.packageWidth_zero]
      exact supported_of_tail freshZero

/-- One honest refresh step: from a recursive proof that the terminal accepts and whose slots are
honest, a valid application step whose circuit is the refresh circuit gives a recursive proof
that the terminal accepts at the next iteration, with honest slots; the refreshed slot returns
to depth 0. -/
theorem refresh_extend (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (holds : Terminal.HoldsFor cap ajtai circuitKeys step control statement (.recursive payload))
    (honest : Honest (ajtai := ajtai) complete.packageWidth payload)
    (witness : AppWitness) (valid : exports.program.valid statement.zi witness)
    (refreshNext : control statement.zi witness = 1)
    (iterationNext : statement.iteration + 1 < goldilocksModulus)
    (nextLength : (step statement.zi witness).length = Stage1.Application.stateWordCount)
    (witnessLength : witness.length = exports.program.witnessWordCount) :
    ∃ next, Terminal.HoldsFor cap ajtai circuitKeys step control
        ⟨statement.iteration + 1, statement.z0, step statement.zi witness⟩ (.recursive next) ∧
      Honest (ajtai := ajtai) complete.packageWidth next := by
  obtain ⟨statementValid, pcValid, positive, freshPublic, slots, freshHolds⟩ :=
    (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control statement payload).mp holds
  let selected := selectedIndex pcValid
  obtain ⟨proof, folded, foldedWitness, accepted, foldedHolds, foldedNorm, foldedSupported⟩ :=
    FoldCompleteness.refresh_folds_bounded ajtai (circuitKeys selected)
      (payload.running selected).1 payload.fresh (payload.runningWitness selected)
      payload.freshWitness (slots selected) freshHolds
  let input : StepInput logicalWidth publicFits :=
    { iteration := statement.iteration, z0 := statement.z0, zi := statement.zi,
      running := payload.running, fresh := payload.fresh, priorPc := payload.pc,
      witness := witness, nifsProof := .refresh proof }
  let runningNext := Function.update payload.running selected
    (folded, 0)
  let draft : StepOutput logicalWidth publicFits :=
    { zNext := step statement.zi witness, runningNext := runningNext, pcNext := 1, x := [] }
  let output : StepOutput logicalWidth publicFits :=
    { draft with x := stateHash (toState (Paper.nextHashPreimage
        ((setup Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit 1) input draft)) }
  have stepHolds : StepHolds Schedule.productionLightFoldCap ajtai circuitKeys step control 1
      input output := by
    apply (stepHolds_iff _ _ _ _ _ _ _ _).mpr
    refine ⟨refreshNext, rfl, rfl, rfl, Or.inr ⟨pcValid, positive, freshPublic,
      fun light => by simp [BoundedDepth.circuitStep] at light, ?_, ?_, ?_⟩⟩
    · show OptionBFolds.folds .refresh (keys (circuitKeys selected).relation ajtai)
          (toClaim ((circuitKeys selected).system ajtai) (payload.running selected).1) payload.fresh
          (.refresh proof) = some (toClaim ((circuitKeys selected).system ajtai)
            (output.runningNext selected).1)
      rw [show output.runningNext selected = (folded, 0) from Function.update_self _ _ _]
      exact accepted
    · show (output.runningNext selected).2 =
        Schedule.nextDepth (payload.running selected).2 .refresh
      rw [show output.runningNext selected = (folded, 0) from Function.update_self _ _ _]
      rfl
    · intro slot other
      exact Function.update_of_ne other _ _
  have small : (Schedule.productionLightFoldCap : Nat) < goldilocksModulus := by decide
  have pcBound : payload.pc < goldilocksModulus := by
    have := pcValid.2
    have : (2 : Nat) < goldilocksModulus := by decide
    omega
  have nextSelected : runningNext selected = (folded, 0) :=
    Function.update_self _ _ _
  have nextOther : ∀ slot, slot ≠ selected → runningNext slot = payload.running slot :=
    fun slot other => Function.update_of_ne other _ _
  have nextDepthLe : ∀ slot, (runningNext slot).2 ≤ Schedule.productionLightFoldCap := by
    intro slot
    by_cases same : slot = selected
    · rw [same, nextSelected]
      exact Nat.zero_le _
    · rw [nextOther slot same]
      exact honest.depth slot
  obtain ⟨env, rows, represents⟩ := RefreshAssemblerCompleteness.complete exports.program relation
    base ajtai circuitKeys step control 1 rfl exports.programStep input output proof selected
    folded stepHolds valid
    (fun _ => by
      show payload.pc = (payload.pc - 1) + 1
      have := pcValid.1
      omega)
    freshPublic (fun _ => rfl) accepted statementValid.1 pcBound
    (fun slot => lt_of_le_of_lt (honest.depth slot) small)
    (fun slot => lt_of_le_of_lt (nextDepthLe slot) small) keysShaped statementValid.2.1
    statementValid.2.2 nextLength witnessLength
  obtain ⟨assignment, freshNext, freshZero⟩ := refresh_complete exports.program relation ajtai
    circuitKeys input output proof env rows represents complete.refreshFree exports.refreshFresh
    exports.refreshWidth exports.refreshFits complete.refreshPacked complete.refreshBelow
    complete.refreshPublicBelow complete.refreshBatches complete.refreshWide cubeFits
  have keyEq : (circuitKeys 1).relation = (CircuitRelation.plan publicColumns _
      exports.refreshFresh exports.refreshWidth exports.refreshFits
      (WideLayout.widen (SlotLayout.layout _) complete.refreshWide)).logicalRelation
        (publicFits := publicFits) cubeFits := by
    rw [exports.refreshKey, complete.refreshLayout]
  let next : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2 :=
    { running := runningNext
      runningWitness := Function.update payload.runningWitness selected foldedWitness
      fresh := { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai
                   (Phi81CarrierLayout.extendAssignment 0 assignment)
                 publicInputs := fun _ => encHash (publicFits := publicFits) output.x }
      freshWitness := Phi81CarrierLayout.extendAssignment 0 assignment
      pc := 2 }
  have pcTwo : InRange 2 2 := ⟨by decide, le_refl 2⟩
  refine ⟨next, (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control _ next).mpr
    ⟨⟨iterationNext, statementValid.2.1, nextLength⟩, pcTwo, Nat.succ_pos _, rfl, ?_, ?_⟩, ?_⟩
  · intro slot
    by_cases same : slot = selected
    · rw [same]
      show CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (toClaim ((circuitKeys selected).system ajtai) (runningNext selected).1)
        (Function.update payload.runningWitness selected foldedWitness selected)
      rw [nextSelected, Function.update_self]
      exact foldedHolds
    · show CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (toClaim ((circuitKeys slot).system ajtai) (runningNext slot).1)
        (Function.update payload.runningWitness selected foldedWitness slot)
      rw [nextOther slot same, Function.update_of_ne same]
      exact slots slot
  · show CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (freshStatement (circuitKeys 1).relation next.fresh) next.freshWitness
    rw [keyEq]
    exact freshNext
  · refine ⟨nextDepthLe, fun slot => ?_, fun slot => ?_, fun pcNext => ?_⟩
    · show (PaperAlgebra.semantics ajtai).normBounded
        (productionGlobalParams.accumulatorBound (runningNext slot).2 + 1)
        (Function.update payload.runningWitness selected foldedWitness slot)
      by_cases same : slot = selected
      · rw [same, nextSelected, Function.update_self]
        exact foldedNorm
      · rw [nextOther slot same, Function.update_of_ne same]
        exact honest.norm slot
    · show Supported (complete.packageWidth slot)
        (Function.update payload.runningWitness selected foldedWitness slot)
      by_cases same : slot = selected
      · rw [same, Function.update_self]
        exact foldedSupported _ (honest.support selected) (honest.freshSupport pcValid)
      · rw [Function.update_of_ne same]
        exact honest.support slot
    · rw [show selectedIndex pcNext = (1 : Fin 2) from Fin.ext rfl, Complete.packageWidth_one]
      exact supported_of_tail freshZero

/-- The first honest step: from the bottom proof, a valid application step on the light circuit
gives a recursive proof that the terminal accepts at iteration 1, with both slots at the default
claim and the zero witness. The fold is HyperNova's zero proof (`BaseFold.zero_folds`). -/
theorem base_extend (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) (statement : TerminalStatement AppState)
    (holds : Terminal.HoldsFor cap ajtai circuitKeys step control statement .bottom)
    (witness : AppWitness) (valid : exports.program.valid statement.zi witness)
    (lightNext : control statement.zi witness = 0)
    (nextLength : (step statement.zi witness).length = Stage1.Application.stateWordCount)
    (witnessLength : witness.length = exports.program.witnessWordCount) :
    ∃ next, Terminal.HoldsFor cap ajtai circuitKeys step control
        ⟨statement.iteration + 1, statement.z0, step statement.zi witness⟩ (.recursive next) ∧
      Honest (ajtai := ajtai) complete.packageWidth next := by
  obtain ⟨statementValid, iterationZero, ziEq⟩ :=
    (Terminal.holdsFor_bottom_iff cap ajtai circuitKeys step control statement).mp holds
  let defaults : Fin 2 → SlotData logicalWidth publicFits := fun _ => (defaultSlot, 0)
  let prior : Paper.HashPreimage (CircuitKey logicalWidth publicFits) AppState
      (SlotData logicalWidth publicFits) 2 :=
    { verifierKeys := circuitKeys, iteration := statement.iteration, z0 := statement.z0,
      current := statement.zi, running := defaults, pc := 0 }
  let digest := stateHash (toState prior)
  let input : StepInput logicalWidth publicFits :=
    { iteration := statement.iteration, z0 := statement.z0, zi := statement.zi,
      running := defaults, fresh := BaseFold.baseFresh digest, priorPc := 0,
      witness := witness, nifsProof := .light BaseFold.zeroProof }
  obtain ⟨folded, accepted⟩ := BaseFold.zero_folds ajtai (circuitKeys 0) digest
  let draft : StepOutput logicalWidth publicFits :=
    { zNext := step statement.zi witness, runningNext := defaults, pcNext := 0, x := [] }
  let output : StepOutput logicalWidth publicFits :=
    { draft with x := stateHash (toState (Paper.nextHashPreimage
        ((setup Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit 0) input draft)) }
  have stepHolds : StepHolds Schedule.productionLightFoldCap ajtai circuitKeys step control 0
      input output :=
    (stepHolds_iff _ _ _ _ _ _ _ _).mpr ⟨lightNext, rfl, rfl, rfl,
      Or.inl ⟨iterationZero, ziEq.symm, rfl⟩⟩
  have priorSame : Paper.priorHashPreimage
      ((setup Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit 0) input = prior := by
    unfold Paper.priorHashPreimage
    rfl
  have freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (publicFits := publicFits) (stateHash (toState (Paper.priorHashPreimage
        ((setup Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit 0) input))) := by
    rw [priorSame]
    rfl
  obtain ⟨env, rows, represents⟩ := LightAssemblerCompleteness.complete exports.program relation
    base ajtai circuitKeys step control 0 rfl exports.programStep input output BaseFold.zeroProof 0
    folded stepHolds valid (fun nonzero => absurd iterationZero nonzero) freshPublic
    (fun nonzero => absurd iterationZero nonzero) accepted
    (show (0 : Nat) < Schedule.productionLightFoldCap by decide) statementValid.1
    (show (0 : Nat) < goldilocksModulus by decide)
    (fun _ => show (0 : Nat) < goldilocksModulus by decide)
    (fun _ => show (0 : Nat) < goldilocksModulus by decide)
    keysShaped statementValid.2.1 statementValid.2.2 nextLength witnessLength
  obtain ⟨assignment, freshNext, freshZero⟩ := light_complete exports.program relation ajtai
    circuitKeys input output BaseFold.zeroProof env rows represents complete.lightFree
    exports.lightFresh exports.lightWidth exports.lightFits complete.lightPacked
    complete.lightBelow complete.lightPublicBelow complete.lightBatches complete.lightWide cubeFits
  have keyEq : (circuitKeys 0).relation = (CircuitRelation.plan publicColumns _ exports.lightFresh
      exports.lightWidth exports.lightFits
      (WideLayout.widen (SlotLayout.layout _) complete.lightWide)).logicalRelation
        (publicFits := publicFits) cubeFits := by
    rw [exports.lightKey, complete.lightLayout]
  let next : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2 :=
    { running := defaults
      runningWitness := fun _ => Phi81Relation.EvaluationHomomorphism.BaseLinear.assignmentZero
      fresh := { commitments := fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai
                   (Phi81CarrierLayout.extendAssignment 0 assignment)
                 publicInputs := fun _ => encHash (publicFits := publicFits) output.x }
      freshWitness := Phi81CarrierLayout.extendAssignment 0 assignment
      pc := 1 }
  have pcOne : InRange 2 1 := ⟨le_refl 1, by decide⟩
  have iterationOne : statement.iteration + 1 < goldilocksModulus := by
    rw [iterationZero]
    decide
  refine ⟨next, (Terminal.holdsFor_recursive_iff cap ajtai circuitKeys step control _ next).mpr
    ⟨⟨iterationOne, statementValid.2.1, nextLength⟩, pcOne, Nat.succ_pos _, rfl, ?_, ?_⟩, ?_⟩
  · intro slot
    have zero := PilotZeroRunning.defaultRunning_holds (circuitKeys slot).relation ajtai
      ⟨0, by decide⟩
    exact ⟨⟨zero.1.1, zero.1.2.1, normBounded_mono zero.1.2.2
      (show productionGlobalParams.b ≤ productionGlobalParams.bigB by decide)⟩, zero.2.1,
      zero.2.2⟩
  · show CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
      (freshStatement (circuitKeys 0).relation next.fresh) next.freshWitness
    rw [keyEq]
    exact freshNext
  · refine ⟨fun _ => Nat.zero_le _, fun slot column => ?_, fun slot => supported_zero _,
      fun pcNext => ?_⟩
    · change centeredMagnitude (0 : F) < productionGlobalParams.accumulatorBound 0 + 1
      simp [centeredMagnitude]
    · rw [show selectedIndex pcNext = (0 : Fin 2) from Fin.ext rfl, Complete.packageWidth_zero]
      exact supported_of_tail freshZero

end NightstreamFPrime.Export.OptionB.StepExtend
