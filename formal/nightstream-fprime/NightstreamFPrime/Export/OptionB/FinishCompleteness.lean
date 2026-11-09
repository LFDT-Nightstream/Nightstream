import NightstreamFPrime.Export.OptionB.RunCompleteness
import NightstreamFPrime.Export.OptionB.FinishRelation
import NightstreamFPrime.Export.OptionB.FinishWalk
import NightstreamFPrime.Layout.OptionB.FinishAssemblerCompleteness

/-! Completeness of the finish package (CW-10.6): the final program's checks on the finish columns
accept the honest finish of every valid run, and both next slots' claims hold with their witnesses.
No statement here says that the shrink proof or the slots' layer-1 proofs are accepted.

The finish package's checks are the converse of `FinishRelation.package_finish`
(`package_complete`). Take values on which every row of the finish circuit holds
(`Layout.OptionB.FinishAssemblerCompleteness.complete`). Set the public columns to `encHash` of the
output digest that the words carry (`StepCompleteness.withPublic`). The finish rows read no public
column, so every finish constraint holds. The lowering's fresh columns from the first fresh column
then satisfy the rows of the package's pieces, and each block's outputs are the permutation of its
lanes (`RelationCompleteness.checks_complete`).

The honest finish is the last refresh fold of a recursive proof that the terminal accepts and
whose slots are honest (`StepExtend.Honest`): the fresh instance folds into the selected slot
(`FoldCompleteness.refresh_folds_bounded`), the witness words are 0, and the next state is the
current state, as Rust's `final_witness` builds them. In the finish columns, the words carry the
setup's key digests and the proof's statement, the decoded step passes the finish checks
(`FinishRelation.constraints_finish`), and both next slots are the honest ones, whose claims
hold with witnesses within the accumulator bound of the light cap. Each next slot witness is zero
on every ring block from its circuit's package width (`BlockSupport.Supported`), because the
fold keeps zero blocks (`finish_complete`). With
`RunCompleteness.run_complete`, this holds for every valid run of at least one step
(`run_finish_complete`).

Three premises are computed checks on the finish package (`FinishComplete`): every piece reads
only columns below the first fresh column, no row of the finish circuit reads a public column,
and the shared words lie below the first fresh column. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.FinishCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Lifecycle.OptionB.BlockSupport
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.OptionB.StepRelation
open NightstreamFPrime.Export.OptionB.TerminalBridge
open NightstreamFPrime.Export.OptionB.StepCompleteness
open NightstreamFPrime.Export.OptionB.StepExtend

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-! ### The finish package -/

section Package

variable (program : Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)

/-- The finish circuit's rows at the refresh step's shared end. -/
abbrev finishOps : List Op :=
  Circuit.ops (FinishStep.main program relation 1
      (RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
        program.witnessWordCount))
    (RefreshAssembler.sharedEnd base program.witnessWordCount)

/-- The finish package that the emitter writes (`EmitMain.emitFinish`). -/
abbrev package (firstFresh width : Nat) (batches : List WitnessBatch)
    (application : Nat × Nat × Nat × Nat × Nat) : StepPackage.Package :=
  StepPackage.packageOf 2 CircuitRelation.publicColumns (FinishRelation.pieces program relation)
    firstFresh width batches application

/-- The finish package's checks that its completeness reads: every piece reads only columns below
the first fresh column, no row of the finish circuit reads a public column, and the shared words
lie below the first fresh column. The emitter evaluates them and writes no finish package when one
fails (`EmitMain.emitPackage`). -/
structure FinishComplete (firstFresh : Nat) : Prop where
  below : (FinishRelation.pieces program relation).all (pieceBelow firstFresh) = true
  free : publicFree (flatConstraints (finishOps program relation)) = true
  sharedBelow : RefreshAssembler.sharedEnd base program.witnessWordCount ≤ firstFresh

/-- Completeness of the finish package, the converse of `FinishRelation.package_finish`: values on
which every finish row holds, whose words `refreshDigestColumn` carry `digest`, give columns that
agree with them below the first fresh column, except the public columns, which carry
`encHash digest`. There every finish constraint and every row of the package's pieces hold, and
each block's outputs are the permutation of its lanes' values. -/
theorem package_complete (firstFresh width : Nat) (batches : List WitnessBatch)
    (application : Nat × Nat × Nat × Nat × Nat)
    (complete : FinishComplete program relation firstFresh) (env : Env)
    (holds : holdsFlat env (finishOps program relation)) (digest : Digest)
    (digestWords : ∀ word, env (refreshDigestColumn program word) = digest.getD word.val 0) :
    ∃ completed : Env,
      (∀ column, column < firstFresh →
        completed column = withPublic (publicFits := publicFits) digest env column) ∧
      ConstraintsHold completed (FinishRelation.constraints program relation) ∧
      (∀ r ∈ (package program relation firstFresh width batches application).pieces.flatMap
        StepPackage.PackagePiece.relationRows, r.Holds completed) ∧
      ∀ block ∈ (package program relation firstFresh width batches application).blocks,
        (fun lane : Fin 16 => completed (block.start + 1080 + lane)) =
          Gadgets.Poseidon2.Permutation.runF Gadgets.Poseidon2.Permutation.schedule
            fun lane => (block.lanes.getD lane 0).eval completed := by
  have all := constraints_complete (publicFits := publicFits) (finishOps program relation)
    (refreshDigestColumn program) env holds digest digestWords (digestColumn_above program)
    complete.free
  obtain ⟨completed, kept, piecesHold, rows, outputs⟩ := RelationCompleteness.checks_complete
    (FinishRelation.pieces program relation) (Pieces.valid_split _ _) firstFresh complete.below
    (withPublic (publicFits := publicFits) digest env) fun piece member expression inPiece =>
      all expression (by
        show expression ∈ FinishRelation.constraints program relation
        rw [← Pieces.flatMap_split (PilotData.canonicalConstraints ())
          (FinishRelation.constraints program relation)]
        exact List.mem_flatMap.mpr ⟨piece, member, inPiece⟩)
  refine ⟨completed, kept, CircuitRelation.split_holds piecesHold, ?_, ?_⟩
  · rw [package, StepPackage.pieceRows_packageOf]
    exact rows
  · rw [package, StepPackage.blocks_packageOf]
    intro block member
    have linear : Pieces.LinearLanes block.lanes :=
      Pieces.valid_split _ _ _ (RelationCompleteness.permutation_mem_of_block member)
    have inputs : (fun lane : Fin 16 => (block.lanes.getD lane 0).eval completed) =
        block.input completed :=
      funext fun lane => (RelationCompleteness.block_input linear completed lane).symm
    rw [inputs]
    exact outputs block member

end Package

/-! ### The honest finish -/

theorem natWord_val_of_lt {count : Nat} (bound : count < goldilocksModulus) :
    (natWord count).val = count := by
  simp [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt bound]

theorem accumulatorBound_mono {small large : Nat} (le : small ≤ large) :
    productionGlobalParams.accumulatorBound small ≤
      productionGlobalParams.accumulatorBound large := by
  unfold GlobalParams.accumulatorBound
  exact Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (by omega))

/-- The words of a slot from `start` read only its 2,055 columns. -/
theorem slotAt_evalList (start : Nat) {left right : Env}
    (agree : ∀ column, start ≤ column → column < start + LightAssembler.slotWordCount →
      left column = right column) :
    Gadgets.Poseidon2.Hash.evalList left (LightAssembler.slotAt (logicalWidth := logicalWidth)
        (publicFits := publicFits) start).words =
      Gadgets.Poseidon2.Hash.evalList right (LightAssembler.slotAt (logicalWidth := logicalWidth)
        (publicFits := publicFits) start).words := by
  unfold Gadgets.Poseidon2.Hash.evalList
  apply List.map_congr_left
  intro word member
  simp only [SlotExpr.words, LightAssembler.slotAt, LightAssembler.kAt,
    StatementAbsorption.serializeCommitmentExpr, StatementAbsorption.serializeEvalKExpr,
    StatementAbsorption.serializeEvalAExpr, StatementAbsorption.serializePointExpr,
    StatementAbsorption.serializePublicInputExpr, StatementAbsorption.serializeKExpr,
    List.mem_append, List.mem_flatMap, List.mem_map, List.mem_finRange, List.mem_cons,
    List.not_mem_nil, or_false, true_and] at member
  simp only [LightAssembler.slotWordCount] at agree
  rcases member with ((((commitment | pad | matrix) | point) | publicWords) | depth)
  · obtain ⟨row, lane, rfl⟩ := commitment
    have rowBound := row.isLt
    have laneBound := lane.isLt
    change row.val < 22 at rowBound
    change lane.val < 54 at laneBound
    exact agree (start + row.val * 54 + lane.val) (by omega) (by omega)
  · obtain ⟨coefficient, rfl | rfl⟩ := pad <;>
    · have coefficientBound := coefficient.isLt
      change coefficient.val < 54 at coefficientBound
      exact agree _ (by omega) (by omega)
  · obtain ⟨matrix, coefficient, rfl | rfl⟩ := matrix <;>
    · have matrixBound := matrix.isLt
      have coefficientBound := coefficient.isLt
      change matrix.val < 4 at matrixBound
      change coefficient.val < 54 at coefficientBound
      simp only [ringDegree]
      exact agree _ (by omega) (by omega)
  · obtain ⟨coordinate, rfl | rfl⟩ := point <;>
    · have coordinateBound := coordinate.isLt
      change coordinate.val < 28 at coordinateBound
      exact agree _ (by omega) (by omega)
  · obtain ⟨column, rfl⟩ := publicWords
    have columnBound := column.isLt
    change column.val < 270 at columnBound
    exact agree (start + 1784 + column.val) (by omega) (by omega)
  · subst depth
    exact agree (start + 2054) (by omega) (by omega)

section Finish

variable (program : Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)

/-- The final program's checks accept `statement` on the finish package: some finish columns
hold one at column 0, satisfy every row of the package's pieces, and make each Poseidon2 block's
outputs the permutation of its lanes. Their key words are the setup's key digests, and they
decode to a refresh step input with the statement's iteration and states that passes the finish
checks with its output. Both next slots' claims hold, with witnesses within the accumulator bound
of the light cap that are zero on every ring block from the package width of the slot's circuit
(`packageWidths`). -/
def FinishAccepts (firstFresh width : Nat) (batches : List WitnessBatch)
    (application : Nat × Nat × Nat × Nat × Nat) (packageWidths : Fin 2 → Nat)
    (statement : TerminalStatement AppState) : Prop :=
  ∃ (env : Env)
    (witnesses : Fin 2 →
      Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)),
    env 0 = 1 ∧
    (∀ r ∈ (package program relation firstFresh width batches application).pieces.flatMap
      StepPackage.PackagePiece.relationRows, r.Holds env) ∧
    (∀ block ∈ (package program relation firstFresh width batches application).blocks,
      (fun lane : Fin 16 => env (block.start + 1080 + lane)) =
        Gadgets.Poseidon2.Permutation.runF Gadgets.Poseidon2.Permutation.schedule
          fun lane => (block.lanes.getD lane 0).eval env) ∧
    decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) program).verifierKeys env = circuitKeys ∧
    FinishWalk.finishStatement (refreshInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) program env) = statement ∧
    FinishStep.FinishHolds Schedule.productionLightFoldCap ajtai circuitKeys 1
      (refreshInput program env) (finishStepOutput program 1 env) ∧
    ∀ slot, CE.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (toClaim ((circuitKeys slot).system ajtai)
          ((finishStepOutput (logicalWidth := logicalWidth) (publicFits := publicFits) program 1
            env).runningNext slot).1)
        (witnesses slot) ∧
      (PaperAlgebra.semantics ajtai).normBounded
        (productionGlobalParams.accumulatorBound Schedule.productionLightFoldCap + 1)
        (witnesses slot) ∧
      Supported (logicalWidth := logicalWidth) (packageWidths slot) (witnesses slot)

/-- The honest finish: from a recursive proof that the terminal accepts and whose slots are
honest, the last refresh fold of the fresh instance into the selected slot gives finish columns
that the final program's checks accept for the proof's statement. -/
theorem finish_complete (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (firstFresh width : Nat) (batches : List WitnessBatch)
    (application : Nat × Nat × Nat × Nat × Nat)
    (complete : FinishComplete program relation firstFresh) (cap : Nat)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (statement : TerminalStatement AppState)
    (payload : TerminalProof (SlotData logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits))
      (StepFresh logicalWidth publicFits)
      (Terminal.Witness (logicalWidth := logicalWidth) (publicFits := publicFits)) 2)
    (holds : Terminal.HoldsFor cap ajtai circuitKeys step control statement (.recursive payload))
    (packageWidths : Fin 2 → Nat) (honest : Honest (ajtai := ajtai) packageWidths payload) :
    FinishAccepts program relation ajtai circuitKeys firstFresh width batches application
      packageWidths statement := by
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
      witness := List.replicate program.witnessWordCount 0, nifsProof := .refresh proof }
  let runningNext := Function.update payload.running selected (folded, 0)
  let draft : StepOutput logicalWidth publicFits :=
    { zNext := statement.zi, runningNext := runningNext, pcNext := 1, x := [] }
  let output : StepOutput logicalWidth publicFits :=
    { draft with x := stateHash (toState (Paper.nextHashPreimage
        ((setup Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit 1) input draft)) }
  have nextSelected : runningNext selected = (folded, 0) := Function.update_self _ _ _
  have nextOther : ∀ slot, slot ≠ selected → runningNext slot = payload.running slot :=
    fun slot other => Function.update_of_ne other _ _
  have finishHolds : FinishStep.FinishHolds Schedule.productionLightFoldCap ajtai circuitKeys 1
      input output := by
    refine ⟨pcValid, ?_⟩
    dsimp only
    refine ⟨freshPublic, ?_, ?_, nextOther⟩
    · show OptionBFolds.folds .refresh (keys (circuitKeys selected).relation ajtai)
          (toClaim ((circuitKeys selected).system ajtai) (payload.running selected).1) payload.fresh
          (.refresh proof) = some (toClaim ((circuitKeys selected).system ajtai)
            (runningNext selected).1)
      rw [nextSelected]
      exact accepted
    · show (runningNext selected).2 = 0
      rw [nextSelected]
  have small : (Schedule.productionLightFoldCap : Nat) < goldilocksModulus := by decide
  have pcBound : payload.pc < goldilocksModulus := by
    have := pcValid.2
    have : (2 : Nat) < goldilocksModulus := by decide
    omega
  have nextDepthLe : ∀ slot, (runningNext slot).2 ≤ Schedule.productionLightFoldCap := by
    intro slot
    by_cases same : slot = selected
    · rw [same, nextSelected]
      exact Nat.zero_le _
    · rw [nextOther slot same]
      exact honest.depth slot
  obtain ⟨env, rows, represents, nextWords⟩ := FinishAssemblerCompleteness.complete program
    relation base
    ajtai circuitKeys 1 input output proof finishHolds (Nat.pos_iff_ne_zero.mp positive) rfl rfl
    statementValid.1 pcBound (fun slot => lt_of_le_of_lt (honest.depth slot) small)
    (fun slot => lt_of_le_of_lt (nextDepthLe slot) small) keysShaped statementValid.2.1
    statementValid.2.2 statementValid.2.2 List.length_replicate (Spec.Poseidon2.hash_length _)
  obtain ⟨completed, kept, constraintsHold, packageRows, packageBlocks⟩ :=
    package_complete program relation firstFresh width batches application complete env rows
      output.x (digest_getD represents.outputDigest)
  have sharedEq := RefreshAssembler.sharedEnd_eq base program.witnessWordCount
  have shared : ∀ column, base ≤ column →
      column < RefreshAssembler.sharedEnd base program.witnessWordCount →
        completed column = env column := by
    intro column atLeast below
    rw [kept column (lt_of_lt_of_le below complete.sharedBelow)]
    unfold withPublic
    rw [dite_eq_right (show ¬ column < CircuitRelation.publicColumns from not_lt.mpr atLeast)]
  have keysEq : decodedKeys circuitKeys (refreshInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) program).verifierKeys completed = circuitKeys := by
    funext key
    have words : List.ofFn (fun lane => ((refreshInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) program).verifierKeys key lane).eval completed) =
          (circuitKeys key).digest := by
      rw [← represents.verifierKeys key]
      refine congrArg List.ofFn (funext fun lane => ?_)
      have keyBound := key.isLt
      have laneBound := lane.isLt
      exact shared _ (by simp only [LightAssembler.keysStart]; omega)
        (by simp only [LightAssembler.keysStart]; omega)
    show (⟨_, (circuitKeys key).relation⟩ : CircuitKey logicalWidth publicFits) =
      circuitKeys key
    rw [words]
  have iterationEq : (refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits)
      program completed).iteration = statement.iteration := by
    have word : completed base = natWord statement.iteration :=
      (shared base le_rfl (by omega)).trans represents.iteration
    show (completed base).val = statement.iteration
    rw [word, natWord_val_of_lt statementValid.1]
  have z0Eq : (refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits)
      program completed).z0 = statement.z0 := by
    refine Eq.trans ?_ represents.initialState
    refine congrArg List.ofFn (funext fun lane => ?_)
    have laneBound := lane.isLt
    exact shared _ (by simp only [LightAssembler.statesStart]; omega)
      (by simp only [LightAssembler.statesStart]; omega)
  have ziEq : (refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits)
      program completed).zi = statement.zi := by
    refine Eq.trans ?_ represents.currentState
    refine congrArg List.ofFn (funext fun lane => ?_)
    have laneBound := lane.isLt
    exact shared _ (by simp only [LightAssembler.statesStart]; omega)
      (by simp only [LightAssembler.statesStart]; omega)
  have statementEq : FinishWalk.finishStatement (refreshInput (logicalWidth := logicalWidth)
      (publicFits := publicFits) program completed) = statement := by
    show (⟨(refreshInput (logicalWidth := logicalWidth) (publicFits := publicFits) program
        completed).iteration, (refreshInput (logicalWidth := logicalWidth)
        (publicFits := publicFits) program completed).z0, (refreshInput
        (logicalWidth := logicalWidth) (publicFits := publicFits) program completed).zi⟩ :
          TerminalStatement AppState) = statement
    rw [iterationEq, z0Eq, ziEq]
  have decodedFinish := FinishRelation.constraints_finish program relation ajtai circuitKeys
    completed constraintsHold (by rw [iterationEq]; exact Nat.pos_iff_ne_zero.mp positive)
  rw [keysEq] at decodedFinish
  have nextEq : ∀ slot, (finishStepOutput (logicalWidth := logicalWidth) (publicFits := publicFits)
      program 1 completed).runningNext slot = runningNext slot := by
    intro slot
    have slotBound := slot.isLt
    show ((RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits)
      base program.witnessWordCount).next slot).value completed = _
    apply SlotExpr.value_eq_of_words _ completed _ (lt_of_le_of_lt (nextDepthLe slot) small)
    refine Eq.trans ?_ (nextWords slot)
    apply slotAt_evalList
    intro column atLeast below
    simp only [LightAssembler.slotsStart, LightAssembler.witnessStart,
      LightAssembler.slotWordCount] at atLeast below
    exact shared column (by omega) (by omega)
  refine ⟨completed, Function.update payload.runningWitness selected foldedWitness, ?_,
    packageRows, packageBlocks, keysEq, statementEq, decodedFinish, fun slot => ?_⟩
  · rw [kept 0 (by have := complete.sharedBelow; omega)]
    have atMarker := congrFun (publicInput_withPublic (publicFits := publicFits) output.x env)
      encHashMarkerIndex
    rw [encHash_marker] at atMarker
    simpa [LogicalPlan.publicInput, encHashMarkerIndex] using atMarker
  · rw [nextEq slot]
    by_cases same : slot = selected
    · rw [same, nextSelected, Function.update_self]
      exact ⟨foldedHolds, normBounded_mono foldedNorm (by
        have := accumulatorBound_mono (Nat.zero_le Schedule.productionLightFoldCap)
        omega), foldedSupported _ (honest.support selected) (honest.freshSupport pcValid)⟩
    · rw [nextOther slot same, Function.update_of_ne same]
      exact ⟨slots slot, normBounded_mono (honest.norm slot) (by
        have := accumulatorBound_mono (honest.depth slot)
        omega), honest.support slot⟩

end Finish

/-! ### Every valid run -/

variable {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)}
  {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
  {step : AppState → AppWitness → AppState}
  {cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}

/-- Completeness of the finish package along a run: for every valid run of at least one step
from `z0` whose witnesses have the application's witness length, the honest finish of the honest
prover's proof gives finish columns that the final program's checks accept for the run's
statement. -/
theorem run_finish_complete (exports : Exports relation ajtai circuitKeys step cubeFits)
    (complete : Complete exports) (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (cap : Nat) (control : AppState → AppWitness → Fin 2) (z0 : AppState)
    (z0Length : z0.length = Stage1.Application.stateWordCount) (firstFresh width : Nat)
    (batches : List WitnessBatch) (application : Nat × Nat × Nat × Nat × Nat)
    (finish : FinishComplete exports.program relation firstFresh) :
    ∀ witnesses : List AppWitness, IvcWalk.ValidFrom exports.program z0 witnesses →
      (∀ witness ∈ witnesses, witness.length = exports.program.witnessWordCount) →
      0 < witnesses.length → witnesses.length < goldilocksModulus →
        FinishAccepts exports.program relation ajtai circuitKeys firstFresh width batches
          application complete.packageWidth ⟨witnesses.length, z0, witnesses.foldl step z0⟩ := by
  intro witnesses valid lengths nonempty bound
  rcases RunCompleteness.run_complete exports complete keysShaped cap control z0 z0Length witnesses
    valid lengths bound with bottom | ⟨payload, holds, honest⟩
  · obtain ⟨-, iterationZero, -⟩ :=
      (Terminal.holdsFor_bottom_iff cap ajtai circuitKeys step control _).mp bottom
    exact absurd iterationZero (Nat.pos_iff_ne_zero.mp nonempty)
  · exact finish_complete exports.program relation ajtai circuitKeys keysShaped firstFresh width
      batches application finish cap step control _ payload holds complete.packageWidth honest

end NightstreamFPrime.Export.OptionB.FinishCompleteness
