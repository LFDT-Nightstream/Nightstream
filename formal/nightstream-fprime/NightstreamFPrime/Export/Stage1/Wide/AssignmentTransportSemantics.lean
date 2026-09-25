import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportExecution
import NightstreamFPrime.Layout.Stage1.Wide.StepPhysicalCompleteness

/-! Value preservation for the exact schema-4 plan and its constructed
physical input. The source permutation and application insertion are
explicit; range auxiliary values come from the actual completion program. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportSemantics

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open ProductionRelation Layout.Stage1.Wide
open AssignmentTransport AssignmentTransportExecution
open Spec.Folding.PiCCS.PaperJoint

private theorem assemble_parts
    (commonBuild : Except String (List Values)) (outputBuild : Except String Values)
    (valueBuild : Except String (List AffineRuns.Run)) (expressionBuild : Except String (List Expr))
    (poseidon quotient : Values) (ranges : List Values)
    (families : List PerApplicationAssignmentTransport.Phi81FamilyShape) (challengeRuns : List AffineRuns.Run)
    (result : AssignmentTransport.Plan)
    (emitted : (do
      let common ← commonBuild
      let outputs ← outputBuild
      return { blocks := common ++ [poseidon] ++ ranges ++ [outputs, quotient]
               families := families
               valueSources := ← valueBuild
               challengeSources := challengeRuns
               outputDigestExpressions := ← expressionBuild } : Except String AssignmentTransport.Plan) = .ok result) :
    ∃ common outputs,
      commonBuild = .ok common ∧ outputBuild = .ok outputs ∧
      result.blocks = common ++ [poseidon] ++ ranges ++ [outputs, quotient] ∧
      result.families = families ∧ valueBuild = .ok result.valueSources ∧
      result.challengeSources = challengeRuns ∧ expressionBuild = .ok result.outputDigestExpressions := by
  cases commonBuild with
  | error message => simp [Bind.bind, Except.bind] at emitted
  | ok common =>
    cases outputBuild with
    | error message => simp [Bind.bind, Except.bind] at emitted
    | ok outputs =>
      cases valueBuild with
      | error message => simp [Bind.bind, Except.bind] at emitted
      | ok values =>
        cases expressionBuild with
        | error message => simp [Bind.bind, Except.bind] at emitted
        | ok expressions =>
          simp only [Bind.bind, Except.bind, Pure.pure, Except.pure, Except.ok.injEq] at emitted
          subst result
          exact ⟨common, outputs, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem emitted_parts (program : AssignmentTransport.Program) (physicalWidth : Nat)
    (result : AssignmentTransport.Plan) (emitted : AssignmentTransport.plan program physicalWidth = .ok result) :
    ∃ common outputs,
      commonKinds.mapM (commonBlock program) = .ok common ∧
      commonBlock program .productOutput = .ok outputs ∧
      result.blocks = common ++ [samplerPoseidon] ++ samplerRanges ++
        [outputs, ⟨.field, 52326, [⟨physicalWidth, 1, 52326⟩]⟩] ∧
      result.families = PerApplicationAssignmentTransport.phi81FamilyShapes ∧
      moveRuns (PerApplicationAssignmentTransport.phi81ValueSources program) = .ok result.valueSources ∧
      result.challengeSources = challenges ∧
      (PerApplicationAssignmentTransport.outputDigestExpressions program).mapM finalMap.expression =
        .ok result.outputDigestExpressions := by
  exact assemble_parts (commonKinds.mapM (commonBlock program)) (commonBlock program .productOutput)
    (moveRuns (PerApplicationAssignmentTransport.phi81ValueSources program))
    ((PerApplicationAssignmentTransport.outputDigestExpressions program).mapM finalMap.expression)
    samplerPoseidon ⟨.field, 52326, [⟨physicalWidth, 1, 52326⟩]⟩ samplerRanges
    PerApplicationAssignmentTransport.phi81FamilyShapes challenges result emitted

theorem commonBlock_geometry (program : AssignmentTransport.Program)
    (kind : PerApplicationAssignmentPlan.BlockKind) (block : Values)
    (emitted : commonBlock program kind = .ok block) :
    block.kind = (PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind).slotKind ∧
      block.count = commonLimit program kind := by
  let original := PerApplicationAssignmentBlocks.BlockPlan.ofKind program kind
  have emitted := (commonBlock_checks program kind block emitted).2
  change (Values.mk original.slotKind (commonLimit program kind) <$>
    moveRuns (takeRuns (commonLimit program kind) original.sourceRuns)) = .ok block at emitted
  cases runs : moveRuns (takeRuns (commonLimit program kind) original.sourceRuns) with
  | error message => simp [runs] at emitted
  | ok moved =>
    simp only [runs, Except.map_ok, Except.ok.injEq] at emitted
    subst block
    exact ⟨rfl, rfl⟩

theorem invocationIndex_eq (result : AssignmentTransport.Plan)
    (families : result.families = PerApplicationAssignmentTransport.phi81FamilyShapes)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    invocationIndex result descriptor = descriptor.invocation.val := by
  rw [PiRLCProductSchedule.Descriptor.invocation_val]
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp only [invocationIndex, familyOffset, familyCount, familyShape, families, familyOrdinal,
      PerApplicationAssignmentTransport.phi81FamilyShapes, List.getD_cons_zero, List.getD_cons_succ,
      PiRLCProductSchedule.Family.blockCount, PiRLCProductSchedule.Family.cellCount, ringDegree]
  all_goals omega

theorem valueSources_count (program : AssignmentTransport.Program) :
    ((PerApplicationAssignmentTransport.phi81ValueSources program).map AffineRuns.Run.count).sum =
      PiRLCProductSchedule.invocationCount := by
  rw [← AffineRuns.expand_length, PerApplicationAssignmentTransport.phi81ValueSources,
    AffineRuns.expand_compress, List.length_ofFn]

/-- Exact operand lookup after the checked physical relocation. -/
theorem valueRing_relocated (program : AssignmentTransport.Program) (physicalWidth : Nat)
    (result : AssignmentTransport.Plan) (emitted : AssignmentTransport.plan program physicalWidth = .ok result)
    (physical : Env) (descriptor : PiRLCProductSchedule.Descriptor) (lane : Fin ringDegree) :
    valueRing result physical descriptor lane =
      finalMap.pullback physical
        (PiRLCProductPlan.valueColumn program (descriptor.withLane lane) lane).val := by
  obtain ⟨_, _, _, _, _, families, values, _, _⟩ := emitted_parts program physicalWidth result emitted
  unfold valueRing
  rw [invocationIndex_eq result families]
  have read := moveRuns_value (PerApplicationAssignmentTransport.phi81ValueSources program)
    result.valueSources values physical (descriptor.withLane lane).invocation.val
    (by rw [valueSources_count]; exact (descriptor.withLane lane).invocation.isLt)
  rw [PerApplicationAssignmentTransport.phi81ValueSources_at,
    PiRLCProductSchedule.descriptor_invocation] at read
  exact read

def prefixValues (env : Env) : Env := fun column =>
  if column = SourceOrder.constantColumn then 1
  else ((Layout.Stage1.Spartan.spartanToSource (SourceOrder.expand column)).map env).getD 0

theorem prefixValues_source (env : Env) (source : Nat) (bounded : source < SourceOrder.sourceWidth) :
    prefixValues env (SourceOrder.column source) = env source := by
  unfold prefixValues
  rw [if_neg (SourceOrder.column_ne_constant source bounded)]
  unfold SourceOrder.column
  rw [SourceOrder.expand_relocate _ (SourceOrder.reference_region source bounded),
    Layout.Stage1.Spartan.spartanToSource_sourceToSpartan source
      (lt_of_lt_of_le bounded SourceOrder.sourceWidth_le_reference)]
  rfl

def physicalValues (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F) : Env := fun column =>
  if before : column < SourceOrder.privateColumns then prefixValues env column
  else if inside : column < SourceOrder.privateColumns + PerApplicationPackage.addedPrivateColumnCount program then
    application ⟨column - SourceOrder.privateColumns, by omega⟩
  else prefixValues env (column - PerApplicationPackage.addedPrivateColumnCount program)

def shiftedColumn (program : AssignmentTransport.Program) (column : Nat) : Nat :=
  if column < SourceOrder.privateColumns then column
  else column + PerApplicationPackage.addedPrivateColumnCount program

theorem physicalValues_shifted (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F) (column : Nat) :
    physicalValues program env application (shiftedColumn program column) = prefixValues env column := by
  unfold shiftedColumn
  by_cases before : column < SourceOrder.privateColumns
  · rw [if_pos before, physicalValues, dif_pos before]
  · rw [if_neg before, physicalValues, dif_neg (by omega), dif_neg (by omega), Nat.add_sub_cancel_right]

theorem physicalValues_source (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (source : Nat) (bounded : source < SourceOrder.sourceWidth) :
    physicalValues program env application (shiftedColumn program (SourceOrder.column source)) = env source := by
  rw [physicalValues_shifted, prefixValues_source env source bounded]

theorem physicalValues_private_source (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (source : Nat) (bounded : source < SourceOrder.sourceWidth)
    (beforeConstant : SourceOrder.column source < SourceOrder.privateColumns) :
    physicalValues program env application (SourceOrder.column source) = env source := by
  have value := physicalValues_source program env application source bounded
  simpa only [shiftedColumn, if_pos beforeConstant] using value

/-- The source at a completed physical range cell equals the standalone
source consumed by the compact assignment constructor. -/
theorem completed_range_value (interface : PiRLC.Wide.Scalar.Interface) (source start : Nat) (env : Env)
    (inputs : PiRLC.Wide.Scalar.Assumptions interface start)
    (completed : PiRLC.Wide.Scalar.RangeCompleted interface source start env)
    (index : Nat) (bounded : index < Gadgets.Sampling.WideReduction.privateCount) :
    let draw : Fin 4 → F := fun lane =>
      ((PiRLC.Wide.Scalar.rangeInterface interface source start).source lane
        (PiRLC.Wide.Scalar.rangeOffset start)).eval env
    env (Gadgets.Sampling.WideReduction.Program.coreOffset (PiRLC.Wide.Scalar.rangeOffset start) + index) =
      PiRlcWideSampler.Witness.rangeValues draw (1408 + index) := by
  dsimp only
  rw [completed index bounded]
  unfold PiRlcWideSampler.Witness.rangeValues Gadgets.Sampling.WideReduction.replaceTemporary
  rw [if_neg (by change ¬(4 ≤ 1408 + index ∧ 1408 + index < 4 + 1404); omega)]
  apply Gadgets.Sampling.WideReduction.ProgramValues.completeEnv_retained_congr
    (PiRLC.Wide.Scalar.rangeInterface interface source start) PiRlcWideSampler.RangePlan.interface
    env (PiRlcWideSampler.Witness.inputEnv _) (PiRLC.Wide.Scalar.rangeOffset start) 4
    (PiRLC.Wide.Scalar.range_inputs interface source start inputs) (fun lane => lane.isLt) _ index bounded
  intro lane
  change _ = PiRlcWideSampler.Witness.inputEnv _ lane.val
  simp only [PiRlcWideSampler.Witness.inputEnv, dif_pos lane.isLt]

private theorem uniform_source (starts : List Nat) (width source position : Nat)
    (sourceBound : source < starts.length) (positionBound : position < width) :
    AffineRuns.sourceAt (starts.map (fun start => ⟨start, 1, width⟩)) (source * width + position) =
      starts.getD source 0 + position := by
  induction starts generalizing source with
  | nil => simp at sourceBound
  | cons first rest ih =>
    cases source with
    | zero => simp [AffineRuns.sourceAt, positionBound]
    | succ source =>
      simp only [List.map_cons, AffineRuns.sourceAt, Nat.succ_mul]
      rw [if_neg (by omega)]
      have subtraction : source * width + width + position - width = source * width + position := by omega
      rw [subtraction, ih source (by simp only [List.length_cons] at sourceBound; omega), List.getD_cons_succ]

/-- Each challenge source addresses a checked digit bit, in source/lane/bit order. -/
theorem challenge_source (source : Fin 17) (lane : Fin ringDegree) (bit : Fin 3) :
    AffineRuns.sourceAt challenges (source.val * (ringDegree * 3) + lane.val * 3 + bit.val) =
      SourceOrder.column (Gadgets.Sampling.WideReduction.digitStart
        (Gadgets.Sampling.WideReduction.Program.coreOffset (PiRLCStarts.rangeLogicalStart source.val)) +
          3 * lane.val + bit.val) := by
  let start := fun scalar => SourceOrder.column (Gadgets.Sampling.WideReduction.digitStart
    (Gadgets.Sampling.WideReduction.Program.coreOffset (PiRLCStarts.rangeLogicalStart scalar)))
  have localBound : lane.val * 3 + bit.val < 162 := by
    have a : lane.val < 54 := lane.isLt
    have b : bit.val < 3 := bit.isLt
    omega
  have read := uniform_source ((List.range 17).map start) 162 source.val (lane.val * 3 + bit.val)
    (by simpa using source.isLt) localBound
  have selected : ((List.range 17).map start).getD source.val 0 = start source.val := by
    rw [List.getD_eq_get _ _ ⟨source.val, by simpa using source.isLt⟩]
    simp
  rw [selected] at read
  change AffineRuns.sourceAt ((List.range 17).map (fun scalar => ⟨start scalar, 1, 162⟩))
      (source.val * 162 + lane.val * 3 + bit.val) = _
  rw [show source.val * 162 + lane.val * 3 + bit.val = source.val * 162 + (lane.val * 3 + bit.val) by omega]
  rw [show (List.range 17).map (fun scalar => (⟨start scalar, 1, 162⟩ : AffineRuns.Run)) =
      ((List.range 17).map start).map (fun first => ⟨first, 1, 162⟩) by simp only [List.map_map]; rfl, read]
  have lower : 14751804 ≤ Gadgets.Sampling.WideReduction.digitStart
      (Gadgets.Sampling.WideReduction.Program.coreOffset (PiRLCStarts.rangeLogicalStart source.val)) := by
    change 14751804 ≤ 19513117 + source.val * 3205 + 592 + 1404 + 264 + 131
    omega
  have upper : Gadgets.Sampling.WideReduction.digitStart
      (Gadgets.Sampling.WideReduction.Program.coreOffset (PiRLCStarts.rangeLogicalStart source.val)) +
        (3 * lane.val + bit.val) < SourceOrder.sourceWidth := by
    have sourceBound : source.val < 17 := source.isLt
    change 19513117 + source.val * 3205 + 592 + 1404 + 264 + 131 + (3 * lane.val + bit.val) < 27859538
    omega
  rw [show Gadgets.Sampling.WideReduction.digitStart
      (Gadgets.Sampling.WideReduction.Program.coreOffset (PiRLCStarts.rangeLogicalStart source.val)) +
        3 * lane.val + bit.val =
      Gadgets.Sampling.WideReduction.digitStart
        (Gadgets.Sampling.WideReduction.Program.coreOffset (PiRLCStarts.rangeLogicalStart source.val)) +
        (3 * lane.val + bit.val) by omega, SourceOrder.column_late_add _ _ lower upper]
  dsimp only [start]
  omega

private theorem range_rows (draw : Fin 4 → F) :
    holds (PiRlcWideSampler.Witness.rangeValues draw)
      (Gadgets.Sampling.WideReduction.operations
        (Gadgets.Sampling.WideReduction.Program.coreInterface PiRlcWideSampler.RangePlan.interface 4)
        (Gadgets.Sampling.WideReduction.HintProgram.resultHints 4) 1408) := by
  apply holdsFlat_implies_holds
  have rows := PiRlcWideSampler.Witness.rangeValues_rows draw
  rw [PiRlcWideSampler.RangePlan.constraints, Gadgets.Sampling.WideReduction.Program.constraints_eq] at rows
  exact rows

theorem range_digit_bit (draw : Fin 4 → F) (lane : Fin ringDegree) (bit : Fin 3) :
    PiRlcWideSampler.Witness.rangeValues draw (1803 + 3 * lane.val + bit.val) = 0 ∨
      PiRlcWideSampler.Witness.rangeValues draw (1803 + 3 * lane.val + bit.val) = 1 := by
  have localBound : 395 + 3 * lane.val + bit.val < Gadgets.Sampling.WideReduction.privateCount := by
    have a : lane.val < 54 := lane.isLt
    have b : bit.val < 3 := bit.isLt
    change 395 + 3 * lane.val + bit.val < 617
    omega
  have value := Gadgets.Sampling.WideReduction.retained_bit_le_one
    (Gadgets.Sampling.WideReduction.Program.coreInterface PiRlcWideSampler.RangePlan.interface 4)
    (Gadgets.Sampling.WideReduction.HintProgram.resultHints 4)
    (PiRlcWideSampler.Witness.rangeValues draw) 1408 (395 + 3 * lane.val + bit.val)
    (fun lane => by change lane.val < 1408; have h : lane.val < 4 := lane.isLt; omega) (range_rows draw) localBound
    (by intro impossible; change 395 + 3 * lane.val + bit.val < 264 at impossible; omega)
  have column : 1408 + (395 + 3 * lane.val + bit.val) = 1803 + 3 * lane.val + bit.val := by omega
  rw [column] at value
  have alternatives : (PiRlcWideSampler.Witness.rangeValues draw (1803 + 3 * lane.val + bit.val)).val = 0 ∨
      (PiRlcWideSampler.Witness.rangeValues draw (1803 + 3 * lane.val + bit.val)).val = 1 := by omega
  exact alternatives.imp (fun zero => Fin.ext zero) (fun one => Fin.ext one)

theorem range_digit (draw : Fin 4 → F) (lane : Fin ringDegree) :
    Gadgets.Sampling.WideReduction.digitValue (PiRlcWideSampler.Witness.rangeValues draw) 1408 lane.val =
      (Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.sample draw lane).val := by
  have decoded := Gadgets.Sampling.WideReduction.soundness
    (Gadgets.Sampling.WideReduction.Program.coreInterface PiRlcWideSampler.RangePlan.interface 4)
    (Gadgets.Sampling.WideReduction.HintProgram.resultHints 4)
    (PiRlcWideSampler.Witness.rangeValues draw) 1408
    (fun lane => by change lane.val < 1408; have h : lane.val < 4 := lane.isLt; omega) (range_rows draw) lane
  have same : Gadgets.Sampling.WideReduction.drawOf
      (Gadgets.Sampling.WideReduction.Program.coreInterface PiRlcWideSampler.RangePlan.interface 4)
      (PiRlcWideSampler.Witness.rangeValues draw) 1408 = draw := by
    funext lane
    exact PiRlcWideSampler.Witness.rangeValues_input draw lane
  rw [same] at decoded
  exact decoded

end NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportSemantics
