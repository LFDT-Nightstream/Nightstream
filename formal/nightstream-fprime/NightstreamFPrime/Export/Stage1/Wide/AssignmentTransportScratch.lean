import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCorrectness
import NightstreamFPrime.Export.Stage1.PiRLCCombinationDirectWitness

/-! The serialized wide assignment never reads ring-product scratch.
The interval is the relocated allocation used by the existing compact
output-only executor; its required output columns remain live. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportScratch

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation CanonicalBlockAssignment
open Layout.Stage1.Wide
open AssignmentTransport AssignmentTransportExecution AssignmentTransportSemantics

def scratchStart : Nat := SourceOrder.column PiRLCStarts.commitmentFreshStart
def scratchEnd : Nat := SourceOrder.column PiRLCStarts.outputFreshStart

theorem scratchStart_eq : scratchStart = 19646884 := by
  change SourceOrder.column 19647162 = 19646884
  rw [SourceOrder.column_late _ (by decide) (by decide)]

theorem scratchEnd_eq : scratchEnd = 27495784 := by
  change SourceOrder.column 27496062 = 27495784
  rw [SourceOrder.column_late _ (by decide) (by decide)]

def Outside (column : Nat) : Prop := column < scratchStart ∨ scratchEnd ≤ column

def Agree (left right : Env) : Prop := ∀ column, Outside column → left column = right column

def erase (env : Env) : Env :=
  PiRLCCombinationDirectWitness.eraseScratch scratchStart (scratchEnd - scratchStart) env

theorem erase_agree (env : Env) : Agree (erase env) env := by
  intro column outside
  unfold erase PiRLCCombinationDirectWitness.eraseScratch
  rw [if_neg]
  simp only [Outside, scratchStart_eq, scratchEnd_eq] at outside ⊢
  omega

theorem source_column_outside (source : Nat) (bounded : source < SourceOrder.sourceWidth)
    (outside : source < PiRLCStarts.commitmentFreshStart ∨ PiRLCStarts.outputFreshStart ≤ source) :
    Outside (SourceOrder.column source) := by
  by_contra inside
  simp only [Outside, scratchStart_eq, scratchEnd_eq] at inside
  have inverse := Layout.Stage1.Spartan.spartanToSource_sourceToSpartan source
    (lt_of_lt_of_le bounded SourceOrder.sourceWidth_le_reference)
  have expanded := SourceOrder.expand_relocate _ (SourceOrder.reference_region source bounded)
  change SourceOrder.expand (SourceOrder.column source) = Layout.Stage1.Spartan.sourceToSpartan source at expanded
  rw [← expanded, SourceOrder.expand, if_neg (by rw [SourceOrder.privateColumns_eq]; omega)] at inverse
  unfold Layout.Stage1.Spartan.spartanToSource at inverse
  rw [if_neg (by change ¬SourceOrder.column source < 98786; omega),
    if_neg (by change ¬SourceOrder.column source < 128074; omega),
    if_neg (by change ¬SourceOrder.column source < 14751526; omega),
    if_pos (by rw [Layout.Stage1.Spartan.privateColumnCount_eq]; omega)] at inverse
  have coordinate := Option.some.inj inverse
  change 14751804 + (SourceOrder.column source - 14751526) = source at coordinate
  change source < 19647162 ∨ 27496062 ≤ source at outside
  omega

private theorem source_view_outside (source target : Nat)
    (mapped : SourceAssignment.source? source = some target) :
    target < PiRLCStarts.commitmentFreshStart ∨ PiRLCStarts.outputFreshStart ≤ target := by
  change (if source < 19513117 then some source
    else if 19776685 ≤ source ∧ source < 19829011 then some (19568520 + (source - 19776685))
    else if 28421542 ≤ source ∧ source < 28785018 then some (27496062 + (source - 28421542))
    else none) = some target at mapped
  change target < 19647162 ∨ 27496062 ≤ target
  split_ifs at mapped <;> simp only [Option.some.injEq] at mapped <;> omega

/-- Strict retained-source relocation cannot select a discarded product cell. -/
theorem finalColumn_outside (source target : Nat) (mapped : finalColumn source = .ok target) : Outside target := by
  by_cases suffix : Layout.Stage1.Spartan.privateColumnCount ≤ source
  · rw [finalColumn, if_pos suffix] at mapped
    have same := Except.ok.inj mapped
    rw [SourceOrder.privateColumns_eq] at same
    right
    rw [scratchEnd_eq]
    omega
  · obtain ⟨original, current, _, selected, column, _⟩ :=
      AssignmentTransportCommonSource.finalColumn_private source target (Nat.lt_of_not_ge suffix) mapped
    rw [← column]
    exact source_column_outside current (SourceAssignment.source?_lt original current selected)
      (source_view_outside original current selected)

private theorem sampler_source_outside (index : Fin (34 * 86)) :
    Outside (AffineRuns.sourceAt samplerPoseidon.sources index.val) := by
  rw [samplerPoseidon, Values.ofSource_source]
  let invocation := index.val / 86
  let start := if invocation % 2 = 0 then PiRLCStarts.entryLogicalStart (invocation / 2)
    else PiRLCStarts.advanceLogicalStart (invocation / 2)
  have sourceBound : invocation / 2 < 17 := by dsimp only [invocation]; omega
  have lower : 14751804 ≤ start := by
    dsimp only [start]
    split
    · change 14751804 ≤ 19513117 + invocation / 2 * 3205; omega
    · change 14751804 ≤ 19513117 + invocation / 2 * 3205 + 592 + 2021; omega
  have upper : start + 592 < 19647162 := by
    dsimp only [start]
    split
    · change 19513117 + invocation / 2 * 3205 + 592 < 19647162; omega
    · change 19513117 + invocation / 2 * 3205 + 592 + 2021 + 592 < 19647162; omega
  left
  change SourceOrder.column start + (PoseidonRetainedSlots.localOutput _).val < scratchStart
  rw [SourceOrder.column_late start lower (by rw [SourceOrder.sourceWidth_eq]; omega), scratchStart_eq]
  have localBound : (PoseidonRetainedSlots.localOutput
      ⟨index.val % 86, by rw [PoseidonRetainedSlots.rows_length]; omega⟩).val < 592 :=
    (PoseidonRetainedSlots.localOutput _).isLt
  omega

private theorem range_source_outside (source : Fin 17) (block : LowNormBlock.Block 2025)
    (slot : Fin block.slotCount) : Outside (AffineRuns.sourceAt (rangeBlock source.val block).sources slot.val) := by
  rw [rangeBlock_source]
  have sourceBound : source.val < 17 := source.isLt
  have cellBound : (block.source slot).val < 2025 := (block.source slot).isLt
  have lower : 14751804 ≤ PiRLCStarts.rangeLogicalStart source.val := by
    change 14751804 ≤ 19513117 + source.val * 3205 + 592; omega
  have upper : PiRLCStarts.rangeLogicalStart source.val + 2025 < 19647162 := by
    change 19513117 + source.val * 3205 + 592 + 2025 < 19647162; omega
  left
  rw [SourceOrder.column_late _ lower (by rw [SourceOrder.sourceWidth_eq]; omega), scratchStart_eq]
  omega

theorem challenge_source_outside (source : Fin 17) (lane : Fin ringDegree) (bit : Fin 3) :
    Outside (AffineRuns.sourceAt challenges (source.val * (ringDegree * 3) + lane.val * 3 + bit.val)) := by
  rw [challenge_source]
  apply source_column_outside
  · have s : source.val < 17 := source.isLt
    have l : lane.val < 54 := lane.isLt
    have b : bit.val < 3 := bit.isLt
    change 19513117 + source.val * 3205 + 592 + 1404 + 264 + 131 + 3 * lane.val + bit.val < 27859538
    omega
  · left
    have s : source.val < 17 := source.isLt
    have l : lane.val < 54 := lane.isLt
    have b : bit.val < 3 := bit.isLt
    change 19513117 + source.val * 3205 + 592 + 1404 + 264 + 131 + 3 * lane.val + bit.val < 19647162
    omega

private theorem common_source_outside (program : RetainedLayout.Program)
    (kind : PerApplicationAssignmentPlan.BlockKind) (block : Values)
    (emitted : commonBlock program kind = .ok block) (slot : Fin block.count) :
    Outside (AffineRuns.sourceAt block.sources slot.val) := by
  have geometry := commonBlock_geometry program kind block emitted
  have inside : slot.val < commonLimit program kind := by rw [← geometry.2]; exact slot.isLt
  exact finalColumn_outside _ _ (commonBlock_source program kind block emitted ⟨slot.val, inside⟩)

private theorem common_sources_outside (program : RetainedLayout.Program)
    (kinds : List PerApplicationAssignmentPlan.BlockKind) (blocks : List Values)
    (pairs : List.Forall₂ (fun kind block => commonBlock program kind = .ok block) kinds blocks)
    (block : Values) (member : block ∈ blocks) (slot : Fin block.count) :
    Outside (AffineRuns.sourceAt block.sources slot.val) := by
  induction pairs with
  | nil => cases member
  | @cons kind head kinds blocks emitted pairs ih =>
    rcases List.mem_cons.mp member with rfl | rest
    · exact common_source_outside program kind _ emitted slot
    · exact ih rest

theorem blocks_outside (program : RetainedLayout.Program) (physicalWidth : Nat) (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program physicalWidth = .ok plan)
    (block : Values) (member : block ∈ plan.blocks) (slot : Fin block.count)
    (physical : AffineRuns.sourceAt block.sources slot.val < physicalWidth) :
    Outside (AffineRuns.sourceAt block.sources slot.val) := by
  obtain ⟨common, outputs, commonEmitted, outputEmitted, blocks, _, _, _, _⟩ :=
    emitted_parts program physicalWidth plan emitted
  rw [blocks] at member
  simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with ((member | rfl) | member) | rfl | rfl
  · exact common_sources_outside program _ _ (PhysicalRelabel.mapM_pairs _ _ _ commonEmitted) block member slot
  · exact sampler_source_outside slot
  · obtain ⟨source, sourceMember, member⟩ := List.mem_flatMap.mp member
    have bound : source < 17 := List.mem_range.mp sourceMember
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl <;> exact range_source_outside ⟨source, bound⟩ _ slot
  · exact common_source_outside program .productOutput _ outputEmitted slot
  · have inside : slot.val < 52326 := slot.isLt
    simp only [AffineRuns.sourceAt, if_pos inside, Nat.one_mul] at physical
    omega

theorem pullback_eq (left right : Env) (agree : Agree left right) :
    finalMap.pullback left = finalMap.pullback right := by
  funext source
  cases mapped : finalColumn source with
  | error message => simp only [PhysicalRelabel.Map.pullback, finalMap, mapped]
  | ok target =>
    simp only [PhysicalRelabel.Map.pullback, finalMap, mapped]
    exact agree target (finalColumn_outside source target mapped)

theorem digitBit_eq (left right : Env) (agree : Agree left right) (plan : AssignmentTransport.Plan)
    (sources : plan.challengeSources = challenges) (source : Fin 17) (lane : Fin ringDegree) (bit : Fin 3) :
    digitBit plan left source lane bit = digitBit plan right source lane bit := by
  unfold digitBit
  rw [sources]
  exact agree _ (challenge_source_outside source lane bit)

theorem quotientValue_eq (program : RetainedLayout.Program) (physicalWidth : Nat) (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program physicalWidth = .ok plan)
    (left right : Env) (agree : Agree left right) (slot : Fin PiRLCProductSchedule.invocationCount) :
    quotientValue plan left slot = quotientValue plan right slot := by
  obtain ⟨_, _, _, _, _, _, _, sources, _⟩ := emitted_parts program physicalWidth plan emitted
  have bits := digitBit_eq left right agree plan sources
  have challenge : ∀ source, challengeRing plan left source = challengeRing plan right source := by
    intro source
    funext lane
    simp only [challengeRing, digit, bits]
  have values : ∀ descriptor, valueRing plan left descriptor = valueRing plan right descriptor := by
    intro descriptor
    funext lane
    rw [valueRing_relocated program physicalWidth plan emitted left,
      valueRing_relocated program physicalWidth plan emitted right, pullback_eq left right agree]
  simp only [quotientValue, challenge, values]

private theorem digest_values_eq (left right : Env) (agree : Agree left right)
    (before after : List Expr)
    (pairs : List.Forall₂ (fun original moved => finalMap.expression original = .ok moved) before after) :
    after.map (fun expression => expression.eval left) = after.map (fun expression => expression.eval right) := by
  induction pairs with
  | nil => rfl
  | @cons original moved before after emitted pairs ih =>
    simp only [List.map_cons, finalMap.expression_eval original moved emitted left,
      finalMap.expression_eval original moved emitted right, pullback_eq left right agree, ih]

theorem outputDigest_eq (program : RetainedLayout.Program) (physicalWidth : Nat) (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program physicalWidth = .ok plan)
    (left right : Env) (agree : Agree left right) : outputDigest plan left = outputDigest plan right := by
  obtain ⟨_, _, _, _, _, _, _, _, expressions⟩ := emitted_parts program physicalWidth plan emitted
  exact digest_values_eq left right agree _ _ (PhysicalRelabel.mapM_pairs _ _ _ expressions)

theorem executeUnchecked_eq (program : RetainedLayout.Program) (physicalWidth : Nat) (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program physicalWidth = .ok plan)
    (left right : Env) (agree : Agree left right) :
    executeUnchecked program plan physicalWidth left = executeUnchecked program plan physicalWidth right := by
  have quotient := quotientValue_eq program physicalWidth plan emitted left right agree
  have blockEq (block : Values) (member : block ∈ plan.blocks) :
      blockValue plan physicalWidth left block = blockValue plan physicalWidth right block := by
    unfold blockValue
    apply congrArg (ofBlock (identityBlock block))
    funext slot
    unfold sourceValue
    split
    · rename_i physical
      exact agree _ (blocks_outside program physicalWidth plan emitted block member slot physical)
    · split
      · exact quotient _
      · rfl
  have schedules := List.map_congr_left blockEq
  unfold executeUnchecked schedule
  rw [outputDigest_eq program physicalWidth plan emitted left right agree, schedules]

theorem execute_eq (program : RetainedLayout.Program) (physicalWidth : Nat) (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program physicalWidth = .ok plan)
    (left right : Env) (agree : Agree left right) :
    execute program plan physicalWidth left = execute program plan physicalWidth right := by
  obtain ⟨_, _, _, _, _, _, _, sources, _⟩ := emitted_parts program physicalWidth plan emitted
  have bits := digitBit_eq left right agree plan sources
  have checks : DigitsValid plan left = DigitsValid plan right := by
    simp only [DigitsValid, digit, bits]
  unfold execute
  simp only [checks, executeUnchecked_eq program physicalWidth plan emitted left right agree]

/-- Erasing the omitted physical interval preserves execution and all CCS
coordinates. The read condition is proved from the emitted plan itself. -/
theorem execute_erase (program : RetainedLayout.Program) (physicalWidth : Nat) (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program physicalWidth = .ok plan) (env : Env) :
    execute program plan physicalWidth (erase env) = execute program plan physicalWidth env :=
  execute_eq program physicalWidth plan emitted (erase env) env (erase_agree env)

end NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportScratch
