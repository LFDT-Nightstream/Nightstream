import NightstreamFPrime.Export.Stage1.PreparedPhysicalArray
import NightstreamFPrime.Export.Stage1.PreparedPhysicalPackageRows
import NightstreamFPrime.Export.Stage1.StoredPhysicalRowContext
import NightstreamFPrime.Export.Stage1.CompactPlanTemplateBounds
import NightstreamFPrime.Export.Stage1.PackagePlan
import NightstreamFPrime.Export.Stage1.WitnessPlan
import NightstreamFPrime.Export.Stage1.StoredPhysicalRowCheck
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Stage1.PerApplicationCachedShift
import NightstreamFPrime.Export.Stage1.PreparedPhysicalInputs
import NightstreamFPrime.Export.RowSemantics

/-!
Canonical physical preparation with final-state row coverage. Events are built
once in canonical source order. Execution sorts a copy; final checks use the
unsorted array. This proves row satisfaction from completed checks, not witness
execution order, caller parsing, or source IO provenance.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredPhysicalPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package

inductive Event where
  | hash (chain : HashChain) (ordinal : Nat)
  | permutation (invocation : PermutationInvocation)
  | compact (outputTarget : Nat) (invocation : CompactRowInvocation)
  | batch (value : WitnessBatch)
  | instruction (value : WitnessInstruction)

/-- The native assignment schedule orders these same global write targets. -/
def Event.target : Event → Nat
  | .hash chain ordinal => chain.witnessStart +
      ordinal * PilotData.poseidonSchedule.recipesPerPermutation
  | .permutation invocation => invocation.witnessStart
  | .compact outputTarget _ => outputTarget
  | .batch value => value.start
  | .instruction value => value.target

/-- Final-state equations checked for each prepared event. Batches carry write
recipes; the package gives them no separate row-satisfaction clause. -/
def Event.RowsHold (event : Event) (package : CircuitPackage) (env : Env) : Prop :=
  match event with
  | .hash chain ordinal => TemplateInvocationHolds package chain ordinal env
  | .permutation invocation => PermutationInvocationHolds package invocation env
  | .compact _ invocation => CompactRowInvocationHolds package invocation env
  | .batch _ => True
  | .instruction item => item.Holds env

def Event.check (event : Event) (package : CircuitPackage) (env : Env) : Bool :=
  match event with
  | .hash chain ordinal => StoredPhysicalRowCheck.hashInvocation package chain ordinal env
  | .permutation invocation => StoredPhysicalRowCheck.permutationInvocation package invocation env
  | .compact _ invocation => StoredPhysicalRowCheck.compactInvocation package invocation env
  | .batch _ => true
  | .instruction item => StoredPhysicalRowCheck.instruction item env

theorem Event.check_iff (event : Event) (package : CircuitPackage) (env : Env) :
    event.check package env = true ↔ event.RowsHold package env := by
  cases event with
  | hash chain ordinal => exact StoredPhysicalRowCheck.hashInvocation_iff package chain ordinal env
  | permutation invocation =>
      exact StoredPhysicalRowCheck.permutationInvocation_iff package invocation env
  | compact target invocation =>
      exact StoredPhysicalRowCheck.compactInvocation_iff package invocation env
  | batch value => simp [Event.check, Event.RowsHold]
  | instruction item => exact StoredPhysicalRowCheck.instruction_iff item env

open PreparedPhysicalArray (appendMap mem_appendMap)

local notation "selectedApplication" => Poseidon2HashChainV1Package.application
local notation "selectedShift" =>
  PerApplicationCachedShift.Context.ofProgram Poseidon2HashChainV1Package.application

structure Plan where
  pilot : { value : CircuitPackage // value = PilotData.circuitPackage () }
  layout : PhysicalLayout
  templates : Array CompactRowTemplate
  templates_eq : templates.toList = Data.compactRowTemplates ()
  rowEvents : Array Event
  events : Array Event
  assertions : Array SparseRow
  sound : ∀ env : Env,
    (∀ event ∈ rowEvents,
      event.check { pilot.val with compactRowTemplates := templates.toList } env = true) →
    (∀ row ∈ assertions, StoredPhysicalRowCheck.sparseRow row env = true) →
    (Poseidon2HashChainV1Package.package ()).RowsHold env

private structure Assembly where
  rowEvents : Array Event
  assertions : Array SparseRow

private def compactEvent (shift : PerApplicationCachedShift.Context)
    (templates : Array CompactRowTemplate)
    (canonical : templates.toList = Data.compactRowTemplates ())
    (block : PackagePlan.CompactInvocationBlock)
    (invocation : { value : CompactRowInvocation // value ∈ block.expand }) : Event :=
  let shifted := PerApplicationCachedShift.shiftCompactRowInvocation shift invocation.val
  let bounded : shifted.templateIndex < templates.size := by
    change invocation.val.templateIndex < templates.size
    have sizeEqual : templates.size = (Data.compactRowTemplates ()).length := by
      simpa using congrArg List.length canonical
    rw [sizeEqual]
    exact CompactPlanTemplateBounds.block_templateIndex_lt block invocation.val
      invocation.property
  let template := templates[shifted.templateIndex]'bounded
  .compact (compactInputColumn shifted.inputRanges template.outputInput) shifted

private def assemble (sources : PreparedPhysicalInputs.Inputs)
    (templates : Array CompactRowTemplate)
    (canonical : templates.toList = Data.compactRowTemplates ()) : Assembly :=
  let shift := selectedShift
  let application := PerApplicationPackage.directApplicationPlan selectedApplication
  let groups := sources.groups
  let batchGroups := [groups.initialClaim.batches, groups.sumcheck.batches,
    groups.evalK.batches, groups.evalA.batches, groups.ccs.batches,
    groups.norm.batches, groups.finalIdentity.batches]
  let events := [Data.priorChain, Data.outputChain].foldl (fun events chain =>
    let shifted := PerApplicationCachedShift.shiftHashChain shift chain
    appendMap events (List.range (shifted.absorbCount + 1)) (Event.hash shifted))
    (#[] : Array Event)
  let events := sources.permutations.blocks.foldl (fun events block =>
    appendMap events block.expand (fun invocation => Event.permutation
      (PerApplicationCachedShift.shiftPermutationInvocation shift invocation))) events
  let events := appendMap events application.permutationInvocations Event.permutation
  let events := PackagePlan.canonicalCompactBlocks.foldl (fun events block =>
    appendMap events block.expand.attach (compactEvent shift templates canonical block)) events
  let events := appendMap events (Data.liftPilotBatches (PilotData.priorWordBatches ()))
    (fun batch => Event.batch (PerApplicationCachedShift.shiftBatch shift batch))
  let events := batchGroups.foldl (fun events batches => appendMap events batches
    (fun batch => Event.batch (PerApplicationCachedShift.shiftBatch shift batch))) events
  let events := (WitnessPlan.canonicalBlocks Data.logicalWidth Data.publicFits).foldl
    (fun events block => appendMap events block.expand
      (fun batch => Event.batch (PerApplicationCachedShift.shiftBatch shift batch))) events
  let events := appendMap events application.witnessBatches Event.batch
  let events := appendMap events
    (Data.liftPilotInstructions (PilotData.witnessInstructions ()))
    (fun instruction => Event.instruction
      (PerApplicationCachedShift.shiftWitnessInstruction shift instruction))
  let events := sources.ordinary.blocks.foldl (fun events block =>
    appendMap events block.witnessInstructions (fun instruction => Event.instruction
      (PerApplicationCachedShift.shiftWitnessInstruction shift instruction))) events
  let events := appendMap events application.witnessInstructions Event.instruction
  let assertions := appendMap (#[] : Array SparseRow)
    (Data.liftPilotRows (PilotData.assertionRows ()))
    (PerApplicationCachedShift.shiftSparseRow shift)
  let assertions := sources.ordinary.blocks.foldl (fun assertions block =>
    appendMap assertions block.assertionRows
      (PerApplicationCachedShift.shiftSparseRow shift)) assertions
  let assertions := appendMap assertions application.assertionRows id
  let assertions := appendMap assertions
    (NextPreimagePackage.assertionRows
      (PerApplicationPackage.nextPreimageRowStart selectedApplication)) id
  { rowEvents := events, assertions := assertions }

private theorem all_appendMap {Alpha Beta : Type} (initial : Array Beta)
    (items : List Alpha) (transform : Alpha → Beta) (predicate : Beta → Prop) :
    (∀ value ∈ appendMap initial items transform, predicate value) ↔
      (∀ value ∈ initial, predicate value) ∧
        ∀ item ∈ items, predicate (transform item) := by
  constructor
  · intro checked
    constructor
    · intro value member
      exact checked value ((mem_appendMap initial items transform value).mpr (Or.inl member))
    · intro item member
      exact checked (transform item)
        ((mem_appendMap initial items transform (transform item)).mpr
          (Or.inr ⟨item, member, rfl⟩))
  · rintro ⟨priorChecks, mapped⟩ value member
    rcases (mem_appendMap initial items transform value).mp member with earlier |
      ⟨item, sourceMember, rfl⟩
    · exact priorChecks value earlier
    · exact mapped item sourceMember

private theorem all_foldl_appendMap {Alpha Gamma : Type} {Beta : Alpha → Type}
    (initial : Array Gamma) (items : List Alpha)
    (children : (item : Alpha) → List (Beta item))
    (transform : (item : Alpha) → Beta item → Gamma) (predicate : Gamma → Prop) :
    (∀ value ∈ items.foldl (fun accumulated item =>
      appendMap accumulated (children item) (transform item)) initial, predicate value) ↔
      (∀ value ∈ initial, predicate value) ∧
        ∀ item ∈ items, ∀ child ∈ children item, predicate (transform item child) := by
  induction items generalizing initial with
  | nil => simp
  | cons item rest inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis, all_appendMap]
      constructor
      · rintro ⟨⟨priorChecks, current⟩, remaining⟩
        refine ⟨priorChecks, ?_⟩
        intro selected member
        rcases List.mem_cons.mp member with rfl | member
        · exact current
        · exact remaining selected member
      · rintro ⟨priorChecks, checked⟩
        refine ⟨⟨priorChecks, checked item (by simp)⟩, ?_⟩
        intro selected member
        exact checked selected (List.mem_cons_of_mem item member)

private theorem assemble_sound (sources : PreparedPhysicalInputs.Inputs)
    (templates : Array CompactRowTemplate)
    (canonical : templates.toList = Data.compactRowTemplates ()) (env : Env)
    (eventChecks : ∀ event ∈ (assemble sources templates canonical).rowEvents,
      event.check { sources.pilot.val with compactRowTemplates := templates.toList } env = true)
    (assertionChecks : ∀ row ∈ (assemble sources templates canonical).assertions,
      StoredPhysicalRowCheck.sparseRow row env = true) :
    (Poseidon2HashChainV1Package.package ()).RowsHold env := by
  let header : CircuitPackage :=
    { sources.pilot.val with compactRowTemplates := templates.toList }
  have headerEqual : header =
      { PilotData.circuitPackage () with compactRowTemplates := templates.toList } := by
    dsimp only [header]
    rw [sources.pilot.property]
  change ∀ event ∈ (assemble sources templates canonical).rowEvents,
    event.check header env = true at eventChecks
  simp only [assemble, all_appendMap, all_foldl_appendMap] at eventChecks assertionChecks
  rcases eventChecks with ⟨e10, applicationInstructionChecks⟩
  rcases e10 with ⟨e9, ordinaryInstructionChecks⟩
  rcases e9 with ⟨e8, pilotInstructionChecks⟩
  rcases e8 with ⟨e7, _applicationBatchChecks⟩
  rcases e7 with ⟨e6, _witnessBlockBatchChecks⟩
  rcases e6 with ⟨e5, _groupBatchChecks⟩
  rcases e5 with ⟨e4, _pilotBatchChecks⟩
  rcases e4 with ⟨e3, compactChecks⟩
  rcases e3 with ⟨e2, _applicationPermutationChecks⟩
  rcases e2 with ⟨e1, permutationChecks⟩
  rcases e1 with ⟨_emptyEventChecks, hashChecks⟩
  rcases assertionChecks with ⟨a3, nextPreimageChecks⟩
  rcases a3 with ⟨a2, applicationAssertionChecks⟩
  rcases a2 with ⟨a1, ordinaryAssertionChecks⟩
  rcases a1 with ⟨_emptyAssertionChecks, pilotAssertionChecks⟩
  have ordinaryRows := PreparedPhysicalOrdinary.holds sources.ordinary
    (fun column => env (PerApplicationPackage.shiftColumn selectedApplication column))
    (by
      intro block blockMember instruction member
      have held := (StoredPhysicalRowCheck.instruction_iff _ env).mp
        (ordinaryInstructionChecks block blockMember instruction member)
      simpa only [PerApplicationCachedShift.shiftWitnessInstruction_eq,
        PerApplicationPackage.shiftWitnessInstruction_holds] using! held)
    (by
      intro block blockMember row member
      have held := (StoredPhysicalRowCheck.sparseRow_iff _ env).mp
        (ordinaryAssertionChecks block blockMember row member)
      simpa only [PerApplicationCachedShift.shiftSparseRow_eq,
        PerApplicationPackage.shiftSparseRow_holds] using! held)
  apply PreparedPhysicalPackageRows.rowsHold env
  · intro chain member ordinal bounded
    apply (StoredPhysicalRowContext.hashInvocation_iff header templates headerEqual
      _ ordinal env).mp
    exact hashChecks chain member ordinal (List.mem_range.mpr (by omega))
  · intro block blockMember invocation member
    have actualMember : block ∈ sources.permutations.blocks := by
      rw [sources.permutations.blocks_eq]
      exact blockMember
    apply (StoredPhysicalRowContext.permutationInvocation_iff header templates
      headerEqual _ env).mp
    exact permutationChecks block actualMember invocation member
  · intro block blockMember invocation member
    have attached : (⟨invocation, member⟩ :
        { value : CompactRowInvocation // value ∈ block.expand }) ∈ block.expand.attach := by
      simp
    have checked := compactChecks block blockMember ⟨invocation, member⟩ attached
    change StoredPhysicalRowCheck.compactInvocation header
      (PerApplicationCachedShift.shiftCompactRowInvocation selectedShift invocation) env = true
      at checked
    exact (StoredPhysicalRowContext.compactInvocation_iff header templates headerEqual
      canonical _ env).mp checked
  · constructor
    · intro instruction member
      exact (StoredPhysicalRowCheck.instruction_iff _ env).mp
        (pilotInstructionChecks instruction member)
    · intro row member
      exact (StoredPhysicalRowCheck.sparseRow_iff _ env).mp (pilotAssertionChecks row member)
  · constructor
    · intro instruction member
      simpa only [PerApplicationCachedShift.shiftWitnessInstruction_eq,
        PerApplicationPackage.shiftWitnessInstruction_holds] using! ordinaryRows.1 instruction member
    · intro row member
      simpa only [PerApplicationCachedShift.shiftSparseRow_eq,
        PerApplicationPackage.shiftSparseRow_holds] using! ordinaryRows.2 row member
  · constructor
    · intro instruction member
      exact (StoredPhysicalRowCheck.instruction_iff _ env).mp
        (applicationInstructionChecks instruction member)
    · intro row member
      exact (StoredPhysicalRowCheck.sparseRow_iff _ env).mp
        (applicationAssertionChecks row member)
  · intro row member
    exact (StoredPhysicalRowCheck.sparseRow_iff _ env).mp (nextPreimageChecks row member)

/-- Construct each event once. Final checks retain canonical source order;
only the execution array is sorted by the existing write-target comparison. -/
def ofSources (sources : PreparedPhysicalInputs.Inputs) : Plan :=
  let templates := (Data.compactRowTemplates ()).toArray
  let canonical : templates.toList = Data.compactRowTemplates () := List.toList_toArray
  let assembled := assemble sources templates canonical
  { pilot := sources.pilot
    layout := PerApplicationPackage.directFinalLayout selectedApplication
    templates := templates
    templates_eq := canonical
    rowEvents := assembled.rowEvents
    events := assembled.rowEvents.qsort (fun left right => decide (left.target < right.target))
    assertions := assembled.assertions
    sound := assemble_sound sources templates canonical }

/-- Concrete assembly coverage, independent of the execution sort or schedule. -/
theorem ofSources_rowsHold (sources : PreparedPhysicalInputs.Inputs) (env : Env)
    (eventChecks : ∀ event ∈ (ofSources sources).rowEvents,
      event.check { (ofSources sources).pilot.val with
        compactRowTemplates := (ofSources sources).templates.toList } env = true)
    (assertionChecks : ∀ row ∈ (ofSources sources).assertions,
      StoredPhysicalRowCheck.sparseRow row env = true) :
    (Poseidon2HashChainV1Package.package ()).RowsHold env := by
  exact (ofSources sources).sound env eventChecks assertionChecks

def prepare : IO Plan := do
  let sources ← PreparedPhysicalInputs.prepare
  return ofSources sources

end NightstreamFPrime.Export.Stage1.StoredPhysicalPlan
