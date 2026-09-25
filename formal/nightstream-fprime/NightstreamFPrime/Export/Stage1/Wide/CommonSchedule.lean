import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCommonSource
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportExecution
import NightstreamFPrime.Export.Stage1.Wide.PhysicalRelabel

/-! The emitted common blocks read the canonical common schedule. Successful
checked source maps supply every value; no source-read equality is assumed. -/

namespace NightstreamFPrime.Export.Stage1.Wide.CommonSchedule

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle ProductionRelation CanonicalBlockAssignment
open PerApplicationAssignmentPlan PerApplicationAssignmentBlocks PerApplicationCanonicalAssignment
open AssignmentTransport AssignmentTransportExecution AssignmentTransportSemantics

def physicalWidth (program : RetainedLayout.Program) : Nat :=
  Layout.Stage1.Wide.SourceOrder.totalColumns + PerApplicationPackage.addedPrivateColumnCount program

theorem finalColumn_lt (program : RetainedLayout.Program)
    (source target : Nat) (bounded : source < PiRLCProductPlan.baseSourceWidth program)
    (mapped : finalColumn source = .ok target) : target < physicalWidth program := by
  by_cases suffix : Layout.Stage1.Spartan.privateColumnCount ≤ source
  · rw [finalColumn, if_pos suffix] at mapped
    have same := Except.ok.inj mapped
    unfold PiRLCProductPlan.baseSourceWidth at bounded
    rw [PerApplicationPackage.package_totalColumnCount,
      PerApplicationPackage.basePackage_totalColumnCount_eq] at bounded
    rw [Layout.Stage1.Spartan.privateColumnCount_eq] at suffix same
    rw [Layout.Stage1.Wide.SourceOrder.privateColumns_eq] at same
    rw [physicalWidth, Layout.Stage1.Wide.SourceOrder.totalColumns_eq]
    omega
  · obtain ⟨_, _, _, _, _, bounded⟩ := AssignmentTransportCommonSource.finalColumn_private source target
      (Nat.lt_of_not_ge suffix) mapped
    rw [Layout.Stage1.Wide.SourceOrder.privateColumns_eq] at bounded
    rw [physicalWidth, Layout.Stage1.Wide.SourceOrder.totalColumns_eq]
    omega

private theorem domainValue_base (program : RetainedLayout.Program) (raw : RawValues program)
    (domain : SourceDomain) (source : Fin (PiRLCProductPlan.baseSourceWidth program)) :
    PerApplicationAssignmentTransportExecution.domainValue program raw domain source.val = raw.base source := by
  cases domain with
  | physicalBase =>
    exact SourceCompiler.sourceEnv_at raw.base source
  | retained =>
    let retained := PiRLCRetainedPreservation.baseSourceColumn program source
    have same : retained.val = source.val := rfl
    have read := SourceCompiler.sourceEnv_at raw.retainedSource retained
    rw [same] at read
    exact read.trans (PiRLCRetainedPreservation.sourceAssignment_base program
      raw.base raw.groupValue raw.products source)

private def referenceBlock {program : RetainedLayout.Program} (raw : RawValues program) : BlockKind → BlockValue
  | .laterPoseidon => ofBlock (LaterPoseidonRetainedBlocks.piCcsBlock program) raw.retainedSource
  | kind => kind.expand raw

private theorem reference_kind (program : RetainedLayout.Program) (raw : RawValues program) (kind : BlockKind) :
    (referenceBlock raw kind).block.kind = (BlockPlan.ofKind program kind).slotKind := by
  cases kind <;> rfl

private theorem reference_count (program : RetainedLayout.Program) (raw : RawValues program) (kind : BlockKind) :
    (referenceBlock raw kind).block.slotCount = commonLimit program kind := by
  cases kind <;> rfl

private theorem limit_le_expanded (program : RetainedLayout.Program) (raw : RawValues program) (kind : BlockKind) :
    commonLimit program kind ≤ (kind.expand raw).block.slotCount := by
  rw [← (entry_geometry_eq_expand program raw kind).2]
  exact commonLimit_le program kind

private theorem reference_source (program : RetainedLayout.Program) (raw : RawValues program) (kind : BlockKind)
    (slot : Nat) (inside : slot < commonLimit program kind)
    (referenceBound : slot < (referenceBlock raw kind).block.slotCount) :
    (referenceBlock raw kind).source ((referenceBlock raw kind).block.source ⟨slot, referenceBound⟩) =
      (kind.expand raw).source ((kind.expand raw).block.source
        ⟨slot, lt_of_lt_of_le inside (limit_le_expanded program raw kind)⟩) := by
  cases kind <;> try rfl
  simp only [referenceBlock, BlockKind.expand, BlockKind.template,
    Canonical.ofBlock, CanonicalBlockAssignment.ofBlock, LaterPoseidonRetainedBlocks.piCcsBlock,
    LowNormBlock.Block.slice, Nat.zero_add]

private theorem blockValue_source (plan : AssignmentTransport.Plan) (width : Nat) (env : Env)
    (block : Values) (slot : Fin block.count) :
    (blockValue plan width env block).source ((blockValue plan width env block).block.source slot) =
      sourceValue plan width env (AffineRuns.sourceAt block.sources slot.val) := rfl

private theorem block_correct (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan) (kind : BlockKind) (block : Values)
    (emitted : commonBlock program kind = .ok block) :
    (blockValue plan (physicalWidth program) (physicalValues program env application) block).coordinateCount =
      (referenceBlock (SourceAssignment.raw program env application) kind).coordinateCount ∧
    ∀ index, (blockValue plan (physicalWidth program) (physicalValues program env application) block).coordinateAt index =
      (referenceBlock (SourceAssignment.raw program env application) kind).coordinateAt index := by
  let raw := SourceAssignment.raw program env application
  have geometry := commonBlock_geometry program kind block emitted
  have kinds : (blockValue plan (physicalWidth program) (physicalValues program env application) block).block.kind =
      (referenceBlock raw kind).block.kind := geometry.1.trans (reference_kind program raw kind).symm
  have counts : (blockValue plan (physicalWidth program) (physicalValues program env application) block).block.slotCount =
      (referenceBlock raw kind).block.slotCount := geometry.2.trans (reference_count program raw kind).symm
  refine ⟨?_, ?_⟩
  · unfold BlockValue.coordinateCount LowNormBlock.Block.coordinateCount
    rw [kinds, counts]
  · intro index
    apply PerApplicationAssignmentTransportExecution.blockValue_coordinateAt_eq _ _ kinds counts
    intro slot leftBound rightBound
    have inside : slot < commonLimit program kind := by
      change slot < block.count at leftBound
      rwa [geometry.2] at leftBound
    let selected : Fin (commonLimit program kind) := ⟨slot, inside⟩
    let source := sourceIndex program kind ⟨slot, lt_of_lt_of_le inside (commonLimit_le program kind)⟩
    have sourceBound : source < PiRLCProductPlan.baseSourceWidth program :=
      commonBlock_source_lt program kind block emitted selected
    have mapped : finalColumn source = .ok (AffineRuns.sourceAt block.sources slot) :=
      commonBlock_source program kind block emitted selected
    have targetBound := finalColumn_lt program source _ sourceBound mapped
    rw [blockValue_source, sourceValue_physical _ _ _ _ targetBound,
      AssignmentTransportCommonSource.base_value program env application ⟨source, sourceBound⟩ _ mapped,
      reference_source program raw kind slot inside rightBound]
    have canonical := PerApplicationAssignmentTransportExecution.canonical_domain_source program raw kind slot
      (lt_of_lt_of_le inside (commonLimit_le program kind))
      (lt_of_lt_of_le inside (limit_le_expanded program raw kind))
    exact (domainValue_base program raw (sourceDomainOf kind) ⟨source, sourceBound⟩).symm.trans canonical

private theorem reference_schedule {program : RetainedLayout.Program} (raw : RawValues program) :
    commonKinds.map (referenceBlock raw) = AssignmentTransportCommon.common raw := by
  rfl

private theorem mapped_schedule (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan) (kinds : List BlockKind) (blocks : List Values)
    (pairs : List.Forall₂ (fun kind block => commonBlock program kind = .ok block) kinds blocks) :
    coordinateCount (blocks.map (blockValue plan (physicalWidth program) (physicalValues program env application))) =
      coordinateCount (kinds.map (referenceBlock (SourceAssignment.raw program env application))) ∧
    ∀ index, coordinateAt
      (blocks.map (blockValue plan (physicalWidth program) (physicalValues program env application))) index =
      coordinateAt (kinds.map (referenceBlock (SourceAssignment.raw program env application))) index := by
  induction pairs with
  | nil => exact ⟨rfl, fun _ => rfl⟩
  | @cons kind block kinds blocks emitted pairs ih =>
    obtain ⟨count, value⟩ := block_correct program env application plan kind block emitted
    refine ⟨?_, ?_⟩
    · simp only [List.map_cons, coordinateCount, count, ih.1]
    · intro index
      simp only [List.map_cons, coordinateAt, count]
      split
      · exact value index
      · exact ih.2 _

/-- Every emitted common block has the canonical count and value sequence.
The plan parameter cannot affect these physical-prefix reads. -/
theorem common_correct (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan) (blocks : List Values)
    (emitted : commonKinds.mapM (commonBlock program) = .ok blocks) :
    coordinateCount (blocks.map (blockValue plan (physicalWidth program) (physicalValues program env application))) =
      coordinateCount (AssignmentTransportCommon.common (SourceAssignment.raw program env application)) ∧
    ∀ index, coordinateAt
      (blocks.map (blockValue plan (physicalWidth program) (physicalValues program env application))) index =
      coordinateAt (AssignmentTransportCommon.common (SourceAssignment.raw program env application)) index := by
  have result := mapped_schedule program env application plan commonKinds blocks
    (PhysicalRelabel.mapM_pairs _ _ _ emitted)
  rwa [reference_schedule] at result

/-- Before the common boundary, executing the emitted common blocks gives
the same seed assignment used by the complete wide witness. -/
theorem assignment_before (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan) (blocks : List Values)
    (emitted : commonKinds.mapM (commonBlock program) = .ok blocks)
    (target : Fin (RetainedLayout.logicalWidth program)) (before : target.val < RetainedLayout.commonCount program) :
    CanonicalBlockAssignment.assignment
      (encodedHashCells (SourceAssignment.raw program env application).outputDigest)
      (blocks.map (blockValue plan (physicalWidth program) (physicalValues program env application))) target =
      AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment target := by
  have values := (common_correct program env application plan blocks emitted).2
  have same : CanonicalBlockAssignment.assignment
      (encodedHashCells (SourceAssignment.raw program env application).outputDigest)
      (blocks.map (blockValue plan (physicalWidth program) (physicalValues program env application))) target =
      CanonicalBlockAssignment.assignment
        (encodedHashCells (SourceAssignment.raw program env application).outputDigest)
        (AssignmentTransportCommon.common (SourceAssignment.raw program env application)) target := by
    unfold CanonicalBlockAssignment.assignment
    split
    · rfl
    · exact values _
  exact same.trans (AssignmentTransportCommon.assignment_before (SourceAssignment.raw program env application) target before)

end NightstreamFPrime.Export.Stage1.Wide.CommonSchedule
