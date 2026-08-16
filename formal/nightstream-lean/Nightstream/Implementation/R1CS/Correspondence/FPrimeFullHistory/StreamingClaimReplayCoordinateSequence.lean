import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinateOverlay
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram

/-!
Contract: exact 86-phase composition of linked claim-coordinate source rows.

Owns the verifier-selected active/carry overlay schedule, the field-value
meaning of the private phase-to-overlay links, and the proof that accepted
linked source rows produce the direct 21,220-field PiCCS coordinate
commitment.

Does not own generated low-norm link rows, Rust trace or column conformance,
Poseidon2 frame recovery, state-digest collision resistance, the other 314
program phases, or recursive lifecycle integration.

Emits constraints: no. Its premises name the exact source-row and private-link
facts that the generated production artifact must derive.
-/

set_option autoImplicit false
set_option maxRecDepth 262144

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateSequence

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateAccumulator
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution

/-- Exactly the 23 claim chunks that contain a selected PiCCS coordinate. -/
def ActiveChunk (chunk : Fin claimChunkCount) : Prop :=
  chunk.val = 0 \/ (60 ≤ chunk.val /\ chunk.val ≤ 81)

instance activeChunkDecidable (chunk : Fin claimChunkCount) :
    Decidable (ActiveChunk chunk) := by
  unfold ActiveChunk
  infer_instance

/-- Rust overlay-kind code for one claim phase. Kind zero is reserved for
non-claim program arms. -/
def overlayKindCodeAt (chunk : Fin claimChunkCount) : Nat :=
  if chunk.val = 0 then
    3
  else if 60 ≤ chunk.val /\ chunk.val ≤ 81 then
    4 + (chunk.val - 60)
  else if chunk.val = 85 then
    2
  else
    1

theorem overlayKindCodeAt_lt (chunk : Fin claimChunkCount) :
    overlayKindCodeAt chunk < 26 := by
  have bound := chunk.isLt
  unfold claimChunkCount at bound
  by_cases zero : chunk.val = 0
  · simp [overlayKindCodeAt, zero]
  by_cases range : 60 ≤ chunk.val /\ chunk.val ≤ 81
  · simp [overlayKindCodeAt, zero, range]
    omega
  by_cases final : chunk.val = 85
  · simp [overlayKindCodeAt, final]
  · simp [overlayKindCodeAt, zero, range, final]

def overlayKindAt (chunk : Fin claimChunkCount) : Fin 26 :=
  ⟨overlayKindCodeAt chunk, overlayKindCodeAt_lt chunk⟩

/-- Overlay kind for one verifier-owned work item. Invalid claim indices map
to no-op; the production program never contains such an index. -/
def overlayKindForWorkItem (item : WorkItem) : Fin 26 :=
  if item.phase = .claimReplay then
    if bound : item.index < claimChunkCount then
      overlayKindAt ⟨item.index, bound⟩
    else
      0
  else
    0

def productionOverlayKindMap : List Nat :=
  (program productionConfig).map fun item =>
    (overlayKindForWorkItem item).val

/-- Exact compact source-field link runs for the 25 non-no-op overlay kinds.
Each entry is `(overlay kind, phase kind, claim chunk, active offset, active
field count)`. Every run also carries the fixed 216 before/after commitment
links. -/
def productionOverlayLinkRuns : List (Nat × Nat × Nat × Nat × Nat) :=
  [ (1, 3, 1, 0, 0)
  , (2, 4, 85, 0, 0)
  , (3, 3, 0, 383, 52)
  , (4, 3, 60, 987, 37)
  , (5, 3, 61, 0, 1024)
  , (6, 3, 62, 0, 1024)
  , (7, 3, 63, 0, 1024)
  , (8, 3, 64, 0, 1024)
  , (9, 3, 65, 0, 1024)
  , (10, 3, 66, 0, 1024)
  , (11, 3, 67, 0, 1024)
  , (12, 3, 68, 0, 1024)
  , (13, 3, 69, 0, 1024)
  , (14, 3, 70, 0, 1024)
  , (15, 3, 71, 0, 1024)
  , (16, 3, 72, 0, 1024)
  , (17, 3, 73, 0, 1024)
  , (18, 3, 74, 0, 1024)
  , (19, 3, 75, 0, 1024)
  , (20, 3, 76, 0, 1024)
  , (21, 3, 77, 0, 1024)
  , (22, 3, 78, 0, 1024)
  , (23, 3, 79, 0, 1024)
  , (24, 3, 80, 0, 1024)
  , (25, 3, 81, 0, 651)
  ]

@[simp] theorem productionOverlayLinkRuns_length :
    productionOverlayLinkRuns.length = 25 := by
  decide

theorem productionOverlayLinkRuns_census :
    (productionOverlayLinkRuns.map fun run => run.2.2.2.2).sum = 21_220 /\
      (productionOverlayLinkRuns.map fun run => 216 + run.2.2.2.2).sum =
        26_620 := by
  decide

@[simp] theorem productionOverlayKindMap_length :
    productionOverlayKindMap.length = 400 := by
  unfold productionOverlayKindMap
  rw [List.length_map, production_program_length]

@[simp] theorem overlayKindAt_zero :
    (overlayKindAt ⟨0, by decide⟩).val = 3 := by
  rfl

theorem overlayKindAt_evaluation
    (chunk : Fin claimChunkCount) (notZero : chunk.val ≠ 0)
    (range : 60 ≤ chunk.val /\ chunk.val ≤ 81) :
    (overlayKindAt chunk).val = 4 + (chunk.val - 60) := by
  simp [overlayKindAt, overlayKindCodeAt, notZero, range]

@[simp] theorem overlayKindAt_final :
    (overlayKindAt ⟨85, by decide⟩).val = 2 := by
  rfl

theorem overlayKindAt_fullCarry
    (chunk : Fin claimChunkCount) (inactive : ¬ ActiveChunk chunk)
    (notFinal : chunk.val ≠ 85) :
    (overlayKindAt chunk).val = 1 := by
  have notZero : chunk.val ≠ 0 := by
    intro zero
    exact inactive (Or.inl zero)
  have notRange : ¬ (60 ≤ chunk.val /\ chunk.val ≤ 81) := by
    intro range
    exact inactive (Or.inr range)
  simp [overlayKindAt, overlayKindCodeAt, notZero, notRange, notFinal]

/-- All chunks outside the fixed active set have an empty coordinate mask. -/
theorem inactiveChunk_of_not_active
    (chunk : Fin claimChunkCount) (inactive : ¬ ActiveChunk chunk) :
    InactiveChunk chunk := by
  have notZero : chunk.val ≠ 0 := by
    intro zero
    exact inactive (Or.inl zero)
  apply inactiveChunk_of_gap chunk notZero
  have bound := chunk.isLt
  unfold claimChunkCount at bound
  unfold ActiveChunk at inactive
  omega

/-- The last claim chunk uses the final replay body. Every earlier claim chunk
uses the shared full replay body. -/
def replayKindAt (chunk : Fin claimChunkCount) : ArmKind :=
  if chunk.val = 85 then .final else .full

theorem replayKindAt_active
    (chunk : Fin claimChunkCount) (active : ActiveChunk chunk) :
    replayKindAt chunk = .full := by
  unfold replayKindAt
  rw [if_neg]
  intro final
  rcases active with zero | range <;> omega

def phaseCommitmentColumn
    (chunk : Fin claimChunkCount) (side : StateSide)
    (output : Fin (shape.rows * shape.degree)) : Nat :=
  commitmentColumn (replayKindAt chunk) side output

def phaseChunkColumn
    (chunk : Fin claimChunkCount) (offset : Fin claimChunkWidth) : Nat :=
  chunkColumn (armFor (replayKindAt chunk)) offset.val

/-- Exact placement of one external accumulator in a phase assignment. -/
def AccumulatorPlaced
    (assignment : Nat → Nat)
    (columns : Fin (shape.rows * shape.degree) → Nat)
    (accumulator : Accumulator) : Prop :=
  ∀ output, assignment (columns output) = (accumulator output).val

private theorem decodedAccumulator_eq_of_links
    {overlayAssignment phaseAssignment : Nat → Nat}
    {overlayColumns phaseColumns :
      Fin (shape.rows * shape.degree) → Nat}
    {accumulator : Accumulator}
    (canonical : ∀ column, overlayAssignment column < goldilocksP)
    (linked : ∀ output,
      overlayAssignment (overlayColumns output) =
        phaseAssignment (phaseColumns output))
    (placed : AccumulatorPlaced phaseAssignment phaseColumns accumulator) :
    decodedAccumulator overlayAssignment overlayColumns = accumulator := by
  funext output
  apply Fin.ext
  rw [decodedAccumulator_val overlayAssignment overlayColumns canonical output]
  exact (linked output).trans (placed output)

/-- Exact source-level evidence for one selected active overlay. The chunk and
commitment equalities are the field meanings of the generated low-norm link
rows. -/
structure ActiveLinkedRows
    (production : ProductionSetup) (fields : Fields)
    (chunk : Fin claimChunkCount) (before after : Accumulator) where
  layout : ActiveLayout
  phaseAssignment : Nat → Nat
  overlayAssignment : Nat → Nat
  overlayCanonical : ∀ column, overlayAssignment column < goldilocksP
  overlayOne : overlayAssignment 0 = 1
  forChunk : ForClaimChunk layout.coordinate chunk
  phaseBeforePlaced : AccumulatorPlaced phaseAssignment
    (phaseCommitmentColumn chunk .before) before
  phaseAfterPlaced : AccumulatorPlaced phaseAssignment
    (phaseCommitmentColumn chunk .after) after
  beforeLinked : ∀ output,
    overlayAssignment (layout.beforeColumn output) =
      phaseAssignment (phaseCommitmentColumn chunk .before output)
  afterLinked : ∀ output,
    overlayAssignment (layout.afterColumn output) =
      phaseAssignment (phaseCommitmentColumn chunk .after output)
  activeFieldLinked : ∀ field ∈ layout.coordinate.activeFields,
    overlayAssignment (layout.coordinate.fieldColumn field) =
      phaseAssignment (phaseChunkColumn chunk (claimChunkOffset field))
  phaseChunkMatchesFields : ∀ field, claimChunk field = chunk →
    phaseAssignment (phaseChunkColumn chunk (claimChunkOffset field)) =
      (fields field).val
  satisfies : Satisfies (activeRows production chunk layout) overlayAssignment

namespace ActiveLinkedRows

theorem activeFieldsPlaced
    {production : ProductionSetup} {fields : Fields}
    {chunk : Fin claimChunkCount} {before after : Accumulator}
    (rows : ActiveLinkedRows production fields chunk before after) :
    ActiveFieldsPlaced rows.layout.coordinate rows.overlayAssignment fields := by
  intro field active
  have selected : claimChunk field = chunk := by
    rw [show rows.layout.coordinate.activeFields = activeFields chunk from
      rows.forChunk] at active
    exact (mem_activeFields chunk field).mp active
  exact (rows.activeFieldLinked field active).trans
    (rows.phaseChunkMatchesFields field selected)

theorem beforeDecoded
    {production : ProductionSetup} {fields : Fields}
    {chunk : Fin claimChunkCount} {before after : Accumulator}
    (rows : ActiveLinkedRows production fields chunk before after) :
    decodedAccumulator rows.overlayAssignment rows.layout.beforeColumn =
      before := by
  exact decodedAccumulator_eq_of_links rows.overlayCanonical rows.beforeLinked
    rows.phaseBeforePlaced

theorem afterDecoded
    {production : ProductionSetup} {fields : Fields}
    {chunk : Fin claimChunkCount} {before after : Accumulator}
    (rows : ActiveLinkedRows production fields chunk before after) :
    decodedAccumulator rows.overlayAssignment rows.layout.afterColumn =
      after := by
  exact decodedAccumulator_eq_of_links rows.overlayCanonical rows.afterLinked
    rows.phaseAfterPlaced

theorem step
    {production : ProductionSetup} {fields : Fields}
    {chunk : Fin claimChunkCount} {before after : Accumulator}
    (rows : ActiveLinkedRows production fields chunk before after) :
    StepAt production fields chunk before after := by
  have localStep := activeRows_imply_step rows.forChunk rows.overlayCanonical
    rows.overlayOne rows.activeFieldsPlaced rows.satisfies
  rw [rows.beforeDecoded, rows.afterDecoded] at localStep
  exact localStep

theorem initial
    {production : ProductionSetup} {fields : Fields}
    {chunk : Fin claimChunkCount} {before after : Accumulator}
    (rows : ActiveLinkedRows production fields chunk before after)
    (chunkZero : chunk.val = 0) :
    before = zeroAccumulator := by
  have zero := activeRows_chunkZero_initial chunkZero rows.overlayCanonical
    rows.overlayOne rows.satisfies
  rw [rows.beforeDecoded] at zero
  exact zero

end ActiveLinkedRows

/-- Exact source-level evidence for one selected carry overlay. -/
structure CarryLinkedRows
    (production : ProductionSetup) (fields : Fields)
    (chunk : Fin claimChunkCount) (before after : Accumulator) where
  beforeColumn : Fin (shape.rows * shape.degree) → Nat
  afterColumn : Fin (shape.rows * shape.degree) → Nat
  phaseAssignment : Nat → Nat
  overlayAssignment : Nat → Nat
  overlayCanonical : ∀ column, overlayAssignment column < goldilocksP
  overlayOne : overlayAssignment 0 = 1
  phaseBeforePlaced : AccumulatorPlaced phaseAssignment
    (phaseCommitmentColumn chunk .before) before
  phaseAfterPlaced : AccumulatorPlaced phaseAssignment
    (phaseCommitmentColumn chunk .after) after
  beforeLinked : ∀ output,
    overlayAssignment (beforeColumn output) =
      phaseAssignment (phaseCommitmentColumn chunk .before output)
  afterLinked : ∀ output,
    overlayAssignment (afterColumn output) =
      phaseAssignment (phaseCommitmentColumn chunk .after output)
  satisfies : Satisfies (carryRows beforeColumn afterColumn) overlayAssignment

namespace CarryLinkedRows

theorem beforeDecoded
    {production : ProductionSetup} {fields : Fields}
    {chunk : Fin claimChunkCount} {before after : Accumulator}
    (rows : CarryLinkedRows production fields chunk before after) :
    decodedAccumulator rows.overlayAssignment rows.beforeColumn = before := by
  exact decodedAccumulator_eq_of_links rows.overlayCanonical rows.beforeLinked
    rows.phaseBeforePlaced

theorem afterDecoded
    {production : ProductionSetup} {fields : Fields}
    {chunk : Fin claimChunkCount} {before after : Accumulator}
    (rows : CarryLinkedRows production fields chunk before after) :
    decodedAccumulator rows.overlayAssignment rows.afterColumn = after := by
  exact decodedAccumulator_eq_of_links rows.overlayCanonical rows.afterLinked
    rows.phaseAfterPlaced

theorem step
    {production : ProductionSetup} {fields : Fields}
    {chunk : Fin claimChunkCount} {before after : Accumulator}
    (rows : CarryLinkedRows production fields chunk before after)
    (inactive : InactiveChunk chunk) :
    StepAt production fields chunk before after := by
  have localStep :
      StepAt production fields chunk
        (decodedAccumulator rows.overlayAssignment rows.beforeColumn)
        (decodedAccumulator rows.overlayAssignment rows.afterColumn) :=
    carryRows_imply_step (production := production) (fields := fields)
      inactive rows.overlayCanonical rows.overlayOne rows.satisfies
  rw [rows.beforeDecoded, rows.afterDecoded] at localStep
  exact localStep

end CarryLinkedRows

/-- The exact production choice for one claim phase: active rows for chunks
zero and 60 through 81, and carry rows for every other chunk. -/
def PhaseRowsAt
    (production : ProductionSetup) (fields : Fields)
    (chunk : Fin claimChunkCount) (before after : Accumulator) : Prop :=
  if ActiveChunk chunk then
    Nonempty (ActiveLinkedRows production fields chunk before after)
  else
    Nonempty (CarryLinkedRows production fields chunk before after)

namespace PhaseRowsAt

theorem step
    {production : ProductionSetup} {fields : Fields}
    {chunk : Fin claimChunkCount} {before after : Accumulator}
    (accepted : PhaseRowsAt production fields chunk before after) :
    StepAt production fields chunk before after := by
  by_cases active : ActiveChunk chunk
  · rw [PhaseRowsAt, if_pos active] at accepted
    rcases accepted with ⟨rows⟩
    exact rows.step
  · rw [PhaseRowsAt, if_neg active] at accepted
    rcases accepted with ⟨rows⟩
    exact rows.step (inactiveChunk_of_not_active chunk active)

theorem zero_initial
    {production : ProductionSetup} {fields : Fields}
    {before after : Accumulator}
    (accepted : PhaseRowsAt production fields ⟨0, by decide⟩ before after) :
    before = zeroAccumulator := by
  have active : ActiveChunk (⟨0, by decide⟩ : Fin claimChunkCount) :=
    Or.inl rfl
  rw [PhaseRowsAt, if_pos active] at accepted
  rcases accepted with ⟨rows⟩
  exact rows.initial rfl

end PhaseRowsAt

/-- One exact linked-row witness for every claim chunk, all connected through
one ordered accumulator state. -/
structure AcceptedLinkedRun
    (production : ProductionSetup) (fields : Fields) where
  state : Nat → Accumulator
  phase : ∀ chunk : Fin claimChunkCount,
    PhaseRowsAt production fields chunk (state chunk.val)
      (state (chunk.val + 1))

namespace AcceptedLinkedRun

def toAccumulatorRun
    {production : ProductionSetup} {fields : Fields}
    (run : AcceptedLinkedRun production fields) :
    FPrimeFullHistoryStreamingClaimReplayCoordinateAccumulator.AcceptedRun
      production fields where
  state := run.state
  initial := (run.phase ⟨0, by decide⟩).zero_initial
  step := fun chunk => (run.phase chunk).step

/-- The linked 86-phase source-row chain has algebraic authority: its final
carried value is the direct commitment to all 21,220 fields. -/
theorem final_eq_direct
    {production : ProductionSetup} {fields : Fields}
    (run : AcceptedLinkedRun production fields) :
    run.state claimChunkCount = directCoordinate production fields := by
  exact run.toAccumulatorRun.final_eq_direct

end AcceptedLinkedRun

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateSequence
