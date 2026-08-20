import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayCoordinateOverlaySchema
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinateCallRefinement

/-!
Contract: structural refinement of one claim-coordinate overlay arm.

Assurance tier: model-level row refinement.

Owns the proof that one active overlay arm updates all three authoritative
claim accumulators. A present map call supplies one exact chunk contribution;
an absent call is allowed only when that map has no fields in the chunk.

Does not own generated arm identity, phase-state links, chunk-zero initial
state, complete accumulator runs, lifecycle selection, or Module-SIS
hardness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlayArtifact

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateCallRefinement
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay.Artifact
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.SeededPhi81RingRefinement
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

/-- Small generated-identity contract consumed by the structural row proof. -/
structure ArmContract (arm : RawActiveArm) : Prop where
  callFor_valid :
    ∀ kind call, arm.callFor kind = some call →
      call ∈ arm.coordinateCalls ∧
        call.mapKind = kind ∧
        call.chunk = arm.chunk ∧
        call.ScheduleValid
  noCall_empty :
    ∀ kind, arm.callFor kind = none →
      kind.activeFields arm.chunk = []

def decodedBefore
    (layout : StateBases) (assignment : Nat → Nat) : State :=
  fun kind output => residueNat (assignment (layout.beforeColumn kind output))

def decodedAfter
    (layout : StateBases) (assignment : Nat → Nat) : State :=
  fun kind output => residueNat (assignment (layout.afterColumn kind output))

private theorem kind_mem_mapOrder (kind : MapKind) : kind ∈ mapOrder := by
  cases kind <;> simp [mapOrder]

private theorem map_rows_satisfy
    (layout : StateBases) (arm : RawActiveArm)
    (assignment : Nat → Nat) (satisfied : arm.Satisfied layout assignment)
    (kind : MapKind) :
    Satisfies (mapRows layout arm kind) assignment := by
  intro row rowMember
  apply satisfied row
  apply List.mem_append_left
  exact List.mem_flatMap.mpr
    ⟨kind, kind_mem_mapOrder kind, rowMember⟩

private theorem call_rows_satisfy
    (layout : StateBases) (arm : RawActiveArm)
    (assignment : Nat → Nat) (satisfied : arm.Satisfied layout assignment)
    (kind : MapKind) (call : CoordinateCall)
    (selected : arm.callFor kind = some call) :
    call.Satisfied assignment := by
  intro row rowMember
  apply map_rows_satisfy layout arm assignment satisfied kind row
  rw [mapRows, selected]
  exact List.mem_append_left _ rowMember

private theorem update_rows_satisfy
    (layout : StateBases) (arm : RawActiveArm)
    (assignment : Nat → Nat) (satisfied : arm.Satisfied layout assignment)
    (kind : MapKind) (call : CoordinateCall)
    (selected : arm.callFor kind = some call) :
    Satisfies (updateRows layout call) assignment := by
  intro row rowMember
  apply map_rows_satisfy layout arm assignment satisfied kind row
  rw [mapRows, selected]
  exact List.mem_append_right _ rowMember

private theorem carry_rows_satisfy
    (layout : StateBases) (arm : RawActiveArm)
    (assignment : Nat → Nat) (satisfied : arm.Satisfied layout assignment)
    (kind : MapKind) (selected : arm.callFor kind = none) :
    Satisfies (carryRows layout kind) assignment := by
  simpa [mapRows, selected] using
    map_rows_satisfy layout arm assignment satisfied kind

private theorem rowHolds_of_operand_perms
    (assignment : Nat → Nat) {source target : Row}
    (a : source.a.Perm target.a)
    (b : source.b.Perm target.b)
    (c : source.c.Perm target.c)
    (holds : RowHolds assignment source) :
    RowHolds assignment target := by
  unfold RowHolds at holds ⊢
  calc
    lcEval assignment target.a * lcEval assignment target.b % goldilocksP =
        lcEval assignment source.a * lcEval assignment source.b %
          goldilocksP := by
      rw [Program.lcEval_eq_of_perm assignment a,
        Program.lcEval_eq_of_perm assignment b]
    _ = lcEval assignment source.c := holds
    _ = lcEval assignment target.c :=
      Program.lcEval_eq_of_perm assignment c

private theorem updateRow_perms
    (layout : StateBases) (call : CoordinateCall)
    (output : Fin outputWidth) :
    (updateRow layout call output).a.Perm
        (builderLinearRow (layout.afterColumn call.mapKind output)
          [(layout.beforeColumn call.mapKind output, 1),
            (call.outputColumn output, 1)]).a ∧
      (updateRow layout call output).b.Perm
        (builderLinearRow (layout.afterColumn call.mapKind output)
          [(layout.beforeColumn call.mapKind output, 1),
            (call.outputColumn output, 1)]).b ∧
      (updateRow layout call output).c.Perm
        (builderLinearRow (layout.afterColumn call.mapKind output)
          [(layout.beforeColumn call.mapKind output, 1),
            (call.outputColumn output, 1)]).c := by
  exact ⟨List.Perm.swap _ _ _, List.Perm.refl _, List.Perm.refl _⟩

private theorem update_facts
    (layout : StateBases) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) (call : CoordinateCall)
    (satisfies : Satisfies (updateRows layout call) assignment) :
    ∀ output : Fin outputWidth,
      assignment (layout.afterColumn call.mapKind output) =
        (assignment (layout.beforeColumn call.mapKind output) +
          assignment (call.outputColumn output)) % goldilocksP := by
  intro output
  have emitted := satisfies (updateRow layout call output)
    (List.mem_ofFn.mpr ⟨output, rfl⟩)
  have builderHolds := rowHolds_of_operand_perms assignment
    (updateRow_perms layout call output).1
    (updateRow_perms layout call output).2.1
    (updateRow_perms layout call output).2.2 emitted
  have defined := builderLinearRow_sound canonical one
    (layout.afterColumn call.mapKind output)
    [(layout.beforeColumn call.mapKind output, 1),
      (call.outputColumn output, 1)]
    (by simp [CanonicalTerms]; decide) builderHolds
  simpa [lcEval, Nat.add_comm, Nat.mul_comm] using defined

private theorem carryRow_perms
    (layout : StateBases) (kind : MapKind)
    (output : Fin outputWidth) :
    (carryRow layout kind output).a.Perm
        (builderLinearRow (layout.afterColumn kind output)
          [(layout.beforeColumn kind output, 1)]).a ∧
      (carryRow layout kind output).b.Perm
        (builderLinearRow (layout.afterColumn kind output)
          [(layout.beforeColumn kind output, 1)]).b ∧
      (carryRow layout kind output).c.Perm
        (builderLinearRow (layout.afterColumn kind output)
          [(layout.beforeColumn kind output, 1)]).c := by
  exact ⟨List.Perm.swap _ _ _, List.Perm.refl _, List.Perm.refl _⟩

private theorem carry_facts
    (layout : StateBases) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) (kind : MapKind)
    (satisfies : Satisfies (carryRows layout kind) assignment) :
    ∀ output : Fin outputWidth,
      assignment (layout.afterColumn kind output) =
        assignment (layout.beforeColumn kind output) := by
  intro output
  have emitted := satisfies (carryRow layout kind output)
    (List.mem_ofFn.mpr ⟨output, rfl⟩)
  have builderHolds := rowHolds_of_operand_perms assignment
    (carryRow_perms layout kind output).1
    (carryRow_perms layout kind output).2.1
    (carryRow_perms layout kind output).2.2 emitted
  have defined := builderLinearRow_sound canonical one
    (layout.afterColumn kind output)
    [(layout.beforeColumn kind output, 1)]
    (by simp [CanonicalTerms]; decide) builderHolds
  simpa [lcEval, Nat.mod_eq_of_lt (canonical _)] using defined

private theorem selectedDigit_zero_of_activeFields_nil
    (frame : ClaimFrame) (kind : MapKind) (chunk : Fin claimChunkCount)
    (empty : kind.activeFields chunk = [])
    (field : Fin kind.fieldCount) (digit : Fin digitCount) :
    selectedDigit frame kind chunk field digit = 0 := by
  unfold selectedDigit
  rw [if_neg]
  intro selected
  have active := (kind.mem_activeFields chunk field).2 selected
  rw [empty] at active
  simpa using active

private theorem semanticAssignment_zero_of_activeFields_nil
    (frame : ClaimFrame) (kind : MapKind) (chunk : Fin claimChunkCount)
    (empty : kind.activeFields chunk = []) :
    ∀ column, semanticAssignment frame kind chunk column = 0 := by
  intro column
  unfold semanticAssignment
  split
  · exact selectedDigit_zero_of_activeFields_nil frame kind chunk empty _ _
  · rfl

private theorem inputValue_zero
    (block : SeededPhi81.Block) (assignment : Nat → Nat)
    (zero : ∀ column, assignment column = 0)
    (messageCol messageRow : Nat) :
    block.inputValue assignment messageCol messageRow = 0 := by
  cases selected : block.bitColumn
      (messageRow * block.messageCols + messageCol) with
  | none => simp [SeededPhi81.Block.inputValue, selected]
  | some column => simp [SeededPhi81.Block.inputValue, selected, zero]

private theorem foldl_self
    {alpha beta : Type} (items : List alpha) (step : beta → alpha → beta)
    (same : ∀ state item, step state item = state) (initial : beta) :
    items.foldl step initial = initial := by
  induction items generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [same]
      exact inductionHypothesis _

private theorem linearValue_zero
    (block : SeededPhi81.Block) (assignment : Nat → Nat)
    (zero : ∀ column, assignment column = 0)
    (output coordinate : Nat) :
    block.linearValue assignment output coordinate = 0 := by
  unfold SeededPhi81.Block.linearValue
  have sumZero :
      (List.range block.messageCols).foldl (fun outer messageCol =>
        (List.range SeededPhi81.dimension).foldl (fun inner messageRow =>
          inner + block.termValue assignment output coordinate
            messageCol messageRow) outer) 0 = 0 := by
    apply foldl_self
    intro outer messageCol
    apply foldl_self
    intro inner messageRow
    unfold SeededPhi81.Block.termValue
    rw [inputValue_zero block assignment zero]
    simp
  rw [sumZero]
  rfl

private theorem chunkContribution_zero_of_activeFields_nil
    (frame : ClaimFrame) (kind : MapKind) (chunk : Fin claimChunkCount)
    (empty : kind.activeFields chunk = [])
    (output : Fin outputWidth) :
    chunkContribution frame kind chunk output = 0 := by
  unfold chunkContribution
  rw [linearValue_zero (semanticBlock kind)
    (semanticAssignment frame kind chunk)
    (semanticAssignment_zero_of_activeFields_nil frame kind chunk empty)]
  rfl

/-- Exact rows and source links for one active overlay arm imply the
authoritative three-map claim accumulator step. -/
theorem rows_and_frame_link_imply_step
    (layout : StateBases) (arm : RawActiveArm)
    (frame : ClaimFrame) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : arm.Satisfied layout assignment)
    (linked : arm.FrameLinked frame assignment)
    (contract : ArmContract arm) :
    Step frame arm.chunk
      (decodedBefore layout assignment) (decodedAfter layout assignment) := by
  intro kind output
  change residueNat (assignment (layout.afterColumn kind output)) =
    residueNat (assignment (layout.beforeColumn kind output)) +
      chunkContribution frame kind arm.chunk output
  cases selected : arm.callFor kind with
  | none =>
      have physical := carry_facts layout assignment canonical one kind
        (carry_rows_satisfy layout arm assignment satisfied kind selected)
        output
      have carried := congrArg residueNat physical
      rw [chunkContribution_zero_of_activeFields_nil frame kind arm.chunk
        (contract.noCall_empty kind selected) output]
      simpa using carried
  | some call =>
      rcases contract.callFor_valid kind call selected with
        ⟨member, callKind, callChunk, schedule⟩
      subst kind
      have physical := update_facts layout assignment canonical one call
        (update_rows_satisfy layout arm assignment satisfied call.mapKind call
          selected) output
      have updated := congrArg residueNat physical
      rw [residueNat_mod, residueNat_add] at updated
      have contribution := output_eq_chunkContribution frame assignment
        canonical one call
        (call_rows_satisfy layout arm assignment satisfied call.mapKind call
          selected)
        (linked call member) arm.chunk callChunk schedule output
      rw [contribution] at updated
      exact updated

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlayArtifact
