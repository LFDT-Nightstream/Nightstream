import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayoutCertificate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallSound

/-!
Contract: family-wide composition and emitted-row binding for the four exact
production PiRLC Poseidon2 replay-call runs.

Assurance tier: artifact-checked and Rust-conformant call-family semantics for
the Nightstream b2/k16 profile.

Owns all 486 call indices, the selector owned by each call, the global local-
slot index, exact local-slot arithmetic, the canonical 86-row block at every
emitted range, and transport from block satisfaction to the typed Poseidon2
S-box equations.

Does not own selector activation, replay input authority, complete PiRLC
semantics, lifecycle authority, or cryptographic security.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallRowProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel

private abbrev RawRun :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.RawRun

structure Run where
  raw : RawRun
  globalStart : Nat

def Run.leafClassAt (run : Run) (index : Nat) : LeafClass :=
  if index = 0 then
    match run.raw.firstClass with
    | .direct => .direct
    | .partialStart => .partialStart
  else
    .chained run.raw.selectorColumn

def Run.globalIndexAt (run : Run) (index : Nat) : Nat :=
  run.globalStart + index

def Run.localFinalAt (run : Run) (index : Nat) : Nat :=
  run.raw.localFinalStart + index *
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.localFinalStride

structure Run.Valid (run : Run) : Prop where
  callCountPositive : 0 < run.raw.callCount
  selectorOwned :
    selectorColumn (run.leafClassAt 0) = run.raw.selectorColumn
  localRoot :
    run.raw.localFinalStart =
      firstLocalSlotStart +
        run.globalStart * (localSlotCount * slotWidth)

theorem Run.Valid.selectorColumn_owned
    {run : Run} (valid : run.Valid) (index : Nat) :
    selectorColumn (run.leafClassAt index) = run.raw.selectorColumn := by
  cases index with
  | zero => exact valid.selectorOwned
  | succ index => simp [Run.leafClassAt, selectorColumn]

theorem Run.Valid.localFinalAt_eq_projection
    {run : Run} (valid : run.Valid) (index : Nat) :
    run.localFinalAt index = currentLocalSlotStart (run.globalIndexAt index) := by
  rw [Run.localFinalAt, valid.localRoot]
  rw [Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.localFinalStride_exact]
  simp only [Run.globalIndexAt, currentLocalSlotStart]
  norm_num [localSlotCount, slotWidth]
  omega

def evenInputRun : Run where
  raw :=
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenInput
  globalStart := 0

def evenOutputRun : Run where
  raw :=
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenOutput
  globalStart := 229

def oddInputRun : Run where
  raw :=
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddInput
  globalStart := 0

def oddOutputRun : Run where
  raw :=
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddOutput
  globalStart := 230

def runs : List Run :=
  [evenInputRun, evenOutputRun, oddInputRun, oddOutputRun]

theorem artifact_runs_exact :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.rawRuns =
      runs.map Run.raw := by
  rw [Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.rawRuns_exact]
  rfl

theorem evenInputRun_valid : evenInputRun.Valid := by
  constructor <;>
    norm_num [evenInputRun, Run.leafClassAt, selectorColumn,
      directSelectorColumn, partialSelectorColumn, firstLocalSlotStart,
      localSlotCount, slotWidth,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenInput]

theorem evenOutputRun_valid : evenOutputRun.Valid := by
  constructor <;>
    norm_num [evenOutputRun, Run.leafClassAt, selectorColumn,
      directSelectorColumn, partialSelectorColumn, firstLocalSlotStart,
      localSlotCount, slotWidth,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedEvenOutput]

theorem oddInputRun_valid : oddInputRun.Valid := by
  constructor <;>
    norm_num [oddInputRun, Run.leafClassAt, selectorColumn,
      directSelectorColumn, partialSelectorColumn, firstLocalSlotStart,
      localSlotCount, slotWidth,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddInput]

theorem oddOutputRun_valid : oddOutputRun.Valid := by
  constructor <;>
    norm_num [oddOutputRun, Run.leafClassAt, selectorColumn,
      directSelectorColumn, partialSelectorColumn, firstLocalSlotStart,
      localSlotCount, slotWidth,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.expectedOddOutput]

theorem runs_valid : ∀ run ∈ runs, run.Valid := by
  intro run member
  simp only [runs, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · exact evenInputRun_valid
  · exact evenOutputRun_valid
  · exact oddInputRun_valid
  · exact oddOutputRun_valid

theorem callCounts_exact :
    runs.map (fun run => run.raw.callCount) = [229, 13, 230, 14] := by
  rfl

theorem totalCallCount_exact :
    (runs.map (fun run => run.raw.callCount)).sum = 486 := by
  rw [callCounts_exact]
  rfl

private abbrev directRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedRows

private abbrev partialRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate.partialDecodedRows

private abbrev chainedRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafCertificate.decodedRows

private abbrev sharedSteps :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedSteps

private abbrev StepSboxHolds :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.StepSboxHolds

private abbrev directSource :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction.reconstructedSource

private abbrev chainedSource :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.reconstructedSource

def rowsFor : LeafClass → List Wire.Row
  | .direct => directRows
  | .partialStart => partialRows
  | .chained _ => chainedRows

theorem rowsFor_length (kind : LeafClass) : (rowsFor kind).length = 86 := by
  cases kind with
  | direct =>
      exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decoded_rows_length
  | partialStart =>
      calc
        partialRows.length =
            (Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate.canonicalRowHead ++
              Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate.partialDecodedRowTail).length := by
          unfold partialRows
            Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate.partialDecodedRows
            Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate.canonicalRowHead
          have split := congrArg List.length
            (List.take_append_drop 8
              Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate.partialDecodedRowHead)
          simp only [List.length_append, List.length_map] at split ⊢
          omega
        _ = directRows.length := by
          exact congrArg List.length
            Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate.canonical_rows_eq_direct
        _ = 86 :=
          Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decoded_rows_length
  | chained selector =>
      exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafCertificate.decoded_rows_length

structure EmittedBlock where
  finalRowStart : Nat
  kind : LeafClass
  globalIndex : Nat
  rows : List Wire.Row

def EmittedBlock.finalRows (block : EmittedBlock) : List Nat :=
  List.range' block.finalRowStart block.rows.length

def EmittedBlock.Satisfied (block : EmittedBlock)
    (assignment : Fin productionFinalColumns → F) : Prop :=
  ∀ row ∈ block.rows,
    absoluteResidual block.kind block.globalIndex assignment row = 0

def Run.emittedBlockAt (run : Run) (index : Fin run.raw.callCount) :
    EmittedBlock where
  finalRowStart :=
    run.raw.emittedRowStart + index.val *
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows
  kind := run.leafClassAt index.val
  globalIndex := run.globalIndexAt index.val
  rows := rowsFor (run.leafClassAt index.val)

theorem Run.emittedBlockAt_finalRows_exact
    (run : Run) (index : Fin run.raw.callCount) :
    (run.emittedBlockAt index).finalRows =
      List.range'
        (run.raw.emittedRowStart + index.val * 86) 86 := by
  unfold Run.emittedBlockAt EmittedBlock.finalRows
  rw [rowsFor_length]
  rw [Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows_exact]

def sourceFor (kind : LeafClass) (final : FinalAssignment) : SourceAssignment :=
  match kind with
  | .direct | .partialStart => directSource final
  | .chained _ => chainedSource final

theorem rows_imply_step_sboxes
    (run : Run) (index : Nat)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment (selectorColumn (run.leafClassAt index)) = 1)
    (holds : ∀ row ∈ rowsFor (run.leafClassAt index),
      absoluteResidual (run.leafClassAt index) (run.globalIndexAt index)
        assignment row = 0) :
    ∀ step ∈ sharedSteps,
      StepSboxHolds
        (sourceFor (run.leafClassAt index)
          (projectFinalAssignment (run.leafClassAt index)
            (run.globalIndexAt index) assignment)) step := by
  cases index with
  | zero =>
      cases firstClass : run.raw.firstClass with
      | direct =>
          simp only [Run.leafClassAt, firstClass, if_pos, Run.globalIndexAt,
            rowsFor, sourceFor, selectorColumn] at selectorOne holds ⊢
          exact direct_absolute_rows_imply_step_sboxes
            run.globalStart assignment one selectorOne holds
      | partialStart =>
          simp only [Run.leafClassAt, firstClass, if_pos, Run.globalIndexAt,
            rowsFor, sourceFor, selectorColumn] at selectorOne holds ⊢
          exact partial_absolute_rows_imply_step_sboxes
            run.globalStart assignment one selectorOne holds
  | succ index =>
      simp only [Run.leafClassAt, Nat.succ_ne_zero, if_false,
        Run.globalIndexAt, rowsFor, sourceFor, selectorColumn]
        at selectorOne holds ⊢
      exact chained_absolute_rows_imply_step_sboxes
        run.raw.selectorColumn (run.globalStart + index.succ)
        assignment one selectorOne holds

theorem emitted_block_implies_step_sboxes
    (run : Run) (index : Fin run.raw.callCount)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment
        (selectorColumn (run.leafClassAt index.val)) = 1)
    (satisfied : (run.emittedBlockAt index).Satisfied assignment) :
    ∀ step ∈ sharedSteps,
      StepSboxHolds
        (sourceFor (run.leafClassAt index.val)
          (projectFinalAssignment (run.leafClassAt index.val)
            (run.globalIndexAt index.val) assignment)) step := by
  exact rows_imply_step_sboxes run index.val assignment one selectorOne satisfied

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
