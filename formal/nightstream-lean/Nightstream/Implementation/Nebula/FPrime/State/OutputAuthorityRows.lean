import Nightstream.Implementation.Nebula.FPrime.State.OutputFrameRows
import Nightstream.Implementation.Nebula.Core.U64HalvesRows

/-!
Contract: exact authority placement for all 26 non-memory fields in the V2
stateful-with-Nebula state-output frame.

Assurance tier: implementation model.

Owns the typed payload, its exact field order, three canonical unsigned-64
split encodings, structural identity with the authoritative recursive-state
columns, row soundness, and honest local completeness.

Does not own the recursive transition that computes these state values, the
carry digest, either Poseidon2 sponge, absolute generated columns, or Rust
conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.StateOutputAuthorityRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open Nightstream.Protocol.FPrime

abbrev Digest := Fin 4 → Nat

/-- Exact non-memory payload of the mandatory stateful V2 XOut preimage. -/
structure Payload where
  vkFsDigest : Digest
  piCcsHeader : Digest
  chunkCount : Nat
  stepCount : Nat
  pc : Nat
  currentBoundary : Digest
  semanticState : Digest
  accumulatorDigest : Digest
deriving DecidableEq

def Payload.toXOutPreimage (payload : Payload) (carryDigest : Digest) :
    XOut.XOutPreimage Digest Digest Digest where
  vkFsDigest := payload.vkFsDigest
  piCcsHeader := payload.piCcsHeader
  chunkCount := payload.chunkCount
  stepCount := payload.stepCount
  pc := payload.pc
  currentBoundary := payload.currentBoundary
  semanticState := some payload.semanticState
  construction2Accumulator := payload.accumulatorDigest
  nebula := some carryDigest

@[simp] theorem Payload.toXOutPreimage_stateful
    (payload : Payload) (carryDigest : Digest) :
    (payload.toXOutPreimage carryDigest).semanticState =
        some payload.semanticState ∧
      (payload.toXOutPreimage carryDigest).nebula = some carryDigest := by
  simp [Payload.toXOutPreimage]

/-- The exact 26 fields between the outer domain tag and Nebula marker. -/
def payloadFields (payload : Payload) : List Nat :=
  List.ofFn payload.vkFsDigest ++
  List.ofFn payload.piCcsHeader ++
  u64Halves payload.chunkCount ++
  u64Halves payload.stepCount ++
  u64Halves payload.pc ++
  List.ofFn payload.currentBoundary ++
  List.ofFn payload.semanticState ++
  List.ofFn payload.accumulatorDigest

theorem payloadFields_length (payload : Payload) :
    (payloadFields payload).length = 26 := by
  simp [payloadFields, u64Halves]

/-- The 26-field typed payload encoding is injective. In particular, the
three unsigned words cannot use a second field encoding of the same value. -/
theorem payloadFields_injective : Function.Injective payloadFields := by
  intro left right equal
  have vkEqual :
      List.ofFn left.vkFsDigest = List.ofFn right.vkFsDigest := by
    have selected := congrArg (fun values => values.take 4) equal
    simpa [payloadFields, u64Halves] using selected
  have headerEqual :
      List.ofFn left.piCcsHeader = List.ofFn right.piCcsHeader := by
    have selected := congrArg (fun values => (values.drop 4).take 4) equal
    simpa [payloadFields, u64Halves] using selected
  have chunkEqual : u64Halves left.chunkCount = u64Halves right.chunkCount := by
    have selected := congrArg (fun values => (values.drop 8).take 2) equal
    simpa [payloadFields, u64Halves] using selected
  have stepEqual : u64Halves left.stepCount = u64Halves right.stepCount := by
    have selected := congrArg (fun values => (values.drop 10).take 2) equal
    simpa [payloadFields, u64Halves] using selected
  have pcEqual : u64Halves left.pc = u64Halves right.pc := by
    have selected := congrArg (fun values => (values.drop 12).take 2) equal
    simpa [payloadFields, u64Halves] using selected
  have boundaryEqual :
      List.ofFn left.currentBoundary = List.ofFn right.currentBoundary := by
    have selected := congrArg (fun values => (values.drop 14).take 4) equal
    simpa [payloadFields, u64Halves] using selected
  have semanticEqual :
      List.ofFn left.semanticState = List.ofFn right.semanticState := by
    have selected := congrArg (fun values => (values.drop 18).take 4) equal
    simpa [payloadFields, u64Halves] using selected
  have accumulatorEqual :
      List.ofFn left.accumulatorDigest =
        List.ofFn right.accumulatorDigest := by
    have selected := congrArg (fun values => (values.drop 22).take 4) equal
    simpa [payloadFields, u64Halves] using selected
  rw [Payload.mk.injEq]
  exact ⟨List.ofFn_injective vkEqual,
    List.ofFn_injective headerEqual,
    U64HalvesRows.u64Halves_injective chunkEqual,
    U64HalvesRows.u64Halves_injective stepEqual,
    U64HalvesRows.u64Halves_injective pcEqual,
    List.ofFn_injective boundaryEqual,
    List.ofFn_injective semanticEqual,
    List.ofFn_injective accumulatorEqual⟩

/-- Exact 32-field stateful-with-Nebula outer frame. -/
def fullFrame (payload : Payload) (carryDigest : Digest) : List Nat :=
  [StateOutputFrameRows.domainTag] ++ payloadFields payload ++
    [StateOutputFrameRows.nebulaMarker] ++ List.ofFn carryDigest

theorem fullFrame_length (payload : Payload) (carryDigest : Digest) :
    (fullFrame payload carryDigest).length = 32 := by
  simp [fullFrame, payloadFields_length]

/-- Equality of complete canonical source messages recovers both independent
typed inputs before any collision-resistance assumption is used. -/
theorem payload_and_carry_eq_of_fullFrame_eq
    {leftPayload rightPayload : Payload}
    {leftCarry rightCarry : Digest}
    (equal : fullFrame leftPayload leftCarry =
      fullFrame rightPayload rightCarry) :
    leftPayload = rightPayload ∧ leftCarry = rightCarry := by
  have payloadEqual : payloadFields leftPayload = payloadFields rightPayload := by
    have selected := congrArg (fun values => (values.drop 1).take 26) equal
    simpa [fullFrame, payloadFields_length] using selected
  have carryEqual : List.ofFn leftCarry = List.ofFn rightCarry := by
    have selected := congrArg (fun values => values.drop 28) equal
    simpa [fullFrame, payloadFields, u64Halves] using selected
  exact ⟨payloadFields_injective payloadEqual,
    List.ofFn_injective carryEqual⟩

structure Layout where
  frame : StateOutputFrameRows.Layout
  vkFsDigestColumn : Fin 4 → Nat
  piCcsHeaderColumn : Fin 4 → Nat
  chunkCount : U64HalvesRows.Layout
  stepCount : U64HalvesRows.Layout
  pc : U64HalvesRows.Layout
  currentBoundaryColumn : Fin 4 → Nat
  semanticStateColumn : Fin 4 → Nat
  accumulatorDigestColumn : Fin 4 → Nat

def rows (layout : Layout) : List Row :=
  U64HalvesRows.rows layout.chunkCount ++
    U64HalvesRows.rows layout.stepCount ++
    U64HalvesRows.rows layout.pc

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 198 := by
  simp [rows, U64HalvesRows.rows_length_exact]

/-- This certificate contains only column identities. It does not contain a
payload value, a row-satisfaction conclusion, or a state-transition fact. -/
structure Layout.Valid (layout : Layout) : Prop where
  exactVkFsDigestColumns :
    List.ofFn layout.frame.vkFsDigestColumn =
      List.ofFn layout.vkFsDigestColumn
  exactPiCcsHeaderColumns :
    List.ofFn layout.frame.piCcsHeaderColumn =
      List.ofFn layout.piCcsHeaderColumn
  exactChunkCountColumns :
    List.ofFn layout.frame.chunkCountHalfColumn =
      [layout.chunkCount.lowColumn, layout.chunkCount.highColumn]
  exactStepCountColumns :
    List.ofFn layout.frame.stepCountHalfColumn =
      [layout.stepCount.lowColumn, layout.stepCount.highColumn]
  exactPcColumns :
    List.ofFn layout.frame.pcHalfColumn =
      [layout.pc.lowColumn, layout.pc.highColumn]
  exactCurrentBoundaryColumns :
    List.ofFn layout.frame.currentBoundaryColumn =
      List.ofFn layout.currentBoundaryColumn
  exactSemanticStateColumns :
    List.ofFn layout.frame.semanticStateColumn =
      List.ofFn layout.semanticStateColumn
  exactAccumulatorDigestColumns :
    List.ofFn layout.frame.accumulatorDigestColumn =
      List.ofFn layout.accumulatorDigestColumn

def payload (layout : Layout) (assignment : Nat → Nat) : Payload where
  vkFsDigest := fun lane => assignment (layout.vkFsDigestColumn lane)
  piCcsHeader := fun lane => assignment (layout.piCcsHeaderColumn lane)
  chunkCount := U64HalvesRows.value layout.chunkCount assignment
  stepCount := U64HalvesRows.value layout.stepCount assignment
  pc := U64HalvesRows.value layout.pc assignment
  currentBoundary := fun lane => assignment (layout.currentBoundaryColumn lane)
  semanticState := fun lane => assignment (layout.semanticStateColumn lane)
  accumulatorDigest := fun lane =>
    assignment (layout.accumulatorDigestColumn lane)

private theorem chunk_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (U64HalvesRows.rows layout.chunkCount) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem step_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (U64HalvesRows.rows layout.stepCount) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem pc_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (U64HalvesRows.rows layout.pc) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem mapped_ofFn_eq
    {left right : Fin 4 → Nat} {assignment : Nat → Nat}
    (equal : List.ofFn left = List.ofFn right) :
    (List.ofFn left).map assignment =
      List.ofFn (fun lane => assignment (right lane)) := by
  rw [equal]
  exact (List.ofFn_comp' right assignment).symm

private theorem mapped_halves_eq
    {left : Fin 2 → Nat} {word : U64HalvesRows.Layout}
    {assignment : Nat → Nat}
    (columns : List.ofFn left = [word.lowColumn, word.highColumn])
    (halves : [assignment word.lowColumn, assignment word.highColumn] =
      u64Halves (U64HalvesRows.value word assignment)) :
    (List.ofFn left).map assignment =
      u64Halves (U64HalvesRows.value word assignment) := by
  rw [columns]
  exact halves

/-- Satisfying the authority rows makes the exact 26 outer-frame columns the
typed encoding of the same recursive-state payload. -/
theorem payload_column_values
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    (StateOutputFrameRows.payloadColumns layout.frame).map assignment =
      payloadFields (payload layout assignment) := by
  have vk := mapped_ofFn_eq valid.exactVkFsDigestColumns
    (assignment := assignment)
  have header := mapped_ofFn_eq valid.exactPiCcsHeaderColumns
    (assignment := assignment)
  have chunk := mapped_halves_eq valid.exactChunkCountColumns
    (U64HalvesRows.half_column_values canonical one (chunk_rows_hold holds))
  have step := mapped_halves_eq valid.exactStepCountColumns
    (U64HalvesRows.half_column_values canonical one (step_rows_hold holds))
  have programCounter := mapped_halves_eq valid.exactPcColumns
    (U64HalvesRows.half_column_values canonical one (pc_rows_hold holds))
  have boundary := mapped_ofFn_eq valid.exactCurrentBoundaryColumns
    (assignment := assignment)
  have semantic := mapped_ofFn_eq valid.exactSemanticStateColumns
    (assignment := assignment)
  have accumulator := mapped_ofFn_eq valid.exactAccumulatorDigestColumns
    (assignment := assignment)
  simp only [StateOutputFrameRows.payloadColumns, List.map_append]
  rw [vk, header, chunk, step, programCounter, boundary, semantic, accumulator]
  rfl

/-- The assignment-shaped outer frame is definitionally the independent
typed frame after the authority rows are checked. -/
theorem sourceFrame_eq_fullFrame
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (carryDigest : Digest) :
    StateOutputFrameRows.sourceFrame layout.frame assignment carryDigest =
      fullFrame (payload layout assignment) carryDigest := by
  unfold StateOutputFrameRows.sourceFrame fullFrame
  rw [payload_column_values valid canonical one holds]

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (payloadValue : Payload) : Prop where
  vkFsDigestPlaced : ∀ lane,
    assignment (layout.vkFsDigestColumn lane) = payloadValue.vkFsDigest lane
  piCcsHeaderPlaced : ∀ lane,
    assignment (layout.piCcsHeaderColumn lane) = payloadValue.piCcsHeader lane
  chunkCount : U64HalvesRows.Honest layout.chunkCount assignment
    payloadValue.chunkCount
  stepCount : U64HalvesRows.Honest layout.stepCount assignment
    payloadValue.stepCount
  pc : U64HalvesRows.Honest layout.pc assignment payloadValue.pc
  currentBoundaryPlaced : ∀ lane,
    assignment (layout.currentBoundaryColumn lane) =
      payloadValue.currentBoundary lane
  semanticStatePlaced : ∀ lane,
    assignment (layout.semanticStateColumn lane) = payloadValue.semanticState lane
  accumulatorDigestPlaced : ∀ lane,
    assignment (layout.accumulatorDigestColumn lane) =
      payloadValue.accumulatorDigest lane

theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat} {payloadValue : Payload}
    (one : assignment 0 = 1)
    (honest : Honest layout assignment payloadValue) :
    Satisfies (rows layout) assignment := by
  intro row member
  simp only [rows, List.mem_append] at member
  rcases member with (chunkMember | stepMember) | pcMember
  · exact U64HalvesRows.rows_complete one honest.chunkCount row chunkMember
  · exact U64HalvesRows.rows_complete one honest.stepCount row stepMember
  · exact U64HalvesRows.rows_complete one honest.pc row pcMember

end Nightstream.Implementation.Nebula.StateOutputAuthorityRows
