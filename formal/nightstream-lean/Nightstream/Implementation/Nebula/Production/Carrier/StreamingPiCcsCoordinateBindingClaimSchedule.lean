import Mathlib.Algebra.BigOperators.Fin
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingCompleteRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsStatementChunkSelection

/-!
Contract: exact claim-replay schedule for the production PiCCS
variable-coordinate commitment.

Assurance tier: phase-composition and commitment-binding bridge.

Owns the fixed claim-frame position of every one of the 24,244 PiCCS
variable fields, its unique 1,024-field claim chunk, the exact active-field
list for each of the 98 claim phases, and the proof that all phase-masked
Ajtai commitments add to the direct full-vector commitment.

Does not own generated Rust traces, physical claim-chunk columns, carried
commitment rows, PiCCS-start rows, Module-SIS hardness, or recursive
lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 262144

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule

open scoped BigOperators
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.AjtaiBinding
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

def claimChunkWidth : Nat := 1024
def claimChunkCount : Nat := 98
def claimFrameLength : Nat := 99903
def pointFieldCount : Nat := 52
def pointFrameOffset : Nat := 383
def evaluationFrameOffset : Nat := 71283

/-- Claim-frame position of one field in the selected 24,244-field order.
The first 52 fields are the prior point. The remaining 24,192 fields are the
carried evaluations. -/
def claimFramePosition (field : Fin fieldCount) : Nat :=
  if field.val < pointFieldCount then
    pointFrameOffset + field.val
  else
    evaluationFrameOffset + (field.val - pointFieldCount)

theorem claimFramePosition_lt (field : Fin fieldCount) :
    claimFramePosition field < claimFrameLength := by
  unfold claimFramePosition fieldCount pointFieldCount pointFrameOffset
    evaluationFrameOffset claimFrameLength
  split <;> omega

/-- Verifier-owned claim-chunk index for one selected field. -/
def claimChunk (field : Fin fieldCount) : Fin claimChunkCount :=
  ⟨claimFramePosition field / claimChunkWidth, by
    have positionBound := claimFramePosition_lt field
    unfold claimFrameLength at positionBound
    unfold claimChunkWidth claimChunkCount
    omega⟩

/-- Offset of one selected field inside its claim-replay chunk. -/
def claimChunkOffset (field : Fin fieldCount) : Fin claimChunkWidth :=
  ⟨claimFramePosition field % claimChunkWidth,
    Nat.mod_lt _ (by decide)⟩

theorem claimFramePosition_recompose (field : Fin fieldCount) :
    (claimChunk field).val * claimChunkWidth +
        (claimChunkOffset field).val =
      claimFramePosition field := by
  unfold claimChunk claimChunkOffset
  simpa [Nat.mul_comm] using
    Nat.div_add_mod (claimFramePosition field) claimChunkWidth

/-- The prior point is wholly inside claim chunk zero, starting at offset
383. -/
theorem point_chunk_geometry
    (field : Fin fieldCount) (point : field.val < pointFieldCount) :
    (claimChunk field).val = 0 /\
      (claimChunkOffset field).val = pointFrameOffset + field.val := by
  change claimFramePosition field / claimChunkWidth = 0 /\
    claimFramePosition field % claimChunkWidth =
      pointFrameOffset + field.val
  unfold claimFramePosition
  rw [if_pos point]
  unfold pointFieldCount at point
  unfold claimChunkWidth pointFrameOffset
  omega

/-- Every carried evaluation is in claim chunks 69 through 93. -/
theorem evaluation_chunk_geometry
    (field : Fin fieldCount) (evaluation : pointFieldCount ≤ field.val) :
    69 ≤ (claimChunk field).val /\
      (claimChunk field).val ≤ 93 := by
  have fieldBound := field.isLt
  change 60 ≤ claimFramePosition field / claimChunkWidth /\
    claimFramePosition field / claimChunkWidth ≤ 81
  unfold claimFramePosition
  rw [if_neg (Nat.not_lt.mpr evaluation)]
  unfold pointFieldCount at evaluation
  unfold fieldCount at fieldBound
  unfold evaluationFrameOffset claimChunkWidth pointFieldCount
  omega

/-- Exact mask used by one claim-replay phase. -/
def chunkMask
    (chunk : Fin claimChunkCount) (field : Fin fieldCount) : Bool :=
  decide (claimChunk field = chunk)

/-- Canonical global field order selected by one claim phase. -/
def activeFields (chunk : Fin claimChunkCount) : List (Fin fieldCount) :=
  (List.finRange fieldCount).filter fun field => claimChunk field = chunk

theorem activeFields_nodup (chunk : Fin claimChunkCount) :
    (activeFields chunk).Nodup := by
  exact (List.nodup_finRange fieldCount).filter _

@[simp] theorem mem_activeFields
    (chunk : Fin claimChunkCount) (field : Fin fieldCount) :
    field ∈ activeFields chunk ↔ claimChunk field = chunk := by
  simp [activeFields]

theorem chunkMask_eq_decide_mem
    (chunk : Fin claimChunkCount) (field : Fin fieldCount) :
    chunkMask chunk field = decide (field ∈ activeFields chunk) := by
  simp [chunkMask]

/-- A physical coordinate-binding layout for one claim phase uses exactly
the verifier-owned global positions for that phase. -/
def ForClaimChunk
    (layout : Layout) (chunk : Fin claimChunkCount) : Prop :=
  layout.activeFields = activeFields chunk

theorem layout_selected_eq_chunkMask
    (layout : Layout) (chunk : Fin claimChunkCount)
    (exact : ForClaimChunk layout chunk) :
    layout.selected = chunkMask chunk := by
  funext field
  unfold Layout.selected chunkMask
  rw [exact]
  simp

/-- Accepted complete coordinate rows for one claim phase derive that
phase's exact verifier-owned partial commitment. -/
theorem rows_imply_claimChunkCommitment
    {production : ProductionSetup} {layout : Layout}
    {assignment : Nat → Nat} {fields : Fields}
    (chunk : Fin claimChunkCount)
    (forChunk : ForClaimChunk layout chunk)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : ActiveFieldsPlaced layout assignment fields)
    (satisfies : Satisfies
      (ProductionStreamingPiCcsCoordinateBindingCompleteRows.rows
        production layout)
      assignment) :
    ∀ output : Fin verifierRows, ∀ coordinate : Fin ringDegree,
      assignment (layout.outputColumn (outputIndex output coordinate)) =
        (maskedConcreteBinding production fields (chunkMask chunk)
          (outputIndex output coordinate)).val := by
  have exactRows :=
    ProductionStreamingPiCcsCoordinateBindingCompleteRows.rows_sound
      canonical one placed satisfies
  intro output coordinate
  rw [← layout_selected_eq_chunkMask layout chunk forChunk]
  exact exactRows.output output coordinate

private theorem chunkMask_sum
    (field : Fin fieldCount) (value : Int) :
    (∑ chunk : Fin claimChunkCount,
        if chunkMask chunk field then value else 0) = value := by
  classical
  rw [Fintype.sum_eq_single (claimChunk field)]
  · simp [chunkMask]
  · intro other different
    simp [chunkMask, Ne.symm different]

/-- The 98 verifier-owned masks form one exact partition of the complete
coordinate witness. No field is skipped or counted twice. -/
theorem maskedWitness_sum
    (fields : Fields) :
    (fun column coefficient =>
      ∑ chunk : Fin claimChunkCount,
        maskedWitness fields (chunkMask chunk) column coefficient) =
      coordinateWitness fields := by
  funext column coefficient
  unfold maskedWitness coordinateWitness
  by_cases valid : flatIndex column coefficient < fieldCount * digitCount
  · simp only [dif_pos valid]
    exact chunkMask_sum
      ⟨flatIndex column coefficient / digitCount, by
        unfold fieldCount digitCount at valid ⊢
        omega⟩
      (signedDigit
        (fields ⟨flatIndex column coefficient / digitCount, by
          unfold fieldCount digitCount at valid ⊢
          omega⟩)
        ⟨flatIndex column coefficient % digitCount,
          Nat.mod_lt _ (by decide)⟩)
  · simp [valid]

/-- Additive form used by a carried commitment accumulator. The sum of all
86 phase outputs is exactly the direct 21,220-field Ajtai commitment. -/
theorem commitments_sum
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (fields : Fields) :
    (fun row =>
      ∑ chunk : Fin claimChunkCount,
        commit matrix coefficientMap
          (maskedWitness fields (chunkMask chunk)) row) =
      commit matrix coefficientMap (coordinateWitness fields) := by
  funext row
  unfold commit
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro column _
  rw [← Finset.sum_mul, ← map_sum]
  apply congrArg (fun value => value * matrix row column)
  apply congrArg coefficientMap
  funext coefficient
  exact congrFun (congrFun (maskedWitness_sum fields) column) coefficient

/-- Concrete production setup specialization of `commitments_sum`. -/
theorem concrete_commitments_sum
    (production : ProductionSetup) (fields : Fields) :
    (fun row =>
      ∑ chunk : Fin claimChunkCount,
        commit (seededMatrix production.setup) coefficientMap
          (maskedWitness fields (chunkMask chunk)) row) =
      bindingMap (seededMatrix production.setup) coefficientMap fields := by
  exact commitments_sum (seededMatrix production.setup) coefficientMap fields

/-- Only claim chunk zero and evaluation chunks 69 through 93 can contain a
selected PiCCS field. -/
theorem claimChunk_active_range (field : Fin fieldCount) :
    (claimChunk field).val = 0 \/
      (69 ≤ (claimChunk field).val /\ (claimChunk field).val ≤ 93) := by
  by_cases point : field.val < pointFieldCount
  · exact Or.inl (point_chunk_geometry field point).1
  · exact Or.inr
      (evaluation_chunk_geometry field (Nat.le_of_not_gt point))

/-- The 26 nonempty coordinate phases contain all 24,244 active fields.
Their source-row total remains a union cost; one recursive step uses only one
local phase. -/
theorem active_phase_row_census :
    26 * (41 + 2 + 108) + fieldCount * 124 = 3010182 := by
  decide

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
