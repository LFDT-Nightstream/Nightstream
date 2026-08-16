import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinateAccumulator
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingCompleteRows
import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: source-row semantics for one fixed-position claim-coordinate
overlay.

Owns the exact additive update, carry, and zero-initialization row families,
their composition with one complete coordinate-binding row family, and the
proof that accepted active or inactive overlays implement the corresponding
accumulator step.

Does not own Rust trace conformance, normalized column placement, selective
low-norm lowering, schedule selection, phase-to-overlay field links, Poseidon2
replay, Module-SIS hardness, or recursive lifecycle integration.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 262144

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateAccumulator
open Nightstream.Implementation.R1CS.Program
open Nightstream.Protocol.Nebula.AjtaiBinding
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.ShiftedTernary41V1
open Nightstream.SuperNeo.Concrete

private theorem moduli_eq : goldilocksModulus = goldilocksP := rfl

/-- Source columns owned by one active coordinate overlay. The coordinate
layout owns the partial output columns. The two state maps own private copies
of the carried commitment before and after the phase. -/
structure ActiveLayout where
  coordinate : Layout
  beforeColumn : Fin (shape.rows * shape.degree) → Nat
  afterColumn : Fin (shape.rows * shape.degree) → Nat

def decodedAccumulator
    (assignment : Nat → Nat)
    (columns : Fin (shape.rows * shape.degree) → Nat) : Accumulator :=
  fun output =>
    ⟨assignment (columns output) % goldilocksP,
      Nat.mod_lt _ (by decide)⟩

theorem decodedAccumulator_val
    (assignment : Nat → Nat)
    (columns : Fin (shape.rows * shape.degree) → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (output : Fin (shape.rows * shape.degree)) :
    (decodedAccumulator assignment columns output).val =
      assignment (columns output) := by
  simp [decodedAccumulator, Nat.mod_eq_of_lt (canonical _)]

/-- Exact Rust `enforce_eq(after, before + partial)` source rows. -/
def updateRows (layout : ActiveLayout) : List Row :=
  List.ofFn fun output : Fin (shape.rows * shape.degree) =>
    builderLinearRow (layout.afterColumn output)
      [(layout.beforeColumn output, 1),
       (layout.coordinate.outputColumn output, 1)]

theorem updateRows_length (layout : ActiveLayout) :
    (updateRows layout).length = shape.rows * shape.degree := by
  simp [updateRows]

theorem updateRows_sound
    {layout : ActiveLayout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (updateRows layout) assignment) :
    ∀ output : Fin (shape.rows * shape.degree),
      assignment (layout.afterColumn output) =
        (assignment (layout.beforeColumn output) +
          assignment (layout.coordinate.outputColumn output)) %
            goldilocksP := by
  intro output
  have holds := satisfies _ (List.mem_ofFn.mpr ⟨output, rfl⟩)
  have defined := builderLinearRow_sound canonical one
    (layout.afterColumn output)
    [(layout.beforeColumn output, 1),
     (layout.coordinate.outputColumn output, 1)]
    (by simp [CanonicalTerms]; decide) holds
  simpa [lcEval, Nat.add_comm, Nat.mul_comm] using defined

def zeroPins (layout : ActiveLayout) : List (Nat × Nat) :=
  List.ofFn fun output : Fin (shape.rows * shape.degree) =>
    (layout.beforeColumn output, 0)

/-- Exact Rust chunk-zero `enforce_eq(before, 0)` rows. -/
def zeroRows (layout : ActiveLayout) : List Row :=
  ConstantPins.rows (zeroPins layout)

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true member

theorem zeroRows_sound
    {layout : ActiveLayout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (zeroRows layout) assignment) :
    decodedAccumulator assignment layout.beforeColumn = zeroAccumulator := by
  have valuesCanonical : ConstantPins.ValuesCanonical (zeroPins layout) := by
    intro pin member
    have valueZero : pin.2 = 0 := by
      rcases List.mem_ofFn.mp member with ⟨output, equal⟩
      rw [← equal]
    rw [valueZero]
    decide
  have facts := ConstantPins.sound
    (pins := zeroPins layout) (programRows := zeroRows layout)
    valuesCanonical (by exact rowsIncluded_self _) canonical one satisfies
  funext output
  apply Fin.ext
  rw [decodedAccumulator_val assignment layout.beforeColumn canonical output]
  exact facts (layout.beforeColumn output, 0)
    (List.mem_ofFn.mpr ⟨output, rfl⟩)

/-- Complete active source program. Chunk zero also pins the initial
accumulator to zero. -/
def activeRows
    (production : ProductionSetup) (chunk : Fin claimChunkCount)
    (layout : ActiveLayout) : List Row :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows.rows
      production layout.coordinate ++
    updateRows layout ++
      if chunk.val = 0 then zeroRows layout else []

private theorem coordinateRows_satisfy
    {production : ProductionSetup} {chunk : Fin claimChunkCount}
    {layout : ActiveLayout} {assignment : Nat → Nat}
    (satisfies : Satisfies (activeRows production chunk layout) assignment) :
    Satisfies
      (Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows.rows
        production layout.coordinate)
      assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_left _ (List.mem_append_left _ member))

private theorem updateRows_satisfy
    {production : ProductionSetup} {chunk : Fin claimChunkCount}
    {layout : ActiveLayout} {assignment : Nat → Nat}
    (satisfies : Satisfies (activeRows production chunk layout) assignment) :
    Satisfies (updateRows layout) assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_left _ (List.mem_append_right _ member))

private theorem zeroRows_satisfy
    {production : ProductionSetup} {chunk : Fin claimChunkCount}
    {layout : ActiveLayout} {assignment : Nat → Nat}
    (chunkZero : chunk.val = 0)
    (satisfies : Satisfies (activeRows production chunk layout) assignment) :
    Satisfies (zeroRows layout) assignment := by
  intro row member
  exact satisfies row
    (List.mem_append_right _ (by simpa [chunkZero] using member))

theorem partialCoordinate_val_eq_maskedConcreteBinding
    (production : ProductionSetup) (fields : Fields)
    (chunk : Fin claimChunkCount)
    (output : Fin (shape.rows * shape.degree)) :
    (partialCoordinate production fields chunk output).val =
      (maskedConcreteBinding production fields (chunkMask chunk) output).val := by
  let pair := outputPair output
  have outputExact : outputIndex pair.1 pair.2 = output := by
    exact Equiv.apply_symm_apply _ output
  rw [← outputExact, maskedConcreteBinding_outputIndex]
  simp only [partialCoordinate, outputPair_outputIndex]

/-- Accepted active source rows implement the exact verifier-owned additive
accumulator step for the selected global claim mask. -/
theorem activeRows_imply_step
    {production : ProductionSetup} {fields : Fields}
    {chunk : Fin claimChunkCount} {layout : ActiveLayout}
    {assignment : Nat → Nat}
    (forChunk : ForClaimChunk layout.coordinate chunk)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : ActiveFieldsPlaced layout.coordinate assignment fields)
    (satisfies : Satisfies (activeRows production chunk layout) assignment) :
    StepAt production fields chunk
      (decodedAccumulator assignment layout.beforeColumn)
      (decodedAccumulator assignment layout.afterColumn) := by
  have partials := rows_imply_claimChunkCommitment chunk forChunk canonical one
    placed (coordinateRows_satisfy satisfies)
  have updates := updateRows_sound canonical one
    (updateRows_satisfy satisfies)
  intro output
  let pair := outputPair output
  have outputExact : outputIndex pair.1 pair.2 = output := by
    exact Equiv.apply_symm_apply _ output
  have partialExact :
      assignment (layout.coordinate.outputColumn output) =
        (partialCoordinate production fields chunk output).val := by
    have atPair := partials pair.1 pair.2
    rw [partialCoordinate_val_eq_maskedConcreteBinding]
    simpa [outputExact] using atPair
  apply Fin.ext
  simp only [Fin.val_add, moduli_eq]
  rw [decodedAccumulator_val assignment layout.afterColumn canonical output,
    decodedAccumulator_val assignment layout.beforeColumn canonical output]
  rw [updates output, partialExact]

theorem activeRows_chunkZero_initial
    {production : ProductionSetup} {chunk : Fin claimChunkCount}
    {layout : ActiveLayout} {assignment : Nat → Nat}
    (chunkZero : chunk.val = 0)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (activeRows production chunk layout) assignment) :
    decodedAccumulator assignment layout.beforeColumn = zeroAccumulator := by
  exact zeroRows_sound canonical one (zeroRows_satisfy chunkZero satisfies)

/-- Exact Rust `enforce_eq(after, before)` carry rows. -/
def carryPairs
    (beforeColumn afterColumn : Fin (shape.rows * shape.degree) → Nat) :
    List (Nat × Nat) :=
  List.ofFn fun output : Fin (shape.rows * shape.degree) =>
    (afterColumn output, beforeColumn output)

def carryRows
    (beforeColumn afterColumn : Fin (shape.rows * shape.degree) → Nat) :
    List Row :=
  EqualityPins.rows (carryPairs beforeColumn afterColumn)

theorem carryRows_sound
    {beforeColumn afterColumn : Fin (shape.rows * shape.degree) → Nat}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (carryRows beforeColumn afterColumn) assignment) :
    decodedAccumulator assignment afterColumn =
      decodedAccumulator assignment beforeColumn := by
  have facts := EqualityPins.rows_sound canonical one satisfies
  funext output
  apply Fin.ext
  rw [decodedAccumulator_val assignment afterColumn canonical output,
    decodedAccumulator_val assignment beforeColumn canonical output]
  exact facts (afterColumn output, beforeColumn output)
    (List.mem_ofFn.mpr ⟨output, rfl⟩)

def InactiveChunk (chunk : Fin claimChunkCount) : Prop :=
  ∀ field : Fin fieldCount, claimChunk field ≠ chunk

theorem inactiveChunk_of_gap
    (chunk : Fin claimChunkCount)
    (notZero : chunk.val ≠ 0)
    (gap : chunk.val < 60 ∨ 81 < chunk.val) :
    InactiveChunk chunk := by
  intro field equal
  have range := claimChunk_active_range field
  rw [equal] at range
  rcases range with zero | evaluation
  · exact notZero zero
  · omega

private theorem maskedWitness_eq_zero_of_inactive
    (fields : Fields) (chunk : Fin claimChunkCount)
    (inactive : InactiveChunk chunk) :
    maskedWitness fields (chunkMask chunk) = 0 := by
  funext column coordinate
  by_cases valid : flatIndex column coordinate < fieldCount * digitCount
  · rw [show maskedWitness fields (chunkMask chunk) column coordinate =
        (if chunkMask chunk
            ⟨flatIndex column coordinate / digitCount, by
              unfold fieldCount digitCount at valid ⊢
              omega⟩ then
          signedDigit
            (fields ⟨flatIndex column coordinate / digitCount, by
              unfold fieldCount digitCount at valid ⊢
              omega⟩)
            ⟨flatIndex column coordinate % digitCount,
              Nat.mod_lt _ (by decide)⟩
        else 0) by
          unfold maskedWitness
          rw [dif_pos valid]]
    simp [chunkMask, inactive
      ⟨flatIndex column coordinate / digitCount, by
        unfold fieldCount digitCount at valid ⊢
        omega⟩]
  · unfold maskedWitness
    rw [dif_neg valid]
    rfl

theorem partialCoordinate_eq_zero_of_inactive
    (production : ProductionSetup) (fields : Fields)
    (chunk : Fin claimChunkCount) (inactive : InactiveChunk chunk) :
    partialCoordinate production fields chunk = 0 := by
  funext output
  unfold partialCoordinate
  rw [maskedWitness_eq_zero_of_inactive fields chunk inactive]
  simp [commit]
  rfl

/-- A carry overlay is the exact accumulator step for every verifier-owned
claim chunk with an empty coordinate mask. -/
theorem carryRows_imply_step
    {production : ProductionSetup} {fields : Fields}
    {chunk : Fin claimChunkCount}
    {beforeColumn afterColumn : Fin (shape.rows * shape.degree) → Nat}
    {assignment : Nat → Nat}
    (inactive : InactiveChunk chunk)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (carryRows beforeColumn afterColumn) assignment) :
    StepAt production fields chunk
      (decodedAccumulator assignment beforeColumn)
      (decodedAccumulator assignment afterColumn) := by
  have carried := carryRows_sound canonical one satisfies
  have partialZero := partialCoordinate_eq_zero_of_inactive
    production fields chunk inactive
  intro output
  rw [congrFun carried output]
  have zeroAt : partialCoordinate production fields chunk output = 0 := by
    simpa using congrFun partialZero output
  rw [zeroAt, add_zero]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay
