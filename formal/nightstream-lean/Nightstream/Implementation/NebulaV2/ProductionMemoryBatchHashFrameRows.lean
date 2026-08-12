import Nightstream.Implementation.NebulaV2.MemoryClaimHashFrameRows
import Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonBinding
import Nightstream.Implementation.NebulaV2.ProductionMemoryCheckedBatchRows

/-!
Contract: exact generated input frame for one field-native memory batch hash.

The eight candidate-specific constants are pinned by rows. The remaining
columns are the 83 typed fields of every row-derived memory suffix in exact
batch order. No digest or typed batch equality is an input to the soundness
theorem.

Does not own the Poseidon2 trace, the CCS public link, absolute generated
columns, Rust refinement, candidate selection, or a verifier key.

Emits constraints: yes, for the eight fixed prefix values.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryBatchHashFrameRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonBinding
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

structure Layout (candidate : Id) where
  memory : ProductionMemoryCheckedBatchRows.Layout candidate
  prefixStart : Nat

def Layout.prefixColumn {candidate : Id}
    (layout : Layout candidate) (index : Nat) : Nat :=
  layout.prefixStart + index

def prefixPins {candidate : Id} (layout : Layout candidate) : List (Nat × Nat) :=
  [ (layout.prefixColumn 0, domainTag)
  , (layout.prefixColumn 1, frameVersion)
  , (layout.prefixColumn 2, 3)
  , (layout.prefixColumn 3, version candidate)
  , (layout.prefixColumn 4, checkedStepsPerFreshClaim candidate)
  , (layout.prefixColumn 5, 1)
  , (layout.prefixColumn 6, MemoryWireGeometry.stepPublicBits)
  , (layout.prefixColumn 7, MemoryClaimCodec.schema.length)
  ]

def prefixColumns {candidate : Id} (layout : Layout candidate) : List Nat :=
  (prefixPins layout).map Prod.fst

def prefixValues {candidate : Id} (layout : Layout candidate) : List Nat :=
  (prefixPins layout).map Prod.snd

theorem prefixValues_exact
    {candidate : Id} (layout : Layout candidate) :
    prefixValues layout = fixedPrefix candidate := by
  rfl

theorem prefixColumns_length
    {candidate : Id} (layout : Layout candidate) :
    (prefixColumns layout).length = 8 := by
  simp [prefixColumns, prefixPins]

theorem prefixPins_valuesCanonical
    {candidate : Id} (layout : Layout candidate) :
    ConstantPins.ValuesCanonical (prefixPins layout) := by
  intro pin member
  simp only [prefixPins, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    cases candidate <;>
      norm_num [domainTag, frameVersion, version,
        checkedStepsPerFreshClaim, MemoryWireGeometry.stepPublicBits_exact,
        MemoryClaimPoseidonBinding.schema_length_exact, goldilocksP]

def claimLayout {candidate : Id} (layout : Layout candidate)
    (index : Fin (checkedStepsPerFreshClaim candidate)) :
    MemoryClaimHashFrameRows.Layout where
  claim := (layout.memory.steps index).claim.reference
  prefixStart := layout.prefixStart

def claimColumnsAt {candidate : Id} (layout : Layout candidate)
    (index : Fin (checkedStepsPerFreshClaim candidate)) : List Nat :=
  MemoryClaimHashFrameRows.claimColumns (claimLayout layout index)

theorem claimColumnsAt_length
    {candidate : Id} (layout : Layout candidate)
    (index : Fin (checkedStepsPerFreshClaim candidate)) :
    (claimColumnsAt layout index).length = 83 := by
  exact MemoryClaimHashFrameRows.claimColumns_length _

def batchColumns {candidate : Id} (layout : Layout candidate) : List Nat :=
  (List.ofFn fun index => claimColumnsAt layout index).flatten

private theorem flatten_ofFn_length
    {Alpha : Type} {count width : Nat} (blocks : Fin count -> List Alpha)
    (each : forall index, (blocks index).length = width) :
    (List.ofFn blocks).flatten.length = count * width := by
  rw [List.length_flatten]
  have constant : forall value, value ∈ (List.ofFn blocks).map List.length ->
      value = width := by
    intro value member
    rcases List.mem_map.mp member with ⟨block, blockMember, rfl⟩
    rcases List.mem_ofFn.mp blockMember with ⟨index, rfl⟩
    exact each index
  rw [List.sum_eq_card_nsmul _ width constant]
  simp

theorem batchColumns_length
    {candidate : Id} (layout : Layout candidate) :
    (batchColumns layout).length =
      checkedStepsPerFreshClaim candidate * 83 := by
  exact flatten_ofFn_length _ (claimColumnsAt_length layout)

def inputColumns {candidate : Id} (layout : Layout candidate) : List Nat :=
  prefixColumns layout ++ batchColumns layout

theorem inputColumns_length
    {candidate : Id} (layout : Layout candidate) :
    (inputColumns layout).length = frameFieldCount candidate := by
  rw [inputColumns, List.length_append, prefixColumns_length,
    batchColumns_length]
  rfl

def rows {candidate : Id} (layout : Layout candidate) : List Row :=
  ConstantPins.rows (prefixPins layout)

theorem rows_length_exact
    {candidate : Id} (layout : Layout candidate) :
    (rows layout).length = 8 := by
  simp [rows, ConstantPins.rows, prefixPins]

private theorem selfIncluded (program : List Row) :
    rowsIncluded program program = true := by
  rw [rowsIncluded, List.all_eq_true]
  intro row member
  exact decide_eq_true member

private theorem prefixFacts
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    ∀ pin ∈ prefixPins layout, assignment pin.1 = pin.2 := by
  exact ConstantPins.sound (prefixPins_valuesCanonical layout)
    (selfIncluded (rows layout)) canonical one satisfied

theorem prefix_column_values
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    (prefixColumns layout).map assignment = fixedPrefix candidate := by
  rw [← prefixValues_exact layout]
  simp only [prefixColumns, prefixValues, List.map_map]
  apply List.map_congr_left
  intro pin member
  exact prefixFacts canonical one satisfied pin member

theorem claim_column_values_at
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat} {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result
      layout.memory assignment headers)
    (index : Fin (checkedStepsPerFreshClaim candidate)) :
    (claimColumnsAt layout index).map assignment =
      MemoryClaimPoseidonBinding.claimFields (result.claim index) := by
  exact MemoryClaimHashFrameRows.claim_column_values
    (result.claimParsed index)

theorem batch_column_values
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat} {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result
      layout.memory assignment headers) :
    (batchColumns layout).map assignment =
      batchFields result.suffixBatch := by
  rw [batchColumns, List.map_flatten, batchFields,
    ProductionMemoryBatchPoseidonBinding.claimBlocks]
  simp only [ProductionMemoryCheckedBatchRows.Result.suffixBatch]
  rw [List.map_ofFn, List.map_ofFn]
  congr 1
  apply congrArg List.ofFn
  funext index
  simpa [Function.comp_apply] using claim_column_values_at result index

/-- The ordered assigned columns are the exact candidate frame of the same
typed batch derived by the checked-memory rows. -/
theorem input_column_values
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat} {headers : FPrime.ChainHeaders Digest.Value}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (result : ProductionMemoryCheckedBatchRows.Result
      layout.memory assignment headers)
    (satisfied : Satisfies (rows layout) assignment) :
    (inputColumns layout).map assignment = frame result.suffixBatch := by
  rw [inputColumns, List.map_append,
    prefix_column_values canonical one satisfied,
    batch_column_values result]
  rfl

end Nightstream.Implementation.NebulaV2.ProductionMemoryBatchHashFrameRows
