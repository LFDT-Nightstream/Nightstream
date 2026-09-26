import NightstreamFPrime.Layout.PiRlcWideSampler.RangePlan
import NightstreamFPrime.Layout.LowNormBlock

/-! Three disjoint retained blocks for one scalar: 256 canonical bits,
eight canonical field auxiliaries, and 353 result bits. Source columns 0–3
are supplied by Poseidon forms. Temporary source columns reconstruct as zero and have no committed slot. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.Retained

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def canonicalBits : LowNormBlock.Block 2025 where
  kind := .bit
  slotCount := 256
  source := fun index => ⟨1408 + 66 * (index.val / 64) + index.val % 64, by omega⟩

def canonicalFields : LowNormBlock.Block 2025 where
  kind := .field
  slotCount := 8
  source := fun index => ⟨1408 + 66 * (index.val / 2) + 64 + index.val % 2, by omega⟩

def resultBits : LowNormBlock.Block 2025 where
  kind := .bit
  slotCount := 353
  source := fun index => ⟨1672 + index.val, by omega⟩

theorem coordinateCounts : canonicalBits.coordinateCount = 256 ∧
    canonicalFields.coordinateCount = 328 ∧ resultBits.coordinateCount = 353 := by decide

def Fits (columns start : Nat) : Prop := start + 937 ≤ columns

private theorem bits_fit {columns start : Nat} (fits : Fits columns start) :
    start + canonicalBits.coordinateCount ≤ columns := by
  rw [coordinateCounts.1]
  exact le_trans (by omega) fits

private theorem fields_fit {columns start : Nat} (fits : Fits columns start) :
    (start + 256) + canonicalFields.coordinateCount ≤ columns := by
  rw [coordinateCounts.2.1]
  exact le_trans (by omega) fits

private theorem results_fit {columns start : Nat} (fits : Fits columns start) :
    (start + 584) + resultBits.coordinateCount ≤ columns := by
  rw [coordinateCounts.2.2]
  exact le_trans (by omega) fits

structure Encodes {columns : Nat} (start : Nat) (fits : Fits columns start)
    (assignment : Assignment F columns) (env : Env) : Prop where
  bits : canonicalBits.EncodesAt start (bits_fit fits) assignment (fun column => env column.val)
  fields : canonicalFields.EncodesAt (start + 256) (fields_fit fits) assignment (fun column => env column.val)
  results : resultBits.EncodesAt (start + 584) (results_fit fits) assignment (fun column => env column.val)

def sourceMap {columns : Nat} (start : Nat) (fits : Fits columns start)
    (input : Fin 4 → SparseForm columns) : SourceCompiler.SourceMap 2025 columns where
  form := fun column =>
    if caller : column.val < 4 then input ⟨column.val, caller⟩
    else if temporary : column.val < 1408 then .empty
    else if canonical : column.val < 1672 then
      if bit : (column.val - 1408) % 66 < 64 then
        canonicalBits.form start (bits_fit fits)
          ⟨64 * ((column.val - 1408) / 66) + (column.val - 1408) % 66, by change _ < 256; omega⟩
      else
        canonicalFields.form (start + 256) (fields_fit fits)
          ⟨2 * ((column.val - 1408) / 66) + (column.val - 1408) % 66 - 64, by change _ < 8; omega⟩
    else
      resultBits.form (start + 584) (results_fit fits) ⟨column.val - 1672, by change _ < 353; omega⟩

theorem sourceMap_preserves {columns : Nat} (start : Nat) (fits : Fits columns start)
    (input : Fin 4 → SparseForm columns) (assignment : Assignment F columns) (env : Env)
    (inputs : ∀ lane, (input lane).eval assignment = env lane.val)
    (helpers : ∀ column, 4 ≤ column → column < 1408 → env column = 0)
    (encoded : Encodes start fits assignment env) :
    (sourceMap start fits input).Preserves assignment env := by
  intro column
  unfold sourceMap
  dsimp only
  split_ifs with caller temporary canonical bit
  · exact inputs ⟨column.val, caller⟩
  · rw [SparseForm.empty_eval]
    exact (helpers column.val (by omega) temporary).symm
  · rw [LowNormBlock.Block.form_eval _ _ _ _ _ encoded.bits]
    apply congrArg env
    change 1408 + 66 * ((64 * ((column.val - 1408) / 66) + (column.val - 1408) % 66) / 64) +
      (64 * ((column.val - 1408) / 66) + (column.val - 1408) % 66) % 64 = column.val
    omega
  · rw [LowNormBlock.Block.form_eval _ _ _ _ _ encoded.fields]
    apply congrArg env
    change 1408 + 66 * ((2 * ((column.val - 1408) / 66) + (column.val - 1408) % 66 - 64) / 2) + 64 +
      (2 * ((column.val - 1408) / 66) + (column.val - 1408) % 66 - 64) % 2 = column.val
    omega
  · rw [LowNormBlock.Block.form_eval _ _ _ _ _ encoded.results]
    apply congrArg env
    change 1672 + (column.val - 1672) = column.val
    omega

end NightstreamFPrime.Layout.PiRlcWideSampler.Retained
