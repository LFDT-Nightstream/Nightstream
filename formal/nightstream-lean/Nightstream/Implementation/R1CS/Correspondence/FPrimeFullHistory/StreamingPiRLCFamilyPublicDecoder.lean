import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyDigest

/-!
Contract: exact public-prefix value decoder for one normalized PiRLC family
body.

Owns the Rust-checked direct source run `1..640 -> 1..640`, the seven public
zero-padding columns, and the pointwise and whole-word transport from raw
public bit columns to the final selective assignment.

Does not own private source decoding, row acceptance, lifecycle public-input
construction, selectors, or collision resistance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicDecoder

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact
open Nightstream.SuperNeo.Concrete

def finalColumns : Nat := 8858862

def publicColumns : Nat := 648

theorem finalColumns_positive : 0 < finalColumns := by decide

/-- Compact direct decoder run emitted by Rust for both parity arms. -/
def expectedPublicRun :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema.RawResidualBatch :=
  { sourceStart := 1
    instanceCount := 1
    instanceStride := 0
    width := 640
    resolution := .direct 1 1 1 false }

/-- Both generated parity decoders start with the same direct public run. -/
theorem generated_public_run_exact :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.evenArm.residualBatches.head? =
        some expectedPublicRun /\
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.oddArm.residualBatches.head? =
        some expectedPublicRun := by
  exact ⟨rfl, rfl⟩

/-- Normalized source and final column for one ordered public bit. -/
def publicBitIndex (word : Fin 10) (bit : Fin 64) : Nat :=
  1 + word.val * 64 + bit.val

def finalPublicBitColumn (word : Fin 10) (bit : Fin 64) :
    Fin finalColumns :=
  ⟨publicBitIndex word bit, by
    unfold publicBitIndex finalColumns
    omega⟩

def finalPaddingColumn (padding : Fin 7) : Fin finalColumns :=
  ⟨641 + padding.val, by
    unfold finalColumns
    omega⟩

/-- Exact value relation implemented by Rust's public-prefix encoder. -/
structure PublicAssignmentBinding
    (source : Nat → Nat) (kind : ArmKind)
    (final : Fin finalColumns → F) : Prop where
  constantOne : final ⟨0, finalColumns_positive⟩ = 1
  bit : ∀ word bit,
    final (finalPublicBitColumn word bit) =
      NumericRowBridge.residue
        (source (publicBitSourceColumn kind word bit))
  padding : ∀ padding, final (finalPaddingColumn padding) = 0

def finalBitNat
    (final : Fin finalColumns → F) (word : Fin 10) (index : Nat) : Nat :=
  if bounded : index < 64 then
    (final (finalPublicBitColumn word ⟨index, bounded⟩)).val
  else
    0

/-- Integer represented by one final 64-bit public block. -/
def finalWordValue
    (final : Fin finalColumns → F) (word : Fin 10) : Nat :=
  (List.range 64).foldl
    (fun value index => value + 2 ^ index * finalBitNat final word index) 0

private theorem finalBitNat_eq_sourceBit
    (source : Nat → Nat) (kind : ArmKind)
    (final : Fin finalColumns → F)
    (canonical : ∀ column, source column < goldilocksP)
    (one : source 0 = 1)
    (satisfied : (armFor kind).Satisfied source)
    (binding : PublicAssignmentBinding source kind final)
    (word : Fin 10) (index : Nat) (bounded : index < 64) :
    finalBitNat final word index =
      CanonicalU64RecipeSound.bitValue source
        (publicWordCall kind word).layout index := by
  let bit : Fin 64 := ⟨index, bounded⟩
  have sourceBinary := public_bit_binary kind word bit source canonical one
    satisfied
  have sourceLt :
      source (publicBitSourceColumn kind word bit) <
        Nightstream.SuperNeo.Concrete.goldilocksModulus := by
    have modulusLarge :
        2 < Nightstream.SuperNeo.Concrete.goldilocksModulus := by decide
    omega
  have unfoldedSourceLt :
      source
          (CanonicalU64Recipe.bitColumn
            (publicWordCall kind word).layout index) <
        Nightstream.SuperNeo.Concrete.goldilocksModulus := by
    simpa [publicBitSourceColumn, bit] using sourceLt
  have values := congrArg Fin.val (binding.bit word bit)
  simpa [finalBitNat, bounded, bit, NumericRowBridge.residue,
    Nat.mod_eq_of_lt unfoldedSourceLt, publicBitSourceColumn,
    CanonicalU64RecipeSound.bitValue] using values

private theorem foldl_finalBits_eq_sourceBits
    (source : Nat → Nat) (kind : ArmKind)
    (final : Fin finalColumns → F)
    (canonical : ∀ column, source column < goldilocksP)
    (one : source 0 = 1)
    (satisfied : (armFor kind).Satisfied source)
    (binding : PublicAssignmentBinding source kind final)
    (word : Fin 10) (indices : List Nat)
    (bounded : ∀ index ∈ indices, index < 64) (initial : Nat) :
    indices.foldl
        (fun value index => value + 2 ^ index * finalBitNat final word index)
        initial =
      indices.foldl
        (fun value index => value + 2 ^ index *
          CanonicalU64RecipeSound.bitValue source
            (publicWordCall kind word).layout index)
        initial := by
  induction indices generalizing initial with
  | nil => rfl
  | cons index tail inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [finalBitNat_eq_sourceBit source kind final canonical one satisfied
        binding word index (bounded index (by simp))]
      apply inductionHypothesis
      intro next member
      exact bounded next (by simp [member])

/-- Every final public word is exactly the Rust-selected raw source word. -/
theorem finalWordValue_eq_publicWordValue
    (source : Nat → Nat) (kind : ArmKind)
    (final : Fin finalColumns → F)
    (canonical : ∀ column, source column < goldilocksP)
    (one : source 0 = 1)
    (satisfied : (armFor kind).Satisfied source)
    (binding : PublicAssignmentBinding source kind final)
    (word : Fin 10) :
    finalWordValue final word = publicWordValue source kind word := by
  unfold finalWordValue publicWordValue CanonicalU64RecipeSound.bitsValue
  exact foldl_finalBits_eq_sourceBits source kind final canonical one satisfied
    binding word (List.range 64)
      (fun index member => List.mem_range.mp member) 0

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicDecoder
