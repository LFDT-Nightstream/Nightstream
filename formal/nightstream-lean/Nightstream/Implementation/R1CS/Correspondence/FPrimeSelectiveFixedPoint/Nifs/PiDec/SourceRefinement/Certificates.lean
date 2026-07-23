import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.PaperBridge
import Nightstream.Implementation.R1CS.Correspondence.Projection.IndexedRows

/-!
Resource-bounded coefficient certificates for the active strict-`PiDEC`
source artifact.

Owns: the concrete compiler layout, forty-eight bounded coefficient
comparisons, exact ordered coverage of both row streams, and kernel-only
aggregation into the public coefficient-equivalence theorem.

Does not own: source-row satisfaction, typed or paper acceptance, selective
CCS lowering, commitment binding, delayed projection authority, or row
removal.

Emits constraints: no.

Every executable comparison consumes proof-free sparse `Row` data.  Chunks
0--46 compare exactly 250 source/compiler row pairs; chunk 47 compares the
exact 95-row remainder.  No executable proposition contains the concatenated
11,845-row stream.

Assurance tier: artifact-checked for the bounded `kappa = 4` fixture.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.nifs.pi_dec.source_refinement.coefficients` | the 48 bounded proof-free chunks cover the complete source/compiler row streams with identical sparse coefficients | checked artifact |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.SourceRefinement.Certificates

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Rows

namespace GeneratedPiDec

abbrev rawLayout :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.rawLayout

abbrev sourceRows :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.sourceRows

abbrev commitmentRows :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.commitmentRows

def commitmentLayout
    (raw : Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.RawCommitment) :
    PiDecStrictCompiler.CommitmentLayout where
  dCol := raw.dCol
  kappaCol := raw.kappaCol
  dataCols := raw.dataCols

def claimLayout
    (raw : Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.RawClaim) :
    PiDecStrictCompiler.ClaimLayout where
  commitment := commitmentLayout raw.commitment
  adv := none
  xActiveCols := raw.xActiveCols
  xInactiveCol := raw.xInactiveCol
  xRows := raw.xRows
  xWidth := raw.xWidth
  xRowsCol := raw.xRowsCol
  xWidthCol := raw.xWidthCol
  mIn := raw.mIn
  mInCol := raw.mInCol
  yRingCols := raw.yRingCols
  ctCols := raw.ctCols
  rCols := raw.rCols
  sColCols := raw.sColCols
  foldDigestCols := raw.foldDigestCols

def baseLayout : PiDecStrictCompiler.Layout where
  radix := rawLayout.radix
  ringDimension := rawLayout.ringDimension
  extensionLimbs := rawLayout.extensionLimbs
  firstAllocatedColumn := rawLayout.firstAllocatedColumn
  parent := claimLayout rawLayout.parent
  children := rawLayout.children.map claimLayout

def layout : PiDecStrictProductionCompiler.Layout where
  base := baseLayout
  xSignTraces := rawLayout.xSignTraces
  childCount := by native_decide

end GeneratedPiDec

def compilerRows : List Row :=
  PiDecStrictProductionCompiler.rows GeneratedPiDec.layout

private def iteratedDrop {alpha : Type} (width : Nat) :
    Nat -> List alpha -> List alpha
  | 0, values => values
  | count + 1, values => iteratedDrop width count (values.drop width)

private def partition {alpha : Type} (width : Nat) :
    Nat -> List alpha -> List (List alpha)
  | 0, values => [values]
  | count + 1, values =>
      values.take width :: partition width count (values.drop width)

private theorem partition_flatten
    {alpha : Type} (width count : Nat) (values : List alpha) :
    (partition width count values).flatten = values := by
  induction count generalizing values with
  | zero => simp [partition]
  | succ count inductionHypothesis =>
      simp only [partition, List.flatten_cons]
      rw [inductionHypothesis]
      exact List.take_append_drop width values

private theorem partition_length
    {alpha : Type} (width count : Nat) (values : List alpha) :
    (partition width count values).length = count + 1 := by
  induction count generalizing values with
  | zero => simp [partition]
  | succ count inductionHypothesis =>
      simp [partition, inductionHypothesis, Nat.succ_eq_add_one,
        Nat.add_assoc]

private theorem iteratedDrop_eq_drop_mul
    {alpha : Type} (width count : Nat) (values : List alpha) :
    iteratedDrop width count values = values.drop (count * width) := by
  induction count generalizing values with
  | zero => simp [iteratedDrop]
  | succ count inductionHypothesis =>
      simp only [iteratedDrop]
      rw [inductionHypothesis, List.drop_drop]
      congr 2
      simp [Nat.add_one_mul, Nat.add_comm]

private def compilerRest (index : Nat) : List Row :=
  iteratedDrop 250 index compilerRows

private def compilerPrefixChunk (index : Nat) : List Row :=
  (compilerRest index).take 250

private def compilerRemainder : List Row :=
  compilerRest 47

private def sourceChunks : List (List Row) := [
  Chunk0.values,
  Chunk1.values,
  Chunk2.values,
  Chunk3.values,
  Chunk4.values,
  Chunk5.values,
  Chunk6.values,
  Chunk7.values,
  Chunk8.values,
  Chunk9.values,
  Chunk10.values,
  Chunk11.values,
  Chunk12.values,
  Chunk13.values,
  Chunk14.values,
  Chunk15.values,
  Chunk16.values,
  Chunk17.values,
  Chunk18.values,
  Chunk19.values,
  Chunk20.values,
  Chunk21.values,
  Chunk22.values,
  Chunk23.values,
  Chunk24.values,
  Chunk25.values,
  Chunk26.values,
  Chunk27.values,
  Chunk28.values,
  Chunk29.values,
  Chunk30.values,
  Chunk31.values,
  Chunk32.values,
  Chunk33.values,
  Chunk34.values,
  Chunk35.values,
  Chunk36.values,
  Chunk37.values,
  Chunk38.values,
  Chunk39.values,
  Chunk40.values,
  Chunk41.values,
  Chunk42.values,
  Chunk43.values,
  Chunk44.values,
  Chunk45.values,
  Chunk46.values,
  Chunk47.values]

private def compilerChunks : List (List Row) := [
  compilerPrefixChunk 0,
  compilerPrefixChunk 1,
  compilerPrefixChunk 2,
  compilerPrefixChunk 3,
  compilerPrefixChunk 4,
  compilerPrefixChunk 5,
  compilerPrefixChunk 6,
  compilerPrefixChunk 7,
  compilerPrefixChunk 8,
  compilerPrefixChunk 9,
  compilerPrefixChunk 10,
  compilerPrefixChunk 11,
  compilerPrefixChunk 12,
  compilerPrefixChunk 13,
  compilerPrefixChunk 14,
  compilerPrefixChunk 15,
  compilerPrefixChunk 16,
  compilerPrefixChunk 17,
  compilerPrefixChunk 18,
  compilerPrefixChunk 19,
  compilerPrefixChunk 20,
  compilerPrefixChunk 21,
  compilerPrefixChunk 22,
  compilerPrefixChunk 23,
  compilerPrefixChunk 24,
  compilerPrefixChunk 25,
  compilerPrefixChunk 26,
  compilerPrefixChunk 27,
  compilerPrefixChunk 28,
  compilerPrefixChunk 29,
  compilerPrefixChunk 30,
  compilerPrefixChunk 31,
  compilerPrefixChunk 32,
  compilerPrefixChunk 33,
  compilerPrefixChunk 34,
  compilerPrefixChunk 35,
  compilerPrefixChunk 36,
  compilerPrefixChunk 37,
  compilerPrefixChunk 38,
  compilerPrefixChunk 39,
  compilerPrefixChunk 40,
  compilerPrefixChunk 41,
  compilerPrefixChunk 42,
  compilerPrefixChunk 43,
  compilerPrefixChunk 44,
  compilerPrefixChunk 45,
  compilerPrefixChunk 46,
  compilerRemainder]

private def rowsPermutationEquivalentListDecidable :
    (source reconstructed : List Row) ->
      Decidable (RowsPermutationEquivalentList source reconstructed)
  | [], [] => isTrue True.intro
  | [], _ :: _ => isFalse id
  | _ :: _, [] => isFalse id
  | source :: sources, reconstructed :: reconstructions =>
      match inferInstanceAs
          (Decidable (RowsPermutationEquivalent source reconstructed)),
        rowsPermutationEquivalentListDecidable sources reconstructions with
      | isTrue head, isTrue tail => isTrue ⟨head, tail⟩
      | isFalse head, isTrue _ => isFalse fun equivalent => head equivalent.1
      | isTrue _, isFalse tail => isFalse fun equivalent => tail equivalent.2
      | isFalse head, isFalse _ => isFalse fun equivalent => head equivalent.1

local instance (source reconstructed : List Row) :
    Decidable (RowsPermutationEquivalentList source reconstructed) :=
  rowsPermutationEquivalentListDecidable source reconstructed

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

/- Each theorem checks exactly the proof-free row-pair cardinality stated in
its first two conjuncts.  The cardinality check and coefficient relation share
one bounded executable subject, so a silent oversized tail cannot pass. -/
private theorem chunk0_certificate :
    Chunk0.values.length = 250 ∧ (compilerPrefixChunk 0).length = 250 ∧
      RowsPermutationEquivalentList Chunk0.values (compilerPrefixChunk 0) := by native_decide
private theorem chunk1_certificate :
    Chunk1.values.length = 250 ∧ (compilerPrefixChunk 1).length = 250 ∧
      RowsPermutationEquivalentList Chunk1.values (compilerPrefixChunk 1) := by native_decide
private theorem chunk2_certificate :
    Chunk2.values.length = 250 ∧ (compilerPrefixChunk 2).length = 250 ∧
      RowsPermutationEquivalentList Chunk2.values (compilerPrefixChunk 2) := by native_decide
private theorem chunk3_certificate :
    Chunk3.values.length = 250 ∧ (compilerPrefixChunk 3).length = 250 ∧
      RowsPermutationEquivalentList Chunk3.values (compilerPrefixChunk 3) := by native_decide
private theorem chunk4_certificate :
    Chunk4.values.length = 250 ∧ (compilerPrefixChunk 4).length = 250 ∧
      RowsPermutationEquivalentList Chunk4.values (compilerPrefixChunk 4) := by native_decide
private theorem chunk5_certificate :
    Chunk5.values.length = 250 ∧ (compilerPrefixChunk 5).length = 250 ∧
      RowsPermutationEquivalentList Chunk5.values (compilerPrefixChunk 5) := by native_decide
private theorem chunk6_certificate :
    Chunk6.values.length = 250 ∧ (compilerPrefixChunk 6).length = 250 ∧
      RowsPermutationEquivalentList Chunk6.values (compilerPrefixChunk 6) := by native_decide
private theorem chunk7_certificate :
    Chunk7.values.length = 250 ∧ (compilerPrefixChunk 7).length = 250 ∧
      RowsPermutationEquivalentList Chunk7.values (compilerPrefixChunk 7) := by native_decide
private theorem chunk8_certificate :
    Chunk8.values.length = 250 ∧ (compilerPrefixChunk 8).length = 250 ∧
      RowsPermutationEquivalentList Chunk8.values (compilerPrefixChunk 8) := by native_decide
private theorem chunk9_certificate :
    Chunk9.values.length = 250 ∧ (compilerPrefixChunk 9).length = 250 ∧
      RowsPermutationEquivalentList Chunk9.values (compilerPrefixChunk 9) := by native_decide
private theorem chunk10_certificate :
    Chunk10.values.length = 250 ∧ (compilerPrefixChunk 10).length = 250 ∧
      RowsPermutationEquivalentList Chunk10.values (compilerPrefixChunk 10) := by native_decide
private theorem chunk11_certificate :
    Chunk11.values.length = 250 ∧ (compilerPrefixChunk 11).length = 250 ∧
      RowsPermutationEquivalentList Chunk11.values (compilerPrefixChunk 11) := by native_decide
private theorem chunk12_certificate :
    Chunk12.values.length = 250 ∧ (compilerPrefixChunk 12).length = 250 ∧
      RowsPermutationEquivalentList Chunk12.values (compilerPrefixChunk 12) := by native_decide
private theorem chunk13_certificate :
    Chunk13.values.length = 250 ∧ (compilerPrefixChunk 13).length = 250 ∧
      RowsPermutationEquivalentList Chunk13.values (compilerPrefixChunk 13) := by native_decide
private theorem chunk14_certificate :
    Chunk14.values.length = 250 ∧ (compilerPrefixChunk 14).length = 250 ∧
      RowsPermutationEquivalentList Chunk14.values (compilerPrefixChunk 14) := by native_decide
private theorem chunk15_certificate :
    Chunk15.values.length = 250 ∧ (compilerPrefixChunk 15).length = 250 ∧
      RowsPermutationEquivalentList Chunk15.values (compilerPrefixChunk 15) := by native_decide
private theorem chunk16_certificate :
    Chunk16.values.length = 250 ∧ (compilerPrefixChunk 16).length = 250 ∧
      RowsPermutationEquivalentList Chunk16.values (compilerPrefixChunk 16) := by native_decide
private theorem chunk17_certificate :
    Chunk17.values.length = 250 ∧ (compilerPrefixChunk 17).length = 250 ∧
      RowsPermutationEquivalentList Chunk17.values (compilerPrefixChunk 17) := by native_decide
private theorem chunk18_certificate :
    Chunk18.values.length = 250 ∧ (compilerPrefixChunk 18).length = 250 ∧
      RowsPermutationEquivalentList Chunk18.values (compilerPrefixChunk 18) := by native_decide
private theorem chunk19_certificate :
    Chunk19.values.length = 250 ∧ (compilerPrefixChunk 19).length = 250 ∧
      RowsPermutationEquivalentList Chunk19.values (compilerPrefixChunk 19) := by native_decide
private theorem chunk20_certificate :
    Chunk20.values.length = 250 ∧ (compilerPrefixChunk 20).length = 250 ∧
      RowsPermutationEquivalentList Chunk20.values (compilerPrefixChunk 20) := by native_decide
private theorem chunk21_certificate :
    Chunk21.values.length = 250 ∧ (compilerPrefixChunk 21).length = 250 ∧
      RowsPermutationEquivalentList Chunk21.values (compilerPrefixChunk 21) := by native_decide
private theorem chunk22_certificate :
    Chunk22.values.length = 250 ∧ (compilerPrefixChunk 22).length = 250 ∧
      RowsPermutationEquivalentList Chunk22.values (compilerPrefixChunk 22) := by native_decide
private theorem chunk23_certificate :
    Chunk23.values.length = 250 ∧ (compilerPrefixChunk 23).length = 250 ∧
      RowsPermutationEquivalentList Chunk23.values (compilerPrefixChunk 23) := by native_decide
private theorem chunk24_certificate :
    Chunk24.values.length = 250 ∧ (compilerPrefixChunk 24).length = 250 ∧
      RowsPermutationEquivalentList Chunk24.values (compilerPrefixChunk 24) := by native_decide
private theorem chunk25_certificate :
    Chunk25.values.length = 250 ∧ (compilerPrefixChunk 25).length = 250 ∧
      RowsPermutationEquivalentList Chunk25.values (compilerPrefixChunk 25) := by native_decide
private theorem chunk26_certificate :
    Chunk26.values.length = 250 ∧ (compilerPrefixChunk 26).length = 250 ∧
      RowsPermutationEquivalentList Chunk26.values (compilerPrefixChunk 26) := by native_decide
private theorem chunk27_certificate :
    Chunk27.values.length = 250 ∧ (compilerPrefixChunk 27).length = 250 ∧
      RowsPermutationEquivalentList Chunk27.values (compilerPrefixChunk 27) := by native_decide
private theorem chunk28_certificate :
    Chunk28.values.length = 250 ∧ (compilerPrefixChunk 28).length = 250 ∧
      RowsPermutationEquivalentList Chunk28.values (compilerPrefixChunk 28) := by native_decide
private theorem chunk29_certificate :
    Chunk29.values.length = 250 ∧ (compilerPrefixChunk 29).length = 250 ∧
      RowsPermutationEquivalentList Chunk29.values (compilerPrefixChunk 29) := by native_decide
private theorem chunk30_certificate :
    Chunk30.values.length = 250 ∧ (compilerPrefixChunk 30).length = 250 ∧
      RowsPermutationEquivalentList Chunk30.values (compilerPrefixChunk 30) := by native_decide
private theorem chunk31_certificate :
    Chunk31.values.length = 250 ∧ (compilerPrefixChunk 31).length = 250 ∧
      RowsPermutationEquivalentList Chunk31.values (compilerPrefixChunk 31) := by native_decide
private theorem chunk32_certificate :
    Chunk32.values.length = 250 ∧ (compilerPrefixChunk 32).length = 250 ∧
      RowsPermutationEquivalentList Chunk32.values (compilerPrefixChunk 32) := by native_decide
private theorem chunk33_certificate :
    Chunk33.values.length = 250 ∧ (compilerPrefixChunk 33).length = 250 ∧
      RowsPermutationEquivalentList Chunk33.values (compilerPrefixChunk 33) := by native_decide
private theorem chunk34_certificate :
    Chunk34.values.length = 250 ∧ (compilerPrefixChunk 34).length = 250 ∧
      RowsPermutationEquivalentList Chunk34.values (compilerPrefixChunk 34) := by native_decide
private theorem chunk35_certificate :
    Chunk35.values.length = 250 ∧ (compilerPrefixChunk 35).length = 250 ∧
      RowsPermutationEquivalentList Chunk35.values (compilerPrefixChunk 35) := by native_decide
private theorem chunk36_certificate :
    Chunk36.values.length = 250 ∧ (compilerPrefixChunk 36).length = 250 ∧
      RowsPermutationEquivalentList Chunk36.values (compilerPrefixChunk 36) := by native_decide
private theorem chunk37_certificate :
    Chunk37.values.length = 250 ∧ (compilerPrefixChunk 37).length = 250 ∧
      RowsPermutationEquivalentList Chunk37.values (compilerPrefixChunk 37) := by native_decide
private theorem chunk38_certificate :
    Chunk38.values.length = 250 ∧ (compilerPrefixChunk 38).length = 250 ∧
      RowsPermutationEquivalentList Chunk38.values (compilerPrefixChunk 38) := by native_decide
private theorem chunk39_certificate :
    Chunk39.values.length = 250 ∧ (compilerPrefixChunk 39).length = 250 ∧
      RowsPermutationEquivalentList Chunk39.values (compilerPrefixChunk 39) := by native_decide
private theorem chunk40_certificate :
    Chunk40.values.length = 250 ∧ (compilerPrefixChunk 40).length = 250 ∧
      RowsPermutationEquivalentList Chunk40.values (compilerPrefixChunk 40) := by native_decide
private theorem chunk41_certificate :
    Chunk41.values.length = 250 ∧ (compilerPrefixChunk 41).length = 250 ∧
      RowsPermutationEquivalentList Chunk41.values (compilerPrefixChunk 41) := by native_decide
private theorem chunk42_certificate :
    Chunk42.values.length = 250 ∧ (compilerPrefixChunk 42).length = 250 ∧
      RowsPermutationEquivalentList Chunk42.values (compilerPrefixChunk 42) := by native_decide
private theorem chunk43_certificate :
    Chunk43.values.length = 250 ∧ (compilerPrefixChunk 43).length = 250 ∧
      RowsPermutationEquivalentList Chunk43.values (compilerPrefixChunk 43) := by native_decide
private theorem chunk44_certificate :
    Chunk44.values.length = 250 ∧ (compilerPrefixChunk 44).length = 250 ∧
      RowsPermutationEquivalentList Chunk44.values (compilerPrefixChunk 44) := by native_decide
private theorem chunk45_certificate :
    Chunk45.values.length = 250 ∧ (compilerPrefixChunk 45).length = 250 ∧
      RowsPermutationEquivalentList Chunk45.values (compilerPrefixChunk 45) := by native_decide
private theorem chunk46_certificate :
    Chunk46.values.length = 250 ∧ (compilerPrefixChunk 46).length = 250 ∧
      RowsPermutationEquivalentList Chunk46.values (compilerPrefixChunk 46) := by native_decide
private theorem chunk47_certificate :
    Chunk47.values.length = 95 ∧ compilerRemainder.length = 95 ∧
      RowsPermutationEquivalentList Chunk47.values compilerRemainder := by native_decide

private theorem compilerChunks_eq_partition :
    compilerChunks = partition 250 47 compilerRows := by
  rfl

/-- The generated shard list is the source stream in exact order. -/
theorem sourceChunks_cover_in_order_without_overlap :
    sourceChunks.flatten = GeneratedPiDec.sourceRows := by
  simp [sourceChunks, GeneratedPiDec.sourceRows,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.sourceRows,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.sourceRows]

/-- Recursive `take`/`drop` consumes each compiler row exactly once and in
order; there is no unaccounted prefix, gap, overlap, or suffix. -/
theorem compilerChunks_cover_in_order_without_overlap :
    compilerChunks.flatten = compilerRows := by
  rw [compilerChunks_eq_partition]
  exact partition_flatten 250 47 compilerRows

theorem compilerChunks_count : compilerChunks.length = 48 := by
  rw [compilerChunks_eq_partition]
  simpa using partition_length 250 47 compilerRows

/-- Distinct 250-row prefix intervals do not overlap. -/
theorem prefixChunkIntervals_disjoint
    {left right : Nat} (leftLt : left < 47) (rightLt : right < 47)
    (ordered : left < right) :
    left * 250 + 250 <= right * 250 := by
  omega

/-- Every prefix interval ends before the exact remainder boundary. -/
theorem prefixChunks_before_remainder
    {index : Nat} (indexLt : index < 47) :
    index * 250 + 250 <= 11750 := by
  omega

theorem compilerRemainder_starts_at_11750 :
    compilerRemainder = compilerRows.drop 11750 := by
  unfold compilerRemainder compilerRest
  rw [iteratedDrop_eq_drop_mul]

theorem sourceRemainder_length_exact : Chunk47.values.length = 95 :=
  chunk47_certificate.1

theorem compilerRemainder_length_exact : compilerRemainder.length = 95 :=
  chunk47_certificate.2.1

private def expectedChunkLengths : List Nat := [
  250, 250, 250, 250, 250, 250, 250, 250,
  250, 250, 250, 250, 250, 250, 250, 250,
  250, 250, 250, 250, 250, 250, 250, 250,
  250, 250, 250, 250, 250, 250, 250, 250,
  250, 250, 250, 250, 250, 250, 250, 250,
  250, 250, 250, 250, 250, 250, 250, 95]

theorem sourceChunkLengths_exact :
    sourceChunks.map List.length = expectedChunkLengths := by
  unfold sourceChunks expectedChunkLengths
  simp only [List.map_cons, List.map_nil]
  rw [chunk0_certificate.1, chunk1_certificate.1, chunk2_certificate.1,
    chunk3_certificate.1, chunk4_certificate.1, chunk5_certificate.1,
    chunk6_certificate.1, chunk7_certificate.1, chunk8_certificate.1,
    chunk9_certificate.1, chunk10_certificate.1, chunk11_certificate.1,
    chunk12_certificate.1, chunk13_certificate.1, chunk14_certificate.1,
    chunk15_certificate.1, chunk16_certificate.1, chunk17_certificate.1,
    chunk18_certificate.1, chunk19_certificate.1, chunk20_certificate.1,
    chunk21_certificate.1, chunk22_certificate.1, chunk23_certificate.1,
    chunk24_certificate.1, chunk25_certificate.1, chunk26_certificate.1,
    chunk27_certificate.1, chunk28_certificate.1, chunk29_certificate.1,
    chunk30_certificate.1, chunk31_certificate.1, chunk32_certificate.1,
    chunk33_certificate.1, chunk34_certificate.1, chunk35_certificate.1,
    chunk36_certificate.1, chunk37_certificate.1, chunk38_certificate.1,
    chunk39_certificate.1, chunk40_certificate.1, chunk41_certificate.1,
    chunk42_certificate.1, chunk43_certificate.1, chunk44_certificate.1,
    chunk45_certificate.1, chunk46_certificate.1, chunk47_certificate.1]

theorem compilerChunkLengths_exact :
    compilerChunks.map List.length = expectedChunkLengths := by
  unfold compilerChunks expectedChunkLengths
  simp only [List.map_cons, List.map_nil]
  rw [chunk0_certificate.2.1, chunk1_certificate.2.1,
    chunk2_certificate.2.1, chunk3_certificate.2.1,
    chunk4_certificate.2.1, chunk5_certificate.2.1,
    chunk6_certificate.2.1, chunk7_certificate.2.1,
    chunk8_certificate.2.1, chunk9_certificate.2.1,
    chunk10_certificate.2.1, chunk11_certificate.2.1,
    chunk12_certificate.2.1, chunk13_certificate.2.1,
    chunk14_certificate.2.1, chunk15_certificate.2.1,
    chunk16_certificate.2.1, chunk17_certificate.2.1,
    chunk18_certificate.2.1, chunk19_certificate.2.1,
    chunk20_certificate.2.1, chunk21_certificate.2.1,
    chunk22_certificate.2.1, chunk23_certificate.2.1,
    chunk24_certificate.2.1, chunk25_certificate.2.1,
    chunk26_certificate.2.1, chunk27_certificate.2.1,
    chunk28_certificate.2.1, chunk29_certificate.2.1,
    chunk30_certificate.2.1, chunk31_certificate.2.1,
    chunk32_certificate.2.1, chunk33_certificate.2.1,
    chunk34_certificate.2.1, chunk35_certificate.2.1,
    chunk36_certificate.2.1, chunk37_certificate.2.1,
    chunk38_certificate.2.1, chunk39_certificate.2.1,
    chunk40_certificate.2.1, chunk41_certificate.2.1,
    chunk42_certificate.2.1, chunk43_certificate.2.1,
    chunk44_certificate.2.1, chunk45_certificate.2.1,
    chunk46_certificate.2.1, chunk47_certificate.2.1]

theorem sourceChunks_max_length
    {chunk : List Row} (member : chunk ∈ sourceChunks) :
    chunk.length <= 250 := by
  have lengthMember : chunk.length ∈ sourceChunks.map List.length :=
    List.mem_map.mpr ⟨chunk, member, rfl⟩
  rw [sourceChunkLengths_exact] at lengthMember
  simp [expectedChunkLengths] at lengthMember
  omega

theorem compilerChunks_max_length
    {chunk : List Row} (member : chunk ∈ compilerChunks) :
    chunk.length <= 250 := by
  have lengthMember : chunk.length ∈ compilerChunks.map List.length :=
    List.mem_map.mpr ⟨chunk, member, rfl⟩
  rw [compilerChunkLengths_exact] at lengthMember
  simp [expectedChunkLengths] at lengthMember
  omega

private theorem rowsPermutationEquivalentList_append
    {leftSource rightSource leftExpected rightExpected : List Row}
    (left : RowsPermutationEquivalentList leftSource leftExpected)
    (right : RowsPermutationEquivalentList rightSource rightExpected) :
    RowsPermutationEquivalentList (leftSource ++ rightSource)
      (leftExpected ++ rightExpected) := by
  induction leftSource generalizing leftExpected with
  | nil =>
      cases leftExpected with
      | nil => simpa using right
      | cons _ _ => simp [RowsPermutationEquivalentList] at left
  | cons source sources inductionHypothesis =>
      cases leftExpected with
      | nil => simp [RowsPermutationEquivalentList] at left
      | cons expected expecteds =>
          rcases left with ⟨head, tail⟩
          exact ⟨head, inductionHypothesis tail⟩

private inductive ChunksEquivalent :
    List (List Row) → List (List Row) → Prop where
  | nil : ChunksEquivalent [] []
  | cons {source expected sources expecteds} :
      RowsPermutationEquivalentList source expected →
      ChunksEquivalent sources expecteds →
      ChunksEquivalent (source :: sources) (expected :: expecteds)

private theorem rowsPermutationEquivalentList_flatten
    {sources expecteds : List (List Row)}
    (related : ChunksEquivalent sources expecteds) :
    RowsPermutationEquivalentList sources.flatten expecteds.flatten := by
  induction related with
  | nil => trivial
  | cons head tail inductionHypothesis =>
      exact rowsPermutationEquivalentList_append head inductionHypothesis

private theorem chunks_exact :
    ChunksEquivalent sourceChunks compilerChunks := by
  unfold sourceChunks compilerChunks
  apply ChunksEquivalent.cons chunk0_certificate.2.2
  apply ChunksEquivalent.cons chunk1_certificate.2.2
  apply ChunksEquivalent.cons chunk2_certificate.2.2
  apply ChunksEquivalent.cons chunk3_certificate.2.2
  apply ChunksEquivalent.cons chunk4_certificate.2.2
  apply ChunksEquivalent.cons chunk5_certificate.2.2
  apply ChunksEquivalent.cons chunk6_certificate.2.2
  apply ChunksEquivalent.cons chunk7_certificate.2.2
  apply ChunksEquivalent.cons chunk8_certificate.2.2
  apply ChunksEquivalent.cons chunk9_certificate.2.2
  apply ChunksEquivalent.cons chunk10_certificate.2.2
  apply ChunksEquivalent.cons chunk11_certificate.2.2
  apply ChunksEquivalent.cons chunk12_certificate.2.2
  apply ChunksEquivalent.cons chunk13_certificate.2.2
  apply ChunksEquivalent.cons chunk14_certificate.2.2
  apply ChunksEquivalent.cons chunk15_certificate.2.2
  apply ChunksEquivalent.cons chunk16_certificate.2.2
  apply ChunksEquivalent.cons chunk17_certificate.2.2
  apply ChunksEquivalent.cons chunk18_certificate.2.2
  apply ChunksEquivalent.cons chunk19_certificate.2.2
  apply ChunksEquivalent.cons chunk20_certificate.2.2
  apply ChunksEquivalent.cons chunk21_certificate.2.2
  apply ChunksEquivalent.cons chunk22_certificate.2.2
  apply ChunksEquivalent.cons chunk23_certificate.2.2
  apply ChunksEquivalent.cons chunk24_certificate.2.2
  apply ChunksEquivalent.cons chunk25_certificate.2.2
  apply ChunksEquivalent.cons chunk26_certificate.2.2
  apply ChunksEquivalent.cons chunk27_certificate.2.2
  apply ChunksEquivalent.cons chunk28_certificate.2.2
  apply ChunksEquivalent.cons chunk29_certificate.2.2
  apply ChunksEquivalent.cons chunk30_certificate.2.2
  apply ChunksEquivalent.cons chunk31_certificate.2.2
  apply ChunksEquivalent.cons chunk32_certificate.2.2
  apply ChunksEquivalent.cons chunk33_certificate.2.2
  apply ChunksEquivalent.cons chunk34_certificate.2.2
  apply ChunksEquivalent.cons chunk35_certificate.2.2
  apply ChunksEquivalent.cons chunk36_certificate.2.2
  apply ChunksEquivalent.cons chunk37_certificate.2.2
  apply ChunksEquivalent.cons chunk38_certificate.2.2
  apply ChunksEquivalent.cons chunk39_certificate.2.2
  apply ChunksEquivalent.cons chunk40_certificate.2.2
  apply ChunksEquivalent.cons chunk41_certificate.2.2
  apply ChunksEquivalent.cons chunk42_certificate.2.2
  apply ChunksEquivalent.cons chunk43_certificate.2.2
  apply ChunksEquivalent.cons chunk44_certificate.2.2
  apply ChunksEquivalent.cons chunk45_certificate.2.2
  apply ChunksEquivalent.cons chunk46_certificate.2.2
  apply ChunksEquivalent.cons chunk47_certificate.2.2
  exact ChunksEquivalent.nil

/-- Exact sparse coefficient agreement, aggregated in the kernel from the
forty-eight bounded executable certificates. -/
theorem sourceRows_exact :
    RowsPermutationEquivalentList GeneratedPiDec.sourceRows compilerRows := by
  rw [<- sourceChunks_cover_in_order_without_overlap,
    <- compilerChunks_cover_in_order_without_overlap]
  exact rowsPermutationEquivalentList_flatten chunks_exact

private theorem rowsPermutationEquivalentList_lengths
    {left right : List Row}
    (equivalent : RowsPermutationEquivalentList left right) :
    left.length = right.length := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons _ _ => simp [RowsPermutationEquivalentList] at equivalent
  | cons head tail inductionHypothesis =>
      cases right with
      | nil => simp [RowsPermutationEquivalentList] at equivalent
      | cons rightHead rightTail =>
          simp only [List.length_cons]
          rw [inductionHypothesis equivalent.2]

/-- The compiler count follows from bounded coefficient lockstep and the
separately bounded generated-source census; no global row-list computation is
used. -/
theorem compilerRows_length :
    compilerRows.length =
      Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Metadata.sourceRowCount := by
  rw [<- rowsPermutationEquivalentList_lengths sourceRows_exact]
  exact Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.sourceRows_length

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Nifs.PiDec.SourceRefinement.Certificates
