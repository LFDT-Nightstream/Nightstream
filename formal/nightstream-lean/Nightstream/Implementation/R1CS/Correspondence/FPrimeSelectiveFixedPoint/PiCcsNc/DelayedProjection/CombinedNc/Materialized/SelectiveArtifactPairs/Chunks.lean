import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Core

/-!
Exact bounded partition of the production combined-NC emitted/provenance
stream.

Owns: the 52 retained pairs, 24 rewrite-pair shards (23 of 64 records and one
of 21), exact concatenation to all 1,545 emitted rows, and proof-free
cardinality data.

Does not own: certificate truth, decoding, row satisfaction, source-program
execution, selector truth, transcript or commitment authority, costs, or row
removal.

Emits constraints: none.

The 52-record prefix is retained provenance.  Rewrite provenance begins at
offset 52, so each 64-record generated provenance shard crosses exactly one
boundary between adjacent 64-record emitted-row shards.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs

private abbrev retainedCount : Nat := 52
private abbrev generatedChunkSize : Nat := 64

theorem retainedCount_le_certificateLimit : retainedCount ≤ 256 := by decide
theorem generatedChunkSize_le_certificateLimit :
    generatedChunkSize ≤ 256 := by decide

def retainedRows : List RawEmittedRow :=
  EmittedRows.Chunk0.values.take retainedCount

def retainedSteps : List RawRetainedStep :=
  Provenance.RetainedSteps.Chunk0.values

def retainedPairs : List RawRetainedPair :=
  List.zipWith RawRetainedPair.mk retainedRows retainedSteps

private def across (left right : List RawEmittedRow) : List RawEmittedRow :=
  left.drop retainedCount ++ right.take retainedCount

def rewriteRows0 := across EmittedRows.Chunk0.values EmittedRows.Chunk1.values
def rewriteRows1 := across EmittedRows.Chunk1.values EmittedRows.Chunk2.values
def rewriteRows2 := across EmittedRows.Chunk2.values EmittedRows.Chunk3.values
def rewriteRows3 := across EmittedRows.Chunk3.values EmittedRows.Chunk4.values
def rewriteRows4 := across EmittedRows.Chunk4.values EmittedRows.Chunk5.values
def rewriteRows5 := across EmittedRows.Chunk5.values EmittedRows.Chunk6.values
def rewriteRows6 := across EmittedRows.Chunk6.values EmittedRows.Chunk7.values
def rewriteRows7 := across EmittedRows.Chunk7.values EmittedRows.Chunk8.values
def rewriteRows8 := across EmittedRows.Chunk8.values EmittedRows.Chunk9.values
def rewriteRows9 := across EmittedRows.Chunk9.values EmittedRows.Chunk10.values
def rewriteRows10 := across EmittedRows.Chunk10.values EmittedRows.Chunk11.values
def rewriteRows11 := across EmittedRows.Chunk11.values EmittedRows.Chunk12.values
def rewriteRows12 := across EmittedRows.Chunk12.values EmittedRows.Chunk13.values
def rewriteRows13 := across EmittedRows.Chunk13.values EmittedRows.Chunk14.values
def rewriteRows14 := across EmittedRows.Chunk14.values EmittedRows.Chunk15.values
def rewriteRows15 := across EmittedRows.Chunk15.values EmittedRows.Chunk16.values
def rewriteRows16 := across EmittedRows.Chunk16.values EmittedRows.Chunk17.values
def rewriteRows17 := across EmittedRows.Chunk17.values EmittedRows.Chunk18.values
def rewriteRows18 := across EmittedRows.Chunk18.values EmittedRows.Chunk19.values
def rewriteRows19 := across EmittedRows.Chunk19.values EmittedRows.Chunk20.values
def rewriteRows20 := across EmittedRows.Chunk20.values EmittedRows.Chunk21.values
def rewriteRows21 := across EmittedRows.Chunk21.values EmittedRows.Chunk22.values
def rewriteRows22 := across EmittedRows.Chunk22.values EmittedRows.Chunk23.values
def rewriteRows23 : List RawEmittedRow :=
  EmittedRows.Chunk23.values.drop retainedCount ++ EmittedRows.Chunk24.values

def rewriteSteps0 := Provenance.RewriteSteps.Chunk0.values
def rewriteSteps1 := Provenance.RewriteSteps.Chunk1.values
def rewriteSteps2 := Provenance.RewriteSteps.Chunk2.values
def rewriteSteps3 := Provenance.RewriteSteps.Chunk3.values
def rewriteSteps4 := Provenance.RewriteSteps.Chunk4.values
def rewriteSteps5 := Provenance.RewriteSteps.Chunk5.values
def rewriteSteps6 := Provenance.RewriteSteps.Chunk6.values
def rewriteSteps7 := Provenance.RewriteSteps.Chunk7.values
def rewriteSteps8 := Provenance.RewriteSteps.Chunk8.values
def rewriteSteps9 := Provenance.RewriteSteps.Chunk9.values
def rewriteSteps10 := Provenance.RewriteSteps.Chunk10.values
def rewriteSteps11 := Provenance.RewriteSteps.Chunk11.values
def rewriteSteps12 := Provenance.RewriteSteps.Chunk12.values
def rewriteSteps13 := Provenance.RewriteSteps.Chunk13.values
def rewriteSteps14 := Provenance.RewriteSteps.Chunk14.values
def rewriteSteps15 := Provenance.RewriteSteps.Chunk15.values
def rewriteSteps16 := Provenance.RewriteSteps.Chunk16.values
def rewriteSteps17 := Provenance.RewriteSteps.Chunk17.values
def rewriteSteps18 := Provenance.RewriteSteps.Chunk18.values
def rewriteSteps19 := Provenance.RewriteSteps.Chunk19.values
def rewriteSteps20 := Provenance.RewriteSteps.Chunk20.values
def rewriteSteps21 := Provenance.RewriteSteps.Chunk21.values
def rewriteSteps22 := Provenance.RewriteSteps.Chunk22.values
def rewriteSteps23 := Provenance.RewriteSteps.Chunk23.values

def pairRewrite (rows : List RawEmittedRow)
    (steps : List RawRewriteStep) : List RawRewritePair :=
  List.zipWith RawRewritePair.mk rows steps

def rewritePairs0 := pairRewrite rewriteRows0 rewriteSteps0
def rewritePairs1 := pairRewrite rewriteRows1 rewriteSteps1
def rewritePairs2 := pairRewrite rewriteRows2 rewriteSteps2
def rewritePairs3 := pairRewrite rewriteRows3 rewriteSteps3
def rewritePairs4 := pairRewrite rewriteRows4 rewriteSteps4
def rewritePairs5 := pairRewrite rewriteRows5 rewriteSteps5
def rewritePairs6 := pairRewrite rewriteRows6 rewriteSteps6
def rewritePairs7 := pairRewrite rewriteRows7 rewriteSteps7
def rewritePairs8 := pairRewrite rewriteRows8 rewriteSteps8
def rewritePairs9 := pairRewrite rewriteRows9 rewriteSteps9
def rewritePairs10 := pairRewrite rewriteRows10 rewriteSteps10
def rewritePairs11 := pairRewrite rewriteRows11 rewriteSteps11
def rewritePairs12 := pairRewrite rewriteRows12 rewriteSteps12
def rewritePairs13 := pairRewrite rewriteRows13 rewriteSteps13
def rewritePairs14 := pairRewrite rewriteRows14 rewriteSteps14
def rewritePairs15 := pairRewrite rewriteRows15 rewriteSteps15
def rewritePairs16 := pairRewrite rewriteRows16 rewriteSteps16
def rewritePairs17 := pairRewrite rewriteRows17 rewriteSteps17
def rewritePairs18 := pairRewrite rewriteRows18 rewriteSteps18
def rewritePairs19 := pairRewrite rewriteRows19 rewriteSteps19
def rewritePairs20 := pairRewrite rewriteRows20 rewriteSteps20
def rewritePairs21 := pairRewrite rewriteRows21 rewriteSteps21
def rewritePairs22 := pairRewrite rewriteRows22 rewriteSteps22
def rewritePairs23 := pairRewrite rewriteRows23 rewriteSteps23

def rewriteRows : List RawEmittedRow :=
  rewriteRows0 ++ rewriteRows1 ++ rewriteRows2 ++ rewriteRows3 ++
  rewriteRows4 ++ rewriteRows5 ++ rewriteRows6 ++ rewriteRows7 ++
  rewriteRows8 ++ rewriteRows9 ++ rewriteRows10 ++ rewriteRows11 ++
  rewriteRows12 ++ rewriteRows13 ++ rewriteRows14 ++ rewriteRows15 ++
  rewriteRows16 ++ rewriteRows17 ++ rewriteRows18 ++ rewriteRows19 ++
  rewriteRows20 ++ rewriteRows21 ++ rewriteRows22 ++ rewriteRows23

def rewriteSteps : List RawRewriteStep :=
  rewriteSteps0 ++ rewriteSteps1 ++ rewriteSteps2 ++ rewriteSteps3 ++
  rewriteSteps4 ++ rewriteSteps5 ++ rewriteSteps6 ++ rewriteSteps7 ++
  rewriteSteps8 ++ rewriteSteps9 ++ rewriteSteps10 ++ rewriteSteps11 ++
  rewriteSteps12 ++ rewriteSteps13 ++ rewriteSteps14 ++ rewriteSteps15 ++
  rewriteSteps16 ++ rewriteSteps17 ++ rewriteSteps18 ++ rewriteSteps19 ++
  rewriteSteps20 ++ rewriteSteps21 ++ rewriteSteps22 ++ rewriteSteps23

def rewritePairs : List RawRewritePair :=
  rewritePairs0 ++ rewritePairs1 ++ rewritePairs2 ++ rewritePairs3 ++
  rewritePairs4 ++ rewritePairs5 ++ rewritePairs6 ++ rewritePairs7 ++
  rewritePairs8 ++ rewritePairs9 ++ rewritePairs10 ++ rewritePairs11 ++
  rewritePairs12 ++ rewritePairs13 ++ rewritePairs14 ++ rewritePairs15 ++
  rewritePairs16 ++ rewritePairs17 ++ rewritePairs18 ++ rewritePairs19 ++
  rewritePairs20 ++ rewritePairs21 ++ rewritePairs22 ++ rewritePairs23

/- The three cardinality inputs are proof-free scalar lists: 25 emitted
chunk lengths, 24 rewrite-step chunk lengths, and 25 pair-list lengths. -/
def emittedChunkLengths : List Nat := [
  EmittedRows.Chunk0.values.length, EmittedRows.Chunk1.values.length,
  EmittedRows.Chunk2.values.length, EmittedRows.Chunk3.values.length,
  EmittedRows.Chunk4.values.length, EmittedRows.Chunk5.values.length,
  EmittedRows.Chunk6.values.length, EmittedRows.Chunk7.values.length,
  EmittedRows.Chunk8.values.length, EmittedRows.Chunk9.values.length,
  EmittedRows.Chunk10.values.length, EmittedRows.Chunk11.values.length,
  EmittedRows.Chunk12.values.length, EmittedRows.Chunk13.values.length,
  EmittedRows.Chunk14.values.length, EmittedRows.Chunk15.values.length,
  EmittedRows.Chunk16.values.length, EmittedRows.Chunk17.values.length,
  EmittedRows.Chunk18.values.length, EmittedRows.Chunk19.values.length,
  EmittedRows.Chunk20.values.length, EmittedRows.Chunk21.values.length,
  EmittedRows.Chunk22.values.length, EmittedRows.Chunk23.values.length,
  EmittedRows.Chunk24.values.length]

def rewriteStepChunkLengths : List Nat := [
  rewriteSteps0.length, rewriteSteps1.length, rewriteSteps2.length,
  rewriteSteps3.length, rewriteSteps4.length, rewriteSteps5.length,
  rewriteSteps6.length, rewriteSteps7.length, rewriteSteps8.length,
  rewriteSteps9.length, rewriteSteps10.length, rewriteSteps11.length,
  rewriteSteps12.length, rewriteSteps13.length, rewriteSteps14.length,
  rewriteSteps15.length, rewriteSteps16.length, rewriteSteps17.length,
  rewriteSteps18.length, rewriteSteps19.length, rewriteSteps20.length,
  rewriteSteps21.length, rewriteSteps22.length, rewriteSteps23.length]

def pairChunkLengths : List Nat := [
  retainedPairs.length, rewritePairs0.length, rewritePairs1.length,
  rewritePairs2.length, rewritePairs3.length, rewritePairs4.length,
  rewritePairs5.length, rewritePairs6.length, rewritePairs7.length,
  rewritePairs8.length, rewritePairs9.length, rewritePairs10.length,
  rewritePairs11.length, rewritePairs12.length, rewritePairs13.length,
  rewritePairs14.length, rewritePairs15.length, rewritePairs16.length,
  rewritePairs17.length, rewritePairs18.length, rewritePairs19.length,
  rewritePairs20.length, rewritePairs21.length, rewritePairs22.length,
  rewritePairs23.length]

set_option maxRecDepth 100000 in
theorem emittedChunkLengthsExact :
    emittedChunkLengths = List.replicate 24 64 ++ [9] := by
  native_decide

set_option maxRecDepth 100000 in
theorem rewriteStepChunkLengthsExact :
    rewriteStepChunkLengths = List.replicate 23 64 ++ [21] := by
  native_decide

set_option maxRecDepth 100000 in
theorem pairChunkLengthsExact :
    pairChunkLengths = [52] ++ List.replicate 23 64 ++ [21] := by
  native_decide

theorem allPairChunksWithinCertificateLimit :
    ∀ count ∈ pairChunkLengths, count ≤ 256 := by
  rw [pairChunkLengthsExact]
  decide

private theorem take_drop_append {alpha : Type} (count : Nat)
    (values rest : List alpha) :
    values.take count ++ (values.drop count ++ rest) = values ++ rest := by
  rw [← List.append_assoc, List.take_append_drop]

/-- Every generated emitted row is owned by exactly one side of the pair
partition.  This is list equality, not count or interval evidence. -/
theorem emittedRowsExact :
    retainedRows ++ rewriteRows = EmittedRows.values := by
  simp only [retainedRows, rewriteRows, rewriteRows0, rewriteRows1,
    rewriteRows2, rewriteRows3, rewriteRows4, rewriteRows5, rewriteRows6,
    rewriteRows7, rewriteRows8, rewriteRows9, rewriteRows10, rewriteRows11,
    rewriteRows12, rewriteRows13, rewriteRows14, rewriteRows15,
    rewriteRows16, rewriteRows17, rewriteRows18, rewriteRows19,
    rewriteRows20, rewriteRows21, rewriteRows22, rewriteRows23, across,
    retainedCount, EmittedRows.values]
  simp only [List.append_assoc, take_drop_append]

theorem rewriteStepsExact : rewriteSteps = Provenance.rewriteSteps := by
  rfl

theorem retainedStepsExact : retainedSteps = Provenance.retainedSteps := by
  rfl

theorem emittedRowCountExact :
    (retainedRows ++ rewriteRows).length = 1545 := by
  rw [emittedRowsExact]
  have lengthAsScalarSum :
      EmittedRows.values.length = emittedChunkLengths.sum := by
    simp only [EmittedRows.values, emittedChunkLengths,
      List.length_append, List.sum_cons, List.sum_nil, Nat.add_assoc,
      Nat.add_zero]
  have summed := congrArg List.sum emittedChunkLengthsExact
  calc
    EmittedRows.values.length = emittedChunkLengths.sum := lengthAsScalarSum
    _ = (List.replicate 24 64 ++ [9]).sum := summed
    _ = 1545 := by decide

theorem rewritePairCountExact : rewritePairs.length = 1493 := by
  have lengthAsScalarSum :
      rewritePairs.length = (pairChunkLengths.drop 1).sum := by
    simp only [rewritePairs, pairChunkLengths, List.drop,
      List.length_append, List.sum_cons, List.sum_nil, Nat.add_assoc,
      Nat.add_zero]
  calc
    rewritePairs.length = (pairChunkLengths.drop 1).sum := lengthAsScalarSum
    _ = (([52] ++ List.replicate 23 64 ++ [21]).drop 1).sum := by
      rw [pairChunkLengthsExact]
    _ = 1493 := by decide

theorem retainedPairCountExact : retainedPairs.length = 52 := by
  have first := congrArg (fun values : List Nat => values.head?)
    pairChunkLengthsExact
  simpa [pairChunkLengths] using first

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectiveArtifactPairs.Chunks
