import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients.Core

/-!
Exact bounded partition of the selective `y_zcol` coefficient certificates.

Owns: five 250-record rewrite partitions, the four-record retained
certificate, exact concatenation back to the canonical pair stream, and
ordered half-open position ownership.

Does not own: the coefficient computation inside a record, row satisfaction,
selector truth, source authority, security events, or permission to remove
rows.

Emits constraints: no.

Assurance tier: artifact-checked partition bookkeeping for the bounded fixture.

Both partition obligations are retained for this bounded profile.

| Stable stage path | Exact obligation | Authority class | Artifact owner | Lean owner | Multiplicity |
|---|---|---|---|---|---|
| `pi_rlc.y_zcol.selective.coefficients.rewrite_chunks` | five disjoint bounded certificates cover every rewrite pair exactly | computed | canonical rewrite pair stream | `rewriteCoefficientChunksExact` | five chunks |
| `pi_rlc.y_zcol.selective.coefficients.retained` | the retained certificate covers every retained pair exactly | computed | canonical retained pair stream | `retainedCoefficientDataLengthExact` | four records |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

private abbrev chunkSize : Nat := 250

theorem chunkSize_le_certificateLimit : chunkSize ≤ 256 := by
  decide

def rewriteCoefficientTail0 : List RewritePair := rewritePairs

def rewriteCoefficientChunk0 : List RewritePair :=
  rewriteCoefficientTail0.take chunkSize

def rewriteCoefficientTail1 : List RewritePair :=
  rewriteCoefficientTail0.drop chunkSize

def rewriteCoefficientChunk1 : List RewritePair :=
  rewriteCoefficientTail1.take chunkSize

def rewriteCoefficientTail2 : List RewritePair :=
  rewriteCoefficientTail1.drop chunkSize

def rewriteCoefficientChunk2 : List RewritePair :=
  rewriteCoefficientTail2.take chunkSize

def rewriteCoefficientTail3 : List RewritePair :=
  rewriteCoefficientTail2.drop chunkSize

def rewriteCoefficientChunk3 : List RewritePair :=
  rewriteCoefficientTail3.take chunkSize

def rewriteCoefficientTail4 : List RewritePair :=
  rewriteCoefficientTail3.drop chunkSize

def rewriteCoefficientChunk4 : List RewritePair := rewriteCoefficientTail4

def rewriteCoefficientDataChunk0 : List CoefficientMatchShape :=
  rewriteCoefficientChunk0.map rewritePairCoefficientShape

def rewriteCoefficientDataChunk1 : List CoefficientMatchShape :=
  rewriteCoefficientChunk1.map rewritePairCoefficientShape

def rewriteCoefficientDataChunk2 : List CoefficientMatchShape :=
  rewriteCoefficientChunk2.map rewritePairCoefficientShape

def rewriteCoefficientDataChunk3 : List CoefficientMatchShape :=
  rewriteCoefficientChunk3.map rewritePairCoefficientShape

def rewriteCoefficientDataChunk4 : List CoefficientMatchShape :=
  rewriteCoefficientChunk4.map rewritePairCoefficientShape

def retainedCoefficientData : List CoefficientMatchShape :=
  retainedPairs.map retainedPairCoefficientShape

/- The executable inputs here are two scalar natural-number cardinalities,
not decoded records. -/
set_option maxRecDepth 100000 in
theorem rewritePairsLengthExact : rewritePairs.length = 1250 := by
  native_decide

set_option maxRecDepth 100000 in
theorem retainedPairsLengthExact : retainedPairs.length = 4 := by
  native_decide

/-- Half-open pair positions owned by the five data certificates. -/
def rewriteCoefficientChunkRanges : List (Nat × Nat) :=
  [(0, 250), (250, 500), (500, 750), (750, 1000), (1000, 1250)]

theorem rewriteCoefficientChunkRangesOrdered :
    rewriteCoefficientChunkRanges.Pairwise
      (fun left right => left.2 ≤ right.1) := by
  decide

theorem rewriteCoefficientChunkLengthsExact :
    rewriteCoefficientChunk0.length = 250 ∧
      rewriteCoefficientChunk1.length = 250 ∧
      rewriteCoefficientChunk2.length = 250 ∧
      rewriteCoefficientChunk3.length = 250 ∧
      rewriteCoefficientChunk4.length = 250 := by
  simp [rewriteCoefficientChunk0, rewriteCoefficientChunk1,
    rewriteCoefficientChunk2, rewriteCoefficientChunk3,
    rewriteCoefficientChunk4, rewriteCoefficientTail0,
    rewriteCoefficientTail1, rewriteCoefficientTail2,
    rewriteCoefficientTail3, rewriteCoefficientTail4, chunkSize,
    rewritePairsLengthExact]

theorem rewriteCoefficientDataChunkLengthsExact :
    rewriteCoefficientDataChunk0.length = 250 ∧
      rewriteCoefficientDataChunk1.length = 250 ∧
      rewriteCoefficientDataChunk2.length = 250 ∧
      rewriteCoefficientDataChunk3.length = 250 ∧
      rewriteCoefficientDataChunk4.length = 250 := by
  simpa only [rewriteCoefficientDataChunk0,
    rewriteCoefficientDataChunk1, rewriteCoefficientDataChunk2,
    rewriteCoefficientDataChunk3, rewriteCoefficientDataChunk4,
    List.length_map] using rewriteCoefficientChunkLengthsExact

theorem rewriteCoefficientDataWithinCertificateLimit :
    rewriteCoefficientDataChunk0.length ≤ 256 ∧
      rewriteCoefficientDataChunk1.length ≤ 256 ∧
      rewriteCoefficientDataChunk2.length ≤ 256 ∧
      rewriteCoefficientDataChunk3.length ≤ 256 ∧
      rewriteCoefficientDataChunk4.length ≤ 256 := by
  rcases rewriteCoefficientDataChunkLengthsExact with
    ⟨chunk0, chunk1, chunk2, chunk3, chunk4⟩
  simp only [chunk0, chunk1, chunk2, chunk3, chunk4]
  decide

theorem retainedCoefficientDataLengthExact :
    retainedCoefficientData.length = 4 := by
  simpa only [retainedCoefficientData, List.length_map] using
    retainedPairsLengthExact

theorem retainedCoefficientDataWithinCertificateLimit :
    retainedCoefficientData.length ≤ 256 := by
  rw [retainedCoefficientDataLengthExact]
  decide

theorem rewriteCoefficientChunksExact :
    rewriteCoefficientChunk0 ++
        (rewriteCoefficientChunk1 ++
          (rewriteCoefficientChunk2 ++
            (rewriteCoefficientChunk3 ++ rewriteCoefficientChunk4))) =
      rewritePairs := by
  have split3 :
      rewriteCoefficientChunk3 ++ rewriteCoefficientChunk4 =
        rewriteCoefficientTail3 := by
    exact List.take_append_drop chunkSize rewriteCoefficientTail3
  have split2 :
      rewriteCoefficientChunk2 ++ rewriteCoefficientTail3 =
        rewriteCoefficientTail2 := by
    exact List.take_append_drop chunkSize rewriteCoefficientTail2
  have split1 :
      rewriteCoefficientChunk1 ++ rewriteCoefficientTail2 =
        rewriteCoefficientTail1 := by
    exact List.take_append_drop chunkSize rewriteCoefficientTail1
  have split0 :
      rewriteCoefficientChunk0 ++ rewriteCoefficientTail1 =
        rewriteCoefficientTail0 := by
    exact List.take_append_drop chunkSize rewriteCoefficientTail0
  simp only [split3, split2, split1, split0, rewriteCoefficientTail0]

theorem rewriteCoefficientDataChunksExact :
    rewriteCoefficientDataChunk0 ++
        (rewriteCoefficientDataChunk1 ++
          (rewriteCoefficientDataChunk2 ++
            (rewriteCoefficientDataChunk3 ++
              rewriteCoefficientDataChunk4))) =
      rewritePairs.map rewritePairCoefficientShape := by
  simpa only [rewriteCoefficientDataChunk0,
    rewriteCoefficientDataChunk1, rewriteCoefficientDataChunk2,
    rewriteCoefficientDataChunk3, rewriteCoefficientDataChunk4,
    List.map_append] using
    congrArg (List.map rewritePairCoefficientShape)
      rewriteCoefficientChunksExact

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge.Coefficients
