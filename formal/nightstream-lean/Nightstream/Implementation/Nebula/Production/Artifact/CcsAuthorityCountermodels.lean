import Nightstream.Implementation.Nebula.Production.Memory.BoundCcsPublic

/-!
Contract: concrete countermodels for incomplete production CCS authority.

The examples prove that matching only the memory-digest half does not fix
the affine coordinate, the state-digest half, or the padding suffix.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionCcsAuthorityCountermodels

open Nightstream.Implementation.Nebula.MemoryBoundCcsPublic
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ProductionProfileCandidates

abbrev PCanonicalDigest :=
  ProductionMemoryBoundCcsPublic.CanonicalDigest
abbrev PBatch := ProductionMemoryBoundCcsPublic.Batch

def badAffineEncode {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate) : List Nat :=
  [0] ++ digestBits stateDigest ++
    digestBits (ProductionMemoryBoundCcsPublic.memoryDigest batch) ++
    List.replicate paddingBitCount 0

theorem badAffineEncode_length {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate) :
    (badAffineEncode stateDigest batch).length = coordinateCount := by
  norm_num [badAffineEncode, digestBits_length, digestBitCount,
    paddingBitCount, coordinateCount]

theorem badAffineEncode_binary {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate)
    (digit : Nat) (member : digit ∈ badAffineEncode stateDigest batch) :
    digit < 2 := by
  simp only [badAffineEncode, List.mem_append, List.mem_cons,
    List.not_mem_nil, or_false, List.mem_replicate] at member
  rcases member with ((zero | stateMember) | memoryMember) | paddingMember
  · subst digit
    decide
  · exact digestBits_binary stateDigest digit stateMember
  · exact digestBits_binary
      (ProductionMemoryBoundCcsPublic.memoryDigest batch) digit memoryMember
  · exact paddingMember.2 ▸ (by decide)

def badAffineWord {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate) :
    FixedBits.Word coordinateCount :=
  ⟨badAffineEncode stateDigest batch,
    badAffineEncode_length stateDigest batch,
    badAffineEncode_binary stateDigest batch⟩

theorem badAffine_memoryMatches {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate) :
    ProductionMemoryBoundCcsPublic.MemoryMatches
      (badAffineWord stateDigest batch) batch := by
  simp [ProductionMemoryBoundCcsPublic.MemoryMatches,
    badAffineWord, badAffineEncode,
    digestBits_length, digestBitCount]

theorem badAffine_not_fullMatches {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate) :
    ¬ ProductionMemoryBoundCcsPublic.FullMatches
      (badAffineWord stateDigest batch) stateDigest batch := by
  intro full
  have first := congrArg (fun values : List Nat => values.getD 0 0) full
  norm_num [ProductionMemoryBoundCcsPublic.FullMatches, badAffineWord,
    badAffineEncode, ProductionMemoryBoundCcsPublic.encode] at first

def zeroStateDigest : PCanonicalDigest :=
  fun _ => ⟨0, by decide⟩

def oneStateDigest : PCanonicalDigest :=
  fun lane => if lane = 0 then ⟨1, by decide⟩ else ⟨0, by decide⟩

theorem zeroStateDigest_ne_oneStateDigest :
    zeroStateDigest ≠ oneStateDigest := by
  intro equal
  have lane := congrFun equal (0 : Fin 4)
  have values := congrArg Subtype.val lane
  norm_num [zeroStateDigest, oneStateDigest] at values

theorem wrongState_memoryMatches {candidate : Id} (batch : PBatch candidate) :
    ProductionMemoryBoundCcsPublic.MemoryMatches
      (ProductionMemoryBoundCcsPublic.word zeroStateDigest batch) batch :=
  ProductionMemoryBoundCcsPublic.word_memoryMatches zeroStateDigest batch

theorem wrongState_not_fullMatches {candidate : Id} (batch : PBatch candidate) :
    ¬ ProductionMemoryBoundCcsPublic.FullMatches
      (ProductionMemoryBoundCcsPublic.word zeroStateDigest batch)
      oneStateDigest batch := by
  intro full
  have encodingEqual :
      ProductionMemoryBoundCcsPublic.encode zeroStateDigest batch =
        ProductionMemoryBoundCcsPublic.encode oneStateDigest batch := by
    exact full
  exact zeroStateDigest_ne_oneStateDigest
    (ProductionMemoryBoundCcsPublic.stateDigest_eq_of_encode_eq encodingEqual)

def badPaddingEncode {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate) : List Nat :=
  [1] ++ digestBits stateDigest ++
    digestBits (ProductionMemoryBoundCcsPublic.memoryDigest batch) ++
    List.replicate 26 0 ++ [1]

theorem badPaddingEncode_length {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate) :
    (badPaddingEncode stateDigest batch).length = coordinateCount := by
  norm_num [badPaddingEncode, digestBits_length, digestBitCount,
    coordinateCount]

theorem badPaddingEncode_binary {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate)
    (digit : Nat) (member : digit ∈ badPaddingEncode stateDigest batch) :
    digit < 2 := by
  simp only [badPaddingEncode, List.mem_append, List.mem_cons,
    List.not_mem_nil, or_false, List.mem_replicate] at member
  rcases member with (((one | stateMember) | memoryMember) |
      zeroPadding) | finalOne
  · subst digit
    decide
  · exact digestBits_binary stateDigest digit stateMember
  · exact digestBits_binary
      (ProductionMemoryBoundCcsPublic.memoryDigest batch) digit memoryMember
  · exact zeroPadding.2 ▸ (by decide)
  · subst digit
    decide

def badPaddingWord {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate) :
    FixedBits.Word coordinateCount :=
  ⟨badPaddingEncode stateDigest batch,
    badPaddingEncode_length stateDigest batch,
    badPaddingEncode_binary stateDigest batch⟩

theorem badPadding_memoryMatches {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate) :
    ProductionMemoryBoundCcsPublic.MemoryMatches
      (badPaddingWord stateDigest batch) batch := by
  simp [ProductionMemoryBoundCcsPublic.MemoryMatches,
    badPaddingWord, badPaddingEncode,
    digestBits_length, digestBitCount]

theorem badPadding_not_fullMatches {candidate : Id}
    (stateDigest : PCanonicalDigest) (batch : PBatch candidate) :
    ¬ ProductionMemoryBoundCcsPublic.FullMatches
      (badPaddingWord stateDigest batch) stateDigest batch := by
  intro full
  have last := congrArg (fun values : List Nat => values.getD 539 0) full
  norm_num [ProductionMemoryBoundCcsPublic.FullMatches, badPaddingWord,
    badPaddingEncode, ProductionMemoryBoundCcsPublic.encode,
    digestBits_length, digestBitCount, paddingBitCount] at last

end Nightstream.Implementation.Nebula.ProductionCcsAuthorityCountermodels
