import Nightstream.Protocol.Nebula

set_option autoImplicit false

namespace tests.NebulaFingerprint

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint

def firstEntry : BoundedTuple :=
  ⟨⟨0, 0, 1⟩, by
    constructor
    · decide
    constructor <;> decide⟩

def secondEntry : BoundedTuple :=
  ⟨⟨0, 0, 2⟩, by
    constructor
    · decide
    constructor <;> decide⟩

def rationalEncoding (value : Nat) : ℚ := value

theorem rational_encoding_injective :
    InjectiveBelowGoldilocks rationalEncoding := by
  intro left right _ _ equal
  change (left : ℚ) = (right : ℚ) at equal
  exact_mod_cast equal

theorem singleton_multisets_differ :
    ({firstEntry} : Multiset BoundedTuple) ≠ {secondEntry} := by
  intro equal
  have entryEqual : firstEntry = secondEntry := by
    simpa using equal
  have valueEqual := congrArg
    (fun entry : BoundedTuple => entry.1.value) entryEqual
  change 1 = 2 at valueEqual
  omega

theorem unequal_singletons_give_nonzero_polynomial :
    difference
      (packedCoordinate rationalEncoding)
      (valueCoordinate rationalEncoding)
      ({firstEntry} : Multiset BoundedTuple)
      ({secondEntry} : Multiset BoundedTuple) ≠ 0 :=
  boundedDifference_ne_zero rationalEncoding rational_encoding_injective
    singleton_multisets_differ

theorem duplicate_multisets_differ :
    ({firstEntry, firstEntry} : Multiset BoundedTuple) ≠ {firstEntry} := by
  intro equal
  have cardEqual := congrArg Multiset.card equal
  simp at cardEqual

theorem set_conversion_loses_multiplicity :
    ({firstEntry, firstEntry} : Multiset BoundedTuple).toFinset =
      ({firstEntry} : Multiset BoundedTuple).toFinset := by
  simp

theorem duplicate_difference_is_nonzero :
    difference
      (packedCoordinate rationalEncoding)
      (valueCoordinate rationalEncoding)
      ({firstEntry, firstEntry} : Multiset BoundedTuple)
      ({firstEntry} : Multiset BoundedTuple) ≠ 0 :=
  boundedDifference_ne_zero rationalEncoding rational_encoding_injective
    duplicate_multisets_differ

theorem duplicate_difference_has_profile_degree_bound :
    (difference
      (packedCoordinate rationalEncoding)
      (valueCoordinate rationalEncoding)
      ({firstEntry, firstEntry} : Multiset BoundedTuple)
      ({firstEntry} : Multiset BoundedTuple)).totalDegree ≤
        maxSegmentFactors := by
  apply difference_totalDegree_le
  · simp [maxSegmentFactors, scannedCells, romCells, ramCells]
  · simp [maxSegmentFactors, scannedCells, romCells, ramCells]

/- If the canonical integer embedding is not injective, distinct records can
produce the same factor. This is why canonical field decoding is necessary. -/
namespace MissingCanonicalEmbedding

def zeroEncoding (_value : Nat) : ℚ := 0

theorem distinct_singletons_have_equal_product :
    product
      (packedCoordinate zeroEncoding)
      (valueCoordinate zeroEncoding)
      ({firstEntry} : Multiset BoundedTuple) =
    product
      (packedCoordinate zeroEncoding)
      (valueCoordinate zeroEncoding)
      ({secondEntry} : Multiset BoundedTuple) := by
  simp [product, factor, packedCoordinate, valueCoordinate, zeroEncoding]

end MissingCanonicalEmbedding

end tests.NebulaFingerprint
