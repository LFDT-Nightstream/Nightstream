import Nightstream.Protocol.NebulaV2.CompactCommit

set_option autoImplicit false

namespace Nightstream.Tests.NebulaV2CompactCommit

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CompactCommit
open Nightstream.Protocol.NebulaV2.ShiftedTernary41V1

example :
    commitmentFieldCount = 972 ∧
      primaryOutputFieldCount = 108 ∧
      tokenFieldCount = 54 ∧
      primaryMessageRingColumns = 738 ∧
      shortMessageRingColumns = 82 :=
  exact_dimensions

example {fieldCount : Nat} :
    Function.Injective
      (encodeFields : FieldVector fieldCount → CenteredMessage fieldCount) :=
  encodeFields_injective

example : Function.Injective (packFields primaryPacking) :=
  packFields_injective primaryPacking

example (commitment : CommitmentEncoding)
    (column : Fin primaryMessageRingColumns)
    (coefficient : Fin ringDegree) :
    (packFields primaryPacking commitment column coefficient).natAbs ≤ 1 :=
  packFields_unit_bound primaryPacking commitment column coefficient

example :
    primaryPacking.indexEquiv
        (⟨0, by decide⟩, ⟨0, by decide⟩) =
      (⟨0, by decide⟩, ⟨0, by decide⟩) := by
  decide

example :
    primaryPacking.indexEquiv
        (⟨737, by decide⟩, ⟨53, by decide⟩) =
      (⟨971, by decide⟩, ⟨40, by decide⟩) := by
  decide

example {Plan Seed : Type} (key : Key Plan Seed) (role : Role)
    {left right : CommitmentEncoding}
    (different : left ≠ right)
    (collision : key.token role left = key.token role right) :
    PrimaryBindingFailure key role ∨ ShortBindingFailure key role :=
  token_collision_implies_primary_or_short_failure key role different collision

example {Plan Seed : Type} (key : Key Plan Seed) (role : Role)
    (primarySecure : ¬ PrimaryBindingFailure key role)
    (shortSecure : ¬ ShortBindingFailure key role) :
    Function.Injective (key.token role) :=
  token_injective_of_no_binding_failure key role primarySecure shortSecure

def canonicalZero : CanonicalGoldilocks :=
  ⟨0, by decide⟩

def canonicalOne : CanonicalGoldilocks :=
  ⟨1, by decide⟩

def firstCommitmentCoordinate : Fin commitmentFieldCount :=
  ⟨0, by decide⟩

def zeroCommitment : CommitmentEncoding :=
  fun _ => canonicalZero

def oneAtFirstCommitment : CommitmentEncoding :=
  fun index => if index = firstCommitmentCoordinate then canonicalOne
    else canonicalZero

theorem zeroCommitment_ne_oneAtFirstCommitment :
    zeroCommitment ≠ oneAtFirstCommitment := by
  intro equal
  have atFirst := congrFun equal firstCommitmentCoordinate
  have valuesEqual := congrArg Subtype.val atFirst
  simp [zeroCommitment, oneAtFirstCommitment, canonicalZero, canonicalOne]
    at valuesEqual

/-- This key meets the structural seed-separation field, but its primary map
is constant. The security model reports an exact primary binding failure; it
does not infer binding from linearity or seed labels. -/
def zeroPrimaryKey : Key Unit Bool where
  profile := Profile.v2
  plan := ()
  primarySeed
    | .operations => false
    | .memory => true
  primarySeedIndependent := by decide
  shortSeed
    | .operations => false
    | .memory => true
  shortSeedIndependent := by decide
  primaryFromSeed := fun _seed _message _index => canonicalZero
  shortFromSeed := fun _seed _message _index => canonicalZero

theorem zero_primary_map_is_a_named_failure :
    PrimaryBindingFailure zeroPrimaryKey .operations := by
  refine ⟨zeroCommitment, oneAtFirstCommitment,
    zeroCommitment_ne_oneAtFirstCommitment, ?_⟩
  change (fun _ : Fin primaryOutputFieldCount => canonicalZero) =
    (fun _ : Fin primaryOutputFieldCount => canonicalZero)
  rfl

end Nightstream.Tests.NebulaV2CompactCommit
