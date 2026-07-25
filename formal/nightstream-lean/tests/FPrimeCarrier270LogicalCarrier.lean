import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LogicalCarrier

/-!
Focused executable and theorem-surface regression for the paper-owned
fixed-active public carrier.

These checks cover the carrier edge cases that are independent of any row
artifact: nonzero tail rejection, exact ordering, public-coordinate
permutation, coordinate coverage, PiDEC closure, and a sampler-valid PiRLC
shift into coordinate 257.
-/

set_option autoImplicit false

namespace tests.FPrimeCarrier270LogicalCarrier

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.LogicalCarrier

/-- Small semantic dimensions; the carrier width is fixed independently of
these non-public fields. -/
def dimensions : Dimensions where
  rowVariables := 1
  legacyLogicalWidth := legacyPublicWidth
  matrixCount := 1
  legacyPublicFits := Nat.le_refl _

#check coordinateEquiv_bijective
#check encodeFresh_externalOfLegacy_eq_expectedPublicInput
#check decodeFresh_sound
#check decodeFresh_complete
#check encodeFresh_injective
#check decodeFresh_rejects_nonzero_padding
#check projectExternal_not_injective
#check piDecSplit_recompose
#check shiftChallenge_valid
#check shift_enters_first_padding
#check freshImage_not_piRlcClosed


/-- The new codec agrees exactly with the existing typed fresh public
projection, without using any row artifact. -/
example (legacy : LegacyAssignment dimensions) :
    encodeFresh dimensions (externalOfLegacy dimensions legacy) =
      expectedPublicInput dimensions legacy :=
  encodeFresh_externalOfLegacy_eq_expectedPublicInput dimensions legacy

/-- One at external coordinate zero. -/
def firstExternalOne : ExternalInput :=
  fun logical => if logical.val = 0 then 1 else 0

/-- The same value moved to external coordinate one. -/
def swappedExternalOne : ExternalInput :=
  fun logical => if logical.val = 1 then 1 else 0

/-- Swapping two public coordinates changes the authoritative external input. -/
theorem firstExternalOne_ne_swappedExternalOne :
    firstExternalOne ≠ swappedExternalOne := by
  intro equal
  have atZero := congrFun equal (⟨0, by decide⟩ : Fin legacyPublicWidth)
  have oneEqZero : (1 : F) = 0 := by
    simpa [firstExternalOne, swappedExternalOne] using atZero
  exact (by decide : (1 : F) ≠ 0) oneEqZero

/-- The typed encoder preserves that distinction; coordinate permutation is
not accepted as an alternate ordering. -/
example :
    encodeFresh dimensions firstExternalOne ≠
      encodeFresh dimensions swappedExternalOne := by
  intro equal
  exact firstExternalOne_ne_swappedExternalOne
    ((encodeFresh_injective dimensions) equal)

/-- Exact ordering of all 257 externally owned coordinates. -/
example (logical : Fin legacyPublicWidth) :
    (carrierColumn dimensions (.inl logical)).val = logical.val := by
  rfl

/-- Exact ordering of all thirteen verifier-owned fresh coordinates. -/
example (padding : Fin fixedPaddingWidth) :
    (carrierColumn dimensions (.inr padding)).val =
      legacyPublicWidth + padding.val := by
  rfl

/-- No public coordinate can be omitted, duplicated, or aliased. -/
example :
    Function.Injective (carrierColumn dimensions) /\
      Function.Surjective (carrierColumn dimensions) :=
  coordinateEquiv_bijective dimensions

/-- Honest fresh encodings execute successfully. -/
example :
    freshCanonicalCheck dimensions
      (encodeFresh dimensions firstExternalOne) = true :=
  encodeFresh_freshCanonical dimensions firstExternalOne

/-- A nonzero value in the first of the thirteen tail coordinates fails the
executable fresh check. -/
example :
    freshCanonicalCheck dimensions (firstPaddingOne dimensions) = false := by
  cases canonical : freshCanonicalCheck dimensions (firstPaddingOne dimensions) with
  | false => rfl
  | true =>
      have zero :=
        (freshCanonical_iff dimensions (firstPaddingOne dimensions)).1
          canonical firstPadding
      have one :
          paddingValue dimensions (firstPaddingOne dimensions) firstPadding = 1 := by
        simp [paddingValue, firstPaddingOne]
      exact False.elim ((by decide : (1 : F) ≠ 0) (one.symm.trans zero))

/-- The fail-closed decoder rejects that nonzero tail. -/
example : decodeFresh dimensions (firstPaddingOne dimensions) = none := by
  apply decodeFresh_rejects_nonzero_padding dimensions _ firstPadding
  have one :
      paddingValue dimensions (firstPaddingOne dimensions) firstPadding = 1 := by
    simp [paddingValue, firstPaddingOne]
  rw [one]
  decide

/-- The legacy 257-coordinate view is ambiguous on complete running inputs. -/
example : ¬ Function.Injective (projectExternal dimensions) :=
  projectExternal_not_injective dimensions

/-- PiDEC splitting/recomposition stays in the complete paper carrier. -/
example (input : LIn dimensions) :
    piDecRecompose dimensions (piDecSplit dimensions input) = input :=
  piDecSplit_recompose dimensions input

/-- The production-valid shift challenge moves coordinate 256 into the first
authoritative tail coordinate. -/
example :
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicAct
        shiftChallenge (encodeFresh dimensions finalExternalOne)
        (firstPaddingColumn dimensions) = 1 :=
  shift_enters_first_padding dimensions

/-- Therefore the zero-tail fresh image is not a running-state invariant. -/
example :
    ¬ FreshCanonical dimensions
      (Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicAct
        shiftChallenge (encodeFresh dimensions finalExternalOne)) :=
  freshImage_not_piRlcClosed dimensions

end tests.FPrimeCarrier270LogicalCarrier
