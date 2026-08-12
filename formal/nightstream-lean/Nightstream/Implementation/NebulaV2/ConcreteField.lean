import Mathlib.Algebra.QuadraticAlgebra.Basic
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentitySecurity
import Nightstream.Protocol.NebulaV2.Fingerprint

/-!
Contract: instantiate the Nebula V2 row field with SuperNeo's exact
Goldilocks quadratic extension.

Owns the field construction, the canonical integer embedding, its injectivity
below the Goldilocks modulus, and coefficient-level correspondence with the
concrete SuperNeo carrier `K`.

Does not own transcript sampling, Fiat--Shamir security, generated rows, Rust
arithmetic, or a probability union bound.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ConcreteField

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Protocol.NebulaV2.Fingerprint
open Nightstream.SuperNeo.Concrete

instance goldilocksPrimeFact : Fact (Nat.Prime goldilocksP) :=
  ⟨GoldilocksField.goldilocks_natPrime⟩

/-- Base field used by the exact challenge carrier. -/
abbrev Base := ZMod goldilocksP

/-- Seven is not a square in the Goldilocks base field. This proof reuses the
same checked generator certificate as SuperNeo's extension proof. -/
theorem seven_not_square (value : Base) :
    value ^ 2 ≠ (7 : Base) := by
  intro square
  have valueNonzero : value ≠ 0 := by
    intro valueZero
    subst value
    have sevenZero : (7 : Base) = 0 := by simpa using square.symm
    exact (by decide : ¬ goldilocksP ∣ 7)
      ((ZMod.natCast_eq_zero_iff 7 goldilocksP).mp sevenZero)
  let exponent := (goldilocksP - 1) / 2
  have twiceExponent : 2 * exponent = goldilocksP - 1 := by
    decide
  have raised :
      value ^ (goldilocksP - 1) = (7 : Base) ^ exponent := by
    calc
      value ^ (goldilocksP - 1) = value ^ (2 * exponent) := by
        rw [twiceExponent]
      _ = (value ^ 2) ^ exponent := by rw [pow_mul]
      _ = (7 : Base) ^ exponent := by rw [square]
  have sevenPowerOne : (7 : Base) ^ exponent = 1 := by
    rw [ZMod.pow_card_sub_one_eq_one valueNonzero] at raised
    exact raised.symm
  exact GoldilocksField.order_not_halved_zmod (by
    simpa [exponent] using sevenPowerOne)

instance sevenRootFact :
    Fact (∀ value : Base,
      value ^ 2 ≠ (7 : Base) + (0 : Base) * value) :=
  ⟨by
    intro value
    simpa using seven_not_square value⟩

/-- Exact V2 challenge field `Goldilocks[U]/(U^2 - 7)`. -/
abbrev ChallengeField := QuadraticAlgebra Base (7 : Base) (0 : Base)

/-- Canonical coefficient embedding used by record fingerprints. -/
def encode (value : Nat) : ChallengeField :=
  ⟨(value : Base), 0⟩

theorem encode_injective_below_goldilocks :
    InjectiveBelowGoldilocks encode := by
  intro left right leftBound rightBound equal
  have coefficients := congrArg QuadraticAlgebra.re equal
  change (left : Base) = (right : Base) at coefficients
  have values := congrArg ZMod.val coefficients
  have leftBound' : left < goldilocksP := by
    simpa [Nightstream.Protocol.NebulaV2.Fingerprint.goldilocksModulus,
      goldilocksP] using leftBound
  have rightBound' : right < goldilocksP := by
    simpa [Nightstream.Protocol.NebulaV2.Fingerprint.goldilocksModulus,
      goldilocksP] using rightBound
  simpa [ZMod.val_natCast_of_lt leftBound',
    ZMod.val_natCast_of_lt rightBound'] using values

/-- Coefficient representation of the exact challenge field. -/
def coefficientPairEquiv : (Base × Base) ≃ ChallengeField where
  toFun pair := ⟨pair.1, pair.2⟩
  invFun value := (value.re, value.im)
  left_inv := fun _ => rfl
  right_inv := fun _ => rfl

noncomputable instance challengeFieldFintype : Fintype ChallengeField :=
  Fintype.ofEquiv (Base × Base) coefficientPairEquiv

theorem challengeField_cardinality :
    Fintype.card ChallengeField = goldilocksP * goldilocksP := by
  calc
    Fintype.card ChallengeField = Fintype.card (Base × Base) :=
      (Fintype.card_congr coefficientPairEquiv).symm
    _ = goldilocksP * goldilocksP := by simp

/-- Exact coefficient bijection from SuperNeo's executable pair carrier to
the Mathlib field carrier used by the fingerprint proof. -/
def superNeoEquiv : K ≃ ChallengeField where
  toFun value :=
    ⟨ZMod.finEquiv goldilocksP value.c0,
      ZMod.finEquiv goldilocksP value.c1⟩
  invFun value :=
    ⟨(ZMod.finEquiv goldilocksP).symm value.re,
      (ZMod.finEquiv goldilocksP).symm value.im⟩
  left_inv := by
    intro value
    cases value
    simp
  right_inv := by
    intro value
    cases value
    simp

theorem superNeoEquiv_zero :
    superNeoEquiv K.zero = (0 : ChallengeField) := by
  rfl

theorem superNeoEquiv_one :
    superNeoEquiv K.one = (1 : ChallengeField) := by
  rfl

theorem superNeoEquiv_add (left right : K) :
    superNeoEquiv (K.add left right) =
      superNeoEquiv left + superNeoEquiv right := by
  cases left
  cases right
  ext
  · exact (ZMod.finEquiv goldilocksP).map_add _ _
  · exact (ZMod.finEquiv goldilocksP).map_add _ _

theorem superNeoEquiv_sub (left right : K) :
    superNeoEquiv (K.sub left right) =
      superNeoEquiv left - superNeoEquiv right := by
  cases left
  cases right
  ext
  · exact (ZMod.finEquiv goldilocksP).map_sub _ _
  · exact (ZMod.finEquiv goldilocksP).map_sub _ _

theorem superNeoEquiv_mul (left right : K) :
    superNeoEquiv (K.mul left right) =
      superNeoEquiv left * superNeoEquiv right := by
  rcases left with ⟨leftReal, leftImaginary⟩
  rcases right with ⟨rightReal, rightImaginary⟩
  have sevenMap :
      (ZMod.finEquiv goldilocksP) (7 : F) = (7 : Base) := by
    rfl
  apply QuadraticAlgebra.ext
  · change
      (ZMod.finEquiv goldilocksP) (_ * _ + 7 * _ * _) =
        (ZMod.finEquiv goldilocksP) _ *
            (ZMod.finEquiv goldilocksP) _ +
          7 * (ZMod.finEquiv goldilocksP) _ *
            (ZMod.finEquiv goldilocksP) _
    rw [(ZMod.finEquiv goldilocksP).map_add,
      (ZMod.finEquiv goldilocksP).map_mul,
      (ZMod.finEquiv goldilocksP).map_mul,
      (ZMod.finEquiv goldilocksP).map_mul]
    exact congrArg
      (fun constant : Base =>
        (ZMod.finEquiv goldilocksP) leftReal *
            (ZMod.finEquiv goldilocksP) rightReal +
          constant * (ZMod.finEquiv goldilocksP) leftImaginary *
            (ZMod.finEquiv goldilocksP) rightImaginary)
      sevenMap
  · change
      (ZMod.finEquiv goldilocksP) (_ * _ + _ * _) =
        (ZMod.finEquiv goldilocksP) _ *
            (ZMod.finEquiv goldilocksP) _ +
          (ZMod.finEquiv goldilocksP) _ *
            (ZMod.finEquiv goldilocksP) _ +
          0 * (ZMod.finEquiv goldilocksP) _ *
            (ZMod.finEquiv goldilocksP) _
    rw [(ZMod.finEquiv goldilocksP).map_add,
      (ZMod.finEquiv goldilocksP).map_mul,
      (ZMod.finEquiv goldilocksP).map_mul,
      zero_mul, zero_mul, add_zero]

theorem superNeoEquiv_embed (value : F) :
    superNeoEquiv (K.embed value) = encode value.val := by
  ext
  · change (ZMod.finEquiv goldilocksP) value =
      (value.val : ZMod goldilocksP)
    apply ZMod.val_injective
    have leftValue :
        ((ZMod.finEquiv goldilocksP) value).val = value.val := by
      rfl
    calc
      ((ZMod.finEquiv goldilocksP) value).val = value.val := leftValue
      _ = (value.val : ZMod goldilocksP).val :=
        (ZMod.val_natCast_of_lt value.isLt).symm
  · rfl

end Nightstream.Implementation.NebulaV2.ConcreteField
