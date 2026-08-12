import Mathlib.Algebra.MvPolynomial.Equiv
import Mathlib.Algebra.MvPolynomial.CommRing
import Mathlib.Algebra.MvPolynomial.Degrees
import Mathlib.Algebra.Polynomial.Roots
import Nightstream.Protocol.NebulaV2.Chain

/-!
Contract: algebraic fingerprint obligations for PaddedRowIdentityMemoryV2.

Assurance tier: model-level.

Owns V2 integer packing, its injectivity on typed records, the two-variable
fingerprint polynomial, its nonzero proof for unequal multisets, and its total
degree bound.

Does not own the concrete quadratic-extension field certificate, challenge
sampling, Fiat-Shamir, or a probability theorem.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.Fingerprint

open Nightstream.Protocol.NebulaV2

def goldilocksModulus : Nat := 18446744069414584321

def timestampRadix : Nat := timestampLimit

def packedNat (entry : MemTuple) : Nat :=
  entry.timestamp + timestampRadix * entry.globalIndex

/-- Exact typed domain on which the V2 field encodings must be injective. -/
def TupleInRange (entry : MemTuple) : Prop :=
  entry.timestamp < timestampLimit ∧
    entry.globalIndex < scannedCells ∧
    entry.value < valueLimit

abbrev BoundedTuple := {entry : MemTuple // TupleInRange entry}

theorem packedNat_lt_twoPowForty
    {entry : MemTuple}
    (inRange : TupleInRange entry) :
    packedNat entry < 2 ^ 40 := by
  rcases inRange with ⟨timestampBound, indexBound, _⟩
  simp only [timestampLimit, timestampBits] at timestampBound
  simp only [scannedCells, romCells, ramCells] at indexBound
  simp only [packedNat, timestampRadix, timestampLimit, timestampBits]
  omega

theorem packedNat_lt_goldilocks
    {entry : MemTuple}
    (inRange : TupleInRange entry) :
    packedNat entry < goldilocksModulus := by
  have packedBound := packedNat_lt_twoPowForty inRange
  have fieldBound : 2 ^ 40 < goldilocksModulus := by decide
  exact packedBound.trans fieldBound

theorem value_lt_goldilocks
    {entry : MemTuple}
    (inRange : TupleInRange entry) :
    entry.value < goldilocksModulus := by
  have valueBound := inRange.2.2
  have fieldBound : valueLimit < goldilocksModulus := by decide
  exact valueBound.trans fieldBound

theorem packedNat_injective
    {left right : MemTuple}
    (leftInRange : TupleInRange left)
    (rightInRange : TupleInRange right)
    (equal : packedNat left = packedNat right) :
    left.timestamp = right.timestamp ∧
      left.globalIndex = right.globalIndex := by
  rcases leftInRange with ⟨leftTimestamp, _, _⟩
  rcases rightInRange with ⟨rightTimestamp, _, _⟩
  unfold packedNat timestampRadix at equal
  simp only [timestampLimit, timestampBits] at leftTimestamp rightTimestamp equal
  omega

theorem natCoordinates_injective :
    Function.Injective
      (fun entry : BoundedTuple =>
        (packedNat entry.1, entry.1.value)) := by
  intro left right equal
  have packedEqual := congrArg Prod.fst equal
  have valueEqual := congrArg Prod.snd equal
  have coordinates := packedNat_injective left.2 right.2 packedEqual
  apply Subtype.ext
  exact MemTuple.ext coordinates.1 coordinates.2 valueEqual

noncomputable section Polynomial

variable {Base Entry : Type}
variable [Field Base]

/-- Coefficient of the monic factor in `gamma2`. It is linear in `gamma1`. -/
def outerCoefficient
    (packed value : Entry → Base) (entry : Entry) :
  MvPolynomial (Fin 1) Base :=
  MvPolynomial.C (packed entry) +
    MvPolynomial.C (value entry) * MvPolynomial.X 0

def outerProduct
    (packed value : Entry → Base) (entries : Multiset Entry) :
    Polynomial (MvPolynomial (Fin 1) Base) :=
  ((entries.map (outerCoefficient packed value)).map
    (fun coefficient => Polynomial.X - Polynomial.C coefficient)).prod

theorem outerCoefficient_injective
    (packed value : Entry → Base)
    (coordinatesInjective :
      Function.Injective (fun entry => (packed entry, value entry))) :
    Function.Injective (outerCoefficient packed value) := by
  intro left right equal
  apply coordinatesInjective
  apply Prod.ext
  · have coefficientEqual := congrArg
      (fun p : MvPolynomial (Fin 1) Base => p.coeff 0) equal
    simpa [outerCoefficient] using coefficientEqual
  · have coefficientEqual := congrArg
      (fun p : MvPolynomial (Fin 1) Base =>
        p.coeff (Finsupp.single 0 1)) equal
    have zeroNeSingle :
        (0 : Fin 1 →₀ Nat) ≠ Finsupp.single 0 1 := by
      intro impossible
      have atZero := congrArg (fun exponent => exponent (0 : Fin 1)) impossible
      simp at atZero
    simpa [outerCoefficient, zeroNeSingle] using coefficientEqual

/-- Unique factorization in the monic `gamma2` variable makes the complete
factor product injective on multisets, including multiplicity. -/
theorem outerProduct_injective
    (packed value : Entry → Base)
    (coordinatesInjective :
      Function.Injective (fun entry => (packed entry, value entry))) :
    Function.Injective (outerProduct packed value) := by
  intro left right equal
  have rootsEqual := congrArg Polynomial.roots equal
  have coefficientMultisetsEqual :
      left.map (outerCoefficient packed value) =
        right.map (outerCoefficient packed value) := by
    unfold outerProduct at rootsEqual
    rw [Polynomial.roots_multiset_prod_X_sub_C,
      Polynomial.roots_multiset_prod_X_sub_C] at rootsEqual
    exact rootsEqual
  exact Multiset.map_injective
    (outerCoefficient_injective packed value coordinatesInjective)
    coefficientMultisetsEqual

/-- The bivariate factor in variables `(gamma2,gamma1)`. -/
def factor
    (packed value : Entry → Base) (entry : Entry) :
    MvPolynomial (Fin 2) Base :=
  MvPolynomial.X 0 -
    (MvPolynomial.C (packed entry) +
      MvPolynomial.C (value entry) * MvPolynomial.X 1)

def product
    (packed value : Entry → Base) (entries : Multiset Entry) :
    MvPolynomial (Fin 2) Base :=
  (entries.map (factor packed value)).prod

def difference
    (packed value : Entry → Base)
    (left right : Multiset Entry) : MvPolynomial (Fin 2) Base :=
  product packed value left - product packed value right

theorem factor_under_finSuccEquiv
    (packed value : Entry → Base) (entry : Entry) :
    MvPolynomial.finSuccEquiv Base 1 (factor packed value entry) =
      Polynomial.X -
        Polynomial.C (outerCoefficient packed value entry) := by
  have mapConstant (constant : Base) :
      MvPolynomial.finSuccEquiv Base 1 (MvPolynomial.C constant) =
        Polynomial.C (MvPolynomial.C constant) := by
    simp [MvPolynomial.finSuccEquiv_apply]
  have oneAsSucc : (1 : Fin 2) = Fin.succ (0 : Fin 1) := by decide
  unfold factor outerCoefficient
  rw [map_sub, map_add, map_mul, mapConstant, mapConstant,
    MvPolynomial.finSuccEquiv_X_zero, oneAsSucc,
    MvPolynomial.finSuccEquiv_X_succ]
  rw [Polynomial.C_add, Polynomial.C_mul]

theorem product_under_finSuccEquiv
    (packed value : Entry → Base) (entries : Multiset Entry) :
    MvPolynomial.finSuccEquiv Base 1 (product packed value entries) =
      outerProduct packed value entries := by
  unfold product outerProduct
  rw [map_multiset_prod]
  apply congrArg Multiset.prod
  simp only [Multiset.map_map]
  apply Multiset.map_congr rfl
  intro entry _
  exact factor_under_finSuccEquiv packed value entry

/-- Unequal typed multisets give a nonzero bivariate difference polynomial.
No set conversion occurs, so repeated records remain significant. -/
theorem difference_ne_zero
    (packed value : Entry → Base)
    (coordinatesInjective :
      Function.Injective (fun entry => (packed entry, value entry)))
    {left right : Multiset Entry}
    (unequal : left ≠ right) :
    difference packed value left right ≠ 0 := by
  intro zero
  have productsEqual :
      product packed value left = product packed value right :=
    sub_eq_zero.mp zero
  have outerEqual := congrArg (MvPolynomial.finSuccEquiv Base 1) productsEqual
  rw [product_under_finSuccEquiv, product_under_finSuccEquiv] at outerEqual
  exact unequal (outerProduct_injective packed value coordinatesInjective outerEqual)

theorem factor_totalDegree_le_one
    (packed value : Entry → Base) (entry : Entry) :
    (factor packed value entry).totalDegree ≤ 1 := by
  have innerBound :
      (MvPolynomial.C (packed entry) +
        MvPolynomial.C (value entry) *
          MvPolynomial.X (1 : Fin 2)).totalDegree ≤ 1 := by
    exact (MvPolynomial.totalDegree_add _ _).trans
      (max_le (by simp)
        ((MvPolynomial.totalDegree_mul _ _).trans (by simp)))
  unfold factor
  exact (MvPolynomial.totalDegree_sub _ _).trans
    (max_le (by simp) innerBound)

theorem product_totalDegree_le_card
    (packed value : Entry → Base) (entries : Multiset Entry) :
    (product packed value entries).totalDegree ≤ entries.card := by
  induction entries using Multiset.induction_on with
  | empty => simp [product]
  | @cons entry rest inductionHypothesis =>
      unfold product at inductionHypothesis ⊢
      simp only [Multiset.map_cons, Multiset.prod_cons, Multiset.card_cons]
      calc
        (factor packed value entry *
          (Multiset.map (factor packed value) rest).prod).totalDegree ≤
            (factor packed value entry).totalDegree +
              (Multiset.map (factor packed value) rest).prod.totalDegree :=
          MvPolynomial.totalDegree_mul _ _
        _ ≤ 1 + rest.card := Nat.add_le_add
          (factor_totalDegree_le_one packed value entry)
          inductionHypothesis
        _ = rest.card + 1 := Nat.add_comm _ _

/-- If both sides contain at most `factorLimit` records, the difference has
total degree at most that one-side limit. -/
theorem difference_totalDegree_le
    (packed value : Entry → Base)
    {left right : Multiset Entry}
    {factorLimit : Nat}
    (leftBound : left.card ≤ factorLimit)
    (rightBound : right.card ≤ factorLimit) :
    (difference packed value left right).totalDegree ≤ factorLimit := by
  exact (MvPolynomial.totalDegree_sub _ _).trans
    (max_le
      ((product_totalDegree_le_card packed value left).trans leftBound)
      ((product_totalDegree_le_card packed value right).trans rightBound))

end Polynomial

section BoundedEncoding

variable {ChallengeField : Type}

/-- The exact embedding property required from the concrete quadratic
extension: canonical integers below the Goldilocks modulus remain distinct. -/
def InjectiveBelowGoldilocks
    (encode : Nat → ChallengeField) : Prop :=
  ∀ {left right : Nat},
    left < goldilocksModulus →
    right < goldilocksModulus →
    encode left = encode right →
    left = right

def packedCoordinate
    (encode : Nat → ChallengeField) (entry : BoundedTuple) : ChallengeField :=
  encode (packedNat entry.1)

def valueCoordinate
    (encode : Nat → ChallengeField) (entry : BoundedTuple) : ChallengeField :=
  encode entry.1.value

theorem boundedCoordinates_injective
    (encode : Nat → ChallengeField)
    (encodeInjective : InjectiveBelowGoldilocks encode) :
    Function.Injective
      (fun entry : BoundedTuple =>
        (packedCoordinate encode entry, valueCoordinate encode entry)) := by
  intro left right equal
  apply natCoordinates_injective
  apply Prod.ext
  · apply encodeInjective
      (packedNat_lt_goldilocks left.2)
      (packedNat_lt_goldilocks right.2)
    exact congrArg Prod.fst equal
  · apply encodeInjective
      (value_lt_goldilocks left.2)
      (value_lt_goldilocks right.2)
    exact congrArg Prod.snd equal

variable [Field ChallengeField]

/-- Concrete V2 nonzero-polynomial bridge. A release must instantiate the
field and embedding premises for `F[U]/(U^2-7)`. -/
theorem boundedDifference_ne_zero
    (encode : Nat → ChallengeField)
    (encodeInjective : InjectiveBelowGoldilocks encode)
    {left right : Multiset BoundedTuple}
    (unequal : left ≠ right) :
    difference (packedCoordinate encode) (valueCoordinate encode)
      left right ≠ 0 :=
  difference_ne_zero _ _
    (boundedCoordinates_injective encode encodeInjective) unequal

end BoundedEncoding

/-- Maximum factors on either side of the V2 segment identity. -/
def maxSegmentFactors : Nat := scannedCells + 63 * 1088

theorem maxSegmentFactors_eq : maxSegmentFactors = 138176 := by
  decide

end Nightstream.Protocol.NebulaV2.Fingerprint
