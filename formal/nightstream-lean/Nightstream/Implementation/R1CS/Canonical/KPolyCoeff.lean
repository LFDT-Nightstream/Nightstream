import Nightstream.Implementation.R1CS.Canonical.KConcreteBridge
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Contract: coefficient extraction for the list polynomial representation.

Owns: `coeffAt`, the lengths of `polyAdd`, `polyScale` and `polyMul`, the
coefficient of a sum, of a scaling and of a product, and the conversion from
the frozen `RingK`.

Does **not** own, and does not prove: the coefficient of a product. That is the
convolution characterization, and it is the remaining content of
`KPOLYMUL-RAWMULCOEFF-AGREEMENT`. This module supplies what that induction
needs and stops there.

## Why lengths come first

The frozen `rawMulCoeffK` folds over `List.range ringDegree` and guards each
term with `i ≤ degree ∧ degree - i < ringDegree`. Matching it against a list
representation requires knowing exactly where `coeffAt` starts returning the
default, which is a statement about `polyMul`'s length.

`polyAdd` takes the longer list, so its length is a maximum rather than a sum —
easy to state wrongly, and the reason `polyMul`'s length is not simply
`a.length + b.length`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPolyCoeff

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KPolyHom
open Nightstream.Implementation.R1CS.Canonical.KPairLaws

/-- The coefficient at an index, zero beyond the end. -/
def coeffAt (poly : List Pair) (index : Nat) : Pair :=
  poly.getD index ⟨0, 0⟩

theorem coeffAt_nil (index : Nat) : coeffAt [] index = ⟨0, 0⟩ := rfl

theorem coeffAt_cons_zero (c : Pair) (rest : List Pair) :
    coeffAt (c :: rest) 0 = c := rfl

theorem coeffAt_cons_succ (c : Pair) (rest : List Pair) (index : Nat) :
    coeffAt (c :: rest) (index + 1) = coeffAt rest index := rfl

/-- Past the end there is nothing, at every index.  What lets a length bound on
the raw convolution stand in for a degree bound. -/
theorem coeffAt_beyond_length (poly : List Pair) (index : Nat)
    (beyond : poly.length ≤ index) : coeffAt poly index = ⟨0, 0⟩ := by
  unfold coeffAt
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none beyond]
  rfl

/-- **Coefficients determine a polynomial**, given the length.  The length is
not redundant: `coeffAt` returns zero past the end, so `[x]` and `[x, ⟨0,0⟩]`
agree at every index while being different lists. -/
theorem list_ext_coeffAt (left right : List Pair)
    (sameLength : left.length = right.length)
    (sameCoeff : ∀ index, coeffAt left index = coeffAt right index) :
    left = right := by
  refine List.ext_getElem sameLength (fun index inLeft inRight => ?_)
  have entry := sameCoeff index
  unfold coeffAt at entry
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem inLeft, List.getElem?_eq_getElem inRight] at entry
  exact entry

/-! ## Lengths

`polyAdd` keeps the longer list, so this is a maximum, not a sum. -/

theorem polyAdd_length :
    ∀ left right : List Pair,
      (polyAdd left right).length = max left.length right.length
  | [], right => by simp [polyAdd]
  | a :: left, [] => by simp [polyAdd]
  | a :: left, b :: right => by
      show (addPair a b :: polyAdd left right).length
        = max (a :: left).length (b :: right).length
      simp only [List.length_cons, polyAdd_length left right]
      omega

theorem polyScale_length (scalar : Pair) (poly : List Pair) :
    (polyScale scalar poly).length = poly.length := by
  unfold polyScale
  exact List.length_map _

/-- The recursion step's length, stated on a cons so the induction needs no
case analysis: one scaled copy of the right side, overlaid on a shifted
product. -/
theorem polyMul_length_cons :
    ∀ (a : Pair) (left right : List Pair), 1 ≤ right.length →
      (polyMul (a :: left) right).length = left.length + right.length
  | a, [], right, _ => by
      show (polyAdd (polyScale a right) (⟨0, 0⟩ :: polyMul [] right)).length
        = 0 + right.length
      rw [polyAdd_length, polyScale_length]
      show max right.length ((⟨0, 0⟩ :: ([] : List Pair)).length)
        = 0 + right.length
      simp only [List.length_cons, List.length_nil]
      omega
  | a, b :: left, right, wide => by
      show (polyAdd (polyScale a right)
          (⟨0, 0⟩ :: polyMul (b :: left) right)).length
        = (b :: left).length + right.length
      rw [polyAdd_length, polyScale_length]
      simp only [List.length_cons, polyMul_length_cons b left right wide]
      omega

/-- **The length of a product.**  `m + n − 1`, not `m + n`: the shifted tail
overlaps the scaled head by one place.

Both nonempty hypotheses are real. With `right = []` the recursion still emits
the leading zero, so `polyMul [a] []` has length 1 while `m + n − 1` would say
0; with `left = []` the product is empty and the formula would underflow. -/
theorem polyMul_length :
    ∀ (left right : List Pair), 1 ≤ left.length → 1 ≤ right.length →
      (polyMul left right).length = left.length + right.length - 1
  | [], _, tall, _ => by simp at tall
  | a :: left, right, _, wide => by
      rw [polyMul_length_cons a left right wide]
      simp only [List.length_cons]
      omega

/-! ## Coefficient of a scaling -/

/-- **The coefficient of a scaling is the scaled coefficient**, at every index.

Stated unconditionally rather than as a disjunction with an out-of-range case.
The out-of-range case is not an exception: `coeffAt` returns zero there and
`mulPair scalar ⟨0,0⟩ = ⟨0,0⟩`, so the same equation holds. A disjunction whose
second arm is implied by the first is a weaker statement for no reason. -/
theorem coeffAt_polyScale (scalar : Pair) (poly : List Pair) (index : Nat) :
    coeffAt (polyScale scalar poly) index
      = mulPair scalar (coeffAt poly index) := by
  unfold coeffAt polyScale
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_map]
  by_cases inRange : index < poly.length
  · rw [List.getElem?_eq_getElem inRange]
    rfl
  · rw [List.getElem?_eq_none (by omega)]
    show ⟨0, 0⟩ = mulPair scalar ⟨0, 0⟩
    rw [mulPair_zero_right]

/-! ## Coefficient of a sum

Needs the coefficients to be residues, for the reason `polyEval_polyAdd` did:
`polyAdd`'s base cases return a list unchanged while the statement applies
`addPair`, which reduces. -/

/-- Every coefficient of a polynomial is a residue. -/
def Canonical (poly : List Pair) : Prop :=
  ∀ index, (coeffAt poly index).low < goldilocksP
    ∧ (coeffAt poly index).high < goldilocksP

theorem canonical_nil : Canonical [] := by
  intro index
  rw [coeffAt_nil]
  exact ⟨by decide, by decide⟩

theorem coeffAt_polyAdd :
    ∀ (left right : List Pair), Canonical left → Canonical right →
      ∀ index, coeffAt (polyAdd left right) index
        = addPair (coeffAt left index) (coeffAt right index)
  | [], right, _, canonicalRight, index => by
      show coeffAt right index
        = addPair (coeffAt [] index) (coeffAt right index)
      rw [coeffAt_nil, addPair_zero_left_canonical _
        (canonicalRight index).1 (canonicalRight index).2]
  | a :: left, [], canonicalLeft, _, index => by
      show coeffAt (a :: left) index
        = addPair (coeffAt (a :: left) index) (coeffAt [] index)
      rw [coeffAt_nil, addPair_comm, addPair_zero_left_canonical _
        (canonicalLeft index).1 (canonicalLeft index).2]
  | a :: left, b :: right, canonicalLeft, canonicalRight, 0 => rfl
  | a :: left, b :: right, canonicalLeft, canonicalRight, index + 1 => by
      show coeffAt (polyAdd left right) index
        = addPair (coeffAt left index) (coeffAt right index)
      exact coeffAt_polyAdd left right
        (fun other => canonicalLeft (other + 1))
        (fun other => canonicalRight (other + 1)) index

/-! ## The convolution

Defined by the same recursion `polyMul` uses — head times the whole right side,
plus a shift — so the characterization below is structural rather than range
arithmetic over a `foldl`. -/

def convolution : List Pair → List Pair → Nat → Pair
  | [], _, _ => ⟨0, 0⟩
  | a :: _, right, 0 => mulPair a (coeffAt right 0)
  | a :: left, right, index + 1 =>
      addPair (mulPair a (coeffAt right (index + 1)))
        (convolution left right index)

theorem convolution_nil (right : List Pair) (index : Nat) :
    convolution [] right index = ⟨0, 0⟩ := by
  cases index <;> rfl

/-! ## Canonicity is preserved

Every constructor lands in residues, so `polyMul` of anything is canonical and
the sum lemma applies at each recursion step. -/

theorem canonical_polyScale (scalar : Pair) (poly : List Pair) :
    Canonical (polyScale scalar poly) := by
  intro index
  rw [coeffAt_polyScale]
  exact mulPair_canonical _ _

theorem canonical_polyAdd (left right : List Pair)
    (canonicalLeft : Canonical left) (canonicalRight : Canonical right) :
    Canonical (polyAdd left right) := by
  intro index
  rw [coeffAt_polyAdd left right canonicalLeft canonicalRight]
  exact addPair_canonical _ _

theorem canonical_cons_zero (poly : List Pair) (canonical : Canonical poly) :
    Canonical (⟨0, 0⟩ :: poly) := by
  intro index
  cases index with
  | zero =>
      rw [coeffAt_cons_zero]
      exact ⟨by decide, by decide⟩
  | succ index => exact canonical index

theorem canonical_polyMul :
    ∀ left right : List Pair, Canonical (polyMul left right)
  | [], _ => canonical_nil
  | a :: left, right =>
      canonical_polyAdd _ _ (canonical_polyScale a right)
        (canonical_cons_zero _ (canonical_polyMul left right))

/-! ## The characterization

`polyMul`'s coefficients are the convolution.  The proof is structural because
`convolution` was defined by the same recursion. -/

/-- **The coefficient of a product is the convolution.** -/
theorem coeffAt_polyMul :
    ∀ (left right : List Pair) (index : Nat),
      coeffAt (polyMul left right) index = convolution left right index
  | [], right, index => by
      rw [convolution_nil]
      rfl
  | a :: left, right, 0 => by
      show coeffAt (polyAdd (polyScale a right)
          (⟨0, 0⟩ :: polyMul left right)) 0 = mulPair a (coeffAt right 0)
      rw [coeffAt_polyAdd _ _ (canonical_polyScale a right)
        (canonical_cons_zero _ (canonical_polyMul left right)),
        coeffAt_polyScale]
      show addPair (mulPair a (coeffAt right 0)) ⟨0, 0⟩
        = mulPair a (coeffAt right 0)
      rw [addPair_comm, addPair_zero_left_canonical _
        (mulPair_canonical a (coeffAt right 0)).1
        (mulPair_canonical a (coeffAt right 0)).2]
  | a :: left, right, index + 1 => by
      show coeffAt (polyAdd (polyScale a right)
          (⟨0, 0⟩ :: polyMul left right)) (index + 1)
        = addPair (mulPair a (coeffAt right (index + 1)))
            (convolution left right index)
      rw [coeffAt_polyAdd _ _ (canonical_polyScale a right)
        (canonical_cons_zero _ (canonical_polyMul left right)),
        coeffAt_polyScale, coeffAt_cons_succ,
        coeffAt_polyMul left right index]

/-! ## The convolution is canonical

Needed to absorb the trailing zero the guarded-fold form produces:
`sumOver` bottoms out at `⟨0,0⟩` and adds it, while `convolution` bottoms out
at a bare product. The two agree exactly when the convolution is a residue,
which every branch makes it. -/

theorem convolution_canonical :
    ∀ (left right : List Pair) (index : Nat),
      (convolution left right index).low < goldilocksP
        ∧ (convolution left right index).high < goldilocksP
  | [], right, index => by
      rw [convolution_nil]
      exact ⟨by decide, by decide⟩
  | a :: left, right, 0 => by
      show (mulPair a (coeffAt right 0)).low < goldilocksP ∧ _
      exact mulPair_canonical _ _
  | a :: left, right, index + 1 => by
      show (addPair (mulPair a (coeffAt right (index + 1)))
        (convolution left right index)).low < goldilocksP ∧ _
      exact addPair_canonical _ _

/-- **The trailing zero is absorbed.**  This is the exact shape difference
between `convolution` and the guarded-fold form, isolated. -/
theorem convolution_add_zero (left right : List Pair) (index : Nat) :
    addPair (convolution left right index) ⟨0, 0⟩
      = convolution left right index := by
  rw [addPair_comm, addPair_zero_left_canonical _
    (convolution_canonical left right index).1
    (convolution_canonical left right index).2]

/-! ## The frozen ring as a list

`RingK` is `Fin ringDegree → K`; the polynomial layer is `List Pair`. This is
the conversion, and the point where `ringKCoeff`'s out-of-range default meets
`coeffAt`'s.

The two defaults agree — `ringKCoeff` returns `K.zero` beyond the degree and
`coeffAt` returns `⟨0,0⟩` — so `coeffAt_toList` holds at *every* index rather
than only in range. That is what lets the index-set argument avoid a case
split. -/

open Nightstream.SuperNeo.Concrete in
/-- A frozen ring element as a coefficient list. -/
def toList (value : RingK) : List Pair :=
  (List.finRange ringDegree).map (fun index => KConcreteBridge.ofConcrete (value index))

open Nightstream.SuperNeo.Concrete in
theorem toList_length (value : RingK) : (toList value).length = ringDegree := by
  unfold toList
  rw [List.length_map, List.length_finRange]

open Nightstream.SuperNeo.Concrete in
/-- **The raw product of two ring elements has 107 coefficients**, so its
degree is 106. This is what forces `KQuotient.quotientLength = 53`, and it is
the constructor for the length bound `KQuotient.quotient_identity_high` asks
for. -/
theorem rawProduct_length (left right : RingK) :
    (polyMul (toList left) (toList right)).length = 107 := by
  rw [polyMul_length _ _ (by rw [toList_length]; decide)
      (by rw [toList_length]; decide),
    toList_length, toList_length]
  decide

open Nightstream.SuperNeo.Concrete in
/-- **The conversion respects coefficients at every index**, in range and out.
Out of range both sides are zero, so no case split reaches the caller. -/
theorem coeffAt_toList (value : RingK) (index : Nat) :
    coeffAt (toList value) index
      = KConcreteBridge.ofConcrete (ringKCoeff value index) := by
  unfold coeffAt toList ringKCoeff
  by_cases inRange : index < ringDegree
  · rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_eq_getElem (by rw [List.length_finRange]; exact inRange),
      dif_pos inRange]
    simp [List.getElem_finRange]
  · rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_eq_none (by rw [List.length_finRange]; omega),
      dif_neg inRange]
    rfl

end Nightstream.Implementation.R1CS.Canonical.KPolyCoeff
