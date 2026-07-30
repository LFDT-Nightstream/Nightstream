import Nightstream.Implementation.R1CS.Canonical.KFoldSum

/-!
Contract: the mod-`Φ₈₁` reduction quotient and the identity it satisfies.

Owns: the quotient's definition and length, the survivor structure of the
modulus convolution, the coefficient identity `raw = reduced + q · Φ₈₁`, its
list form, and its evaluation at a point.

Also owns the link to the frozen ring: `toList_ringKMul` proves `reducedList` of
the raw convolution **is** `SuperNeo.Concrete.ringKMul`, so every theorem here is
a statement about the protocol's ring multiplication rather than about
`polyMul`. That needed `KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul_all` — the
agreement at *every* degree — because `ringKMul`'s off-diagonal reads all land
at degrees 54 through 106.

Does **not** own: the row program that emits this identity, its cost, or where
the challenge comes from.

## The closed form

Cycle 277 derived the quotient as two sums; cycle 282 showed they combine,
because the second contributes at indices 27 through 52 — exactly the range the
first leaves empty. So the positive part is uniform and only the subtraction is
conditional:

```text
q_j = c_{j+54} − (c_{j+81} if j ≤ 25 else 0)      for j = 0 … 52
```

Degree 52 is forced: `q · Φ₈₁` must reach 106 to match the raw convolution, and
`Φ₈₁` has degree 54.

## Why the conditional survives

`c_{j+81}` is out of range for `j ≥ 26`, since the raw convolution stops at 106
and `26 + 81 = 107`. `coeffAt` would return zero there anyway, so the guard is
not strictly needed for correctness — it is kept because it makes the two
branches of the boundary explicit rather than relying on an out-of-range
default to do the work silently.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KQuotient

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KPairLaws
open Nightstream.Implementation.R1CS.Canonical.KPolyCoeff

/-- Length of the quotient: one more than its degree bound of 52. -/
def quotientLength : Nat := 53

/-- **The reduction quotient.** -/
def reductionQuotient (raw : List Pair) : List Pair :=
  (List.range quotientLength).map (fun index =>
    subPair (coeffAt raw (index + 54))
      (if index ≤ 25 then coeffAt raw (index + 81) else ⟨0, 0⟩))

theorem reductionQuotient_length (raw : List Pair) :
    (reductionQuotient raw).length = 53 := by
  unfold reductionQuotient quotientLength
  rw [List.length_map, List.length_range]

/-- **The quotient's coefficients, in range.** -/
theorem coeffAt_reductionQuotient
    (raw : List Pair) (index : Nat) (inRange : index < quotientLength) :
    coeffAt (reductionQuotient raw) index
      = subPair (coeffAt raw (index + 54))
          (if index ≤ 25 then coeffAt raw (index + 81) else ⟨0, 0⟩) := by
  unfold coeffAt reductionQuotient
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_getElem (by rw [List.length_range]; exact inRange)]
  simp [List.getElem_range, coeffAt, List.getD_eq_getElem?_getD]

/-- Beyond the degree bound the quotient is zero, so the identity's degree
bookkeeping needs no separate case. -/
theorem coeffAt_reductionQuotient_beyond
    (raw : List Pair) (index : Nat) (beyond : quotientLength ≤ index) :
    coeffAt (reductionQuotient raw) index = ⟨0, 0⟩ := by
  unfold coeffAt reductionQuotient
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_none (by rw [List.length_range]; omega)]
  rfl

/-- Every quotient coefficient is a residue, since `subPair` reduces. -/
theorem reductionQuotient_canonical (raw : List Pair) :
    Canonical (reductionQuotient raw) := by
  intro index
  by_cases inRange : index < quotientLength
  · rw [coeffAt_reductionQuotient raw index inRange]
    exact subPair_canonical _ _
  · rw [coeffAt_reductionQuotient_beyond raw index (by omega)]
    exact ⟨by decide, by decide⟩

/-! ## The modulus is sparse

`Φ₈₁` has three nonzero coefficients out of 55. Convolving against it therefore
picks out three terms and kills the rest, which is what makes the coefficient
identity a three-way sum rather than a 53-term one.

Stated at every index, in range and out: beyond 54 both sides are zero, so no
case split reaches the caller. -/

theorem coeffAt_modulus (index : Nat) :
    coeffAt KRingProjection.modulusCoefficients index
      = if index = 0 ∨ index = KRingProjection.ringMiddleDegree
          ∨ index = KRingProjection.ringDegree then ⟨1, 0⟩ else ⟨0, 0⟩ := by
  unfold coeffAt KRingProjection.modulusCoefficients
  by_cases inRange : index < KRingProjection.ringDegree + 1
  · rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_eq_getElem (by rw [List.length_range]; exact inRange)]
    simp [List.getElem_range]
  · rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_eq_none (by rw [List.length_range]; omega)]
    rw [if_neg (by
      simp only [KRingProjection.ringDegree, KRingProjection.ringMiddleDegree]
        at inRange ⊢
      omega)]
    rfl

/-- The three surviving positions, named. -/
theorem modulus_nonzero_positions :
    coeffAt KRingProjection.modulusCoefficients 0 = ⟨1, 0⟩
      ∧ coeffAt KRingProjection.modulusCoefficients 27 = ⟨1, 0⟩
      ∧ coeffAt KRingProjection.modulusCoefficients 54 = ⟨1, 0⟩ := by
  refine ⟨?_, ?_, ?_⟩ <;> rw [coeffAt_modulus] <;> simp
    [KRingProjection.ringMiddleDegree, KRingProjection.ringDegree]

/-- Everywhere else it vanishes. -/
theorem modulus_zero_elsewhere (index : Nat)
    (notZero : index ≠ 0) (notMiddle : index ≠ 27) (notTop : index ≠ 54) :
    coeffAt KRingProjection.modulusCoefficients index = ⟨0, 0⟩ := by
  rw [coeffAt_modulus, if_neg (by
    simp only [KRingProjection.ringMiddleDegree, KRingProjection.ringDegree]
    omega)]

/-! ## Convolution terms vanish off the survivors

`sumOver_filter`'s hypothesis, for the modulus convolution: at any index whose
modulus offset is not 0, 27 or 54 the product is zero, because the modulus
coefficient is.

Note the Nat-subtraction subtlety. Over `List.range (degree + 1)` every index
satisfies `j ≤ degree`, so `degree − j = 0` holds exactly at `j = degree` and
truncation never fires. Outside that range it would: at `j > degree` the
difference is 0 and the term would be wrongly retained. -/

theorem modulus_term_vanishes
    (quotient : List Pair) (degree index : Nat)
    (notSurvivor : ¬(degree - index = 0 ∨ degree - index = 27
      ∨ degree - index = 54)) :
    mulPair (coeffAt quotient index)
        (coeffAt KRingProjection.modulusCoefficients (degree - index))
      = ⟨0, 0⟩ := by
  rw [modulus_zero_elsewhere (degree - index)
      (fun isZero => notSurvivor (Or.inl isZero))
      (fun isMiddle => notSurvivor (Or.inr (Or.inl isMiddle)))
      (fun isTop => notSurvivor (Or.inr (Or.inr isTop))),
    mulPair_zero_right]

/-- Over the convolution's own index range the truncation never fires, so
`degree − index = 0` really does isolate `index = degree`. -/
theorem offset_zero_iff (degree index : Nat) (inRange : index ≤ degree) :
    degree - index = 0 ↔ index = degree := by
  omega

/-! ## Identifying the survivors

Sampling the filter at degrees 30, 40, 53, 54, 80 and 106 shows one shape, not
three: the survivors are always `degree − 27k` for `k = 0, 1, 2` with
`27k ≤ degree`. What looked like three regimes needing three extraction
techniques is a single list whose *length* is just how many of the three
offsets fit below the degree.

That turns the proof into an induction on the degree. The step is exact:
raising the degree by one shifts every survivor up by one, and a new survivor
appears at index 0 exactly when the degree reaches 27 or 54. -/

/-- The survivor predicate, named once so the list and the sum share it. -/
def survives (degree index : Nat) : Bool :=
  decide (degree - index = 0 ∨ degree - index = 27 ∨ degree - index = 54)

theorem survives_self (degree : Nat) : survives degree degree = true := by
  unfold survives
  simp

/-- **The survivors, ascending.**  `degree − 27k` for each admissible `k`,
written out rather than as a comprehension so that `sumOver` computes on it. -/
def survivorList (degree : Nat) : List Nat :=
  (if 54 ≤ degree then [degree - 54] else [])
    ++ (if 27 ≤ degree then [degree - 27] else [])
    ++ [degree]

/-- Away from the two boundaries the survivors just shift up by one. -/
theorem survivorList_succ (degree : Nat)
    (notMiddle : degree + 1 ≠ 27) (notTop : degree + 1 ≠ 54) :
    (survivorList degree).map Nat.succ = survivorList (degree + 1) := by
  unfold survivorList
  by_cases top : 54 ≤ degree
  · have middle : 27 ≤ degree := by omega
    have shiftTop : degree - 54 + 1 = degree + 1 - 54 := by omega
    have shiftMiddle : degree - 27 + 1 = degree + 1 - 27 := by omega
    rw [if_pos top, if_pos middle, if_pos (show 54 ≤ degree + 1 by omega),
      if_pos (show 27 ≤ degree + 1 by omega)]
    simp [shiftTop, shiftMiddle]
  · by_cases middle : 27 ≤ degree
    · have shiftMiddle : degree - 27 + 1 = degree + 1 - 27 := by omega
      rw [if_neg top, if_pos middle,
        if_neg (show ¬ 54 ≤ degree + 1 by omega),
        if_pos (show 27 ≤ degree + 1 by omega)]
      simp [shiftMiddle]
    · rw [if_neg top, if_neg middle,
        if_neg (show ¬ 54 ≤ degree + 1 by omega),
        if_neg (show ¬ 27 ≤ degree + 1 by omega)]
      simp

/-- Filtering a shifted list is the shifted filter of the shifted predicate.
Needed because `List.range_succ_eq_map` presents the range's tail as a map. -/
theorem filter_map_succ (keep : Nat → Bool) :
    ∀ indices : List Nat,
      (indices.map Nat.succ).filter keep
        = (indices.filter (fun index => keep (Nat.succ index))).map Nat.succ
  | [] => rfl
  | index :: rest => by
      have inner := filter_map_succ keep rest
      cases fires : keep (Nat.succ index) <;> simp [fires, inner]

/-- **The filter is exactly the survivor list**, at every degree — one
statement, not one per regime. -/
theorem filter_survivors :
    ∀ degree : Nat,
      (List.range (degree + 1)).filter (survives degree) = survivorList degree
  | 0 => by decide
  | degree + 1 => by
      have inner := filter_survivors degree
      have shift : ∀ index : Nat,
          survives (degree + 1) (Nat.succ index) = survives degree index := by
        intro index
        unfold survives
        rw [show degree + 1 - Nat.succ index = degree - index from by omega]
      have predicate : (fun index => survives (degree + 1) (Nat.succ index))
          = survives degree := funext shift
      have tail :
          ((List.range (degree + 1)).map Nat.succ).filter (survives (degree + 1))
            = (survivorList degree).map Nat.succ := by
        rw [filter_map_succ, predicate, inner]
      rw [show List.range (degree + 1 + 1)
            = 0 :: (List.range (degree + 1)).map Nat.succ
          from List.range_succ_eq_map]
      by_cases fires : survives (degree + 1) 0 = true
      · rw [List.filter_cons_of_pos fires, tail]
        have boundary : degree + 1 = 27 ∨ degree + 1 = 54 := by
          unfold survives at fires
          simp only [decide_eq_true_eq] at fires
          omega
        rcases boundary with middle | top
        · have reached : degree = 26 := by omega
          subst reached
          decide
        · have reached : degree = 53 := by omega
          subst reached
          decide
      · have notMiddle : degree + 1 ≠ 27 := fun isMiddle =>
          fires (by unfold survives; simp only [decide_eq_true_eq]; omega)
        have notTop : degree + 1 ≠ 54 := fun isTop =>
          fires (by unfold survives; simp only [decide_eq_true_eq]; omega)
        rw [List.filter_cons_of_neg fires, tail,
          survivorList_succ degree notMiddle notTop]

/-! ## The convolution collapses

With the survivors identified, `sumOver_filter` discards the other 104 terms in
one step: each vanishes because the modulus coefficient at its offset does.

The result is stated over `survivorList`, whose `if`s are the only place the
degree's regime is visible. Everything above this line is regime-free. -/

open Nightstream.Implementation.R1CS.Canonical.KFoldSum in
/-- **The modulus convolution is the sum over the survivors.** -/
theorem convolution_modulus_eq_survivor_sum
    (quotient : List Pair) (degree : Nat) :
    convolution quotient KRingProjection.modulusCoefficients degree
      = sumOver (fun _ => true)
          (fun index => mulPair (coeffAt quotient index)
            (coeffAt KRingProjection.modulusCoefficients (degree - index)))
          (survivorList degree) := by
  rw [convolution_eq_sumOver, ← filter_survivors degree]
  refine (sumOver_filter _ (survives degree) (List.range (degree + 1))
    (fun index _ notKept => ?_)).symm
  refine modulus_term_vanishes quotient degree index ?_
  simp only [survives, decide_eq_false_iff_not] at notKept
  exact notKept

/-! ## The modulus factor is one

Every survivor sits at offset 0, 27 or 54, and `Φ₈₁` is one at each. So the
modulus contributes nothing to the surviving terms but its support: the
convolution against it is a plain sum of quotient coefficients.

That is the shape the coefficient identity needs, because
`ringKMul`'s reduction adds and subtracts *raw* coefficients with no factors. -/

/-- Membership in the survivor list is exactly the survivor condition.  Read
off `filter_survivors` rather than by unfolding the regimes, which is what makes
that theorem load-bearing rather than decorative. -/
theorem mem_survivorList (degree index : Nat)
    (member : index ∈ survivorList degree) :
    degree - index = 0 ∨ degree - index = 27 ∨ degree - index = 54 := by
  rw [← filter_survivors degree] at member
  have kept := (List.mem_filter.1 member).2
  simpa [survives] using kept

/-- **At a survivor the modulus factor drops out.** -/
theorem term_at_survivor (quotient : List Pair)
    (canonical : Canonical quotient) (degree index : Nat)
    (member : index ∈ survivorList degree) :
    mulPair (coeffAt quotient index)
        (coeffAt KRingProjection.modulusCoefficients (degree - index))
      = coeffAt quotient index := by
  rw [coeffAt_modulus, if_pos (by
      simpa [KRingProjection.ringMiddleDegree, KRingProjection.ringDegree]
        using mem_survivorList degree index member),
    mulPair_comm, mulPair_one_left _ (canonical index).1 (canonical index).2]

open Nightstream.Implementation.R1CS.Canonical.KFoldSum in
/-- **The modulus convolution is the sum of the quotient's coefficients at the
survivors.**  No products remain. -/
theorem convolution_modulus_eq_quotient_sum
    (quotient : List Pair) (canonical : Canonical quotient) (degree : Nat) :
    convolution quotient KRingProjection.modulusCoefficients degree
      = sumOver (fun _ => true) (coeffAt quotient) (survivorList degree) := by
  rw [convolution_modulus_eq_survivor_sum]
  exact sumOver_congr_term _ _ _ _
    (fun index member => term_at_survivor quotient canonical degree index member)

/-! ## The sum, per regime

`survivorList`'s `if`s are the last place the degree's regime is visible, so
this is where it is resolved — into one, two or three terms. The trailing
`⟨0,0⟩` that `sumOver` bottoms out at is absorbed by canonicity at each arity,
which is the fifth place in the tower that condition is load-bearing. -/

open Nightstream.Implementation.R1CS.Canonical.KFoldSum in
/-- Below 27: one term. -/
theorem quotientSum_low (quotient : List Pair) (canonical : Canonical quotient)
    (degree : Nat) (below : degree < 27) :
    sumOver (fun _ => true) (coeffAt quotient) (survivorList degree)
      = coeffAt quotient degree := by
  unfold survivorList
  rw [if_neg (by omega : ¬ 54 ≤ degree), if_neg (by omega : ¬ 27 ≤ degree)]
  show addPair (coeffAt quotient degree) ⟨0, 0⟩ = coeffAt quotient degree
  rw [addPair_comm, addPair_zero_left_canonical _
    (canonical degree).1 (canonical degree).2]

open Nightstream.Implementation.R1CS.Canonical.KFoldSum in
/-- Between 27 and 53: two terms. -/
theorem quotientSum_middle (quotient : List Pair)
    (canonical : Canonical quotient) (degree : Nat)
    (atLeast : 27 ≤ degree) (below : degree < 54) :
    sumOver (fun _ => true) (coeffAt quotient) (survivorList degree)
      = addPair (coeffAt quotient (degree - 27)) (coeffAt quotient degree) := by
  unfold survivorList
  rw [if_neg (by omega : ¬ 54 ≤ degree), if_pos atLeast]
  show addPair (coeffAt quotient (degree - 27))
      (addPair (coeffAt quotient degree) ⟨0, 0⟩) = _
  rw [addPair_comm (coeffAt quotient degree), addPair_zero_left_canonical _
    (canonical degree).1 (canonical degree).2]

open Nightstream.Implementation.R1CS.Canonical.KFoldSum in
/-- From 54 up: three terms. -/
theorem quotientSum_high (quotient : List Pair)
    (canonical : Canonical quotient) (degree : Nat) (atLeast : 54 ≤ degree) :
    sumOver (fun _ => true) (coeffAt quotient) (survivorList degree)
      = addPair (coeffAt quotient (degree - 54))
          (addPair (coeffAt quotient (degree - 27)) (coeffAt quotient degree)) := by
  unfold survivorList
  rw [if_pos atLeast, if_pos (by omega : 27 ≤ degree)]
  show addPair (coeffAt quotient (degree - 54))
      (addPair (coeffAt quotient (degree - 27))
        (addPair (coeffAt quotient degree) ⟨0, 0⟩)) = _
  rw [addPair_comm (coeffAt quotient degree), addPair_zero_left_canonical _
    (canonical degree).1 (canonical degree).2]

/-! ## Above the reduced range

`ringKMul` produces 54 coefficients, so at degree 54 and above the reduced
polynomial contributes nothing and the quotient must reproduce the raw
coefficient by itself. It does.

The *statement* has no split by degree. The splits live in the two shift lemmas
below, and each collapses for the same reason: past index 106 the raw
convolution is empty, so "the quotient ran out" and "the raw coefficient is
zero" are the same fact. That is why the length bound is a hypothesis rather
than a degree hypothesis — it is what makes the boundary cases agree. -/

/-- Raw coefficients vanish past the convolution's reach. -/
theorem raw_vanishes (raw : List Pair) (bounded : raw.length ≤ 107)
    (index : Nat) (beyond : 107 ≤ index) : coeffAt raw index = ⟨0, 0⟩ :=
  coeffAt_beyond_length raw index (by omega)

/-- **The 27-shifted survivor is a raw coefficient.**  In range the
subtrahend's guard is always false at these degrees; out of range both sides
vanish. -/
theorem quotientCoeff_shift27 (raw : List Pair) (canonical : Canonical raw)
    (bounded : raw.length ≤ 107) (degree : Nat) (atLeast : 54 ≤ degree) :
    coeffAt (reductionQuotient raw) (degree - 27)
      = coeffAt raw (degree + 27) := by
  by_cases inRange : degree - 27 < quotientLength
  · have small : degree - 27 < 53 := inRange
    rw [coeffAt_reductionQuotient raw _ inRange, if_neg (by omega),
      show degree - 27 + 54 = degree + 27 from by omega,
      subPair_zero_right _ (canonical (degree + 27)).1
        (canonical (degree + 27)).2]
  · have large : ¬ (degree - 27 < 53) := inRange
    rw [coeffAt_reductionQuotient_beyond raw _ (Nat.not_lt.1 inRange),
      raw_vanishes raw bounded (degree + 27) (by omega)]

/-- **The 54-shifted survivor is the raw difference.**  Three boundary cases,
all collapsing to one form once the length bound is available. -/
theorem quotientCoeff_shift54 (raw : List Pair)
    (bounded : raw.length ≤ 107) (degree : Nat) (atLeast : 54 ≤ degree) :
    coeffAt (reductionQuotient raw) (degree - 54)
      = subPair (coeffAt raw degree) (coeffAt raw (degree + 27)) := by
  by_cases inRange : degree - 54 < quotientLength
  · have small : degree - 54 < 53 := inRange
    rw [coeffAt_reductionQuotient raw _ inRange,
      show degree - 54 + 54 = degree from by omega]
    by_cases early : degree - 54 ≤ 25
    · rw [if_pos early, show degree - 54 + 81 = degree + 27 from by omega]
    · rw [if_neg early, raw_vanishes raw bounded (degree + 27) (by omega)]
  · have large : ¬ (degree - 54 < 53) := inRange
    rw [coeffAt_reductionQuotient_beyond raw _ (Nat.not_lt.1 inRange),
      raw_vanishes raw bounded degree (by omega),
      raw_vanishes raw bounded (degree + 27) (by omega),
      subPair_zero_right _ (by decide) (by decide)]

open Nightstream.Implementation.R1CS.Canonical.KFoldSum in
/-- **Above the reduced range the quotient reproduces the raw coefficient
exactly.**  The upper half of `raw = reduced + q · Φ₈₁`, where the reduced part
is zero.

The three survivors telescope: the top one is out of the quotient's range, the
27-shifted one is `raw[d+27]`, and the 54-shifted one is
`raw[d] − raw[d+27]`. `addPair_subPair` collapses them. -/
theorem quotient_identity_high (raw : List Pair) (canonical : Canonical raw)
    (bounded : raw.length ≤ 107) (degree : Nat) (atLeast : 54 ≤ degree) :
    convolution (reductionQuotient raw) KRingProjection.modulusCoefficients
        degree
      = coeffAt raw degree := by
  rw [convolution_modulus_eq_quotient_sum _ (reductionQuotient_canonical raw),
    quotientSum_high _ (reductionQuotient_canonical raw) degree atLeast,
    coeffAt_reductionQuotient_beyond raw degree
      (Nat.le_trans (by decide) atLeast),
    quotientCoeff_shift27 raw canonical bounded degree atLeast,
    quotientCoeff_shift54 raw bounded degree atLeast,
    addPair_comm (coeffAt raw (degree + 27)) ⟨0, 0⟩,
    addPair_zero_left_canonical _ (canonical (degree + 27)).1
      (canonical (degree + 27)).2,
    addPair_subPair _ _ (canonical degree).1 (canonical degree).2]

/-! ## Inside the reduced range

Below degree 54 the reduced polynomial contributes, so the identity is the full
`raw = reduced + q · Φ₈₁` rather than the quotient alone.

`reducedCoeff` is written in `SuperNeo.Concrete.ringKMul`'s exact shape,
including its `index + 81 ≤ 106` guard rather than the equivalent `index ≤ 25`,
so that the correspondence with the frozen definition is visible rather than
argued. The cost is one `by_cases` where the two guards have to be aligned. -/

/-- The frozen reduction by `X⁵⁴ = −X²⁷ − 1`, at the list level.

Derivation of the shape: for `j` in 54…80, `X^j ≡ −X^{j−27} − X^{j−54}`; for `j`
in 81…106 a second pass collapses to `X^j ≡ X^{j−81}`. Collecting contributions
at index `i` gives exactly the three terms below. -/
def reducedCoeff (raw : List Pair) (index : Nat) : Pair :=
  addPair
    (subPair (coeffAt raw index)
      (if index < 27 then coeffAt raw (index + 54)
       else coeffAt raw (index + 27)))
    (if index + 81 ≤ 106 then coeffAt raw (index + 81) else ⟨0, 0⟩)

/-- In the middle regime the top survivor is a raw coefficient.  Uniform across
degree 53, where the quotient has run out and so has the raw list. -/
theorem quotientCoeff_mid_top (raw : List Pair) (canonical : Canonical raw)
    (bounded : raw.length ≤ 107) (degree : Nat)
    (atLeast : 27 ≤ degree) (below : degree < 54) :
    coeffAt (reductionQuotient raw) degree = coeffAt raw (degree + 54) := by
  by_cases inRange : degree < quotientLength
  · have small : degree < 53 := inRange
    rw [coeffAt_reductionQuotient raw _ inRange, if_neg (by omega),
      subPair_zero_right _ (canonical (degree + 54)).1
        (canonical (degree + 54)).2]
  · have large : ¬ (degree < 53) := inRange
    rw [coeffAt_reductionQuotient_beyond raw _ (Nat.not_lt.1 inRange),
      raw_vanishes raw bounded (degree + 54) (by omega)]

/-- In the middle regime the 27-shifted survivor is a raw difference. -/
theorem quotientCoeff_mid_shift (raw : List Pair) (bounded : raw.length ≤ 107)
    (degree : Nat) (atLeast : 27 ≤ degree) (below : degree < 54) :
    coeffAt (reductionQuotient raw) (degree - 27)
      = subPair (coeffAt raw (degree + 27)) (coeffAt raw (degree + 54)) := by
  have inRange : degree - 27 < quotientLength := by
    show degree - 27 < 53
    omega
  rw [coeffAt_reductionQuotient raw _ inRange,
    show degree - 27 + 54 = degree + 27 from by omega]
  by_cases early : degree - 27 ≤ 25
  · rw [if_pos early, show degree - 27 + 81 = degree + 54 from by omega]
  · rw [if_neg early, raw_vanishes raw bounded (degree + 54) (by omega)]

open Nightstream.Implementation.R1CS.Canonical.KFoldSum in
/-- **Degrees 27 to 53.**  Two survivors; the 27-shifted one cancels the top
one, leaving `raw[d+27]`, which is exactly what the reduction subtracted. -/
theorem quotient_identity_middle (raw : List Pair) (canonical : Canonical raw)
    (bounded : raw.length ≤ 107) (degree : Nat)
    (atLeast : 27 ≤ degree) (below : degree < 54) :
    addPair (reducedCoeff raw degree)
        (convolution (reductionQuotient raw)
          KRingProjection.modulusCoefficients degree)
      = coeffAt raw degree := by
  rw [convolution_modulus_eq_quotient_sum _ (reductionQuotient_canonical raw),
    quotientSum_middle _ (reductionQuotient_canonical raw) degree atLeast below,
    quotientCoeff_mid_top raw canonical bounded degree atLeast below,
    quotientCoeff_mid_shift raw bounded degree atLeast below]
  unfold reducedCoeff
  rw [if_neg (by omega : ¬ degree < 27),
    if_neg (by omega : ¬ degree + 81 ≤ 106),
    addPair_comm (subPair (coeffAt raw degree) (coeffAt raw (degree + 27)))
      ⟨0, 0⟩,
    addPair_zero_left_canonical _ (subPair_canonical _ _).1
      (subPair_canonical _ _).2,
    addPair_subPair (coeffAt raw (degree + 27)) (coeffAt raw (degree + 54))
      (canonical (degree + 27)).1 (canonical (degree + 27)).2,
    addPair_subPair (coeffAt raw degree) (coeffAt raw (degree + 27))
      (canonical degree).1 (canonical degree).2]

open Nightstream.Implementation.R1CS.Canonical.KFoldSum in
/-- **Degrees 0 to 26.**  One survivor, but the reduction's second-pass term is
live here, so the cancellation is two-stage: the quotient's own subtrahend
cancels it first, then the result cancels the reduction's subtraction. -/
theorem quotient_identity_low (raw : List Pair) (canonical : Canonical raw)
    (degree : Nat) (below : degree < 27) :
    addPair (reducedCoeff raw degree)
        (convolution (reductionQuotient raw)
          KRingProjection.modulusCoefficients degree)
      = coeffAt raw degree := by
  have inRange : degree < quotientLength := by
    show degree < 53
    omega
  rw [convolution_modulus_eq_quotient_sum _ (reductionQuotient_canonical raw),
    quotientSum_low _ (reductionQuotient_canonical raw) degree below,
    coeffAt_reductionQuotient raw degree inRange]
  unfold reducedCoeff
  rw [if_pos below]
  by_cases early : degree ≤ 25
  · rw [if_pos early, if_pos (by omega : degree + 81 ≤ 106), addPair_assoc,
      addPair_comm (coeffAt raw (degree + 81)),
      addPair_subPair (coeffAt raw (degree + 54)) (coeffAt raw (degree + 81))
        (canonical (degree + 54)).1 (canonical (degree + 54)).2,
      addPair_subPair (coeffAt raw degree) (coeffAt raw (degree + 54))
        (canonical degree).1 (canonical degree).2]
  · rw [if_neg early, if_neg (by omega : ¬ degree + 81 ≤ 106),
      subPair_zero_right (coeffAt raw (degree + 54))
        (canonical (degree + 54)).1 (canonical (degree + 54)).2,
      addPair_comm (subPair (coeffAt raw degree) (coeffAt raw (degree + 54)))
        ⟨0, 0⟩,
      addPair_zero_left_canonical _ (subPair_canonical _ _).1
        (subPair_canonical _ _).2,
      addPair_subPair (coeffAt raw degree) (coeffAt raw (degree + 54))
        (canonical degree).1 (canonical degree).2]

/-! ## The coefficient identity

The three ranges assembled into one statement. `reducedList` carries
`ringKMul`'s 54 coefficients as a list so that both sides are `coeffAt` of a
list and the degree disappears from the statement.

Above 53 the reduced list is empty and `quotient_identity_high` carries the
whole coefficient; below, the reduction contributes and the other two halves
apply. No degree appears in the conclusion. -/

/-- The reduced polynomial: `ringKMul`'s 54 coefficients. -/
def reducedList (raw : List Pair) : List Pair :=
  (List.range 54).map (reducedCoeff raw)

theorem reducedList_length (raw : List Pair) :
    (reducedList raw).length = 54 := by
  unfold reducedList
  rw [List.length_map, List.length_range]

theorem coeffAt_reducedList (raw : List Pair) (index : Nat)
    (inRange : index < 54) :
    coeffAt (reducedList raw) index = reducedCoeff raw index := by
  unfold coeffAt reducedList
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_getElem (by rw [List.length_range]; exact inRange)]
  simp [List.getElem_range]

theorem coeffAt_reducedList_beyond (raw : List Pair) (index : Nat)
    (beyond : 54 ≤ index) : coeffAt (reducedList raw) index = ⟨0, 0⟩ := by
  unfold coeffAt reducedList
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_none (by rw [List.length_range]; omega)]
  rfl

/-- **The coefficient identity: `raw = reduced + q · Φ₈₁`, at every index.**

This is what makes `reductionQuotient` *the* quotient rather than a plausible
candidate, and it is the statement `KPolyEval.polyEval_quotientForm` needs in
order to carry the frozen check to the challenge — with no condition on the
challenge. -/
theorem coefficient_identity (raw : List Pair) (canonical : Canonical raw)
    (bounded : raw.length ≤ 107) (degree : Nat) :
    coeffAt raw degree
      = addPair (coeffAt (reducedList raw) degree)
          (coeffAt (KPolyHom.polyMul (reductionQuotient raw)
            KRingProjection.modulusCoefficients) degree) := by
  rw [coeffAt_polyMul]
  by_cases high : 54 ≤ degree
  · rw [coeffAt_reducedList_beyond raw degree high,
      quotient_identity_high raw canonical bounded degree high,
      addPair_zero_left_canonical _ (canonical degree).1 (canonical degree).2]
  · rw [coeffAt_reducedList raw degree (by omega)]
    by_cases middle : 27 ≤ degree
    · exact (quotient_identity_middle raw canonical bounded degree middle
        (by omega)).symm
    · exact (quotient_identity_low raw canonical degree (by omega)).symm

/-! ## From coefficients to polynomials, and to the challenge

The coefficient identity holds at every index; with both lengths pinned at 107
that upgrades to a list equality, and from there `polyEval_quotientForm`
delivers the frozen check's shape at any point.

Note where the length arithmetic lands: `polyMul` of a 53-entry quotient and a
55-entry modulus is `53 + 55 − 1 = 107`, the same 107 the raw convolution has.
That coincidence is not luck — it is what fixed `quotientLength = 53` in the
first place. -/

theorem reducedList_canonical (raw : List Pair) : Canonical (reducedList raw) := by
  intro index
  by_cases inRange : index < 54
  · rw [coeffAt_reducedList raw index inRange]
    exact addPair_canonical _ _
  · rw [coeffAt_reducedList_beyond raw index (by omega)]
    exact ⟨by decide, by decide⟩

/-- The quotient multiple has exactly the raw convolution's length. -/
theorem quotientMultiple_length (raw : List Pair) :
    (KPolyHom.polyMul (reductionQuotient raw)
      KRingProjection.modulusCoefficients).length = 107 := by
  rw [polyMul_length _ _
      (by rw [reductionQuotient_length]; decide)
      (by rw [KRingProjection.modulusCoefficients_length]; decide),
    reductionQuotient_length, KRingProjection.modulusCoefficients_length]

/-- **The list identity: `raw = reduced + q · Φ₈₁` as polynomials.**  This is
what makes `reductionQuotient` *the* quotient rather than a candidate. -/
theorem raw_eq_reduced_add_quotient_multiple (raw : List Pair)
    (canonical : Canonical raw) (sized : raw.length = 107) :
    raw = KPolyHom.polyAdd (reducedList raw)
      (KPolyHom.polyMul (reductionQuotient raw)
        KRingProjection.modulusCoefficients) := by
  refine list_ext_coeffAt _ _ ?_ (fun index => ?_)
  · rw [polyAdd_length, reducedList_length, quotientMultiple_length, sized]
    omega
  · rw [coeffAt_polyAdd _ _ (reducedList_canonical raw)
      (canonical_polyMul _ _)]
    exact coefficient_identity raw canonical (by omega) index

/-- **The frozen check's shape, at any point.**  Evaluating the raw convolution
gives the reduced value plus the quotient's contribution.

No hypothesis on the point. This is the statement the withdrawn root route was
trying to reach by assuming `Φ₈₁(point) = 0`, which is impossible over `K`; the
quotient term is carried instead, exactly as production does. -/
theorem polyEval_raw_eq_quotientForm (point : Pair) (raw : List Pair)
    (canonical : Canonical raw) (sized : raw.length = 107) :
    KPolyHom.polyEval point raw
      = addPair (KPolyHom.polyEval point (reducedList raw))
          (mulPair (KPolyHom.polyEval point (reductionQuotient raw))
            (KPolyHom.polyEval point KRingProjection.modulusCoefficients)) := by
  rw [← KPolyEval.polyEval_quotientForm point (reducedList raw)
      KRingProjection.modulusCoefficients (reductionQuotient raw),
    ← raw_eq_reduced_add_quotient_multiple raw canonical sized]

/-! ## The link to the frozen ring multiplication

Everything above is stated over `reducedCoeff`, which was written in
`ringKMul`'s shape by hand. This closes the gap: the two are the same function,
so every theorem in this module is a statement about the protocol's ring
multiplication.

It needs the agreement at **every** degree, not just below 54:
`ringKMul`'s three reads are at `index`, at `index + 27` or `index + 54`, and
at `index + 81`, and only the first is ever below the ring degree. -/

open Nightstream.SuperNeo.Concrete in
/-- **`reducedList` of the raw convolution is the frozen `ringKMul`.** -/
theorem toList_ringKMul (left right : RingK) :
    toList (ringKMul left right)
      = reducedList (KPolyHom.polyMul (toList left) (toList right)) := by
  refine list_ext_coeffAt _ _
    (by rw [toList_length, reducedList_length]; rfl) ?_
  intro index
  rw [coeffAt_toList]
  by_cases inRange : index < 54
  · have expand : ringKCoeff (ringKMul left right) index
        = K.add
            (K.sub (rawMulCoeffK left right index)
              (if index < 27 then rawMulCoeffK left right (index + 54)
               else rawMulCoeffK left right (index + 27)))
            (if index + 81 ≤ 106 then rawMulCoeffK left right (index + 81)
             else K.zero) := by
      unfold ringKCoeff
      rw [dif_pos (show index < ringDegree from inRange)]
      rfl
    rw [expand, coeffAt_reducedList _ _ inRange,
      KConcreteBridge.ofConcrete_add, KConcreteBridge.ofConcrete_sub]
    unfold reducedCoeff
    by_cases low : index < 27
    · by_cases early : index + 81 ≤ 106 <;>
        rw [if_pos low, if_pos low] <;>
        simp only [early, if_true, if_false, reduceIte,
          KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul_all,
          KConcreteBridge.ofConcrete_zero]
    · rw [if_neg low, if_neg low,
        if_neg (show ¬ index + 81 ≤ 106 by omega),
        if_neg (show ¬ index + 81 ≤ 106 by omega)]
      simp only [KFoldSum.rawMulCoeffK_eq_coeffAt_polyMul_all,
        KConcreteBridge.ofConcrete_zero]
  · rw [coeffAt_reducedList_beyond _ _ (by omega)]
    unfold ringKCoeff
    rw [dif_neg (show ¬ index < ringDegree from inRange)]
    rfl

open Nightstream.SuperNeo.Concrete in
/-- **The frozen check's shape, for the protocol's own ring multiplication.**

Evaluating the raw convolution of two `RingK` elements at any point gives the
frozen `ringKMul`'s evaluation plus the quotient's contribution. Every symbol
on the right is either a frozen definition or the quotient this module
constructs, and there is **no hypothesis on the point**.

This is the algebraic content of what `ProjectionProgram.ProjectionTrace.identity`
tests, derived in Lean rather than read from an artifact. -/
theorem polyEval_ringKMul_quotientForm
    (point : Pair) (left right : RingK) :
    KPolyHom.polyEval point (KPolyHom.polyMul (toList left) (toList right))
      = addPair (KPolyHom.polyEval point (toList (ringKMul left right)))
          (mulPair
            (KPolyHom.polyEval point
              (reductionQuotient (KPolyHom.polyMul (toList left) (toList right))))
            (KPolyHom.polyEval point KRingProjection.modulusCoefficients)) := by
  rw [toList_ringKMul,
    polyEval_raw_eq_quotientForm point _ (canonical_polyMul _ _)
      (rawProduct_length left right)]

end Nightstream.Implementation.R1CS.Canonical.KQuotient
