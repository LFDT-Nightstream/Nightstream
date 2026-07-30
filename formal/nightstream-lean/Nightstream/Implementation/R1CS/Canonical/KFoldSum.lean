import Nightstream.Implementation.R1CS.Canonical.KPolyCoeff

/-!
Contract: guarded `foldl` accumulation over `Pair`.

Owns: the accumulator lemma that turns a `foldl` starting from an arbitrary
value into that value plus a `foldl` from zero, and the recursive sum it
equals.

Does **not** own, and does not prove: agreement of `convolution` with
`rawMulCoeffK`. That needs this plus an index-range argument, and it is not
written here.

## Why this lemma is the crux

`rawMulCoeffK` accumulates left-to-right over `List.range ringDegree`, guarding
each term. `convolution` nests to the right, head first. The two are the same
sum in different association, and every attempt to relate them directly runs
into the accumulator: `foldl` threads a partial result that the recursion does
not have.

Peeling that accumulator out is what makes the shapes comparable, and it needs
`addPair`'s associativity — not just its commutativity — because the
accumulator sits on the left of every step.

## Canonicity, again

The zero-start form only equals the accumulated form up to `addPair`'s
reduction, so the caller's `init` must be a residue. This is the fourth place in
the tower where that condition is load-bearing rather than bookkeeping.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KFoldSum

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KPairLaws

/-- One guarded accumulation step. -/
def step (guard : Nat → Bool) (term : Nat → Pair) (acc : Pair) (index : Nat) :
    Pair :=
  if guard index then addPair acc (term index) else acc

/-- The same sum, nested to the right and started from zero. -/
def sumOver (guard : Nat → Bool) (term : Nat → Pair) : List Nat → Pair
  | [] => ⟨0, 0⟩
  | index :: rest =>
      if guard index then addPair (term index) (sumOver guard term rest)
      else sumOver guard term rest

theorem sumOver_canonical (guard : Nat → Bool) (term : Nat → Pair) :
    ∀ indices : List Nat,
      (sumOver guard term indices).low < goldilocksP
        ∧ (sumOver guard term indices).high < goldilocksP
  | [] => by
      show ((⟨0, 0⟩ : Pair)).low < goldilocksP ∧ ((⟨0, 0⟩ : Pair)).high < goldilocksP
      exact ⟨by decide, by decide⟩
  | index :: rest => by
      rw [show sumOver guard term (index :: rest)
        = (if guard index then addPair (term index) (sumOver guard term rest)
           else sumOver guard term rest) from rfl]
      by_cases fires : guard index
      · rw [if_pos fires]
        exact addPair_canonical _ _
      · rw [if_neg fires]
        exact sumOver_canonical guard term rest

/-- **The accumulator peels out.**  A guarded `foldl` from `init` is `init` plus
the same fold from zero — which is what makes it comparable to a right-nested
recursion. -/
theorem foldl_step_accumulator (guard : Nat → Bool) (term : Nat → Pair) :
    ∀ (indices : List Nat) (init : Pair),
      init.low < goldilocksP → init.high < goldilocksP →
      indices.foldl (step guard term) init
        = addPair init (sumOver guard term indices)
  | [], init, lowLt, highLt => by
      show init = addPair init ⟨0, 0⟩
      rw [addPair_comm, addPair_zero_left_canonical init lowLt highLt]
  | index :: rest, init, lowLt, highLt => by
      show (rest.foldl (step guard term) (step guard term init index))
        = addPair init (if guard index then
            addPair (term index) (sumOver guard term rest)
          else sumOver guard term rest)
      by_cases fires : guard index
      · rw [if_pos fires]
        show rest.foldl (step guard term)
            (if guard index then addPair init (term index) else init) = _
        rw [if_pos fires,
          foldl_step_accumulator guard term rest (addPair init (term index))
            (addPair_canonical _ _).1 (addPair_canonical _ _).2,
          addPair_assoc]
      · rw [if_neg fires]
        show rest.foldl (step guard term)
            (if guard index then addPair init (term index) else init) = _
        rw [if_neg fires,
          foldl_step_accumulator guard term rest init lowLt highLt]

/-- **The zero-start form.**  The shape `rawMulCoeffK` presents, rewritten as a
right-nested sum. -/
theorem foldl_step_from_zero (guard : Nat → Bool) (term : Nat → Pair)
    (indices : List Nat) :
    indices.foldl (step guard term) ⟨0, 0⟩ = sumOver guard term indices := by
  rw [foldl_step_accumulator guard term indices ⟨0, 0⟩ (by decide) (by decide),
    addPair_zero_left_canonical _ (sumOver_canonical guard term indices).1
      (sumOver_canonical guard term indices).2]

/-! ## Splitting the range

`rawMulCoeffK` folds over all of `List.range ringDegree`, but its guard fires
only below the degree. These two lemmas let the inert tail be discarded, which
is what makes the fold's length independent of `ringDegree` and comparable to a
convolution that stops at the degree. -/

/-- Sums split over concatenation. -/
theorem sumOver_append (guard : Nat → Bool) (term : Nat → Pair) :
    ∀ left right : List Nat,
      sumOver guard term (left ++ right)
        = addPair (sumOver guard term left) (sumOver guard term right)
  | [], right => by
      show sumOver guard term right
        = addPair (sumOver guard term []) (sumOver guard term right)
      rw [show sumOver guard term [] = (⟨0, 0⟩ : Pair) from rfl,
        addPair_zero_left_canonical _ (sumOver_canonical guard term right).1
          (sumOver_canonical guard term right).2]
  | index :: left, right => by
      rw [List.cons_append,
        show sumOver guard term (index :: (left ++ right))
          = (if guard index then
              addPair (term index) (sumOver guard term (left ++ right))
            else sumOver guard term (left ++ right)) from rfl,
        show sumOver guard term (index :: left)
          = (if guard index then
              addPair (term index) (sumOver guard term left)
            else sumOver guard term left) from rfl,
        sumOver_append guard term left right]
      by_cases fires : guard index
      · rw [if_pos fires, if_pos fires, addPair_assoc]
      · rw [if_neg fires, if_neg fires]

/-- **An inert stretch contributes nothing.**  Where the guard never fires the
sum is zero, so the tail of the range can be dropped. -/
theorem sumOver_of_no_guard (guard : Nat → Bool) (term : Nat → Pair) :
    ∀ indices : List Nat, (∀ index ∈ indices, guard index = false) →
      sumOver guard term indices = ⟨0, 0⟩
  | [], _ => rfl
  | index :: rest, silent => by
      rw [show sumOver guard term (index :: rest)
        = (if guard index then addPair (term index) (sumOver guard term rest)
           else sumOver guard term rest) from rfl,
        if_neg (by simp [silent index (by simp)])]
      exact sumOver_of_no_guard guard term rest
        (fun other member => silent other (List.mem_cons_of_mem _ member))

/-- **Sums commute with re-indexing.**  Mapping the index list is the same as
composing the guard and term with the map. This is what lets
`List.range_succ_eq_map`'s shifted tail be matched against a recursion that
drops a list head. -/
theorem sumOver_map (guard : Nat → Bool) (term : Nat → Pair) (shift : Nat → Nat) :
    ∀ indices : List Nat,
      sumOver guard term (indices.map shift)
        = sumOver (fun index => guard (shift index))
            (fun index => term (shift index)) indices
  | [] => rfl
  | index :: rest => by
      rw [List.map_cons,
        show sumOver guard term (shift index :: rest.map shift)
          = (if guard (shift index) then
              addPair (term (shift index)) (sumOver guard term (rest.map shift))
            else sumOver guard term (rest.map shift)) from rfl,
        show sumOver (fun i => guard (shift i)) (fun i => term (shift i))
            (index :: rest)
          = (if guard (shift index) then
              addPair (term (shift index))
                (sumOver (fun i => guard (shift i)) (fun i => term (shift i)) rest)
            else sumOver (fun i => guard (shift i)) (fun i => term (shift i)) rest)
            from rfl,
        sumOver_map guard term shift rest]

/-- **All-zero terms sum to zero**, whatever the guard does.  The empty-left
base case of the re-association needs this: every term is a product with a
missing coefficient, hence zero, but the guard still fires. -/
theorem sumOver_of_zero_terms (guard : Nat → Bool) (term : Nat → Pair) :
    ∀ indices : List Nat, (∀ index ∈ indices, term index = ⟨0, 0⟩) →
      sumOver guard term indices = ⟨0, 0⟩
  | [], _ => rfl
  | index :: rest, vanish => by
      rw [show sumOver guard term (index :: rest)
        = (if guard index then addPair (term index) (sumOver guard term rest)
           else sumOver guard term rest) from rfl,
        sumOver_of_zero_terms guard term rest
          (fun other member => vanish other (List.mem_cons_of_mem _ member))]
      by_cases fires : guard index
      · rw [if_pos fires, vanish index (by simp)]
        show addPair ⟨0, 0⟩ ⟨0, 0⟩ = ⟨0, 0⟩
        rw [addPair_zero_left_canonical _ (by decide) (by decide)]
      · rw [if_neg fires]

/-- Guards agreeing on the index list give the same sum. -/
theorem sumOver_congr_guard (term : Nat → Pair) (first second : Nat → Bool) :
    ∀ indices : List Nat, (∀ index ∈ indices, first index = second index) →
      sumOver first term indices = sumOver second term indices
  | [], _ => rfl
  | index :: rest, agree => by
      rw [show sumOver first term (index :: rest)
        = (if first index then addPair (term index) (sumOver first term rest)
           else sumOver first term rest) from rfl,
        show sumOver second term (index :: rest)
          = (if second index then
              addPair (term index) (sumOver second term rest)
            else sumOver second term rest) from rfl,
        agree index (by simp),
        sumOver_congr_guard term first second rest
          (fun other member => agree other (List.mem_cons_of_mem _ member))]

/-- Terms agreeing on the index list give the same sum.  The companion of
`sumOver_congr_guard`, for replacing a term function that has been simplified
only at the indices actually summed over. -/
theorem sumOver_congr_term (guard : Nat → Bool) (first second : Nat → Pair) :
    ∀ indices : List Nat, (∀ index ∈ indices, first index = second index) →
      sumOver guard first indices = sumOver guard second indices
  | [], _ => rfl
  | index :: rest, agree => by
      rw [show sumOver guard first (index :: rest)
        = (if guard index then addPair (first index) (sumOver guard first rest)
           else sumOver guard first rest) from rfl,
        show sumOver guard second (index :: rest)
          = (if guard index then
              addPair (second index) (sumOver guard second rest)
            else sumOver guard second rest) from rfl,
        agree index (by simp),
        sumOver_congr_term guard first second rest
          (fun other member => agree other (List.mem_cons_of_mem _ member))]

/-- **A guard whose false positions have zero terms can be dropped.**  The
companion to `sumOver_range_truncate`: that one shortens the range when the
guard goes silent, this one removes the guard when silence costs nothing.

Which of the two applies depends on the degree. Below the ring degree the guard
is false at indices whose terms are *not* zero, so the range must shrink first;
at or above it, the guard is false exactly where a coefficient is out of range,
so the guard can go first. -/
theorem sumOver_of_guard_zero (guard : Nat → Bool) (term : Nat → Pair) :
    ∀ indices : List Nat,
      (∀ index ∈ indices, guard index = false → term index = ⟨0, 0⟩) →
      sumOver guard term indices = sumOver (fun _ => true) term indices
  | [], _ => rfl
  | index :: rest, vanish => by
      have inner := sumOver_of_guard_zero guard term rest
        (fun other member => vanish other (List.mem_cons_of_mem _ member))
      rw [show sumOver guard term (index :: rest)
          = (if guard index then addPair (term index) (sumOver guard term rest)
             else sumOver guard term rest) from rfl,
        show sumOver (fun _ => true) term (index :: rest)
          = addPair (term index) (sumOver (fun _ => true) term rest) from rfl,
        inner]
      by_cases fires : guard index
      · rw [if_pos fires]
      · rw [if_neg fires, vanish index (by simp) (by simpa using fires),
          addPair_zero_left_canonical _
            (sumOver_canonical (fun _ => true) term rest).1
            (sumOver_canonical (fun _ => true) term rest).2]

/-- **A range can be truncated where the terms vanish**, whatever the guard
does. The term-side counterpart of `sumOver_range_truncate`. -/
theorem sumOver_range_truncate_terms (guard : Nat → Bool) (term : Nat → Pair)
    (bound : Nat) (vanish : ∀ index, bound ≤ index → term index = ⟨0, 0⟩) :
    ∀ size, bound ≤ size →
      sumOver guard term (List.range size) = sumOver guard term (List.range bound)
  | 0, atLeast => by
      rw [show bound = 0 from by omega]
  | size + 1, atLeast => by
      by_cases reached : bound = size + 1
      · rw [reached]
      · rw [List.range_succ, sumOver_append,
          sumOver_of_zero_terms guard term [size]
            (fun index member => by
              have isSize : index = size := by simpa using member
              exact vanish index (by omega)),
          addPair_comm, addPair_zero_left_canonical _
            (sumOver_canonical guard term (List.range size)).1
            (sumOver_canonical guard term (List.range size)).2,
          sumOver_range_truncate_terms guard term bound vanish size (by omega)]

/-- **A range can be truncated where the guard goes silent.**  Everything past
the last firing index contributes nothing, so `rawMulCoeffK`'s fixed
`ringDegree` range collapses to the degree that matters.

Proved by peeling the range from the top rather than splitting with `drop`,
which keeps the membership reasoning to a single comparison. -/
theorem sumOver_range_truncate (guard : Nat → Bool) (term : Nat → Pair)
    (bound : Nat) (silent : ∀ index, bound < index → guard index = false) :
    ∀ size, bound + 1 ≤ size →
      sumOver guard term (List.range size)
        = sumOver guard term (List.range (bound + 1))
  | 0, atLeast => by omega
  | size + 1, atLeast => by
      by_cases exact : bound + 1 = size + 1
      · rw [exact]
      · have smaller : bound + 1 ≤ size := by omega
        rw [List.range_succ, sumOver_append,
          sumOver_of_no_guard guard term [size]
            (fun index member => by
              have : index = size := by simpa using member
              exact silent index (by omega)),
          addPair_comm, addPair_zero_left_canonical _
            (sumOver_canonical guard term (List.range size)).1
            (sumOver_canonical guard term (List.range size)).2,
          sumOver_range_truncate guard term bound silent size smaller]

/-! ## Translating the frozen definition

`rawMulCoeffK` is a guarded fold whose guard and term are stated in the
semantic vocabulary. These two lemmas restate them in the polynomial
vocabulary, which is what lets the accumulator lemma above apply to it at all.

Neither is the agreement itself: after both, the remaining step is showing the
guarded fold over `List.range ringDegree` visits exactly the terms
`convolution` does. -/

open Nightstream.SuperNeo.Concrete in
/-- **The guard collapses below the degree.**  `degree - index < ringDegree` is
implied by `index ≤ degree` once `degree < ringDegree`, so the frozen
two-part guard is just `index ≤ degree` on the range that matters.

Above the degree the second conjunct does real work, which is why the frozen
definition carries it. -/
theorem guard_collapses (degree index : Nat) (below : degree < ringDegree) :
    (index ≤ degree ∧ degree - index < ringDegree) ↔ index ≤ degree := by
  constructor
  · exact fun both => both.1
  · intro atMost
    exact ⟨atMost, by omega⟩

open Nightstream.SuperNeo.Concrete in
/-- **The frozen term is the polynomial term.**  One `ofConcrete_mul` and two
`coeffAt_toList`s. -/
theorem term_is_polynomial (left right : RingK) (degree index : Nat) :
    KConcreteBridge.ofConcrete
        (K.mul (ringKCoeff left index) (ringKCoeff right (degree - index)))
      = mulPair (KPolyCoeff.coeffAt (KPolyCoeff.toList left) index)
          (KPolyCoeff.coeffAt (KPolyCoeff.toList right) (degree - index)) := by
  rw [KConcreteBridge.ofConcrete_mul, KPolyCoeff.coeffAt_toList,
    KPolyCoeff.coeffAt_toList]

/-! ## The convolution is the guarded sum

The last step of `KPOLYMUL-RAWMULCOEFF-AGREEMENT`'s shape reconciliation.
`convolution` drops a list head and decrements the degree; `List.range (d+2)`
splits as `0 :: (List.range (d+1)).map Nat.succ`. `sumOver_map` re-indexes the
shifted tail so the two recursions line up. -/

open Nightstream.Implementation.R1CS.Canonical.KPolyCoeff in
/-- **The convolution is the sum over its own index range.** -/
theorem convolution_eq_sumOver (right : List Pair) :
    ∀ (left : List Pair) (degree : Nat),
      convolution left right degree
        = sumOver (fun _ => true)
            (fun index => mulPair (coeffAt left index)
              (coeffAt right (degree - index)))
            (List.range (degree + 1))
  | [], degree => by
      rw [convolution_nil]
      refine (sumOver_of_zero_terms _ _ _ (fun index _ => ?_)).symm
      rw [coeffAt_nil, mulPair_zero_left]
  | a :: left, 0 => by
      show mulPair a (coeffAt right 0) = _
      rw [show List.range 1 = [0] from rfl,
        show sumOver (fun _ => true)
            (fun index => mulPair (coeffAt (a :: left) index)
              (coeffAt right (0 - index))) [0]
          = addPair (mulPair (coeffAt (a :: left) 0) (coeffAt right 0))
              (⟨0, 0⟩ : Pair) from rfl,
        coeffAt_cons_zero, addPair_comm,
        addPair_zero_left_canonical _ (mulPair_canonical _ _).1
          (mulPair_canonical _ _).2]
  | a :: left, degree + 1 => by
      have shifted :
          (fun index => mulPair (coeffAt (a :: left) (Nat.succ index))
              (coeffAt right (degree + 1 - Nat.succ index)))
            = fun index => mulPair (coeffAt left index)
                (coeffAt right (degree - index)) := by
        funext index
        rw [coeffAt_cons_succ,
          show degree + 1 - Nat.succ index = degree - index from by omega]
      show addPair (mulPair a (coeffAt right (degree + 1)))
          (convolution left right degree) = _
      rw [show List.range (degree + 1 + 1)
            = 0 :: (List.range (degree + 1)).map Nat.succ
          from List.range_succ_eq_map,
        convolution_eq_sumOver right left degree,
        show sumOver (fun _ => true)
            (fun index => mulPair (coeffAt (a :: left) index)
              (coeffAt right (degree + 1 - index)))
            (0 :: (List.range (degree + 1)).map Nat.succ)
          = addPair (mulPair (coeffAt (a :: left) 0)
              (coeffAt right (degree + 1)))
              (sumOver (fun _ => true)
                (fun index => mulPair (coeffAt (a :: left) index)
                  (coeffAt right (degree + 1 - index)))
                ((List.range (degree + 1)).map Nat.succ)) from rfl,
        sumOver_map, shifted, coeffAt_cons_zero]

/-! ## Transferring the frozen fold

`rawMulCoeffK` accumulates in `K` with `K.add` and `K.mul`; everything above
accumulates in `Pair`. `ofConcrete` carries both operations, so it carries the
whole fold — with the accumulator generalized, since it changes at every step.

This is the last missing link: with it, the frozen definition becomes a
`sumOver` and every lemma above applies. -/

open Nightstream.SuperNeo.Concrete in
open Nightstream.Implementation.R1CS.Canonical.KPolyCoeff in
/-- **The fold transfers.**  `ofConcrete` of the frozen `K`-accumulation is the
`Pair`-accumulation of the transferred terms. -/
theorem ofConcrete_foldl (left right : RingK) (degree : Nat) :
    ∀ (indices : List Nat) (accumulator : K),
      KConcreteBridge.ofConcrete (indices.foldl (fun current index =>
          if index ≤ degree ∧ degree - index < ringDegree then
            K.add current
              (K.mul (ringKCoeff left index) (ringKCoeff right (degree - index)))
          else current) accumulator)
        = indices.foldl
            (step (fun index => decide (index ≤ degree ∧ degree - index < ringDegree))
              (fun index => mulPair (coeffAt (toList left) index)
                (coeffAt (toList right) (degree - index))))
            (KConcreteBridge.ofConcrete accumulator)
  | [], _ => rfl
  | index :: rest, accumulator => by
      simp only [List.foldl_cons, step]
      by_cases fires : index ≤ degree ∧ degree - index < ringDegree
      · rw [if_pos fires, if_pos (by simpa using fires),
          ofConcrete_foldl left right degree rest _,
          KConcreteBridge.ofConcrete_add, term_is_polynomial]
      · rw [if_neg fires, if_neg (by simpa using fires),
          ofConcrete_foldl left right degree rest accumulator]

/-! ## Discarding vanishing terms

The three-way reduction needs the surviving terms extracted from a sum whose
other terms are zero. Rather than splitting the range around each survivor,
filter: dropping indices whose term vanishes leaves the sum unchanged.

That turns "extract three terms from 107" into "filter, then evaluate the
filtered list", and the filtered list is short enough to compute. -/

/-- **Filtering out vanishing terms preserves the sum.** -/
theorem sumOver_filter (term : Nat → Pair) (keep : Nat → Bool) :
    ∀ indices : List Nat,
      (∀ index ∈ indices, keep index = false → term index = ⟨0, 0⟩) →
      sumOver (fun _ => true) term (indices.filter keep)
        = sumOver (fun _ => true) term indices
  | [], _ => rfl
  | index :: rest, vanish => by
      have tail : ∀ other ∈ rest, keep other = false → term other = ⟨0, 0⟩ :=
        fun other member => vanish other (List.mem_cons_of_mem _ member)
      by_cases kept : keep index
      · rw [List.filter_cons_of_pos kept]
        show addPair (term index) (sumOver _ term (rest.filter keep))
          = addPair (term index) (sumOver _ term rest)
        rw [sumOver_filter term keep rest tail]
      · rw [List.filter_cons_of_neg (by simpa using kept),
          sumOver_filter term keep rest tail]
        show _ = addPair (term index) (sumOver _ term rest)
        rw [vanish index (by simp) (by simpa using kept),
          addPair_zero_left_canonical _ (sumOver_canonical _ term rest).1
            (sumOver_canonical _ term rest).2]

/-! ## The agreement

Composed from the chain above.  Nothing new is proved here; every step is an
existing theorem, and the point of stating it is that the pieces have been put
together rather than merely all existing. -/

open Nightstream.SuperNeo.Concrete in
open Nightstream.Implementation.R1CS.Canonical.KPolyCoeff in
/-- **`polyMul`'s coefficients are the frozen `rawMulCoeffK`'s.**

This is what makes every theorem in the `K` tower a statement about the
protocol's multiplication rather than about a definition of my own. -/
theorem rawMulCoeffK_eq_coeffAt_polyMul
    (left right : RingK) (degree : Nat) (below : degree < ringDegree) :
    KConcreteBridge.ofConcrete (rawMulCoeffK left right degree)
      = coeffAt (KPolyHom.polyMul (toList left) (toList right)) degree := by
  have guardTrue : ∀ index ∈ List.range (degree + 1),
      decide (index ≤ degree ∧ degree - index < ringDegree) = true := by
    intro index member
    have : index < degree + 1 := List.mem_range.1 member
    simp only [decide_eq_true_eq]
    exact ⟨by omega, by omega⟩
  have guardFalse : ∀ index, degree < index →
      decide (index ≤ degree ∧ degree - index < ringDegree) = false := by
    intro index above
    simp only [decide_eq_false_iff_not, not_and]
    omega
  rw [show rawMulCoeffK left right degree
      = (List.range ringDegree).foldl (fun current index =>
          if index ≤ degree ∧ degree - index < ringDegree then
            K.add current
              (K.mul (ringKCoeff left index) (ringKCoeff right (degree - index)))
          else current) K.zero from rfl,
    ofConcrete_foldl left right degree,
    show KConcreteBridge.ofConcrete K.zero = (⟨0, 0⟩ : Pair) from rfl,
    foldl_step_from_zero,
    sumOver_range_truncate _ _ degree guardFalse ringDegree (by omega),
    sumOver_congr_guard _ _ (fun _ => true) _ guardTrue,
    ← convolution_eq_sumOver, ← coeffAt_polyMul]

/-! ## Above the ring degree

The theorem above stops at `degree < ringDegree` because its proof shortens the
frozen fold's `List.range ringDegree` down to `List.range (degree + 1)`. At or
above the ring degree that direction is wrong — the frozen range is already the
shorter one — so the two steps swap:

| | below the degree | at or above it |
|---|---|---|
| first | shorten the range, since the guard is silent past the degree | drop the guard, since it is false exactly where a coefficient is out of range |
| then | drop the guard, now true everywhere | lengthen the range, since the added terms are zero |

The swap is forced, not stylistic. Below the degree the guard is false at
indices whose terms are *not* zero (any `index` in `(degree, 54)` reads
`coeffAt right 0`), so the guard cannot be dropped first. This matters because
`ringKMul` reads `rawMulCoeffK` at degrees 54 through 106 — **every**
off-diagonal read of the frozen reduction lands in this range. -/

open Nightstream.SuperNeo.Concrete in
open Nightstream.Implementation.R1CS.Canonical.KPolyCoeff in
/-- **The agreement at or above the ring degree.** -/
theorem rawMulCoeffK_eq_coeffAt_polyMul_high
    (left right : RingK) (degree : Nat) (atLeast : ringDegree ≤ degree) :
    KConcreteBridge.ofConcrete (rawMulCoeffK left right degree)
      = coeffAt (KPolyHom.polyMul (toList left) (toList right)) degree := by
  have guardZero : ∀ index ∈ List.range ringDegree,
      decide (index ≤ degree ∧ degree - index < ringDegree) = false →
      mulPair (coeffAt (toList left) index)
        (coeffAt (toList right) (degree - index)) = ⟨0, 0⟩ := by
    intro index member notFired
    have small : index < ringDegree := List.mem_range.1 member
    simp only [decide_eq_false_iff_not, not_and, Nat.not_lt] at notFired
    rw [coeffAt_beyond_length (toList right) _
        (by rw [toList_length]; exact notFired (by omega)),
      mulPair_zero_right]
  have termVanishes : ∀ index, ringDegree ≤ index →
      mulPair (coeffAt (toList left) index)
        (coeffAt (toList right) (degree - index)) = ⟨0, 0⟩ := by
    intro index beyond
    rw [coeffAt_beyond_length (toList left) _ (by rw [toList_length]; exact beyond),
      mulPair_zero_left]
  rw [show rawMulCoeffK left right degree
      = (List.range ringDegree).foldl (fun current index =>
          if index ≤ degree ∧ degree - index < ringDegree then
            K.add current
              (K.mul (ringKCoeff left index) (ringKCoeff right (degree - index)))
          else current) K.zero from rfl,
    ofConcrete_foldl left right degree,
    show KConcreteBridge.ofConcrete K.zero = (⟨0, 0⟩ : Pair) from rfl,
    foldl_step_from_zero,
    sumOver_of_guard_zero _ _ _ guardZero,
    ← sumOver_range_truncate_terms (fun _ => true) _ ringDegree termVanishes
      (degree + 1) (by omega),
    ← convolution_eq_sumOver, ← coeffAt_polyMul]

open Nightstream.SuperNeo.Concrete in
open Nightstream.Implementation.R1CS.Canonical.KPolyCoeff in
/-- **The agreement, at every degree.**  This is what makes the whole `K` tower
— including `KQuotient`'s reduction identity — a set of statements about the
protocol's multiplication rather than about `polyMul`. -/
theorem rawMulCoeffK_eq_coeffAt_polyMul_all
    (left right : RingK) (degree : Nat) :
    KConcreteBridge.ofConcrete (rawMulCoeffK left right degree)
      = coeffAt (KPolyHom.polyMul (toList left) (toList right)) degree := by
  by_cases below : degree < ringDegree
  · exact rawMulCoeffK_eq_coeffAt_polyMul left right degree below
  · exact rawMulCoeffK_eq_coeffAt_polyMul_high left right degree
      (Nat.not_lt.1 below)

end Nightstream.Implementation.R1CS.Canonical.KFoldSum
