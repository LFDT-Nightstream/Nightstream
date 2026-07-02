import SuperNeo.SumCheck.Defs

/-!
Full-field challenge-coin counting and the uniform-coin probability
model used by the SumCheck soundness games.
-/

namespace SuperNeo.ProofSystem

namespace Sumcheck

private theorem list_sum_map_const_nat
  {α : Type}
  (xs : List α)
  (c : Nat) :
  (xs.map (fun _ => c)).sum = xs.length * c := by
  induction xs with
  | nil =>
      simp
  | cons _ xs ih =>
      simp [ih, Nat.succ_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

theorem fullFieldCoinSpace_length (m : Nat) :
    (fullFieldCoinSpace m).length = Goldilocks.q ^ m := by
  induction m with
  | zero =>
      simp [fullFieldCoinSpace]
  | succ m ih =>
      calc
        (fullFieldCoinSpace (m + 1)).length
            = ((fullFieldCoinSpace m).map (fun _ => Goldilocks.q)).sum := by
                simp [fullFieldCoinSpace, List.length_flatMap, List.length_map, fullFieldChallengeDomain_length]
        _ = (fullFieldCoinSpace m).length * Goldilocks.q := by
              simpa using list_sum_map_const_nat (fullFieldCoinSpace m) Goldilocks.q
        _ = Goldilocks.q ^ m * Goldilocks.q := by
              simpa [ih]
        _ = Goldilocks.q ^ (m + 1) := by
              simp [Nat.pow_succ, Nat.mul_comm]

/--
Count verifier-coin assignments in the full-field product space that satisfy `E`.
-/
noncomputable def fullFieldCoinEventCount (m : Nat) (E : Array F → Prop) : Nat :=
  by
    classical
    exact ((fullFieldCoinSpace m).filter (fun coins => decide (E coins))).length

/--
Boolean-form event count over full-field verifier-coin assignments.

This is a helper surface for exact finite counting proofs; it aligns with
`fullFieldCoinEventCount` via `E = fun coins => B coins = true`.
-/
noncomputable def fullFieldCoinEventCountBool (m : Nat) (E : Array F → Bool) : Nat :=
  ((fullFieldCoinSpace m).filter E).length

private theorem list_sum_map_eq_length_mul_const
  {α : Type}
  (f : α → Nat)
  (l : List α)
  (c : Nat)
  (hConst : ∀ x ∈ l, f x = c) :
  (l.map f).sum = l.length * c := by
  induction l with
  | nil =>
      simp
  | cons x xs ih =>
      have hx : f x = c := hConst x (by simp)
      have hxs : ∀ y ∈ xs, f y = c := by
        intro y hy
        exact hConst y (by simp [hy])
      have ih' := ih hxs
      calc
        (List.map f (x :: xs)).sum = f x + (List.map f xs).sum := by simp
        _ = c + xs.length * c := by simp [hx, ih']
        _ = (x :: xs).length * c := by simp [Nat.succ_mul, Nat.add_comm]

private theorem list_sum_map_le_length_mul_const
  {α : Type}
  (f : α → Nat)
  (l : List α)
  (c : Nat)
  (hBound : ∀ x ∈ l, f x ≤ c) :
  (l.map f).sum ≤ l.length * c := by
  induction l with
  | nil =>
      simp
  | cons x xs ih =>
      have hx : f x ≤ c := hBound x (by simp)
      have hxs : ∀ y ∈ xs, f y ≤ c := by
        intro y hy
        exact hBound y (by simp [hy])
      have ih' := ih hxs
      calc
        (List.map f (x :: xs)).sum = f x + (List.map f xs).sum := by simp
        _ ≤ c + xs.length * c := Nat.add_le_add hx ih'
        _ = (x :: xs).length * c := by simp [Nat.succ_mul, Nat.add_comm]

private theorem list_sum_map_ite_eq_mul_filter_length
  {α : Type}
  (l : List α)
  (P : α → Bool)
  (c : Nat) :
  (l.map (fun a => if P a then c else 0)).sum = c * (l.filter P).length := by
  induction l with
  | nil =>
      simp
  | cons a l ih =>
      by_cases hPa : P a
      · simp [hPa, ih, Nat.mul_add, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
      · simp [hPa, ih, Nat.mul_add, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

/--
Exact count for last-coordinate predicates over full-field coin space:
`count({coins | B coins[m]}) = rootCount(B) * |F|^m`.
-/
theorem fullFieldCoinEventCountBool_last
  (m : Nat)
  (B : F → Bool) :
  fullFieldCoinEventCountBool (m + 1) (fun coins => B (coins[m]!))
    = (fullFieldChallengeDomain.filter B).length * (fullFieldCoinSpace m).length := by
  classical
  let rootCount := (fullFieldChallengeDomain.filter B).length
  let f : Array F → Nat :=
    fun a =>
      (List.filter (fun coins => B (coins[m]!))
        (fullFieldChallengeDomain.map (fun r => a.push r))).length
  unfold fullFieldCoinEventCountBool
  rw [fullFieldCoinSpace]
  rw [List.filter_flatMap, List.length_flatMap]
  have hConst : ∀ a ∈ fullFieldCoinSpace m, f a = rootCount := by
    intro a ha
    have hsize : a.size = m := mem_fullFieldCoinSpace_size ha
    have hPredEq : (fun r => B ((a.push r)[m]!)) = (fun r => B r) := by
      funext r
      have hm : m = a.size := by omega
      subst hm
      simp
    show
      (List.filter (fun coins => B (coins[m]!))
        (fullFieldChallengeDomain.map (fun r => a.push r))).length = rootCount
    calc
      (List.filter (fun coins => B (coins[m]!)) (fullFieldChallengeDomain.map (fun r => a.push r))).length
          = (List.filter ((fun coins => B (coins[m]!)) ∘ fun r => a.push r) fullFieldChallengeDomain).length := by
              simp [List.filter_map]
      _ = (List.filter (fun r => B r) fullFieldChallengeDomain).length := by
            change (List.filter (fun r => B ((a.push r)[m]!)) fullFieldChallengeDomain).length =
              (List.filter (fun r => B r) fullFieldChallengeDomain).length
            rw [hPredEq]
      _ = rootCount := rfl
  have hSum : (List.map f (fullFieldCoinSpace m)).sum = (fullFieldCoinSpace m).length * rootCount := by
    exact list_sum_map_eq_length_mul_const (f := f) (l := fullFieldCoinSpace m) (c := rootCount) hConst
  calc
    (List.map f (fullFieldCoinSpace m)).sum = (fullFieldCoinSpace m).length * rootCount := hSum
    _ = rootCount * (fullFieldCoinSpace m).length := by simp [Nat.mul_comm]

/--
Recurrence for non-last coordinate predicates:
adding one trailing coordinate multiplies event count by `|F|`.
-/
theorem fullFieldCoinEventCountBool_coord_succ_lt
  (m i : Nat)
  (hi : i < m)
  (B : F → Bool) :
  fullFieldCoinEventCountBool (m + 1) (fun coins => B (coins[i]!))
    = Goldilocks.q * fullFieldCoinEventCountBool m (fun coins => B (coins[i]!)) := by
  classical
  unfold fullFieldCoinEventCountBool
  rw [fullFieldCoinSpace]
  rw [List.filter_flatMap, List.length_flatMap]
  have hInner :
      List.map
        (fun a =>
          (List.filter (fun coins => B (coins[i]!))
            (fullFieldChallengeDomain.map (fun r => a.push r))).length)
        (fullFieldCoinSpace m)
      =
      List.map
        (fun a => if B (a[i]!) then Goldilocks.q else 0)
        (fullFieldCoinSpace m) := by
    apply List.ext_get
    · simp
    · intro j hj1 hj2
      simp at hj1
      have hjMem : (fullFieldCoinSpace m).get ⟨j, hj1⟩ ∈ fullFieldCoinSpace m := by
        exact List.get_mem _ _
      let a : Array F := (fullFieldCoinSpace m).get ⟨j, hj1⟩
      have hsize : a.size = m := mem_fullFieldCoinSpace_size hjMem
      have hPredEq : (fun r => B ((a.push r)[i]!)) = (fun _ : F => B (a[i]!)) := by
        funext r
        have hiA : i < a.size := by simpa [hsize] using hi
        have hiPush : i < (a.push r).size := by
          have hlt : i < a.size + 1 := Nat.lt_trans hiA (Nat.lt_succ_self a.size)
          simpa [Array.size_push] using hlt
        calc
          B ((a.push r)[i]!) = B ((a.push r)[i]) := by
            rw [getElem!_pos (c := a.push r) (i := i) hiPush]
          _ = B (a[i]) := by
            simpa using congrArg B (Array.getElem_push_lt (xs := a) (x := r) (i := i) hiA)
          _ = B (a[i]!) := by
            rw [getElem!_pos (c := a) (i := i) hiA]
      have hFilterEq :
          (List.filter (fun coins => B (coins[i]!))
            (fullFieldChallengeDomain.map (fun r => a.push r))).length
            = (if B (a[i]!) then Goldilocks.q else 0) := by
        calc
          (List.filter (fun coins => B (coins[i]!))
            (fullFieldChallengeDomain.map (fun r => a.push r))).length
              = (List.filter ((fun coins => B (coins[i]!)) ∘ fun r => a.push r)
                    fullFieldChallengeDomain).length := by
                    simp [List.filter_map]
          _ = (List.filter (fun _ : F => B (a[i]!)) fullFieldChallengeDomain).length := by
                change (List.filter (fun r => B ((a.push r)[i]!)) fullFieldChallengeDomain).length =
                  (List.filter (fun _ : F => B (a[i]!)) fullFieldChallengeDomain).length
                rw [hPredEq]
          _ = (if B (a[i]!) then Goldilocks.q else 0) := by
                by_cases hBa : B (a[i]!)
                · simp [hBa, fullFieldChallengeDomain_length]
                · simp [hBa]
      dsimp [a] at hFilterEq
      simpa using hFilterEq
  rw [hInner]
  have hSum := list_sum_map_ite_eq_mul_filter_length
      (l := fullFieldCoinSpace m)
      (P := fun a => B (a[i]!))
      (c := Goldilocks.q)
  calc
    (List.map (fun a => if B (a[i]!) then Goldilocks.q else 0) (fullFieldCoinSpace m)).sum
        = Goldilocks.q * (List.filter (fun a => B (a[i]!)) (fullFieldCoinSpace m)).length := hSum
    _ = Goldilocks.q * fullFieldCoinEventCountBool m (fun coins => B (coins[i]!)) := by rfl

/--
Exact coordinate-lift counting theorem:
for `i < m`, predicates on `coins[i]` satisfy
`count = rootCount * |F|^(m-1)`.
-/
theorem fullFieldCoinEventCountBool_coord_exact
  (m i : Nat)
  (hi : i < m)
  (B : F → Bool) :
  fullFieldCoinEventCountBool m (fun coins => B (coins[i]!))
    = (fullFieldChallengeDomain.filter B).length * Goldilocks.q ^ (m - 1) := by
  set rootCount : Nat := (fullFieldChallengeDomain.filter B).length
  have hDecomp : ∃ t, m = i + 1 + t := by
    refine ⟨m - (i + 1), ?_⟩
    omega
  rcases hDecomp with ⟨t, hm⟩
  subst hm
  induction t with
  | zero =>
      simpa [rootCount, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, fullFieldCoinSpace_length]
        using fullFieldCoinEventCountBool_last i B
  | succ t ih =>
      have hiStep : i < i + 1 + t := by omega
      have hRec := fullFieldCoinEventCountBool_coord_succ_lt (m := i + 1 + t) (i := i) hiStep B
      have hsub : (i + 1 + t) - 1 = i + t := by omega
      have ih' :
          fullFieldCoinEventCountBool (i + 1 + t) (fun coins => B (coins[i]!))
            = rootCount * Goldilocks.q ^ (i + t) := by
        exact (ih hiStep).trans (by simp [rootCount, hsub])
      have hsub2 : (i + 1 + t + 1) - 1 = i + t + 1 := by omega
      calc
        fullFieldCoinEventCountBool (i + 1 + t + 1) (fun coins => B (coins[i]!))
            = Goldilocks.q * fullFieldCoinEventCountBool (i + 1 + t) (fun coins => B (coins[i]!)) := by
                simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hRec
        _ = Goldilocks.q * (rootCount * Goldilocks.q ^ (i + t)) := by simpa [ih']
        _ = rootCount * Goldilocks.q ^ (i + t + 1) := by
              simp [Nat.pow_succ, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm]
        _ = rootCount * Goldilocks.q ^ ((i + 1 + t + 1) - 1) := by
              simp [hsub2]

private theorem fullFieldChallengeDomain_filter_length_eq_finset_card
  (B : F → Bool) :
  (fullFieldChallengeDomain.filter B).length =
    (Finset.univ.filter (fun r : F => B r = true)).card := by
  have hNodup : (fullFieldChallengeDomain.filter B).Nodup := by
    exact (List.nodup_finRange Goldilocks.q).filter B
  calc
    (fullFieldChallengeDomain.filter B).length
        = (fullFieldChallengeDomain.filter B).toFinset.card :=
            (List.toFinset_card_of_nodup hNodup).symm
    _ = (fullFieldChallengeDomain.toFinset.filter (fun r : F => B r = true)).card := by
          simp [List.toFinset_filter]
    _ = (Finset.univ.filter (fun r : F => B r = true)).card := by
          simp [fullFieldChallengeDomain, List.toFinset_finRange]

/--
Exact coordinate-lift counting in proposition form.
-/
theorem fullFieldCoinEventCount_coordPredicate_exact
  (m i : Nat)
  (hi : i < m)
  (P : F → Prop)
  [DecidablePred P] :
  fullFieldCoinEventCount m (fun coins => P (coins[i]!))
    = (Finset.univ.filter (fun r : F => P r)).card * Goldilocks.q ^ (m - 1) := by
  classical
  let BP : F → Bool := fun r => @decide (P r) (Classical.propDecidable (P r))
  have hBool := fullFieldCoinEventCountBool_coord_exact m i hi BP
  have hCard :
      (fullFieldChallengeDomain.filter BP).length =
        (Finset.univ.filter (fun r : F => P r)).card := by
    calc
      (fullFieldChallengeDomain.filter BP).length
          = (Finset.univ.filter (fun r : F => BP r = true)).card :=
              fullFieldChallengeDomain_filter_length_eq_finset_card BP
      _ = (Finset.univ.filter (fun r : F => P r)).card := by
            have hEq :
                (Finset.univ.filter (fun r : F => BP r = true)) =
                  (Finset.univ.filter (fun r : F => P r)) := by
              ext r
              simp [BP]
            simpa [hEq]
  unfold fullFieldCoinEventCount
  simpa [fullFieldCoinEventCountBool, BP, hCard] using hBool

theorem fullFieldCoinEventCount_le_space
  (m : Nat)
  (E : Array F → Prop) :
  fullFieldCoinEventCount m E ≤ (fullFieldCoinSpace m).length := by
  classical
  simpa [fullFieldCoinEventCount] using
    (List.length_filter_le (fun coins => decide (E coins)) (fullFieldCoinSpace m))

private theorem filter_length_mono_of_imp
  {α : Type}
  (p q : α → Bool)
  (hImp : ∀ a, p a = true → q a = true) :
  ∀ l : List α, (l.filter p).length ≤ (l.filter q).length := by
  intro l
  induction l with
  | nil =>
      simp
  | cons a l ih =>
      by_cases hp : p a = true
      · have hq : q a = true := hImp a hp
        simp [List.filter, hp, hq, ih]
      · by_cases hq : q a = true
        · have hStep : (l.filter p).length ≤ Nat.succ (l.filter q).length := by
            exact Nat.le_trans ih (Nat.le_succ _)
          simpa [List.filter, hp, hq] using hStep
        · simpa [List.filter, hp, hq] using ih

private theorem filter_length_union_le
  {α : Type}
  (p q : α → Bool) :
  ∀ l : List α,
    (l.filter (fun a => p a || q a)).length ≤
      (l.filter p).length + (l.filter q).length := by
  intro l
  induction l with
  | nil =>
      simp
  | cons a l ih =>
      by_cases hp : p a = true
      · by_cases hq : q a = true
        · have hStep1 :
            Nat.succ ((l.filter (fun a => p a || q a)).length) ≤
              Nat.succ ((l.filter p).length + (l.filter q).length) := by
            exact Nat.succ_le_succ ih
          have hStep2 :
              Nat.succ ((l.filter p).length + (l.filter q).length) ≤
                Nat.succ (l.filter p).length + Nat.succ (l.filter q).length := by
            have h := Nat.le_add_right (Nat.succ ((l.filter p).length + (l.filter q).length)) 1
            simpa [Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using h
          have hStep :
              Nat.succ ((l.filter (fun a => p a || q a)).length) ≤
                Nat.succ (l.filter p).length + Nat.succ (l.filter q).length :=
            Nat.le_trans hStep1 hStep2
          simpa [List.filter, hp, hq, Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
            using hStep
        · have hStep :
            Nat.succ ((l.filter (fun a => p a || q a)).length) ≤
              Nat.succ ((l.filter p).length + (l.filter q).length) := by
            exact Nat.succ_le_succ ih
          simpa [List.filter, hp, hq, Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
            using hStep
      · by_cases hq : q a = true
        · have hStep :
            Nat.succ ((l.filter (fun a => p a || q a)).length) ≤
              Nat.succ ((l.filter p).length + (l.filter q).length) := by
            exact Nat.succ_le_succ ih
          simpa [List.filter, hp, hq, Bool.false_or, Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_left_comm,
            Nat.add_comm]
            using hStep
        · simpa [List.filter, hp, hq, Bool.false_or] using ih

theorem fullFieldCoinEventCount_mono
  (m : Nat)
  {E1 E2 : Array F → Prop}
  (hImp : ∀ coins, E1 coins → E2 coins) :
  fullFieldCoinEventCount m E1 ≤ fullFieldCoinEventCount m E2 := by
  classical
  unfold fullFieldCoinEventCount
  apply filter_length_mono_of_imp
  intro coins hDec
  have hE1 : E1 coins := of_decide_eq_true hDec
  exact decide_eq_true (hImp coins hE1)

/--
Coordinate-lift upper bound from implication into a coordinate predicate.
-/
theorem fullFieldCoinEventCount_le_coordPredicate
  (m i : Nat)
  (hi : i < m)
  (E : Array F → Prop)
  (P : F → Prop)
  [DecidablePred P]
  (hImp : ∀ coins, E coins → P (coins[i]!)) :
  fullFieldCoinEventCount m E ≤
    (Finset.univ.filter (fun r : F => P r)).card * Goldilocks.q ^ (m - 1) := by
  have hMono :
      fullFieldCoinEventCount m E ≤
        fullFieldCoinEventCount m (fun coins => P (coins[i]!)) :=
    fullFieldCoinEventCount_mono m hImp
  have hExact :=
    fullFieldCoinEventCount_coordPredicate_exact m i hi P
  exact hMono.trans (by simpa [hExact] using (Nat.le_refl _))

private def lastPrefixRootSliceCount
  (m : Nat)
  (rootSet : Array F → Finset F)
  (a : Array F) : Nat :=
  (List.filter
    (fun coins => coins[m]! ∈ rootSet (coins.extract 0 m))
    (fullFieldChallengeDomain.map (fun r => a.push r))).length

/--
Last-coordinate slice counting bound.

For each fixed prefix `a : F^m`, if the set of allowed last-coordinate values
`rootSet a` has size at most `d`, then the event
`coins[m] ∈ rootSet(coins[0..m))` occurs on at most `d * |F|^m` full coins.
-/
theorem fullFieldCoinEventCount_lastPrefixRootSet_le
  (m d : Nat)
  (rootSet : Array F → Finset F)
  (hBound :
    ∀ a ∈ fullFieldCoinSpace m, (rootSet a).card ≤ d) :
  fullFieldCoinEventCount (m + 1)
      (fun coins => coins[m]! ∈ rootSet (coins.extract 0 m)) ≤
    (fullFieldCoinSpace m).length * d := by
  classical
  have hInnerLe :
      ∀ a ∈ fullFieldCoinSpace m, lastPrefixRootSliceCount m rootSet a ≤ d := by
    intro a ha
    have hSize : a.size = m := mem_fullFieldCoinSpace_size ha
    have hExtractPush : ∀ r : F, (a.push r).extract 0 m = a := by
      intro r
      apply Array.ext
      · simp [hSize]
      · intro j hj1 hj2
        have hj : j < m := by simpa [hSize] using hj1
        simp [Array.getElem_extract, Array.getElem_push_lt, hSize, hj]
    have hFilterEq :
        (List.filter
          (fun coins => coins[m]! ∈ rootSet (coins.extract 0 m))
          (fullFieldChallengeDomain.map (fun r => a.push r))).length =
          (fullFieldChallengeDomain.filter (fun r => decide (r ∈ rootSet a))).length := by
      have hPredEq :
          (fun r : F => decide ((a.push r)[m]! ∈ rootSet ((a.push r).extract 0 m))) =
            (fun r : F => decide (r ∈ rootSet a)) := by
        funext r
        have hmPush : m < (a.push r).size := by simpa [hSize]
        have hLast : (a.push r)[m]! = r := by
          rw [getElem!_pos (c := a.push r) (i := m) hmPush]
          simpa [hSize] using Array.getElem_push_eq (xs := a) (x := r)
        simp [hLast, hExtractPush r]
      calc
        (List.filter
          (fun coins => coins[m]! ∈ rootSet (coins.extract 0 m))
          (fullFieldChallengeDomain.map (fun r => a.push r))).length
            =
            (List.filter
              ((fun coins => coins[m]! ∈ rootSet (coins.extract 0 m)) ∘ fun r => a.push r)
              fullFieldChallengeDomain).length := by
                simp [List.filter_map]
        _ =
            (List.filter (fun r => decide (r ∈ rootSet a)) fullFieldChallengeDomain).length := by
              simpa using congrArg (fun p => (List.filter p fullFieldChallengeDomain).length) hPredEq
    have hCard :
        (fullFieldChallengeDomain.filter (fun r => decide (r ∈ rootSet a))).length =
          (rootSet a).card := by
      simpa using
        (fullFieldChallengeDomain_filter_length_eq_finset_card
          (fun r => decide (r ∈ rootSet a)))
    calc
      lastPrefixRootSliceCount m rootSet a
          = (fullFieldChallengeDomain.filter (fun r => decide (r ∈ rootSet a))).length := by
              unfold lastPrefixRootSliceCount
              exact hFilterEq
      _ = (rootSet a).card := hCard
      _ ≤ d := hBound a ha
  have hBoundList :
      ∀ l : List (Array F),
        (∀ a ∈ l, lastPrefixRootSliceCount m rootSet a ≤ d) →
        (List.filter
          (fun coins => decide (coins[m]! ∈ rootSet (coins.extract 0 m)))
          (l.flatMap (fun a => fullFieldChallengeDomain.map (fun r => a.push r)))).length
          ≤ l.length * d := by
    intro l
    induction l with
    | nil =>
        intro _h
        simp
    | cons a t ih =>
        intro hSlice
        have hHead : lastPrefixRootSliceCount m rootSet a ≤ d := hSlice a (by simp)
        have hTail :
            ∀ b ∈ t, lastPrefixRootSliceCount m rootSet b ≤ d := by
          intro b hb
          exact hSlice b (by simp [hb])
        have ih' := ih hTail
        calc
          (List.filter
              (fun coins => decide (coins[m]! ∈ rootSet (coins.extract 0 m)))
              ((a :: t).flatMap (fun a => fullFieldChallengeDomain.map (fun r => a.push r)))).length
              =
                (List.filter
                  (fun coins => decide (coins[m]! ∈ rootSet (coins.extract 0 m)))
                  (fullFieldChallengeDomain.map (fun r => a.push r))).length
                  +
                (List.filter
                  (fun coins => decide (coins[m]! ∈ rootSet (coins.extract 0 m)))
                  (t.flatMap (fun a => fullFieldChallengeDomain.map (fun r => a.push r)))).length := by
                    simp
          _ ≤ d + t.length * d := by
                gcongr
                simpa [lastPrefixRootSliceCount] using hHead
          _ = (a :: t).length * d := by
                simp [Nat.succ_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
  have hDecEq :
      (fun coins : Array F =>
        @decide (coins[m]! ∈ rootSet (coins.extract 0 m))
          (Finset.decidableMem coins[m]! (rootSet (coins.extract 0 m)))) =
      (fun coins : Array F =>
        @decide ((fun coins => coins[m]! ∈ rootSet (coins.extract 0 m)) coins)
          (Classical.propDecidable ((fun coins => coins[m]! ∈ rootSet (coins.extract 0 m)) coins))) := by
    funext coins
    by_cases h : coins[m]! ∈ rootSet (coins.extract 0 m)
    · simp [h]
    · simp [h]
  unfold fullFieldCoinEventCount
  rw [fullFieldCoinSpace]
  simpa only [hDecEq] using hBoundList (fullFieldCoinSpace m) hInnerLe

theorem fullFieldCoinEventCount_union_le
  (m : Nat)
  (E1 E2 : Array F → Prop) :
  fullFieldCoinEventCount m (fun coins => E1 coins ∨ E2 coins) ≤
    fullFieldCoinEventCount m E1 + fullFieldCoinEventCount m E2 := by
  classical
  have hDecOr :
      (fun coins => decide (E1 coins ∨ E2 coins)) =
        (fun coins => decide (E1 coins) || decide (E2 coins)) := by
    funext coins
    simpa using (Bool.decide_or (E1 coins) (E2 coins))
  unfold fullFieldCoinEventCount
  simpa [hDecOr] using
    (filter_length_union_le
      (fun coins => decide (E1 coins))
      (fun coins => decide (E2 coins))
      (fullFieldCoinSpace m))

/--
Concrete full-field product probability over verifier coins:
`Pr[E] = #E / #F^m`.
-/
noncomputable def fullFieldCoinPr (m : Nat) (E : Array F → Prop) : Rat :=
  Rat.divInt (fullFieldCoinEventCount m E) ((fullFieldCoinSpace m).length)

theorem fullFieldCoinPr_nonneg (m : Nat) (E : Array F → Prop) :
  0 ≤ fullFieldCoinPr m E := by
  unfold fullFieldCoinPr
  exact Rat.divInt_nonneg
    (by exact_mod_cast (Nat.zero_le (fullFieldCoinEventCount m E)))
    (by exact_mod_cast (Nat.zero_le (fullFieldCoinSpace m).length))

theorem fullFieldCoinPr_le_one (m : Nat) (E : Array F → Prop) :
  fullFieldCoinPr m E ≤ 1 := by
  have hDenPosNat : 0 < (fullFieldCoinSpace m).length := fullFieldCoinSpace_length_pos m
  have hDenPosRat : 0 < ((fullFieldCoinSpace m).length : Rat) := by
    exact (Rat.natCast_pos).2 hDenPosNat
  have hCountLeNat :
      fullFieldCoinEventCount m E ≤ (fullFieldCoinSpace m).length :=
    fullFieldCoinEventCount_le_space m E
  have hCountLeRat :
      (fullFieldCoinEventCount m E : Rat) ≤ ((fullFieldCoinSpace m).length : Rat) := by
    exact (Rat.natCast_le_natCast).2 hCountLeNat
  have hNotLt : ¬ 1 < fullFieldCoinPr m E := by
    intro hLt
    have hLtDiv :
        1 <
          (fullFieldCoinEventCount m E : Rat) /
            ((fullFieldCoinSpace m).length : Rat) := by
      simpa [fullFieldCoinPr, Rat.divInt_eq_div] using hLt
    have hDenLtCount :
        ((fullFieldCoinSpace m).length : Rat) <
          (fullFieldCoinEventCount m E : Rat) := by
      have hMulLt :
          (1 : Rat) * ((fullFieldCoinSpace m).length : Rat) <
            (fullFieldCoinEventCount m E : Rat) :=
        (Rat.lt_div_iff hDenPosRat).1 hLtDiv
      simpa [Rat.one_mul] using hMulLt
    have hNo : ¬ ((fullFieldCoinSpace m).length : Rat) <
        (fullFieldCoinEventCount m E : Rat) := (Rat.not_lt).2 hCountLeRat
    exact hNo hDenLtCount
  exact (Rat.not_lt).1 hNotLt

theorem fullFieldCoinPr_false (m : Nat) :
  fullFieldCoinPr m (fun _ => False) = 0 := by
  classical
  have hCountZero : fullFieldCoinEventCount m (fun _ => False) = 0 := by
    unfold fullFieldCoinEventCount
    simp
  unfold fullFieldCoinPr
  simpa [hCountZero] using (Rat.zero_divInt ((fullFieldCoinSpace m).length : Int))

private theorem fullFieldCoinPr_mul_den
  (m : Nat)
  (E : Array F → Prop) :
  fullFieldCoinPr m E * (((fullFieldCoinSpace m).length : Int) : Rat) =
    (fullFieldCoinEventCount m E : Rat) := by
  have hDenPosNat : 0 < (fullFieldCoinSpace m).length := fullFieldCoinSpace_length_pos m
  have hDenNeInt : ((fullFieldCoinSpace m).length : Int) ≠ 0 := by
    exact_mod_cast (Nat.ne_of_gt hDenPosNat)
  unfold fullFieldCoinPr
  simpa [Rat.divInt_eq_div, hDenNeInt] using
    (Rat.div_mul_cancel
      (a := ((fullFieldCoinEventCount m E : Int) : Rat))
      (b := (((fullFieldCoinSpace m).length : Int) : Rat))
      (by exact_mod_cast (Nat.ne_of_gt hDenPosNat)))

theorem fullFieldCoinPr_mul_den_nat
  (m : Nat)
  (E : Array F → Prop) :
  fullFieldCoinPr m E * ((fullFieldCoinSpace m).length : Rat) =
    (fullFieldCoinEventCount m E : Rat) := by
  simpa using fullFieldCoinPr_mul_den m E

/--
Count-scaled bound transfer for the concrete full-field probability model.

If `count(E) * k ≤ d * |coinSpace|`, then
`Pr(E) * k ≤ d`.
-/
theorem fullFieldCoinPr_mul_nat_le_of_countScaled
  (m : Nat)
  (E : Array F → Prop)
  (k d : Nat)
  (hScaled :
      fullFieldCoinEventCount m E * k ≤
        d * (fullFieldCoinSpace m).length) :
  fullFieldCoinPr m E * (k : Rat) ≤ (d : Rat) := by
  have hDenPosNat : 0 < (fullFieldCoinSpace m).length := fullFieldCoinSpace_length_pos m
  have hDenPosRat : 0 < ((fullFieldCoinSpace m).length : Rat) := by
    exact (Rat.natCast_pos).2 hDenPosNat
  have hScaledRat :
      (fullFieldCoinEventCount m E : Rat) * (k : Rat) ≤
        (d : Rat) * ((fullFieldCoinSpace m).length : Rat) := by
    have hScaledNatRat :
        ((fullFieldCoinEventCount m E * k : Nat) : Rat) ≤
          ((d * (fullFieldCoinSpace m).length : Nat) : Rat) := by
      exact_mod_cast hScaled
    simpa [Rat.natCast_mul, Rat.mul_assoc, Rat.mul_comm] using hScaledNatRat
  have hMul :
      (fullFieldCoinPr m E * (k : Rat)) * ((fullFieldCoinSpace m).length : Rat) ≤
        (d : Rat) * ((fullFieldCoinSpace m).length : Rat) := by
    calc
      (fullFieldCoinPr m E * (k : Rat)) * ((fullFieldCoinSpace m).length : Rat)
          = (fullFieldCoinEventCount m E : Rat) * (k : Rat) := by
              calc
                (fullFieldCoinPr m E * (k : Rat)) * ((fullFieldCoinSpace m).length : Rat)
                    = fullFieldCoinPr m E * ((fullFieldCoinSpace m).length : Rat) * (k : Rat) := by
                        simp [Rat.mul_assoc, Rat.mul_comm]
                _ = (fullFieldCoinEventCount m E : Rat) * (k : Rat) := by
                      simp [fullFieldCoinPr_mul_den_nat, Rat.mul_assoc, Rat.mul_comm]
      _ ≤ (d : Rat) * ((fullFieldCoinSpace m).length : Rat) := hScaledRat
  exact Rat.le_of_mul_le_mul_right hMul hDenPosRat

/--
Reverse count-scaled transfer for the concrete full-field probability model.

If `Pr(E) * k ≤ d`, then
`count(E) * k ≤ d * |coinSpace|`.
-/
theorem fullFieldCoinEventCount_scaled_of_pr_mul_nat_le
  (m : Nat)
  (E : Array F → Prop)
  (k d : Nat)
  (hPr : fullFieldCoinPr m E * (k : Rat) ≤ (d : Rat)) :
  fullFieldCoinEventCount m E * k ≤ d * (fullFieldCoinSpace m).length := by
  have hLenNonneg : 0 ≤ ((fullFieldCoinSpace m).length : Rat) := by
    exact_mod_cast (Nat.zero_le (fullFieldCoinSpace m).length)
  have hMul :
      (fullFieldCoinPr m E * (k : Rat)) * ((fullFieldCoinSpace m).length : Rat) ≤
        (d : Rat) * ((fullFieldCoinSpace m).length : Rat) := by
    exact Rat.mul_le_mul_of_nonneg_right hPr hLenNonneg
  have hLeft :
      ((fullFieldCoinEventCount m E * k : Nat) : Rat) =
        (fullFieldCoinPr m E * (k : Rat)) * ((fullFieldCoinSpace m).length : Rat) := by
    calc
      ((fullFieldCoinEventCount m E * k : Nat) : Rat)
          = (fullFieldCoinEventCount m E : Rat) * (k : Rat) := by
              simp [Rat.natCast_mul]
      _ = (fullFieldCoinPr m E * ((fullFieldCoinSpace m).length : Rat)) * (k : Rat) := by
            simp [fullFieldCoinPr_mul_den_nat]
      _ = (fullFieldCoinPr m E * (k : Rat)) * ((fullFieldCoinSpace m).length : Rat) := by
            simp [Rat.mul_assoc, Rat.mul_comm]
  have hRight :
      (d : Rat) * ((fullFieldCoinSpace m).length : Rat) =
        ((d * (fullFieldCoinSpace m).length : Nat) : Rat) := by
    simp [Rat.natCast_mul]
  have hRat :
      ((fullFieldCoinEventCount m E * k : Nat) : Rat) ≤
        ((d * (fullFieldCoinSpace m).length : Nat) : Rat) := by
    calc
      ((fullFieldCoinEventCount m E * k : Nat) : Rat)
          = (fullFieldCoinPr m E * (k : Rat)) * ((fullFieldCoinSpace m).length : Rat) := hLeft
      _ ≤ (d : Rat) * ((fullFieldCoinSpace m).length : Rat) := hMul
      _ = ((d * (fullFieldCoinSpace m).length : Nat) : Rat) := hRight
  exact (Rat.natCast_le_natCast).1 hRat

theorem fullFieldCoinPr_mono
  (m : Nat)
  {E1 E2 : Array F → Prop}
  (hImp : ∀ coins, E1 coins → E2 coins) :
  fullFieldCoinPr m E1 ≤ fullFieldCoinPr m E2 := by
  have hDenPosNat : 0 < (fullFieldCoinSpace m).length := fullFieldCoinSpace_length_pos m
  have hDenPosRat : 0 < (((fullFieldCoinSpace m).length : Int) : Rat) := by
    exact_mod_cast hDenPosNat
  have hCountLeNat :
      fullFieldCoinEventCount m E1 ≤ fullFieldCoinEventCount m E2 :=
    fullFieldCoinEventCount_mono m hImp
  have hCountLeRat :
      (fullFieldCoinEventCount m E1 : Rat) ≤
        (fullFieldCoinEventCount m E2 : Rat) := by
    exact (Rat.natCast_le_natCast).2 hCountLeNat
  have hMul :
      fullFieldCoinPr m E1 * (((fullFieldCoinSpace m).length : Int) : Rat) ≤
        fullFieldCoinPr m E2 * (((fullFieldCoinSpace m).length : Int) : Rat) := by
    calc
      fullFieldCoinPr m E1 * (((fullFieldCoinSpace m).length : Int) : Rat)
          = (fullFieldCoinEventCount m E1 : Rat) := fullFieldCoinPr_mul_den m E1
      _ ≤ (fullFieldCoinEventCount m E2 : Rat) := hCountLeRat
      _ = fullFieldCoinPr m E2 * (((fullFieldCoinSpace m).length : Int) : Rat) := by
            symm
            exact fullFieldCoinPr_mul_den m E2
  exact Rat.le_of_mul_le_mul_right hMul hDenPosRat

theorem fullFieldCoinPr_union_le_add
  (m : Nat)
  (E1 E2 : Array F → Prop) :
  fullFieldCoinPr m (fun coins => E1 coins ∨ E2 coins) ≤
    fullFieldCoinPr m E1 + fullFieldCoinPr m E2 := by
  have hDenPosNat : 0 < (fullFieldCoinSpace m).length := fullFieldCoinSpace_length_pos m
  have hDenPosRat : 0 < (((fullFieldCoinSpace m).length : Int) : Rat) := by
    exact_mod_cast hDenPosNat
  have hCountUnionNat :
      fullFieldCoinEventCount m (fun coins => E1 coins ∨ E2 coins) ≤
        fullFieldCoinEventCount m E1 + fullFieldCoinEventCount m E2 :=
    fullFieldCoinEventCount_union_le m E1 E2
  have hCountUnionRat :
      (fullFieldCoinEventCount m (fun coins => E1 coins ∨ E2 coins) : Rat) ≤
        (fullFieldCoinEventCount m E1 : Rat) + (fullFieldCoinEventCount m E2 : Rat) := by
    have hCountUnionRatNat :
        (fullFieldCoinEventCount m (fun coins => E1 coins ∨ E2 coins) : Rat) ≤
          ((fullFieldCoinEventCount m E1 + fullFieldCoinEventCount m E2) : Rat) := by
      exact_mod_cast hCountUnionNat
    calc
      (fullFieldCoinEventCount m (fun coins => E1 coins ∨ E2 coins) : Rat)
          ≤ ((fullFieldCoinEventCount m E1 + fullFieldCoinEventCount m E2) : Rat) := hCountUnionRatNat
      _ = (fullFieldCoinEventCount m E1 : Rat) + (fullFieldCoinEventCount m E2 : Rat) := by
            simpa [Rat.natCast_add]
  have hMul :
      fullFieldCoinPr m (fun coins => E1 coins ∨ E2 coins) * (((fullFieldCoinSpace m).length : Int) : Rat) ≤
        (fullFieldCoinPr m E1 + fullFieldCoinPr m E2) * (((fullFieldCoinSpace m).length : Int) : Rat) := by
    calc
      (fullFieldCoinPr m (fun coins => E1 coins ∨ E2 coins)) * (((fullFieldCoinSpace m).length : Int) : Rat)
          = (fullFieldCoinEventCount m (fun coins => E1 coins ∨ E2 coins) : Rat) := by
              exact fullFieldCoinPr_mul_den m (fun coins => E1 coins ∨ E2 coins)
      _ ≤ (fullFieldCoinEventCount m E1 : Rat) + (fullFieldCoinEventCount m E2 : Rat) := hCountUnionRat
      _ = (fullFieldCoinPr m E1 + fullFieldCoinPr m E2) * (((fullFieldCoinSpace m).length : Int) : Rat) := by
            calc
              (fullFieldCoinEventCount m E1 : Rat) + (fullFieldCoinEventCount m E2 : Rat)
                  = fullFieldCoinPr m E1 * (((fullFieldCoinSpace m).length : Int) : Rat) +
                      fullFieldCoinPr m E2 * (((fullFieldCoinSpace m).length : Int) : Rat) := by
                        rw [fullFieldCoinPr_mul_den, fullFieldCoinPr_mul_den]
              _ = (fullFieldCoinPr m E1 + fullFieldCoinPr m E2) *
                    (((fullFieldCoinSpace m).length : Int) : Rat) := by
                  symm
                  exact Rat.add_mul
                    (fullFieldCoinPr m E1)
                    (fullFieldCoinPr m E2)
                    (((fullFieldCoinSpace m).length : Int) : Rat)
  exact Rat.le_of_mul_le_mul_right hMul hDenPosRat

/-- Concrete full-field product `CoinProbModel` for `m` verifier rounds. -/
noncomputable def fullFieldUniformCoinProbModel (m : Nat) : CoinProbModel where
  Pr := fullFieldCoinPr m
  prNonneg := fullFieldCoinPr_nonneg m
  prLeOne := fullFieldCoinPr_le_one m
  prFalse := fullFieldCoinPr_false m
  prMonotone := by
    intro E1 E2 hImp
    exact fullFieldCoinPr_mono m hImp
  prUnionLeAdd := by
    intro E1 E2
    exact fullFieldCoinPr_union_le_add m E1 E2

/--
Online (non-anticipatory) SumCheck prover strategy.

`roundPoly i coins` may only depend on the prefix `coins[0..i)`, captured by
`nonanticipatory`.
-/
structure OnlineProverStrategy (inst : Instance) where
  roundPoly : Nat → Array F → Array F
  roundPolyShape :
    ∀ i : Nat, i < inst.rounds →
      ∀ coins : Array F, (roundPoly i coins).size = inst.maxDegree + 1
  nonanticipatory :
    ∀ i : Nat, i < inst.rounds →
      ∀ {coins1 coins2 : Array F},
        (∀ j : Nat, j < i → coins1[j]! = coins2[j]!) →
          roundPoly i coins1 = roundPoly i coins2

end Sumcheck

end SuperNeo.ProofSystem
