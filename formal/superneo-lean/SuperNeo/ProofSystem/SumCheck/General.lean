import SuperNeo.ProofSystem.SumCheck.SingleRound
import SuperNeo.ProofSystem.Types
import SuperNeo.ProofSystem.Security
import SuperNeo.GoldilocksPrime
import Mathlib
import Init.Data.List.Lemmas
import Init.Data.Rat.Lemmas

namespace SuperNeo.ProofSystem

/- Paper-facing SumCheck namespace wrappers. -/
namespace Sumcheck

abbrev Instance := SuperNeo.SumCheckInstance
abbrev Transcript := SuperNeo.SumCheckTranscript

abbrev RoundConsistent := SuperNeo.SumCheckRoundConsistent
abbrev InitialRoundConsistent := SuperNeo.sumcheckInitialRoundConsistent
abbrev Accepted := SuperNeo.SumCheckAccepted
abbrev ClaimTrue := SuperNeo.SumCheckClaimTrue

abbrev SoundnessAssumption := SuperNeo.SumcheckSoundnessAssumption
abbrev CompletenessAssumption := SuperNeo.SumcheckCompletenessAssumption
abbrev Assumptions := SuperNeo.SumCheckAssumptions

/-- Soundness-failure event for a fixed instance/transcript. -/
def SoundnessFailureEvent (inst : Instance) (tr : Transcript) : Prop :=
  Accepted inst tr ∧ ¬ ClaimTrue inst

/-- Advantage of soundness failure under the ambient probability model. -/
def SoundnessFailureAdvantage
  (prob : ProbModel)
  (inst : Instance)
  (tr : Transcript) : Rat :=
  prob.Pr (SoundnessFailureEvent inst tr)

/--
Theorem-facing bound shape for soundness-failure advantage against a
security-parameter indexed error function.
-/
def SoundnessFailureAdvantageBound
  (inst : Instance)
  (tr : Transcript)
  (eps : ErrorFn) : Prop :=
  ∀ prob : ProbModel, ∀ n : Nat,
    SoundnessFailureAdvantage prob inst tr ≤ eps n

/-- Paper-facing SumCheck soundness bound `(ℓ·d, |K|)` for an instance. -/
def lundSchwartzZippelSoundnessBound (inst : Instance) : Nat × Nat :=
  SuperNeo.sumcheckLundSoundnessBound inst

/--
Probability model over verifier-coin events.

This is the event space needed for non-scaffold SumCheck soundness games:
events are predicates over sampled verifier challenge arrays.
-/
structure CoinProbModel where
  Pr : (Array F → Prop) → Rat
  prNonneg : ∀ E : Array F → Prop, 0 ≤ Pr E
  prLeOne : ∀ E : Array F → Prop, Pr E ≤ 1
  prFalse : Pr (fun _ => False) = 0
  prMonotone :
    ∀ {E1 E2 : Array F → Prop},
      (∀ coins, E1 coins → E2 coins) → Pr E1 ≤ Pr E2
  prUnionLeAdd :
    ∀ E1 E2 : Array F → Prop,
      Pr (fun coins => E1 coins ∨ E2 coins) ≤ Pr E1 + Pr E2

/--
Canonical full-field challenge domain for concrete SumCheck coin sampling.

This uses all field elements of `F = Fin Goldilocks.q`.
-/
def fullFieldChallengeDomain : List F :=
  List.finRange Goldilocks.q

/-- Full-field product coin space for `m` verifier rounds. -/
def fullFieldCoinSpace : Nat → List (Array F)
  | 0 => [#[]]
  | m + 1 =>
      (fullFieldCoinSpace m).flatMap (fun coins =>
        fullFieldChallengeDomain.map (fun r => coins.push r))

@[simp] theorem fullFieldChallengeDomain_length :
    fullFieldChallengeDomain.length = Goldilocks.q := by
  simp [fullFieldChallengeDomain]

/-- Canonical zero coin-vector used to witness non-emptiness of coin spaces. -/
def zeroCoins : Nat → Array F
  | 0 => #[]
  | m + 1 => (zeroCoins m).push 0

@[simp] theorem zeroCoins_size (m : Nat) : (zeroCoins m).size = m := by
  induction m with
  | zero =>
      simp [zeroCoins]
  | succ m ih =>
      simpa [zeroCoins, ih]

theorem mem_fullFieldCoinSpace_size
  {m : Nat}
  {coins : Array F}
  (hMem : coins ∈ fullFieldCoinSpace m) :
  coins.size = m := by
  induction m generalizing coins with
  | zero =>
      simp [fullFieldCoinSpace] at hMem
      rcases hMem with rfl
      simp
  | succ m ih =>
      simp [fullFieldCoinSpace] at hMem
      rcases hMem with ⟨base, hBaseMem, r, _hRMem, hEq⟩
      rcases hEq with rfl
      simpa [ih hBaseMem]

theorem zeroCoins_mem_fullFieldCoinSpace (m : Nat) :
  zeroCoins m ∈ fullFieldCoinSpace m := by
  induction m with
  | zero =>
      simp [fullFieldCoinSpace, zeroCoins]
  | succ m ih =>
      have hZeroMem : (0 : F) ∈ fullFieldChallengeDomain := by
        simpa [fullFieldChallengeDomain] using (List.mem_finRange (0 : F))
      apply List.mem_flatMap.mpr
      refine ⟨zeroCoins m, ih, ?_⟩
      apply List.mem_map.mpr
      refine ⟨(0 : F), hZeroMem, ?_⟩
      simp [zeroCoins]

theorem fullFieldCoinSpace_length_pos (m : Nat) :
  0 < (fullFieldCoinSpace m).length := by
  have hMem : zeroCoins m ∈ fullFieldCoinSpace m := zeroCoins_mem_fullFieldCoinSpace m
  have hNe : fullFieldCoinSpace m ≠ [] := by
    intro hNil
    simpa [hNil] using hMem
  exact (List.length_pos_iff).2 hNe

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

/--
Paper-style SumCheck soundness game:
- an externally fixed table witness,
- a false-claim condition against `inst.claimedValue`,
- an online (non-anticipatory) prover strategy.
-/
structure SoundnessGame where
  inst : Instance
  table : Array F
  tableSize : table.size = 2 ^ inst.rounds
  falseClaim : SuperNeo.sumcheckTableSum table ≠ inst.claimedValue
  prover : OnlineProverStrategy inst

/-- Build a transcript by running an online prover strategy on verifier coins. -/
def SoundnessGame.transcript (g : SoundnessGame) (coins : Array F) : Transcript :=
  { challenges := coins
    roundPolys := Array.ofFn (fun i : Fin g.inst.rounds =>
      g.prover.roundPoly i.1 coins) }

/-- Game acceptance event on a specific verifier-coin sample. -/
def SoundnessGame.acceptsOn (g : SoundnessGame) (coins : Array F) : Prop :=
  let tr := g.transcript coins
  SuperNeo.sumcheckAcceptedForTable g.inst g.table tr

/-- Soundness-failure event family over verifier coins. -/
def SoundnessGame.failureEvent (g : SoundnessGame) : Array F → Prop :=
  fun coins => g.acceptsOn coins

/--
Challenge-vector replacement helper: substitute coordinate `i` with `x` while
keeping all other challenge coordinates unchanged.
-/
def SoundnessGame.challengeWith
  (g : SoundnessGame)
  (coins : Array F)
  (i : Nat)
  (x : F) : Array F :=
  Array.ofFn (fun k : Fin coins.size =>
    if k.1 = i then x else coins[k.1]!)

@[simp] theorem SoundnessGame.challengeWith_size
  (g : SoundnessGame)
  (coins : Array F)
  (i : Nat)
  (x : F) :
  (g.challengeWith coins i x).size = coins.size := by
  simp [SoundnessGame.challengeWith]

theorem SoundnessGame.challengeWith_eq_self_of_lt
  (g : SoundnessGame)
  (coins : Array F)
  (i : Nat)
  (hi : i < coins.size) :
  g.challengeWith coins i (coins[i]!) = coins := by
  apply Array.ext
  · simp [SoundnessGame.challengeWith]
  · intro j hj1 hj2
    by_cases hji : j = i
    · subst hji
      simp [SoundnessGame.challengeWith, hi]
    · simp [SoundnessGame.challengeWith, hji]

theorem SoundnessGame.challengeWith_getElem!_eq
  (g : SoundnessGame)
  (coins : Array F)
  (i : Nat)
  (x : F)
  (hi : i < coins.size) :
  (g.challengeWith coins i x)[i]! = x := by
  rw [getElem!_pos (c := g.challengeWith coins i x) (i := i)]
  · simp [SoundnessGame.challengeWith]
  · simpa [SoundnessGame.challengeWith]

theorem SoundnessGame.challengeWith_getElem!_of_ne
  (g : SoundnessGame)
  (coins : Array F)
  (i j : Nat)
  (x : F)
  (hji : j ≠ i) :
  (g.challengeWith coins i x)[j]! = coins[j]! := by
  by_cases hj : j < coins.size
  · rw [getElem!_pos (c := g.challengeWith coins i x) (i := j)]
    · rw [getElem!_pos (c := coins) (i := j) hj]
      simp [SoundnessGame.challengeWith, hji]
    · simpa [SoundnessGame.challengeWith] using hj
  · have hOut : ¬ j < (g.challengeWith coins i x).size := by
      simpa [SoundnessGame.challengeWith] using hj
    rw [getElem!_neg (c := g.challengeWith coins i x) (i := j) (h := hOut)]
    rw [getElem!_neg (c := coins) (i := j) (h := hj)]

theorem SoundnessGame.challengeWith_overwrite
  (g : SoundnessGame)
  (coins : Array F)
  (i : Nat)
  (x y : F) :
  g.challengeWith (g.challengeWith coins i x) i y = g.challengeWith coins i y := by
  apply Array.ext
  · simp [SoundnessGame.challengeWith]
  · intro j hj1 hj2
    by_cases hji : j = i
    · subst hji
      simp [SoundnessGame.challengeWith]
    · have hLeftPos :
          (g.challengeWith (g.challengeWith coins i x) i y)[j]! =
            (g.challengeWith (g.challengeWith coins i x) i y)[j] :=
        getElem!_pos (c := g.challengeWith (g.challengeWith coins i x) i y) (i := j) hj1
      have hRightPos :
          (g.challengeWith coins i y)[j]! = (g.challengeWith coins i y)[j] :=
        getElem!_pos (c := g.challengeWith coins i y) (i := j) hj2
      calc
        (g.challengeWith (g.challengeWith coins i x) i y)[j]
            = (g.challengeWith (g.challengeWith coins i x) i y)[j]! := by
                simpa using hLeftPos.symm
        _ = (g.challengeWith coins i x)[j]! := by
              exact g.challengeWith_getElem!_of_ne (g.challengeWith coins i x) i j y hji
        _ = coins[j]! := by
              exact g.challengeWith_getElem!_of_ne coins i j x hji
        _ = (g.challengeWith coins i y)[j]! := by
              symm
              exact g.challengeWith_getElem!_of_ne coins i j y hji
        _ = (g.challengeWith coins i y)[j] := by
              simpa using hRightPos

/--
Canonical prefix weight for table index `j` at round `i` under verifier coins.

This keeps only the first `i` challenge coordinates and multiplies the Boolean
selector factors against the corresponding index bits.
-/
def SoundnessGame.prefixWeight
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (j : Nat) : F :=
  (List.range i).foldl
    (fun acc k =>
      acc * eqTerm ((bitsToFieldArray g.inst.rounds j)[k]!) (coins[k]!))
    1

/--
Table-induced canonical round target evaluator at round `i` and point `x`.

This is the pointwise sum of index contributions weighted by:
- prefix constraints on coordinates `< i`,
- the current coordinate selector at `i`.
-/
def SoundnessGame.roundTargetEval
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (x : F) : F :=
  mleByInnerProduct g.table (g.challengeWith coins i x)

theorem SoundnessGame.roundTargetEval_invariant_at_index
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (x y : F) :
  g.roundTargetEval i (g.challengeWith coins i y) x = g.roundTargetEval i coins x := by
  unfold SoundnessGame.roundTargetEval
  simp [SoundnessGame.challengeWith_overwrite]

/--
Canonical linear target polynomial coefficients for round `i`, truncated to the
configured degree shape `maxDegree + 1`.

For degrees `>= 2`, coefficients are zero.
-/
def SoundnessGame.roundTargetPoly
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) : Array F :=
  let v0 := g.roundTargetEval i coins 0
  let v1 := g.roundTargetEval i coins 1
  Array.ofFn (fun k : Fin (g.inst.maxDegree + 1) =>
    if h0 : k.1 = 0 then
      v0
    else if h1 : k.1 = 1 then
      v1 - v0
    else
      0)

theorem SoundnessGame.roundTargetPoly_invariant_at_index
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (x : F) :
  g.roundTargetPoly i (g.challengeWith coins i x) = g.roundTargetPoly i coins := by
  unfold SoundnessGame.roundTargetPoly
  have h0 :
      g.roundTargetEval i (g.challengeWith coins i x) 0 =
        g.roundTargetEval i coins 0 :=
    g.roundTargetEval_invariant_at_index i coins 0 x
  have h1 :
      g.roundTargetEval i (g.challengeWith coins i x) 1 =
        g.roundTargetEval i coins 1 :=
    g.roundTargetEval_invariant_at_index i coins 1 x
  simp [h0, h1]

/--
Canonical round witness polynomial:
`proverRoundPoly_i - canonicalTargetPoly_i`.

This is the algebraic object whose sampled root-event is used in the
Schwartz-Zippel/Lund soundness path.
-/
def SoundnessGame.roundPolyWitness
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) : Array F :=
  let p := g.prover.roundPoly i coins
  let q := g.roundTargetPoly i coins
  Array.ofFn (fun k : Fin (g.inst.maxDegree + 1) => p[k.1]! - q[k.1]!)

theorem SoundnessGame.roundPolyWitness_invariant_at_index
  (g : SoundnessGame)
  (i : Nat)
  (hi : i < g.inst.rounds)
  (coins : Array F)
  (x : F) :
  g.roundPolyWitness i (g.challengeWith coins i x) = g.roundPolyWitness i coins := by
  have hProver :
      g.prover.roundPoly i (g.challengeWith coins i x) = g.prover.roundPoly i coins := by
    apply g.prover.nonanticipatory i hi
    intro j hj
    exact g.challengeWith_getElem!_of_ne coins i j x (by omega)
  have hTarget :
      g.roundTargetPoly i (g.challengeWith coins i x) = g.roundTargetPoly i coins :=
    g.roundTargetPoly_invariant_at_index i coins x
  unfold SoundnessGame.roundPolyWitness
  simp [hProver, hTarget]

/--
Canonical per-round event: the round witness polynomial vanishes at the sampled
challenge coordinate.
-/
def SoundnessGame.roundFailureCanonical
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) : Prop :=
  i < g.inst.rounds ∧
    sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) =
      g.roundTargetEval i coins (coins[i]!)

@[simp] theorem SoundnessGame.roundTargetPoly_size
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) :
  (g.roundTargetPoly i coins).size = g.inst.maxDegree + 1 := by
  simp [SoundnessGame.roundTargetPoly]

@[simp] theorem SoundnessGame.roundPolyWitness_size
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) :
  (g.roundPolyWitness i coins).size = g.inst.maxDegree + 1 := by
  simp [SoundnessGame.roundPolyWitness]

@[simp] theorem SoundnessGame.roundTargetPoly_get_zero
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) :
  (g.roundTargetPoly i coins)[0]! = g.roundTargetEval i coins 0 := by
  have hZero : 0 < (g.roundTargetPoly i coins).size := by
    simpa using Nat.succ_pos g.inst.maxDegree
  simp [SoundnessGame.roundTargetPoly, hZero]

theorem SoundnessGame.roundTargetPoly_get_one
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (hDegPos : 0 < g.inst.maxDegree) :
  (g.roundTargetPoly i coins)[1]! =
    g.roundTargetEval i coins 1 - g.roundTargetEval i coins 0 := by
  have hOne' : 1 < g.inst.maxDegree + 1 := Nat.succ_lt_succ hDegPos
  have hOne : 1 < (g.roundTargetPoly i coins).size := by
    simpa [SoundnessGame.roundTargetPoly_size] using hOne'
  rw [getElem!_pos (c := g.roundTargetPoly i coins) (i := 1) hOne]
  simp [SoundnessGame.roundTargetPoly]

theorem SoundnessGame.roundTargetPoly_get_fin
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (k : Fin (g.inst.maxDegree + 1)) :
  (g.roundTargetPoly i coins)[k] =
    if k.1 = 0 then g.roundTargetEval i coins 0
    else if k.1 = 1 then g.roundTargetEval i coins 1 - g.roundTargetEval i coins 0
    else 0 := by
  simp [SoundnessGame.roundTargetPoly]

theorem SoundnessGame.roundTargetPoly_get_ge_two
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  {k : Nat}
  (hk : k < g.inst.maxDegree + 1)
  (hk2 : 2 ≤ k) :
  (g.roundTargetPoly i coins)[k]! = 0 := by
  have hPos : k < (g.roundTargetPoly i coins).size := by
    simpa [SoundnessGame.roundTargetPoly_size] using hk
  rw [getElem!_pos (c := g.roundTargetPoly i coins) (i := k) hPos]
  have hk0 : k ≠ 0 := by omega
  have hk1 : k ≠ 1 := by omega
  simp [SoundnessGame.roundTargetPoly, hk0, hk1]

theorem SoundnessGame.roundPolyWitness_get
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  {k : Nat}
  (hk : k < g.inst.maxDegree + 1) :
  (g.roundPolyWitness i coins)[k]! =
    (g.prover.roundPoly i coins)[k]! - (g.roundTargetPoly i coins)[k]! := by
  simp [SoundnessGame.roundPolyWitness, hk]

theorem SoundnessGame.roundPolyWitness_eval_zero_iff
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) :
  g.roundFailureCanonical i coins ↔
    i < g.inst.rounds ∧
      sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) =
        g.roundTargetEval i coins (coins[i]!) := by
  rfl

theorem SoundnessGame.roundFailureCanonical_implies_lt
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (h : g.roundFailureCanonical i coins) :
  i < g.inst.rounds := h.1

theorem SoundnessGame.roundFailureCanonical_implies_root
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (h : g.roundFailureCanonical i coins) :
  sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) =
    g.roundTargetEval i coins (coins[i]!) := h.2

theorem SoundnessGame.roundFailureCanonical_implies_sub_eq_zero
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (h : g.roundFailureCanonical i coins) :
  sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) -
      g.roundTargetEval i coins (coins[i]!) = 0 := by
  exact sub_eq_zero.mpr h.2

theorem SoundnessGame.roundFailureCanonical_of_sub_eq_zero
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (hi : i < g.inst.rounds)
  (hSub :
    sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) -
      g.roundTargetEval i coins (coins[i]!) = 0) :
  g.roundFailureCanonical i coins := by
  exact ⟨hi, sub_eq_zero.mp hSub⟩

theorem SoundnessGame.roundFailureCanonical_iff_sub_eq_zero
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F) :
  g.roundFailureCanonical i coins ↔
    i < g.inst.rounds ∧
      sumcheckEvalPoly (g.prover.roundPoly i coins) (coins[i]!) -
        g.roundTargetEval i coins (coins[i]!) = 0 := by
  constructor
  · intro h
    exact ⟨h.1, sub_eq_zero.mpr h.2⟩
  · intro h
    exact ⟨h.1, sub_eq_zero.mp h.2⟩

theorem SoundnessGame.roundTargetEval_at_challenge_eq_mleByInnerProduct
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (hi : i < coins.size) :
  g.roundTargetEval i coins (coins[i]!) = mleByInnerProduct g.table coins := by
  unfold SoundnessGame.roundTargetEval
  simpa [SoundnessGame.challengeWith_eq_self_of_lt (g := g) (coins := coins) (i := i) hi]

theorem SoundnessGame.roundTargetEval_at_challenge_eq_mleByFolding
  (g : SoundnessGame)
  (i : Nat)
  (coins : Array F)
  (hi : i < coins.size)
  (hSize : coins.size = g.inst.rounds) :
  g.roundTargetEval i coins (coins[i]!) = mleByFolding g.table coins := by
  have hInner :
      g.roundTargetEval i coins (coins[i]!) = mleByInnerProduct g.table coins :=
    g.roundTargetEval_at_challenge_eq_mleByInnerProduct i coins hi
  have hTableSize :
      g.table.size = 2 ^ coins.size := by
    simpa [hSize] using g.tableSize
  have hBridge :
      mleByInnerProduct g.table coins = mleByFolding g.table coins :=
    mleByInnerProduct_eq_mleByFolding_of_size (v := g.table) (r := coins) hTableSize
  exact hInner.trans hBridge

theorem SoundnessGame.roundFailureCanonical_last_of_failureEvent
  (g : SoundnessGame)
  (coins : Array F)
  (hFail : g.failureEvent coins)
  (hRoundsPos : 0 < g.inst.rounds) :
  g.roundFailureCanonical (g.inst.rounds - 1) coins := by
  have hAccepted :
      SuperNeo.sumcheckAcceptedForTable g.inst g.table (g.transcript coins) := by
    simpa [SoundnessGame.failureEvent, SoundnessGame.acceptsOn] using hFail
  have hCore : SuperNeo.sumcheckAcceptedCore g.inst (g.transcript coins) := hAccepted.1
  have hRoundCons : SuperNeo.sumcheckRoundConsistent g.inst (g.transcript coins) := hCore.2.2.1
  have hChSize : (g.transcript coins).challenges.size = g.inst.rounds := hRoundCons.1
  have hRpSize : (g.transcript coins).roundPolys.size = g.inst.rounds := hRoundCons.2
  have hCoinsSize : coins.size = g.inst.rounds := by
    simpa [SoundnessGame.transcript] using hChSize
  have hRoundsNe : g.inst.rounds ≠ 0 := Nat.ne_of_gt hRoundsPos
  have hLastLtRounds : g.inst.rounds - 1 < g.inst.rounds := by omega
  have hLastLtRp : g.inst.rounds - 1 < (g.transcript coins).roundPolys.size := by
    omega
  have hLastLtCoins : g.inst.rounds - 1 < coins.size := by
    omega
  have hFinalEval :
      sumcheckEvalPoly (g.transcript coins).roundPolys[g.inst.rounds - 1]!
        (g.transcript coins).challenges[g.inst.rounds - 1]! =
      mleByFolding g.table (g.transcript coins).challenges := by
    simpa [SuperNeo.sumcheckFinalOracleConsistentWithTable, hRoundsNe] using hAccepted.2.2.2
  have hPolyLast :
      (g.transcript coins).roundPolys[g.inst.rounds - 1]! =
        g.prover.roundPoly (g.inst.rounds - 1) coins := by
    simpa [SoundnessGame.transcript, hLastLtRounds]
  have hChallengeLast :
      (g.transcript coins).challenges[g.inst.rounds - 1]! =
        coins[g.inst.rounds - 1]! := by
    simp [SoundnessGame.transcript, hLastLtCoins]
  have hProverEval :
      sumcheckEvalPoly (g.prover.roundPoly (g.inst.rounds - 1) coins)
          (coins[g.inst.rounds - 1]!) =
        mleByFolding g.table coins := by
    calc
      sumcheckEvalPoly (g.prover.roundPoly (g.inst.rounds - 1) coins)
          (coins[g.inst.rounds - 1]!)
          = sumcheckEvalPoly (g.transcript coins).roundPolys[g.inst.rounds - 1]!
              (g.transcript coins).challenges[g.inst.rounds - 1]! := by
                simp [hPolyLast, hChallengeLast]
      _ = mleByFolding g.table (g.transcript coins).challenges := hFinalEval
      _ = mleByFolding g.table coins := by simp [SoundnessGame.transcript]
  have hTargetEval :
      g.roundTargetEval (g.inst.rounds - 1) coins (coins[g.inst.rounds - 1]!) =
        mleByFolding g.table coins :=
    g.roundTargetEval_at_challenge_eq_mleByFolding
      (i := g.inst.rounds - 1) (coins := coins) hLastLtCoins hCoinsSize
  refine ⟨hLastLtRounds, ?_⟩
  exact hProverEval.trans hTargetEval.symm

/-- Soundness-failure advantage for a fixed game under a coin-probability model. -/
def SoundnessGame.advantage (prob : CoinProbModel) (g : SoundnessGame) : Rat :=
  prob.Pr g.failureEvent

theorem SoundnessGame.advantage_nonneg
  (prob : CoinProbModel) (g : SoundnessGame) :
  0 ≤ g.advantage prob := by
  exact prob.prNonneg g.failureEvent

theorem SoundnessGame.advantage_le_one
  (prob : CoinProbModel) (g : SoundnessGame) :
  g.advantage prob ≤ 1 := by
  exact prob.prLeOne g.failureEvent

/--
Cross-multiplied Lund/Schwartz-Zippel soundness bound shape:
`advantage * |K| ≤ ℓ·d`.

This avoids explicit division and remains well-defined for all `Nat` parameters.
-/
def SoundnessGame.lundBoundHolds (prob : CoinProbModel) (g : SoundnessGame) : Prop :=
  g.advantage prob * (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
    SuperNeo.sumcheckLundSoundnessNumerator g.inst

/--
Paper-facing boundary assumption for SumCheck soundness over the explicit game.

This is the non-scaffolded endpoint: probability is taken over verifier coins,
with fixed false claim and an adversarial prover strategy.
-/
def LundSoundnessAssumption : Prop :=
  ∀ (prob : CoinProbModel) (g : SoundnessGame), g.lundBoundHolds prob

/-- Finite union of round failure events over the first `n` rounds. -/
def roundFailureUnion (E : Nat → Prop) : Nat → Prop
  | 0 => False
  | n + 1 => roundFailureUnion E n ∨ E n

/-- Finite sum of per-round error bounds over the first `n` rounds. -/
def roundErrorSum (eps : Nat → Rat) : Nat → Rat
  | 0 => 0
  | n + 1 => roundErrorSum eps n + eps n

/-- Finite union of round-failure events over verifier coins. -/
def roundFailureUnionCoins (E : Nat → Array F → Prop) : Nat → (Array F → Prop)
  | 0 => fun _ => False
  | n + 1 => fun coins => roundFailureUnionCoins E n coins ∨ E n coins

theorem roundFailureUnionCoins_of_mem
  {E : Nat → Array F → Prop}
  {n i : Nat}
  {coins : Array F}
  (hi : i < n)
  (hEi : E i coins) :
  roundFailureUnionCoins E n coins := by
  induction n generalizing i with
  | zero =>
      exact (Nat.not_lt_zero _ hi).elim
  | succ n ih =>
      by_cases hEq : i = n
      · subst hEq
        simpa [roundFailureUnionCoins] using Or.inr hEi
      · have hi' : i < n := by omega
        exact Or.inl (ih hi' hEi)

theorem SoundnessGame.failureEvent_covered_by_roundFailureCanonical_of_rounds_pos
  (g : SoundnessGame)
  (coins : Array F)
  (hFail : g.failureEvent coins)
  (hRoundsPos : 0 < g.inst.rounds) :
  roundFailureUnionCoins g.roundFailureCanonical g.inst.rounds coins := by
  have hLast : g.roundFailureCanonical (g.inst.rounds - 1) coins :=
    g.roundFailureCanonical_last_of_failureEvent coins hFail hRoundsPos
  have hLastLt : g.inst.rounds - 1 < g.inst.rounds := by omega
  exact roundFailureUnionCoins_of_mem hLastLt hLast

theorem pr_roundFailureUnionCoins_le_roundErrorSum
  (prob : CoinProbModel)
  (E : Nat → Array F → Prop)
  (eps : Nat → Rat)
  (n : Nat)
  (hBound : ∀ i : Nat, i < n → prob.Pr (E i) ≤ eps i) :
  prob.Pr (roundFailureUnionCoins E n) ≤ roundErrorSum eps n := by
  induction n with
  | zero =>
      simpa [roundFailureUnionCoins, roundErrorSum, prob.prFalse] using (Rat.le_refl : (0 : Rat) ≤ 0)
  | succ n ih =>
      have hBoundPrev : ∀ i : Nat, i < n → prob.Pr (E i) ≤ eps i := by
        intro i hi
        exact hBound i (Nat.lt_trans hi (Nat.lt_succ_self n))
      have hBoundN : prob.Pr (E n) ≤ eps n := hBound n (Nat.lt_succ_self n)
      have hAddPrev :
          prob.Pr (roundFailureUnionCoins E n) + prob.Pr (E n) ≤
            roundErrorSum eps n + prob.Pr (E n) := by
        exact (Rat.add_le_add_right (c := prob.Pr (E n))).2 (ih hBoundPrev)
      have hAddLast :
          roundErrorSum eps n + prob.Pr (E n) ≤
            roundErrorSum eps n + eps n := by
        exact (Rat.add_le_add_left (c := roundErrorSum eps n)).2 hBoundN
      calc
        prob.Pr (roundFailureUnionCoins E (n + 1))
            = prob.Pr (fun coins => roundFailureUnionCoins E n coins ∨ E n coins) := by
                simp [roundFailureUnionCoins]
        _ ≤ prob.Pr (roundFailureUnionCoins E n) + prob.Pr (E n) := prob.prUnionLeAdd _ _
        _ ≤ roundErrorSum eps n + prob.Pr (E n) := hAddPrev
        _ ≤ roundErrorSum eps n + eps n := hAddLast
        _ = roundErrorSum eps (n + 1) := by
              simp [roundErrorSum]

/--
Round-by-round boundary sufficient to derive the Lund-style soundness bound.

This isolates the remaining closure work to:
1) constructing round-failure events that cover global failure,
2) proving per-round probability bounds (typically via Schwartz-Zippel),
3) proving the final accumulated-error inequality.
-/
structure LundRoundBoundary
  (prob : CoinProbModel)
  (g : SoundnessGame) where
  roundFailure : Nat → Array F → Prop
  epsRound : Nat → Rat
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundBound :
    ∀ i : Nat, i < g.inst.rounds → prob.Pr (roundFailure i) ≤ epsRound i
  totalBound :
    roundErrorSum epsRound g.inst.rounds *
      (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
        SuperNeo.sumcheckLundSoundnessNumerator g.inst

theorem SoundnessGame.lundBoundHolds_of_roundBoundary
  (prob : CoinProbModel)
  (g : SoundnessGame)
  (hRbr : LundRoundBoundary prob g) :
  g.lundBoundHolds prob := by
  unfold SoundnessGame.lundBoundHolds SoundnessGame.advantage
  let dRat : Rat := (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat)
  have hdNonnegCast :
      0 ≤ (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat) := by
    exact_mod_cast (Nat.zero_le (SuperNeo.sumcheckLundSoundnessDenominator g.inst))
  have hdNonneg : 0 ≤ dRat := by
    simpa [dRat] using hdNonnegCast
  have hCover :
      prob.Pr g.failureEvent ≤
        prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) := by
    exact prob.prMonotone hRbr.covered
  have hUnion :
      prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) ≤
        roundErrorSum hRbr.epsRound g.inst.rounds := by
    exact pr_roundFailureUnionCoins_le_roundErrorSum
      prob hRbr.roundFailure hRbr.epsRound g.inst.rounds hRbr.roundBound
  have hMul1 : prob.Pr g.failureEvent * dRat ≤
      prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat := by
    exact Rat.mul_le_mul_of_nonneg_right hCover hdNonneg
  have hMul2 :
      prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat ≤
        roundErrorSum hRbr.epsRound g.inst.rounds * dRat := by
    exact Rat.mul_le_mul_of_nonneg_right hUnion hdNonneg
  calc
    prob.Pr g.failureEvent * (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
        = prob.Pr g.failureEvent * dRat := by simp [dRat]
    _ ≤ prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat := hMul1
    _ ≤ roundErrorSum hRbr.epsRound g.inst.rounds * dRat := hMul2
    _ = roundErrorSum hRbr.epsRound g.inst.rounds *
          (SuperNeo.sumcheckLundSoundnessDenominator g.inst) := by simp [dRat]
    _ ≤ SuperNeo.sumcheckLundSoundnessNumerator g.inst := hRbr.totalBound

/--
Theorem-native closure surface for Lund soundness:
constructing round-by-round boundaries for every game suffices to prove the
global `LundSoundnessAssumption`.
-/
def LundRoundBoundaryAssumption : Prop :=
  ∀ (prob : CoinProbModel) (g : SoundnessGame), Nonempty (LundRoundBoundary prob g)

theorem lundSoundnessAssumption_of_roundBoundary
  (hRound : LundRoundBoundaryAssumption) :
  LundSoundnessAssumption := by
  intro prob g
  rcases hRound prob g with ⟨hRbr⟩
  exact SoundnessGame.lundBoundHolds_of_roundBoundary prob g hRbr

/--
Cross-multiplied union-bound helper:
if every per-round event satisfies `Pr(E_i) * d ≤ k`, then the finite union up to
`n` rounds satisfies `Pr(⋃_{i<n} E_i) * d ≤ n * k`.
-/
theorem pr_roundFailureUnionCoins_mul_le_const
  (prob : CoinProbModel)
  (E : Nat → Array F → Prop)
  (n : Nat)
  (d k : Rat)
  (hdNonneg : 0 ≤ d)
  (hBound : ∀ i : Nat, i < n → prob.Pr (E i) * d ≤ k) :
  prob.Pr (roundFailureUnionCoins E n) * d ≤ (n : Rat) * k := by
  induction n with
  | zero =>
      simpa [roundFailureUnionCoins, prob.prFalse]
        using (Rat.le_refl : (0 : Rat) ≤ 0)
  | succ n ih =>
      have hBoundPrev : ∀ i : Nat, i < n → prob.Pr (E i) * d ≤ k := by
        intro i hi
        exact hBound i (Nat.lt_trans hi (Nat.lt_succ_self n))
      have hBoundN : prob.Pr (E n) * d ≤ k := hBound n (Nat.lt_succ_self n)
      have hUnion :
          prob.Pr (roundFailureUnionCoins E (n + 1)) ≤
            prob.Pr (roundFailureUnionCoins E n) + prob.Pr (E n) := by
        simpa [roundFailureUnionCoins] using
          (prob.prUnionLeAdd (roundFailureUnionCoins E n) (E n))
      have hMulUnion :
          prob.Pr (roundFailureUnionCoins E (n + 1)) * d ≤
            (prob.Pr (roundFailureUnionCoins E n) + prob.Pr (E n)) * d := by
        exact Rat.mul_le_mul_of_nonneg_right hUnion hdNonneg
      have hAddPrev :
          prob.Pr (roundFailureUnionCoins E n) * d + prob.Pr (E n) * d ≤
            (n : Rat) * k + prob.Pr (E n) * d := by
        exact (Rat.add_le_add_right (c := prob.Pr (E n) * d)).2 (ih hBoundPrev)
      have hAddLast :
          (n : Rat) * k + prob.Pr (E n) * d ≤
            (n : Rat) * k + k := by
        exact (Rat.add_le_add_left (c := (n : Rat) * k)).2 hBoundN
      calc
        prob.Pr (roundFailureUnionCoins E (n + 1)) * d
            ≤ (prob.Pr (roundFailureUnionCoins E n) + prob.Pr (E n)) * d := hMulUnion
        _ = prob.Pr (roundFailureUnionCoins E n) * d + prob.Pr (E n) * d := by
              simpa using
                (Rat.add_mul (prob.Pr (roundFailureUnionCoins E n)) (prob.Pr (E n)) d)
        _ ≤ (n : Rat) * k + prob.Pr (E n) * d := hAddPrev
        _ ≤ (n : Rat) * k + k := hAddLast
        _ = ((n : Rat) + 1) * k := by
              calc
                (n : Rat) * k + k = (n : Rat) * k + 1 * k := by simp [Rat.one_mul]
                _ = ((n : Rat) + 1) * k := by
                      simpa [Rat.one_mul, Rat.add_comm, Rat.add_left_comm, Rat.add_assoc] using
                        (Rat.add_mul (n : Rat) 1 k).symm
        _ = ((n + 1 : Nat) : Rat) * k := by simp

/--
Cross-multiplied round-bound package (Schwartz-Zippel style):
per-round event bounds are stated directly as `Pr(E_i) * |K| ≤ d`.
-/
structure LundRoundBoundaryScaled
  (prob : CoinProbModel)
  (g : SoundnessGame) where
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundBoundScaled :
    ∀ i : Nat, i < g.inst.rounds →
      prob.Pr (roundFailure i) * (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
        (g.inst.maxDegree : Rat)

/--
Lower-level round-event kernel (Schwartz-Zippel style):
each round carries an explicit root-budget witness `d_i`, with
`d_i ≤ maxDegree` and cross-multiplied bound
`Pr(E_i) * |K| ≤ d_i`.
-/
structure LundRoundKernel
  (prob : CoinProbModel)
  (g : SoundnessGame) where
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundRootBudget : Nat → Nat
  roundRootBudgetBound :
    ∀ i : Nat, i < g.inst.rounds →
      roundRootBudget i ≤ g.inst.maxDegree
  roundProbBound :
    ∀ i : Nat, i < g.inst.rounds →
      prob.Pr (roundFailure i) * (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
        (roundRootBudget i : Rat)

/--
Kernel-to-boundary lift:
a Schwartz-Zippel round kernel induces the scaled Lund round boundary.
-/
def LundRoundBoundaryScaled.of_kernel
  (prob : CoinProbModel)
  (g : SoundnessGame)
  (hK : LundRoundKernel prob g) :
  LundRoundBoundaryScaled prob g := by
  refine
    { roundFailure := hK.roundFailure
      covered := hK.covered
      roundBoundScaled := ?_ }
  intro i hi
  have hProb : prob.Pr (hK.roundFailure i) * (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
      (hK.roundRootBudget i : Rat) := hK.roundProbBound i hi
  have hBudgetNat : hK.roundRootBudget i ≤ g.inst.maxDegree := hK.roundRootBudgetBound i hi
  have hBudget : (hK.roundRootBudget i : Rat) ≤ (g.inst.maxDegree : Rat) := by
    exact_mod_cast hBudgetNat
  exact Rat.le_trans hProb hBudget

theorem SoundnessGame.lundBoundHolds_of_scaledRoundBoundary
  (prob : CoinProbModel)
  (g : SoundnessGame)
  (hRbr : LundRoundBoundaryScaled prob g) :
  g.lundBoundHolds prob := by
  unfold SoundnessGame.lundBoundHolds SoundnessGame.advantage
  let dRat : Rat := (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat)
  have hdNonnegCast :
      0 ≤ (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat) := by
    exact_mod_cast (Nat.zero_le (SuperNeo.sumcheckLundSoundnessDenominator g.inst))
  have hdNonneg : 0 ≤ dRat := by
    simpa [dRat] using hdNonnegCast
  have hCover :
      prob.Pr g.failureEvent ≤
        prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) := by
    exact prob.prMonotone hRbr.covered
  have hCoverMul :
      prob.Pr g.failureEvent * dRat ≤
        prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat := by
    exact Rat.mul_le_mul_of_nonneg_right hCover hdNonneg
  have hRoundBound :
      ∀ i : Nat, i < g.inst.rounds →
        prob.Pr (hRbr.roundFailure i) * dRat ≤ (g.inst.maxDegree : Rat) := by
    intro i hi
    simpa [dRat] using hRbr.roundBoundScaled i hi
  have hUnionMul :
      prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat ≤
        (g.inst.rounds : Rat) * (g.inst.maxDegree : Rat) := by
    simpa [dRat] using
      (pr_roundFailureUnionCoins_mul_le_const
        prob hRbr.roundFailure g.inst.rounds dRat (g.inst.maxDegree : Rat) hdNonneg hRoundBound)
  calc
    prob.Pr g.failureEvent * (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
        = prob.Pr g.failureEvent * dRat := by simp [dRat]
    _ ≤ prob.Pr (roundFailureUnionCoins hRbr.roundFailure g.inst.rounds) * dRat := hCoverMul
    _ ≤ (g.inst.rounds : Rat) * (g.inst.maxDegree : Rat) := hUnionMul
    _ = SuperNeo.sumcheckLundSoundnessNumerator g.inst := by
          simp [SuperNeo.sumcheckLundSoundnessNumerator]

/--
Theorem-native closure surface using cross-multiplied per-round bounds.
-/
def LundRoundScaledBoundaryAssumption : Prop :=
  ∀ (prob : CoinProbModel) (g : SoundnessGame), Nonempty (LundRoundBoundaryScaled prob g)

theorem lundSoundnessAssumption_of_scaledRoundBoundary
  (hRound : LundRoundScaledBoundaryAssumption) :
  LundSoundnessAssumption := by
  intro prob g
  rcases hRound prob g with ⟨hRbr⟩
  exact SoundnessGame.lundBoundHolds_of_scaledRoundBoundary prob g hRbr

/--
Kernel-level assumption surface:
for every game, lower-level round-event lemmas produce a Schwartz-Zippel kernel.
-/
def LundRoundKernelAssumption : Prop :=
  ∀ (prob : CoinProbModel) (g : SoundnessGame), Nonempty (LundRoundKernel prob g)

theorem lundRoundScaledBoundaryAssumption_of_kernel
  (hKernel : LundRoundKernelAssumption) :
  LundRoundScaledBoundaryAssumption := by
  intro prob g
  rcases hKernel prob g with ⟨hK⟩
  exact ⟨LundRoundBoundaryScaled.of_kernel prob g hK⟩

theorem lundSoundnessAssumption_of_kernel
  (hKernel : LundRoundKernelAssumption) :
  LundSoundnessAssumption := by
  exact lundSoundnessAssumption_of_scaledRoundBoundary
    (lundRoundScaledBoundaryAssumption_of_kernel hKernel)

/--
Lower-level Schwartz-Zippel/round-event lemma package for a fixed game.

This is the intended theorem-native input surface from lower algebraic/probabilistic
proofs: a concrete round-event family, event coverage of global failure, and
cross-multiplied per-round bounds with explicit root budgets.
-/
structure SchwartzZippelRoundEventLemmas
  (prob : CoinProbModel)
  (g : SoundnessGame) where
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundRootBudget : Nat → Nat
  roundRootBudgetBound :
    ∀ i : Nat, i < g.inst.rounds →
      roundRootBudget i ≤ g.inst.maxDegree
  roundProbBoundScaled :
    ∀ i : Nat, i < g.inst.rounds →
      prob.Pr (roundFailure i) * (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
        (roundRootBudget i : Rat)

/--
Canonical cross-multiplied per-round bound surface for `roundFailureCanonical`.
-/
def CanonicalRoundBoundScaled
  (prob : CoinProbModel)
  (g : SoundnessGame) : Prop :=
  ∀ i : Nat, i < g.inst.rounds →
    prob.Pr (g.roundFailureCanonical i) *
      (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
        (g.inst.maxDegree : Rat)

/--
Constructive instantiation of `SchwartzZippelRoundEventLemmas` from the
canonical event family plus canonical per-round scaled bounds.

This theorem closes the event-packaging step once per-round bounds are available.
-/
def SchwartzZippelRoundEventLemmas.of_canonicalRoundBoundScaled
  (prob : CoinProbModel)
  (g : SoundnessGame)
  (hRoundsPos : 0 < g.inst.rounds)
  (hBound : CanonicalRoundBoundScaled prob g) :
  SchwartzZippelRoundEventLemmas prob g := by
  refine
    { roundFailure := g.roundFailureCanonical
      covered := ?_
      roundRootBudget := fun _ => g.inst.maxDegree
      roundRootBudgetBound := ?_
      roundProbBoundScaled := ?_ }
  · intro coins hFail
    exact g.failureEvent_covered_by_roundFailureCanonical_of_rounds_pos coins hFail hRoundsPos
  · intro i hi
    exact Nat.le_refl _
  · intro i hi
    simpa using hBound i hi

/--
Canonical lower-level SZ round-event package for the full-field model.

This is a named wrapper around
`SchwartzZippelRoundEventLemmas.of_canonicalRoundBoundScaled` used by the
aligned/positive-round closure chain below.
-/
noncomputable def canonicalRoundSzLemmas
  (g : SoundnessGame)
  (hRoundsPos : 0 < g.inst.rounds)
  (hBound : CanonicalRoundBoundScaled (fullFieldUniformCoinProbModel g.inst.rounds) g) :
  SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g :=
  SchwartzZippelRoundEventLemmas.of_canonicalRoundBoundScaled
    (prob := fullFieldUniformCoinProbModel g.inst.rounds)
    (g := g)
    hRoundsPos
    hBound

/-- Build a `LundRoundKernel` directly from lower-level Schwartz-Zippel lemmas. -/
def LundRoundKernel.of_schwartzZippelRoundEventLemmas
  (prob : CoinProbModel)
  (g : SoundnessGame)
  (hSz : SchwartzZippelRoundEventLemmas prob g) :
  LundRoundKernel prob g :=
  { roundFailure := hSz.roundFailure
    covered := hSz.covered
    roundRootBudget := hSz.roundRootBudget
    roundRootBudgetBound := hSz.roundRootBudgetBound
    roundProbBound := hSz.roundProbBoundScaled }

/--
Global closure surface: for every game, lower-level Schwartz-Zippel round-event
lemmas are available.
-/
def SchwartzZippelRoundEventAssumption : Prop :=
  ∀ (prob : CoinProbModel) (g : SoundnessGame),
    Nonempty (SchwartzZippelRoundEventLemmas prob g)

theorem lundRoundKernelAssumption_of_schwartzZippelRoundEvent
  (hSz : SchwartzZippelRoundEventAssumption) :
  LundRoundKernelAssumption := by
  intro prob g
  rcases hSz prob g with ⟨hSzGame⟩
  exact ⟨LundRoundKernel.of_schwartzZippelRoundEventLemmas prob g hSzGame⟩

theorem lundRoundScaledBoundaryAssumption_of_schwartzZippelRoundEvent
  (hSz : SchwartzZippelRoundEventAssumption) :
  LundRoundScaledBoundaryAssumption := by
  exact lundRoundScaledBoundaryAssumption_of_kernel
    (lundRoundKernelAssumption_of_schwartzZippelRoundEvent hSz)

theorem lundSoundnessAssumption_of_schwartzZippelRoundEvent
  (hSz : SchwartzZippelRoundEventAssumption) :
  LundSoundnessAssumption := by
  exact lundSoundnessAssumption_of_kernel
    (lundRoundKernelAssumption_of_schwartzZippelRoundEvent hSz)

/--
Full-field lower-level round-event package:
for the canonical coin model `fullFieldUniformCoinProbModel rounds`, each round
provides a root-budget witness and a count-scaled inequality.
-/
structure FullFieldRoundEventCardinalityLemmas
  (g : SoundnessGame) where
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundRootBudget : Nat → Nat
  roundRootBudgetBound :
    ∀ i : Nat, i < g.inst.rounds →
      roundRootBudget i ≤ g.inst.maxDegree
  roundCountBoundScaled :
    ∀ i : Nat, i < g.inst.rounds →
      fullFieldCoinEventCount g.inst.rounds (roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
          roundRootBudget i * (fullFieldCoinSpace g.inst.rounds).length

/--
Lower-level full-field root-count package (paper-style shape):
for each round event, the event count over `F^ℓ` is bounded by
`dᵢ * |F|^(ℓ-1)`.

This is the direct finite-field root-count form typically produced by
Schwartz-Zippel style arguments before converting to probability-scaled bounds.
-/
structure FullFieldRoundEventRootCountLemmas
  (g : SoundnessGame) where
  domainAligned :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundRootBudget : Nat → Nat
  roundRootBudgetBound :
    ∀ i : Nat, i < g.inst.rounds →
      roundRootBudget i ≤ g.inst.maxDegree
  roundCountBoundPow :
    ∀ i : Nat, i < g.inst.rounds →
      fullFieldCoinEventCount g.inst.rounds (roundFailure i) ≤
        roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1)

/--
Full-field root count of a univariate coefficient-array polynomial over `F`.

Mathlib bridge:
- map `F = Fin q` values into `ZMod q`,
- build the corresponding univariate polynomial in `ZMod q[X]`,
- count roots by filtering all `F`-challenges through polynomial evaluation.
-/
abbrev Fq : Type := ZMod Goldilocks.q

/-- Canonical coercion from `F = Fin q` into `ZMod q`. -/
def fToZMod (a : F) : Fq :=
  (a.val : Fq)

theorem fToZMod_injective : Function.Injective fToZMod := by
  intro a b hEq
  apply Fin.ext
  have hMod :
      a.val % Goldilocks.q = b.val % Goldilocks.q :=
    (ZMod.natCast_eq_natCast_iff' a.val b.val Goldilocks.q).1 hEq
  simpa [Nat.mod_eq_of_lt a.isLt, Nat.mod_eq_of_lt b.isLt] using hMod

/-- Mathlib polynomial corresponding to a coefficient array (low degree first). -/
def sumcheckPolynomialZMod (poly : Array F) : Polynomial Fq :=
  Polynomial.ofFn poly.size (fun i => fToZMod (poly[i.1]!))

/--
Polynomial that vanishes exactly on a chosen finite set of field points
(up to root multiplicity one): `∏_{r∈S} (X - r)`.
-/
noncomputable def rootVanishingPoly (S : Finset F) : Polynomial Fq :=
  S.prod (fun r => (Polynomial.X - Polynomial.C (fToZMod r)))

theorem rootVanishingPoly_eval_eq_zero_of_mem
    {S : Finset F} {r : F}
    (hr : r ∈ S) :
    (rootVanishingPoly S).eval (fToZMod r) = 0 := by
  classical
  induction S using Finset.induction_on with
  | empty =>
      cases hr
  | @insert a S ha ih =>
      simp [rootVanishingPoly, Finset.prod_insert, ha] at hr ⊢
      rcases hr with rfl | hr'
      · simp
      · right
        simpa [rootVanishingPoly] using ih hr'

theorem rootVanishingPoly_natDegree_eq_card
    (S : Finset F) :
    (rootVanishingPoly S).natDegree = S.card := by
  classical
  unfold rootVanishingPoly
  simpa using
    (Polynomial.natDegree_finset_prod_X_sub_C_eq_card
      (s := S) (f := fun r : F => fToZMod r))

theorem rootVanishingPoly_ne_zero
    (S : Finset F) :
    rootVanishingPoly S ≠ 0 := by
  classical
  have hMonic : (rootVanishingPoly S).Monic := by
    unfold rootVanishingPoly
    simpa using
      (Polynomial.monic_prod_X_sub_C (s := S) (b := fun r : F => fToZMod r))
  exact hMonic.ne_zero

/-- Canonical conversion from `ZMod q` into `F = Fin q`. -/
noncomputable def zmodToF (z : Fq) : F :=
  ⟨z.val, z.val_lt⟩

theorem fToZMod_zmodToF (z : Fq) : fToZMod (zmodToF z) = z := by
  unfold fToZMod zmodToF
  simp

/--
Truncate/pad a `ZMod q[X]` polynomial into exactly `n` coefficients in `F`.
This is the executable coefficient-array surface used by the SumCheck bridge.
-/
noncomputable def zmodPolyToCoeffArray (n : Nat) (p : Polynomial Fq) : Array F :=
  Array.ofFn (fun i : Fin n => zmodToF ((Polynomial.toFn n p) i))

@[simp] theorem zmodPolyToCoeffArray_size (n : Nat) (p : Polynomial Fq) :
    (zmodPolyToCoeffArray n p).size = n := by
  simp [zmodPolyToCoeffArray]

/--
If a `ZMod q[X]` polynomial has degree `< n`, converting it to `n` coefficients
and back through `sumcheckPolynomialZMod` is exact.
-/
theorem sumcheckPolynomialZMod_zmodPolyToCoeffArray
    (n : Nat)
    (p : Polynomial Fq)
    (hdeg : p.natDegree < n) :
    sumcheckPolynomialZMod (zmodPolyToCoeffArray n p) = p := by
  let arr : Array F := zmodPolyToCoeffArray n p
  have hSize : arr.size = n := by
    simp [arr, zmodPolyToCoeffArray]
  unfold sumcheckPolynomialZMod
  rw [hSize]
  have hfun :
      (fun i : Fin n => fToZMod (arr[i.1]!)) = Polynomial.toFn n p := by
    funext i
    simp [arr, zmodPolyToCoeffArray, fToZMod_zmodToF]
  calc
    Polynomial.ofFn n (fun i : Fin n => fToZMod (arr[i.1]!))
      = Polynomial.ofFn n (Polynomial.toFn n p) := by
          simp [hfun]
    _ = p := Polynomial.ofFn_comp_toFn_eq_id_of_natDegree_lt hdeg

/-- Root count over the full finite challenge domain using the Mathlib polynomial bridge. -/
noncomputable def fullFieldPolyRootCount (poly : Array F) : Nat :=
  (Finset.univ.filter (fun r : F =>
      (sumcheckPolynomialZMod poly).eval (fToZMod r) = 0)).card

theorem sumcheckPolynomialZMod_natDegree_lt_size
    {poly : Array F}
    (hSizePos : 0 < poly.size) :
    (sumcheckPolynomialZMod poly).natDegree < poly.size := by
  unfold sumcheckPolynomialZMod
  have hOneLe : 1 ≤ poly.size := Nat.succ_le_of_lt hSizePos
  simpa using
    (Polynomial.ofFn_natDegree_lt (R := Fq) hOneLe
      (fun i => fToZMod (poly[i.1]!)))

/--
Mathlib root-count bridge:
for nonzero bridged polynomials, counted full-field roots are bounded by
the multiset-cardinality of roots.
-/
theorem fullFieldPolyRootCount_le_card_roots
    [Fact (Nat.Prime Goldilocks.q)]
    {poly : Array F}
    (hPolyNeZero : sumcheckPolynomialZMod poly ≠ 0) :
    fullFieldPolyRootCount poly ≤ (sumcheckPolynomialZMod poly).roots.card := by
  classical
  let rootsF : Finset F :=
    Finset.univ.filter (fun r : F =>
      (sumcheckPolynomialZMod poly).eval (fToZMod r) = 0)
  have hDef : fullFieldPolyRootCount poly = rootsF.card := by
    simp [fullFieldPolyRootCount, rootsF]
  have hImageSub :
      rootsF.image fToZMod ⊆ (sumcheckPolynomialZMod poly).roots.toFinset := by
    intro z hz
    rcases Finset.mem_image.mp hz with ⟨r, hrMem, rfl⟩
    have hrEval : (sumcheckPolynomialZMod poly).eval (fToZMod r) = 0 :=
      (Finset.mem_filter.mp hrMem).2
    have hrRoot : (sumcheckPolynomialZMod poly).IsRoot (fToZMod r) := by
      simpa [Polynomial.IsRoot] using hrEval
    exact Multiset.mem_toFinset.mpr ((Polynomial.mem_roots hPolyNeZero).2 hrRoot)
  have hCardImage :
      (rootsF.image fToZMod).card = rootsF.card :=
    Finset.card_image_of_injective rootsF fToZMod_injective
  calc
    fullFieldPolyRootCount poly = rootsF.card := hDef
    _ = (rootsF.image fToZMod).card := hCardImage.symm
    _ ≤ (sumcheckPolynomialZMod poly).roots.toFinset.card := Finset.card_le_card hImageSub
    _ ≤ (sumcheckPolynomialZMod poly).roots.card :=
      Multiset.toFinset_card_le (sumcheckPolynomialZMod poly).roots

theorem fullFieldPolyRootCount_le_natDegree_of_nonzero
    [Fact (Nat.Prime Goldilocks.q)]
    {poly : Array F}
    (hPolyNeZero : sumcheckPolynomialZMod poly ≠ 0) :
    fullFieldPolyRootCount poly ≤ (sumcheckPolynomialZMod poly).natDegree := by
  calc
    fullFieldPolyRootCount poly ≤ (sumcheckPolynomialZMod poly).roots.card :=
      fullFieldPolyRootCount_le_card_roots hPolyNeZero
    _ ≤ (sumcheckPolynomialZMod poly).natDegree :=
      Polynomial.card_roots' (sumcheckPolynomialZMod poly)

theorem fullFieldPolyRootCount_le_pred_size_of_nonzero
    [Fact (Nat.Prime Goldilocks.q)]
    {poly : Array F}
    (hPolyNeZero : sumcheckPolynomialZMod poly ≠ 0)
    (hSizePos : 0 < poly.size) :
    fullFieldPolyRootCount poly ≤ poly.size - 1 := by
  have hDegLt : (sumcheckPolynomialZMod poly).natDegree < poly.size :=
    sumcheckPolynomialZMod_natDegree_lt_size (poly := poly) hSizePos
  have hDegLe : (sumcheckPolynomialZMod poly).natDegree ≤ poly.size - 1 :=
    Nat.le_pred_of_lt hDegLt
  exact Nat.le_trans
    (fullFieldPolyRootCount_le_natDegree_of_nonzero (poly := poly) hPolyNeZero)
    hDegLe

theorem fullFieldPolyRootCount_le_maxDegree_of_shape_nonzero
    [Fact (Nat.Prime Goldilocks.q)]
    {poly : Array F}
    {maxDegree : Nat}
    (hShape : poly.size = maxDegree + 1)
    (hPolyNeZero : sumcheckPolynomialZMod poly ≠ 0) :
    fullFieldPolyRootCount poly ≤ maxDegree := by
  have hSizePos : 0 < poly.size := by
    simpa [hShape]
  have hPred :
      fullFieldPolyRootCount poly ≤ poly.size - 1 :=
    fullFieldPolyRootCount_le_pred_size_of_nonzero
      (poly := poly) hPolyNeZero hSizePos
  simpa [hShape] using hPred

/--
Lower-level polynomial-root package for one game.

This captures the intended "real math" input layer:
1) each round event is controlled by a concrete polynomial witness,
2) each polynomial has a full-field root-count budget,
3) round-event counting over `F^ℓ` is bounded by the corresponding root set
   times `|F|^(ℓ-1)` (coordinate-lift counting).

From this package we can construct `FullFieldRoundEventRootCountLemmas`.
-/
structure FullFieldRoundPolynomialRootLemmas
  (g : SoundnessGame) where
  domainAligned :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundRootBudget : Nat → Nat
  roundRootBudgetBound :
    ∀ i : Nat, i < g.inst.rounds →
      roundRootBudget i ≤ g.inst.maxDegree
  roundPoly : Nat → Array F
  /-- Event-to-root relation for each round witness polynomial (Mathlib bridge form). -/
  roundFailureImpliesPolyRoot :
    ∀ i : Nat, i < g.inst.rounds →
      ∀ coins : Array F,
        roundFailure i coins →
          (sumcheckPolynomialZMod (roundPoly i)).eval (fToZMod (coins[i]!)) = 0
  /-- Root-budget bound in the full field for each round witness polynomial. -/
  roundPolyRootCountBound :
    ∀ i : Nat, i < g.inst.rounds →
      fullFieldPolyRootCount (roundPoly i) ≤ roundRootBudget i
  /--
  Coordinate-lift counting bridge:
  event count over `F^ℓ` is bounded by root count times `|F|^(ℓ-1)`.
  -/
  roundFailureCountLePolyRoots :
    ∀ i : Nat, i < g.inst.rounds →
      fullFieldCoinEventCount g.inst.rounds (roundFailure i) ≤
        fullFieldPolyRootCount (roundPoly i) * Goldilocks.q ^ (g.inst.rounds - 1)

/--
Mathlib-root-count flavored lower-level polynomial package for one game.

This variant fixes round root budgets to `inst.maxDegree` and derives the
per-round polynomial root-count bounds from Mathlib theorems using:
- bridged polynomial nonzero proofs,
- bridged polynomial shape (`size = maxDegree + 1`).
-/
structure FullFieldRoundPolynomialRootMathlibLemmas
  (g : SoundnessGame) where
  domainAligned :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundPoly : Nat → Array F
  roundPolyShape :
    ∀ i : Nat, i < g.inst.rounds →
      (roundPoly i).size = g.inst.maxDegree + 1
  roundPolyNonzero :
    ∀ i : Nat, i < g.inst.rounds →
      sumcheckPolynomialZMod (roundPoly i) ≠ 0
  roundFailureImpliesPolyRoot :
    ∀ i : Nat, i < g.inst.rounds →
      ∀ coins : Array F,
        roundFailure i coins →
          (sumcheckPolynomialZMod (roundPoly i)).eval (fToZMod (coins[i]!)) = 0

/--
Constructive conversion from Mathlib-root-count flavored polynomial lemmas to
the existing `FullFieldRoundPolynomialRootLemmas` package.
-/
def FullFieldRoundPolynomialRootLemmas.of_mathlib
  (g : SoundnessGame)
  (hMathlib : FullFieldRoundPolynomialRootMathlibLemmas g) :
  FullFieldRoundPolynomialRootLemmas g := by
  refine
    { domainAligned := hMathlib.domainAligned
      roundFailure := hMathlib.roundFailure
      covered := hMathlib.covered
      roundRootBudget := fun _ => g.inst.maxDegree
      roundRootBudgetBound := ?_
      roundPoly := hMathlib.roundPoly
      roundFailureImpliesPolyRoot := hMathlib.roundFailureImpliesPolyRoot
      roundPolyRootCountBound := ?_
      roundFailureCountLePolyRoots := ?_ }
  · intro i _hi
    exact le_rfl
  · intro i hi
    have hShape : (hMathlib.roundPoly i).size = g.inst.maxDegree + 1 :=
      hMathlib.roundPolyShape i hi
    have hNz : sumcheckPolynomialZMod (hMathlib.roundPoly i) ≠ 0 :=
      hMathlib.roundPolyNonzero i hi
    simpa using
      (fullFieldPolyRootCount_le_maxDegree_of_shape_nonzero
        (poly := hMathlib.roundPoly i)
        (maxDegree := g.inst.maxDegree)
        hShape hNz)
  · intro i hi
    classical
    have hImp :
        ∀ coins, hMathlib.roundFailure i coins →
          (sumcheckPolynomialZMod (hMathlib.roundPoly i)).eval (fToZMod (coins[i]!)) = 0 := by
      intro coins hFail
      exact hMathlib.roundFailureImpliesPolyRoot i hi coins hFail
    simpa [fullFieldPolyRootCount] using
      (fullFieldCoinEventCount_le_coordPredicate
        (m := g.inst.rounds)
        (i := i)
        (hi := hi)
        (E := hMathlib.roundFailure i)
        (P := fun r : F =>
          (sumcheckPolynomialZMod (hMathlib.roundPoly i)).eval (fToZMod r) = 0)
        hImp)

/--
Constructive lift from polynomial-root lemmas to paper-style full-field
round-event root-count lemmas.
-/
def FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas
  (g : SoundnessGame)
  (hPoly : FullFieldRoundPolynomialRootLemmas g) :
  FullFieldRoundEventRootCountLemmas g := by
  refine
    { domainAligned := hPoly.domainAligned
      roundFailure := hPoly.roundFailure
      covered := hPoly.covered
      roundRootBudget := hPoly.roundRootBudget
      roundRootBudgetBound := hPoly.roundRootBudgetBound
      roundCountBoundPow := ?_ }
  intro i hi
  have hCountRoots :
      fullFieldCoinEventCount g.inst.rounds (hPoly.roundFailure i) ≤
        fullFieldPolyRootCount (hPoly.roundPoly i) * Goldilocks.q ^ (g.inst.rounds - 1) :=
    hPoly.roundFailureCountLePolyRoots i hi
  have hRootBound :
      fullFieldPolyRootCount (hPoly.roundPoly i) ≤ hPoly.roundRootBudget i :=
    hPoly.roundPolyRootCountBound i hi
  have hMul :
      fullFieldPolyRootCount (hPoly.roundPoly i) * Goldilocks.q ^ (g.inst.rounds - 1) ≤
        hPoly.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1) := by
    exact Nat.mul_le_mul_right (Goldilocks.q ^ (g.inst.rounds - 1)) hRootBound
  exact Nat.le_trans hCountRoots hMul

/--
Constructive conversion from full-field Schwartz-Zippel round-event lemmas
to count-scaled cardinality lemmas.
-/
def FullFieldRoundEventCardinalityLemmas.of_schwartzZippel
  (g : SoundnessGame)
  (hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g) :
  FullFieldRoundEventCardinalityLemmas g := by
  refine
    { roundFailure := hSz.roundFailure
      covered := hSz.covered
      roundRootBudget := hSz.roundRootBudget
      roundRootBudgetBound := hSz.roundRootBudgetBound
      roundCountBoundScaled := ?_ }
  intro i hi
  have hProb :
      fullFieldCoinPr g.inst.rounds (hSz.roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat) ≤
          (hSz.roundRootBudget i : Rat) := by
    simpa [fullFieldUniformCoinProbModel] using hSz.roundProbBoundScaled i hi
  exact fullFieldCoinEventCount_scaled_of_pr_mul_nat_le
    g.inst.rounds
    (hSz.roundFailure i)
    (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
    (hSz.roundRootBudget i)
    hProb

/--
Constructive conversion from count-scaled cardinality lemmas to paper-style
root-count lemmas for full-field games.

Requires denominator alignment `|K| = |F| = Goldilocks.q`.
-/
def FullFieldRoundEventRootCountLemmas.of_cardinality
  (g : SoundnessGame)
  (hDomain :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hCard : FullFieldRoundEventCardinalityLemmas g) :
  FullFieldRoundEventRootCountLemmas g := by
  refine
    { domainAligned := hDomain
      roundFailure := hCard.roundFailure
      covered := hCard.covered
      roundRootBudget := hCard.roundRootBudget
      roundRootBudgetBound := hCard.roundRootBudgetBound
      roundCountBoundPow := ?_ }
  intro i hi
  have hScaled :
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
          hCard.roundRootBudget i * (fullFieldCoinSpace g.inst.rounds).length :=
    hCard.roundCountBoundScaled i hi
  have hScaledQ :
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) * Goldilocks.q ≤
        hCard.roundRootBudget i * Goldilocks.q ^ g.inst.rounds := by
    calc
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) * Goldilocks.q
          = fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) *
              (SuperNeo.sumcheckLundSoundnessDenominator g.inst) := by
                simp [hDomain]
      _ ≤ hCard.roundRootBudget i * (fullFieldCoinSpace g.inst.rounds).length := hScaled
      _ = hCard.roundRootBudget i * Goldilocks.q ^ g.inst.rounds := by
            simp [fullFieldCoinSpace_length]
  have hRoundsPos : 0 < g.inst.rounds := by
    exact Nat.pos_of_ne_zero (by
      intro hZero
      simpa [hZero] using hi)
  have hPowStep :
      Goldilocks.q ^ g.inst.rounds =
        Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q := by
    have hIdx : (g.inst.rounds - 1) + 1 = g.inst.rounds := by
      omega
    calc
      Goldilocks.q ^ g.inst.rounds
          = Goldilocks.q ^ ((g.inst.rounds - 1) + 1) := by
              simp [hIdx]
      _ = Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q := by
            simp [Nat.pow_succ, Nat.mul_comm]
  have hScaledQ' :
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) * Goldilocks.q ≤
        (hCard.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1)) *
          Goldilocks.q := by
    calc
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) * Goldilocks.q
          ≤ hCard.roundRootBudget i * Goldilocks.q ^ g.inst.rounds := hScaledQ
      _ = hCard.roundRootBudget i * (Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q) := by
            rw [hPowStep]
      _ = (hCard.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1)) * Goldilocks.q := by
            simp [Nat.mul_assoc]
  exact Nat.le_of_mul_le_mul_right hScaledQ' Goldilocks.q_pos

/--
Constructive conversion from root-count bounds
`count(Eᵢ) ≤ dᵢ * |F|^(ℓ-1)` into the cross-multiplied cardinality surface
`count(Eᵢ) * |K| ≤ dᵢ * |F|^ℓ`, using `|K| = |F|`.
-/
def FullFieldRoundEventCardinalityLemmas.of_rootCount
  (g : SoundnessGame)
  (hRoot : FullFieldRoundEventRootCountLemmas g) :
  FullFieldRoundEventCardinalityLemmas g := by
  refine
    { roundFailure := hRoot.roundFailure
      covered := hRoot.covered
      roundRootBudget := hRoot.roundRootBudget
      roundRootBudgetBound := hRoot.roundRootBudgetBound
      roundCountBoundScaled := ?_ }
  intro i hi
  have hRoundsPos : 0 < g.inst.rounds := by
    exact Nat.pos_of_ne_zero (by
      intro hZero
      simpa [hZero] using hi)
  have hPowStep :
      Goldilocks.q ^ g.inst.rounds =
        Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q := by
    have hIdx : (g.inst.rounds - 1) + 1 = g.inst.rounds := by
      omega
    calc
      Goldilocks.q ^ g.inst.rounds
          = Goldilocks.q ^ ((g.inst.rounds - 1) + 1) := by
              simp [hIdx]
      _ = Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q := by
            simp [Nat.pow_succ, Nat.mul_comm]
  have hCountPow :
      fullFieldCoinEventCount g.inst.rounds (hRoot.roundFailure i) ≤
        hRoot.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1) :=
    hRoot.roundCountBoundPow i hi
  have hMul :
      fullFieldCoinEventCount g.inst.rounds (hRoot.roundFailure i) * Goldilocks.q ≤
        (hRoot.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1)) * Goldilocks.q := by
    exact Nat.mul_le_mul_right Goldilocks.q hCountPow
  calc
    fullFieldCoinEventCount g.inst.rounds (hRoot.roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
        = fullFieldCoinEventCount g.inst.rounds (hRoot.roundFailure i) * Goldilocks.q := by
            simp [hRoot.domainAligned]
    _ ≤ (hRoot.roundRootBudget i * Goldilocks.q ^ (g.inst.rounds - 1)) * Goldilocks.q := hMul
    _ = hRoot.roundRootBudget i * (Goldilocks.q ^ (g.inst.rounds - 1) * Goldilocks.q) := by
          simp [Nat.mul_assoc]
    _ = hRoot.roundRootBudget i * Goldilocks.q ^ g.inst.rounds := by
          rw [← hPowStep]
    _ = hRoot.roundRootBudget i * (fullFieldCoinSpace g.inst.rounds).length := by
          simp [fullFieldCoinSpace_length]

/-- Global lower-level closure surface in root-count form for full-field games. -/
def FullFieldRoundEventRootCountAssumption : Prop :=
  ∀ g : SoundnessGame, Nonempty (FullFieldRoundEventRootCountLemmas g)

/--
Domain-aligned variant of the full-field root-count closure surface:
for aligned games (`|K| = |F| = q`), per-game root-count lemmas exist.
-/
def FullFieldRoundEventRootCountAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      Nonempty (FullFieldRoundEventRootCountLemmas g)

/--
Global lower-level closure surface in polynomial-root form for full-field games.

This is the stronger theorem-native input that can be converted constructively
to `FullFieldRoundEventRootCountAssumption`.
-/
def FullFieldRoundPolynomialRootAssumption : Prop :=
  ∀ g : SoundnessGame, Nonempty (FullFieldRoundPolynomialRootLemmas g)

/--
Domain-aligned variant of the full-field polynomial-root closure surface:
for aligned games (`|K| = |F| = q`), per-game polynomial-root lemmas exist.
-/
def FullFieldRoundPolynomialRootAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      Nonempty (FullFieldRoundPolynomialRootLemmas g)

/--
Global lower-level closure surface in Mathlib-root-count form for full-field games.

This is a theorem-native strengthening: it carries nonzero/shape witnesses for
bridged polynomials and derives the original polynomial-root package
constructively (`FullFieldRoundPolynomialRootLemmas.of_mathlib`).
-/
def FullFieldRoundPolynomialRootMathlibAssumption : Prop :=
  ∀ g : SoundnessGame, Nonempty (FullFieldRoundPolynomialRootMathlibLemmas g)

/--
Domain-aligned variant of the full-field Mathlib-root closure surface:
for aligned games (`|K| = |F| = q`), per-game Mathlib-root lemmas exist.
-/
def FullFieldRoundPolynomialRootMathlibAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      Nonempty (FullFieldRoundPolynomialRootMathlibLemmas g)

/--
Lower-level algebraic witness package for constructing
`FullFieldRoundPolynomialRootMathlibAssumption` from full-field
Schwartz-Zippel round-event lemmas.

This intentionally separates:
- probabilistic/event coverage lemmas (`SchwartzZippelRoundEventLemmas`), and
- polynomial witness/root lemmas (this structure).
-/
structure FullFieldRoundPolynomialRootMathlibWitness
  (g : SoundnessGame)
  (hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g) where
  domainAligned :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q
  roundPoly : Nat → Array F
  roundPolyShape :
    ∀ i : Nat, i < g.inst.rounds →
      (roundPoly i).size = g.inst.maxDegree + 1
  roundPolyNonzero :
    ∀ i : Nat, i < g.inst.rounds →
      sumcheckPolynomialZMod (roundPoly i) ≠ 0
  roundFailureImpliesPolyRoot :
    ∀ i : Nat, i < g.inst.rounds →
      ∀ coins : Array F,
        hSz.roundFailure i coins →
          (sumcheckPolynomialZMod (roundPoly i)).eval (fToZMod (coins[i]!)) = 0

/--
Global all-games witness assumption for constructing the Mathlib-root package
from internal probabilistic + algebraic lemmas.
-/
def FullFieldRoundPolynomialRootMathlibWitnessAssumption : Prop :=
  ∀ g : SoundnessGame,
    ∃ hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g,
      Nonempty (FullFieldRoundPolynomialRootMathlibWitness g hSz)

/--
Domain-aligned variant of the witness-layer assumption:
for aligned games (`|K| = |F| = q`), per-game SZ+witness packages exist.
-/
def FullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      ∃ hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g,
        Nonempty (FullFieldRoundPolynomialRootMathlibWitness g hSz)

/--
Lower-level executable root-set witness package for one game.

This is strictly weaker/more primitive than directly providing witness
polynomials: for each round we only require a finite root set that covers every
failing challenge coordinate and is budgeted by `maxDegree`.

From this, witness polynomials are constructed internally as finite products
`∏ (X - r)` and bridged back to coefficient arrays.
-/
structure FullFieldRoundPolynomialRootSetWitness
  (g : SoundnessGame)
  (hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g) where
  domainAligned :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q
  roundRootSet : Nat → Finset F
  roundRootSetBound :
    ∀ i : Nat, i < g.inst.rounds →
      (roundRootSet i).card ≤ g.inst.maxDegree
  roundFailureInRootSet :
    ∀ i : Nat, i < g.inst.rounds →
      ∀ coins : Array F,
        hSz.roundFailure i coins →
          coins[i]! ∈ roundRootSet i

/--
Constructive lift:
root-set witnesses induce full polynomial witnesses by using vanishing products
`∏_{r∈Sᵢ} (X-r)` and the coefficient-array bridge.
-/
noncomputable def FullFieldRoundPolynomialRootMathlibWitness.of_rootSetWitness
  (g : SoundnessGame)
  (hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g)
  (hSet : FullFieldRoundPolynomialRootSetWitness g hSz) :
  FullFieldRoundPolynomialRootMathlibWitness g hSz := by
  classical
  refine
    { domainAligned := hSet.domainAligned
      roundPoly := fun i =>
        zmodPolyToCoeffArray (g.inst.maxDegree + 1)
          (rootVanishingPoly (hSet.roundRootSet i))
      roundPolyShape := ?_
      roundPolyNonzero := ?_
      roundFailureImpliesPolyRoot := ?_ }
  · intro i hi
    simp [zmodPolyToCoeffArray]
  · intro i hi
    let pRoot : Polynomial Fq := rootVanishingPoly (hSet.roundRootSet i)
    have hDegLe : pRoot.natDegree ≤ g.inst.maxDegree := by
      simpa [pRoot, rootVanishingPoly_natDegree_eq_card] using
        hSet.roundRootSetBound i hi
    have hDegLt : pRoot.natDegree < g.inst.maxDegree + 1 :=
      Nat.lt_succ_of_le hDegLe
    have hEq :
        sumcheckPolynomialZMod
          (zmodPolyToCoeffArray (g.inst.maxDegree + 1) pRoot) = pRoot :=
      sumcheckPolynomialZMod_zmodPolyToCoeffArray (g.inst.maxDegree + 1) pRoot hDegLt
    have hNe : pRoot ≠ 0 := by
      simpa [pRoot] using rootVanishingPoly_ne_zero (hSet.roundRootSet i)
    intro hZero
    exact hNe (hEq.symm.trans hZero)
  · intro i hi coins hFail
    let pRoot : Polynomial Fq := rootVanishingPoly (hSet.roundRootSet i)
    have hDegLe : pRoot.natDegree ≤ g.inst.maxDegree := by
      simpa [pRoot, rootVanishingPoly_natDegree_eq_card] using
        hSet.roundRootSetBound i hi
    have hDegLt : pRoot.natDegree < g.inst.maxDegree + 1 :=
      Nat.lt_succ_of_le hDegLe
    have hEq :
        sumcheckPolynomialZMod
          (zmodPolyToCoeffArray (g.inst.maxDegree + 1) pRoot) = pRoot :=
      sumcheckPolynomialZMod_zmodPolyToCoeffArray (g.inst.maxDegree + 1) pRoot hDegLt
    have hMem : coins[i]! ∈ hSet.roundRootSet i :=
      hSet.roundFailureInRootSet i hi coins hFail
    have hEval :
        pRoot.eval (fToZMod (coins[i]!)) = 0 := by
      simpa [pRoot] using rootVanishingPoly_eval_eq_zero_of_mem hMem
    have hEval' :
        (sumcheckPolynomialZMod
          (zmodPolyToCoeffArray (g.inst.maxDegree + 1) pRoot)).eval
            (fToZMod (coins[i]!)) = 0 := by
      simpa [hEq] using hEval
    simpa [pRoot] using hEval'

/-- Global all-games root-set witness assumption (lower than polynomial witness layer). -/
def FullFieldRoundPolynomialRootSetWitnessAssumption : Prop :=
  ∀ g : SoundnessGame,
    ∃ hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g,
      Nonempty (FullFieldRoundPolynomialRootSetWitness g hSz)

/--
Domain-aligned variant of the root-set witness-layer assumption:
for aligned games (`|K| = |F| = q`), per-game SZ+root-set packages exist.
-/
def FullFieldRoundPolynomialRootSetWitnessAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      ∃ hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g,
        Nonempty (FullFieldRoundPolynomialRootSetWitness g hSz)

theorem fullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned_of_full
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumption) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned := by
  intro g _hAligned
  exact hWit g

theorem fullFieldRoundPolynomialRootSetWitnessAssumptionAligned_of_full
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumption) :
  FullFieldRoundPolynomialRootSetWitnessAssumptionAligned := by
  intro g _hAligned
  exact hSet g

theorem fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_aligned
  (hAligned :
    ∀ g : SoundnessGame,
      SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumption := by
  intro g
  exact hWit g (hAligned g)

theorem fullFieldRoundPolynomialRootSetWitnessAssumption_of_aligned
  (hAligned :
    ∀ g : SoundnessGame,
      SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumptionAligned) :
  FullFieldRoundPolynomialRootSetWitnessAssumption := by
  intro g
  exact hSet g (hAligned g)

theorem fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_rootSetWitness
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumption) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumption := by
  intro g
  rcases hSet g with ⟨hSz, hSetGameNonempty⟩
  rcases hSetGameNonempty with ⟨hSetGame⟩
  refine ⟨hSz, ?_⟩
  exact ⟨FullFieldRoundPolynomialRootMathlibWitness.of_rootSetWitness g hSz hSetGame⟩

theorem fullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned_of_rootSetWitnessAligned
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumptionAligned) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned := by
  intro g hAligned
  rcases hSet g hAligned with ⟨hSz, hSetGameNonempty⟩
  rcases hSetGameNonempty with ⟨hSetGame⟩
  refine ⟨hSz, ?_⟩
  exact ⟨FullFieldRoundPolynomialRootMathlibWitness.of_rootSetWitness g hSz hSetGame⟩

/--
Combined lower-level package for one game:
- full-field Schwartz-Zippel round-event lemmas, and
- polynomial witness/root lemmas for the same round events.

This is the theorem-native "single-source" surface that can instantiate both
`SchwartzZippelRoundEventAssumptionFullField` and
`FullFieldRoundPolynomialRootMathlibAssumption` without an extra witness layer.
-/
structure FullFieldRoundMathlibLemmas
  (g : SoundnessGame) where
  domainAligned :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q
  roundFailure : Nat → Array F → Prop
  covered :
    ∀ coins : Array F,
      g.failureEvent coins →
        roundFailureUnionCoins roundFailure g.inst.rounds coins
  roundRootBudget : Nat → Nat
  roundRootBudgetBound :
    ∀ i : Nat, i < g.inst.rounds →
      roundRootBudget i ≤ g.inst.maxDegree
  roundProbBoundScaled :
    ∀ i : Nat, i < g.inst.rounds →
      (fullFieldUniformCoinProbModel g.inst.rounds).Pr (roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
          (roundRootBudget i : Rat)
  roundPoly : Nat → Array F
  roundPolyShape :
    ∀ i : Nat, i < g.inst.rounds →
      (roundPoly i).size = g.inst.maxDegree + 1
  roundPolyNonzero :
    ∀ i : Nat, i < g.inst.rounds →
      sumcheckPolynomialZMod (roundPoly i) ≠ 0
  roundFailureImpliesPolyRoot :
    ∀ i : Nat, i < g.inst.rounds →
      ∀ coins : Array F,
        roundFailure i coins →
          (sumcheckPolynomialZMod (roundPoly i)).eval (fToZMod (coins[i]!)) = 0

/-- Global all-games combined lower-level package. -/
def FullFieldRoundMathlibAssumption : Prop :=
  ∀ g : SoundnessGame, Nonempty (FullFieldRoundMathlibLemmas g)

/--
Domain-aligned variant of the combined full-field package:
for aligned games (`|K| = |F| = q`), per-game Mathlib round lemmas exist.
-/
def FullFieldRoundMathlibAssumptionAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      Nonempty (FullFieldRoundMathlibLemmas g)

/--
Aligned + positive-round full-field Lund endpoint.
-/
def LundSoundnessAssumptionFullFieldAlignedPosRounds : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      0 < g.inst.rounds →
        g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds)

/--
Domain-mismatch blocker: full-field round Mathlib lemmas are impossible for a game
whose challenge-domain denominator is not `Goldilocks.q`.
-/
theorem no_fullFieldRoundMathlibLemmas_of_domain_mismatch
  (g : SoundnessGame)
  (hMismatch :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q) :
  ¬ Nonempty (FullFieldRoundMathlibLemmas g) := by
  intro h
  rcases h with ⟨hGame⟩
  exact hMismatch hGame.domainAligned

theorem fullFieldDomainAligned_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q := by
  intro g
  rcases hMath g with ⟨hGame⟩
  exact hGame.domainAligned

theorem fullFieldRoundMathlibAssumptionAligned_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldRoundMathlibAssumptionAligned := by
  intro g _hAligned
  exact hMath g

theorem fullFieldRoundMathlibAssumption_of_aligned
  (hAligned :
    ∀ g : SoundnessGame,
      SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q)
  (hMath : FullFieldRoundMathlibAssumptionAligned) :
  FullFieldRoundMathlibAssumption := by
  intro g
  exact hMath g (hAligned g)

private def fullFieldRoundMathlibMismatchGame : SoundnessGame where
  inst := { rounds := 0, maxDegree := 0, domainSize := 0, claimedValue := 0 }
  table := #[1]
  tableSize := by simp
  falseClaim := by
    simp [SuperNeo.sumcheckTableSum]
  prover :=
    { roundPoly := fun _ _ => #[0]
      roundPolyShape := by
        intro i hi
        exact (False.elim (Nat.not_lt_zero i hi))
      nonanticipatory := by
        intro i hi
        exact (False.elim (Nat.not_lt_zero i hi)) }

theorem not_fullFieldRoundMathlibAssumption :
  ¬ FullFieldRoundMathlibAssumption := by
  intro hAll
  let g : SoundnessGame := fullFieldRoundMathlibMismatchGame
  have hMismatch :
      SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q := by
    have hqNe : (0 : Nat) ≠ Goldilocks.q := Nat.ne_of_lt Goldilocks.q_pos
    simpa [g, fullFieldRoundMathlibMismatchGame,
      SuperNeo.sumcheckLundSoundnessDenominator] using hqNe
  exact (no_fullFieldRoundMathlibLemmas_of_domain_mismatch g hMismatch) (hAll g)

theorem no_fullFieldRoundEventRootCountLemmas_of_domain_mismatch
  (g : SoundnessGame)
  (hMismatch :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q) :
  ¬ Nonempty (FullFieldRoundEventRootCountLemmas g) := by
  intro h
  rcases h with ⟨hGame⟩
  exact hMismatch hGame.domainAligned

theorem no_fullFieldRoundPolynomialRootLemmas_of_domain_mismatch
  (g : SoundnessGame)
  (hMismatch :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q) :
  ¬ Nonempty (FullFieldRoundPolynomialRootLemmas g) := by
  intro h
  rcases h with ⟨hGame⟩
  exact hMismatch hGame.domainAligned

theorem no_fullFieldRoundPolynomialRootMathlibLemmas_of_domain_mismatch
  (g : SoundnessGame)
  (hMismatch :
    SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q) :
  ¬ Nonempty (FullFieldRoundPolynomialRootMathlibLemmas g) := by
  intro h
  rcases h with ⟨hGame⟩
  exact hMismatch hGame.domainAligned

theorem not_fullFieldRoundEventRootCountAssumption :
  ¬ FullFieldRoundEventRootCountAssumption := by
  intro hAll
  let g : SoundnessGame := fullFieldRoundMathlibMismatchGame
  have hMismatch :
      SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q := by
    have hqNe : (0 : Nat) ≠ Goldilocks.q := Nat.ne_of_lt Goldilocks.q_pos
    simpa [g, fullFieldRoundMathlibMismatchGame,
      SuperNeo.sumcheckLundSoundnessDenominator] using hqNe
  exact (no_fullFieldRoundEventRootCountLemmas_of_domain_mismatch g hMismatch) (hAll g)

theorem not_fullFieldRoundPolynomialRootAssumption :
  ¬ FullFieldRoundPolynomialRootAssumption := by
  intro hAll
  let g : SoundnessGame := fullFieldRoundMathlibMismatchGame
  have hMismatch :
      SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q := by
    have hqNe : (0 : Nat) ≠ Goldilocks.q := Nat.ne_of_lt Goldilocks.q_pos
    simpa [g, fullFieldRoundMathlibMismatchGame,
      SuperNeo.sumcheckLundSoundnessDenominator] using hqNe
  exact (no_fullFieldRoundPolynomialRootLemmas_of_domain_mismatch g hMismatch) (hAll g)

theorem not_fullFieldRoundPolynomialRootMathlibAssumption :
  ¬ FullFieldRoundPolynomialRootMathlibAssumption := by
  intro hAll
  let g : SoundnessGame := fullFieldRoundMathlibMismatchGame
  have hMismatch :
      SuperNeo.sumcheckLundSoundnessDenominator g.inst ≠ Goldilocks.q := by
    have hqNe : (0 : Nat) ≠ Goldilocks.q := Nat.ne_of_lt Goldilocks.q_pos
    simpa [g, fullFieldRoundMathlibMismatchGame,
      SuperNeo.sumcheckLundSoundnessDenominator] using hqNe
  exact (no_fullFieldRoundPolynomialRootMathlibLemmas_of_domain_mismatch g hMismatch) (hAll g)

/-- Forgetful projection: combined package -> full-field SZ round-event lemmas. -/
def FullFieldRoundMathlibLemmas.to_schwartzZippel
  (g : SoundnessGame)
  (h : FullFieldRoundMathlibLemmas g) :
  SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g :=
  { roundFailure := h.roundFailure
    covered := h.covered
    roundRootBudget := h.roundRootBudget
    roundRootBudgetBound := h.roundRootBudgetBound
    roundProbBoundScaled := h.roundProbBoundScaled }

/-- Forgetful projection: combined package -> Mathlib witness package. -/
def FullFieldRoundMathlibLemmas.to_witness
  (g : SoundnessGame)
  (h : FullFieldRoundMathlibLemmas g) :
  FullFieldRoundPolynomialRootMathlibWitness g
    (FullFieldRoundMathlibLemmas.to_schwartzZippel g h) :=
  { domainAligned := h.domainAligned
    roundPoly := h.roundPoly
    roundPolyShape := h.roundPolyShape
    roundPolyNonzero := h.roundPolyNonzero
    roundFailureImpliesPolyRoot := h.roundFailureImpliesPolyRoot }

theorem fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumption := by
  intro g
  rcases hMath g with ⟨hGame⟩
  refine ⟨FullFieldRoundMathlibLemmas.to_schwartzZippel g hGame, ?_⟩
  exact ⟨FullFieldRoundMathlibLemmas.to_witness g hGame⟩

theorem fullFieldRoundPolynomialRootSetWitnessAssumption_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldRoundPolynomialRootSetWitnessAssumption := by
  intro g
  rcases hMath g with ⟨hGame⟩
  let hSzGame := FullFieldRoundMathlibLemmas.to_schwartzZippel g hGame
  refine ⟨hSzGame, ?_⟩
  refine ⟨{
    domainAligned := hGame.domainAligned
    roundRootSet := fun i =>
      Finset.univ.filter (fun r : F =>
        (sumcheckPolynomialZMod (hGame.roundPoly i)).eval (fToZMod r) = 0)
    roundRootSetBound := ?_
    roundFailureInRootSet := ?_
  }⟩
  · intro i hi
    have hShape : (hGame.roundPoly i).size = g.inst.maxDegree + 1 :=
      hGame.roundPolyShape i hi
    have hNz : sumcheckPolynomialZMod (hGame.roundPoly i) ≠ 0 :=
      hGame.roundPolyNonzero i hi
    simpa [fullFieldPolyRootCount] using
      (fullFieldPolyRootCount_le_maxDegree_of_shape_nonzero
        (poly := hGame.roundPoly i)
        (maxDegree := g.inst.maxDegree)
        hShape hNz)
  · intro i hi coins hFail
    have hEval :
        (sumcheckPolynomialZMod (hGame.roundPoly i)).eval (fToZMod (coins[i]!)) = 0 := by
      simpa [hSzGame, FullFieldRoundMathlibLemmas.to_schwartzZippel] using
        (hGame.roundFailureImpliesPolyRoot i hi coins hFail)
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, hEval⟩

theorem schwartzZippelRoundEventAssumptionFullField_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  ∀ g : SoundnessGame,
    Nonempty (SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g) := by
  intro g
  rcases hMath g with ⟨hGame⟩
  exact ⟨FullFieldRoundMathlibLemmas.to_schwartzZippel g hGame⟩

theorem fullFieldRoundPolynomialRootMathlibAssumption_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldRoundPolynomialRootMathlibAssumption := by
  intro g
  rcases hMath g with ⟨hGame⟩
  let hSzGame := FullFieldRoundMathlibLemmas.to_schwartzZippel g hGame
  let hWitGame := FullFieldRoundMathlibLemmas.to_witness g hGame
  exact ⟨{
    domainAligned := hWitGame.domainAligned
    roundFailure := hSzGame.roundFailure
    covered := hSzGame.covered
    roundPoly := hWitGame.roundPoly
    roundPolyShape := hWitGame.roundPolyShape
    roundPolyNonzero := hWitGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := hWitGame.roundFailureImpliesPolyRoot
  }⟩

theorem fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_fullFieldRoundMathlibAligned
  (hMath : FullFieldRoundMathlibAssumptionAligned) :
  FullFieldRoundPolynomialRootMathlibAssumptionAligned := by
  intro g hAligned
  rcases hMath g hAligned with ⟨hGame⟩
  let hSzGame := FullFieldRoundMathlibLemmas.to_schwartzZippel g hGame
  let hWitGame := FullFieldRoundMathlibLemmas.to_witness g hGame
  exact ⟨{
    domainAligned := hWitGame.domainAligned
    roundFailure := hSzGame.roundFailure
    covered := hSzGame.covered
    roundPoly := hWitGame.roundPoly
    roundPolyShape := hWitGame.roundPolyShape
    roundPolyNonzero := hWitGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := hWitGame.roundFailureImpliesPolyRoot
  }⟩

/--
Constructive instantiation of the global Mathlib-root package from:
1) full-field Schwartz-Zippel round-event lemmas for each game, and
2) polynomial witness/root lemmas for those round events.
-/
theorem fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelWitness
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumption) :
  FullFieldRoundPolynomialRootMathlibAssumption := by
  intro g
  rcases hWit g with ⟨hSzGame, hWitGameNonempty⟩
  rcases hWitGameNonempty with ⟨hWitGame⟩
  refine ⟨{
    domainAligned := hWitGame.domainAligned
    roundFailure := hSzGame.roundFailure
    covered := hSzGame.covered
    roundPoly := hWitGame.roundPoly
    roundPolyShape := hWitGame.roundPolyShape
    roundPolyNonzero := hWitGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := ?_
  }⟩
  intro i hi coins hFail
  exact hWitGame.roundFailureImpliesPolyRoot i hi coins hFail

theorem fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_schwartzZippelWitnessAligned
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned) :
  FullFieldRoundPolynomialRootMathlibAssumptionAligned := by
  intro g hAligned
  rcases hWit g hAligned with ⟨hSzGame, hWitGameNonempty⟩
  rcases hWitGameNonempty with ⟨hWitGame⟩
  refine ⟨{
    domainAligned := hWitGame.domainAligned
    roundFailure := hSzGame.roundFailure
    covered := hSzGame.covered
    roundPoly := hWitGame.roundPoly
    roundPolyShape := hWitGame.roundPolyShape
    roundPolyNonzero := hWitGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := ?_
  }⟩
  intro i hi coins hFail
  exact hWitGame.roundFailureImpliesPolyRoot i hi coins hFail

theorem fullFieldRoundPolynomialRootAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundPolynomialRootAssumption := by
  intro g
  rcases hMathlib g with ⟨hMathlibGame⟩
  exact ⟨FullFieldRoundPolynomialRootLemmas.of_mathlib g hMathlibGame⟩

/--
Constructive closure:
polynomial-root lemmas imply paper-style round-event root-count lemmas.
-/
theorem fullFieldRoundEventRootCountAssumption_of_polynomialRoot
  (hPoly : FullFieldRoundPolynomialRootAssumption) :
  FullFieldRoundEventRootCountAssumption := by
  intro g
  rcases hPoly g with ⟨hPolyGame⟩
  exact ⟨FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas g hPolyGame⟩

/--
Direct constructive closure:
Mathlib-root-count package implies full-field root-count assumption.
-/
theorem fullFieldRoundEventRootCountAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundEventRootCountAssumption := by
  exact fullFieldRoundEventRootCountAssumption_of_polynomialRoot
    (fullFieldRoundPolynomialRootAssumption_of_mathlib hMathlib)

/--
Global denominator-alignment surface for full-field soundness games:
`|K| = |F| = Goldilocks.q`.
-/
def FullFieldDomainAlignedAssumption : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q

theorem fullFieldDomainAlignedAssumption_of_fullFieldRoundEventRootCount
  (hRoot : FullFieldRoundEventRootCountAssumption) :
  FullFieldDomainAlignedAssumption := by
  intro g
  rcases hRoot g with ⟨hGame⟩
  exact hGame.domainAligned

theorem fullFieldDomainAlignedAssumption_of_fullFieldRoundPolynomialRoot
  (hPoly : FullFieldRoundPolynomialRootAssumption) :
  FullFieldDomainAlignedAssumption := by
  intro g
  rcases hPoly g with ⟨hGame⟩
  exact hGame.domainAligned

theorem fullFieldDomainAlignedAssumption_of_fullFieldRoundPolynomialRootMathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldDomainAlignedAssumption := by
  intro g
  rcases hMathlib g with ⟨hGame⟩
  exact hGame.domainAligned

theorem fullFieldRoundEventRootCountAssumptionAligned_of_fullFieldRoundEventRootCount
  (hRoot : FullFieldRoundEventRootCountAssumption) :
  FullFieldRoundEventRootCountAssumptionAligned := by
  intro g _hAligned
  exact hRoot g

theorem fullFieldRoundEventRootCountAssumption_of_aligned
  (hAligned : FullFieldDomainAlignedAssumption)
  (hRoot : FullFieldRoundEventRootCountAssumptionAligned) :
  FullFieldRoundEventRootCountAssumption := by
  intro g
  exact hRoot g (hAligned g)

theorem fullFieldRoundPolynomialRootAssumptionAligned_of_fullFieldRoundPolynomialRoot
  (hPoly : FullFieldRoundPolynomialRootAssumption) :
  FullFieldRoundPolynomialRootAssumptionAligned := by
  intro g _hAligned
  exact hPoly g

theorem fullFieldRoundPolynomialRootAssumption_of_aligned
  (hAligned : FullFieldDomainAlignedAssumption)
  (hPoly : FullFieldRoundPolynomialRootAssumptionAligned) :
  FullFieldRoundPolynomialRootAssumption := by
  intro g
  exact hPoly g (hAligned g)

theorem fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_fullFieldRoundPolynomialRootMathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundPolynomialRootMathlibAssumptionAligned := by
  intro g _hAligned
  exact hMathlib g

theorem fullFieldRoundPolynomialRootMathlibAssumption_of_aligned
  (hAligned : FullFieldDomainAlignedAssumption)
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumptionAligned) :
  FullFieldRoundPolynomialRootMathlibAssumption := by
  intro g
  exact hMathlib g (hAligned g)

theorem fullFieldRoundEventRootCountAssumptionAligned_of_fullFieldRoundMathlibAligned
  (hMath : FullFieldRoundMathlibAssumptionAligned) :
  FullFieldRoundEventRootCountAssumptionAligned := by
  intro g hAligned
  have hMathlibAligned :
      FullFieldRoundPolynomialRootMathlibAssumptionAligned :=
    fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_fullFieldRoundMathlibAligned hMath
  rcases hMathlibAligned g hAligned with ⟨hMathGame⟩
  let hPoly : FullFieldRoundPolynomialRootLemmas g :=
    FullFieldRoundPolynomialRootLemmas.of_mathlib g hMathGame
  exact ⟨FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas g hPoly⟩

theorem fullFieldDomainAlignedAssumption_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldDomainAlignedAssumption := by
  exact fullFieldDomainAligned_of_fullFieldRoundMathlib hMath

theorem fullFieldRoundMathlibAssumption_of_domainAlignedAssumption
  (hAligned : FullFieldDomainAlignedAssumption)
  (hMath : FullFieldRoundMathlibAssumptionAligned) :
  FullFieldRoundMathlibAssumption := by
  exact fullFieldRoundMathlibAssumption_of_aligned hAligned hMath

/--
Constructive lift from full-field round-event cardinality lemmas to the
Schwartz-Zippel round-event theorem surface.
-/
def SchwartzZippelRoundEventLemmas.of_fullFieldCardinality
  (g : SoundnessGame)
  (hCard : FullFieldRoundEventCardinalityLemmas g) :
  SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g := by
  refine
    { roundFailure := hCard.roundFailure
      covered := hCard.covered
      roundRootBudget := hCard.roundRootBudget
      roundRootBudgetBound := hCard.roundRootBudgetBound
      roundProbBoundScaled := ?_ }
  intro i hi
  have hScaledNat :
      fullFieldCoinEventCount g.inst.rounds (hCard.roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst) ≤
          hCard.roundRootBudget i * (fullFieldCoinSpace g.inst.rounds).length :=
    hCard.roundCountBoundScaled i hi
  have hProb :
      fullFieldCoinPr g.inst.rounds (hCard.roundFailure i) *
        (SuperNeo.sumcheckLundSoundnessDenominator g.inst : Rat) ≤
          (hCard.roundRootBudget i : Rat) := by
    exact fullFieldCoinPr_mul_nat_le_of_countScaled
      g.inst.rounds
      (hCard.roundFailure i)
      (SuperNeo.sumcheckLundSoundnessDenominator g.inst)
      (hCard.roundRootBudget i)
      hScaledNat
  simpa [fullFieldUniformCoinProbModel] using hProb

/-- Global full-field closure surface for round-event cardinality lemmas. -/
def FullFieldRoundEventCardinalityAssumption : Prop :=
  ∀ g : SoundnessGame, Nonempty (FullFieldRoundEventCardinalityLemmas g)

theorem fullFieldRoundEventCardinalityAssumption_of_rootCount
  (hRoot : FullFieldRoundEventRootCountAssumption) :
  FullFieldRoundEventCardinalityAssumption := by
  intro g
  rcases hRoot g with ⟨hRootGame⟩
  exact ⟨FullFieldRoundEventCardinalityLemmas.of_rootCount g hRootGame⟩

/--
Direct constructive closure:
Mathlib-root-count package implies full-field cardinality assumption.
-/
theorem fullFieldRoundEventCardinalityAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundEventCardinalityAssumption := by
  exact fullFieldRoundEventCardinalityAssumption_of_rootCount
    (fullFieldRoundEventRootCountAssumption_of_mathlib hMathlib)

/--
Constructive combined-package instantiation from Mathlib-root packages.

Given per-game Mathlib-root polynomial witnesses, we build the full-field SZ
round-event lemmas through the canonical root-count/cardinality conversions and
package both layers together as `FullFieldRoundMathlibLemmas`.
-/
theorem fullFieldRoundMathlibAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundMathlibAssumption := by
  intro g
  rcases hMathlib g with ⟨hMathGame⟩
  let hPoly : FullFieldRoundPolynomialRootLemmas g :=
    FullFieldRoundPolynomialRootLemmas.of_mathlib g hMathGame
  let hRoot : FullFieldRoundEventRootCountLemmas g :=
    FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas g hPoly
  let hCard : FullFieldRoundEventCardinalityLemmas g :=
    FullFieldRoundEventCardinalityLemmas.of_rootCount g hRoot
  let hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g :=
    SchwartzZippelRoundEventLemmas.of_fullFieldCardinality g hCard
  refine ⟨{
    domainAligned := hMathGame.domainAligned
    roundFailure := hSz.roundFailure
    covered := hSz.covered
    roundRootBudget := hSz.roundRootBudget
    roundRootBudgetBound := hSz.roundRootBudgetBound
    roundProbBoundScaled := hSz.roundProbBoundScaled
    roundPoly := hMathGame.roundPoly
    roundPolyShape := hMathGame.roundPolyShape
    roundPolyNonzero := hMathGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := ?_
  }⟩
  intro i hi coins hFail
  have hFailMath : hMathGame.roundFailure i coins := by
    simpa [hSz, hCard, hRoot, hPoly] using hFail
  exact hMathGame.roundFailureImpliesPolyRoot i hi coins hFailMath

theorem fullFieldRoundMathlibAssumptionAligned_of_mathlibAligned
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumptionAligned) :
  FullFieldRoundMathlibAssumptionAligned := by
  intro g hAligned
  rcases hMathlib g hAligned with ⟨hMathGame⟩
  let hPoly : FullFieldRoundPolynomialRootLemmas g :=
    FullFieldRoundPolynomialRootLemmas.of_mathlib g hMathGame
  let hRoot : FullFieldRoundEventRootCountLemmas g :=
    FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas g hPoly
  let hCard : FullFieldRoundEventCardinalityLemmas g :=
    FullFieldRoundEventCardinalityLemmas.of_rootCount g hRoot
  let hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g :=
    SchwartzZippelRoundEventLemmas.of_fullFieldCardinality g hCard
  refine ⟨{
    domainAligned := hMathGame.domainAligned
    roundFailure := hSz.roundFailure
    covered := hSz.covered
    roundRootBudget := hSz.roundRootBudget
    roundRootBudgetBound := hSz.roundRootBudgetBound
    roundProbBoundScaled := hSz.roundProbBoundScaled
    roundPoly := hMathGame.roundPoly
    roundPolyShape := hMathGame.roundPolyShape
    roundPolyNonzero := hMathGame.roundPolyNonzero
    roundFailureImpliesPolyRoot := ?_
  }⟩
  intro i hi coins hFail
  have hFailMath : hMathGame.roundFailure i coins := by
    simpa [hSz, hCard, hRoot, hPoly] using hFail
  exact hMathGame.roundFailureImpliesPolyRoot i hi coins hFailMath

theorem fullFieldRoundPolynomialRootSetWitnessAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundPolynomialRootSetWitnessAssumption := by
  exact fullFieldRoundPolynomialRootSetWitnessAssumption_of_fullFieldRoundMathlib
    (fullFieldRoundMathlibAssumption_of_mathlib hMathlib)

/--
Direct constructive closure:
root-set witness packages imply global Mathlib-root packages.
-/
theorem fullFieldRoundPolynomialRootMathlibAssumption_of_rootSetWitness
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumption) :
  FullFieldRoundPolynomialRootMathlibAssumption := by
  exact fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelWitness
    (fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_rootSetWitness hSet)

/--
Direct constructive closure:
root-set witness packages imply the combined full-field round package.
-/
theorem fullFieldRoundMathlibAssumption_of_rootSetWitness
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumption) :
  FullFieldRoundMathlibAssumption := by
  exact fullFieldRoundMathlibAssumption_of_mathlib
    (fullFieldRoundPolynomialRootMathlibAssumption_of_rootSetWitness hSet)

theorem fullFieldRoundMathlibAssumptionAligned_of_rootSetWitnessAligned
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumptionAligned) :
  FullFieldRoundMathlibAssumptionAligned := by
  exact fullFieldRoundMathlibAssumptionAligned_of_mathlibAligned
    (fullFieldRoundPolynomialRootMathlibAssumptionAligned_of_schwartzZippelWitnessAligned
      (fullFieldRoundPolynomialRootMathlibWitnessAssumptionAligned_of_rootSetWitnessAligned hSet))

/--
Direct constructive closure:
combined full-field round package implies full-field root-count assumption.
-/
theorem fullFieldRoundEventRootCountAssumption_of_fullFieldRoundMathlib
  (hMath : FullFieldRoundMathlibAssumption) :
  FullFieldRoundEventRootCountAssumption := by
  exact fullFieldRoundEventRootCountAssumption_of_mathlib
    (fullFieldRoundPolynomialRootMathlibAssumption_of_fullFieldRoundMathlib hMath)

/--
Direct constructive closure:
root-set witness packages imply full-field root-count assumption.
-/
theorem fullFieldRoundEventRootCountAssumption_of_rootSetWitness
  (hSet : FullFieldRoundPolynomialRootSetWitnessAssumption) :
  FullFieldRoundEventRootCountAssumption := by
  exact fullFieldRoundEventRootCountAssumption_of_mathlib
    (fullFieldRoundPolynomialRootMathlibAssumption_of_rootSetWitness hSet)

/--
Global full-field Schwartz-Zippel round-event theorem surface.
-/
def SchwartzZippelRoundEventAssumptionFullField : Prop :=
  ∀ g : SoundnessGame,
    Nonempty (SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g)

/--
Constructive witness-layer instantiation from the Mathlib-root package itself.

This removes the need to provide a separate witness assumption when an all-games
`FullFieldRoundPolynomialRootMathlibAssumption` package is already available:
the required full-field SZ event package is constructed canonically from the
Mathlib-root chain and paired with the original polynomial witnesses.
-/
theorem fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  FullFieldRoundPolynomialRootMathlibWitnessAssumption := by
  exact fullFieldRoundPolynomialRootMathlibWitnessAssumption_of_rootSetWitness
    (fullFieldRoundPolynomialRootSetWitnessAssumption_of_mathlib hMathlib)

theorem schwartzZippelRoundEventAssumptionFullField_of_witness
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumption) :
  SchwartzZippelRoundEventAssumptionFullField := by
  intro g
  rcases hWit g with ⟨hSzGame, _hWitGame⟩
  exact ⟨hSzGame⟩

theorem fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelFullFieldWitness
  (_hSz : SchwartzZippelRoundEventAssumptionFullField)
  (hWit : FullFieldRoundPolynomialRootMathlibWitnessAssumption) :
  FullFieldRoundPolynomialRootMathlibAssumption := by
  exact fullFieldRoundPolynomialRootMathlibAssumption_of_schwartzZippelWitness hWit

theorem fullFieldRoundEventCardinalityAssumption_of_schwartzZippelFullField
  (hSz : SchwartzZippelRoundEventAssumptionFullField) :
  FullFieldRoundEventCardinalityAssumption := by
  intro g
  rcases hSz g with ⟨hSzGame⟩
  exact ⟨FullFieldRoundEventCardinalityLemmas.of_schwartzZippel g hSzGame⟩

theorem fullFieldRoundEventRootCountAssumption_of_cardinality
  (hDomain : FullFieldDomainAlignedAssumption)
  (hCard : FullFieldRoundEventCardinalityAssumption) :
  FullFieldRoundEventRootCountAssumption := by
  intro g
  rcases hCard g with ⟨hCardGame⟩
  exact ⟨FullFieldRoundEventRootCountLemmas.of_cardinality g (hDomain g) hCardGame⟩

theorem fullFieldRoundEventRootCountAssumption_of_schwartzZippelFullField
  (hDomain : FullFieldDomainAlignedAssumption)
  (hSz : SchwartzZippelRoundEventAssumptionFullField) :
  FullFieldRoundEventRootCountAssumption := by
  exact fullFieldRoundEventRootCountAssumption_of_cardinality
    hDomain
    (fullFieldRoundEventCardinalityAssumption_of_schwartzZippelFullField hSz)

theorem schwartzZippelRoundEventAssumptionFullField_of_cardinality
  (hCard : FullFieldRoundEventCardinalityAssumption) :
  SchwartzZippelRoundEventAssumptionFullField := by
  intro g
  rcases hCard g with ⟨hCardGame⟩
  exact ⟨SchwartzZippelRoundEventLemmas.of_fullFieldCardinality g hCardGame⟩

theorem schwartzZippelRoundEventAssumptionFullField_of_rootCount
  (hRoot : FullFieldRoundEventRootCountAssumption) :
  SchwartzZippelRoundEventAssumptionFullField := by
  exact schwartzZippelRoundEventAssumptionFullField_of_cardinality
    (fullFieldRoundEventCardinalityAssumption_of_rootCount hRoot)

/--
Direct constructive closure:
Mathlib-root-count package implies full-field Schwartz-Zippel round-event assumption.
-/
theorem schwartzZippelRoundEventAssumptionFullField_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  SchwartzZippelRoundEventAssumptionFullField := by
  exact schwartzZippelRoundEventAssumptionFullField_of_cardinality
    (fullFieldRoundEventCardinalityAssumption_of_mathlib hMathlib)

/--
Full-field Lund soundness endpoint:
for every game, the canonical full-field coin model satisfies the Lund bound.
-/
def LundSoundnessAssumptionFullField : Prop :=
  ∀ g : SoundnessGame, g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds)

/--
Aligned full-field Lund soundness endpoint:
for aligned games (`|K| = |F| = q`), the canonical full-field coin model
satisfies the Lund bound.
-/
def LundSoundnessAssumptionFullFieldAligned : Prop :=
  ∀ g : SoundnessGame,
    SuperNeo.sumcheckLundSoundnessDenominator g.inst = Goldilocks.q →
      g.lundBoundHolds (fullFieldUniformCoinProbModel g.inst.rounds)

theorem lundSoundnessAssumptionFullField_of_schwartzZippelRoundEvent
  (hSz : SchwartzZippelRoundEventAssumptionFullField) :
  LundSoundnessAssumptionFullField := by
  intro g
  rcases hSz g with ⟨hSzGame⟩
  let prob := fullFieldUniformCoinProbModel g.inst.rounds
  have hKernel : LundRoundKernel prob g :=
    LundRoundKernel.of_schwartzZippelRoundEventLemmas prob g hSzGame
  have hScaled : LundRoundBoundaryScaled prob g :=
    LundRoundBoundaryScaled.of_kernel prob g hKernel
  exact SoundnessGame.lundBoundHolds_of_scaledRoundBoundary prob g hScaled

theorem lundSoundnessAssumptionFullField_of_rootCount
  (hRoot : FullFieldRoundEventRootCountAssumption) :
  LundSoundnessAssumptionFullField := by
  exact lundSoundnessAssumptionFullField_of_schwartzZippelRoundEvent
    (schwartzZippelRoundEventAssumptionFullField_of_rootCount hRoot)

/--
Direct constructive closure:
Mathlib-root-count package implies full-field Lund soundness endpoint.
-/
theorem lundSoundnessAssumptionFullField_of_mathlib
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumption) :
  LundSoundnessAssumptionFullField := by
  exact lundSoundnessAssumptionFullField_of_schwartzZippelRoundEvent
    (schwartzZippelRoundEventAssumptionFullField_of_mathlib hMathlib)

theorem lundSoundnessAssumptionFullFieldAligned_of_mathlibAligned
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumptionAligned) :
  LundSoundnessAssumptionFullFieldAligned := by
  intro g hAligned
  rcases hMathlib g hAligned with ⟨hMathGame⟩
  let hPoly : FullFieldRoundPolynomialRootLemmas g :=
    FullFieldRoundPolynomialRootLemmas.of_mathlib g hMathGame
  let hRoot : FullFieldRoundEventRootCountLemmas g :=
    FullFieldRoundEventRootCountLemmas.of_polynomialRootLemmas g hPoly
  let hCard : FullFieldRoundEventCardinalityLemmas g :=
    FullFieldRoundEventCardinalityLemmas.of_rootCount g hRoot
  let hSz : SchwartzZippelRoundEventLemmas (fullFieldUniformCoinProbModel g.inst.rounds) g :=
    SchwartzZippelRoundEventLemmas.of_fullFieldCardinality g hCard
  let prob := fullFieldUniformCoinProbModel g.inst.rounds
  have hKernel : LundRoundKernel prob g :=
    LundRoundKernel.of_schwartzZippelRoundEventLemmas prob g hSz
  have hScaled : LundRoundBoundaryScaled prob g :=
    LundRoundBoundaryScaled.of_kernel prob g hKernel
  exact SoundnessGame.lundBoundHolds_of_scaledRoundBoundary prob g hScaled


theorem lundSoundnessAssumptionFullField_of_mathlibAligned
  (hAligned : FullFieldDomainAlignedAssumption)
  (hMathlib : FullFieldRoundPolynomialRootMathlibAssumptionAligned) :
  LundSoundnessAssumptionFullField := by
  exact lundSoundnessAssumptionFullField_of_mathlib
    (fullFieldRoundPolynomialRootMathlibAssumption_of_aligned hAligned hMathlib)

/--
Round-by-round soundness boundary:
for each round, a failure event with an explicit probability bound, and
coverage from global soundness failure to the union of round failures.
-/
structure RoundByRoundSoundnessBoundary
  (prob : ProbModel)
  (inst : Instance)
  (tr : Transcript) where
  roundFailure : Nat → Prop
  epsRound : Nat → Rat
  roundFailureBound :
    ∀ i : Nat, i < inst.rounds → prob.Pr (roundFailure i) ≤ epsRound i
  soundnessFailureCovered :
    SoundnessFailureEvent inst tr → roundFailureUnion roundFailure inst.rounds

/-- Aggregate round-by-round soundness error bound up to `inst.rounds`. -/
def RoundByRoundSoundnessBoundary.totalRoundError
  {prob : ProbModel}
  {inst : Instance}
  {tr : Transcript}
  (hRbr : RoundByRoundSoundnessBoundary prob inst tr) : Rat :=
  roundErrorSum hRbr.epsRound inst.rounds

theorem soundnessFailureEvent_not
  (hSound : SoundnessAssumption)
  {inst : Instance}
  {tr : Transcript} :
  ¬ SoundnessFailureEvent inst tr := by
  intro hFail
  exact hFail.2 (hSound inst tr hFail.1)

theorem soundnessFailureAdvantage_eq_zero_of_soundness
  (prob : ProbModel)
  (hSound : SoundnessAssumption)
  {inst : Instance}
  {tr : Transcript} :
  SoundnessFailureAdvantage prob inst tr = 0 := by
  unfold SoundnessFailureAdvantage
  have hEventFalse : SoundnessFailureEvent inst tr → False := by
    exact soundnessFailureEvent_not hSound
  have hLeZero : prob.Pr (SoundnessFailureEvent inst tr) ≤ 0 := by
    calc
      prob.Pr (SoundnessFailureEvent inst tr) ≤ prob.Pr False := prob.prMonotone hEventFalse
      _ = 0 := prob.prFalse
  exact Rat.le_antisymm hLeZero (prob.prNonneg _)

/--
If soundness holds and `eps` is pointwise nonnegative, the soundness-failure
advantage is bounded by `eps`.
-/
theorem soundnessFailureAdvantageBound_of_soundness
  (hSound : SoundnessAssumption)
  {inst : Instance}
  {tr : Transcript}
  {eps : ErrorFn}
  (hEpsNonneg : ∀ n : Nat, 0 ≤ eps n) :
  SoundnessFailureAdvantageBound inst tr eps := by
  intro prob n
  have hZero :
      SoundnessFailureAdvantage prob inst tr = 0 :=
    soundnessFailureAdvantage_eq_zero_of_soundness prob hSound
  have hLeZero : SoundnessFailureAdvantage prob inst tr ≤ 0 := by
    simpa [hZero] using (show (0 : Rat) ≤ 0 by decide)
  exact Rat.le_trans hLeZero (hEpsNonneg n)

theorem pr_roundFailureUnion_le_roundErrorSum
  (prob : ProbModel)
  (E : Nat → Prop)
  (eps : Nat → Rat)
  (n : Nat)
  (hBound : ∀ i : Nat, i < n → prob.Pr (E i) ≤ eps i) :
  prob.Pr (roundFailureUnion E n) ≤ roundErrorSum eps n := by
  induction n with
  | zero =>
      simpa [roundFailureUnion, roundErrorSum, prob.prFalse] using (Rat.le_refl : (0 : Rat) ≤ 0)
  | succ n ih =>
      have hBoundPrev : ∀ i : Nat, i < n → prob.Pr (E i) ≤ eps i := by
        intro i hi
        exact hBound i (Nat.lt_trans hi (Nat.lt_succ_self n))
      have hBoundN : prob.Pr (E n) ≤ eps n := hBound n (Nat.lt_succ_self n)
      have hAddPrev :
          prob.Pr (roundFailureUnion E n) + prob.Pr (E n) ≤
            roundErrorSum eps n + prob.Pr (E n) := by
        exact (Rat.add_le_add_right (c := prob.Pr (E n))).2 (ih hBoundPrev)
      have hAddLast :
          roundErrorSum eps n + prob.Pr (E n) ≤
            roundErrorSum eps n + eps n := by
        exact (Rat.add_le_add_left (c := roundErrorSum eps n)).2 hBoundN
      calc
        prob.Pr (roundFailureUnion E (n + 1))
            = prob.Pr (roundFailureUnion E n ∨ E n) := by
                simp [roundFailureUnion]
        _ ≤ prob.Pr (roundFailureUnion E n) + prob.Pr (E n) := prob.prUnionLeAdd _ _
        _ ≤ roundErrorSum eps n + prob.Pr (E n) := hAddPrev
        _ ≤ roundErrorSum eps n + eps n := hAddLast
        _ = roundErrorSum eps (n + 1) := by
              simp [roundErrorSum]

theorem RoundByRoundSoundnessBoundary.soundnessFailureAdvantage_le_totalRoundError
  {prob : ProbModel}
  {inst : Instance}
  {tr : Transcript}
  (hRbr : RoundByRoundSoundnessBoundary prob inst tr) :
  SoundnessFailureAdvantage prob inst tr ≤ hRbr.totalRoundError := by
  unfold SoundnessFailureAdvantage RoundByRoundSoundnessBoundary.totalRoundError
  have hCover :
      prob.Pr (SoundnessFailureEvent inst tr) ≤
        prob.Pr (roundFailureUnion hRbr.roundFailure inst.rounds) := by
    exact prob.prMonotone hRbr.soundnessFailureCovered
  exact Rat.le_trans hCover
    (pr_roundFailureUnion_le_roundErrorSum
      prob hRbr.roundFailure hRbr.epsRound inst.rounds hRbr.roundFailureBound)

/--
Convert a concrete round-by-round bound into the theorem-facing advantage-bound
contract, for a fixed probability model.
-/
theorem RoundByRoundSoundnessBoundary.soundnessFailureAdvantageBound
  {prob : ProbModel}
  {inst : Instance}
  {tr : Transcript}
  {eps : ErrorFn}
  (hRbr : RoundByRoundSoundnessBoundary prob inst tr)
  (hTotalLe : ∀ n : Nat, hRbr.totalRoundError ≤ eps n) :
  ∀ n : Nat, SoundnessFailureAdvantage prob inst tr ≤ eps n := by
  intro n
  exact Rat.le_trans
    (hRbr.soundnessFailureAdvantage_le_totalRoundError)
    (hTotalLe n)

/-- Explicit soundness-error boundary surface for SumCheck. -/
structure SoundnessErrorBoundary where
  epsSoundness : ErrorFn
  nonnegEpsSoundness : ∀ n : Nat, 0 ≤ epsSoundness n
  negligibleEpsSoundness : IsNegligible epsSoundness

/--
Boundary-complete SumCheck theorem package:
- soundness/completeness are carried as typed parameters,
- soundness error surface is carried explicitly as a theorem-facing boundary.
-/
structure TheoremPackage
  (soundness : SoundnessAssumption)
  (completeness : CompletenessAssumption) where
  soundnessError : SoundnessErrorBoundary

/-- Project SumCheck soundness error function from theorem package. -/
def TheoremPackage.eps
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (hPkg : TheoremPackage soundness completeness) : ErrorFn :=
  hPkg.soundnessError.epsSoundness

/-- Project nonnegativity of the soundness-error function from theorem package. -/
theorem TheoremPackage.nonneg
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (hPkg : TheoremPackage soundness completeness) :
  ∀ n : Nat, 0 ≤ hPkg.eps n := by
  exact hPkg.soundnessError.nonnegEpsSoundness

/-- Project negligible soundness-error boundary from theorem package. -/
theorem TheoremPackage.negligible
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (hPkg : TheoremPackage soundness completeness) :
  IsNegligible hPkg.eps := by
  exact hPkg.soundnessError.negligibleEpsSoundness

/-- Soundness projection from theorem package plus acceptance witness. -/
theorem TheoremPackage.soundness
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (_hPkg : TheoremPackage soundness completeness)
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  ClaimTrue inst := by
  exact soundness inst tr hAccepted

/-- Completeness projection from theorem package plus claim-truth witness. -/
theorem TheoremPackage.completeness
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (_hPkg : TheoremPackage soundness completeness)
  {inst : Instance}
  (hClaim : ClaimTrue inst) :
  ∃ tr, Accepted inst tr := by
  exact completeness inst hClaim

theorem TheoremPackage.soundnessFailureAdvantage_eq_zero
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (hPkg : TheoremPackage soundness completeness)
  (prob : ProbModel)
  {inst : Instance}
  {tr : Transcript} :
  SoundnessFailureAdvantage prob inst tr = 0 := by
  apply soundnessFailureAdvantage_eq_zero_of_soundness (prob := prob)
  intro inst tr hAccepted
  exact hPkg.soundness hAccepted

/--
Theorem-package soundness implies a full theorem-facing soundness-failure
advantage bound against the package error function.
-/
theorem TheoremPackage.soundnessFailureAdvantageBound
  {soundness : SoundnessAssumption}
  {completeness : CompletenessAssumption}
  (hPkg : TheoremPackage soundness completeness)
  {inst : Instance}
  {tr : Transcript} :
  SoundnessFailureAdvantageBound inst tr hPkg.eps := by
  intro prob n
  have hZero :
      SoundnessFailureAdvantage prob inst tr = 0 :=
    hPkg.soundnessFailureAdvantage_eq_zero prob
  have hLeZero : SoundnessFailureAdvantage prob inst tr ≤ 0 := by
    simpa [hZero] using (show (0 : Rat) ≤ 0 by decide)
  exact Rat.le_trans hLeZero (hPkg.nonneg n)

theorem accepted_rounds_eq
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  tr.roundPolys.size = inst.rounds := by
  exact SingleRound.accepted_rounds_eq hAccepted

theorem accepted_challenges_eq
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  tr.challenges.size = tr.roundPolys.size := by
  exact SingleRound.accepted_challenges_eq hAccepted

theorem accepted_fold_step
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
      SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
    SuperNeo.sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) := by
  exact SingleRound.accepted_fold_step hAccepted hi

theorem accepted_initial_round
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  InitialRoundConsistent inst tr := by
  exact SingleRound.accepted_initial_round hAccepted

theorem accepted_round_sum_step
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
      SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
    SuperNeo.sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) := by
  exact SingleRound.accepted_round_sum_step hAccepted hi

/-- Soundness theorem surface (assumption-instantiated). -/
theorem soundness
  (h : SoundnessAssumption)
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  ClaimTrue inst := by
  exact h inst tr hAccepted

/-- Completeness theorem surface (assumption-instantiated). -/
theorem completeness
  (h : CompletenessAssumption)
  {inst : Instance}
  (hClaim : ClaimTrue inst) :
  ∃ tr, Accepted inst tr := by
  exact h inst hClaim

/--
Canonical constructor for the constructive SumCheck closure path from
`SuperNeo.SumCheck`.
-/
def theoremPackage_constructive
  (soundnessError : SoundnessErrorBoundary) :
  TheoremPackage
    SuperNeo.sumcheckSoundness_constructive
    SuperNeo.sumcheckCompleteness_constructive where
  soundnessError := soundnessError

/--
Canonical zero-error soundness boundary for the constructive SumCheck closure
path.
-/
def soundnessErrorBoundary_zero : SoundnessErrorBoundary where
  epsSoundness := fun _ => 0
  nonnegEpsSoundness := by
    intro n
    exact (show (0 : Rat) ≤ 0 by decide)
  negligibleEpsSoundness := by
    simpa using (isNegligible_zero : IsNegligible (fun _ => (0 : Rat)))

/-- Canonical constructive theorem package with zero soundness error. -/
def theoremPackage_constructive_zeroError :
  TheoremPackage
    SuperNeo.sumcheckSoundness_constructive
    SuperNeo.sumcheckCompleteness_constructive :=
  theoremPackage_constructive soundnessErrorBoundary_zero

end Sumcheck

end SuperNeo.ProofSystem
