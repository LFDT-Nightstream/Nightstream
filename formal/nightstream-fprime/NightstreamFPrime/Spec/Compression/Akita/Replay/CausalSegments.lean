import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinStage
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentGiven

/-! The messages of a path before its draw `n + 1` as a function of the values that the code has
before this draw (Akita revision `d50f700`; `docs/akita-work/w20/design.md`, module W;
`Extract/PresentGiven.lean`, `Replay/SumCheck.lean`, `Replay/Range.lean`, `Replay/LevelFold.lean`,
`Replay/Tensor.lean`).

- `Extract.upto path n` is the path before its draw `n + 1`. `Extract.Same n path other`: the two
  paths are equal before their draw `n + 1`. `Same.segments`: they have the same
  `Extract.segments … j` for each `j ≤ n`. `Same.append`, `Same.of_lt`, `Same.recv`, `Same.draw`,
  `Same.of_leading`: the rules for `++` and for the first move.
- `plainPath_same`, `eqCodePath_same`: two lists of rounds with the same first `r` rounds and the
  same message of the round `r` give the same path before the draw `r + 1`.
- `treeThen` is the path of the product stages and then a path `u` when the stages pass
  (`foldPath_eq`, `foldPathSq_eq`). `treeThen_same`: equal first stages. `treeThen_rounds` and
  `treeThen_children`: the rounds and the children of the first different stage.
- `leafPath_rounds`, `normThen_rounds`, `fusedPath_rounds`, `tensorPath_rounds` and the lemmas
  after them: the positions in the last stage, in the fused sum-check and in the tensor
  reduction.
- `treeFull`, `normFull`, `tensorFull`, `closingFull`: the paths of a fold whose checks pass, with
  no condition.
- `Need` is one value that the path of a level reads, with the draw from which the path reads it.
  `Need.Holds j s t`: two states have the same value when the draw is at most `j`. `needs20 ℓ`
  are the needs of the path of the level `ℓ` of `key20.aks` (`Replay/CausalCells.lean`).
- `Need.check k₀ L` is the check of a need on the skeleton of `layout20`: no operation from the
  work of its stage to `L` writes the key, or the challenge is drawn before the work of its stage.
  `needs20_checked₀` to `needs20_checked₅`: each need of each level has the check. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Compression.Game (Move)

/-- The path before its draw `n + 1`: the first `n` draws and each message before the next
draw. -/
def upto : List Move → ℕ → List Move
  | [], _ => []
  | Move.recv message :: path, n => Move.recv message :: upto path n
  | Move.draw _ :: _, 0 => []
  | Move.draw challenge :: path, n + 1 => Move.draw challenge :: upto path n

/-- The two paths are equal before their draw `n + 1`. -/
def Same (n : ℕ) (path other : List Move) : Prop := upto path n = upto other n

theorem upto_nil (n : ℕ) : upto [] n = [] := by cases n <;> rfl

theorem upto_recv {M : Type} (message : M) (path : List Move) (n : ℕ) :
    upto (Move.recv message :: path) n = Move.recv message :: upto path n := by
  cases n <;> rfl

theorem upto_draw_zero {C : Type} (challenge : C) (path : List Move) :
    upto (Move.draw challenge :: path) 0 = [] := rfl

theorem upto_draw_succ {C : Type} (challenge : C) (path : List Move) (n : ℕ) :
    upto (Move.draw challenge :: path) (n + 1) = Move.draw challenge :: upto path n := rfl

theorem upto_upto (path : List Move) : ∀ {m n : ℕ}, m ≤ n → upto (upto path n) m = upto path m := by
  induction path with
  | nil =>
      intro m n _
      simp only [upto_nil]
  | cons move path induction =>
      intro m n le
      cases move with
      | recv message =>
          simp only [upto_recv]
          rw [induction le]
      | draw challenge =>
          cases n with
          | zero =>
              obtain rfl := Nat.le_zero.mp le
              simp only [upto_draw_zero, upto_nil]
          | succ n =>
              cases m with
              | zero => simp only [upto_draw_succ, upto_draw_zero]
              | succ m =>
                  simp only [upto_draw_succ]
                  rw [induction (Nat.le_of_succ_le_succ le)]

theorem segments_nil (j : ℕ) : segments [] j = [] := by cases j <;> rfl

theorem segments_recv_zero {M : Type} (message : M) (path : List Move) :
    segments (Move.recv message :: path) 0 = Move.recv message :: segments path 0 := by
  unfold segments
  rw [afterDraws_zero, afterDraws_zero]
  rfl

theorem segments_recv_succ {M : Type} (message : M) (path : List Move) (j : ℕ) :
    segments (Move.recv message :: path) (j + 1) = segments path (j + 1) := rfl

theorem segments_draw_zero {C : Type} (challenge : C) (path : List Move) :
    segments (Move.draw challenge :: path) 0 = [] := rfl

theorem segments_draw_succ {C : Type} (challenge : C) (path : List Move) (j : ℕ) :
    segments (Move.draw challenge :: path) (j + 1) = segments path j := rfl

/-- The messages after the draw `j ≤ n` are messages of the path before its draw `n + 1`. -/
theorem segments_upto (path : List Move) :
    ∀ {j n : ℕ}, j ≤ n → segments (upto path n) j = segments path j := by
  induction path with
  | nil =>
      intro j n _
      rw [upto_nil]
  | cons move path induction =>
      intro j n le
      cases move with
      | recv message =>
          rw [upto_recv]
          cases j with
          | zero => rw [segments_recv_zero, segments_recv_zero, induction le]
          | succ j => rw [segments_recv_succ, segments_recv_succ, induction le]
      | draw challenge =>
          cases n with
          | zero =>
              obtain rfl := Nat.le_zero.mp le
              rw [upto_draw_zero, segments_nil, segments_draw_zero]
          | succ n =>
              rw [upto_draw_succ]
              cases j with
              | zero => rw [segments_draw_zero, segments_draw_zero]
              | succ j =>
                  rw [segments_draw_succ, segments_draw_succ,
                    induction (Nat.le_of_succ_le_succ le)]

theorem upto_append_of_lt (first second : List Move) :
    ∀ {n : ℕ}, n < countDraws first → upto (first ++ second) n = upto first n := by
  induction first with
  | nil => exact fun lt => absurd lt (Nat.not_lt_zero _)
  | cons move first induction =>
      intro n lt
      cases move with
      | recv message =>
          rw [List.cons_append, upto_recv, upto_recv, induction lt]
      | draw challenge =>
          cases n with
          | zero => rfl
          | succ n =>
              rw [List.cons_append, upto_draw_succ, upto_draw_succ,
                induction (Nat.lt_of_succ_lt_succ lt)]

theorem upto_append (first second : List Move) (n : ℕ) :
    upto (first ++ second) (countDraws first + n) = first ++ upto second n := by
  induction first with
  | nil =>
      show upto second (0 + n) = upto second n
      rw [Nat.zero_add]
  | cons move first induction =>
      cases move with
      | recv message =>
          show upto (Move.recv message :: (first ++ second)) (countDraws first + n) = _
          rw [upto_recv, induction]
          rfl
      | draw challenge =>
          show upto (Move.draw challenge :: (first ++ second)) (countDraws first + 1 + n) = _
          rw [show countDraws first + 1 + n = countDraws first + n + 1 by omega, upto_draw_succ,
            induction]
          rfl

namespace Same

theorem refl (n : ℕ) (path : List Move) : Same n path path := rfl

theorem symm {n : ℕ} {path other : List Move} (same : Same n path other) : Same n other path :=
  Eq.symm same

theorem trans {n : ℕ} {a b c : List Move} (first : Same n a b) (second : Same n b c) :
    Same n a c :=
  Eq.trans first second

theorem mono {n m : ℕ} {path other : List Move} (same : Same n path other) (le : m ≤ n) :
    Same m path other := by
  have equal : upto path n = upto other n := same
  show upto path m = upto other m
  rw [← upto_upto path le, equal, upto_upto other le]

/-- Two paths that are equal before their draw `n + 1` have the same messages after each draw
`j ≤ n`. -/
theorem segments {n j : ℕ} {path other : List Move} (same : Same n path other) (le : j ≤ n) :
    Extract.segments path j = Extract.segments other j := by
  have equal : upto path n = upto other n := same
  rw [← segments_upto path le, equal, segments_upto other le]

theorem recv {n : ℕ} {path other : List Move} {M : Type} (message : M) (same : Same n path other) :
    Same n (Move.recv message :: path) (Move.recv message :: other) := by
  have equal : upto path n = upto other n := same
  show upto _ n = upto _ n
  rw [upto_recv, upto_recv, equal]

theorem draw {n : ℕ} {path other : List Move} {C : Type} (challenge : C)
    (same : Same n path other) :
    Same (n + 1) (Move.draw challenge :: path) (Move.draw challenge :: other) := by
  have equal : upto path n = upto other n := same
  show upto _ (n + 1) = upto _ (n + 1)
  rw [upto_draw_succ, upto_draw_succ, equal]

/-- Two paths that start with a draw: nothing before the first draw, and after it the same
challenge and equal paths. -/
theorem draw_of {j : ℕ} {path other : List Move} {C : Type} {challenge challenge' : C}
    (later : 1 ≤ j → challenge = challenge' ∧ Same (j - 1) path other) :
    Same j (Move.draw challenge :: path) (Move.draw challenge' :: other) := by
  cases j with
  | zero => rfl
  | succ j =>
      obtain ⟨rfl, same⟩ := later (Nat.succ_le_succ (Nat.zero_le j))
      exact same.draw _

/-- Paths with the same messages before the first draw are equal before it. -/
theorem of_leading : ∀ {path other : List Move}, leading path = leading other →
    Same 0 path other
  | [], [], _ => rfl
  | [], Move.draw _ :: _, _ => rfl
  | [], Move.recv _ :: _, equal => by cases equal
  | Move.draw _ :: _, [], _ => rfl
  | Move.draw _ :: _, Move.draw _ :: _, _ => rfl
  | Move.draw _ :: _, Move.recv _ :: _, equal => by cases equal
  | Move.recv _ :: _, [], equal => by cases equal
  | Move.recv _ :: _, Move.draw _ :: _, equal => by cases equal
  | Move.recv a :: path, Move.recv b :: other, equal => by
      have parts : Move.recv a = Move.recv b ∧ leading path = leading other :=
        List.cons_eq_cons.mp equal
      rw [parts.1]
      exact (of_leading parts.2).recv b

/-- A common front with `countDraws first` draws. -/
theorem append (first : List Move) {n : ℕ} {path other : List Move} (same : Same n path other) :
    Same (countDraws first + n) (first ++ path) (first ++ other) := by
  have equal : upto path n = upto other n := same
  show upto _ _ = upto _ _
  rw [upto_append, upto_append, equal]

/-- A common front, and paths after it that are equal before their first draw. -/
theorem append_zero (first : List Move) {path other : List Move} (same : Same 0 path other) :
    Same (countDraws first) (first ++ path) (first ++ other) :=
  append first same

/-- Fronts that are equal before a draw that they have. -/
theorem of_lt {n : ℕ} {first first' : List Move} (second second' : List Move)
    (same : Same n first first') (lt : n < countDraws first) (lt' : n < countDraws first') :
    Same n (first ++ second) (first' ++ second') := by
  have equal : upto first n = upto first' n := same
  show upto _ n = upto _ n
  rw [upto_append_of_lt first second lt, upto_append_of_lt first' second' lt', equal]

end Same

end NightstreamFPrime.Spec.Compression.Akita.Extract

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Same countDraws leading)

section Rounds

variable {K : Type} [Field K] (φ : Ext → K)

theorem plainPath_cons (d : ℕ) (claim : Ext) (message : List Ext) (r : Ext)
    (rounds : List (List Ext × Ext)) :
    plainPath φ d claim ((message, r) :: rounds) =
      Move.recv (coefficients (d + 1) (plainPoly (φ claim) (message.map φ))) ::
        Move.draw (φ r) :: plainPath φ d (plainStep claim message r) rounds := rfl

theorem eqCodePath_cons (D : ℕ) (message : List Ext) (r : Ext) (rounds : List (List Ext × Ext)) :
    eqCodePath φ D ((message, r) :: rounds) =
      Move.recv (listFn D (message.map φ)) :: Move.draw (φ r) :: eqCodePath φ D rounds := rfl

theorem plainPath_draws (d : ℕ) : ∀ (rounds : List (List Ext × Ext)) (claim : Ext),
    countDraws (plainPath φ d claim rounds) = rounds.length
  | [], _ => rfl
  | (message, r) :: rounds, claim => by
      rw [plainPath_cons]
      show countDraws (plainPath φ d (plainStep claim message r) rounds) + 1 = rounds.length + 1
      rw [plainPath_draws d rounds]

theorem eqCodePath_draws (D : ℕ) : ∀ rounds : List (List Ext × Ext),
    countDraws (eqCodePath φ D rounds) = rounds.length
  | [] => rfl
  | (message, r) :: rounds => by
      rw [eqCodePath_cons]
      show countDraws (eqCodePath φ D rounds) + 1 = rounds.length + 1
      rw [eqCodePath_draws D rounds]

/-- Two lists of rounds with the same first `r` rounds and the same message of the round `r`
give the same plain path before its draw `r + 1`, for each paths after them. -/
theorem plainPath_same (d : ℕ) (after after' : List Move) :
    ∀ (r : ℕ) (rounds rounds' : List (List Ext × Ext)) (claim : Ext), r < rounds.length →
      rounds.take r = rounds'.take r → rounds[r]?.map Prod.fst = rounds'[r]?.map Prod.fst →
      Same r (plainPath φ d claim rounds ++ after) (plainPath φ d claim rounds' ++ after')
  | 0, [], _, _, lt, _, _ => absurd lt (Nat.not_lt_zero _)
  | 0, (message, x) :: rounds, [], _, _, _, equal => by cases equal
  | 0, (message, x) :: rounds, (message', x') :: rounds', claim, _, _, equal => by
      have same : message = message' := by
        simpa only [List.getElem?_cons_zero, Option.map_some, Option.some.injEq] using equal
      subst same
      rw [plainPath_cons, plainPath_cons]
      exact (Same.of_leading rfl).recv _
  | r + 1, [], _, _, lt, _, _ => absurd lt (Nat.not_lt_zero _)
  | r + 1, (message, x) :: rounds, [], _, _, front, _ => by cases front
  | r + 1, (message, x) :: rounds, y :: rounds', claim, lt, front, equal => by
      have parts : (message, x) = y ∧ rounds.take r = rounds'.take r := List.cons_eq_cons.mp front
      obtain ⟨rfl, taken⟩ := parts
      rw [plainPath_cons, plainPath_cons]
      exact ((plainPath_same d after after' r rounds rounds' _ (Nat.lt_of_succ_lt_succ lt) taken
        (by simpa only [List.getElem?_cons_succ] using equal)).draw _).recv _

/-- `plainPath_same` for the path of the equality-factored messages. -/
theorem eqCodePath_same (D : ℕ) (after after' : List Move) :
    ∀ (r : ℕ) (rounds rounds' : List (List Ext × Ext)), r < rounds.length →
      rounds.take r = rounds'.take r → rounds[r]?.map Prod.fst = rounds'[r]?.map Prod.fst →
      Same r (eqCodePath φ D rounds ++ after) (eqCodePath φ D rounds' ++ after')
  | 0, [], _, lt, _, _ => absurd lt (Nat.not_lt_zero _)
  | 0, (message, x) :: rounds, [], _, _, equal => by cases equal
  | 0, (message, x) :: rounds, (message', x') :: rounds', _, _, equal => by
      have same : message = message' := by
        simpa only [List.getElem?_cons_zero, Option.map_some, Option.some.injEq] using equal
      subst same
      rw [eqCodePath_cons, eqCodePath_cons]
      exact (Same.of_leading rfl).recv _
  | r + 1, [], _, lt, _, _ => absurd lt (Nat.not_lt_zero _)
  | r + 1, (message, x) :: rounds, [], _, front, _ => by cases front
  | r + 1, (message, x) :: rounds, y :: rounds', lt, front, equal => by
      have parts : (message, x) = y ∧ rounds.take r = rounds'.take r := List.cons_eq_cons.mp front
      obtain ⟨rfl, taken⟩ := parts
      rw [eqCodePath_cons, eqCodePath_cons]
      exact ((eqCodePath_same D after after' r rounds rounds' (Nat.lt_of_succ_lt_succ lt) taken
        (by simpa only [List.getElem?_cons_succ] using equal)).draw _).recv _

/-- The two lists of rounds have the same first `r` rounds and the same message of the round
`r`, and the first list has a round `r`. -/
def Close (r : ℕ) (rounds rounds' : List (List Ext × Ext)) : Prop :=
  r < rounds.length ∧ rounds.take r = rounds'.take r ∧
    rounds[r]?.map Prod.fst = rounds'[r]?.map Prod.fst

/-- The plain path for each draw index: close rounds before the end of the rounds, and equal
rounds and equal paths after them from the end. -/
theorem plainPath_agree (d : ℕ) {rounds rounds' : List (List Ext × Ext)} {after after' : List Move}
    {j : ℕ} (claim : Ext) (close : j < rounds.length → Close j rounds rounds')
    (full : rounds.length ≤ j → rounds = rounds' ∧ Same (j - rounds.length) after after') :
    Same j (plainPath φ d claim rounds ++ after) (plainPath φ d claim rounds' ++ after') := by
  by_cases lt : j < rounds.length
  · exact plainPath_same φ d _ _ j _ _ _ (close lt).1 (close lt).2.1 (close lt).2.2
  · obtain ⟨rfl, same⟩ := full (Nat.le_of_not_lt lt)
    have whole := Same.append (plainPath φ d claim rounds) same
    rw [plainPath_draws, show rounds.length + (j - rounds.length) = j by omega] at whole
    exact whole

/-- `plainPath_agree` for the path of the equality-factored messages. -/
theorem eqCodePath_agree (D : ℕ) {rounds rounds' : List (List Ext × Ext)}
    {after after' : List Move} {j : ℕ} (close : j < rounds.length → Close j rounds rounds')
    (full : rounds.length ≤ j → rounds = rounds' ∧ Same (j - rounds.length) after after') :
    Same j (eqCodePath φ D rounds ++ after) (eqCodePath φ D rounds' ++ after') := by
  by_cases lt : j < rounds.length
  · exact eqCodePath_same φ D _ _ j _ _ (close lt).1 (close lt).2.1 (close lt).2.2
  · obtain ⟨rfl, same⟩ := full (Nat.le_of_not_lt lt)
    have whole := Same.append (eqCodePath φ D rounds) same
    rw [eqCodePath_draws, show rounds.length + (j - rounds.length) = j by omega] at whole
    exact whole

end Rounds

section Tree

variable {K : Type} [Field K] (φ : Ext → K) {β β' : Type}

/-- The end of the rounds of a product stage, when its end check holds. -/
def stageOut (S : Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext) :
    Option Ended :=
  match eqRun claim Ext.one point S.rounds with
  | none => none
  | some ended =>
      if ended.claim.same (stageEnd nodes S.arity weights S.children) then some ended else none

/-- The path of the product stages, and then the path `after` when the stages and the last stage
`run` pass. -/
noncomputable def treeThen (last : ℕ → (ℕ → Ext) → List Ext → Ext → List Move)
    (run : ℕ → (ℕ → Ext) → List Ext → Ext → Option β) (after : List Move)
    (stages : List Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext) :
    List Move :=
  treePath φ last stages nodes weights point claim ++
    if (treeRun run stages nodes weights point claim).isSome then after else []

variable (last last' : ℕ → (ℕ → Ext) → List Ext → Ext → List Move)
  (run : ℕ → (ℕ → Ext) → List Ext → Ext → Option β)
  (run' : ℕ → (ℕ → Ext) → List Ext → Ext → Option β') (after after' : List Move)

theorem treeThen_nil (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext) :
    treeThen φ last run after [] nodes weights point claim =
      last nodes weights point claim ++
        if (run nodes weights point claim).isSome then after else [] := rfl

theorem treeThen_cons (S : Product) (rest : List Product) (nodes : ℕ) (weights : ℕ → Ext)
    (point : List Ext) (claim : Ext) :
    treeThen φ last run after (S :: rest) nodes weights point claim =
      eqCodePath φ S.arity S.rounds ++
        Move.recv (listFn (nodes * S.arity) (S.children.map φ)) ::
          match stageOut S nodes weights point claim with
          | none => []
          | some ended => Move.draw (φ S.gamma) ::
              treeThen φ last run after rest (nodes * S.arity) (powExt S.gamma) ended.point
                (batched (nodes * S.arity) S.children S.gamma) := by
  show (eqCodePath φ S.arity S.rounds ++
      Move.recv (listFn (nodes * S.arity) (S.children.map φ)) ::
        match eqRun claim Ext.one point S.rounds with
        | none => []
        | some ended =>
            if ended.claim.same (stageEnd nodes S.arity weights S.children) then
              Move.draw (φ S.gamma) :: treePath φ last rest (nodes * S.arity) (powExt S.gamma)
                ended.point (batched (nodes * S.arity) S.children S.gamma)
            else []) ++
    (if (match eqRun claim Ext.one point S.rounds with
        | none => none
        | some ended =>
            if ended.claim.same (stageEnd nodes S.arity weights S.children) then
              treeRun run rest (nodes * S.arity) (powExt S.gamma) ended.point
                (batched (nodes * S.arity) S.children S.gamma)
            else none).isSome then after else []) = _
  unfold stageOut
  cases eqRun claim Ext.one point S.rounds with
  | none => simp
  | some ended =>
      by_cases holds : ended.claim.same (stageEnd nodes S.arity weights S.children) = true
      · simp [holds, treeThen]
      · simp [holds]

variable {last last' run run' after after'}

/-- Equal first stages: the paths of two lists of stages with the common front `front` are equal
before the draw `n + 1` after the draws of `front`, when the paths of the two tails are equal
before their draw `n + 1` from each state of the tree. -/
theorem treeThen_same {tail tail' : List Product} {n : ℕ}
    (same : ∀ nodes weights point claim,
      Same n (treeThen φ last run after tail nodes weights point claim)
        (treeThen φ last' run' after' tail' nodes weights point claim)) :
    ∀ (front : List Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext),
      Same ((front.map fun S => S.rounds.length + 1).sum + n)
        (treeThen φ last run after (front ++ tail) nodes weights point claim)
        (treeThen φ last' run' after' (front ++ tail') nodes weights point claim)
  | [], nodes, weights, point, claim => by
      simpa using same nodes weights point claim
  | S :: front, nodes, weights, point, claim => by
      have count : ((S :: front).map fun S => S.rounds.length + 1).sum + n =
          countDraws (eqCodePath φ S.arity S.rounds) +
            ((front.map fun S => S.rounds.length + 1).sum + n + 1) := by
        rw [eqCodePath_draws, List.map_cons, List.sum_cons]
        omega
      rw [List.cons_append, List.cons_append, treeThen_cons, treeThen_cons, count]
      refine Same.append _ (Same.recv _ ?_)
      cases stageOut S nodes weights point claim with
      | none => exact Same.refl _ _
      | some ended => exact (treeThen_same same front _ _ _ _).draw _

/-- The rounds of the first stage: two first stages with the same arity and close rounds. -/
theorem treeThen_rounds {S S' : Product} (rest rest' : List Product) (arity : S.arity = S'.arity)
    {r : ℕ} (close : Close r S.rounds S'.rounds) (nodes : ℕ) (weights : ℕ → Ext)
    (point : List Ext) (claim : Ext) :
    Same r (treeThen φ last run after (S :: rest) nodes weights point claim)
      (treeThen φ last' run' after' (S' :: rest') nodes weights point claim) := by
  rw [treeThen_cons, treeThen_cons, ← arity]
  exact eqCodePath_same φ S.arity _ _ r _ _ close.1 close.2.1 close.2.2

/-- The claims of the children of the first stage: two first stages with the same arity, rounds
and children. -/
theorem treeThen_children {S S' : Product} (rest rest' : List Product)
    (arity : S.arity = S'.arity) (rounds : S.rounds = S'.rounds)
    (children : S.children = S'.children) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext)
    (claim : Ext) :
    Same S.rounds.length (treeThen φ last run after (S :: rest) nodes weights point claim)
      (treeThen φ last' run' after' (S' :: rest') nodes weights point claim) := by
  have out : stageOut S' nodes weights point claim = stageOut S nodes weights point claim := by
    unfold stageOut
    rw [rounds, children, arity]
  rw [treeThen_cons, treeThen_cons, out, ← arity, ← rounds, ← children,
    ← eqCodePath_draws φ S.arity S.rounds]
  refine Same.append_zero _ (Same.recv _ ?_)
  cases stageOut S nodes weights point claim with
  | none => exact Same.refl _ _
  | some ended => exact Same.of_leading rfl

end Tree

section Last

variable {K : Type} [Field K] (φ : Ext → K)

theorem leading_ite (c : Prop) [Decidable c] {after : List Move} (headed : leading after = []) :
    leading (if c then after else []) = [] := by
  split
  · exact headed
  · rfl

theorem leafPath_draws (leaf : ℕ) (rounds : List (List Ext × Ext)) (image : Ext) :
    countDraws (leafPath φ leaf rounds image) = rounds.length := by
  unfold leafPath
  rw [Extract.countDraws_append, eqCodePath_draws]
  rfl

/-- The rounds of the leaf stage. -/
theorem leafPath_rounds (leaf : ℕ) {rounds rounds' : List (List Ext × Ext)} (image image' : Ext)
    (after after' : List Move) {r : ℕ} (close : Close r rounds rounds') :
    Same r (leafPath φ leaf rounds image ++ after) (leafPath φ leaf rounds' image' ++ after') := by
  unfold leafPath
  rw [List.append_assoc, List.append_assoc]
  exact eqCodePath_same φ leaf _ _ r _ _ close.1 close.2.1 close.2.2

/-- After the rounds of the leaf stage: its value `s_claim` and the path after it. -/
theorem leafPath_after (leaf : ℕ) (rounds : List (List Ext × Ext)) (image : Ext)
    {after after' : List Move} {n : ℕ} (same : Same n after after') :
    Same (rounds.length + n) (leafPath φ leaf rounds image ++ after)
      (leafPath φ leaf rounds image ++ after') := by
  rw [← leafPath_draws φ leaf rounds image]
  exact Same.append _ same

theorem normRun_none {leaf limit energy : ℕ} (merge : Ext) (rounds : List (List Ext × Ext))
    (virtual image : Ext) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext)
    (over : ¬energy ≤ limit) :
    normRun leaf limit energy merge rounds virtual image nodes weights point claim = none := by
  unfold normRun energyFits
  simp [over]

/-- The path of the norm stage, and then the path `after` when the stage passes. -/
noncomputable def normThen (leaf limit energy : ℕ) (merge : Ext)
    (rounds : List (List Ext × Ext)) (virtual image : Ext) (after : List Move) (nodes : ℕ)
    (weights : ℕ → Ext) (point : List Ext) (claim : Ext) : List Move :=
  normPath φ leaf limit energy merge rounds virtual image claim ++
    if (normRun leaf limit energy merge rounds virtual image nodes weights point claim).isSome
    then after else []

/-- The value `E_resp` of the norm stage. -/
theorem normThen_energy (leaf limit energy : ℕ) (merge merge' : Ext)
    (rounds rounds' : List (List Ext × Ext)) (virtual virtual' image image' : Ext)
    {after after' : List Move} (headed : leading after = []) (headed' : leading after' = [])
    (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim claim' : Ext) :
    Same 0 (normThen φ leaf limit energy merge rounds virtual image after nodes weights point claim)
      (normThen φ leaf limit energy merge' rounds' virtual' image' after' nodes weights point
        claim') := by
  unfold normThen normPath
  by_cases fits : energy ≤ limit
  · rw [ite_eq_left fits, ite_eq_left fits]
    exact (Same.of_leading rfl).recv _
  · rw [ite_eq_right fits, ite_eq_right fits]
    exact (Same.of_leading ((leading_ite _ headed).trans (leading_ite _ headed').symm)).recv _

/-- The rounds of the norm stage. -/
theorem normThen_rounds (leaf limit energy : ℕ) (merge : Ext)
    {rounds rounds' : List (List Ext × Ext)} (virtual virtual' image image' : Ext)
    (after after' : List Move) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext)
    {r : ℕ} (close : Close r rounds rounds') :
    Same (r + 1)
      (normThen φ leaf limit energy merge rounds virtual image after nodes weights point claim)
      (normThen φ leaf limit energy merge rounds' virtual' image' after' nodes weights point
        claim) := by
  unfold normThen
  by_cases fits : energy ≤ limit
  · unfold normPath
    rw [ite_eq_left fits, ite_eq_left fits, List.cons_append, List.cons_append, List.cons_append,
      List.cons_append, List.append_assoc, List.append_assoc]
    exact ((plainPath_same φ leaf _ _ r _ _ _ close.1 close.2.1 close.2.2).draw _).recv _
  · rw [normRun_none _ _ _ _ _ _ _ _ fits, normRun_none _ _ _ _ _ _ _ _ fits]
    unfold normPath
    rw [ite_eq_right fits, ite_eq_right fits]
    exact Same.refl _ _

/-- After the rounds of the norm stage: its values `z̃(r_virt)` and `s_claim`, and the path after
it. -/
theorem normThen_after (leaf limit energy : ℕ) (merge : Ext) (rounds : List (List Ext × Ext))
    (virtual image : Ext) {after after' : List Move} {n : ℕ} (same : Same n after after')
    (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext) :
    Same (rounds.length + 1 + n)
      (normThen φ leaf limit energy merge rounds virtual image after nodes weights point claim)
      (normThen φ leaf limit energy merge rounds virtual image after' nodes weights point
        claim) := by
  unfold normThen
  by_cases fits : energy ≤ limit
  · have count : countDraws (normPath φ leaf limit energy merge rounds virtual image claim) =
        rounds.length + 1 := by
      unfold normPath
      rw [ite_eq_left fits]
      show countDraws (plainPath φ leaf _ rounds ++ [Move.recv (φ virtual, φ image)]) + 1 = _
      rw [Extract.countDraws_append, plainPath_draws]
      rfl
    rw [← count]
    refine Same.append _ ?_
    split
    · exact same
    · exact Same.refl _ _
  · rw [normRun_none _ _ _ _ _ _ _ _ fits]
    exact Same.refl _ _

end Last

section Fused

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E]

theorem fusedPath_draws (claim : Ext) (rounds : List (List Ext × Ext))
    (finish : List Ext → FusedEnd) :
    countDraws (fusedPath E claim rounds finish) = rounds.length := by
  unfold fusedPath
  rw [Extract.countDraws_append, plainPath_draws]
  rfl

/-- The rounds of the fused sum-check. -/
theorem fusedPath_rounds (claim : Ext) {rounds rounds' : List (List Ext × Ext)}
    (finish finish' : List Ext → FusedEnd) (after after' : List Move) {r : ℕ}
    (close : Close r rounds rounds') :
    Same r (fusedPath E claim rounds finish ++ after)
      (fusedPath E claim rounds' finish' ++ after') := by
  unfold fusedPath
  rw [List.append_assoc, List.append_assoc]
  exact plainPath_same _ 2 _ _ r _ _ _ close.1 close.2.1 close.2.2

/-- After the rounds of the fused sum-check: the message `v_W` and the path after it. -/
theorem fusedPath_after (claim : Ext) (rounds : List (List Ext × Ext))
    {finish finish' : List Ext → FusedEnd}
    (value : (finish (rounds.map (·.2))).value = (finish' (rounds.map (·.2))).value)
    {after after' : List Move} {n : ℕ} (same : Same n after after') :
    Same (rounds.length + n) (fusedPath E claim rounds finish ++ after)
      (fusedPath E claim rounds finish' ++ after') := by
  have paths : fusedPath E claim rounds finish' = fusedPath E claim rounds finish := by
    unfold fusedPath
    rw [value]
  rw [paths, ← fusedPath_draws E claim rounds finish]
  exact Same.append _ same

/-- The column partials of the tensor reduction, before the draw of `η`. -/
theorem tensorPath_partials (ν : ℕ) (point : List Ext) (v p0 p1 η η' : Ext)
    (rounds rounds' : List (List Ext × Ext)) (final final' : Ext) :
    Same 0 (tensorPath E ν point v p0 p1 η rounds final)
      (tensorPath E ν point v p0 p1 η' rounds' final') := by
  unfold tensorPath
  cases point with
  | nil => exact Same.refl _ _
  | cons head tail =>
      by_cases long : tail.length = ν
      · simp only [ite_eq_left long]
        by_cases opens : v.same ((Ext.one - head) * p0 + head * p1) = true
        · simp only [ite_eq_left opens]
          exact (Same.of_leading rfl).recv _
        · simp only [ite_eq_right opens]
          exact Same.refl _ _
      · simp only [ite_eq_right long]
        exact Same.refl _ _

/-- The rounds of the tensor reduction. -/
theorem tensorPath_rounds (ν : ℕ) (point : List Ext) (v p0 p1 η : Ext)
    {rounds rounds' : List (List Ext × Ext)} (final final' : Ext) {r : ℕ}
    (close : Close r rounds rounds') :
    Same (r + 1) (tensorPath E ν point v p0 p1 η rounds final)
      (tensorPath E ν point v p0 p1 η rounds' final') := by
  unfold tensorPath
  cases point with
  | nil => exact Same.refl _ _
  | cons head tail =>
      by_cases long : tail.length = ν
      · simp only [ite_eq_left long]
        by_cases opens : v.same ((Ext.one - head) * p0 + head * p1) = true
        · simp only [ite_eq_left opens]
          exact ((plainPath_same _ 1 _ _ r _ _ _ close.1 close.2.1 close.2.2).draw _).recv _
        · simp only [ite_eq_right opens]
          exact Same.refl _ _
      · simp only [ite_eq_right long]
        exact Same.refl _ _

end Fused

section Folds

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E]

omit [Field E] [Algebra Zq E] [Degree2 E] in
theorem leading_binPath (bin : Bool) (ζ : E) (after : List Move) {C : Type} (challenge : C) :
    leading (binPath bin ζ ++ Move.draw challenge :: after) = [] := by
  cases bin <;> rfl

/-- `foldPath` is the three first draws and `treeThen` of the leaf stage. -/
theorem foldPath_eq (vars batch low b : ℕ) (bin : Bool) (W : FoldWords)
    (leaf : List (List Ext × Ext)) (claim : Ext) (finish : List Ext → FusedEnd) :
    foldPath E vars batch low b bin W leaf claim finish =
      Move.draw (W.alpha.into : E) ::
        Move.draw (listFn vars (W.tau0.map Ext.into) : Fin vars → E) ::
          Move.draw (listFn batch (W.tau1.map Ext.into) : Fin batch → E) ::
            treeThen (Ext.into (E := E))
              (fun _ _ _ _ => leafPath (Ext.into (E := E)) (rangeLeaf b) leaf W.image)
              (leafRun (rangeLeaf b) leaf W.image)
              (binPath bin (W.binary.into : E) ++
                Move.draw (W.batch.into : E) :: fusedPath E claim W.rounds finish)
              W.stages 1 (fun _ => Ext.one) (W.point low) Ext.zero := by
  unfold foldPath treeThen rangeRun
  cases treeRun (leafRun (rangeLeaf b) leaf W.image) W.stages 1 (fun _ => Ext.one) (W.point low)
    Ext.zero <;> rfl

/-- `foldPathSq` is the three first draws and `treeThen` of the norm stage. -/
theorem foldPathSq_eq (vars batch low leaf limit : ℕ) (bin : Bool) (W : FoldWords)
    (N : NormWords) (claim : Ext) (finish : List Ext → FusedEnd) :
    foldPathSq E vars batch low leaf limit bin W N claim finish =
      Move.draw (W.alpha.into : E) ::
        Move.draw (listFn vars (W.tau0.map Ext.into) : Fin vars → E) ::
          Move.draw (listFn batch (W.tau1.map Ext.into) : Fin batch → E) ::
            treeThen (Ext.into (E := E))
              (fun _ _ _ claim => normPath (Ext.into (E := E)) leaf limit N.energy N.merge
                N.rounds N.virtual W.image claim)
              (normRun leaf limit N.energy N.merge N.rounds N.virtual W.image)
              (Move.draw (N.eta.into : E) :: binPath bin (W.binary.into : E) ++
                Move.draw (W.batch.into : E) :: fusedPath E claim W.rounds finish)
              W.stages 1 (fun _ => Ext.one) (W.point low) Ext.zero := by
  unfold foldPathSq treeThen
  cases treeRun (normRun leaf limit N.energy N.merge N.rounds N.virtual W.image) W.stages 1
    (fun _ => Ext.one) (W.point low) Ext.zero <;> rfl

end Folds

section Full

variable {K : Type} [Field K] (φ : Ext → K)

/-- The path of the product stages when each stage passes: the rounds, the claims of the children
and the batch challenge of each stage, then the path `last` of the last stage from the batched
claim. -/
noncomputable def treeFull (last : Ext → List Move) : List Product → ℕ → Ext → List Move
  | [], _, claim => last claim
  | S :: rest, nodes, _ =>
      eqCodePath φ S.arity S.rounds ++
        Move.recv (listFn (nodes * S.arity) (S.children.map φ)) ::
          Move.draw (φ S.gamma) ::
            treeFull last rest (nodes * S.arity) (batched (nodes * S.arity) S.children S.gamma)

/-- The path of the norm stage when `E_resp` is at most the cap, and then the path `after`. -/
noncomputable def normFull (leaf energy : ℕ) (merge : Ext) (rounds : List (List Ext × Ext))
    (virtual image : Ext) (after : List Move) (claim : Ext) : List Move :=
  Move.recv energy :: Move.draw (φ merge) ::
    (plainPath φ leaf (claim + merge * Ext.ofBase (Fq.ofNat energy)) rounds ++
      Move.recv (φ virtual, φ image) :: after)

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E]

/-- The path of the tensor reduction when its opening check passes. -/
noncomputable def tensorFull (p0 p1 η : Ext) (rounds : List (List Ext × Ext)) (final : Ext) :
    List Move :=
  Move.recv ((p0.into, p1.into) : E × E) :: Move.draw (η.into : E) ::
    (plainPath (Ext.into (E := E)) 1 (tensorInput p0 p1 η) rounds ++ [Move.recv (final.into : E)])

/-- The path after the last stage of the range proof: `ζ_bin` at a level with compression digits,
`γ`, the rounds of the fused sum-check from `claim`, the message `v_W` and the path `after`. -/
noncomputable def closingFull (bin : Bool) (binary batch claim : Ext)
    (rounds : List (List Ext × Ext)) (value : Ext) (after : List Move) : List Move :=
  binPath bin (binary.into : E) ++ Move.draw (batch.into : E) ::
    (plainPath (Ext.into (E := E)) 2 claim rounds ++ Move.recv (value.into : E) :: after)

theorem treeRun_cons {β : Type} (run : ℕ → (ℕ → Ext) → List Ext → Ext → Option β) (S : Product)
    (rest : List Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext) :
    treeRun run (S :: rest) nodes weights point claim =
      match stageOut S nodes weights point claim with
      | none => none
      | some ended => treeRun run rest (nodes * S.arity) (powExt S.gamma) ended.point
          (batched (nodes * S.arity) S.children S.gamma) := by
  show (match eqRun claim Ext.one point S.rounds with
    | none => none
    | some ended =>
        if ended.claim.same (stageEnd nodes S.arity weights S.children) then
          treeRun run rest (nodes * S.arity) (powExt S.gamma) ended.point
            (batched (nodes * S.arity) S.children S.gamma)
        else none) = _
  unfold stageOut
  cases eqRun claim Ext.one point S.rounds with
  | none => rfl
  | some ended =>
      by_cases holds : ended.claim.same (stageEnd nodes S.arity weights S.children) = true
      · simp [holds]
      · simp [holds]

/-- The path of product stages that pass, with the path `after` after it, is the path with no
condition. -/
theorem treeThen_full {β : Type} {last : ℕ → (ℕ → Ext) → List Ext → Ext → List Move}
    {run : ℕ → (ℕ → Ext) → List Ext → Ext → Option β} {middle : List Move}
    {last' : Ext → List Move} (after : List Move)
    (ends : ∀ nodes weights point claim, (run nodes weights point claim).isSome = true →
      last nodes weights point claim ++ middle ++ after = last' claim) :
    ∀ (stages : List Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext),
      (treeRun run stages nodes weights point claim).isSome = true →
      treeThen φ last run middle stages nodes weights point claim ++ after =
        treeFull φ last' stages nodes claim
  | [], nodes, weights, point, claim, passes => by
      have passed : (run nodes weights point claim).isSome = true := passes
      rw [treeThen_nil, ite_eq_left passed]
      exact ends nodes weights point claim passed
  | S :: rest, nodes, weights, point, claim, passes => by
      rw [treeRun_cons] at passes
      rw [treeThen_cons]
      cases out : stageOut S nodes weights point claim with
      | none =>
          rw [out] at passes
          cases passes
      | some ended =>
          rw [out] at passes
          show (eqCodePath φ S.arity S.rounds ++ Move.recv _ :: Move.draw _ ::
            treeThen φ last run middle rest _ _ _ _) ++ after =
              eqCodePath φ S.arity S.rounds ++ Move.recv _ :: Move.draw _ ::
                treeFull φ last' rest _ _
          rw [List.append_assoc, List.cons_append, List.cons_append,
            treeThen_full after ends rest _ _ _ _ passes]

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E]

theorem closingFull_eq (bin : Bool) (binary batch claim : Ext) (rounds : List (List Ext × Ext))
    (finish : List Ext → FusedEnd) (after : List Move) :
    (binPath bin (binary.into : E) ++ Move.draw (batch.into : E) ::
        fusedPath E claim rounds finish) ++ after =
      closingFull E bin binary batch claim rounds (finish (rounds.map (·.2))).value after := by
  unfold closingFull fusedPath
  simp only [List.append_assoc, List.cons_append, List.nil_append]

/-- `foldPath` of a range proof that passes, with the path `after` after it. -/
theorem foldPath_full (vars batch low b : ℕ) (bin : Bool) (W : FoldWords)
    (leaf : List (List Ext × Ext)) (claim : Ext) (finish : List Ext → FusedEnd)
    (after : List Move)
    (passes : (rangeRun b W.stages leaf W.image (W.point low)).isSome = true) :
    foldPath E vars batch low b bin W leaf claim finish ++ after =
      Move.draw (W.alpha.into : E) ::
        Move.draw (listFn vars (W.tau0.map Ext.into) : Fin vars → E) ::
          Move.draw (listFn batch (W.tau1.map Ext.into) : Fin batch → E) ::
            treeFull (Ext.into (E := E))
              (fun _ => leafPath (Ext.into (E := E)) (rangeLeaf b) leaf W.image ++
                closingFull E bin W.binary W.batch claim W.rounds
                  (finish (W.rounds.map (·.2))).value after)
              W.stages 1 Ext.zero := by
  rw [foldPath_eq, List.cons_append, List.cons_append, List.cons_append,
    treeThen_full (Ext.into (E := E)) after (fun _ _ _ _ _ => by
      rw [List.append_assoc, closingFull_eq]) W.stages 1 _ _ _ passes]

/-- `foldPathSq` of product stages and a norm stage that pass, with the path `after` after
it. -/
theorem foldPathSq_full (vars batch low leaf limit : ℕ) (bin : Bool) (W : FoldWords)
    (N : NormWords) (claim : Ext) (finish : List Ext → FusedEnd) (after : List Move)
    (passes : (treeRun (normRun leaf limit N.energy N.merge N.rounds N.virtual W.image) W.stages 1
      (fun _ => Ext.one) (W.point low) Ext.zero).isSome = true) :
    foldPathSq E vars batch low leaf limit bin W N claim finish ++ after =
      Move.draw (W.alpha.into : E) ::
        Move.draw (listFn vars (W.tau0.map Ext.into) : Fin vars → E) ::
          Move.draw (listFn batch (W.tau1.map Ext.into) : Fin batch → E) ::
            treeFull (Ext.into (E := E))
              (normFull (Ext.into (E := E)) leaf N.energy N.merge N.rounds N.virtual W.image
                (Move.draw (N.eta.into : E) ::
                  closingFull E bin W.binary W.batch claim W.rounds
                    (finish (W.rounds.map (·.2))).value after))
              W.stages 1 Ext.zero := by
  have ends : ∀ (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (tree : Ext),
      (normRun leaf limit N.energy N.merge N.rounds N.virtual W.image nodes weights point
        tree).isSome = true →
      normPath (Ext.into (E := E)) leaf limit N.energy N.merge N.rounds N.virtual W.image tree ++
          (Move.draw (N.eta.into : E) :: binPath bin (W.binary.into : E) ++
            Move.draw (W.batch.into : E) :: fusedPath E claim W.rounds finish) ++ after =
        normFull (Ext.into (E := E)) leaf N.energy N.merge N.rounds N.virtual W.image
          (Move.draw (N.eta.into : E) ::
            closingFull E bin W.binary W.batch claim W.rounds
              (finish (W.rounds.map (·.2))).value after) tree := by
    intro nodes weights point tree passed
    have fits : N.energy ≤ limit := by
      by_contra over
      rw [normRun_none _ _ _ _ _ _ _ _ over] at passed
      cases passed
    unfold normPath normFull closingFull fusedPath
    rw [ite_eq_left fits]
    simp only [List.append_assoc, List.cons_append, List.nil_append]
  rw [foldPathSq_eq, List.cons_append, List.cons_append, List.cons_append,
    treeThen_full (Ext.into (E := E)) after ends W.stages 1 _ _ _ passes]

/-- `tensorPath` of a tensor reduction that passes. -/
theorem tensorPath_full (ν : ℕ) (point : List Ext) (v p0 p1 η : Ext)
    (rounds : List (List Ext × Ext)) (final : Ext)
    (passes : (tensorRun ν point v p0 p1 η rounds final).isSome = true) :
    tensorPath E ν point v p0 p1 η rounds final = tensorFull E p0 p1 η rounds final := by
  unfold tensorRun at passes
  unfold tensorPath tensorFull
  cases point with
  | nil => cases passes
  | cons head tail =>
      by_cases long : tail.length = ν
      · simp only [ite_eq_left long] at passes ⊢
        by_cases opens : v.same ((Ext.one - head) * p0 + head * p1) = true
        · simp only [ite_eq_left opens]
        · simp only [ite_eq_right opens] at passes
          cases passes
      · simp only [ite_eq_right long] at passes
        cases passes

end Full

section Needs

open Replayer

/-- One value that the path of a level reads, with the draw `from_` from which the path reads it.
`values`: each value of the key. `value`: the value `index` of the recorded values of the key. -/
inductive Need where
  | values (from_ : ℕ) (key : Key)
  | value (from_ : ℕ) (key : Key) (index : ℕ)

/-- The two states have the same value of the need when its draw is at most `j`. -/
def Need.Holds (j : ℕ) (s t : Replayer) : Need → Prop
  | .values from_ key => from_ ≤ j → recordedIn s key = recordedIn t key ∧
      s.fields.get? key = t.fields.get? key ∧ s.integers.get? key = t.integers.get? key
  | .value from_ key index => from_ ≤ j →
      (recordedIn s key).toList.getD index Ext.zero = (recordedIn t key).toList.getD index Ext.zero

theorem Need.Holds.of_values {j : ℕ} {s t : Replayer} {from_ : ℕ} {key : Key}
    (holds : Need.Holds j s t (.values from_ key)) (le : from_ ≤ j) :
    recordedIn s key = recordedIn t key ∧ s.fields.get? key = t.fields.get? key ∧
      s.integers.get? key = t.integers.get? key :=
  (show from_ ≤ j → _ from holds) le

theorem Need.Holds.of_value {j : ℕ} {s t : Replayer} {from_ : ℕ} {key : Key} {index : ℕ}
    (holds : Need.Holds j s t (.value from_ key index)) (le : from_ ≤ j) :
    (recordedIn s key).toList.getD index Ext.zero =
      (recordedIn t key).toList.getD index Ext.zero :=
  (show from_ ≤ j → _ from holds) le

/-- The needs of the rounds of a sum-check whose first message is after the draw `base`: the
message of the round `i` from the draw `base + i`, and its challenge from the draw
`base + i + 1`. -/
def roundNeeds (base : ℕ) (key : Key) (n : ℕ) : List Need :=
  (List.range n).flatMap fun i =>
    [.values (base + i) ⟨key.level, key.site ++ ".round", some i⟩,
      .value (base + i + 1) ⟨key.level, key.site ++ ".point", none⟩ i]

/-- The needs of the tensor reduction of the claim of the level `level`, whose column partials
are after the draw `base`. -/
def tensorNeeds (level ν base : ℕ) : List Need :=
  .values base ⟨level + 1, "tensor.partials", none⟩ ::
    .values (base + 1) ⟨level + 1, "tensor.eta", none⟩ ::
      (roundNeeds (base + 1) ⟨level + 1, "tensor.sumcheck", none⟩ ν ++
        [.values (base + 1 + ν) ⟨level + 1, "tensor.final_claims", none⟩])

/-- The needs of the fused sum-check, with `base` draws before the draw of `γ`. -/
def fusedNeeds (level n : ℕ) (after : ℕ → List Need) (base : ℕ) : List Need :=
  .values (base + 1) ⟨level, "stage2_batch", none⟩ ::
    ((stage2Keys level).map (Need.values (base + 1)) ++
      (roundNeeds (base + 1) ⟨level, "fused", none⟩ n ++
        (.values (base + 1 + n) ⟨level, "witness_evaluation", none⟩ :: after (base + 1 + n))))

/-- The needs of the path after the last stage of the range proof. -/
def closingNeeds (level : ℕ) (bin : Bool) (n : ℕ) (after : ℕ → List Need) (base : ℕ) :
    List Need :=
  match bin with
  | true => .values (base + 1) ⟨level, "binary_batch", none⟩ :: fusedNeeds level n after (base + 1)
  | false => fusedNeeds level n after base

/-- The needs of the last stage of the range proof and of the path after it. -/
def lastNeeds (level : ℕ) (sq bin : Bool) (n : ℕ) (after : ℕ → List Need) (base : ℕ) :
    List Need :=
  match sq with
  | true =>
      .values base ⟨level, "norm.response_l2_sq", none⟩ ::
        .values (base + 1) ⟨level, "norm.merge", none⟩ ::
          (roundNeeds (base + 1) ⟨level, "norm.sumcheck", none⟩ n ++
            (.values (base + 1 + n) ⟨level, "norm.virtual_evaluations", none⟩ ::
              .values (base + 1 + n) ⟨level, "range.image", none⟩ ::
                .values (base + n + 2) ⟨level, "norm.virtual_batch", none⟩ ::
                  closingNeeds level bin n after (base + n + 2)))
  | false =>
      roundNeeds base ⟨level, "range.leaf", none⟩ n ++
        (.values (base + n) ⟨level, "range.image", none⟩ :: closingNeeds level bin n after (base + n))

/-- The needs of the product stages `stages`, and then the needs `after`. -/
def stageNeeds (level n : ℕ) (after : ℕ → List Need) : List (ℕ × ℕ) → ℕ → List Need
  | [], base => after base
  | x :: more, base =>
      roundNeeds base ⟨level, s!"range.{x.2}", none⟩ n ++
        (.values (base + n) ⟨level, s!"range.{x.2}.children", none⟩ ::
          .values (base + n + 1) ⟨level, s!"range.{x.2}.batch", none⟩ ::
            stageNeeds level n after more (base + n + 1))

/-- The needs of `pathAt`. -/
def pathNeeds (level : ℕ) (sq bin : Bool) (n : ℕ) (stages : List (ℕ × ℕ))
    (after : ℕ → List Need) : List Need :=
  .values 1 ⟨level, "alpha", none⟩ :: .values 2 ⟨level, "tau0", none⟩ :: .values 3 ⟨level, "tau1", none⟩ ::
    stageNeeds level n (lastNeeds level sq bin n after) stages 3

/-- The needs of `afterAt`. -/
def afterNeeds20 (ℓ : Fin 6) : ℕ → List Need :=
  if ℓ.val = 0 then fun _ => [] else tensorNeeds ℓ (tail20 ℓ)

/-- The needs of `pathAt20`. -/
def needs20 (ℓ : Fin 6) : List Need :=
  pathNeeds ℓ (fold20 ℓ).euclidean (fold20 ℓ).compressed (fold20 ℓ).vars
    ((productStages (fold20 ℓ).logBasis).map (·.1)).zipIdx (afterNeeds20 ℓ)

/-- The check of a need on the skeleton of `layout20`, for the level whose draw `0` is the work of
the stage `k₀` and whose last state is before the operation `L`. `values`: no operation from the
work of the stage of the need to `L` writes the key. `value`: the operations before the work of
the stage of the need draw more than `index` challenges into the key. -/
def Need.check (k₀ L : ℕ) : Need → Bool
  | .values from_ key => held20 (workIndices20.getD (k₀ + from_) 0) L [key]
  | .value from_ key index =>
      (drawnCount key (sk20.take (workIndices20.getD (k₀ + from_) 0))).any fun count =>
        decide (index + 1 ≤ count)

theorem needs20_checked₀ : (needs20 0).all (Need.check 2 316) = true := by decide +kernel

theorem needs20_checked₁ : (needs20 1).all (Need.check 52 586) = true := by decide +kernel

theorem needs20_checked₂ : (needs20 2).all (Need.check 121 848) = true := by decide +kernel

theorem needs20_checked₃ : (needs20 3).all (Need.check 204 1153) = true := by decide +kernel

theorem needs20_checked₄ : (needs20 4).all (Need.check 285 1456) = true := by decide +kernel

theorem needs20_checked₅ : (needs20 5).all (Need.check 379 1521) = true := by decide +kernel

end Needs

end NightstreamFPrime.Spec.Compression.Akita.Replay
