import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Closed

/-! The counts of the recording prover of Akita (ePrint 2026/1983, revision of 2026-11-23,
Lemma 10.47 with (212) and (213)) on a closed run `Digest.Run.closed deps run`
(`Replay/Sound20Game.lean`, `Replay/Sound20Closed.lean`), with the count `Q` of `run` and not
the count of the closed run. No Akita type is in this file.

- `Digest.Run.guessed_distinct_le`: the second bad event, with a count of the different strings
  in the digest positions. `Digest.Run.closed_bad_le`: the two bad events of a closed run have
  probability at most `(C(Q_dig, 2) + width·Q·Q_dig)/|digests|`, when the keys of one chain
  `deps point ++ [point]` have at most `width` different strings.
- `Simulate.Coherent`: the addresses of the keys of one chain have one class at most.
  `Simulate.compDeps_closed_classes`: the table queries of the recording prover with no named
  key on a closed run have at most `Q` different classes.
- `Simulate.oracleDeps_closed_bad_le`: `Digest.Run.closed_bad_le` on the table that the
  algorithm sees.
- `Simulate.coherent_none`, `Simulate.compDeps_closed_classes_guard`,
  `Digest.Run.closed_bad_guard`: the hypotheses hold for concrete instances. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Digest.Run

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle

attribute [local instance low] Classical.propDecidable

variable {Fs Challenge Msg Digest Output : Type}

theorem toComp_asking (fs : Fs → Challenge) :
    ∀ (keys : List Fs) (run : Run Fs Challenge Msg Digest Output),
      (asking keys run).toComp fs = run.toComp fs
  | [], _ => rfl
  | _ :: keys, run => toComp_asking fs keys run

/-- The closed algorithm has the digest queries of the algorithm. -/
theorem toComp_closed (deps : Fs → List Fs) (fs : Fs → Challenge)
    (run : Run Fs Challenge Msg Digest Output) :
    (run.closed deps).toComp fs = run.toComp fs := by
  induction run with
  | done output => rfl
  | challenge point next induction =>
      show (asking (deps point) _).toComp fs = _
      rw [toComp_asking]
      exact induction (fs point)
  | digest message next induction =>
      show OracleComp.query message (fun answer => ((next answer).closed deps).toComp fs) =
        OracleComp.query message fun answer => (next answer).toComp fs
      exact congrArg (OracleComp.query message) (funext fun answer => induction answer)

/-- The first bad event of the closed algorithm is the first bad event of the algorithm. -/
theorem collided_closed (deps : Fs → List Fs) (fs : Fs → Challenge) (hash : Msg → Digest)
    (run : Run Fs Challenge Msg Digest Output) :
    Collided fs hash (run.closed deps) ↔ Collided fs hash run := by
  unfold Collided
  simp only [← mem_queries_iff, toComp_closed]

theorem mass_distinct_le [Fintype Digest] [DecidableEq Digest] (targets : List Digest) :
    mass {digest | digest ∈ targets} ≤ (targets.toFinset.card : ℝ) / Fintype.card Digest := by
  have same : {digest | digest ∈ targets} = {digest | digest ∈ targets.dedup} := by
    ext digest
    exact List.mem_dedup.symm
  rw [same, List.card_toFinset]
  exact mass_mem_le _

/-- The second bad event has probability at most `Q_o·V_o/2^ℓ` (Akita Lemma 10.47). `few`: the
queries have at most `V_o` different strings in their digest positions. `bounded`: the algorithm
has at most `Q_o` digest queries. -/
theorem guessed_distinct_le [DecidableEq Msg] [Fintype Msg] [Fintype Digest] [DecidableEq Digest]
    [Nonempty Digest] (fs : Fs → Challenge) (positions : Fs ⊕ Msg → List Digest)
    {run : Run Fs Challenge Msg Digest Output} {digests positionCount : ℕ}
    (bounded : (run.toComp fs).QueryBound digests)
    (few : ∀ hash, ((run.trace fs hash).flatMap positions).toFinset.card ≤ positionCount) :
    𝔼 hash : Msg → Digest, (if Guessed fs positions hash run then (1 : ℝ) else 0) ≤
      digests * ((positionCount : ℝ) / Fintype.card Digest) := by
  have escape := escape_le
    (fun message hash => {digest | digest ∈ (run.upTo fs hash message).flatMap positions})
    (fun message hash answer => by simp only [upTo_update])
    ((positionCount : ℝ) / Fintype.card Digest)
    (fun message hash => (mass_distinct_le _).trans (div_le_div_of_nonneg_right (by
      obtain ⟨later, split⟩ := upTo_prefix fs hash message run
      have total := few hash
      rw [← split, List.flatMap_append, List.toFinset_append] at total
      exact_mod_cast (Finset.card_le_card Finset.subset_union_left).trans total)
      (Nat.cast_nonneg _)))
    (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) bounded
  refine (Finset.expect_le_expect fun hash _ => ?_).trans escape
  by_cases guessed : Guessed fs positions hash run
  · obtain ⟨message, queried, hit⟩ := guessed
    have escaped : ∃ point ∈ (run.toComp fs).queries hash,
        hash point ∈ {digest | digest ∈ (run.upTo fs hash point).flatMap positions} :=
      ⟨message, (mem_queries_iff fs hash run message).mpr queried, hit⟩
    rw [ite_eq_left ⟨message, queried, hit⟩, ite_eq_left escaped]
  · rw [ite_eq_right guessed]
    split <;> norm_num

/-- The queries of the closed algorithm have at most `width·Q` different strings in their digest
positions, when the keys of one chain have at most `width`. -/
theorem distinct_closed [DecidableEq Digest] (deps : Fs → List Fs) (fs : Fs → Challenge)
    (positions : Fs ⊕ Msg → List Digest) {width : ℕ}
    (chain : ∀ point,
      ((deps point ++ [point]).flatMap fun key => positions (.inl key)).toFinset.card ≤ width)
    (digest : ∀ message, positions (.inr message) = [])
    {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) (hash : Msg → Digest) :
    (((run.closed deps).trace fs hash).flatMap positions).toFinset.card ≤ width * queries := by
  induction bounded with
  | done output queries digests =>
      show (([] : List (Fs ⊕ Msg)).flatMap positions).toFinset.card ≤ _
      simp
  | challenge point next queries digests each induction =>
      have parts : ((deps point).map Sum.inl ++
          Sum.inl point :: ((next (fs point)).closed deps).trace fs hash).flatMap positions =
          ((deps point ++ [point]).flatMap fun key => positions (.inl key)) ++
            (((next (fs point)).closed deps).trace fs hash).flatMap positions := by
        simp [List.flatMap_map]
      rw [trace_closed_challenge, parts, List.toFinset_append, Nat.mul_succ]
      exact (Finset.card_union_le _ _).trans
        ((Nat.add_comm _ _).trans_le (Nat.add_le_add (induction (fs point)) (chain point)))
  | digest message next queries digests each induction =>
      show ((Sum.inr message :: ((next (hash message)).closed deps).trace fs hash).flatMap
        positions).toFinset.card ≤ _
      rw [List.flatMap_cons, digest, List.nil_append]
      exact induction _

private theorem or_le (p q : Prop) :
    (if p ∨ q then (1 : ℝ) else 0) ≤ (if p then (1 : ℝ) else 0) + if q then (1 : ℝ) else 0 := by
  by_cases first : p <;> by_cases second : q <;> norm_num [first, second]

/-- The bound `ε_dig` of Akita (213) for a closed algorithm, with the count `Q` of the algorithm
before `closed`: when the keys of one chain `deps point ++ [point]` have at most `width`
different strings in their digest positions and a digest input has none, the probability of a
bad event is at most `(C(Q_dig, 2) + width·Q·Q_dig)/2^ℓ`. -/
theorem closed_bad_le [DecidableEq Msg] [Fintype Msg] [Fintype Digest] [DecidableEq Digest]
    [Nonempty Digest] (deps : Fs → List Fs) (fs : Fs → Challenge)
    (positions : Fs ⊕ Msg → List Digest) {width : ℕ}
    (chain : ∀ point,
      ((deps point ++ [point]).flatMap fun key => positions (.inl key)).toFinset.card ≤ width)
    (digest : ∀ message, positions (.inr message) = [])
    {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) :
    𝔼 hash : Msg → Digest,
        (if Collided fs hash (run.closed deps) ∨ Guessed fs positions hash (run.closed deps)
          then (1 : ℝ) else 0) ≤
      ((digests.choose 2 + width * queries * digests : ℕ) : ℝ) / Fintype.card Digest := by
  have comp : ((run.closed deps).toComp fs).QueryBound digests := by
    rw [toComp_closed]
    exact bounded.toComp fs
  have first : 𝔼 hash : Msg → Digest,
      (if Collided fs hash (run.closed deps) then (1 : ℝ) else 0) ≤
        (digests.choose 2 : ℝ) / Fintype.card Digest := by
    refine (Finset.expect_le_expect fun hash _ => ?_).trans (collided_le fs bounded)
    by_cases collided : Collided fs hash run
    · rw [ite_eq_left collided, ite_eq_left ((collided_closed deps fs hash run).mpr collided)]
    · rw [ite_eq_right collided, ite_eq_right (mt (collided_closed deps fs hash run).mp collided)]
  have second := guessed_distinct_le fs positions comp
    (distinct_closed deps fs positions chain digest bounded)
  refine (Finset.expect_le_expect fun hash _ => or_le _ _).trans ?_
  rw [Finset.expect_add_distrib]
  refine (add_le_add first second).trans (le_of_eq ?_)
  push_cast
  ring

/-- The hypotheses of `closed_bad_le` hold for no named key and `width = 2`, when each query to
`H_FS` has at most two digest positions: the guard against a vacuous statement. -/
theorem closed_bad_guard [DecidableEq Msg] [Fintype Msg] [Fintype Digest] [DecidableEq Digest]
    [Nonempty Digest] (fs : Fs → Challenge) (positions : Fs ⊕ Msg → List Digest)
    (challenge : ∀ point, (positions (.inl point)).length ≤ 2)
    (digest : ∀ message, positions (.inr message) = [])
    {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) :
    𝔼 hash : Msg → Digest,
        (if Collided fs hash (run.closed fun _ => []) ∨
          Guessed fs positions hash (run.closed fun _ => []) then (1 : ℝ) else 0) ≤
      ((digests.choose 2 + 2 * queries * digests : ℕ) : ℝ) / Fintype.card Digest :=
  closed_bad_le (fun _ => []) fs positions
    (fun point => (List.toFinset_card_le _).trans (by simpa using challenge point)) digest bounded

end NightstreamFPrime.Spec.Compression.Akita.Digest.Run

namespace NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.RandomOracle

attribute [local instance low] Classical.propDecidable

section Classes

variable {Fs Y Msg Dig Addr Reply Output K : Type}

variable (resolve : (Fs → Y) → List (Msg × Dig) → Fs → Option Addr) (deps : Fs → List Fs)
  (φ : Fs → Reply × Y → Y)

/-- For one table of `H_FS` and one record, the addresses of the keys of one chain
`deps point ++ [point]` have one class at most. -/
def Coherent (class_ : Addr → Option K) : Prop :=
  ∀ (fs : Fs → Y) (record : List (Msg × Dig)) (point first second : Fs) (x y : Addr) (u v : K),
    first ∈ deps point ++ [point] → second ∈ deps point ++ [point] →
    resolve fs record first = some x → resolve fs record second = some y →
    class_ x = some u → class_ y = some v → u = v

/-- `Coherent` holds for the class function with no class. -/
theorem coherent_none : Coherent resolve deps (fun _ : Addr => (none : Option K)) :=
  fun _ _ _ _ _ _ _ _ _ _ _ _ _ classed _ => by cases classed

section Settle

variable [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)

/-- A key that `settleAll` adds to the cache is a key of `pending`, and its entry is `resolve` of
the table that the grown cache gives and of the record. -/
theorem settleAll_fresh (record : List (Msg × Dig)) (local_ : DepsLocal resolve deps)
    (pending : List Fs) :
    ∀ cache : Cache Fs Addr, Ready deps cache pending → ∀ (key : Fs) (slot : Option Addr),
      cache key = none → settleAll resolve φ table aux record cache pending key = some slot →
      key ∈ pending ∧
        slot = resolve (reads φ (settleAll resolve φ table aux record cache pending) table aux)
          record key := by
  induction pending with
  | nil =>
      intro cache _ key slot fresh found
      have old : cache key = some slot := found
      exact absurd (fresh.symm.trans old) (by simp)
  | cons point pending induction =>
      intro cache ready key slot fresh found
      rw [settleAll_cons] at found ⊢
      cases grown : settle resolve φ table aux record cache point key with
      | none =>
          obtain ⟨member, value⟩ := induction _
            (ready_tail deps ready
              (fun other known => settle_isSome resolve φ table aux record point known)
              (settle_self resolve φ table aux record cache point))
            key slot grown found
          exact ⟨List.mem_cons_of_mem point member, value⟩
      | some old =>
          have same : key = point := by
            by_contra apart
            rw [settle_of_ne resolve φ table aux record cache apart, fresh] at grown
            cases grown
          rw [same] at fresh found ⊢
          have atKey : settle resolve φ table aux record cache point point =
              some (resolve (reads φ cache table aux) record point) := by
            rw [settle_fresh resolve φ table aux record fresh, Function.update_self]
          have value : slot = resolve (reads φ cache table aux) record point :=
            Option.some.inj (found.symm.trans
              (settleAll_keeps resolve φ table aux record pending _ point _ atKey))
          refine ⟨List.mem_cons_self, value.trans (local_ _ _ record point fun dep member => ?_)⟩
          obtain ⟨prior, cached⟩ := Option.isSome_iff_exists.mp (ready_head deps ready dep member)
          exact reads_of_eq φ (cached.trans (settleAll_keeps resolve φ table aux record pending _
            dep prior (settle_keeps resolve φ table aux record point cached)).symm) table aux

/-- The keys that one chain adds to the cache have one class at most. -/
theorem settleAll_class [DecidableEq K] (record : List (Msg × Dig))
    (local_ : DepsLocal resolve deps) (class_ : Addr → Option K)
    (coherent : Coherent resolve deps class_) (cache : Cache Fs Addr) (point : Fs)
    (ready : Ready deps cache (deps point ++ [point])) :
    ∃ extra : Finset K, extra.card ≤ 1 ∧ ∀ (key : Fs) (address : Addr) (u : K),
      cache key = none →
      settleAll resolve φ table aux record cache (deps point ++ [point]) key =
        some (some address) →
      class_ address = some u → u ∈ extra := by
  by_cases seen : ∃ (key : Fs) (address : Addr) (u : K), cache key = none ∧
      settleAll resolve φ table aux record cache (deps point ++ [point]) key =
        some (some address) ∧ class_ address = some u
  · obtain ⟨first, x, u, fresh, found, classed⟩ := seen
    obtain ⟨member, value⟩ := settleAll_fresh resolve deps φ table aux record local_ _ cache
      ready first (some x) fresh found
    refine ⟨{u}, (Finset.card_singleton u).le, fun key address v freshKey foundKey classedKey =>
      Finset.mem_singleton.mpr ?_⟩
    obtain ⟨memberKey, valueKey⟩ := settleAll_fresh resolve deps φ table aux record local_ _
      cache ready key (some address) freshKey foundKey
    exact coherent _ record point key first address x v u memberKey member valueKey.symm
      value.symm classedKey classed
  · exact ⟨∅, by simp, fun key address u fresh found classed =>
      absurd ⟨key, address, u, fresh, found, classed⟩ seen⟩

end Settle

section Run

variable [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)

/-- The addresses that the simulation of a closed run adds to the cache have at most `Q`
different classes, `Q` the count of the run before `closed`. -/
theorem simulateDeps_closed_classes [DecidableEq K] (local_ : DepsLocal resolve deps)
    (closed : DepsClosed deps) (class_ : Addr → Option K)
    (coherent : Coherent resolve deps class_) {run : Digest.Run Fs Y Msg Dig Output}
    {queries digests : ℕ} (bounded : run.Bound queries digests) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)), ∃ classes : Finset K,
      classes.card ≤ queries ∧ ∀ (key : Fs) (address : Addr) (u : K), cache key = none →
        (simulateDeps resolve deps φ table aux hash (run.closed deps) cache record).2.1 key =
          some (some address) →
        class_ address = some u → u ∈ classes := by
  induction bounded with
  | done output queries digests =>
      refine fun cache record => ⟨∅, by simp, fun key address u fresh found _ => ?_⟩
      have old : cache key = some (some address) := found
      exact absurd (fresh.symm.trans old) (by simp)
  | digest message next queries digests each induction =>
      intro cache record
      have shape : (Digest.Run.digest message next).closed deps =
          .digest message fun answer => (next answer).closed deps := rfl
      rw [shape, simulateDeps_digest]
      exact induction (hash message) cache _
  | challenge point next queries digests each induction =>
      intro cache record
      have shape : (Digest.Run.challenge point next).closed deps =
          Digest.Run.asking (deps point)
            (.challenge point fun answer => (next answer).closed deps) := rfl
      have ready : Ready deps cache (deps point) := fun before key after split dep member =>
        Or.inl (closed point before key after split dep member)
      obtain ⟨grown, grownDef⟩ : ∃ grown,
          grown = settleAll resolve φ table aux record cache (deps point ++ [point]) := ⟨_, rfl⟩
      have parts : settleAll resolve φ table aux record cache (deps point ++ [point]) =
          settleAll resolve φ table aux record
            (settleAll resolve φ table aux record cache (deps point)) [point] :=
        List.foldl_append
      have chained : settleAll resolve φ table aux record
          (settleAll resolve φ table aux record cache (deps point)) (deps point ++ [point]) =
            grown := by
        rw [settleAll_last resolve φ table aux record
          (cache := settleAll resolve φ table aux record cache (deps point))
          (pending := deps point) point
          fun key member => settleAll_mem resolve φ table aux record _ cache key member,
          grownDef]
        exact parts.symm
      rw [shape,
        simulateDeps_asking resolve deps φ table aux hash _ record (deps point) cache ready,
        simulateDeps_challenge, chained]
      obtain ⟨rest, restCard, restHas⟩ := induction (reads φ grown table aux point) grown record
      obtain ⟨extra, extraCard, extraHas⟩ := settleAll_class resolve deps φ table aux record
        local_ class_ coherent cache point (ready_chain deps closed cache point)
      rw [← grownDef] at extraHas
      refine ⟨rest ∪ extra,
        (Finset.card_union_le rest extra).trans (Nat.add_le_add restCard extraCard),
        fun key address u fresh found classed => Finset.mem_union.mpr ?_⟩
      cases cached : grown key with
      | none => exact Or.inl (restHas key address u cached found classed)
      | some slot =>
          rw [simulateDeps_cache resolve deps φ table aux hash _ grown record key slot cached]
            at found
          exact Or.inr (extraHas key address u fresh (cached.trans found) classed)

/-- Each table query of `compDeps` is the address of a key that the simulation adds to the
cache. -/
theorem compDeps_queries_fixed (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)) (address : Addr),
      address ∈ (compDeps resolve deps φ aux hash run cache record
        (reads φ cache table aux)).queries table →
      ∃ key, cache key = none ∧
        (simulateDeps resolve deps φ table aux hash run cache record).2.1 key =
          some (some address) := by
  induction run with
  | done output => exact fun cache record address member => absurd member List.not_mem_nil
  | digest message next induction =>
      exact fun cache record address member => induction (hash message) cache _ address member
  | challenge point next induction =>
      intro cache record address member
      obtain ⟨grown, grownDef⟩ : ∃ grown,
          grown = settleAll resolve φ table aux record cache (deps point ++ [point]) := ⟨_, rfl⟩
      obtain ⟨first, -, -, fresh, listed⟩ := settleAllComp_fixed resolve φ table aux record
        (fun later answers =>
          compDeps resolve deps φ aux hash (next (answers point)) later record answers)
        (deps point ++ [point]) cache
      rw [compDeps_challenge, listed, ← grownDef] at member
      rw [simulateDeps_challenge, ← grownDef]
      rcases List.mem_append.mp member with early | late
      · obtain ⟨key, inside, joined⟩ := List.mem_filterMap.mp early
        have found : grown key = some (some address) := by
          cases cached : grown key with
          | none =>
              rw [cached] at joined
              cases joined
          | some slot =>
              rw [cached] at joined
              exact congrArg some joined
        exact ⟨key, fresh key inside,
          simulateDeps_cache resolve deps φ table aux hash _ grown record key _ found⟩
      · obtain ⟨key, absent, found⟩ := induction _ grown record address late
        refine ⟨key, ?_, found⟩
        cases cached : cache key with
        | none => rfl
        | some slot =>
            have kept := settleAll_keeps resolve φ table aux record (deps point ++ [point])
              cache key slot cached
            rw [← grownDef, absent] at kept
            cases kept

/-- The table queries of the recording prover with no named key on a closed run, with its
answered queries, have at most `Q` different classes, `Q` the count of the run before `closed`:
each query to `H_FS` of the run adds the keys of one chain, and the new addresses of one chain
have one class at most (`coherent`). -/
theorem compDeps_closed_classes [DecidableEq K] (local_ : DepsLocal resolve deps)
    (closed : DepsClosed deps) (class_ : Addr → Option K)
    (coherent : Coherent resolve deps class_) {run : Digest.Run Fs Y Msg Dig Output}
    {queries digests : ℕ} (bounded : run.Bound queries digests) :
    (((compDeps resolve (fun _ => []) φ aux hash (run.closed deps).logged (fun _ => none) []
      aux).queries table).filterMap class_).toFinset.card ≤ queries := by
  obtain ⟨classes, card, has⟩ := simulateDeps_closed_classes resolve deps φ table aux hash local_
    closed class_ coherent bounded (fun _ => none) []
  refine (Finset.card_le_card fun u member => ?_).trans card
  obtain ⟨address, queried, classed⟩ := List.mem_filterMap.mp (List.mem_toFinset.mp member)
  obtain ⟨key, -, found⟩ := compDeps_queries_fixed resolve (fun _ => []) φ table aux hash
    (run.closed deps).logged (fun _ => none) [] address queried
  rw [(simulateDeps_logged resolve (fun _ => []) φ table aux hash (run.closed deps) _ []).2,
    ← simulateDeps_closed resolve deps φ table aux hash closed run _ [] (named_empty deps)]
    at found
  exact has key address u rfl found classed

end Run

section Bad

private theorem expect_swap_law {T A H F : Type} [Fintype T] [Fintype A] [Fintype H] [Fintype F]
    (X : T → A → H → ℝ) (bad : F → H → ℝ)
    (law : ∀ hash, 𝔼 t, 𝔼 a, X t a hash = 𝔼 fs, bad fs hash) :
    𝔼 t, 𝔼 a, 𝔼 hash, X t a hash = 𝔼 fs, 𝔼 hash, bad fs hash :=
  calc 𝔼 t, 𝔼 a, 𝔼 hash, X t a hash = 𝔼 t, 𝔼 hash, 𝔼 a, X t a hash :=
        Finset.expect_congr rfl fun _ _ => Finset.expect_comm _ _ _
    _ = 𝔼 hash, 𝔼 t, 𝔼 a, X t a hash := Finset.expect_comm _ _ _
    _ = 𝔼 hash, 𝔼 fs, bad fs hash := Finset.expect_congr rfl fun hash _ => law hash
    _ = 𝔼 fs, 𝔼 hash, bad fs hash := Finset.expect_comm _ _ _

variable [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y] [Fintype Addr] [DecidableEq Addr]
  [Fintype Reply] [Nonempty Reply] [Fintype Msg] [DecidableEq Msg] [Fintype Dig]

/-- The two bad events of Akita Lemma 10.47 on the table that the algorithm sees have, for a
uniform `table`, a uniform `aux` and a uniform `hash`, each bound `limit` that they have for
each table of `H_FS` and a uniform `hash`. The other hypotheses are those of
`simulateDeps_law`. -/
theorem oracleDeps_bad_le_of {back : (Fs → Y) → Addr → Fs} (local_ : DepsLocal resolve deps)
    (closed : DepsClosed deps)
    (resolves : ∀ fs record point address, resolve fs record point = some address →
      back fs address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) (positions : (Fs → Y) → Fs ⊕ Msg → List Dig)
    (alike : ∀ fs other hash, run.answered fs hash = run.answered other hash →
      ∀ query ∈ run.trace fs hash, positions fs query = positions other query) {limit : ℝ}
    (each : ∀ fs : Fs → Y, 𝔼 hash : Msg → Dig,
      (if Digest.Run.Collided fs hash run ∨ Digest.Run.Guessed fs (positions fs) hash run
        then (1 : ℝ) else 0) ≤ limit) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, 𝔼 hash : Msg → Dig,
        (if Digest.Run.Collided (oracleDeps resolve deps φ table aux hash run []) hash run ∨
          Digest.Run.Guessed (oracleDeps resolve deps φ table aux hash run [])
            (positions (oracleDeps resolve deps φ table aux hash run [])) hash run
          then (1 : ℝ) else 0) ≤ limit := by
  have law : ∀ hash : Msg → Dig, 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
      (if Digest.Run.Collided (oracleDeps resolve deps φ table aux hash run []) hash run ∨
        Digest.Run.Guessed (oracleDeps resolve deps φ table aux hash run [])
          (positions (oracleDeps resolve deps φ table aux hash run [])) hash run
        then (1 : ℝ) else 0) =
      𝔼 fs : Fs → Y,
        (if Digest.Run.Collided fs hash run ∨ Digest.Run.Guessed fs (positions fs) hash run
          then (1 : ℝ) else 0) :=
    fun hash => oracleDeps_law resolve deps φ (back := back) hash local_ closed
      (fun fs record _ point address resolved => resolves fs record point address resolved)
      fibres run [] (fun _ member => absurd member List.not_mem_nil)
      (fun fs => if Digest.Run.Collided fs hash run ∨
        Digest.Run.Guessed fs (positions fs) hash run then (1 : ℝ) else 0)
      (fun fs other same => by
        have collided := Digest.Run.collided_congr same
        have guessed := Digest.Run.guessed_congr_family positions same
          (alike fs other hash same)
        simp only [collided, guessed])
  refine (le_of_eq (expect_swap_law
    (fun (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig) =>
      if Digest.Run.Collided (oracleDeps resolve deps φ table aux hash run []) hash run ∨
        Digest.Run.Guessed (oracleDeps resolve deps φ table aux hash run [])
          (positions (oracleDeps resolve deps φ table aux hash run [])) hash run
        then (1 : ℝ) else 0)
    (fun (fs : Fs → Y) (hash : Msg → Dig) =>
      if Digest.Run.Collided fs hash run ∨ Digest.Run.Guessed fs (positions fs) hash run
        then (1 : ℝ) else 0) law)).trans ?_
  exact expect_le_const _ _ each

/-- The bound `ε_dig` of Akita (213) in the recording prover on a closed run, with the count `Q`
of the run before `closed`: for a uniform `table`, a uniform `aux` and a uniform `hash`, the two
bad events of Lemma 10.47 on the table that the algorithm sees have probability at most
`(C(Q_dig, 2) + width·Q·Q_dig)/|digests|`. `chain`: for each table, the keys of one chain
`deps point ++ [point]` have at most `width` different strings in their digest positions. The
other hypotheses are those of `oracleDeps_bad_le_of` for the closed run. -/
theorem oracleDeps_closed_bad_le [DecidableEq Dig] [Nonempty Dig]
    {back : (Fs → Y) → Addr → Fs}
    (local_ : DepsLocal resolve deps) (closed : DepsClosed deps)
    (resolves : ∀ fs record point address, resolve fs record point = some address →
      back fs address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) {queries digests : ℕ}
    (bounded : run.Bound queries digests) (positions : (Fs → Y) → Fs ⊕ Msg → List Dig)
    {width : ℕ}
    (chain : ∀ fs point,
      ((deps point ++ [point]).flatMap fun key => positions fs (.inl key)).toFinset.card ≤ width)
    (digest : ∀ fs message, positions fs (.inr message) = [])
    (alike : ∀ fs other hash,
      (run.closed deps).answered fs hash = (run.closed deps).answered other hash →
      ∀ query ∈ (run.closed deps).trace fs hash, positions fs query = positions other query) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, 𝔼 hash : Msg → Dig,
        (if Digest.Run.Collided (oracleDeps resolve deps φ table aux hash (run.closed deps) [])
            hash (run.closed deps) ∨
          Digest.Run.Guessed (oracleDeps resolve deps φ table aux hash (run.closed deps) [])
            (positions (oracleDeps resolve deps φ table aux hash (run.closed deps) [])) hash
            (run.closed deps) then (1 : ℝ) else 0) ≤
      ((digests.choose 2 + width * queries * digests : ℕ) : ℝ) / Fintype.card Dig :=
  oracleDeps_bad_le_of resolve deps φ (back := back) local_ closed resolves fibres
    (run.closed deps) positions alike fun fs =>
      Digest.Run.closed_bad_le deps fs (positions fs) (chain fs) (digest fs) bounded

end Bad

end Classes

section Guard

variable {Y Msg Dig Reply Output : Type}

/-- `compDeps_closed_classes` for two keys where the address of the second key has the answer of
`H_FS` at the first key, and the class of an address is its first entry: the hypotheses
`DepsLocal`, `DepsClosed` and `Coherent` hold for a class function that is not constant. This is
the guard against a vacuous statement. -/
theorem compDeps_closed_classes_guard [DecidableEq Y] (φ : Bool → Reply × Y → Y)
    (table : List Y → Reply) (aux : Bool → Y) (hash : Msg → Dig)
    {run : Digest.Run Bool Y Msg Dig Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) :
    (((compDeps
      (fun (fs : Bool → Y) (_ : List (Msg × Dig)) (point : Bool) =>
        some (if point then [fs false] else []))
      (fun _ => []) φ aux hash (run.closed fun point => if point then [false] else []).logged
      (fun _ => none) [] aux).queries table).filterMap List.head?).toFinset.card ≤ queries :=
  compDeps_closed_classes
    (fun (fs : Bool → Y) (_ : List (Msg × Dig)) (point : Bool) =>
      some (if point then [fs false] else []))
    (fun point => if point then [false] else []) φ table aux hash
    (fun fs other _ point same => by
      cases point with
      | false => rfl
      | true => exact congrArg (fun value => some [value]) (same false (by simp)))
    (fun point before key after split named member => by
      cases point with
      | false => simp at split
      | true =>
          cases before with
          | nil =>
              simp at split
              obtain ⟨first, -⟩ := split
              rw [first] at member
              simp at member
          | cons first before => simp at split)
    (K := Y) List.head?
    (fun fs _ _ first second x y u v _ _ firstAt secondAt firstClass secondClass => by
      cases first with
      | false =>
          have empty : x = [] := (Option.some.inj firstAt).symm
          rw [empty] at firstClass
          cases firstClass
      | true =>
          cases second with
          | false =>
              have empty : y = [] := (Option.some.inj secondAt).symm
              rw [empty] at secondClass
              cases secondClass
          | true =>
              have one : x = [fs false] := (Option.some.inj firstAt).symm
              have other : y = [fs false] := (Option.some.inj secondAt).symm
              rw [one] at firstClass
              rw [other] at secondClass
              exact (Option.some.inj firstClass).symm.trans (Option.some.inj secondClass))
    bounded

end Guard

end NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate
