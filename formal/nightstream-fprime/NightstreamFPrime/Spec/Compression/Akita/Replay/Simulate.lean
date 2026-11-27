import NightstreamFPrime.Spec.Compression.Akita.Digest
import NightstreamFPrime.Spec.Compression.Akita.Replay.Coupling

/-! The recording prover of Akita (ePrint 2026/1983, revision of 2026-11-23, Lemma 10.47,
the paragraphs "The prover P" and "Simulation"), for an algorithm `Digest.Run` with the two
oracles `H_FS` and `H_o`. No Akita type is in this file.

- `Digest.Run.eval` is the run with two tables. `Digest.Run.answered` has the queries with their
  answers, and `Digest.Run.logged` gives them with the output.
- `simulate` answers a digest query from `hash` and records it. The first `H_FS` query of a key
  fixes `resolve record key` in a cache. A key with an address gets `φ key (table address, aux
  key)`, and each other key gets `aux key`. `comp` is `simulate` as an algorithm with the oracle
  `table`, with one table query for each `H_FS` query at most.
- `simulate_law`: for a uniform `table` and a uniform `aux`, the output has the law of the run
  with a uniform `H_FS`, with no condition on the bad events of the lemma. `oracle_law` is this
  law for each value that reads `H_FS` through the answered queries only, for example the events
  `Digest.Run.Collided` and `Digest.Run.Guessed` (`Digest.Run.collided_congr`,
  `Digest.Run.guessed_congr`).
- `simulate_resolved` and `simulate_record_oracle` give the final cache and the final record:
  each entry of the cache is `resolve` of the record at an earlier time, and the record has the
  digest queries of the run. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Digest.Run

variable {Fs Challenge Msg Digest Output : Type}

/-- The output of the algorithm for the tables `fs` of `H_FS` and `hash` of `H_o`. -/
def eval (fs : Fs → Challenge) (hash : Msg → Digest) :
    Run Fs Challenge Msg Digest Output → Output
  | done output => output
  | challenge point next => (next (fs point)).eval fs hash
  | digest message next => (next (hash message)).eval fs hash

theorem eval_challenge (fs : Fs → Challenge) (hash : Msg → Digest) (point : Fs)
    (next : Challenge → Run Fs Challenge Msg Digest Output) :
    (challenge point next).eval fs hash = (next (fs point)).eval fs hash :=
  rfl

theorem eval_eq_toComp (fs : Fs → Challenge) (hash : Msg → Digest)
    (run : Run Fs Challenge Msg Digest Output) :
    run.eval fs hash = (run.toComp fs).run hash := by
  induction run with
  | done output => rfl
  | challenge point next induction => exact induction (fs point)
  | digest message next induction => exact induction (hash message)

/-- The algorithm with `f` on its output. -/
def map {Other : Type} (f : Output → Other) :
    Run Fs Challenge Msg Digest Output → Run Fs Challenge Msg Digest Other
  | done output => done (f output)
  | challenge point next => challenge point fun answer => (next answer).map f
  | digest message next => digest message fun answer => (next answer).map f

theorem eval_map {Other : Type} (f : Output → Other) (fs : Fs → Challenge) (hash : Msg → Digest)
    (run : Run Fs Challenge Msg Digest Output) :
    (run.map f).eval fs hash = f (run.eval fs hash) := by
  induction run with
  | done output => rfl
  | challenge point next induction => exact induction (fs point)
  | digest message next induction => exact induction (hash message)

theorem Bound.map {Other : Type} (f : Output → Other)
    {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) : (run.map f).Bound queries digests := by
  induction bounded with
  | done output queries digests => exact .done _ _ _
  | challenge point next queries digests each induction =>
      exact .challenge point _ queries digests induction
  | digest message next queries digests each induction =>
      exact .digest message _ queries digests induction

/-- The queries of the algorithm to the two oracles with their answers, in their order. -/
def answered (fs : Fs → Challenge) (hash : Msg → Digest) :
    Run Fs Challenge Msg Digest Output → List ((Fs × Challenge) ⊕ (Msg × Digest))
  | done _ => []
  | challenge point next => .inl (point, fs point) :: (next (fs point)).answered fs hash
  | digest message next => .inr (message, hash message) :: (next (hash message)).answered fs hash

theorem map_answered (fs : Fs → Challenge) (hash : Msg → Digest)
    (run : Run Fs Challenge Msg Digest Output) :
    (run.answered fs hash).map (Sum.map Prod.fst Prod.fst) = run.trace fs hash := by
  induction run with
  | done output => rfl
  | challenge point next induction => simp [answered, trace, induction]
  | digest message next induction => simp [answered, trace, induction]

/-- The algorithm that gives its answered queries with its output. -/
def logged : Run Fs Challenge Msg Digest Output →
    Run Fs Challenge Msg Digest (Output × List ((Fs × Challenge) ⊕ (Msg × Digest)))
  | done output => done (output, [])
  | challenge point next => challenge point fun answer =>
      (next answer).logged.map fun result => (result.1, .inl (point, answer) :: result.2)
  | digest message next => digest message fun answer =>
      (next answer).logged.map fun result => (result.1, .inr (message, answer) :: result.2)

theorem logged_challenge (point : Fs) (next : Challenge → Run Fs Challenge Msg Digest Output) :
    (challenge point next).logged = challenge point fun answer =>
      (next answer).logged.map fun result => (result.1, .inl (point, answer) :: result.2) :=
  rfl

theorem logged_digest (message : Msg) (next : Digest → Run Fs Challenge Msg Digest Output) :
    (digest message next).logged = digest message fun answer =>
      (next answer).logged.map fun result => (result.1, .inr (message, answer) :: result.2) :=
  rfl

theorem eval_logged (fs : Fs → Challenge) (hash : Msg → Digest)
    (run : Run Fs Challenge Msg Digest Output) :
    run.logged.eval fs hash = (run.eval fs hash, run.answered fs hash) := by
  induction run with
  | done output => rfl
  | challenge point next induction =>
      rw [logged_challenge, eval_challenge, eval_map, induction]
      rfl
  | digest message next induction =>
      rw [logged_digest]
      exact (eval_map _ fs hash _).trans (by rw [induction]; rfl)

theorem Bound.logged {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) : run.logged.Bound queries digests := by
  induction bounded with
  | done output queries digests => exact .done _ _ _
  | challenge point next queries digests each induction =>
      exact .challenge point _ queries digests fun answer => (induction answer).map _
  | digest message next queries digests each induction =>
      exact .digest message _ queries digests fun answer => (induction answer).map _

/-- The output is a function of the answered queries. -/
theorem eval_congr {fs other : Fs → Challenge} {hash : Msg → Digest}
    {run : Run Fs Challenge Msg Digest Output}
    (same : run.answered fs hash = run.answered other hash) :
    run.eval fs hash = run.eval other hash := by
  induction run with
  | done output => rfl
  | challenge point next induction =>
      have listed : (Sum.inl (point, fs point) :: (next (fs point)).answered fs hash :
          List ((Fs × Challenge) ⊕ (Msg × Digest))) =
          Sum.inl (point, other point) :: (next (other point)).answered other hash := same
      have value : fs point = other point :=
        (Prod.mk.inj (Sum.inl.inj (List.head_eq_of_cons_eq listed))).2
      have rest := List.tail_eq_of_cons_eq listed
      rw [← value] at rest
      rw [eval_challenge, eval_challenge, ← value]
      exact induction (fs point) rest
  | digest message next induction =>
      have listed : (Sum.inr (message, hash message) :: (next (hash message)).answered fs hash :
          List ((Fs × Challenge) ⊕ (Msg × Digest))) =
          Sum.inr (message, hash message) :: (next (hash message)).answered other hash := same
      exact induction (hash message) (List.tail_eq_of_cons_eq listed)

/-- The queries are a function of the answered queries. -/
theorem trace_congr {fs other : Fs → Challenge} {hash : Msg → Digest}
    {run : Run Fs Challenge Msg Digest Output}
    (same : run.answered fs hash = run.answered other hash) :
    run.trace fs hash = run.trace other hash := by
  rw [← map_answered, ← map_answered, same]

theorem upTo_congr [DecidableEq Msg] {fs other : Fs → Challenge} {hash : Msg → Digest}
    {run : Run Fs Challenge Msg Digest Output}
    (same : run.answered fs hash = run.answered other hash) (target : Msg) :
    run.upTo fs hash target = run.upTo other hash target := by
  induction run with
  | done output => rfl
  | challenge point next induction =>
      have listed : (Sum.inl (point, fs point) :: (next (fs point)).answered fs hash :
          List ((Fs × Challenge) ⊕ (Msg × Digest))) =
          Sum.inl (point, other point) :: (next (other point)).answered other hash := same
      have value : fs point = other point :=
        (Prod.mk.inj (Sum.inl.inj (List.head_eq_of_cons_eq listed))).2
      have rest := List.tail_eq_of_cons_eq listed
      rw [← value] at rest
      simp only [upTo, ← value, induction (fs point) rest]
  | digest message next induction =>
      have listed : (Sum.inr (message, hash message) :: (next (hash message)).answered fs hash :
          List ((Fs × Challenge) ⊕ (Msg × Digest))) =
          Sum.inr (message, hash message) :: (next (hash message)).answered other hash := same
      simp only [upTo, induction (hash message) (List.tail_eq_of_cons_eq listed)]

/-- The first bad event of Akita Lemma 10.47 is a function of the answered queries. -/
theorem collided_congr {fs other : Fs → Challenge} {hash : Msg → Digest}
    {run : Run Fs Challenge Msg Digest Output}
    (same : run.answered fs hash = run.answered other hash) :
    Collided fs hash run ↔ Collided other hash run := by
  unfold Collided
  rw [trace_congr same]

/-- The second bad event of Akita Lemma 10.47 is a function of the answered queries. -/
theorem guessed_congr [DecidableEq Msg] (positions : Fs ⊕ Msg → List Digest)
    {fs other : Fs → Challenge} {hash : Msg → Digest} {run : Run Fs Challenge Msg Digest Output}
    (same : run.answered fs hash = run.answered other hash) :
    Guessed fs positions hash run ↔ Guessed other positions hash run := by
  unfold Guessed
  simp only [trace_congr same, upTo_congr same]

end NightstreamFPrime.Spec.Compression.Akita.Digest.Run

namespace NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.RandomOracle
open scoped BigOperators

attribute [local instance low] Classical.propDecidable

variable {Fs Y Msg Dig Addr Reply Output : Type}

/-- The cache of the recording prover: the address of each key of `H_FS` with a query, or `none`
for a key with no address. -/
abbrev Cache (Fs Addr : Type) : Type :=
  Fs → Option (Option Addr)

variable (resolve : List (Msg × Dig) → Fs → Option Addr) (φ : Fs → Reply × Y → Y)

/-- The answer of the recording prover at the key `point` with the address `slot`. -/
def answer (table : Addr → Reply) (aux : Fs → Y) (point : Fs) : Option Addr → Y
  | some address => φ point (table address, aux point)
  | none => aux point

/-- The table of `H_FS` that the cache gives: the answer of the recording prover at each key of
the cache, and `aux` at each other key. -/
def reads (cache : Cache Fs Addr) (table : Addr → Reply) (aux : Fs → Y) (point : Fs) : Y :=
  answer φ table aux point ((cache point).getD none)

/-- The recording prover of Akita Lemma 10.47 on `run`, from a cache and a record of the digest
queries (the newest first). It gives the output, the final cache and the final record. -/
def simulate [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig) :
    Digest.Run Fs Y Msg Dig Output → Cache Fs Addr → List (Msg × Dig) →
      Output × Cache Fs Addr × List (Msg × Dig)
  | .done output, cache, record => (output, cache, record)
  | .challenge point next, cache, record =>
      simulate table aux hash
        (next (answer φ table aux point ((cache point).getD (resolve record point))))
        (Function.update cache point (some ((cache point).getD (resolve record point)))) record
  | .digest message next, cache, record =>
      simulate table aux hash (next (hash message)) cache ((message, hash message) :: record)

theorem simulate_challenge [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)
    (hash : Msg → Dig) (point : Fs) (next : Y → Digest.Run Fs Y Msg Dig Output)
    (cache : Cache Fs Addr) (record : List (Msg × Dig)) :
    simulate resolve φ table aux hash (.challenge point next) cache record =
      simulate resolve φ table aux hash
        (next (answer φ table aux point ((cache point).getD (resolve record point))))
        (Function.update cache point (some ((cache point).getD (resolve record point)))) record :=
  rfl

theorem simulate_digest [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)
    (hash : Msg → Dig) (message : Msg) (next : Dig → Digest.Run Fs Y Msg Dig Output)
    (cache : Cache Fs Addr) (record : List (Msg × Dig)) :
    simulate resolve φ table aux hash (.digest message next) cache record =
      simulate resolve φ table aux hash (next (hash message)) cache
        ((message, hash message) :: record) :=
  rfl

theorem reads_update [DecidableEq Fs] (cache : Cache Fs Addr) (point : Fs) (slot : Option Addr)
    (table : Addr → Reply) (aux : Fs → Y) :
    reads φ (Function.update cache point (some slot)) table aux =
      Function.update (reads φ cache table aux) point (answer φ table aux point slot) := by
  funext other
  by_cases same : other = point
  · rw [same, Function.update_self]
    unfold reads
    rw [Function.update_self, Option.getD_some]
  · rw [Function.update_of_ne same]
    unfold reads
    rw [Function.update_of_ne same]

/-- The simulation keeps each entry of its cache. -/
theorem simulate_cache [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)) (point : Fs) (slot : Option Addr),
      cache point = some slot →
      (simulate resolve φ table aux hash run cache record).2.1 point = some slot := by
  induction run with
  | done output => exact fun cache record point slot found => found
  | challenge asked next induction =>
      intro cache record point slot found
      rw [simulate_challenge]
      refine induction _ _ record point slot ?_
      by_cases same : point = asked
      · rw [same] at found ⊢
        simp [found]
      · rw [Function.update_of_ne same]
        exact found
  | digest message next induction =>
      exact fun cache record point slot found => induction (hash message) cache _ point slot found

/-- The output of the simulation is the output of the run with the table that the final cache
gives. -/
theorem simulate_eval [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      (simulate resolve φ table aux hash run cache record).1 =
        run.eval (reads φ (simulate resolve φ table aux hash run cache record).2.1 table aux)
          hash := by
  induction run with
  | done output => exact fun _ _ => rfl
  | challenge point next induction =>
      intro cache record
      obtain ⟨slot, slotDef⟩ : ∃ slot, slot = (cache point).getD (resolve record point) :=
        ⟨_, rfl⟩
      rw [simulate_challenge, ← slotDef]
      have seen := simulate_cache resolve φ table aux hash
        (next (answer φ table aux point slot)) (Function.update cache point (some slot)) record
        point slot (Function.update_self point (some slot) cache)
      have value : reads φ (simulate resolve φ table aux hash
          (next (answer φ table aux point slot)) (Function.update cache point (some slot))
          record).2.1 table aux point = answer φ table aux point slot := by
        unfold reads
        rw [seen, Option.getD_some]
      refine (induction _ _ record).trans ?_
      rw [Digest.Run.eval_challenge, value]
  | digest message next induction =>
      exact fun cache record => induction (hash message) cache _

theorem simulate_map [DecidableEq Fs] {Other : Type} (f : Output → Other) (table : Addr → Reply)
    (aux : Fs → Y) (hash : Msg → Dig) (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      simulate resolve φ table aux hash (run.map f) cache record =
        (f (simulate resolve φ table aux hash run cache record).1,
          (simulate resolve φ table aux hash run cache record).2) := by
  induction run with
  | done output => exact fun _ _ => rfl
  | challenge point next induction =>
      exact fun cache record => induction
        (answer φ table aux point ((cache point).getD (resolve record point)))
        (Function.update cache point (some ((cache point).getD (resolve record point)))) record
  | digest message next induction =>
      exact fun cache record => induction (hash message) cache ((message, hash message) :: record)

/-- The simulation of `run.logged` has the output, the cache and the record of the simulation
of `run`. -/
theorem simulate_logged [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      (simulate resolve φ table aux hash run.logged cache record).1.1 =
          (simulate resolve φ table aux hash run cache record).1 ∧
        (simulate resolve φ table aux hash run.logged cache record).2 =
          (simulate resolve φ table aux hash run cache record).2 := by
  induction run with
  | done output => exact fun _ _ => ⟨rfl, rfl⟩
  | challenge point next induction =>
      intro cache record
      rw [Digest.Run.logged_challenge, simulate_challenge, simulate_challenge, simulate_map]
      exact induction _ _ record
  | digest message next induction =>
      intro cache record
      rw [Digest.Run.logged_digest, simulate_digest, simulate_digest, simulate_map]
      exact induction _ cache _

/-- The table of `H_FS` that the algorithm sees in the simulation from the empty cache. -/
def oracle [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)
    (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig)) : Fs → Y :=
  reads φ (simulate resolve φ table aux hash run (fun _ => none) record).2.1 table aux

theorem simulate_oracle [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)
    (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig)) :
    (simulate resolve φ table aux hash run (fun _ => none) record).1 =
      run.eval (oracle resolve φ table aux hash run record) hash :=
  simulate_eval resolve φ table aux hash run _ record

/-- The simulation of `run.logged` gives the output and the answered queries of the run with the
table that the algorithm sees. -/
theorem simulate_logged_oracle [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)
    (hash : Msg → Dig) (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig)) :
    (simulate resolve φ table aux hash run.logged (fun _ => none) record).1 =
      (run.eval (oracle resolve φ table aux hash run record) hash,
        run.answered (oracle resolve φ table aux hash run record) hash) := by
  unfold oracle
  rw [simulate_eval, (simulate_logged resolve φ table aux hash run _ record).2,
    Digest.Run.eval_logged]

/-- A property of the cache and the record that each step keeps holds at the end. -/
theorem simulate_invariant [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)
    (hash : Msg → Dig) (P : Cache Fs Addr → List (Msg × Dig) → Prop)
    (asked : ∀ cache record point, P cache record →
      P (Function.update cache point (some ((cache point).getD (resolve record point)))) record)
    (hashed : ∀ cache record message, P cache record →
      P cache ((message, hash message) :: record))
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)), P cache record →
      P (simulate resolve φ table aux hash run cache record).2.1
        (simulate resolve φ table aux hash run cache record).2.2 := by
  induction run with
  | done output => exact fun _ _ holds => holds
  | challenge point next induction =>
      intro cache record holds
      rw [simulate_challenge]
      exact induction _ _ record (asked cache record point holds)
  | digest message next induction =>
      intro cache record holds
      rw [simulate_digest]
      exact induction _ cache _ (hashed cache record message holds)

/-- Each entry of the final cache is `resolve` of the record at an earlier time. -/
theorem simulate_resolved [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)
    (hash : Msg → Dig) (run : Digest.Run Fs Y Msg Dig Output) (cache : Cache Fs Addr)
    (record : List (Msg × Dig))
    (start : ∀ point slot, cache point = some slot →
      ∃ early, early <:+ record ∧ slot = resolve early point) :
    ∀ point slot, (simulate resolve φ table aux hash run cache record).2.1 point = some slot →
      ∃ early, early <:+ (simulate resolve φ table aux hash run cache record).2.2 ∧
        slot = resolve early point := by
  refine simulate_invariant resolve φ table aux hash
    (fun cache record => ∀ point slot, cache point = some slot →
      ∃ early, early <:+ record ∧ slot = resolve early point) ?_ ?_ run cache record start
  · intro cache record queried holds point slot found
    by_cases same : point = queried
    · rw [same] at found ⊢
      rw [Function.update_self] at found
      cases cached : cache queried with
      | none =>
          rw [cached, Option.getD_none] at found
          exact ⟨record, List.suffix_refl record, (Option.some.inj found).symm⟩
      | some old =>
          rw [cached, Option.getD_some] at found
          exact holds queried slot (cached.trans found)
    · rw [Function.update_of_ne same] at found
      exact holds point slot found
  · intro cache record message holds point slot found
    obtain ⟨early, suffix, value⟩ := holds point slot found
    exact ⟨early, suffix.trans (List.suffix_cons _ record), value⟩

/-- The final record has the digest queries of the run, the newest first, before the first
record. -/
theorem simulate_record [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      (simulate resolve φ table aux hash run cache record).2.2 =
        ((simulate resolve φ table aux hash run.logged cache record).1.2.filterMap
          Sum.getRight?).reverse ++ record := by
  induction run with
  | done output => exact fun _ _ => rfl
  | challenge point next induction =>
      intro cache record
      rw [Digest.Run.logged_challenge, simulate_challenge, simulate_challenge, simulate_map,
        induction]
      simp [List.filterMap_cons]
  | digest message next induction =>
      intro cache record
      rw [Digest.Run.logged_digest, simulate_digest, simulate_digest, simulate_map, induction]
      simp

theorem simulate_record_oracle [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)
    (hash : Msg → Dig) (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig)) :
    (simulate resolve φ table aux hash run (fun _ => none) record).2.2 =
      ((run.answered (oracle resolve φ table aux hash run record) hash).filterMap
        Sum.getRight?).reverse ++ record := by
  rw [simulate_record]
  simp only [simulate_logged_oracle]

section Law

private theorem expect_swap {X Z : Type} [Fintype X] [Fintype Z] (f : X → Z → ℝ) :
    𝔼 x, 𝔼 z, f x z = 𝔼 z, 𝔼 x, f x z := by
  rw [Finset.expect_comm]

private theorem expect_sum {X Z : Type} [Fintype X] [Fintype Z] (f : X → Z → ℝ) :
    𝔼 x, ∑ z, f x z = ∑ z, 𝔼 x, f x z := by
  rw [Finset.expect_sum_comm]

private theorem expect_pair {R Z : Type} [Fintype R] [Fintype Z] (g : R × Z → ℝ) :
    𝔼 reply, 𝔼 value, g (reply, value) = 𝔼 x, g x := by
  simp only [Finset.expect_eq_sum_div_card, Finset.card_univ, Fintype.card_prod, Nat.cast_mul,
    ← Finset.sum_div, Fintype.sum_prod_type]
  rw [div_div, mul_comm]

private theorem expect_split {T A Z : Type} [Fintype T] [Fintype A] [Fintype Z] (h : T → A → Z)
    (K K' : Z → T → A → ℝ)
    (each : ∀ z, 𝔼 t, 𝔼 a, (if h t a = z then K z t a else 0) =
      𝔼 t, 𝔼 a, (if h t a = z then K' z t a else 0)) :
    𝔼 t, 𝔼 a, K (h t a) t a = 𝔼 t, 𝔼 a, K' (h t a) t a := by
  have spread (K : Z → T → A → ℝ) : 𝔼 t, 𝔼 a, K (h t a) t a =
      ∑ z, 𝔼 t, 𝔼 a, (if h t a = z then K z t a else 0) := by
    refine Eq.trans ?_ (expect_sum fun t z => 𝔼 a, if h t a = z then K z t a else 0)
    refine Finset.expect_congr rfl fun t _ => ?_
    refine Eq.trans ?_ (expect_sum fun a z => if h t a = z then K z t a else 0)
    exact Finset.expect_congr rfl fun a _ => by simp
  rw [spread K, spread K']
  exact Finset.sum_congr rfl fun z _ => each z

private theorem reads_table [DecidableEq Addr] {back : Addr → Fs} {cache : Cache Fs Addr}
    (sound : ∀ point address, cache point = some (some address) → back address = point)
    {point : Fs} (fresh : cache point = none) {address : Addr} (owner : back address = point)
    (table : Addr → Reply) (aux : Fs → Y) (reply : Reply) :
    reads φ cache (Function.update table address reply) aux = reads φ cache table aux := by
  funext other
  unfold reads
  cases found : cache other with
  | none => rfl
  | some slot =>
      cases slot with
      | none => rfl
      | some cell =>
          have apart : cell ≠ address := fun same => by
            have owned := sound other cell found
            rw [same, owner] at owned
            rw [← owned, fresh] at found
            simp at found
          show φ other (Function.update table address reply cell, aux other) =
            φ other (table cell, aux other)
          rw [Function.update_of_ne apart]

private theorem reads_aux [DecidableEq Fs] {cache : Cache Fs Addr} {point : Fs}
    (fresh : cache point = none) (table : Addr → Reply) (aux : Fs → Y) (value : Y) :
    reads φ cache table (Function.update aux point value) =
      Function.update (reads φ cache table aux) point value := by
  funext other
  by_cases same : other = point
  · rw [same, Function.update_self]
    unfold reads
    rw [fresh]
    exact Function.update_self point value aux
  · rw [Function.update_of_ne same]
    unfold reads
    cases (cache other).getD none with
    | none => exact Function.update_of_ne same _ _
    | some cell =>
        show φ other (table cell, Function.update aux point value other) =
          φ other (table cell, aux other)
        rw [Function.update_of_ne same]

private theorem expect_fibre_pair [Fintype Reply] [Nonempty Reply] [Fintype Y] [Nonempty Y]
    (ψ : Reply × Y → Y) (fibres : ∀ value, Nat.card {x // ψ x = value} = Fintype.card Reply)
    (g : Y → ℝ) : 𝔼 reply, 𝔼 value, g (ψ (reply, value)) = 𝔼 value, g value :=
  (expect_pair fun x => g (ψ x)).trans (expect_fibres ψ fibres g)

private theorem expect_reads_fresh [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Addr] [DecidableEq Addr] [Fintype Reply] [Nonempty Reply]
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    {back : Addr → Fs} {cache : Cache Fs Addr}
    (sound : ∀ point address, cache point = some (some address) → back address = point)
    {point : Fs} (fresh : cache point = none) {address : Addr} (owner : back address = point)
    (H : (Fs → Y) → ℝ) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
        H (Function.update (reads φ cache table aux) point
          (φ point (table address, aux point))) =
      𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, H (reads φ cache table aux) := by
  calc 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
        H (Function.update (reads φ cache table aux) point (φ point (table address, aux point)))
      = 𝔼 table : Addr → Reply, 𝔼 reply : Reply, 𝔼 aux : Fs → Y, 𝔼 value : Y,
          H (Function.update (reads φ cache table aux) point (φ point (reply, value))) := by
        rw [expect_update address]
        refine Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun reply _ => ?_
        rw [expect_update point]
        refine Finset.expect_congr rfl fun aux _ => Finset.expect_congr rfl fun value _ => ?_
        rw [Function.update_self, Function.update_self, reads_table φ sound fresh owner,
          reads_aux φ fresh, Function.update_idem]
    _ = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, 𝔼 reply : Reply, 𝔼 value : Y,
          H (Function.update (reads φ cache table aux) point (φ point (reply, value))) :=
        Finset.expect_congr rfl fun table _ => expect_swap _
    _ = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, 𝔼 value : Y,
          H (Function.update (reads φ cache table aux) point value) :=
        Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun aux _ =>
          expect_fibre_pair (φ point) (fibres point) fun value =>
            H (Function.update (reads φ cache table aux) point value)
    _ = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, H (reads φ cache table aux) :=
        Finset.expect_congr rfl fun table _ =>
          ((expect_update point fun aux => H (reads φ cache table aux)).trans
            (Finset.expect_congr rfl fun aux _ => Finset.expect_congr rfl fun value _ => by
              rw [reads_aux φ fresh])).symm

private theorem expect_reads_grow [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Addr] [DecidableEq Addr] [Fintype Reply] [Nonempty Reply]
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    {back : Addr → Fs} {cache : Cache Fs Addr}
    (sound : ∀ point address, cache point = some (some address) → back address = point)
    (point : Fs) (slot : Option Addr)
    (owner : ∀ address, slot = some address → back address = point)
    (kept : ∀ old, cache point = some old → old = slot) (H : (Fs → Y) → ℝ) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
        H (reads φ (Function.update cache point (some slot)) table aux) =
      𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, H (reads φ cache table aux) := by
  cases cached : cache point with
  | some old =>
      have same : Function.update cache point (some slot) = cache := by
        rw [← kept old cached, ← cached]
        exact Function.update_eq_self point cache
      rw [same]
  | none =>
      cases slot with
      | none =>
          have same : ∀ (table : Addr → Reply) (aux : Fs → Y),
              reads φ (Function.update cache point (some none)) table aux =
                reads φ cache table aux := by
            intro table aux
            have here : answer φ table aux point none = reads φ cache table aux point := by
              unfold reads
              rw [cached]
              rfl
            rw [reads_update, here]
            exact Function.update_eq_self point _
          simp only [same]
      | some address =>
          simp only [reads_update]
          exact expect_reads_fresh φ fibres sound cached (owner address rfl) H

private theorem simulate_reads [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Addr] [DecidableEq Addr] [Fintype Reply] [Nonempty Reply] {back : Addr → Fs}
    (hash : Msg → Dig)
    (resolves : ∀ record, (∀ entry ∈ record, entry.2 = hash entry.1) →
      ∀ point address, resolve record point = some address → back address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      (∀ entry ∈ record, entry.2 = hash entry.1) →
      (∀ point address, cache point = some (some address) → back address = point) →
      ∀ G : (Fs → Y) → Output → ℝ,
        (∀ f f' output, (∀ point, (cache point).isSome → f point = f' point) →
          G f output = G f' output) →
        𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
            (simulate resolve φ table aux hash run cache record).1 =
          𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
            (run.eval (reads φ cache table aux) hash) := by
  induction run with
  | done output => exact fun cache record honest sound G near => rfl
  | digest message next induction =>
      intro cache record honest sound G near
      refine induction (hash message) cache _ (fun entry member => ?_) sound G near
      rcases List.mem_cons.mp member with first | inside
      · rw [first]
      · exact honest entry inside
  | challenge point next induction =>
      intro cache record honest sound G near
      obtain ⟨slot, slotDef⟩ : ∃ slot, slot = (cache point).getD (resolve record point) :=
        ⟨_, rfl⟩
      obtain ⟨grown, grownDef⟩ : ∃ grown, grown = Function.update cache point (some slot) :=
        ⟨_, rfl⟩
      have owner : ∀ address, slot = some address → back address = point := by
        intro address found
        rw [slotDef] at found
        cases cached : cache point with
        | none =>
            rw [cached] at found
            exact resolves record honest point address found
        | some old =>
            rw [cached, Option.getD_some] at found
            exact sound point address (cached.trans (congrArg some found))
      have kept : ∀ old, cache point = some old → old = slot := by
        intro old cached
        rw [slotDef, cached, Option.getD_some]
      have unfolded : ∀ (table : Addr → Reply) (aux : Fs → Y),
          simulate resolve φ table aux hash (.challenge point next) cache record =
            simulate resolve φ table aux hash (next (reads φ grown table aux point)) grown
              record := by
        intro table aux
        rw [grownDef, reads_update, Function.update_self, slotDef, simulate_challenge]
      have sound' : ∀ other address, grown other = some (some address) →
          back address = other := by
        intro other address found
        rw [grownDef] at found
        by_cases same : other = point
        · rw [same] at found ⊢
          rw [Function.update_self] at found
          exact owner address (Option.some.inj found)
        · rw [Function.update_of_ne same] at found
          exact sound other address found
      have agree : ∀ (table : Addr → Reply) (aux : Fs → Y) (other : Fs),
          (cache other).isSome → reads φ cache table aux other = reads φ grown table aux other := by
        intro table aux other cachedOther
        rw [grownDef]
        by_cases same : other = point
        · obtain ⟨old, found⟩ := Option.isSome_iff_exists.mp cachedOther
          rw [same] at found ⊢
          unfold reads
          rw [Function.update_self, found, kept old found]
        · unfold reads
          rw [Function.update_of_ne same]
      have near' : ∀ f f' output, (∀ other, (grown other).isSome → f other = f' other) →
          G f output = G f' output := by
        intro f f' output same
        refine near f f' output fun other cachedOther => same other ?_
        rw [grownDef]
        by_cases here : other = point
        · rw [here]
          simp
        · rw [Function.update_of_ne here]
          exact cachedOther
      calc 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
            (simulate resolve φ table aux hash (.challenge point next) cache record).1
          = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ grown table aux)
              (simulate resolve φ table aux hash (next (reads φ grown table aux point)) grown
                record).1 :=
            Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun aux _ => by
              rw [unfolded]
              exact near _ _ _ fun other cachedOther => agree table aux other cachedOther
        _ = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ grown table aux)
              ((next (reads φ grown table aux point)).eval (reads φ grown table aux) hash) :=
            expect_split (fun table aux => reads φ grown table aux point)
              (fun value table aux => G (reads φ grown table aux)
                (simulate resolve φ table aux hash (next value) grown record).1)
              (fun value table aux => G (reads φ grown table aux)
                ((next value).eval (reads φ grown table aux) hash))
              fun value => induction value grown record honest sound'
                (fun f output => if f point = value then G f output else 0)
                fun f f' output same => by
                  have atPoint : f point = f' point := same point (by rw [grownDef]; simp)
                  show (if f point = value then G f output else 0) =
                    if f' point = value then G f' output else 0
                  rw [atPoint, near' f f' output same]
        _ = _ := by
            rw [grownDef]
            exact expect_reads_grow φ fibres sound point slot owner kept fun f =>
              G f ((Digest.Run.challenge point next).eval f hash)

/-- The recording prover of Akita Lemma 10.47 gives the algorithm a uniform `H_FS`: for a
uniform `table` and a uniform `aux`, the output of the simulation has the law of the output of
the run with a uniform table of `H_FS`, for each table `hash` of `H_o` and with no condition on
the bad events. `back` and `resolves`: for a record of answers of `hash`, an address is the
address of one key only. `fibres`: each `φ point` has fibres of `|Reply|` elements. -/
theorem simulate_law [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y] [Fintype Addr]
    [DecidableEq Addr] [Fintype Reply] [Nonempty Reply] {back : Addr → Fs} (hash : Msg → Dig)
    (resolves : ∀ record, (∀ entry ∈ record, entry.2 = hash entry.1) →
      ∀ point address, resolve record point = some address → back address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig))
    (honest : ∀ entry ∈ record, entry.2 = hash entry.1) (F : Output → ℝ) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
        F (simulate resolve φ table aux hash run (fun _ => none) record).1 =
      𝔼 fs : Fs → Y, F (run.eval fs hash) := by
  have core : 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
        F (simulate resolve φ table aux hash run (fun _ => none) record).1 =
      𝔼 _table : Addr → Reply, 𝔼 aux : Fs → Y, F (run.eval aux hash) :=
    simulate_reads resolve φ hash resolves fibres run (fun _ => none) record honest
      (fun _ _ found => by simp at found) (fun _ output => F output) (fun _ _ _ _ => rfl)
  exact core.trans (Finset.expect_const Finset.univ_nonempty _)

/-- `simulate_law` with a uniform table of `H_o`, for a value that reads this table also. -/
theorem simulate_law_hash [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y] [Fintype Addr]
    [DecidableEq Addr] [Fintype Reply] [Nonempty Reply] [Fintype Msg] [DecidableEq Msg]
    [Fintype Dig] {back : (Msg → Dig) → Addr → Fs}
    (resolves : ∀ hash record, (∀ entry ∈ record, entry.2 = hash entry.1) →
      ∀ point address, resolve record point = some address → back hash address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) (F : (Msg → Dig) → Output → ℝ) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, 𝔼 hash : Msg → Dig,
        F hash (simulate resolve φ table aux hash run (fun _ => none) []).1 =
      𝔼 fs : Fs → Y, 𝔼 hash : Msg → Dig, F hash (run.eval fs hash) := by
  calc 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, 𝔼 hash : Msg → Dig,
        F hash (simulate resolve φ table aux hash run (fun _ => none) []).1
      = 𝔼 table : Addr → Reply, 𝔼 hash : Msg → Dig, 𝔼 aux : Fs → Y,
          F hash (simulate resolve φ table aux hash run (fun _ => none) []).1 :=
        Finset.expect_congr rfl fun table _ => expect_swap _
    _ = 𝔼 hash : Msg → Dig, 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
          F hash (simulate resolve φ table aux hash run (fun _ => none) []).1 :=
        expect_swap _
    _ = 𝔼 hash : Msg → Dig, 𝔼 fs : Fs → Y, F hash (run.eval fs hash) :=
        Finset.expect_congr rfl fun hash _ =>
          simulate_law resolve φ hash (resolves hash) fibres run [] (by simp) (F hash)
    _ = 𝔼 fs : Fs → Y, 𝔼 hash : Msg → Dig, F hash (run.eval fs hash) := expect_swap _

/-- `simulate_law` for the output with the answered queries of the run, in their order. -/
theorem simulate_logged_law [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Addr] [DecidableEq Addr] [Fintype Reply] [Nonempty Reply] {back : Addr → Fs}
    (hash : Msg → Dig)
    (resolves : ∀ record, (∀ entry ∈ record, entry.2 = hash entry.1) →
      ∀ point address, resolve record point = some address → back address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig))
    (honest : ∀ entry ∈ record, entry.2 = hash entry.1)
    (F : Output × List ((Fs × Y) ⊕ (Msg × Dig)) → ℝ) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
        F (simulate resolve φ table aux hash run.logged (fun _ => none) record).1 =
      𝔼 fs : Fs → Y, F (run.eval fs hash, run.answered fs hash) := by
  rw [simulate_law resolve φ hash resolves fibres run.logged record honest F]
  exact Finset.expect_congr rfl fun fs _ => by rw [Digest.Run.eval_logged]

/-- The table of `H_FS` that the algorithm sees has the law of a uniform table, for each value
that reads the table through the answered queries of the run only. The output, the queries and
the events `Digest.Run.Collided` and `Digest.Run.Guessed` are such values. -/
theorem oracle_law [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y] [Fintype Addr]
    [DecidableEq Addr] [Fintype Reply] [Nonempty Reply] {back : Addr → Fs} (hash : Msg → Dig)
    (resolves : ∀ record, (∀ entry ∈ record, entry.2 = hash entry.1) →
      ∀ point address, resolve record point = some address → back address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig))
    (honest : ∀ entry ∈ record, entry.2 = hash entry.1) (S : (Fs → Y) → ℝ)
    (factors : ∀ fs other, run.answered fs hash = run.answered other hash → S fs = S other) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, S (oracle resolve φ table aux hash run record) =
      𝔼 fs : Fs → Y, S fs := by
  have pick : ∀ log : List ((Fs × Y) ⊕ (Msg × Dig)), ∃ value : ℝ,
      ∀ fs, run.answered fs hash = log → S fs = value := by
    intro log
    by_cases seen : ∃ fs, run.answered fs hash = log
    · obtain ⟨fs, same⟩ := seen
      exact ⟨S fs, fun other sameOther => factors other fs (sameOther.trans same.symm)⟩
    · exact ⟨0, fun fs same => (seen ⟨fs, same⟩).elim⟩
  choose value valued using pick
  have law := simulate_logged_law resolve φ hash resolves fibres run record honest
    fun result => value result.2
  refine Eq.trans ?_ (law.trans ?_)
  · refine Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun aux _ => ?_
    rw [simulate_logged_oracle]
    exact valued _ _ rfl
  · exact Finset.expect_congr rfl fun fs _ => (valued _ fs rfl).symm

/-- The fibres of the second projection have `|Reply|` elements. -/
theorem card_snd_fibre (Reply : Type) {Y : Type} [Fintype Reply] (value : Y) :
    Nat.card {x : Reply × Y // x.2 = value} = Fintype.card Reply :=
  (Nat.card_congr
    { toFun := fun x => x.1.1
      invFun := fun reply => ⟨(reply, value), rfl⟩
      left_inv := fun x => Subtype.ext (Prod.ext rfl x.2.symm)
      right_inv := fun _ => rfl }).trans Nat.card_eq_fintype_card

/-- The hypotheses of `simulate_law` hold for the keys as addresses and the second projection:
the guard against a vacuous statement. -/
theorem simulate_law_guard [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Reply] [Nonempty Reply] (hash : Msg → Dig) (run : Digest.Run Fs Y Msg Dig Output)
    (F : Output → ℝ) :
    𝔼 table : Fs → Reply, 𝔼 aux : Fs → Y,
        F (simulate (fun _ point => some point) (fun _ (x : Reply × Y) => x.2) table aux hash run
          (fun _ => none) []).1 =
      𝔼 fs : Fs → Y, F (run.eval fs hash) :=
  simulate_law _ _ (back := id) hash (fun _ _ _ _ found => (Option.some.inj found).symm)
    (fun _ value => card_snd_fibre Reply value) run [] (by simp) F

end Law

section Count

variable {Result : Type}

/-- One step of the recording prover as an algorithm with the oracle `table`: a key with an
address reads the table at the address. -/
def ask (aux : Fs → Y) (point : Fs) :
    Option Addr → (Y → OracleComp Addr Reply Result) → OracleComp Addr Reply Result
  | some address, rest => .query address fun reply => rest (φ point (reply, aux point))
  | none, rest => rest (aux point)

theorem ask_run (aux : Fs → Y) (point : Fs) (slot : Option Addr)
    (rest : Y → OracleComp Addr Reply Result) (table : Addr → Reply) :
    (ask φ aux point slot rest).run table = (rest (answer φ table aux point slot)).run table := by
  cases slot <;> rfl

private theorem bound_succ {computation : OracleComp Addr Reply Result} {bound : ℕ}
    (bounded : computation.QueryBound bound) : computation.QueryBound (bound + 1) := by
  induction bounded with
  | done output bound => exact .done output _
  | query point next bound each induction => exact .query point next (bound + 1) induction

theorem ask_bound (aux : Fs → Y) (point : Fs) (slot : Option Addr)
    {rest : Y → OracleComp Addr Reply Result} {bound : ℕ}
    (bounded : ∀ value, (rest value).QueryBound bound) :
    (ask φ aux point slot rest).QueryBound (bound + 1) := by
  cases slot with
  | none => exact bound_succ (bounded _)
  | some address => exact .query address _ bound fun reply => bounded _

end Count

/-- The recording prover for fixed `aux` and `hash` as an algorithm with the oracle `table`. -/
def comp [DecidableEq Fs] (aux : Fs → Y) (hash : Msg → Dig) :
    Digest.Run Fs Y Msg Dig Output → Cache Fs Addr → List (Msg × Dig) →
      OracleComp Addr Reply (Output × Cache Fs Addr × List (Msg × Dig))
  | .done output, cache, record => .done (output, cache, record)
  | .challenge point next, cache, record =>
      ask φ aux point ((cache point).getD (resolve record point)) fun value =>
        comp aux hash (next value)
          (Function.update cache point (some ((cache point).getD (resolve record point)))) record
  | .digest message next, cache, record =>
      comp aux hash (next (hash message)) cache ((message, hash message) :: record)

theorem comp_challenge [DecidableEq Fs] (aux : Fs → Y) (hash : Msg → Dig) (point : Fs)
    (next : Y → Digest.Run Fs Y Msg Dig Output) (cache : Cache Fs Addr)
    (record : List (Msg × Dig)) :
    comp resolve φ aux hash (.challenge point next) cache record =
      ask φ aux point ((cache point).getD (resolve record point)) fun value =>
        comp resolve φ aux hash (next value)
          (Function.update cache point (some ((cache point).getD (resolve record point))))
          record :=
  rfl

/-- The run of `comp` with the oracle `table` is the simulation. -/
theorem comp_run [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      (comp resolve φ aux hash run cache record).run table =
        simulate resolve φ table aux hash run cache record := by
  induction run with
  | done output => exact fun _ _ => rfl
  | challenge point next induction =>
      intro cache record
      rw [comp_challenge, ask_run, simulate_challenge]
      exact induction _ _ record
  | digest message next induction =>
      exact fun cache record => induction (hash message) cache _

/-- The recording prover has at most one table query for each query of the algorithm to `H_FS`
(Akita Lemma 10.47: "`P` makes at most `Q` queries"). -/
theorem comp_bound [DecidableEq Fs] (aux : Fs → Y) (hash : Msg → Dig)
    {run : Digest.Run Fs Y Msg Dig Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      (comp resolve φ aux hash run cache record).QueryBound queries := by
  induction bounded with
  | done output queries digests => exact fun _ _ => .done _ _
  | challenge point next queries digests each induction =>
      intro cache record
      rw [comp_challenge]
      exact ask_bound φ aux point _ fun value => induction value _ record
  | digest message next queries digests each induction =>
      exact fun cache record => induction (hash message) cache _

end NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate
