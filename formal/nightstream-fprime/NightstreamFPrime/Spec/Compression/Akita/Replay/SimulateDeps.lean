import NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate

/-! The recording prover of Akita (ePrint 2026/1983, revision of 2026-11-23, Lemma 10.47, the
paragraphs "The prover P" and "Simulation") for addresses that read earlier answers of `H_FS`
(Appendix E.1, (F1): the stage input reads the table at keys of earlier stages). No Akita type
is in this file.

- `resolve fs record point` is the address of the key `point` for the table `fs` of `H_FS`, and
  `deps point` has the keys at which it reads `fs`, the oldest first (`DepsLocal`). The keys that
  a key of `deps point` names are before it in `deps point` (`DepsClosed`).
- `simulateDeps`: the first `H_FS` query of a key first fixes each key of `deps point` that is
  not in the cache, in order, and then the key, each with `resolve` of the table that the cache
  gives (`settle`, `settleAll`).
- `simulateDeps_resolved`: each entry of the final cache is `resolve` of the table that the
  final cache gives and of the record at an earlier time.
- `compDeps` is `simulateDeps` as an algorithm with the oracle `table`. It keeps the answers
  that it gave. `compDeps_bound`: `(width + 1)` table queries for each `H_FS` query at most.
  `compDeps_fixed`: the table queries are the addresses of the keys that the simulation adds to
  the cache, each key one time, with a count for each class of keys. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.RandomOracle
open scoped BigOperators

attribute [local instance low] Classical.propDecidable

variable {Fs Y Msg Dig Addr Reply Output : Type}

variable (resolve : (Fs → Y) → List (Msg × Dig) → Fs → Option Addr) (deps : Fs → List Fs)
  (φ : Fs → Reply × Y → Y)

/-- `resolve` reads the table of `H_FS` only at the keys of `deps`. -/
def DepsLocal : Prop :=
  ∀ (fs other : Fs → Y) (record : List (Msg × Dig)) (point : Fs),
    (∀ key ∈ deps point, fs key = other key) → resolve fs record point = resolve other record point

/-- The keys that a key of `deps point` names are before this key in `deps point`. -/
def DepsClosed : Prop :=
  ∀ (point : Fs) (before : List Fs) (key : Fs) (after : List Fs),
    deps point = before ++ key :: after → ∀ named ∈ deps key, named ∈ before

theorem reads_of_eq {cache other : Cache Fs Addr} {point : Fs} (same : cache point = other point)
    (table : Addr → Reply) (aux : Fs → Y) :
    reads φ cache table aux point = reads φ other table aux point := by
  unfold reads
  rw [same]

/-- The cache after the first query of `point`: a key that is not in the cache gets `resolve` of
the table that the cache gives. -/
def settle [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (record : List (Msg × Dig))
    (cache : Cache Fs Addr) (point : Fs) : Cache Fs Addr :=
  Function.update cache point
    (some ((cache point).getD (resolve (reads φ cache table aux) record point)))

/-- `settle` for each key of `pending`, in order. -/
def settleAll [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (record : List (Msg × Dig))
    (cache : Cache Fs Addr) (pending : List Fs) : Cache Fs Addr :=
  pending.foldl (settle resolve φ table aux record) cache

section Settle

variable [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (record : List (Msg × Dig))

theorem settleAll_cons (cache : Cache Fs Addr) (point : Fs) (pending : List Fs) :
    settleAll resolve φ table aux record cache (point :: pending) =
      settleAll resolve φ table aux record (settle resolve φ table aux record cache point)
        pending :=
  rfl

theorem settle_cached {cache : Cache Fs Addr} {point : Fs} {slot : Option Addr}
    (cached : cache point = some slot) : settle resolve φ table aux record cache point = cache := by
  unfold settle
  rw [cached, Option.getD_some, ← cached]
  exact Function.update_eq_self point cache

theorem settle_fresh {cache : Cache Fs Addr} {point : Fs} (fresh : cache point = none) :
    settle resolve φ table aux record cache point =
      Function.update cache point (some (resolve (reads φ cache table aux) record point)) := by
  unfold settle
  rw [fresh, Option.getD_none]

theorem settle_of_ne (cache : Cache Fs Addr) {point other : Fs} (apart : other ≠ point) :
    settle resolve φ table aux record cache point other = cache other := by
  unfold settle
  rw [Function.update_of_ne apart]

theorem settle_self (cache : Cache Fs Addr) (point : Fs) :
    (settle resolve φ table aux record cache point point).isSome := by
  unfold settle
  rw [Function.update_self]
  rfl

theorem settle_keeps {cache : Cache Fs Addr} (point : Fs) {other : Fs} {slot : Option Addr}
    (found : cache other = some slot) :
    settle resolve φ table aux record cache point other = some slot := by
  by_cases same : other = point
  · rw [← same, settle_cached resolve φ table aux record found]
    exact found
  · rw [settle_of_ne resolve φ table aux record cache same]
    exact found

theorem settle_isSome {cache : Cache Fs Addr} (point : Fs) {other : Fs}
    (known : (cache other).isSome) :
    (settle resolve φ table aux record cache point other).isSome := by
  obtain ⟨slot, found⟩ := Option.isSome_iff_exists.mp known
  rw [settle_keeps resolve φ table aux record point found]
  rfl

theorem settleAll_keeps (pending : List Fs) :
    ∀ (cache : Cache Fs Addr) (other : Fs) (slot : Option Addr), cache other = some slot →
      settleAll resolve φ table aux record cache pending other = some slot := by
  induction pending with
  | nil => exact fun cache other slot found => found
  | cons point pending induction =>
      exact fun cache other slot found =>
        induction _ other slot (settle_keeps resolve φ table aux record point found)

theorem settleAll_mem (pending : List Fs) :
    ∀ (cache : Cache Fs Addr) (point : Fs), point ∈ pending →
      (settleAll resolve φ table aux record cache pending point).isSome := by
  induction pending with
  | nil => exact fun cache point member => by simp at member
  | cons first pending induction =>
      intro cache point member
      rcases List.mem_cons.mp member with same | inside
      · obtain ⟨slot, found⟩ := Option.isSome_iff_exists.mp
          (settle_self resolve φ table aux record cache first)
        rw [same, settleAll_cons,
          settleAll_keeps resolve φ table aux record pending _ first slot found]
        rfl
      · exact induction _ point inside

end Settle

/-- The recording prover of Akita Lemma 10.47 on `run` with addresses that read earlier answers,
from a cache and a record of the digest queries (the newest first). It gives the output, the
final cache and the final record. -/
def simulateDeps [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig) :
    Digest.Run Fs Y Msg Dig Output → Cache Fs Addr → List (Msg × Dig) →
      Output × Cache Fs Addr × List (Msg × Dig)
  | .done output, cache, record => (output, cache, record)
  | .challenge point next, cache, record =>
      simulateDeps table aux hash
        (next (reads φ (settleAll resolve φ table aux record cache (deps point ++ [point])) table
          aux point))
        (settleAll resolve φ table aux record cache (deps point ++ [point])) record
  | .digest message next, cache, record =>
      simulateDeps table aux hash (next (hash message)) cache ((message, hash message) :: record)

section Run

variable [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)

theorem simulateDeps_challenge (point : Fs) (next : Y → Digest.Run Fs Y Msg Dig Output)
    (cache : Cache Fs Addr) (record : List (Msg × Dig)) :
    simulateDeps resolve deps φ table aux hash (.challenge point next) cache record =
      simulateDeps resolve deps φ table aux hash
        (next (reads φ (settleAll resolve φ table aux record cache (deps point ++ [point])) table
          aux point))
        (settleAll resolve φ table aux record cache (deps point ++ [point])) record :=
  rfl

theorem simulateDeps_digest (message : Msg) (next : Dig → Digest.Run Fs Y Msg Dig Output)
    (cache : Cache Fs Addr) (record : List (Msg × Dig)) :
    simulateDeps resolve deps φ table aux hash (.digest message next) cache record =
      simulateDeps resolve deps φ table aux hash (next (hash message)) cache
        ((message, hash message) :: record) :=
  rfl

/-- The simulation keeps each entry of its cache. -/
theorem simulateDeps_cache (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)) (point : Fs) (slot : Option Addr),
      cache point = some slot →
      (simulateDeps resolve deps φ table aux hash run cache record).2.1 point = some slot := by
  induction run with
  | done output => exact fun cache record point slot found => found
  | challenge asked next induction =>
      intro cache record point slot found
      rw [simulateDeps_challenge]
      exact induction _ _ record point slot
        (settleAll_keeps resolve φ table aux record _ cache point slot found)
  | digest message next induction =>
      exact fun cache record point slot found => induction (hash message) cache _ point slot found

/-- The output of the simulation is the output of the run with the table that the final cache
gives. -/
theorem simulateDeps_eval (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      (simulateDeps resolve deps φ table aux hash run cache record).1 =
        run.eval (reads φ (simulateDeps resolve deps φ table aux hash run cache record).2.1 table
          aux) hash := by
  induction run with
  | done output => exact fun _ _ => rfl
  | challenge point next induction =>
      intro cache record
      obtain ⟨grown, grownDef⟩ : ∃ grown,
          grown = settleAll resolve φ table aux record cache (deps point ++ [point]) := ⟨_, rfl⟩
      rw [simulateDeps_challenge, ← grownDef]
      obtain ⟨slot, found⟩ := Option.isSome_iff_exists.mp (show (grown point).isSome by
        rw [grownDef]
        exact settleAll_mem resolve φ table aux record _ cache point (by simp))
      have value := reads_of_eq φ ((simulateDeps_cache resolve deps φ table aux hash
        (next (reads φ grown table aux point)) grown record point slot found).trans found.symm)
        table aux
      refine (induction _ _ record).trans ?_
      rw [Digest.Run.eval_challenge, value]
  | digest message next induction =>
      exact fun cache record => induction (hash message) cache _

theorem simulateDeps_map {Other : Type} (f : Output → Other)
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      simulateDeps resolve deps φ table aux hash (run.map f) cache record =
        (f (simulateDeps resolve deps φ table aux hash run cache record).1,
          (simulateDeps resolve deps φ table aux hash run cache record).2) := by
  induction run with
  | done output => exact fun _ _ => rfl
  | challenge point next induction => exact fun cache record => induction _ _ record
  | digest message next induction =>
      exact fun cache record => induction (hash message) cache ((message, hash message) :: record)

/-- The simulation of `run.logged` has the output, the cache and the record of the simulation
of `run`. -/
theorem simulateDeps_logged (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      (simulateDeps resolve deps φ table aux hash run.logged cache record).1.1 =
          (simulateDeps resolve deps φ table aux hash run cache record).1 ∧
        (simulateDeps resolve deps φ table aux hash run.logged cache record).2 =
          (simulateDeps resolve deps φ table aux hash run cache record).2 := by
  induction run with
  | done output => exact fun _ _ => ⟨rfl, rfl⟩
  | challenge point next induction =>
      intro cache record
      rw [Digest.Run.logged_challenge, simulateDeps_challenge, simulateDeps_challenge,
        simulateDeps_map]
      exact induction _ _ record
  | digest message next induction =>
      intro cache record
      rw [Digest.Run.logged_digest, simulateDeps_digest, simulateDeps_digest, simulateDeps_map]
      exact induction _ cache _

/-- The table of `H_FS` that the algorithm sees in the simulation from the empty cache. -/
def oracleDeps (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig)) : Fs → Y :=
  reads φ (simulateDeps resolve deps φ table aux hash run (fun _ => none) record).2.1 table aux

theorem simulateDeps_oracle (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig)) :
    (simulateDeps resolve deps φ table aux hash run (fun _ => none) record).1 =
      run.eval (oracleDeps resolve deps φ table aux hash run record) hash :=
  simulateDeps_eval resolve deps φ table aux hash run _ record

/-- The simulation of `run.logged` gives the output and the answered queries of the run with the
table that the algorithm sees. -/
theorem simulateDeps_logged_oracle (run : Digest.Run Fs Y Msg Dig Output)
    (record : List (Msg × Dig)) :
    (simulateDeps resolve deps φ table aux hash run.logged (fun _ => none) record).1 =
      (run.eval (oracleDeps resolve deps φ table aux hash run record) hash,
        run.answered (oracleDeps resolve deps φ table aux hash run record) hash) := by
  unfold oracleDeps
  rw [simulateDeps_eval, (simulateDeps_logged resolve deps φ table aux hash run _ record).2,
    Digest.Run.eval_logged]

/-- A property of the cache and the record that each step keeps holds at the end. -/
theorem simulateDeps_invariant (P : Cache Fs Addr → List (Msg × Dig) → Prop)
    (asked : ∀ cache record point, P cache record →
      P (settleAll resolve φ table aux record cache (deps point ++ [point])) record)
    (hashed : ∀ cache record message, P cache record →
      P cache ((message, hash message) :: record))
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)), P cache record →
      P (simulateDeps resolve deps φ table aux hash run cache record).2.1
        (simulateDeps resolve deps φ table aux hash run cache record).2.2 := by
  induction run with
  | done output => exact fun _ _ holds => holds
  | challenge point next induction =>
      intro cache record holds
      rw [simulateDeps_challenge]
      exact induction _ _ record (asked cache record point holds)
  | digest message next induction =>
      intro cache record holds
      rw [simulateDeps_digest]
      exact induction _ cache _ (hashed cache record message holds)

/-- The final record has the digest queries of the run, the newest first, before the first
record. -/
theorem simulateDeps_record (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      (simulateDeps resolve deps φ table aux hash run cache record).2.2 =
        ((simulateDeps resolve deps φ table aux hash run.logged cache record).1.2.filterMap
          Sum.getRight?).reverse ++ record := by
  induction run with
  | done output => exact fun _ _ => rfl
  | challenge point next induction =>
      intro cache record
      rw [Digest.Run.logged_challenge, simulateDeps_challenge, simulateDeps_challenge,
        simulateDeps_map, induction]
      simp [List.filterMap_cons]
  | digest message next induction =>
      intro cache record
      rw [Digest.Run.logged_digest, simulateDeps_digest, simulateDeps_digest, simulateDeps_map,
        induction]
      simp

theorem simulateDeps_record_oracle (run : Digest.Run Fs Y Msg Dig Output)
    (record : List (Msg × Dig)) :
    (simulateDeps resolve deps φ table aux hash run (fun _ => none) record).2.2 =
      ((run.answered (oracleDeps resolve deps φ table aux hash run record) hash).filterMap
        Sum.getRight?).reverse ++ record := by
  rw [simulateDeps_record]
  simp only [simulateDeps_logged_oracle]

end Run

/-- Each key of `pending` names only keys of the cache and keys before it in `pending`. -/
def Ready (cache : Cache Fs Addr) (pending : List Fs) : Prop :=
  ∀ (before : List Fs) (key : Fs) (after : List Fs), pending = before ++ key :: after →
    ∀ named ∈ deps key, named ∈ before ∨ (cache named).isSome

theorem ready_chain (closed : DepsClosed deps) (cache : Cache Fs Addr) (point : Fs) :
    Ready deps cache (deps point ++ [point]) := by
  intro before key after split named member
  rcases List.eq_nil_or_concat after with empty | ⟨front, last, parts⟩
  · rw [empty] at split
    obtain ⟨same, single⟩ := List.append_inj' split rfl
    rw [← List.head_eq_of_cons_eq single, same] at member
    exact Or.inl member
  · have regroup : before ++ key :: after = (before ++ key :: front) ++ [last] := by
      rw [parts]
      simp
    obtain ⟨same, -⟩ := List.append_inj' (split.trans regroup) rfl
    exact Or.inl (closed point before key front same named member)

theorem ready_head {cache : Cache Fs Addr} {key : Fs} {pending : List Fs}
    (ready : Ready deps cache (key :: pending)) : ∀ named ∈ deps key, (cache named).isSome := by
  intro named member
  rcases ready [] key pending rfl named member with inside | cached
  · simp at inside
  · exact cached

theorem ready_tail {cache grown : Cache Fs Addr} {key : Fs} {pending : List Fs}
    (ready : Ready deps cache (key :: pending))
    (keeps : ∀ other, (cache other).isSome → (grown other).isSome)
    (cached : (grown key).isSome) : Ready deps grown pending := by
  intro before next after split named member
  rcases ready (key :: before) next after (by rw [split]; rfl) named member with inside | old
  · rcases List.mem_cons.mp inside with same | inside
    · rw [same]
      exact Or.inr cached
    · exact Or.inl inside
  · exact Or.inr (keeps named old)

/-- Each entry of the cache is `resolve` of the table that the cache gives and of the record at
an earlier time, and the cache has each key that the entry names. -/
def Resolved (table : Addr → Reply) (aux : Fs → Y) (cache : Cache Fs Addr)
    (record : List (Msg × Dig)) : Prop :=
  ∀ point slot, cache point = some slot →
    (∀ named ∈ deps point, (cache named).isSome) ∧
      ∃ early, early <:+ record ∧ slot = resolve (reads φ cache table aux) early point

theorem resolved_empty (table : Addr → Reply) (aux : Fs → Y) (record : List (Msg × Dig)) :
    Resolved resolve deps φ table aux (fun _ => none) record :=
  fun _ _ found => by simp at found

section Final

variable [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)

theorem settle_resolved (local_ : DepsLocal resolve deps) (record : List (Msg × Dig))
    {cache : Cache Fs Addr} {point : Fs} (named : ∀ key ∈ deps point, (cache key).isSome)
    (resolved : Resolved resolve deps φ table aux cache record) :
    Resolved resolve deps φ table aux (settle resolve φ table aux record cache point) record := by
  cases cached : cache point with
  | some old =>
      rw [settle_cached resolve φ table aux record cached]
      exact resolved
  | none =>
      have apart : ∀ key, (cache key).isSome → key ≠ point := fun key known same => by
        rw [same, cached] at known
        simp at known
      have same : ∀ other, (∀ key ∈ deps other, (cache key).isSome) → ∀ early,
          resolve (reads φ (settle resolve φ table aux record cache point) table aux) early other =
            resolve (reads φ cache table aux) early other :=
        fun other known early => local_ _ _ early other fun key member =>
          reads_of_eq φ (settle_of_ne resolve φ table aux record cache
            (apart key (known key member))) table aux
      intro other slot found
      by_cases here : other = point
      · rw [here, settle_fresh resolve φ table aux record cached, Function.update_self] at found
        rw [here]
        refine ⟨fun key member => settle_isSome resolve φ table aux record point
          (named key member), record, List.suffix_refl record, ?_⟩
        rw [same point named record]
        exact (Option.some.inj found).symm
      · rw [settle_of_ne resolve φ table aux record cache here] at found
        obtain ⟨known, early, suffix, value⟩ := resolved other slot found
        refine ⟨fun key member => settle_isSome resolve φ table aux record point
          (known key member), early, suffix, ?_⟩
        rw [same other known early]
        exact value

theorem settleAll_resolved (local_ : DepsLocal resolve deps) (record : List (Msg × Dig))
    (pending : List Fs) :
    ∀ cache : Cache Fs Addr, Ready deps cache pending →
      Resolved resolve deps φ table aux cache record →
      Resolved resolve deps φ table aux (settleAll resolve φ table aux record cache pending)
        record := by
  induction pending with
  | nil => exact fun cache ready resolved => resolved
  | cons point pending induction =>
      intro cache ready resolved
      exact induction _
        (ready_tail deps ready
          (fun other known => settle_isSome resolve φ table aux record point known)
          (settle_self resolve φ table aux record cache point))
        (settle_resolved resolve deps φ table aux local_ record (ready_head deps ready) resolved)

/-- Each entry of the final cache is `resolve` of the table that the final cache gives and of
the record at an earlier time. -/
theorem simulateDeps_resolved (local_ : DepsLocal resolve deps) (closed : DepsClosed deps)
    (hash : Msg → Dig) (run : Digest.Run Fs Y Msg Dig Output) (cache : Cache Fs Addr)
    (record : List (Msg × Dig)) (start : Resolved resolve deps φ table aux cache record) :
    Resolved resolve deps φ table aux
      (simulateDeps resolve deps φ table aux hash run cache record).2.1
      (simulateDeps resolve deps φ table aux hash run cache record).2.2 :=
  simulateDeps_invariant resolve deps φ table aux hash (Resolved resolve deps φ table aux)
    (fun cache record point resolved => settleAll_resolved resolve deps φ table aux local_ record
      _ cache (ready_chain deps closed cache point) resolved)
    (fun cache record message resolved point slot found => by
      obtain ⟨known, early, suffix, value⟩ := resolved point slot found
      exact ⟨known, early, suffix.trans (List.suffix_cons _ record), value⟩)
    run cache record start

/-- From the empty cache: each entry of the final cache is `resolve` of the table that the
algorithm sees and of the record at an earlier time. -/
theorem simulateDeps_resolved_oracle (local_ : DepsLocal resolve deps)
    (closed : DepsClosed deps) (hash : Msg → Dig) (run : Digest.Run Fs Y Msg Dig Output)
    (record : List (Msg × Dig)) (point : Fs) (slot : Option Addr)
    (found : (simulateDeps resolve deps φ table aux hash run (fun _ => none) record).2.1 point =
      some slot) :
    ∃ early, early <:+ (simulateDeps resolve deps φ table aux hash run (fun _ => none) record).2.2 ∧
      slot = resolve (oracleDeps resolve deps φ table aux hash run record) early point :=
  (simulateDeps_resolved resolve deps φ table aux local_ closed hash run _ record
    (resolved_empty resolve deps φ table aux record) point slot found).2

end Final

section Law

private theorem swap_expect {X Z : Type} [Fintype X] [Fintype Z] (f : X → Z → ℝ) :
    𝔼 x, 𝔼 z, f x z = 𝔼 z, 𝔼 x, f x z := by
  rw [Finset.expect_comm]

private theorem sum_expect {X Z : Type} [Fintype X] [Fintype Z] (f : X → Z → ℝ) :
    𝔼 x, ∑ z, f x z = ∑ z, 𝔼 x, f x z := by
  rw [Finset.expect_sum_comm]

private theorem pair_expect {R Z : Type} [Fintype R] [Fintype Z] (g : R × Z → ℝ) :
    𝔼 reply, 𝔼 value, g (reply, value) = 𝔼 x, g x := by
  simp only [Finset.expect_eq_sum_div_card, Finset.card_univ, Fintype.card_prod, Nat.cast_mul,
    ← Finset.sum_div, Fintype.sum_prod_type]
  rw [div_div, mul_comm]

private theorem split_expect {T A Z : Type} [Fintype T] [Fintype A] [Fintype Z] (h : T → A → Z)
    (K K' : Z → T → A → ℝ)
    (each : ∀ z, 𝔼 t, 𝔼 a, (if h t a = z then K z t a else 0) =
      𝔼 t, 𝔼 a, (if h t a = z then K' z t a else 0)) :
    𝔼 t, 𝔼 a, K (h t a) t a = 𝔼 t, 𝔼 a, K' (h t a) t a := by
  have spread (K : Z → T → A → ℝ) : 𝔼 t, 𝔼 a, K (h t a) t a =
      ∑ z, 𝔼 t, 𝔼 a, (if h t a = z then K z t a else 0) := by
    refine Eq.trans ?_ (sum_expect fun t z => 𝔼 a, if h t a = z then K z t a else 0)
    refine Finset.expect_congr rfl fun t _ => ?_
    refine Eq.trans ?_ (sum_expect fun a z => if h t a = z then K z t a else 0)
    exact Finset.expect_congr rfl fun a _ => by simp
  rw [spread K, spread K']
  exact Finset.sum_congr rfl fun z _ => each z

private theorem table_reads [DecidableEq Addr] {cache : Cache Fs Addr} {address : Addr}
    (apart : ∀ other cell, cache other = some (some cell) → cell ≠ address)
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
          show φ other (Function.update table address reply cell, aux other) =
            φ other (table cell, aux other)
          rw [Function.update_of_ne (apart other cell found)]

private theorem aux_reads [DecidableEq Fs] {cache : Cache Fs Addr} {point : Fs}
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

private theorem fresh_expect [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Addr] [DecidableEq Addr] [Fintype Reply] [Nonempty Reply]
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    {cache : Cache Fs Addr} {address : Addr}
    (apart : ∀ other cell, cache other = some (some cell) → cell ≠ address)
    {point : Fs} (fresh : cache point = none) (H : (Fs → Y) → ℝ) :
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
        rw [Function.update_self, Function.update_self, table_reads φ apart, aux_reads φ fresh,
          Function.update_idem]
    _ = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, 𝔼 reply : Reply, 𝔼 value : Y,
          H (Function.update (reads φ cache table aux) point (φ point (reply, value))) :=
        Finset.expect_congr rfl fun table _ => swap_expect _
    _ = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, 𝔼 value : Y,
          H (Function.update (reads φ cache table aux) point value) :=
        Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun aux _ =>
          (pair_expect fun x => H (Function.update (reads φ cache table aux) point
            (φ point x))).trans (expect_fibres (φ point) (fibres point) fun value =>
              H (Function.update (reads φ cache table aux) point value))
    _ = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, H (reads φ cache table aux) :=
        Finset.expect_congr rfl fun table _ =>
          ((expect_update point fun aux => H (reads φ cache table aux)).trans
            (Finset.expect_congr rfl fun aux _ => Finset.expect_congr rfl fun value _ => by
              rw [aux_reads φ fresh])).symm

private theorem settle_expect [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Addr] [DecidableEq Addr] [Fintype Reply] [Nonempty Reply]
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    {cache : Cache Fs Addr} {point : Fs} (fresh : cache point = none) (slot : Option Addr)
    (apart : ∀ address, slot = some address →
      ∀ other cell, cache other = some (some cell) → cell ≠ address)
    (H : (Fs → Y) → ℝ) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
        H (reads φ (Function.update cache point (some slot)) table aux) =
      𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, H (reads φ cache table aux) := by
  cases slot with
  | none =>
      have same : ∀ (table : Addr → Reply) (aux : Fs → Y),
          reads φ (Function.update cache point (some none)) table aux =
            reads φ cache table aux := by
        intro table aux
        have here : answer φ table aux point none = reads φ cache table aux point := by
          unfold reads
          rw [fresh]
          rfl
        rw [reads_update, here]
        exact Function.update_eq_self point _
      simp only [same]
  | some address =>
      simp only [reads_update]
      exact fresh_expect φ fibres (apart address rfl) fresh H

private def Full (cache : Cache Fs Addr) : Prop :=
  ∀ point, (cache point).isSome → ∀ named ∈ deps point, (cache named).isSome

private def Sound (hash : Msg → Dig) (cache : Cache Fs Addr) (ref : Fs → Y) : Prop :=
  ∀ point address, cache point = some (some address) →
    ∃ early, (∀ entry ∈ early, entry.2 = hash entry.1) ∧ resolve ref early point = some address

private def Seen (cache : Cache Fs Addr) (ref f : Fs → Y) : Prop :=
  ∀ point, (cache point).isSome → f point = ref point

private def Near (cache : Cache Fs Addr) (G : (Fs → Y) → Output → ℝ) : Prop :=
  ∀ f f' output, (∀ point, (cache point).isSome → f point = f' point) → G f output = G f' output

private def Support (cache : Cache Fs Addr) (ref : Fs → Y) (G : (Fs → Y) → Output → ℝ) : Prop :=
  ∀ f output, ¬ Seen cache ref f → G f output = 0

private theorem chain_reads [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Addr] [DecidableEq Addr] [Fintype Reply] [Nonempty Reply]
    {back : (Fs → Y) → Addr → Fs} (hash : Msg → Dig) (local_ : DepsLocal resolve deps)
    (resolves : ∀ fs record, (∀ entry ∈ record, entry.2 = hash entry.1) →
      ∀ point address, resolve fs record point = some address → back fs address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (point : Fs) (next : Y → Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig))
    (honest : ∀ entry ∈ record, entry.2 = hash entry.1)
    (induction : ∀ (value : Y) (cache : Cache Fs Addr) (ref : Fs → Y), Full deps cache →
      Sound resolve hash cache ref → ∀ G : (Fs → Y) → Output → ℝ, Near cache G →
      Support cache ref G →
      𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
          (simulateDeps resolve deps φ table aux hash (next value) cache record).1 =
        𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
          ((next value).eval (reads φ cache table aux) hash))
    (pending : List Fs) :
    ∀ (cache : Cache Fs Addr) (ref : Fs → Y), Full deps cache → Sound resolve hash cache ref →
      Ready deps cache pending → (point ∈ pending ∨ (cache point).isSome) →
      ∀ G : (Fs → Y) → Output → ℝ, Near cache G → Support cache ref G →
      𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
          (simulateDeps resolve deps φ table aux hash
            (next (reads φ (settleAll resolve φ table aux record cache pending) table aux point))
            (settleAll resolve φ table aux record cache pending) record).1 =
        𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
          ((next (reads φ cache table aux point)).eval (reads φ cache table aux) hash) := by
  induction pending with
  | nil =>
      intro cache ref full sound ready listed G near support
      have cached : (cache point).isSome := listed.resolve_left (by simp)
      have pointwise : ∀ (table : Addr → Reply) (aux : Fs → Y) (K : Y → Output),
          G (reads φ cache table aux) (K (reads φ cache table aux point)) =
            G (reads φ cache table aux) (K (ref point)) := by
        intro table aux K
        by_cases seen : Seen cache ref (reads φ cache table aux)
        · rw [seen point cached]
        · rw [support _ _ seen, support _ _ seen]
      calc 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
            (simulateDeps resolve deps φ table aux hash
              (next (reads φ (settleAll resolve φ table aux record cache []) table aux point))
              (settleAll resolve φ table aux record cache []) record).1
          = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
              (simulateDeps resolve deps φ table aux hash (next (ref point)) cache record).1 :=
            Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun aux _ =>
              pointwise table aux fun value =>
                (simulateDeps resolve deps φ table aux hash (next value) cache record).1
        _ = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
              ((next (ref point)).eval (reads φ cache table aux) hash) :=
            induction (ref point) cache ref full sound G near support
        _ = _ :=
            Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun aux _ =>
              (pointwise table aux fun value =>
                (next value).eval (reads φ cache table aux) hash).symm
  | cons key pending inner =>
      intro cache ref full sound ready listed G near support
      cases cached : cache key with
      | some old =>
          have same : ∀ (table : Addr → Reply) (aux : Fs → Y),
              settleAll resolve φ table aux record cache (key :: pending) =
                settleAll resolve φ table aux record cache pending := fun table aux => by
            rw [settleAll_cons, settle_cached resolve φ table aux record cached]
          simp only [same]
          refine inner cache ref full sound
            (ready_tail deps ready (fun _ known => known) (by rw [cached]; rfl)) ?_ G near support
          rcases listed with member | known
          · rcases List.mem_cons.mp member with first | inside
            · exact Or.inr (by rw [first, cached]; rfl)
            · exact Or.inl inside
          · exact Or.inr known
      | none =>
          have named := ready_head deps ready
          have apart : ∀ other, (cache other).isSome → other ≠ key := fun other known same => by
            rw [same, cached] at known
            simp at known
          obtain ⟨slot, slotDef⟩ : ∃ slot, slot = resolve ref record key := ⟨_, rfl⟩
          obtain ⟨grown, grownDef⟩ : ∃ grown, grown = Function.update cache key (some slot) :=
            ⟨_, rfl⟩
          have settled : ∀ (table : Addr → Reply) (aux : Fs → Y),
              Seen cache ref (reads φ cache table aux) →
              settle resolve φ table aux record cache key = grown := by
            intro table aux seen
            rw [settle_fresh resolve φ table aux record cached, grownDef, slotDef,
              local_ (reads φ cache table aux) ref record key fun other member =>
                seen other (named other member)]
          have keeps : ∀ other, (cache other).isSome → (grown other).isSome := by
            intro other known
            rw [grownDef, Function.update_of_ne (apart other known)]
            exact known
          have here : (grown key).isSome := by
            rw [grownDef, Function.update_self]
            rfl
          have nearG : ∀ (table : Addr → Reply) (aux : Fs → Y) (output : Output),
              G (reads φ grown table aux) output = G (reads φ cache table aux) output :=
            fun table aux output => near _ _ output fun other known =>
              reads_of_eq φ (by rw [grownDef, Function.update_of_ne (apart other known)]) table aux
          have full' : Full deps grown := by
            intro other known dep member
            by_cases same : other = key
            · rw [same] at member
              exact keeps dep (named dep member)
            · rw [grownDef, Function.update_of_ne same] at known
              exact keeps dep (full other known dep member)
          have sound' : ∀ value, Sound resolve hash grown (Function.update ref key value) := by
            intro value other address found
            have fixed : ∀ target, (∀ dep ∈ deps target, (cache dep).isSome) → ∀ early,
                resolve (Function.update ref key value) early target = resolve ref early target :=
              fun target known early => local_ _ _ early target fun dep member =>
                Function.update_of_ne (apart dep (known dep member)) _ _
            by_cases same : other = key
            · rw [same, grownDef, Function.update_self] at found
              refine ⟨record, honest, ?_⟩
              rw [same, fixed key named record, ← slotDef]
              exact Option.some.inj found
            · rw [grownDef, Function.update_of_ne same] at found
              obtain ⟨early, true_, value'⟩ := sound other address found
              exact ⟨early, true_,
                (fixed other (full other (by rw [found]; rfl)) early).trans value'⟩
          have distinct : ∀ address, slot = some address →
              ∀ other cell, cache other = some (some cell) → cell ≠ address := by
            intro address found other cell entry same
            obtain ⟨early, true_, value⟩ := sound other cell entry
            have owner := resolves ref early true_ other cell value
            rw [same, resolves ref record honest key address (slotDef.symm.trans found)] at owner
            exact apart other (by rw [entry]; rfl) owner.symm
          have listed' : point ∈ pending ∨ (grown point).isSome := by
            rcases listed with member | known
            · rcases List.mem_cons.mp member with first | inside
              · exact Or.inr (by rw [first]; exact here)
              · exact Or.inl inside
            · exact Or.inr (keeps point known)
          have near' : ∀ value : Y,
              Near grown fun f (output : Output) => if f key = value then G f output else 0 := by
            intro value f f' output same
            have atKey : f key = f' key := same key here
            show (if f key = value then G f output else 0) =
              if f' key = value then G f' output else 0
            rw [atKey, near f f' output fun other known => same other (keeps other known)]
          have support' : ∀ value : Y, Support grown (Function.update ref key value)
              fun f (output : Output) => if f key = value then G f output else 0 := by
            intro value f output unseen
            show (if f key = value then G f output else 0) = 0
            by_cases hit : f key = value
            · rw [ite_eq_left hit]
              refine support f output fun seen => unseen fun other known => ?_
              by_cases same : other = key
              · rw [same, Function.update_self]
                exact hit
              · rw [Function.update_of_ne same]
                refine seen other ?_
                rw [grownDef, Function.update_of_ne same] at known
                exact known
            · rw [ite_eq_right hit]
          calc 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
                (simulateDeps resolve deps φ table aux hash
                  (next (reads φ (settleAll resolve φ table aux record cache (key :: pending))
                    table aux point))
                  (settleAll resolve φ table aux record cache (key :: pending)) record).1
              = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ grown table aux)
                  (simulateDeps resolve deps φ table aux hash
                    (next (reads φ (settleAll resolve φ table aux record grown pending) table aux
                      point))
                    (settleAll resolve φ table aux record grown pending) record).1 :=
                Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun aux _ => by
                  rw [nearG]
                  by_cases seen : Seen cache ref (reads φ cache table aux)
                  · rw [settleAll_cons, settled table aux seen]
                  · rw [support _ _ seen, support _ _ seen]
            _ = 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ grown table aux)
                  ((next (reads φ grown table aux point)).eval (reads φ grown table aux) hash) :=
                split_expect (fun table aux => reads φ grown table aux key)
                  (fun _ table aux => G (reads φ grown table aux)
                    (simulateDeps resolve deps φ table aux hash
                      (next (reads φ (settleAll resolve φ table aux record grown pending) table
                        aux point))
                      (settleAll resolve φ table aux record grown pending) record).1)
                  (fun _ table aux => G (reads φ grown table aux)
                    ((next (reads φ grown table aux point)).eval (reads φ grown table aux) hash))
                  fun value => inner grown (Function.update ref key value) full' (sound' value)
                    (ready_tail deps ready keeps here) listed'
                    (fun f output => if f key = value then G f output else 0) (near' value)
                    (support' value)
            _ = _ := by
                rw [grownDef]
                exact settle_expect φ fibres cached slot distinct fun f =>
                  G f ((next (f point)).eval f hash)

private theorem simulateDeps_reads [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Addr] [DecidableEq Addr] [Fintype Reply] [Nonempty Reply]
    {back : (Fs → Y) → Addr → Fs} (hash : Msg → Dig) (local_ : DepsLocal resolve deps)
    (closed : DepsClosed deps)
    (resolves : ∀ fs record, (∀ entry ∈ record, entry.2 = hash entry.1) →
      ∀ point address, resolve fs record point = some address → back fs address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)) (ref : Fs → Y),
      (∀ entry ∈ record, entry.2 = hash entry.1) → Full deps cache →
      Sound resolve hash cache ref → ∀ G : (Fs → Y) → Output → ℝ, Near cache G →
      Support cache ref G →
      𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
          (simulateDeps resolve deps φ table aux hash run cache record).1 =
        𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, G (reads φ cache table aux)
          (run.eval (reads φ cache table aux) hash) := by
  induction run with
  | done output => exact fun _ _ _ _ _ _ _ _ _ => rfl
  | digest message next induction =>
      intro cache record ref honest full sound G near support
      refine induction (hash message) cache _ ref (fun entry member => ?_) full sound G near
        support
      rcases List.mem_cons.mp member with first | inside
      · rw [first]
      · exact honest entry inside
  | challenge point next induction =>
      intro cache record ref honest full sound G near support
      exact chain_reads resolve deps φ hash local_ resolves fibres point next record honest
        (fun value cache ref => induction value cache record ref honest) (deps point ++ [point])
        cache ref full sound (ready_chain deps closed cache point) (Or.inl (by simp)) G near
        support

/-- The recording prover of Akita Lemma 10.47 with addresses that read earlier answers gives the
algorithm a uniform `H_FS`: for a uniform `table` and a uniform `aux`, the output of the
simulation has the law of the output of the run with a uniform table of `H_FS`, for each table
`hash` of `H_o` and with no condition on the bad events. `local_`: `resolve` reads `H_FS` only
at the keys of `deps`. `closed`: the keys that a key of `deps point` names are before it in
`deps point`. `back` and `resolves`: for one table of `H_FS` and a record of answers of `hash`,
an address is the address of one key only. `fibres`: each `φ point` has fibres of `|Reply|`
elements. -/
theorem simulateDeps_law [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y] [Fintype Addr]
    [DecidableEq Addr] [Fintype Reply] [Nonempty Reply] {back : (Fs → Y) → Addr → Fs}
    (hash : Msg → Dig) (local_ : DepsLocal resolve deps) (closed : DepsClosed deps)
    (resolves : ∀ fs record, (∀ entry ∈ record, entry.2 = hash entry.1) →
      ∀ point address, resolve fs record point = some address → back fs address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig))
    (honest : ∀ entry ∈ record, entry.2 = hash entry.1) (F : Output → ℝ) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
        F (simulateDeps resolve deps φ table aux hash run (fun _ => none) record).1 =
      𝔼 fs : Fs → Y, F (run.eval fs hash) := by
  have core : 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
        F (simulateDeps resolve deps φ table aux hash run (fun _ => none) record).1 =
      𝔼 _table : Addr → Reply, 𝔼 aux : Fs → Y, F (run.eval aux hash) :=
    simulateDeps_reads resolve deps φ hash local_ closed resolves fibres run (fun _ => none)
      record (fun _ => Classical.arbitrary Y) honest (fun _ known => by simp at known)
      (fun _ _ found => by simp at found) (fun _ output => F output) (fun _ _ _ _ => rfl)
      (fun _ _ unseen => (unseen fun _ known => by simp at known).elim)
  exact core.trans (Finset.expect_const Finset.univ_nonempty _)

/-- `simulateDeps_law` with a uniform table of `H_o`, for a value that reads this table also. -/
theorem simulateDeps_law_hash [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Addr] [DecidableEq Addr] [Fintype Reply] [Nonempty Reply] [Fintype Msg]
    [DecidableEq Msg] [Fintype Dig] {back : (Msg → Dig) → (Fs → Y) → Addr → Fs}
    (local_ : DepsLocal resolve deps) (closed : DepsClosed deps)
    (resolves : ∀ hash fs record, (∀ entry ∈ record, entry.2 = hash entry.1) →
      ∀ point address, resolve fs record point = some address → back hash fs address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) (F : (Msg → Dig) → Output → ℝ) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, 𝔼 hash : Msg → Dig,
        F hash (simulateDeps resolve deps φ table aux hash run (fun _ => none) []).1 =
      𝔼 fs : Fs → Y, 𝔼 hash : Msg → Dig, F hash (run.eval fs hash) := by
  calc 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y, 𝔼 hash : Msg → Dig,
        F hash (simulateDeps resolve deps φ table aux hash run (fun _ => none) []).1
      = 𝔼 table : Addr → Reply, 𝔼 hash : Msg → Dig, 𝔼 aux : Fs → Y,
          F hash (simulateDeps resolve deps φ table aux hash run (fun _ => none) []).1 :=
        Finset.expect_congr rfl fun table _ => swap_expect _
    _ = 𝔼 hash : Msg → Dig, 𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
          F hash (simulateDeps resolve deps φ table aux hash run (fun _ => none) []).1 :=
        swap_expect _
    _ = 𝔼 hash : Msg → Dig, 𝔼 fs : Fs → Y, F hash (run.eval fs hash) :=
        Finset.expect_congr rfl fun hash _ =>
          simulateDeps_law resolve deps φ hash local_ closed (resolves hash) fibres run []
            (by simp) (F hash)
    _ = 𝔼 fs : Fs → Y, 𝔼 hash : Msg → Dig, F hash (run.eval fs hash) := swap_expect _

/-- `simulateDeps_law` for the output with the answered queries of the run, in their order. -/
theorem simulateDeps_logged_law [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Addr] [DecidableEq Addr] [Fintype Reply] [Nonempty Reply]
    {back : (Fs → Y) → Addr → Fs} (hash : Msg → Dig) (local_ : DepsLocal resolve deps)
    (closed : DepsClosed deps)
    (resolves : ∀ fs record, (∀ entry ∈ record, entry.2 = hash entry.1) →
      ∀ point address, resolve fs record point = some address → back fs address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig))
    (honest : ∀ entry ∈ record, entry.2 = hash entry.1)
    (F : Output × List ((Fs × Y) ⊕ (Msg × Dig)) → ℝ) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
        F (simulateDeps resolve deps φ table aux hash run.logged (fun _ => none) record).1 =
      𝔼 fs : Fs → Y, F (run.eval fs hash, run.answered fs hash) := by
  rw [simulateDeps_law resolve deps φ hash local_ closed resolves fibres run.logged record honest
    F]
  exact Finset.expect_congr rfl fun fs _ => by rw [Digest.Run.eval_logged]

/-- The table of `H_FS` that the algorithm sees has the law of a uniform table, for each value
that reads the table through the answered queries of the run only. The output, the queries and
the events `Digest.Run.Collided` and `Digest.Run.Guessed` are such values. -/
theorem oracleDeps_law [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y] [Fintype Addr]
    [DecidableEq Addr] [Fintype Reply] [Nonempty Reply] {back : (Fs → Y) → Addr → Fs}
    (hash : Msg → Dig) (local_ : DepsLocal resolve deps) (closed : DepsClosed deps)
    (resolves : ∀ fs record, (∀ entry ∈ record, entry.2 = hash entry.1) →
      ∀ point address, resolve fs record point = some address → back fs address = point)
    (fibres : ∀ point value, Nat.card {x // φ point x = value} = Fintype.card Reply)
    (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig))
    (honest : ∀ entry ∈ record, entry.2 = hash entry.1) (S : (Fs → Y) → ℝ)
    (factors : ∀ fs other, run.answered fs hash = run.answered other hash → S fs = S other) :
    𝔼 table : Addr → Reply, 𝔼 aux : Fs → Y,
        S (oracleDeps resolve deps φ table aux hash run record) =
      𝔼 fs : Fs → Y, S fs := by
  have pick : ∀ log : List ((Fs × Y) ⊕ (Msg × Dig)), ∃ value : ℝ,
      ∀ fs, run.answered fs hash = log → S fs = value := by
    intro log
    by_cases seen : ∃ fs, run.answered fs hash = log
    · obtain ⟨fs, same⟩ := seen
      exact ⟨S fs, fun other sameOther => factors other fs (sameOther.trans same.symm)⟩
    · exact ⟨0, fun fs same => (seen ⟨fs, same⟩).elim⟩
  choose value valued using pick
  have law := simulateDeps_logged_law resolve deps φ hash local_ closed resolves fibres run record
    honest fun result => value result.2
  refine Eq.trans ?_ (law.trans ?_)
  · refine Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun aux _ => ?_
    rw [simulateDeps_logged_oracle]
    exact valued _ _ rfl
  · exact Finset.expect_congr rfl fun fs _ => (valued _ fs rfl).symm

/-- The hypotheses of `simulateDeps_law` hold for the keys as addresses, no named key and the
second projection. -/
theorem simulateDeps_law_guard [Fintype Fs] [DecidableEq Fs] [Fintype Y] [Nonempty Y]
    [Fintype Reply] [Nonempty Reply] (hash : Msg → Dig) (run : Digest.Run Fs Y Msg Dig Output)
    (F : Output → ℝ) :
    𝔼 table : Fs → Reply, 𝔼 aux : Fs → Y,
        F (simulateDeps (fun _ _ point => some point) (fun _ => [])
          (fun _ (x : Reply × Y) => x.2) table aux hash run (fun _ => none) []).1 =
      𝔼 fs : Fs → Y, F (run.eval fs hash) :=
  simulateDeps_law _ _ _ (back := fun _ => id) hash (fun _ _ _ _ _ => rfl)
    (fun _ _ _ _ split => absurd split (by simp))
    (fun _ _ _ _ _ found => (Option.some.inj found).symm)
    (fun _ value => card_snd_fibre Reply value) run [] (by simp) F

/-- The hypotheses of `simulateDeps_law` hold for two keys where the address of the second key
has the answer of `H_FS` at the first key: the guard against a vacuous statement. -/
theorem simulateDeps_law_guard_reads [Fintype Y] [DecidableEq Y] [Nonempty Y] [Fintype Reply]
    [Nonempty Reply] (hash : Msg → Dig) (run : Digest.Run Bool Y Msg Dig Output)
    (F : Output → ℝ) :
    𝔼 table : Bool × Option Y → Reply, 𝔼 aux : Bool → Y,
        F (simulateDeps
          (fun fs _ point => some (point, if point then some (fs false) else none))
          (fun point => if point then [false] else []) (fun _ (x : Reply × Y) => x.2) table aux
          hash run (fun _ => none) []).1 =
      𝔼 fs : Bool → Y, F (run.eval fs hash) :=
  simulateDeps_law _ _ _ (back := fun _ address => address.1) hash
    (fun fs other _ point same => by
      cases point with
      | false => rfl
      | true => exact congrArg (fun value => some (true, some value)) (same false (by simp)))
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
    (fun _ _ _ _ _ found => (congrArg Prod.fst (Option.some.inj found)).symm)
    (fun _ value => card_snd_fibre Reply value) run [] (by simp) F

end Law

section Count

variable {Result : Type}

private theorem bound_mono {computation : OracleComp Addr Reply Result} {bound : ℕ}
    (bounded : computation.QueryBound bound) :
    ∀ larger, bound ≤ larger → computation.QueryBound larger := by
  induction bounded with
  | done output bound => exact fun _ _ => .done output _
  | query point next bound each induction =>
      intro larger le
      obtain ⟨more, rfl⟩ : ∃ more, larger = more + 1 := ⟨larger - 1, by omega⟩
      exact .query point next more fun answer => induction answer more (by omega)

/-- One key of the chain in the algorithm with the oracle `table`, for the entry `entry` of the
cache at the key: a new key with an address reads the table at the address. `seen` has the
answers that the algorithm gave. -/
def settleWith [DecidableEq Fs] (aux : Fs → Y) (point : Fs) (slot : Option Addr)
    (cache : Cache Fs Addr) (seen : Fs → Y)
    (rest : Cache Fs Addr → (Fs → Y) → OracleComp Addr Reply Result) :
    Option (Option Addr) → OracleComp Addr Reply Result
  | some _ => rest cache seen
  | none => ask φ aux point slot fun value =>
      rest (Function.update cache point (some slot)) (Function.update seen point value)

/-- `settle` in the algorithm with the oracle `table`. -/
def settleComp [DecidableEq Fs] (aux : Fs → Y) (record : List (Msg × Dig)) (point : Fs)
    (cache : Cache Fs Addr) (seen : Fs → Y)
    (rest : Cache Fs Addr → (Fs → Y) → OracleComp Addr Reply Result) :
    OracleComp Addr Reply Result :=
  settleWith φ aux point (resolve seen record point) cache seen rest (cache point)

/-- `settleAll` in the algorithm with the oracle `table`. -/
def settleAllComp [DecidableEq Fs] (aux : Fs → Y) (record : List (Msg × Dig)) :
    List Fs → Cache Fs Addr → (Fs → Y) →
      (Cache Fs Addr → (Fs → Y) → OracleComp Addr Reply Result) → OracleComp Addr Reply Result
  | [], cache, seen, rest => rest cache seen
  | point :: pending, cache, seen, rest =>
      settleComp resolve φ aux record point cache seen fun grown answers =>
        settleAllComp aux record pending grown answers rest

theorem settleComp_run [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)
    (record : List (Msg × Dig)) (point : Fs) (cache : Cache Fs Addr)
    (rest : Cache Fs Addr → (Fs → Y) → OracleComp Addr Reply Result) :
    (settleComp resolve φ aux record point cache (reads φ cache table aux) rest).run table =
      (rest (settle resolve φ table aux record cache point)
        (reads φ (settle resolve φ table aux record cache point) table aux)).run table := by
  unfold settleComp
  cases cached : cache point with
  | some old =>
      rw [settle_cached resolve φ table aux record cached]
      rfl
  | none =>
      rw [settle_fresh resolve φ table aux record cached, reads_update]
      exact ask_run φ aux point _ _ table

theorem settleAllComp_run [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)
    (record : List (Msg × Dig))
    (rest : Cache Fs Addr → (Fs → Y) → OracleComp Addr Reply Result) (pending : List Fs) :
    ∀ cache : Cache Fs Addr,
      (settleAllComp resolve φ aux record pending cache (reads φ cache table aux) rest).run table =
        (rest (settleAll resolve φ table aux record cache pending)
          (reads φ (settleAll resolve φ table aux record cache pending) table aux)).run table := by
  induction pending with
  | nil => exact fun cache => rfl
  | cons point pending induction =>
      intro cache
      exact (settleComp_run resolve φ table aux record point cache _).trans (induction _)

theorem settleComp_bound [DecidableEq Fs] (aux : Fs → Y) (record : List (Msg × Dig))
    (point : Fs) (cache : Cache Fs Addr) (seen : Fs → Y)
    {rest : Cache Fs Addr → (Fs → Y) → OracleComp Addr Reply Result} {bound : ℕ}
    (bounded : ∀ cache seen, (rest cache seen).QueryBound bound) :
    (settleComp resolve φ aux record point cache seen rest).QueryBound (bound + 1) := by
  unfold settleComp
  cases cache point with
  | some old => exact bound_mono (bounded cache seen) _ (Nat.le_succ bound)
  | none => exact ask_bound φ aux point _ fun value => bounded _ _

theorem settleAllComp_bound [DecidableEq Fs] (aux : Fs → Y) (record : List (Msg × Dig))
    {rest : Cache Fs Addr → (Fs → Y) → OracleComp Addr Reply Result} {bound : ℕ}
    (bounded : ∀ cache seen, (rest cache seen).QueryBound bound) (pending : List Fs) :
    ∀ (cache : Cache Fs Addr) (seen : Fs → Y),
      (settleAllComp resolve φ aux record pending cache seen rest).QueryBound
        (bound + pending.length) := by
  induction pending with
  | nil => exact fun cache seen => bounded cache seen
  | cons point pending induction =>
      exact fun cache seen => settleComp_bound resolve φ aux record point cache seen induction

end Count

/-- The recording prover with addresses that read earlier answers, for fixed `aux` and `hash`,
as an algorithm with the oracle `table`. `seen` has the answers that the algorithm gave. -/
def compDeps [DecidableEq Fs] (aux : Fs → Y) (hash : Msg → Dig) :
    Digest.Run Fs Y Msg Dig Output → Cache Fs Addr → List (Msg × Dig) → (Fs → Y) →
      OracleComp Addr Reply (Output × Cache Fs Addr × List (Msg × Dig))
  | .done output, cache, record, _ => .done (output, cache, record)
  | .challenge point next, cache, record, seen =>
      settleAllComp resolve φ aux record (deps point ++ [point]) cache seen fun grown answers =>
        compDeps aux hash (next (answers point)) grown record answers
  | .digest message next, cache, record, seen =>
      compDeps aux hash (next (hash message)) cache ((message, hash message) :: record) seen

theorem compDeps_challenge [DecidableEq Fs] (aux : Fs → Y) (hash : Msg → Dig) (point : Fs)
    (next : Y → Digest.Run Fs Y Msg Dig Output) (cache : Cache Fs Addr)
    (record : List (Msg × Dig)) (seen : Fs → Y) :
    compDeps resolve deps φ aux hash (.challenge point next) cache record seen =
      settleAllComp resolve φ aux record (deps point ++ [point]) cache seen fun grown answers =>
        compDeps resolve deps φ aux hash (next (answers point)) grown record answers :=
  rfl

/-- The run of `compDeps` with the oracle `table`, from the answers that the cache gives, is the
simulation. -/
theorem compDeps_run [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)),
      (compDeps resolve deps φ aux hash run cache record (reads φ cache table aux)).run table =
        simulateDeps resolve deps φ table aux hash run cache record := by
  induction run with
  | done output => exact fun _ _ => rfl
  | challenge point next induction =>
      intro cache record
      rw [compDeps_challenge, settleAllComp_run, simulateDeps_challenge]
      exact induction _ _ record
  | digest message next induction =>
      exact fun cache record => induction (hash message) cache _

/-- The recording prover has at most `width + 1` table queries for each query of the algorithm
to `H_FS`, when each key names at most `width` keys. -/
theorem compDeps_bound [DecidableEq Fs] (aux : Fs → Y) (hash : Msg → Dig) {width : ℕ}
    (wide : ∀ point, (deps point).length ≤ width) {run : Digest.Run Fs Y Msg Dig Output}
    {queries digests : ℕ} (bounded : run.Bound queries digests) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)) (seen : Fs → Y),
      (compDeps resolve deps φ aux hash run cache record seen).QueryBound
        ((width + 1) * queries) := by
  induction bounded with
  | done output queries digests => exact fun _ _ _ => .done _ _
  | challenge point next queries digests each induction =>
      intro cache record seen
      rw [compDeps_challenge]
      have expand : (width + 1) * (queries + 1) = (width + 1) * queries + (width + 1) :=
        Nat.mul_succ _ _
      refine bound_mono (settleAllComp_bound resolve φ aux record
        (fun grown answers => induction (answers point) grown record answers)
        (deps point ++ [point]) cache seen) _ ?_
      rw [List.length_append, List.length_singleton, expand]
      exact Nat.add_le_add_left (Nat.add_le_add_right (wide point) 1) _
  | digest message next queries digests each induction =>
      exact fun cache record seen => induction (hash message) cache _ seen

section Fixed

variable {Result : Type}

theorem ask_queries (aux : Fs → Y) (point : Fs) (slot : Option Addr)
    (rest : Y → OracleComp Addr Reply Result) (table : Addr → Reply) :
    (ask φ aux point slot rest).queries table =
      slot.toList ++ (rest (answer φ table aux point slot)).queries table := by
  cases slot <;> rfl

theorem settleComp_cached [DecidableEq Fs] (aux : Fs → Y) (record : List (Msg × Dig))
    {point : Fs} {cache : Cache Fs Addr} {old : Option Addr} (cached : cache point = some old)
    (seen : Fs → Y) (rest : Cache Fs Addr → (Fs → Y) → OracleComp Addr Reply Result) :
    settleComp resolve φ aux record point cache seen rest = rest cache seen := by
  unfold settleComp
  rw [cached]
  rfl

theorem settleComp_queries [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)
    (record : List (Msg × Dig)) {point : Fs} {cache : Cache Fs Addr} (fresh : cache point = none)
    (rest : Cache Fs Addr → (Fs → Y) → OracleComp Addr Reply Result) :
    (settleComp resolve φ aux record point cache (reads φ cache table aux) rest).queries table =
      (resolve (reads φ cache table aux) record point).toList ++
        (rest (settle resolve φ table aux record cache point)
          (reads φ (settle resolve φ table aux record cache point) table aux)).queries table := by
  unfold settleComp
  rw [fresh, settle_fresh resolve φ table aux record fresh, reads_update]
  exact ask_queries φ aux point _ _ table

/-- The table queries of `settleAllComp` are the addresses of the keys that it adds to the
cache, in order. Each such key is in `pending`, and no key is added two times. -/
theorem settleAllComp_fixed [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y)
    (record : List (Msg × Dig))
    (rest : Cache Fs Addr → (Fs → Y) → OracleComp Addr Reply Result) (pending : List Fs) :
    ∀ cache : Cache Fs Addr, ∃ fixed : List Fs, fixed.Nodup ∧ fixed.Sublist pending ∧
      (∀ key ∈ fixed, cache key = none) ∧
      (settleAllComp resolve φ aux record pending cache (reads φ cache table aux) rest).queries
          table =
        fixed.filterMap (fun key =>
            (settleAll resolve φ table aux record cache pending key).join) ++
          (rest (settleAll resolve φ table aux record cache pending)
            (reads φ (settleAll resolve φ table aux record cache pending) table aux)).queries
            table := by
  induction pending with
  | nil =>
      exact fun cache => ⟨[], List.nodup_nil, List.Sublist.refl _,
        fun _ member => by simp at member, rfl⟩
  | cons point pending induction =>
      intro cache
      cases cached : cache point with
      | some old =>
          obtain ⟨fixed, nodup, sub, fresh, listed⟩ := induction cache
          refine ⟨fixed, nodup, sub.cons point, fresh, ?_⟩
          rw [settleAll_cons, settle_cached resolve φ table aux record cached]
          exact (congrArg (fun computation => computation.queries table)
            (settleComp_cached resolve φ aux record cached (reads φ cache table aux)
              fun grown answers =>
                settleAllComp resolve φ aux record pending grown answers rest)).trans listed
      | none =>
          obtain ⟨fixed, nodup, sub, fresh, listed⟩ :=
            induction (settle resolve φ table aux record cache point)
          have grownAt : settle resolve φ table aux record cache point point =
              some (resolve (reads φ cache table aux) record point) := by
            rw [settle_fresh resolve φ table aux record cached, Function.update_self]
          have absent : point ∉ fixed := fun member => by
            rw [fresh point member] at grownAt
            simp at grownAt
          have older : ∀ key ∈ fixed, cache key = none := by
            intro key member
            cases found : cache key with
            | none => rfl
            | some slot =>
                have kept := settle_keeps resolve φ table aux record point found
                rw [fresh key member] at kept
                simp at kept
          refine ⟨point :: fixed, List.nodup_cons.mpr ⟨absent, nodup⟩, sub.cons_cons point, ?_, ?_⟩
          · intro key member
            rcases List.mem_cons.mp member with same | inside
            · rw [same]
              exact cached
            · exact older key inside
          · have head : (settleAll resolve φ table aux record
                (settle resolve φ table aux record cache point) pending point).join =
                resolve (reads φ cache table aux) record point := by
              rw [settleAll_keeps resolve φ table aux record pending _ point _ grownAt]
              rfl
            refine ((settleComp_queries resolve φ table aux record cached _).trans
              (congrArg (_ ++ ·) listed)).trans ?_
            rw [settleAll_cons]
            simp only [List.filterMap_cons, head]
            cases resolve (reads φ cache table aux) record point <;> rfl

/-- The table queries of `compDeps` are the addresses of the keys that the simulation adds to
the cache, in order, and no key is added two times. For each class `kind` of keys: when the
chain of each key has at most `count` keys of the class, the simulation adds at most `count`
keys of the class for each query of the algorithm to `H_FS`. -/
theorem compDeps_fixed [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)
    {run : Digest.Run Fs Y Msg Dig Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)), ∃ fixed : List Fs, fixed.Nodup ∧
      (∀ key ∈ fixed, cache key = none ∧
        ((simulateDeps resolve deps φ table aux hash run cache record).2.1 key).isSome) ∧
      (compDeps resolve deps φ aux hash run cache record (reads φ cache table aux)).queries
          table =
        fixed.filterMap (fun key =>
          ((simulateDeps resolve deps φ table aux hash run cache record).2.1 key).join) ∧
      ∀ (kind : Fs → Bool) (count : ℕ),
        (∀ point, ((deps point ++ [point]).filter kind).length ≤ count) →
        (fixed.filter kind).length ≤ count * queries := by
  induction bounded with
  | done output queries digests =>
      exact fun cache record => ⟨[], List.nodup_nil, fun _ member => by simp at member, rfl,
        fun _ _ _ => Nat.zero_le _⟩
  | digest message next queries digests each induction =>
      exact fun cache record => induction (hash message) cache _
  | challenge point next queries digests each induction =>
      intro cache record
      obtain ⟨grown, grownDef⟩ : ∃ grown,
          grown = settleAll resolve φ table aux record cache (deps point ++ [point]) := ⟨_, rfl⟩
      rw [simulateDeps_challenge, compDeps_challenge, ← grownDef]
      obtain ⟨first, nodup₁, sub, fresh₁, listed₁⟩ := settleAllComp_fixed resolve φ table aux
        record (fun grown answers =>
          compDeps resolve deps φ aux hash (next (answers point)) grown record answers)
        (deps point ++ [point]) cache
      obtain ⟨second, nodup₂, fresh₂, listed₂, counted⟩ :=
        induction (reads φ grown table aux point) grown record
      rw [← grownDef] at listed₁
      have known : ∀ key ∈ first, (grown key).isSome := fun key member => by
        rw [grownDef]
        exact settleAll_mem resolve φ table aux record _ cache key (sub.subset member)
      refine ⟨first ++ second,
        List.nodup_append.mpr ⟨nodup₁, nodup₂, fun key inFirst other inSecond same => ?_⟩,
        fun key member => ?_, ?_, fun kind count few => ?_⟩
      · have cached := known key inFirst
        rw [same, (fresh₂ other inSecond).1] at cached
        simp at cached
      · rcases List.mem_append.mp member with inFirst | inSecond
        · obtain ⟨slot, found⟩ := Option.isSome_iff_exists.mp (known key inFirst)
          refine ⟨fresh₁ key inFirst, ?_⟩
          rw [simulateDeps_cache resolve deps φ table aux hash _ grown record key slot found]
          rfl
        · refine ⟨?_, (fresh₂ key inSecond).2⟩
          cases found : cache key with
          | none => rfl
          | some slot =>
              have kept := settleAll_keeps resolve φ table aux record (deps point ++ [point])
                cache key slot found
              rw [← grownDef, (fresh₂ key inSecond).1] at kept
              simp at kept
      · rw [listed₁, listed₂, List.filterMap_append]
        refine congrArg (· ++ _) (List.filterMap_congr fun key member => ?_)
        obtain ⟨slot, found⟩ := Option.isSome_iff_exists.mp (known key member)
        rw [simulateDeps_cache resolve deps φ table aux hash _ grown record key slot found, found]
      · have expand : count * (queries + 1) = count * queries + count := Nat.mul_succ _ _
        rw [List.filter_append, List.length_append, expand]
        exact (Nat.add_comm _ _).trans_le (Nat.add_le_add (counted kind count few)
          ((sub.filter kind).length_le.trans (few point)))

end Fixed

end NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate
