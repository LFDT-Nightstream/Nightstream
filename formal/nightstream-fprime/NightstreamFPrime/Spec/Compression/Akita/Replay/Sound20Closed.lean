import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Game
import NightstreamFPrime.Spec.Compression.Akita.Replay.SimulateDeps

/-! The recording prover of Akita (ePrint 2026/1983, revision of 2026-11-23, Lemma 10.47, the
paragraphs "The prover P" and "Simulation") on a closed run `Digest.Run.closed deps run`
(`Replay/Sound20Game.lean`, `Replay/SimulateDeps.lean`). No Akita type is in this file.

- `simulateDeps_closed`: on a closed run, each query to `H_FS` fixes the address of its own key
  only, because each key that it names has a query directly before it. The simulation is then
  the simulation with no named key. `oracleDeps_closed`, `simulateDeps_closed_logged`.
- `closed_before`: in the queries of a closed run, each key that a query names has a query
  before this query.
- `simulateDeps_closed_entry`: the address of a key in the final cache is `resolve` of the table
  that the algorithm sees and of the digest queries before the first query of the key.
  `simulateDeps_closed_asked`: each key with a query is in the final cache.
- `compDeps_closed_run`: the algorithm with the oracle `table` and no named key is, on a closed
  run, the simulation with the named keys.
- `closed_guard`, `simulateDeps_closed_entry_guard`: the hypotheses hold for an address that
  does not read the table of `H_FS`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.RandomOracle

attribute [local instance low] Classical.propDecidable

variable {Fs Y Msg Dig Addr Reply Output : Type}

variable (resolve : (Fs → Y) → List (Msg × Dig) → Fs → Option Addr) (deps : Fs → List Fs)
  (φ : Fs → Reply × Y → Y)

/-- The cache has each key that one of its keys names. -/
def Named (cache : Cache Fs Addr) : Prop :=
  ∀ point, (cache point).isSome → ∀ named ∈ deps point, (cache named).isSome

theorem named_empty : Named deps (fun _ => none : Cache Fs Addr) :=
  fun _ known => absurd known Bool.false_ne_true

theorem ready_none (cache : Cache Fs Addr) (pending : List Fs) :
    Ready (fun _ : Fs => []) cache pending :=
  fun _ _ _ _ _ member => absurd member List.not_mem_nil

/-- The record of the digest queries of `queries` with the answers of `hash`, the newest
first. -/
def recorded (hash : Msg → Dig) (queries : List (Fs ⊕ Msg)) : List (Msg × Dig) :=
  (queries.filterMap fun query =>
    query.getRight?.map fun message => (message, hash message)).reverse

theorem recorded_inl (hash : Msg → Dig) (point : Fs) (queries : List (Fs ⊕ Msg)) :
    recorded hash (Sum.inl point :: queries) = recorded hash queries := by
  unfold recorded
  rw [List.filterMap_cons_none rfl]

theorem recorded_inr (hash : Msg → Dig) (message : Msg) (queries : List (Fs ⊕ Msg)) :
    recorded hash (Sum.inr message :: queries) =
      recorded hash queries ++ [(message, hash message)] := by
  unfold recorded
  exact (congrArg List.reverse (List.filterMap_cons_some
    (f := fun query : Fs ⊕ Msg => query.getRight?.map fun message => (message, hash message))
    (a := Sum.inr message) (l := queries) rfl)).trans List.reverse_cons

theorem split_append {α : Type} {x : α} {tail post : List α} (front : List α) :
    ∀ pre : List α, front ++ tail = pre ++ x :: post →
      (∃ mid, front = pre ++ x :: mid) ∨
        ∃ rest, pre = front ++ rest ∧ tail = rest ++ x :: post := by
  induction front with
  | nil => exact fun pre split => Or.inr ⟨pre, rfl, split⟩
  | cons a front induction =>
      intro pre split
      cases pre with
      | nil =>
          have split' : a :: (front ++ tail) = x :: post := split
          exact Or.inl ⟨front, congrArg (· :: front) (List.head_eq_of_cons_eq split')⟩
      | cons b pre =>
          have split' : a :: (front ++ tail) = b :: (pre ++ x :: post) := split
          have head := List.head_eq_of_cons_eq split'
          rcases induction pre (List.tail_eq_of_cons_eq split') with
            ⟨mid, found⟩ | ⟨rest, whole, tailEq⟩
          · exact Or.inl ⟨mid, congrArg₂ List.cons head found⟩
          · exact Or.inr ⟨rest, congrArg₂ List.cons head.symm whole, tailEq⟩

/-- In the queries of the closed algorithm, each key that a query to `H_FS` names has a query
before this query. -/
theorem closed_before (closed : DepsClosed deps) (fs : Fs → Y) (hash : Msg → Dig)
    (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (pre post : List (Fs ⊕ Msg)) (point : Fs),
      (run.closed deps).trace fs hash = pre ++ Sum.inl point :: post →
      ∀ key ∈ deps point, Sum.inl key ∈ pre := by
  induction run with
  | done output =>
      intro pre post point split
      have empty : ([] : List (Fs ⊕ Msg)) = pre ++ Sum.inl point :: post := split
      exact absurd empty.symm (List.append_ne_nil_of_right_ne_nil pre (List.cons_ne_nil _ _))
  | digest message next induction =>
      intro pre post point split key member
      cases pre with
      | nil =>
          have split' : Sum.inr message :: ((next (hash message)).closed deps).trace fs hash =
              Sum.inl point :: post := split
          cases List.head_eq_of_cons_eq split'
      | cons first pre =>
          have split' : Sum.inr message :: ((next (hash message)).closed deps).trace fs hash =
              first :: (pre ++ Sum.inl point :: post) := split
          exact List.mem_cons_of_mem _
            (induction (hash message) pre post point (List.tail_eq_of_cons_eq split') key member)
  | challenge asked next induction =>
      intro pre post point split key member
      rw [Digest.Run.trace_closed_challenge] at split
      rcases split_append _ pre split with ⟨mid, found⟩ | ⟨rest, whole, tailEq⟩
      · obtain ⟨before, tail, parts, beforeMap, tailMap⟩ := List.map_eq_append_iff.mp found
        obtain ⟨dep, after, tailParts, depEq, -⟩ := List.map_eq_cons_iff.mp tailMap
        rw [tailParts, Sum.inl.inj depEq] at parts
        rw [← beforeMap]
        exact List.mem_map_of_mem (closed asked before point after parts key member)
      · cases rest with
        | nil =>
            have tail' : Sum.inl asked :: ((next (fs asked)).closed deps).trace fs hash =
                Sum.inl point :: post := tailEq
            have same : asked = point := Sum.inl.inj (List.head_eq_of_cons_eq tail')
            rw [← same] at member
            rw [whole, List.append_nil]
            exact List.mem_map_of_mem member
        | cons first rest =>
            have tail' : Sum.inl asked :: ((next (fs asked)).closed deps).trace fs hash =
                first :: (rest ++ Sum.inl point :: post) := tailEq
            rw [whole]
            exact List.mem_append_right _ (List.mem_cons_of_mem _
              (induction (fs asked) rest post point (List.tail_eq_of_cons_eq tail') key member))

section Settle

variable [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (record : List (Msg × Dig))

theorem settleAll_cached (pending : List Fs) :
    ∀ cache : Cache Fs Addr, (∀ key ∈ pending, (cache key).isSome) →
      settleAll resolve φ table aux record cache pending = cache := by
  induction pending with
  | nil => exact fun _ _ => rfl
  | cons key pending induction =>
      intro cache known
      obtain ⟨slot, found⟩ := Option.isSome_iff_exists.mp (known key List.mem_cons_self)
      rw [settleAll_cons, settle_cached resolve φ table aux record found]
      exact induction cache fun other member => known other (List.mem_cons_of_mem key member)

/-- A chain with each key but the last in the cache settles only the last key. -/
theorem settleAll_last {cache : Cache Fs Addr} {pending : List Fs} (point : Fs)
    (known : ∀ key ∈ pending, (cache key).isSome) :
    settleAll resolve φ table aux record cache (pending ++ [point]) =
      settle resolve φ table aux record cache point := by
  have parts : settleAll resolve φ table aux record cache (pending ++ [point]) =
      settleAll resolve φ table aux record
        (settleAll resolve φ table aux record cache pending) [point] :=
    List.foldl_append
  rw [parts, settleAll_cached resolve φ table aux record pending cache known, settleAll_cons]
  exact rfl

theorem settle_named {cache : Cache Fs Addr} {point : Fs} (named : Named deps cache)
    (known : ∀ key ∈ deps point, (cache key).isSome) :
    Named deps (settle resolve φ table aux record cache point) := by
  intro other cached key member
  by_cases same : other = point
  · rw [same] at member
    exact settle_isSome resolve φ table aux record point (known key member)
  · rw [settle_of_ne resolve φ table aux record cache same] at cached
    exact settle_isSome resolve φ table aux record point (named other cached key member)

theorem settleAll_named (pending : List Fs) :
    ∀ cache : Cache Fs Addr, Named deps cache → Ready deps cache pending →
      Named deps (settleAll resolve φ table aux record cache pending) := by
  induction pending with
  | nil => exact fun _ named _ => named
  | cons key pending induction =>
      intro cache named ready
      exact induction _
        (settle_named resolve deps φ table aux record named (ready_head deps ready))
        (ready_tail deps ready
          (fun other known => settle_isSome resolve φ table aux record key known)
          (settle_self resolve φ table aux record cache key))

end Settle

section Run

variable [DecidableEq Fs] (table : Addr → Reply) (aux : Fs → Y) (hash : Msg → Dig)

/-- The simulation of the queries at `keys`, each with its named keys in the cache or before it
in `keys`, settles the keys of `keys` only. -/
theorem simulateDeps_asking (rest : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig))
    (keys : List Fs) :
    ∀ cache : Cache Fs Addr, Ready deps cache keys →
      simulateDeps resolve deps φ table aux hash (Digest.Run.asking keys rest) cache record =
        simulateDeps resolve deps φ table aux hash rest
          (settleAll resolve φ table aux record cache keys) record := by
  induction keys with
  | nil => exact fun _ _ => rfl
  | cons key keys induction =>
      intro cache ready
      have shape : Digest.Run.asking (key :: keys) rest =
          .challenge key fun _ => Digest.Run.asking keys rest := rfl
      rw [shape, simulateDeps_challenge,
        settleAll_last resolve φ table aux record key (ready_head deps ready), settleAll_cons]
      exact induction _ (ready_tail deps ready
        (fun other known => settle_isSome resolve φ table aux record key known)
        (settle_self resolve φ table aux record cache key))

/-- On a closed run, the recording prover with the named keys of `deps` is the recording prover
with no named key, from each cache that has each key that one of its keys names. -/
theorem simulateDeps_closed (closed : DepsClosed deps) (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)), Named deps cache →
      simulateDeps resolve deps φ table aux hash (run.closed deps) cache record =
        simulateDeps resolve (fun _ => []) φ table aux hash (run.closed deps) cache record := by
  induction run with
  | done output => exact fun _ _ _ => rfl
  | digest message next induction =>
      intro cache record named
      have shape : (Digest.Run.digest message next).closed deps =
          .digest message fun answer => (next answer).closed deps := rfl
      rw [shape, simulateDeps_digest, simulateDeps_digest]
      exact induction (hash message) cache _ named
  | challenge point next induction =>
      intro cache record named
      have shape : (Digest.Run.challenge point next).closed deps =
          Digest.Run.asking (deps point)
            (.challenge point fun answer => (next answer).closed deps) := rfl
      have ready : Ready deps cache (deps point) := fun before key after split dep member =>
        Or.inl (closed point before key after split dep member)
      obtain ⟨grown, grownDef⟩ : ∃ grown,
          grown = settleAll resolve φ table aux record cache (deps point) := ⟨_, rfl⟩
      have known : ∀ key ∈ deps point, (grown key).isSome := fun key member => by
        subst grownDef
        exact settleAll_mem resolve φ table aux record _ cache key member
      have grownNamed : Named deps grown := by
        subst grownDef
        exact settleAll_named resolve deps φ table aux record (deps point) cache named ready
      have single : settleAll resolve φ table aux record grown ([] ++ [point]) =
          settle resolve φ table aux record grown point := rfl
      rw [shape,
        simulateDeps_asking resolve deps φ table aux hash _ record (deps point) cache ready,
        simulateDeps_asking resolve (fun _ => []) φ table aux hash _ record (deps point) cache
          (ready_none cache _),
        ← grownDef, simulateDeps_challenge, simulateDeps_challenge,
        settleAll_last resolve φ table aux record point known, single]
      exact induction _ _ record (settle_named resolve deps φ table aux record grownNamed known)

/-- On a closed run, the table of `H_FS` that the algorithm sees is the table of the recording
prover with no named key. -/
theorem oracleDeps_closed (closed : DepsClosed deps) (run : Digest.Run Fs Y Msg Dig Output)
    (record : List (Msg × Dig)) :
    oracleDeps resolve deps φ table aux hash (run.closed deps) record =
      oracleDeps resolve (fun _ => []) φ table aux hash (run.closed deps) record := by
  unfold oracleDeps
  rw [simulateDeps_closed resolve deps φ table aux hash closed run _ record (named_empty deps)]

/-- The simulation of `run.logged` is a function of the simulation of `run`. -/
theorem simulateDeps_logged_eq (run : Digest.Run Fs Y Msg Dig Output) (cache : Cache Fs Addr)
    (record : List (Msg × Dig)) :
    simulateDeps resolve deps φ table aux hash run.logged cache record =
      ((run.eval (reads φ (simulateDeps resolve deps φ table aux hash run cache record).2.1 table
          aux) hash,
        run.answered (reads φ (simulateDeps resolve deps φ table aux hash run cache record).2.1
          table aux) hash),
        (simulateDeps resolve deps φ table aux hash run cache record).2) := by
  have value := simulateDeps_eval resolve deps φ table aux hash run.logged cache record
  rw [(simulateDeps_logged resolve deps φ table aux hash run cache record).2,
    Digest.Run.eval_logged] at value
  exact Prod.ext value (simulateDeps_logged resolve deps φ table aux hash run cache record).2

/-- `simulateDeps_closed` for the closed run with its answered queries. -/
theorem simulateDeps_closed_logged (closed : DepsClosed deps)
    (run : Digest.Run Fs Y Msg Dig Output) (cache : Cache Fs Addr) (record : List (Msg × Dig))
    (named : Named deps cache) :
    simulateDeps resolve deps φ table aux hash (run.closed deps).logged cache record =
      simulateDeps resolve (fun _ => []) φ table aux hash (run.closed deps).logged cache
        record := by
  rw [simulateDeps_logged_eq resolve deps, simulateDeps_logged_eq resolve (fun _ => []),
    simulateDeps_closed resolve deps φ table aux hash closed run cache record named]

/-- The answer at the key of a query is the answer of the cache after the query. -/
theorem simulateDeps_reads_asked (point : Fs) (next : Y → Digest.Run Fs Y Msg Dig Output)
    (cache : Cache Fs Addr) (record : List (Msg × Dig)) :
    reads φ (simulateDeps resolve deps φ table aux hash (.challenge point next) cache record).2.1
        table aux point =
      reads φ (settleAll resolve φ table aux record cache (deps point ++ [point])) table aux
        point := by
  obtain ⟨slot, found⟩ := Option.isSome_iff_exists.mp
    (settleAll_mem resolve φ table aux record (deps point ++ [point]) cache point (by simp))
  rw [simulateDeps_challenge]
  exact reads_of_eq φ ((simulateDeps_cache resolve deps φ table aux hash _ _ record point slot
    found).trans found.symm) table aux

/-- Each key with a query to `H_FS`, in the run with the table that the final cache gives, is
in the final cache. -/
theorem simulateDeps_asked (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)) (point : Fs),
      Sum.inl point ∈ run.trace
        (reads φ (simulateDeps resolve deps φ table aux hash run cache record).2.1 table aux)
        hash →
      ((simulateDeps resolve deps φ table aux hash run cache record).2.1 point).isSome := by
  induction run with
  | done output => exact fun _ _ _ member => absurd member List.not_mem_nil
  | digest message next induction =>
      intro cache record point member
      rw [simulateDeps_digest] at member ⊢
      obtain ⟨final, finalDef⟩ : ∃ final, final = (simulateDeps resolve deps φ table aux hash
          (next (hash message)) cache ((message, hash message) :: record)).2.1 := ⟨_, rfl⟩
      rw [← finalDef] at member
      have member' : Sum.inl point ∈ (Sum.inr message ::
          (next (hash message)).trace (reads φ final table aux) hash : List (Fs ⊕ Msg)) := member
      rcases List.mem_cons.mp member' with bad | later
      · cases bad
      · rw [finalDef] at later
        exact induction (hash message) cache _ point later
  | challenge asked next induction =>
      intro cache record point member
      obtain ⟨grown, grownDef⟩ : ∃ grown,
          grown = settleAll resolve φ table aux record cache (deps asked ++ [asked]) := ⟨_, rfl⟩
      have value : reads φ (simulateDeps resolve deps φ table aux hash (.challenge asked next)
          cache record).2.1 table aux asked = reads φ grown table aux asked := by
        subst grownDef
        exact simulateDeps_reads_asked resolve deps φ table aux hash asked next cache record
      have here : (grown asked).isSome := by
        subst grownDef
        exact settleAll_mem resolve φ table aux record _ cache asked (by simp)
      rw [simulateDeps_challenge, ← grownDef] at member value ⊢
      obtain ⟨final, finalDef⟩ : ∃ final, final = (simulateDeps resolve deps φ table aux hash
          (next (reads φ grown table aux asked)) grown record).2.1 := ⟨_, rfl⟩
      rw [← finalDef] at member value ⊢
      have member' : Sum.inl point ∈ (Sum.inl asked ::
          (next (reads φ final table aux asked)).trace (reads φ final table aux) hash :
            List (Fs ⊕ Msg)) := member
      rw [value] at member'
      rcases List.mem_cons.mp member' with same | later
      · obtain ⟨slot, found⟩ := Option.isSome_iff_exists.mp here
        rw [Sum.inl.inj same, finalDef,
          simulateDeps_cache resolve deps φ table aux hash _ grown record asked slot found]
        exact rfl
      · rw [finalDef] at later ⊢
        exact induction _ grown record point later

/-- In the recording prover with no named key, each entry of the final cache is an entry of the
first cache, or the key has a first query in the run with the table that the final cache gives,
and the entry is `resolve` of the digest queries before this query and of a table with the
answers of the final cache at each key of the first cache and at each key with a query before
this query. -/
theorem simulateDeps_entry (run : Digest.Run Fs Y Msg Dig Output) :
    ∀ (cache : Cache Fs Addr) (record : List (Msg × Dig)) (point : Fs) (slot : Option Addr),
      (simulateDeps resolve (fun _ => []) φ table aux hash run cache record).2.1 point =
        some slot →
      cache point = some slot ∨ ∃ pre post fs,
        run.trace (reads φ (simulateDeps resolve (fun _ => []) φ table aux hash run cache
          record).2.1 table aux) hash = pre ++ Sum.inl point :: post ∧
        Sum.inl point ∉ pre ∧ cache point = none ∧
        (∀ key, (cache key).isSome ∨ Sum.inl key ∈ pre →
          fs key = reads φ (simulateDeps resolve (fun _ => []) φ table aux hash run cache
            record).2.1 table aux key) ∧
        slot = resolve fs (recorded hash pre ++ record) point := by
  induction run with
  | done output => exact fun _ _ _ _ found => Or.inl found
  | digest message next induction =>
      intro cache record point slot found
      rw [simulateDeps_digest] at found ⊢
      rcases induction (hash message) cache _ point slot found with
        old | ⟨pre, post, fs, split, first, fresh, agrees, value⟩
      · exact Or.inl old
      · refine Or.inr ⟨Sum.inr message :: pre, post, fs, congrArg (Sum.inr message :: ·) split,
          ?_, fresh, ?_, ?_⟩
        · intro member
          rcases List.mem_cons.mp member with bad | inside
          · cases bad
          · exact first inside
        · intro key known
          refine agrees key (known.imp_right fun member => ?_)
          rcases List.mem_cons.mp member with bad | inside
          · cases bad
          · exact inside
        · rw [recorded_inr, List.append_assoc, List.singleton_append]
          exact value
  | challenge asked next induction =>
      intro cache record point slot found
      obtain ⟨grown, grownDef⟩ : ∃ grown,
          grown = settle resolve φ table aux record cache asked := ⟨_, rfl⟩
      have step : simulateDeps resolve (fun _ => []) φ table aux hash (.challenge asked next)
          cache record = simulateDeps resolve (fun _ => []) φ table aux hash
            (next (reads φ grown table aux asked)) grown record := by
        subst grownDef
        rfl
      have value : reads φ (simulateDeps resolve (fun _ => []) φ table aux hash
          (.challenge asked next) cache record).2.1 table aux asked =
            reads φ grown table aux asked := by
        subst grownDef
        exact simulateDeps_reads_asked resolve (fun _ => []) φ table aux hash asked next cache
          record
      have keeps : ∀ key, (cache key).isSome → (grown key).isSome := fun key known => by
        subst grownDef
        exact settle_isSome resolve φ table aux record asked known
      have here : (grown asked).isSome := by
        subst grownDef
        exact settle_self resolve φ table aux record cache asked
      rw [step] at found value ⊢
      have inner := induction (reads φ grown table aux asked) grown record point slot found
      obtain ⟨final, finalDef⟩ : ∃ final, final = (simulateDeps resolve (fun _ => []) φ table aux
          hash (next (reads φ grown table aux asked)) grown record).2.1 := ⟨_, rfl⟩
      rw [← finalDef] at inner value ⊢
      have trace : (Digest.Run.challenge asked next).trace (reads φ final table aux) hash =
          Sum.inl asked ::
            (next (reads φ grown table aux asked)).trace (reads φ final table aux) hash :=
        congrArg (fun answer => (Sum.inl asked : Fs ⊕ Msg) ::
          (next answer).trace (reads φ final table aux) hash) value
      rcases inner with old | ⟨pre, post, fs, split, first, fresh, agrees, resolved⟩
      · cases cached : cache point with
        | some prior =>
            have kept : grown point = some prior := by
              subst grownDef
              exact settle_keeps resolve φ table aux record asked cached
            exact Or.inl (kept.symm.trans old)
        | none =>
            have equal : point = asked := by
              by_contra apart
              have moved : grown point = cache point := by
                subst grownDef
                exact settle_of_ne resolve φ table aux record cache apart
              rw [old, cached] at moved
              cases moved
            have freshKey : cache asked = none := by
              rw [← equal]
              exact cached
            have atKey : grown asked =
                some (resolve (reads φ cache table aux) record asked) := by
              subst grownDef
              rw [settle_fresh resolve φ table aux record freshKey, Function.update_self]
            rw [equal, atKey] at old
            refine Or.inr ⟨[],
              (next (reads φ grown table aux asked)).trace (reads φ final table aux) hash,
              reads φ cache table aux, ?_, List.not_mem_nil, rfl, ?_, ?_⟩
            · show (Digest.Run.challenge asked next).trace (reads φ final table aux) hash =
                Sum.inl point :: _
              rw [trace, equal]
            · intro key known
              obtain ⟨prior, foundKey⟩ :=
                Option.isSome_iff_exists.mp (known.resolve_right List.not_mem_nil)
              have kept : final key = some prior := by
                subst finalDef grownDef
                exact simulateDeps_cache resolve _ φ table aux hash _ _ record key prior
                  (settle_keeps resolve φ table aux record asked foundKey)
              exact reads_of_eq φ (foundKey.trans kept.symm) table aux
            · rw [equal]
              exact (Option.some.inj old).symm
      · have apart : point ≠ asked := fun equal => by
          rw [← equal, fresh] at here
          exact Bool.false_ne_true here
        have older : cache point = none := by
          cases cached : cache point with
          | none => rfl
          | some prior =>
              have kept := keeps point (by rw [cached]; exact rfl)
              rw [fresh] at kept
              exact absurd kept Bool.false_ne_true
        refine Or.inr ⟨Sum.inl asked :: pre, post, fs,
          trace.trans (congrArg (Sum.inl asked :: ·) split), ?_, older, ?_, ?_⟩
        · intro member
          rcases List.mem_cons.mp member with head | inside
          · exact apart (Sum.inl.inj head)
          · exact first inside
        · intro key known
          refine agrees key ?_
          rcases known with cachedKey | member
          · exact Or.inl (keeps key cachedKey)
          · rcases List.mem_cons.mp member with head | inside
            · rw [Sum.inl.inj head]
              exact Or.inl here
            · exact Or.inr inside
        · rw [recorded_inl]
          exact resolved

/-- The time of an address on a closed run, from the empty cache: the entry of a key in the
final cache is `resolve` of the table that the algorithm sees and of the digest queries before
the first query of the key, the newest first. -/
theorem simulateDeps_closed_entry (local_ : DepsLocal resolve deps) (closed : DepsClosed deps)
    (run : Digest.Run Fs Y Msg Dig Output) (point : Fs) (slot : Option Addr)
    (found : (simulateDeps resolve deps φ table aux hash (run.closed deps) (fun _ => none)
      []).2.1 point = some slot) :
    ∃ pre post,
      (run.closed deps).trace (oracleDeps resolve deps φ table aux hash (run.closed deps) [])
          hash = pre ++ Sum.inl point :: post ∧
        Sum.inl point ∉ pre ∧
        slot = resolve (oracleDeps resolve deps φ table aux hash (run.closed deps) [])
          ((pre.filterMap fun query => query.getRight?.map fun message =>
            (message, hash message)).reverse) point := by
  have same := simulateDeps_closed resolve deps φ table aux hash closed run _ []
    (named_empty deps)
  unfold oracleDeps
  rw [same] at found ⊢
  rcases simulateDeps_entry resolve φ table aux hash (run.closed deps) _ [] point slot found with
    old | ⟨pre, post, fs, split, first, -, agrees, value⟩
  · cases old
  · refine ⟨pre, post, split, first, ?_⟩
    rw [value, List.append_nil]
    exact local_ fs _ (recorded hash pre) point fun key member =>
      agrees key (Or.inr (closed_before deps closed _ hash run pre post point split key member))

/-- Each key with a query to `H_FS` in the run with the table that the algorithm sees is in the
final cache. -/
theorem oracleDeps_asked (run : Digest.Run Fs Y Msg Dig Output) (record : List (Msg × Dig))
    (point : Fs)
    (asked : Sum.inl point ∈ run.trace (oracleDeps resolve deps φ table aux hash run record)
      hash) :
    ((simulateDeps resolve deps φ table aux hash run (fun _ => none) record).2.1 point).isSome :=
  simulateDeps_asked resolve deps φ table aux hash run _ record point asked

/-- `oracleDeps_asked` for a closed run and the empty record. -/
theorem simulateDeps_closed_asked (run : Digest.Run Fs Y Msg Dig Output) (point : Fs)
    (asked : Sum.inl point ∈ (run.closed deps).trace
      (oracleDeps resolve deps φ table aux hash (run.closed deps) []) hash) :
    ((simulateDeps resolve deps φ table aux hash (run.closed deps) (fun _ => none) []).2.1
      point).isSome :=
  oracleDeps_asked resolve deps φ table aux hash (run.closed deps) [] point asked

end Run

section Count

variable [DecidableEq Fs]

/-- The run of the recording prover with no named key, with the oracle `table`, on a closed run
with its answered queries is the simulation with the named keys of `deps`. -/
theorem compDeps_closed_run (closed : DepsClosed deps) (table : Addr → Reply) (aux : Fs → Y)
    (hash : Msg → Dig) (run : Digest.Run Fs Y Msg Dig Output) (cache : Cache Fs Addr)
    (record : List (Msg × Dig)) (named : Named deps cache) :
    (compDeps resolve (fun _ => []) φ aux hash (run.closed deps).logged cache record
      (reads φ cache table aux)).run table =
      simulateDeps resolve deps φ table aux hash (run.closed deps).logged cache record := by
  rw [compDeps_run,
    simulateDeps_closed_logged resolve deps φ table aux hash closed run cache record named]

/-- `compDeps_closed_run` from the empty cache and the empty record. -/
theorem compDeps_closed_run_empty (closed : DepsClosed deps) (table : Addr → Reply)
    (aux : Fs → Y) (hash : Msg → Dig) (run : Digest.Run Fs Y Msg Dig Output) :
    (compDeps resolve (fun _ => []) φ aux hash (run.closed deps).logged (fun _ => none) []
      aux).run table =
      simulateDeps resolve deps φ table aux hash (run.closed deps).logged (fun _ => none) [] :=
  compDeps_closed_run resolve deps φ closed table aux hash run (fun _ => none) []
    (named_empty deps)

end Count

section Guard

/-- The hypotheses of this file hold for an address that does not read the table of `H_FS`, no
named key and the empty cache: the guard against a vacuous statement. -/
theorem closed_guard (address : List (Msg × Dig) → Fs → Option Addr) :
    DepsLocal (fun (_ : Fs → Y) => address) (fun _ => []) ∧
      DepsClosed (fun _ : Fs => ([] : List Fs)) ∧
      Named (fun _ : Fs => []) (fun _ => none : Cache Fs Addr) :=
  ⟨fun _ _ _ _ _ => rfl, fun _ _ _ _ split => absurd split (by simp), named_empty _⟩

/-- `simulateDeps_closed_entry` for an address that does not read the table of `H_FS` and no
named key, with no hypothesis on `resolve` and `deps`. -/
theorem simulateDeps_closed_entry_guard [DecidableEq Fs]
    (address : List (Msg × Dig) → Fs → Option Addr) (table : Addr → Reply) (aux : Fs → Y)
    (hash : Msg → Dig) (run : Digest.Run Fs Y Msg Dig Output) (point : Fs) (slot : Option Addr)
    (found : (simulateDeps (fun _ => address) (fun _ => []) φ table aux hash
      (run.closed fun _ => []) (fun _ => none) []).2.1 point = some slot) :
    ∃ pre post,
      (run.closed fun _ => []).trace (oracleDeps (fun _ => address) (fun _ => []) φ table aux
          hash (run.closed fun _ => []) []) hash = pre ++ Sum.inl point :: post ∧
        Sum.inl point ∉ pre ∧
        slot = address ((pre.filterMap fun query => query.getRight?.map fun message =>
          (message, hash message)).reverse) point :=
  simulateDeps_closed_entry (fun _ => address) (fun _ => []) φ table aux hash
    (closed_guard (Y := Y) address).1 (closed_guard (Y := Y) address).2.1 run point slot found

end Guard

end NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate
