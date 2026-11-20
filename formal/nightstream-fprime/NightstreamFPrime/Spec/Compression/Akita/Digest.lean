import NightstreamFPrime.Spec.RandomOracle

/-! The bad events of the hashed transcript entries of Akita (ePrint 2026/1983, revision of
2026-10-15, Lemma 10.47 with (212) and (213), for one digest type): the counting part of the
proof, and the statement of its paragraph "Acceptance" for the recorded queries.

- `Run` is an algorithm with the two oracles `H_FS` and `H_o`. `Run.trace` has its queries in
  their order, and `Run.Bound` bounds the numbers `Q` and `Q_o` of its queries.
- `Collided` is the first bad event: two different digest queries have one answer. `Guessed` is
  the second bad event: the answer of a first digest query is a string in a digest position of
  this query or of an earlier query to one of the two oracles.
- `bad_le` bounds the probability of the two events by `ε_hash = (C(Q_o, 2) + Q_o·V_o)/2^ℓ` of
  (212), for each table of `H_FS` and a uniform `H_o`. `terminal_le` is (213).
- `preimage` states that, with no bad event, a digest with a recorded preimage has one recorded
  preimage, and that the first query of this preimage is before each query with the digest in a
  digest position.

The protocol `Π_H`, the expansion of a proof and the prover `P` of the lemma are not in this
file. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Digest

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle

attribute [local instance low] Classical.propDecidable

/-- An algorithm with the oracle `H_FS`, from `Fs` to `Challenge`, and the digest oracle `H_o`,
from `Msg` to `Digest`. -/
inductive Run (Fs Challenge Msg Digest Output : Type) : Type where
  | done (output : Output)
  | challenge (point : Fs) (next : Challenge → Run Fs Challenge Msg Digest Output)
  | digest (message : Msg) (next : Digest → Run Fs Challenge Msg Digest Output)

variable {Fs Challenge Msg Digest Output : Type}

namespace Run

/-- `run` has at most `Q` queries to `H_FS` and at most `Q_o` queries to `H_o`. -/
inductive Bound : Run Fs Challenge Msg Digest Output → ℕ → ℕ → Prop
  | done (output : Output) (queries digests : ℕ) : Bound (done output) queries digests
  | challenge (point : Fs) (next : Challenge → Run Fs Challenge Msg Digest Output)
      (queries digests : ℕ) (each : ∀ answer, Bound (next answer) queries digests) :
      Bound (challenge point next) (queries + 1) digests
  | digest (message : Msg) (next : Digest → Run Fs Challenge Msg Digest Output)
      (queries digests : ℕ) (each : ∀ answer, Bound (next answer) queries digests) :
      Bound (digest message next) queries (digests + 1)

variable (fs : Fs → Challenge)

/-- The algorithm with the oracle `H_o` only, for the table `fs` of `H_FS`. -/
def toComp : Run Fs Challenge Msg Digest Output → OracleComp Msg Digest Output
  | done output => .done output
  | challenge point next => (next (fs point)).toComp
  | digest message next => .query message fun answer => (next answer).toComp

/-- The queries of the algorithm to the two oracles, in their order. -/
def trace (hash : Msg → Digest) : Run Fs Challenge Msg Digest Output → List (Fs ⊕ Msg)
  | done _ => []
  | challenge point next => .inl point :: (next (fs point)).trace hash
  | digest message next => .inr message :: (next (hash message)).trace hash

theorem mem_queries_iff (hash : Msg → Digest) (run : Run Fs Challenge Msg Digest Output)
    (message : Msg) :
    message ∈ (run.toComp fs).queries hash ↔ Sum.inr message ∈ run.trace fs hash := by
  induction run with
  | done output => simp [toComp, trace, OracleComp.queries]
  | challenge point next induction => simp [toComp, trace, induction]
  | digest asked next induction => simp [toComp, trace, OracleComp.queries, induction]

theorem Bound.toComp {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) : (run.toComp fs).QueryBound digests := by
  induction bounded with
  | done output queries digests => exact .done _ _
  | challenge point next queries digests each induction => exact induction (fs point)
  | digest message next queries digests each induction => exact .query _ _ _ induction

variable [DecidableEq Msg]

/-- The queries to the first digest query of `target`, with this query. -/
def upTo (hash : Msg → Digest) (target : Msg) :
    Run Fs Challenge Msg Digest Output → List (Fs ⊕ Msg)
  | done _ => []
  | challenge point next => .inl point :: (next (fs point)).upTo hash target
  | digest message next =>
      if message = target then [.inr message]
      else .inr message :: (next (hash message)).upTo hash target

/-- The queries to the first digest query of `target` do not read the answer of `target`. -/
theorem upTo_update (hash : Msg → Digest) (target : Msg) (answer : Digest)
    (run : Run Fs Challenge Msg Digest Output) :
    run.upTo fs (Function.update hash target answer) target = run.upTo fs hash target := by
  induction run with
  | done output => rfl
  | challenge point next induction => simp [upTo, induction]
  | digest message next induction =>
      by_cases same : message = target
      · simp [upTo, same]
      · simp [upTo, same, induction]

theorem upTo_prefix (hash : Msg → Digest) (target : Msg)
    (run : Run Fs Challenge Msg Digest Output) :
    run.upTo fs hash target <+: run.trace fs hash := by
  induction run with
  | done output => exact List.nil_prefix
  | challenge point next induction => exact (List.prefix_cons_inj _).mpr (induction _)
  | digest message next induction =>
      by_cases same : message = target
      · simp [upTo, trace, same]
      · simp only [upTo, trace, same, ite_false]
        exact (List.prefix_cons_inj _).mpr (induction _)

theorem mem_upTo {hash : Msg → Digest} {target : Msg}
    {run : Run Fs Challenge Msg Digest Output}
    (queried : Sum.inr target ∈ run.trace fs hash) : Sum.inr target ∈ run.upTo fs hash target := by
  induction run with
  | done output => simp [trace] at queried
  | challenge point next induction =>
      simp only [trace, List.mem_cons, reduceCtorEq, false_or] at queried
      exact List.mem_cons_of_mem _ (induction _ queried)
  | digest message next induction =>
      by_cases same : message = target
      · simp [upTo, same]
      · simp only [trace, List.mem_cons, Sum.inr.injEq] at queried
        simp only [upTo, same, ite_false]
        exact List.mem_cons_of_mem _
          (induction _ (queried.resolve_left fun equal => same equal.symm))

variable (positions : Fs ⊕ Msg → List Digest)

/-- The first bad event of Akita Lemma 10.47: two different digest queries have one answer. -/
def Collided (hash : Msg → Digest) (run : Run Fs Challenge Msg Digest Output) : Prop :=
  ∃ message, Sum.inr message ∈ run.trace fs hash ∧ ∃ other, Sum.inr other ∈ run.trace fs hash ∧
    message ≠ other ∧ hash message = hash other

/-- The second bad event of Akita Lemma 10.47: the answer of the first digest query of a message
is a string in a digest position of this query or of an earlier query. `positions query` has the
strings in the digest positions of a query. -/
def Guessed (hash : Msg → Digest) (run : Run Fs Challenge Msg Digest Output) : Prop :=
  ∃ message, Sum.inr message ∈ run.trace fs hash ∧
    hash message ∈ (run.upTo fs hash message).flatMap positions

/-- With no bad event, a digest with a recorded preimage `message` has no other recorded
preimage, and the first digest query of `message` is before each query with the digest in a
digest position (Akita Lemma 10.47, the paragraphs "Simulation" and "Acceptance"). For an
accepted proof, the queries of the verifier are in the trace, so `message` is the message that
the verifier recovers for the digest. -/
theorem preimage {hash : Msg → Digest} {run : Run Fs Challenge Msg Digest Output}
    (unique : ¬ Collided fs hash run) (fresh : ¬ Guessed fs positions hash run) {message : Msg}
    (opened : Sum.inr message ∈ run.trace fs hash) :
    (∀ other, Sum.inr other ∈ run.trace fs hash → hash other = hash message → other = message) ∧
      ∃ later, run.trace fs hash = run.upTo fs hash message ++ later ∧
        Sum.inr message ∈ run.upTo fs hash message ∧
        ∀ query ∈ run.upTo fs hash message, hash message ∉ positions query := by
  refine ⟨fun other queried same => ?_, ?_⟩
  · by_contra different
    exact unique ⟨other, queried, message, opened, different, same⟩
  · obtain ⟨later, split⟩ := upTo_prefix fs hash message run
    refine ⟨later, split.symm, mem_upTo fs opened, fun query member hit => fresh ?_⟩
    exact ⟨message, opened, List.mem_flatMap.mpr ⟨query, member, hit⟩⟩

omit [DecidableEq Msg] in
/-- The digest positions of the queries: each query to `H_FS` has at most `width` digest
positions, and a digest input has none. The terminal of Akita Section 8.2 has `width = 2`. -/
theorem positions_le {width : ℕ} (challenge : ∀ point, (positions (.inl point)).length ≤ width)
    (digest : ∀ message, positions (.inr message) = [])
    {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) (hash : Msg → Digest) :
    ((run.trace fs hash).flatMap positions).length ≤ width * queries := by
  induction bounded with
  | done output queries digests => simp [trace]
  | challenge point next queries digests each induction =>
      simp only [trace, List.flatMap_cons, List.length_append]
      rw [Nat.mul_succ, Nat.add_comm]
      exact Nat.add_le_add (induction _) (challenge point)
  | digest message next queries digests each induction =>
      simp only [trace, List.flatMap_cons, digest, List.nil_append]
      exact induction _

section Probability

variable [Fintype Msg] [Fintype Digest] [Nonempty Digest]

omit [DecidableEq Msg] [Fintype Msg] [Nonempty Digest] in
theorem mass_mem_le (targets : List Digest) :
    mass {digest | digest ∈ targets} ≤ targets.length / Fintype.card Digest := by
  unfold mass
  rw [Finset.expect_eq_sum_div_card, Finset.card_univ]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  calc _ = ∑ digest : Digest, if digest ∈ targets.toFinset then (1 : ℝ) else 0 :=
        Finset.sum_congr rfl fun digest _ => by
          by_cases member : digest ∈ targets <;> simp [member]
    _ ≤ targets.length := by
        rw [Finset.sum_ite_mem, Finset.univ_inter, Finset.sum_const, nsmul_eq_mul, mul_one]
        exact_mod_cast List.toFinset_card_le _

/-- The first bad event has probability at most `C(Q_o, 2)/2^ℓ` (Akita Lemma 10.47). -/
theorem collided_le {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) :
    𝔼 hash : Msg → Digest, (if Collided fs hash run then (1 : ℝ) else 0) ≤
      (digests.choose 2 : ℝ) / Fintype.card Digest := by
  refine (Finset.expect_le_expect fun hash _ => ?_).trans (collision_le (bounded.toComp fs))
  by_cases collided : Collided fs hash run
  · obtain ⟨message, queried, other, queriedOther, different, same⟩ := collided
    have collision : Collision ((run.toComp fs).queries hash) hash :=
      ⟨message, (mem_queries_iff fs hash run message).mpr queried, other,
        (mem_queries_iff fs hash run other).mpr queriedOther, different, same⟩
    rw [ite_eq_left ⟨message, queried, other, queriedOther, different, same⟩,
      ite_eq_left collision]
  · rw [ite_eq_right collided]
    split <;> norm_num

/-- The second bad event has probability at most `Q_o·V_o/2^ℓ` (Akita Lemma 10.47). `few`: the
queries have at most `V_o` digest positions. -/
theorem guessed_le {run : Run Fs Challenge Msg Digest Output} {queries digests positionCount : ℕ}
    (bounded : run.Bound queries digests)
    (few : ∀ hash, ((run.trace fs hash).flatMap positions).length ≤ positionCount) :
    𝔼 hash : Msg → Digest, (if Guessed fs positions hash run then (1 : ℝ) else 0) ≤
      digests * ((positionCount : ℝ) / Fintype.card Digest) := by
  have escape := escape_le
    (fun message hash => {digest | digest ∈ (run.upTo fs hash message).flatMap positions})
    (fun message hash answer => by simp only [upTo_update])
    ((positionCount : ℝ) / Fintype.card Digest)
    (fun message hash => (mass_mem_le _).trans (div_le_div_of_nonneg_right (by
      obtain ⟨later, split⟩ := upTo_prefix fs hash message run
      have total := few hash
      rw [← split, List.flatMap_append, List.length_append] at total
      exact_mod_cast Nat.le_of_add_right_le total) (Nat.cast_nonneg _)))
    (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) (bounded.toComp fs)
  refine (Finset.expect_le_expect fun hash _ => ?_).trans escape
  by_cases guessed : Guessed fs positions hash run
  · obtain ⟨message, queried, hit⟩ := guessed
    have escaped : ∃ point ∈ (run.toComp fs).queries hash,
        hash point ∈ {digest | digest ∈ (run.upTo fs hash point).flatMap positions} :=
      ⟨message, (mem_queries_iff fs hash run message).mpr queried, hit⟩
    rw [ite_eq_left ⟨message, queried, hit⟩, ite_eq_left escaped]
  · rw [ite_eq_right guessed]
    split <;> norm_num

/-- The bound `ε_hash` of Akita (212) for one digest type: for each table `fs` of `H_FS` and each
algorithm with at most `Q_o` queries to a uniform `H_o`, whose queries have at most `V_o` digest
positions, the probability of a bad event is at most `(C(Q_o, 2) + Q_o·V_o)/2^ℓ`. The digests are
`2^ℓ`. The queries of the verifier on the output count as queries of the algorithm. -/
theorem bad_le {run : Run Fs Challenge Msg Digest Output} {queries digests positionCount : ℕ}
    (bounded : run.Bound queries digests)
    (few : ∀ hash, ((run.trace fs hash).flatMap positions).length ≤ positionCount) :
    𝔼 hash : Msg → Digest,
        (if Collided fs hash run ∨ Guessed fs positions hash run then (1 : ℝ) else 0) ≤
      ((digests.choose 2 + digests * positionCount : ℕ) : ℝ) / Fintype.card Digest := by
  have split : ∀ hash : Msg → Digest,
      (if Collided fs hash run ∨ Guessed fs positions hash run then (1 : ℝ) else 0) ≤
        (if Collided fs hash run then (1 : ℝ) else 0) +
          if Guessed fs positions hash run then (1 : ℝ) else 0 := by
    intro hash
    by_cases collided : Collided fs hash run
    · rw [ite_eq_left (Or.inl collided), ite_eq_left collided]
      split <;> norm_num
    · by_cases guessed : Guessed fs positions hash run
      · rw [ite_eq_left (Or.inr guessed), ite_eq_right collided, ite_eq_left guessed, zero_add]
      · rw [ite_eq_right (not_or.mpr ⟨collided, guessed⟩), ite_eq_right collided,
          ite_eq_right guessed, zero_add]
  refine (Finset.expect_le_expect fun hash _ => split hash).trans ?_
  rw [Finset.expect_add_distrib]
  refine (add_le_add (collided_le fs bounded) (guessed_le fs positions bounded few)).trans
    (le_of_eq ?_)
  push_cast
  ring

/-- The bound `ε_dig` of Akita (213): when each query to `H_FS` has at most two digest positions
and a digest input has none, the probability of a bad event is at most
`(C(Q_dig, 2) + 2·Q·Q_dig)/2^ℓ` for at most `Q` queries to `H_FS` and `Q_dig` digest queries. -/
theorem terminal_le (challenge : ∀ point, (positions (.inl point)).length ≤ 2)
    (digest : ∀ message, positions (.inr message) = [])
    {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) :
    𝔼 hash : Msg → Digest,
        (if Collided fs hash run ∨ Guessed fs positions hash run then (1 : ℝ) else 0) ≤
      ((digests.choose 2 + 2 * queries * digests : ℕ) : ℝ) / Fintype.card Digest := by
  have bound := bad_le fs positions bounded (positions_le fs positions challenge digest bounded)
  rw [show digests * (2 * queries) = 2 * queries * digests by ring] at bound
  exact bound

end Probability

end Run

/-- `ε_hash` of Akita (212) for one digest type with `ℓ = bits`: `(C(Q_o, 2) + Q_o·V_o)/2^ℓ`. -/
noncomputable def hashError (bits digests positionCount : ℕ) : ℝ :=
  ((digests.choose 2 + digests * positionCount : ℕ) : ℝ) / 2 ^ bits

/-- `ε_hash` is at most the term `(Q_H² + Q_H·V_H)/2^ℓ` of `Schedule.knowledgeError`. -/
theorem hashError_le (bits digests positionCount : ℕ) :
    hashError bits digests positionCount ≤
      ((digests : ℝ) ^ 2 + digests * positionCount) / 2 ^ bits := by
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  have choose : digests.choose 2 ≤ digests ^ 2 := by
    rw [Nat.choose_two_right, sq]
    exact (Nat.div_le_self _ _).trans (Nat.mul_le_mul_left _ (Nat.sub_le _ _))
  exact_mod_cast Nat.add_le_add_right choose _

end NightstreamFPrime.Spec.Compression.Akita.Digest
