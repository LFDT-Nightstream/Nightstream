import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Deps
import NightstreamFPrime.Spec.Compression.Akita.Replay.VerifierRun20

/-! The run "adversary, then verifier" of the experiment of the `key20.aks` claims and its law in
the recording prover (Akita, ePrint 2026/1983, revision of 2026-11-23, Lemma 10.47, the
paragraphs "Simulation" and "Acceptance", with (213); `docs/akita-work/w21/design.md`,
sections 2 and 8; `Replay/SimulateDeps.lean`, `Replay/VerifierRun20.lean`,
`Replay/Sound20Deps.lean`).

- `Digest.Run.bind` with `eval_bind`, `trace_bind` and `Bound.bind`.
  `Digest.Run.guessed_congr_family`: the second bad event is a function of the answered queries
  also when the digest positions of a query read the table of `H_FS`, if they are the same for
  two tables with the same answered queries.
- `Digest.Run.closed deps run` asks `H_FS` at each key of `deps point` before each query `point`
  of `run`: `eval_closed`, `closed_named` and `closed_alike` (the hypothesis `alike` of
  `Simulate.oracleDeps_closed_bad_le` of `Replay/Sound20Count.lean`, for positions that read
  `H_FS` at the named keys).
- `game20 pub adversary`: the adversary gives the proof bytes, then `verifierRun` reads them.
  `game20_eval`, `game20_trace`, `game20_bound` (`Q + 473` sponge queries, `Q_dig + 2` digest
  queries).
- `simulate20_law`: for a uniform table, a uniform `aux` and a uniform digest function, the
  output of a run in the recording prover of `key20.aks` has the law of its output with a
  uniform sponge. `game20_law` is this law for the acceptance of `verifyProof`.
- `oracle20_law`: the table of the sponge that the algorithm sees has the law of a uniform
  table. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Digest.Run

variable {Fs Challenge Msg Digest Output Other : Type}

/-- The algorithm `run`, then the algorithm `f` on its output. -/
def bind : Run Fs Challenge Msg Digest Output → (Output → Run Fs Challenge Msg Digest Other) →
    Run Fs Challenge Msg Digest Other
  | done output, f => f output
  | challenge point next, f => challenge point fun answer => (next answer).bind f
  | digest message next, f => digest message fun answer => (next answer).bind f

theorem eval_bind (fs : Fs → Challenge) (hash : Msg → Digest)
    (f : Output → Run Fs Challenge Msg Digest Other) (run : Run Fs Challenge Msg Digest Output) :
    (run.bind f).eval fs hash = (f (run.eval fs hash)).eval fs hash := by
  induction run with
  | done output => rfl
  | challenge point next induction => exact induction (fs point)
  | digest message next induction => exact induction (hash message)

theorem trace_bind (fs : Fs → Challenge) (hash : Msg → Digest)
    (f : Output → Run Fs Challenge Msg Digest Other) (run : Run Fs Challenge Msg Digest Output) :
    (run.bind f).trace fs hash =
      run.trace fs hash ++ (f (run.eval fs hash)).trace fs hash := by
  induction run with
  | done output => rfl
  | challenge point next induction => exact congrArg (Sum.inl point :: ·) (induction (fs point))
  | digest message next induction =>
      exact congrArg (Sum.inr message :: ·) (induction (hash message))

theorem Bound.bind {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) {f : Output → Run Fs Challenge Msg Digest Other}
    {more moreDigests : ℕ} (each : ∀ output, (f output).Bound more moreDigests) :
    (run.bind f).Bound (queries + more) (digests + moreDigests) := by
  induction bounded with
  | done output queries digests =>
      exact (each output).mono (Nat.le_add_left _ _) (Nat.le_add_left _ _)
  | challenge point next queries digests _ induction =>
      rw [Nat.add_right_comm]
      exact .challenge point _ _ _ induction
  | digest message next queries digests _ induction =>
      rw [Nat.add_right_comm digests]
      exact .digest message _ _ _ induction

theorem flatMap_congr_mem {α β : Type} {f g : α → List β} : ∀ {l : List α},
    (∀ x ∈ l, f x = g x) → l.flatMap f = l.flatMap g
  | [], _ => rfl
  | x :: l, same => by
      rw [List.flatMap_cons, List.flatMap_cons, same x List.mem_cons_self,
        flatMap_congr_mem fun y member => same y (List.mem_cons_of_mem x member)]

/-- The second bad event of Akita Lemma 10.47, with digest positions that read the table of
`H_FS`, is a function of the answered queries, when the positions of each query of the run are
the same for the two tables. -/
theorem guessed_congr_family [DecidableEq Msg]
    (positions : (Fs → Challenge) → Fs ⊕ Msg → List Digest) {fs other : Fs → Challenge}
    {hash : Msg → Digest} {run : Run Fs Challenge Msg Digest Output}
    (same : run.answered fs hash = run.answered other hash)
    (alike : ∀ query ∈ run.trace fs hash, positions fs query = positions other query) :
    Guessed fs (positions fs) hash run ↔ Guessed other (positions other) hash run := by
  have moved : ∀ message, (run.upTo fs hash message).flatMap (positions fs) =
      (run.upTo other hash message).flatMap (positions other) := by
    intro message
    rw [← upTo_congr same message]
    exact flatMap_congr_mem fun query member =>
      alike query ((upTo_prefix fs hash message run).subset member)
  unfold Guessed
  simp only [trace_congr same, moved]

/-- The queries to `H_FS` at each key of `keys`, in order, then `run`. -/
def asking (keys : List Fs) (run : Run Fs Challenge Msg Digest Output) :
    Run Fs Challenge Msg Digest Output :=
  keys.foldr (fun key rest => challenge key fun _ => rest) run

theorem eval_asking (fs : Fs → Challenge) (hash : Msg → Digest) :
    ∀ (keys : List Fs) (run : Run Fs Challenge Msg Digest Output),
      (asking keys run).eval fs hash = run.eval fs hash
  | [], _ => rfl
  | _ :: keys, run => eval_asking fs hash keys run

theorem trace_asking (fs : Fs → Challenge) (hash : Msg → Digest) :
    ∀ (keys : List Fs) (run : Run Fs Challenge Msg Digest Output),
      (asking keys run).trace fs hash = keys.map Sum.inl ++ run.trace fs hash
  | [], _ => rfl
  | key :: keys, run => congrArg (Sum.inl key :: ·) (trace_asking fs hash keys run)

/-- The algorithm that asks `H_FS` at each key of `deps point`, in order, before each query
`point` of `run`. -/
def closed (deps : Fs → List Fs) :
    Run Fs Challenge Msg Digest Output → Run Fs Challenge Msg Digest Output
  | done output => done output
  | challenge point next =>
      asking (deps point) (challenge point fun answer => (next answer).closed deps)
  | digest message next => digest message fun answer => (next answer).closed deps

variable (deps : Fs → List Fs)

theorem eval_closed (fs : Fs → Challenge) (hash : Msg → Digest)
    (run : Run Fs Challenge Msg Digest Output) :
    (run.closed deps).eval fs hash = run.eval fs hash := by
  induction run with
  | done output => rfl
  | challenge point next induction =>
      show (asking (deps point) _).eval fs hash = _
      rw [eval_asking]
      exact induction (fs point)
  | digest message next induction => exact induction (hash message)

theorem trace_closed_challenge (fs : Fs → Challenge) (hash : Msg → Digest) (point : Fs)
    (next : Challenge → Run Fs Challenge Msg Digest Output) :
    ((challenge point next).closed deps).trace fs hash =
      (deps point).map Sum.inl ++
        Sum.inl point :: ((next (fs point)).closed deps).trace fs hash := by
  show (asking (deps point) _).trace fs hash = _
  rw [trace_asking]
  rfl

/-- In the closed algorithm, each key that a query to `H_FS` names is a query. -/
theorem closed_named (named : Replay.Simulate.DepsClosed deps) (fs : Fs → Challenge)
    (hash : Msg → Digest) (run : Run Fs Challenge Msg Digest Output) :
    ∀ point, Sum.inl point ∈ (run.closed deps).trace fs hash →
      ∀ key ∈ deps point, Sum.inl key ∈ (run.closed deps).trace fs hash := by
  induction run with
  | done output => exact fun point member => absurd member List.not_mem_nil
  | challenge asked next induction =>
      intro point member key inside
      rw [trace_closed_challenge] at member ⊢
      rcases List.mem_append.mp member with early | late
      · obtain ⟨dep, listed, same⟩ := List.mem_map.mp early
        rw [Sum.inl.inj same] at listed
        obtain ⟨pre, post, parts⟩ := List.append_of_mem listed
        have before := named asked pre point post parts key inside
        refine List.mem_append_left _ (List.mem_map_of_mem ?_)
        rw [parts]
        exact List.mem_append_left _ before
      · rcases List.mem_cons.mp late with same | later
        · have equal : point = asked := Sum.inl.inj same
          exact List.mem_append_left _ (List.mem_map_of_mem (equal ▸ inside))
        · exact List.mem_append_right _
            (List.mem_cons_of_mem _ (induction (fs asked) point later key inside))
  | digest message next induction =>
      intro point member key inside
      have member' : Sum.inl point ∈
          Sum.inr message :: ((next (hash message)).closed deps).trace fs hash := member
      rcases List.mem_cons.mp member' with same | later
      · cases same
      · exact List.mem_cons_of_mem _ (induction (hash message) point later key inside)

/-- Each answered query to `H_FS` has the answer of the table. -/
theorem answered_inl (fs : Fs → Challenge) (hash : Msg → Digest)
    (run : Run Fs Challenge Msg Digest Output) :
    ∀ key y, Sum.inl (key, y) ∈ run.answered fs hash → y = fs key := by
  induction run with
  | done output => exact fun key y member => absurd member List.not_mem_nil
  | challenge point next induction =>
      intro key y member
      have member' : Sum.inl (key, y) ∈
          (Sum.inl (point, fs point) :: (next (fs point)).answered fs hash :
            List ((Fs × Challenge) ⊕ (Msg × Digest))) := member
      rcases List.mem_cons.mp member' with same | later
      · have pair := Prod.mk.inj (Sum.inl.inj same)
        rw [pair.1, pair.2]
      · exact induction (fs point) key y later
  | digest message next induction =>
      intro key y member
      have member' : Sum.inl (key, y) ∈
          (Sum.inr (message, hash message) :: (next (hash message)).answered fs hash :
            List ((Fs × Challenge) ⊕ (Msg × Digest))) := member
      rcases List.mem_cons.mp member' with same | later
      · cases same
      · exact induction (hash message) key y later

/-- Two tables with the same answered queries have the same answer at each query. -/
theorem answered_same {fs other : Fs → Challenge} {hash : Msg → Digest}
    {run : Run Fs Challenge Msg Digest Output}
    (same : run.answered fs hash = run.answered other hash) {key : Fs}
    (queried : Sum.inl key ∈ run.trace fs hash) : fs key = other key := by
  rw [← map_answered] at queried
  obtain ⟨entry, inside, mapped⟩ := List.mem_map.mp queried
  cases entry with
  | inl pair =>
      obtain ⟨k, y⟩ := pair
      have equal : k = key := Sum.inl.inj mapped
      have value := answered_inl fs hash run k y inside
      rw [same] at inside
      have valueOther := answered_inl other hash run k y inside
      rw [← equal, ← value, valueOther]
  | inr pair =>
      exact absurd (show (Sum.inr pair.1 : Fs ⊕ Msg) = Sum.inl key from mapped) (by simp)

/-- The hypothesis `alike` of `Simulate.oracleDeps_closed_bad_le` for a closed algorithm: digest
positions of a query that read the table of `H_FS` only at the keys that the query names are the
same for two tables with the same answered queries. -/
theorem closed_alike (named : Replay.Simulate.DepsClosed deps)
    (positions : (Fs → Challenge) → Fs ⊕ Msg → List Digest)
    (local_ : ∀ fs other point, (∀ key ∈ deps point, fs key = other key) →
      positions fs (.inl point) = positions other (.inl point))
    (digest : ∀ fs message, positions fs (.inr message) = [])
    (run : Run Fs Challenge Msg Digest Output) (fs other : Fs → Challenge) (hash : Msg → Digest)
    (same : (run.closed deps).answered fs hash = (run.closed deps).answered other hash) :
    ∀ query ∈ (run.closed deps).trace fs hash, positions fs query = positions other query := by
  intro query member
  cases query with
  | inl point =>
      exact local_ fs other point fun key inside =>
        answered_same same (closed_named deps named fs hash run point member key inside)
  | inr message => rw [digest fs message, digest other message]

end NightstreamFPrime.Spec.Compression.Akita.Digest.Run

namespace NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate

open scoped BigOperators

theorem expect_le_const {X : Type} [Fintype X] [Nonempty X] (f : X → ℝ) (limit : ℝ)
    (each : ∀ x, f x ≤ limit) : 𝔼 x, f x ≤ limit :=
  (Finset.expect_le_expect fun x _ => each x).trans_eq
    (Finset.expect_const Finset.univ_nonempty limit)

end NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Address)
open Sponge (split answerOf spongeOf)
open Replayer

attribute [local instance low] Classical.propDecidable

section Game

variable {L B C S D : ℕ}

/-- The run of the experiment: the adversary gives the proof bytes, then the verifier reads them
with the same two oracles. The output has the bytes and the result of the verifier. -/
noncomputable def game20 (pub : Public)
    (adversary : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32)
      ByteArray) :
    Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32)
      (ByteArray × Outcome) :=
  adversary.bind fun proof => (verifierRun pub proof).map fun outcome => (proof, outcome)

variable (pub : Public)
  (adversary : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) ByteArray)
  (fs : Sponge.Base L B C → Sponge.Y S C) (hash : Sponge.Str D → Bytes 32)

/-- The output of the run of the experiment: the bytes of the adversary and the result of
`verifyProof` on them, with the sponge of `fs` and the digest function of `hash`. -/
theorem game20_eval :
    (game20 pub adversary).eval fs hash =
      (adversary.eval fs hash, verifyProof (pub.withOracles fs hash) (adversary.eval fs hash)) := by
  unfold game20
  rw [Digest.Run.eval_bind, Digest.Run.eval_map, verifierRun_eval]

theorem game20_trace :
    (game20 pub adversary).trace fs hash =
      adversary.trace fs hash ++ (verifierRun pub (adversary.eval fs hash)).trace fs hash := by
  unfold game20
  rw [Digest.Run.trace_bind, Digest.Run.trace_map]

variable {pub adversary}

/-- The run of the experiment has at most `Q + 473` sponge queries and `Q_dig + 2` digest
queries, for an adversary with at most `Q` and `Q_dig`. -/
theorem game20_bound {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    {queries digests : ℕ} (bounded : adversary.Bound queries digests) :
    (game20 pub adversary).Bound (queries + 473) (digests + 2) :=
  bounded.bind fun proof => (verifierRun_bound20 pub proof layout).map _

end Game

section Law

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {L B C S D total : ℕ}
  (block : Key20.root.Block → Coord) (block₁ : Fin Key20.level₁.shape.blocks → Coord)
  (block₂ : Fin Key20.level₂.shape.blocks → Coord) (block₃ : Fin Key20.level₃.shape.blocks → Coord)
  (block₄ : Fin Key20.level₄.shape.blocks → Coord) (block₅ : Fin Key20.level₅.shape.blocks → Coord)
  (blockL : Fin Key20.terminal.blocks → Coord) {pub : Public} {Output : Type}

local notation "Reply" => (Option Coord → CodeSlot E)

local notation "Addr" => Address (Key20.Wire E) total

/-- The recording prover of `key20.aks` gives the algorithm a uniform sponge: for a uniform
`table`, a uniform `aux` and a uniform digest function, the output of `run` in the simulation
has the law of its output with a uniform table of the sponge. `pub` is the public input of
`resolveAt`, a script of `layout20` with the work bits of the model. -/
theorem simulate20_law (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (long : 472 ≤ total)
    (run : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) Output)
    (F : (Sponge.Str D → Bytes 32) → Output → ℝ) :
    𝔼 table : Addr → Reply, 𝔼 aux : Sponge.Base L B C → Sponge.Y S C,
        𝔼 hash : Sponge.Str D → Bytes 32,
        F hash (Simulate.simulateDeps (resolveAt total pub) (deps20 S)
          (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux hash run
          (fun _ => none) []).1 =
      𝔼 fs : Sponge.Base L B C → Sponge.Y S C, 𝔼 hash : Sponge.Str D → Bytes 32,
        F hash (run.eval fs hash) :=
  Simulate.simulateDeps_law_hash (resolveAt total pub) (deps20 S)
    (phiAt block block₁ block₂ block₃ block₄ block₅ blockL)
    (back := fun _ fs => backAt pub fs) (resolveAt_local bits total) (deps20_closed S)
    (fun _ fs record _ point address resolved =>
      resolveAt_back erased long fs record point address resolved)
    (card_phiAt block block₁ block₂ block₃ block₄ block₅ blockL) run F

/-- The verifier accepts the bytes of the adversary in the recording prover with the
probability of the run with a uniform sponge and a uniform digest function. -/
theorem game20_law (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (long : 472 ≤ total) (pubV : Public)
    (adversary : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32)
      ByteArray) :
    𝔼 table : Addr → Reply, 𝔼 aux : Sponge.Base L B C → Sponge.Y S C,
        𝔼 hash : Sponge.Str D → Bytes 32,
        (if (Simulate.simulateDeps (resolveAt total pub) (deps20 S)
          (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux hash
          (game20 pubV adversary) (fun _ => none) []).1.2 = .verdict .accept
          then (1 : ℝ) else 0) =
      𝔼 fs : Sponge.Base L B C → Sponge.Y S C, 𝔼 hash : Sponge.Str D → Bytes 32,
        (if verifyProof (pubV.withOracles fs hash) (adversary.eval fs hash) = .verdict .accept
          then (1 : ℝ) else 0) := by
  rw [simulate20_law block block₁ block₂ block₃ block₄ block₅ blockL erased bits long
    (game20 pubV adversary) fun _ output => if output.2 = .verdict .accept then (1 : ℝ) else 0]
  simp only [game20_eval]

/-- The table of the sponge that the algorithm sees in the recording prover of `key20.aks` has
the law of a uniform table, for each value that reads the table through the answered queries of
the run only. -/
theorem oracle20_law (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (long : 472 ≤ total) (hash : Sponge.Str D → Bytes 32)
    (run : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) Output)
    (value : (Sponge.Base L B C → Sponge.Y S C) → ℝ)
    (factors : ∀ fs other, run.answered fs hash = run.answered other hash →
      value fs = value other) :
    𝔼 table : Addr → Reply, 𝔼 aux : Sponge.Base L B C → Sponge.Y S C,
        value (Simulate.oracleDeps (resolveAt total pub) (deps20 S)
          (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux hash run []) =
      𝔼 fs : Sponge.Base L B C → Sponge.Y S C, value fs :=
  Simulate.oracleDeps_law (resolveAt total pub) (deps20 S)
    (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) (back := backAt pub) hash
    (resolveAt_local bits total) (deps20_closed S)
    (fun fs record _ point address resolved =>
      resolveAt_back erased long fs record point address resolved)
    (card_phiAt block block₁ block₂ block₃ block₄ block₅ blockL) run []
    (fun _ member => absurd member List.not_mem_nil) value factors

end Law

end NightstreamFPrime.Spec.Compression.Akita.Replay
