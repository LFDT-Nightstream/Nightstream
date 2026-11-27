import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Positions
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Wire
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageMapsReadings

/-! The recording prover of `key20.aks` on the experiment "adversary, then verifier" gives the
readings of the code on the event "accepted, no bad event" (Akita revision `d50f700`; ePrint
2026/1983, revision of 2026-11-23, Lemma 10.47, the paragraphs "Simulation" and "Acceptance";
`docs/akita-work/w22/design.md`; `Replay/Sound20Game.lean`, `Replay/Sound20Closed.lean`,
`Replay/Sound20Positions.lean`, `Replay/Sound20Wire.lean`, `Replay/StageMapsReadings.lean`).

- `Digest.Run.mem_trace_closed`, `mem_answered_inr`, `mem_digests`: the queries of a closed run
  and the record of its digest queries.
- `expect_fintype`, `indicator_congr`, `expect_union_le`, `expect_cover_le`: the union bound
  over a bad event, and the split of an event in two parts.
- `wires20_of_fresh`, `entry20_of_fresh`: for an accepted proof and a run with the queries of
  the verifier and no bad event, the wire messages `wires20` are the pages of the proof, and the
  table of the sponge at the base of each stage is the stage map of the answer of the table at
  the key of the stage.
- `seen20`: the table of the sponge that the algorithm sees in the recording prover on the closed
  run of the experiment. `accepted20`: the verifier accepts the bytes of the adversary with this
  table. `accepted20_iff`, `closedGame20_law`: this is the result of the simulation, and it has
  the probability of the experiment with a uniform sponge and a uniform digest function.
  `bad20`: a bad event of Lemma 10.47 on the closed run.
- `reads20_of_good`: the hypothesis `reads` of `verifyProof_bound20_of_staged` on the event
  "accepted, no collision, no guess". -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Digest.Run

variable {Fs Challenge Msg Digest Output : Type}

/-- Each query of a run is a query of its closed run. -/
theorem mem_trace_closed (deps : Fs → List Fs) (fs : Fs → Challenge) (hash : Msg → Digest)
    (run : Run Fs Challenge Msg Digest Output) :
    ∀ query ∈ run.trace fs hash, query ∈ (run.closed deps).trace fs hash := by
  induction run with
  | done output => exact fun _ member => member
  | challenge point next induction =>
      intro query member
      have member' : query ∈
          (Sum.inl point :: (next (fs point)).trace fs hash : List (Fs ⊕ Msg)) := member
      rw [trace_closed_challenge]
      rcases List.mem_cons.mp member' with same | later
      · exact List.mem_append_right _ (List.mem_cons.mpr (Or.inl same))
      · exact List.mem_append_right _
          (List.mem_cons_of_mem _ (induction (fs point) query later))
  | digest message next induction =>
      intro query member
      have member' : query ∈
          (Sum.inr message :: (next (hash message)).trace fs hash : List (Fs ⊕ Msg)) := member
      show query ∈
        (Sum.inr message :: ((next (hash message)).closed deps).trace fs hash : List (Fs ⊕ Msg))
      rcases List.mem_cons.mp member' with same | later
      · exact List.mem_cons.mpr (Or.inl same)
      · exact List.mem_cons_of_mem _ (induction (hash message) query later)

/-- The answered digest queries of a run are its digest queries with the answers of `hash`. -/
theorem mem_answered_inr {fs : Fs → Challenge} {hash : Msg → Digest}
    {run : Run Fs Challenge Msg Digest Output} {message : Msg} {digest : Digest} :
    Sum.inr (message, digest) ∈ run.answered fs hash ↔
      Sum.inr message ∈ run.trace fs hash ∧ digest = hash message := by
  induction run with
  | done output =>
      exact ⟨fun member => absurd member List.not_mem_nil,
        fun both => absurd both.1 List.not_mem_nil⟩
  | challenge point next induction =>
      have step := induction (fs point)
      constructor
      · intro member
        have member' : Sum.inr (message, digest) ∈
            (Sum.inl (point, fs point) :: (next (fs point)).answered fs hash :
              List ((Fs × Challenge) ⊕ (Msg × Digest))) := member
        rcases List.mem_cons.mp member' with bad | later
        · cases bad
        · obtain ⟨inside, value⟩ := step.mp later
          exact ⟨List.mem_cons_of_mem _ inside, value⟩
      · rintro ⟨inside, value⟩
        have inside' : Sum.inr message ∈
            (Sum.inl point :: (next (fs point)).trace fs hash : List (Fs ⊕ Msg)) := inside
        rcases List.mem_cons.mp inside' with bad | later
        · cases bad
        · exact List.mem_cons_of_mem _ (step.mpr ⟨later, value⟩)
  | digest asked next induction =>
      have step := induction (hash asked)
      constructor
      · intro member
        have member' : Sum.inr (message, digest) ∈
            (Sum.inr (asked, hash asked) :: (next (hash asked)).answered fs hash :
              List ((Fs × Challenge) ⊕ (Msg × Digest))) := member
        rcases List.mem_cons.mp member' with same | later
        · obtain ⟨first, second⟩ := Prod.mk.inj (Sum.inr.inj same)
          rw [first, second]
          exact ⟨List.mem_cons.mpr (Or.inl rfl), rfl⟩
        · obtain ⟨inside, value⟩ := step.mp later
          exact ⟨List.mem_cons_of_mem _ inside, value⟩
      · rintro ⟨inside, value⟩
        have inside' : Sum.inr message ∈
            (Sum.inr asked :: (next (hash asked)).trace fs hash : List (Fs ⊕ Msg)) := inside
        rcases List.mem_cons.mp inside' with same | later
        · rw [value, Sum.inr.inj same]
          exact List.mem_cons.mpr (Or.inl rfl)
        · exact List.mem_cons_of_mem _ (step.mpr ⟨later, value⟩)

/-- The record of the digest queries of a run, the newest first, has each digest query with the
answer of `hash`. -/
theorem mem_digests {fs : Fs → Challenge} {hash : Msg → Digest}
    {run : Run Fs Challenge Msg Digest Output} {message : Msg} {digest : Digest} :
    (message, digest) ∈ ((run.answered fs hash).filterMap Sum.getRight?).reverse ↔
      Sum.inr message ∈ run.trace fs hash ∧ digest = hash message := by
  rw [List.mem_reverse, List.mem_filterMap, ← mem_answered_inr]
  constructor
  · rintro ⟨entry, member, right⟩
    rw [Sum.getRight?_eq_some_iff] at right
    rw [← right]
    exact member
  · exact fun member => ⟨_, member, rfl⟩

end NightstreamFPrime.Spec.Compression.Akita.Digest.Run

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Prefix Address)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)
open Sponge (split answerOf spongeOf)
open Replayer

/-- The mean over a finite type is the same for each two enumerations of the type. -/
theorem expect_fintype {T : Type} (first second : Fintype T) (f : T → ℝ) :
    Finset.expect (@Finset.univ T first) f = Finset.expect (@Finset.univ T second) f := by
  rw [Subsingleton.elim first second]

/-- The indicator of an event is the same for each two decision procedures of the event. -/
theorem indicator_congr {p q : Prop} {first : Decidable p} {second : Decidable q}
    (same : p ↔ q) : @ite ℝ p first 1 0 = @ite ℝ q second 1 0 :=
  if_congr same rfl rfl

section Union

variable {T A H P : Type} [Fintype T] [Fintype A] [Fintype H] [Fintype P] [Nonempty A]
  [Nonempty H] [Nonempty P]

/-- The union bound over a bad event: an event that gives a guarded event when the bad event
does not occur has at most the probability of the guarded event, for each `a` and `h`, plus the
probability of the bad event. -/
theorem expect_union_le {event guarded : T → A → H → P → Prop} {bad : T → A → H → Prop}
    {dEvent : ∀ t a h p, Decidable (event t a h p)}
    {dGuarded : ∀ t a h p, Decidable (guarded t a h p)}
    {dBad : ∀ t a h, Decidable (bad t a h)} {ε δ : ℝ}
    (model : ∀ a h, 𝔼 t, 𝔼 p, (if guarded t a h p then (1 : ℝ) else 0) ≤ ε)
    (rare : 𝔼 t, 𝔼 a, 𝔼 h, (if bad t a h then (1 : ℝ) else 0) ≤ δ)
    (split : ∀ t a h p, event t a h p → ¬ bad t a h → guarded t a h p) :
    𝔼 t, 𝔼 a, 𝔼 h, 𝔼 p, (if event t a h p then (1 : ℝ) else 0) ≤ ε + δ := by
  have pointwise : ∀ t a h p, (if event t a h p then (1 : ℝ) else 0) ≤
      (if guarded t a h p then (1 : ℝ) else 0) + (if bad t a h then (1 : ℝ) else 0) := by
    intro t a h p
    by_cases e : event t a h p
    · by_cases b : bad t a h
      · rw [ite_eq_left e, ite_eq_left b]
        split_ifs <;> norm_num
      · rw [ite_eq_left e, ite_eq_left (split t a h p e b), ite_eq_right b]
        norm_num
    · rw [ite_eq_right e]
      split_ifs <;> norm_num
  have inner : ∀ t a h, 𝔼 p, ((if guarded t a h p then (1 : ℝ) else 0) +
      (if bad t a h then (1 : ℝ) else 0)) =
      (𝔼 p, (if guarded t a h p then (1 : ℝ) else 0)) + (if bad t a h then (1 : ℝ) else 0) := by
    intro t a h
    rw [Finset.expect_add_distrib,
      Finset.expect_const (Finset.univ_nonempty : (Finset.univ : Finset P).Nonempty)]
  have swapped : 𝔼 t, 𝔼 a, 𝔼 h, 𝔼 p, (if guarded t a h p then (1 : ℝ) else 0) ≤ ε :=
    calc 𝔼 t, 𝔼 a, 𝔼 h, 𝔼 p, (if guarded t a h p then (1 : ℝ) else 0)
        = 𝔼 a, 𝔼 t, 𝔼 h, 𝔼 p, (if guarded t a h p then (1 : ℝ) else 0) :=
          Finset.expect_comm _ _ _
      _ = 𝔼 a, 𝔼 h, 𝔼 t, 𝔼 p, (if guarded t a h p then (1 : ℝ) else 0) :=
          Finset.expect_congr rfl fun _ _ => Finset.expect_comm _ _ _
      _ ≤ ε := Simulate.expect_le_const _ _ fun a => Simulate.expect_le_const _ _ fun h =>
          model a h
  calc 𝔼 t, 𝔼 a, 𝔼 h, 𝔼 p, (if event t a h p then (1 : ℝ) else 0)
      ≤ 𝔼 t, 𝔼 a, 𝔼 h, 𝔼 p, ((if guarded t a h p then (1 : ℝ) else 0) +
          (if bad t a h then (1 : ℝ) else 0)) :=
        Finset.expect_le_expect fun t _ => Finset.expect_le_expect fun a _ =>
          Finset.expect_le_expect fun h _ => Finset.expect_le_expect fun p _ =>
            pointwise t a h p
    _ = (𝔼 t, 𝔼 a, 𝔼 h, 𝔼 p, (if guarded t a h p then (1 : ℝ) else 0)) +
          𝔼 t, 𝔼 a, 𝔼 h, (if bad t a h then (1 : ℝ) else 0) := by
        simp only [inner, Finset.expect_add_distrib]
    _ ≤ ε + δ := add_le_add swapped rare

omit [Nonempty A] [Nonempty H] in
/-- An event with two parts for each `p`: its probability is at most the probability of the
first part plus a bound of the probability of the second part. -/
theorem expect_cover_le {accept : T → A → H → Prop} {won lost : T → A → H → P → Prop}
    {dAccept : ∀ t a h, Decidable (accept t a h)}
    {dWon : ∀ t a h p, Decidable (won t a h p)}
    {dLost : ∀ t a h p, Decidable (lost t a h p)} {ε : ℝ}
    (bound : 𝔼 t, 𝔼 a, 𝔼 h, 𝔼 p, (if lost t a h p then (1 : ℝ) else 0) ≤ ε)
    (cover : ∀ t a h p, accept t a h → won t a h p ∨ lost t a h p) :
    𝔼 t, 𝔼 a, 𝔼 h, (if accept t a h then (1 : ℝ) else 0) ≤
      (𝔼 t, 𝔼 a, 𝔼 h, 𝔼 p, (if won t a h p then (1 : ℝ) else 0)) + ε := by
  have pointwise : ∀ t a h p, (if accept t a h then (1 : ℝ) else 0) ≤
      (if won t a h p then (1 : ℝ) else 0) + (if lost t a h p then (1 : ℝ) else 0) := by
    intro t a h p
    by_cases e : accept t a h
    · rcases cover t a h p e with w | l
      · rw [ite_eq_left e, ite_eq_left w]
        split_ifs <;> norm_num
      · rw [ite_eq_left e, ite_eq_left l]
        split_ifs <;> norm_num
    · rw [ite_eq_right e]
      split_ifs <;> norm_num
  have constant : ∀ t a h, (if accept t a h then (1 : ℝ) else 0) =
      𝔼 _p : P, (if accept t a h then (1 : ℝ) else 0) := fun t a h =>
    (Finset.expect_const (Finset.univ_nonempty : (Finset.univ : Finset P).Nonempty) _).symm
  calc 𝔼 t, 𝔼 a, 𝔼 h, (if accept t a h then (1 : ℝ) else 0)
      = 𝔼 t, 𝔼 a, 𝔼 h, 𝔼 _p : P, (if accept t a h then (1 : ℝ) else 0) :=
        Finset.expect_congr rfl fun t _ => Finset.expect_congr rfl fun a _ =>
          Finset.expect_congr rfl fun h _ => constant t a h
    _ ≤ 𝔼 t, 𝔼 a, 𝔼 h, 𝔼 p, ((if won t a h p then (1 : ℝ) else 0) +
          (if lost t a h p then (1 : ℝ) else 0)) :=
        Finset.expect_le_expect fun t _ => Finset.expect_le_expect fun a _ =>
          Finset.expect_le_expect fun h _ => Finset.expect_le_expect fun p _ =>
            pointwise t a h p
    _ = (𝔼 t, 𝔼 a, 𝔼 h, 𝔼 p, (if won t a h p then (1 : ℝ) else 0)) +
          𝔼 t, 𝔼 a, 𝔼 h, 𝔼 p, (if lost t a h p then (1 : ℝ) else 0) := by
        simp only [Finset.expect_add_distrib]
    _ ≤ _ := add_le_add le_rfl bound

end Union

attribute [local instance low] Classical.propDecidable

section Accepted

/-- An accepted proof has decoded messages. -/
theorem decoded_of_accepted {pub : Public} {proof : ByteArray}
    (accepted : verifyProof pub proof = .verdict .accept) :
    ∃ messages rest, decode pub.sealed.ops proof = .ok (messages, rest) := by
  have ran := ((verifyProof_accept_iff _ _).mp accepted).2
  have scripted := verifyProof_scripted ran
  rw [verifyProof_script _ proof scripted] at ran
  obtain ⟨messages, rest, decoded, -⟩ := (verifyBytes_accept _ _ _).mp ran
  exact ⟨messages, rest, decoded⟩

/-- An accepted proof has a public input whose layout takes its plan. -/
theorem scripted_of_accepted {pub : Public} {proof : ByteArray}
    (accepted : verifyProof pub proof = .verdict .accept) :
    script pub.layout pub.plan = .ok pub.ops := by
  have scripted := verifyProof_scripted ((verifyProof_accept_iff _ _).mp accepted).2
  exact scripted

variable {L B C S D : ℕ}

/-- Each query of the verifier on the bytes of the adversary is a query of the closed run of
the experiment. -/
theorem game20_queried (pubV : Public)
    (adversary : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32)
      ByteArray)
    (fs : Sponge.Base L B C → Sponge.Y S C) (hash : Sponge.Str D → Bytes 32) :
    ∀ query ∈ (verifierRun pubV (adversary.eval fs hash)).trace fs hash,
      query ∈ ((game20 pubV adversary).closed (deps20 S)).trace fs hash := by
  intro query member
  refine Digest.Run.mem_trace_closed (deps20 S) fs hash _ query ?_
  rw [game20_trace]
  exact List.mem_append_right _ member

end Accepted

section Record

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {L B C S D : ℕ}
  {Output Result Store : Type}

/-- The wire messages of the recording prover are the pages of the accepted proof, for a run
with the queries of the verifier and no collision: `wires20_eq` with the answered queries of the
run and a record that has each digest query of the run with the answer of `hash`. -/
theorem wires20_of_fresh {pubV : Public} {zMax rice : ℕ}
    (layout : pubV.layout = layout20 zMax rice)
    {fs : Sponge.Base L B C → Sponge.Y S C} {hash : Sponge.Str D → Bytes 32} {proof : ByteArray}
    (accepted : verifyProof (pubV.withOracles fs hash) proof = .verdict .accept)
    {messages : Array Message} {rest : ℕ}
    (decoded : decode (pubV.withOracles fs hash).sealed.ops proof = .ok (messages, rest))
    (fits : 4128 ≤ D) {tLast : ℕ → Rq Key20.terminal.innerDegree}
    {eLast : Rq Key20.terminal.innerDegree}
    (last : Last20 (E := E) (pubV.withOracles fs hash).sealed proof messages tLast eLast)
    {run : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) Output}
    (unique : ¬ Digest.Run.Collided fs hash run)
    (queried : ∀ query ∈ (verifierRun pubV proof).trace fs hash, query ∈ run.trace fs hash)
    {record : List (Sponge.Str D × Bytes 32)}
    (listed : ∀ message digest, (message, digest) ∈ record ↔
      Sum.inr message ∈ run.trace fs hash ∧ digest = hash message)
    (result : Result) (cache : Store) (k : ℕ) :
    wires20 (E := E) (pubV.withSponge Duplex.empty).sealed
        (((proof, result), run.answered fs hash), cache, record) k =
      stagePage20 (pubV.withOracles fs hash).sealed proof messages tLast eLast k := by
  obtain ⟨asked, tAsked, eAsked⟩ := wires20_guard layout accepted decoded fits last
  refine wires20_eq layout accepted decoded fits last (Digest.Run.answered_inl fs hash run)
    (fun j lt base events => Digest.Run.answered_of_trace (queried _ (asked j lt base events)))
    ((listed _ _).mpr ⟨queried _ tAsked, rfl⟩) ((listed _ _).mpr ⟨queried _ eAsked, rfl⟩)
    (fun a b d first second => ?_) result cache k
  obtain ⟨aAsked, aValue⟩ := (listed a d).mp first
  obtain ⟨bAsked, bValue⟩ := (listed b d).mp second
  by_contra different
  exact unique ⟨a, aAsked, b, bAsked, different, aValue.symm.trans bValue⟩

end Record

section Entry

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {L B C S D total : ℕ}
  (block : Key20.root.Block → Coord) (block₁ : Fin Key20.level₁.shape.blocks → Coord)
  (block₂ : Fin Key20.level₂.shape.blocks → Coord) (block₃ : Fin Key20.level₃.shape.blocks → Coord)
  (block₄ : Fin Key20.level₄.shape.blocks → Coord) (block₅ : Fin Key20.level₅.shape.blocks → Coord)
  (blockL : Fin Key20.terminal.blocks → Coord)

local notation "Reply" => (Option Coord → CodeSlot E)

local notation "Addr" => Address (Key20.Wire E) total

/-- The table of the sponge at the base of a stage is the stage map of the answer of the model
at the key of the stage. For an accepted proof of `layout20` with the tables `fs` and `hash` and
last blocks `Last20`, a run with the queries of the verifier and no bad event for `positions20`,
and a table `fs` whose answer at each key with a query is the answer of the recording prover
with the address `resolveAt` of the digest queries before a query of the key: the answer of `fs`
at the base of the stage `k < 473` is `phi20 … k` of the answer of `table` at the key
`addressAt20 … k` of the pages of the proof. -/
theorem entry20_of_fresh {Output : Type} {pubV : Public} {zMax rice : ℕ}
    (layout : pubV.layout = layout20 zMax rice)
    {fs : Sponge.Base L B C → Sponge.Y S C} {hash : Sponge.Str D → Bytes 32} {proof : ByteArray}
    (accepted : verifyProof (pubV.withOracles fs hash) proof = .verdict .accept)
    {messages : Array Message} {rest : ℕ}
    (decoded : decode (pubV.withOracles fs hash).sealed.ops proof = .ok (messages, rest))
    (fits : 4128 ≤ D) {tLast : ℕ → Rq Key20.terminal.innerDegree}
    {eLast : Rq Key20.terminal.innerDegree}
    (last : Last20 (E := E) (pubV.withOracles fs hash).sealed proof messages tLast eLast)
    {run : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) Output}
    (unique : ¬ Digest.Run.Collided fs hash run)
    (fresh : ¬ Digest.Run.Guessed fs (positions20 (pubV.withSponge Duplex.empty).sealed fs)
      hash run)
    (queried : ∀ query ∈ (verifierRun pubV proof).trace fs hash, query ∈ run.trace fs hash)
    (table : Addr → Reply) (aux : Sponge.Base L B C → Sponge.Y S C)
    (entries : ∀ base, Sum.inl base ∈ run.trace fs hash →
      ∃ pre post, run.trace fs hash = pre ++ Sum.inl base :: post ∧
        fs base = Simulate.answer (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table
          aux base (resolveAt total (pubV.withSponge Duplex.empty).sealed fs
            (Simulate.recorded hash pre) base))
    {k : ℕ} (inside : k < 473) {base : Sponge.Base L B C}
    (events : base.events = eventsAt20 (pubV.withOracles fs hash).sealed proof messages k) :
    fs base = phi20 block block₁ block₂ block₃ block₄ block₅ blockL k
      (table (addressAt20 (pubV.withOracles fs hash).sealed proof messages tLast eLast total k),
        aux base) := by
  obtain ⟨asked, tAsked, eAsked⟩ := wires20_guard layout accepted decoded fits last
  obtain ⟨bits, stage⟩ := stage20_of_accepted layout accepted decoded inside
  obtain ⟨pre, post, parts, value⟩ := entries base (queried _ (asked k inside base events))
  obtain ⟨tListed, eListed⟩ := record20_of_fresh layout accepted decoded fits last inside events
    unique fresh (queried _ tAsked) (queried _ eAsked) parts
  obtain ⟨-, -, ran, same⟩ := stage
  obtain ⟨-, scripted, -⟩ := replay_of_accepted pubV proof fs hash accepted decoded
  have erased : (pubV.withOracles fs hash).sealed.ops.toList.map Op.erase =
      skeleton (layout20 0 0) := by
    rw [layout] at scripted
    exact (script_erase scripted).trans (skeleton20 zMax rice)
  have layoutR : ((pubV.withSponge Duplex.empty).sealed.reideal
      (some (answerOf fs))).layout = layout20 zMax rice := layout
  have resolved := resolveWith20_eq (E := E) (pub := (pubV.withSponge Duplex.empty).sealed)
    (a := answerOf fs) (a' := answerOf fs) bits inside decoded ran (AgreeOn.refl _ _) tListed
    eListed total
  have moved : addressAt20 (E := E) ((pubV.withSponge Duplex.empty).sealed.reideal
      (some (answerOf fs))) proof messages tLast eLast total k =
      addressAt20 (pubV.withOracles fs hash).sealed proof messages tLast eLast total k :=
    (addressAt20_withDigest ((pubV.withSponge Duplex.empty).sealed.reideal (some (answerOf fs)))
      (digestOf hash) proof messages tLast eLast layoutR total k).symm
  have address : resolveAt (E := E) total (pubV.withSponge Duplex.empty).sealed fs
      (Simulate.recorded hash pre) base =
      some (addressAt20 (pubV.withOracles fs hash).sealed proof messages tLast eLast total k) :=
    ((congrArg (resolveWith20 (E := E) total (pubV.withSponge Duplex.empty).sealed (answerOf fs)
      (recordOf (Simulate.recorded hash pre))) (events.trans same.symm)).trans resolved).trans
      (congrArg some moved)
  have stageK : stageOf base = k :=
    stageOf_base (pub := (pubV.withSponge Duplex.empty).sealed.reideal (some (answerOf fs)))
      (a := answerOf fs) erased bits rfl rfl inside ran (events.trans same.symm)
  rw [value, address]
  show phi20 block block₁ block₂ block₃ block₄ block₅ blockL (stageOf base) _ = _
  rw [stageK]

variable (pubV : Public)
  (adversary : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) ByteArray)

/-- The table of the sponge that the algorithm sees in the recording prover of `key20.aks` on
the closed run of the experiment, from the empty cache and the empty record. -/
noncomputable def seen20 (table : Addr → Reply) (aux : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32) : Sponge.Base L B C → Sponge.Y S C :=
  Simulate.oracleDeps (resolveAt total (pubV.withSponge Duplex.empty).sealed) (deps20 S)
    (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux hash
    ((game20 pubV adversary).closed (deps20 S)) []

/-- The verifier accepts the bytes of the adversary in the recording prover: `verifyProof` with
the table `seen20` of the sponge and the digest function `hash` accepts the bytes that the
adversary gives with these two oracles. -/
def accepted20 (table : Addr → Reply) (aux : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32) : Prop :=
  verifyProof
    (pubV.withOracles (seen20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary
      table aux hash) hash)
    (adversary.eval (seen20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary
      table aux hash) hash) = .verdict .accept

local notation "seen" => seen20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary

/-- `accepted20` is the result of the verifier in the simulation of the closed run of the
experiment. -/
theorem accepted20_iff (table : Addr → Reply) (aux : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32) :
    accepted20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux hash ↔
      (Simulate.simulateDeps (resolveAt total (pubV.withSponge Duplex.empty).sealed) (deps20 S)
        (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux hash
        ((game20 pubV adversary).closed (deps20 S)) (fun _ => none) []).1.2 =
          .verdict .accept := by
  rw [Simulate.simulateDeps_oracle, Digest.Run.eval_closed, game20_eval]
  exact Iff.rfl

/-- The answer of `seen20` at each key with a query of the closed run is the answer of the
recording prover with the address `resolveAt` of the digest queries before a query of the
key. -/
theorem seen20_entry (bits : Bits20 (pubV.withSponge Duplex.empty).sealed)
    (table : Addr → Reply) (aux : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32) (base : Sponge.Base L B C)
    (asked : Sum.inl base ∈
      ((game20 pubV adversary).closed (deps20 S)).trace (seen table aux hash) hash) :
    ∃ pre post,
      ((game20 pubV adversary).closed (deps20 S)).trace (seen table aux hash) hash =
          pre ++ Sum.inl base :: post ∧
        seen table aux hash base =
          Simulate.answer (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux base
            (resolveAt total (pubV.withSponge Duplex.empty).sealed (seen table aux hash)
              (Simulate.recorded hash pre) base) := by
  obtain ⟨slot, found⟩ := Option.isSome_iff_exists.mp
    (Simulate.simulateDeps_closed_asked (resolveAt total (pubV.withSponge Duplex.empty).sealed)
      (deps20 S) (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux hash
      (game20 pubV adversary) base asked)
  obtain ⟨pre, post, parts, -, value⟩ :=
    Simulate.simulateDeps_closed_entry (resolveAt total (pubV.withSponge Duplex.empty).sealed)
      (deps20 S) (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux hash
      (resolveAt_local bits total) (deps20_closed S) (game20 pubV adversary) base slot found
  exact ⟨pre, post, parts,
    (congrArg (fun entry : Option (Option Addr) =>
      Simulate.answer (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux base
        (entry.getD none)) found).trans
      (congrArg (Simulate.answer (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table
        aux base) value)⟩

/-- The proof of `closedProver20` on the experiment has the wire messages `wires20` of the bytes
of the adversary, the result of the verifier, the answered queries of the closed run with the
table `seen20`, and the record of its digest queries. -/
theorem seen20_run (table : Addr → Reply) (aux : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32) :
    (closedProver20 (total := total) block block₁ block₂ block₃ block₄ block₅ blockL
      (pubV.withSponge Duplex.empty).sealed aux hash (game20 pubV adversary)).run table =
      wires20 (pubV.withSponge Duplex.empty).sealed
        (((adversary.eval (seen table aux hash) hash,
            verifyProof (pubV.withOracles (seen table aux hash) hash)
              (adversary.eval (seen table aux hash) hash)),
          ((game20 pubV adversary).closed (deps20 S)).answered (seen table aux hash) hash),
          (Simulate.simulateDeps (resolveAt total (pubV.withSponge Duplex.empty).sealed)
            (deps20 S) (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux hash
            ((game20 pubV adversary).closed (deps20 S)) (fun _ => none) []).2.1,
          ((((game20 pubV adversary).closed (deps20 S)).answered (seen table aux hash)
            hash).filterMap Sum.getRight?).reverse) := by
  have value : ((game20 pubV adversary).closed (deps20 S)).eval (seen table aux hash) hash =
      (adversary.eval (seen table aux hash) hash,
        verifyProof (pubV.withOracles (seen table aux hash) hash)
          (adversary.eval (seen table aux hash) hash)) := by
    rw [Digest.Run.eval_closed, game20_eval]
  have record := Simulate.simulateDeps_record_oracle
    (resolveAt total (pubV.withSponge Duplex.empty).sealed) (deps20 S)
    (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux hash
    ((game20 pubV adversary).closed (deps20 S)) []
  rw [List.append_nil] at record
  rw [closedProver20_run, Simulate.simulateDeps_logged_eq]
  exact congrArg (wires20 (pubV.withSponge Duplex.empty).sealed)
    (Prod.ext (Prod.ext value rfl) (Prod.ext rfl record))

/-- The verifier accepts the bytes of the adversary in the recording prover on the closed run
with the probability of the experiment with a uniform table of the sponge and a uniform digest
function (Akita Lemma 10.47, the paragraph "Simulation"). -/
theorem closedGame20_law {zMax rice : ℕ} (layout : pubV.layout = layout20 zMax rice)
    (long : 472 ≤ total) :
    𝔼 table : Addr → Reply, 𝔼 aux : Sponge.Base L B C → Sponge.Y S C,
        𝔼 hash : Sponge.Str D → Bytes 32,
        (if accepted20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux
          hash then (1 : ℝ) else 0) =
      𝔼 fs : Sponge.Base L B C → Sponge.Y S C, 𝔼 hash : Sponge.Str D → Bytes 32,
        (if verifyProof (pubV.withOracles fs hash) (adversary.eval fs hash) = .verdict .accept
          then (1 : ℝ) else 0) := by
  by_cases scripted : script pubV.layout pubV.plan = .ok pubV.ops
  · have erased : (pubV.withSponge Duplex.empty).sealed.ops.toList.map Op.erase =
        skeleton (layout20 0 0) := by
      have scripted' := scripted
      rw [layout] at scripted'
      exact (script_erase scripted').trans (skeleton20 zMax rice)
    have bits : Bits20 (pubV.withSponge Duplex.empty).sealed :=
      bits20_of_scripted (pub := (pubV.withSponge Duplex.empty).sealed) layout scripted
    have law := simulate20_law (E := E) (total := total) block block₁ block₂ block₃ block₄
      block₅ blockL erased bits long ((game20 pubV adversary).closed (deps20 S))
      fun _ output => if output.2 = .verdict .accept then (1 : ℝ) else 0
    have value : ∀ (fs : Sponge.Base L B C → Sponge.Y S C) (hash : Sponge.Str D → Bytes 32),
        (((game20 pubV adversary).closed (deps20 S)).eval fs hash).2 =
          verifyProof (pubV.withOracles fs hash) (adversary.eval fs hash) := fun fs hash =>
      congrArg Prod.snd ((Digest.Run.eval_closed (deps20 S) fs hash (game20 pubV adversary)).trans
        (game20_eval pubV adversary fs hash))
    refine Eq.trans ?_ (law.trans ?_)
    · exact Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun aux _ =>
        Finset.expect_congr rfl fun hash _ => indicator_congr
          (accepted20_iff block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table
            aux hash)
    · exact Finset.expect_congr rfl fun fs _ => Finset.expect_congr rfl fun hash _ =>
        indicator_congr (by rw [value fs hash])
  · have left : ∀ (table : Addr → Reply) (aux : Sponge.Base L B C → Sponge.Y S C)
        (hash : Sponge.Str D → Bytes 32),
        ¬ accepted20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux
          hash :=
      fun _ _ _ accepted => by
        have taken := scripted_of_accepted accepted
        exact scripted taken
    have right : ∀ (fs : Sponge.Base L B C → Sponge.Y S C) (hash : Sponge.Str D → Bytes 32),
        ¬ verifyProof (pubV.withOracles fs hash) (adversary.eval fs hash) = .verdict .accept :=
      fun _ _ accepted => by
        have taken := scripted_of_accepted accepted
        exact scripted taken
    simp [left, right]

/-- A bad event of Akita Lemma 10.47 on the closed run of the experiment with the table
`seen20`: two digest queries with one answer, or the answer of a digest query in a digest
position `positions20` of this query or of an earlier query. -/
def bad20 (table : Addr → Reply) (aux : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32) : Prop :=
  Digest.Run.Collided (seen table aux hash) hash ((game20 pubV adversary).closed (deps20 S)) ∨
    Digest.Run.Guessed (seen table aux hash)
      (positions20 (pubV.withSponge Duplex.empty).sealed (seen table aux hash)) hash
      ((game20 pubV adversary).closed (deps20 S))

end Entry

section Sound

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {L B C S D total : ℕ}
  (block : Key20.root.Block → Coord) (block₁ : Fin Key20.level₁.shape.blocks → Coord)
  (block₂ : Fin Key20.level₂.shape.blocks → Coord) (block₃ : Fin Key20.level₃.shape.blocks → Coord)
  (block₄ : Fin Key20.level₄.shape.blocks → Coord) (block₅ : Fin Key20.level₅.shape.blocks → Coord)
  (blockL : Fin Key20.terminal.blocks → Coord) (pubV : Public)
  (adversary : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) ByteArray)

local notation "Reply" => (Option Coord → CodeSlot E)

local notation "Addr" => Address (Key20.Wire E) total

local notation "seen" => seen20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary

/-- The hypothesis `reads` of `verifyProof_bound20_of_staged` for the recording prover on the
experiment, on the event "accepted, no collision, no guess". For `layout20`, the statement shape
`root20` and the setup facts of the seed: when `verifyProof` with the table `seen20` and the
digest function `hash` accepts the bytes of the adversary, and the closed run of the experiment
has no bad event for `positions20`, the transcript of `closedProver20` with the answers of
`table` has the 473 stages, its wire message `k ≤ 473` is the page `stagePage20 … k` of the
bytes, and its answers have the readings of the code on the answers of `seen20`, for last blocks
of the code. -/
theorem reads20_of_good {zMax rice : ℕ} (layout : pubV.layout = layout20 zMax rice)
    (root : pubV.root = root20) {full : Array Fq}
    (terminalSetup : pubV.sealed.setupFor pubV.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pubV.sealed.setupFor (fold20 ℓ).setupCount).isSome)
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord)
    (inside : 472 < total) (sizes : drawBytes20 ≤ C) (three : 3 ≤ S) (fits : 4128 ≤ D)
    (table : Addr → Reply) (aux : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32)
    (accepted : verifyProof (pubV.withOracles (seen table aux hash) hash)
      (adversary.eval (seen table aux hash) hash) = .verdict .accept)
    (unique : ¬ Digest.Run.Collided (seen table aux hash) hash
      ((game20 pubV adversary).closed (deps20 S)))
    (fresh : ¬ Digest.Run.Guessed (seen table aux hash)
      (positions20 (pubV.withSponge Duplex.empty).sealed (seen table aux hash)) hash
      ((game20 pubV adversary).closed (deps20 S))) :
    ∃ (messages : Array Message) (rest : ℕ) (tLast : ℕ → Rq Key20.terminal.innerDegree)
      (eLast : Rq Key20.terminal.innerDegree) (answer : List Duplex.Event → ℕ → ByteArray),
      decode pubV.ops (adversary.eval (seen table aux hash) hash) = .ok (messages, rest) ∧
      (spongeOf (seen table aux hash)).ideal = some answer ∧
      Last20 (E := E) (pubV.withOracles (seen table aux hash) hash).sealed
        (adversary.eval (seen table aux hash) hash) messages tLast eLast ∧
      473 ≤ ((tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
        472).body.protocol.final
        (closedProver20 block block₁ block₂ block₃ block₄ block₅ blockL
          (pubV.withSponge Duplex.empty).sealed aux hash (game20 pubV adversary))
        table).rest.length ∧
      (∀ k ≤ 473, ((tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL
          total 472).body.protocol.final
          (closedProver20 block block₁ block₂ block₃ block₄ block₅ blockL
            (pubV.withSponge Duplex.empty).sealed aux hash (game20 pubV adversary))
          table).wire k =
        stagePage20 (pubV.withOracles (seen table aux hash) hash).sealed
          (adversary.eval (seen table aux hash) hash) messages tLast eLast k) ∧
      Readings20 (pubV.withOracles (seen table aux hash) hash).sealed
        (adversary.eval (seen table aux hash) hash) messages answer block block₁ block₂ block₃
        block₄ block₅ blockL
        ((tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
          472).body.protocol.final
          (closedProver20 block block₁ block₂ block₃ block₄ block₅ blockL
            (pubV.withSponge Duplex.empty).sealed aux hash (game20 pubV adversary))
          table) := by
  have entries := fun bits base asked =>
    seen20_entry block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary bits table aux
      hash base asked
  have wiresRun := seen20_run block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary
    table aux hash
  generalize seen table aux hash = fs at accepted unique fresh entries wiresRun ⊢
  have queried := game20_queried (S := S) pubV adversary fs hash
  generalize adversary.eval fs hash = proof at accepted wiresRun queried ⊢
  have long : 473 ≤ total := inside
  obtain ⟨messages, rest, decoded⟩ := decoded_of_accepted accepted
  obtain ⟨tLast, eLast, _, last, -⟩ := verifyProof_accepts20 (E := E) (total := total)
    (pub := pubV.withOracles fs hash) (block := block) (block₁ := block₁) (block₂ := block₂)
    (block₃ := block₃) (block₄ := block₄) (block₅ := block₅) (blockL := blockL) layout root
    accepted terminalSetup expanded decoded injective injective₁ injective₂ injective₃ injective₄
    injective₅ injectiveL slots long
  have ranV := ((verifyProof_accept_iff _ _).mp accepted).2
  have scripted := scripted_of_accepted accepted
  have bits : Bits20 (pubV.withSponge Duplex.empty).sealed :=
    bits20_of_scripted (pub := (pubV.withSponge Duplex.empty).sealed) layout scripted
  have wires : ∀ k, (closedProver20 (total := total) block block₁ block₂ block₃ block₄ block₅
      blockL (pubV.withSponge Duplex.empty).sealed aux hash (game20 pubV adversary)).run table k =
      stagePage20 (pubV.withOracles fs hash).sealed proof messages tLast eLast k := fun k => by
    rw [wiresRun]
    exact wires20_of_fresh layout accepted decoded fits last unique queried
      (fun _ _ => Digest.Run.mem_digests) _ _ k
  refine ⟨messages, rest, tLast, eLast, answerOf fs, decoded, rfl, last, ?_, ?_, ?_⟩
  · rw [final_length20]
    exact long
  · intro k le
    rw [final_wire20 pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL 472 _ table
      (le.trans long)]
    exact wires k
  · refine readings20_of_coupled (pub := (pubV.withOracles fs hash).sealed) layout ranV
      terminalSetup decoded fs rfl sizes three injective injective₁ injective₂ injective₃
      injective₄ injective₅ injectiveL slots ?_
    intro k below reply at_ base events
    rw [final_reply20 pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL 472 _ table
      (Nat.lt_of_lt_of_le below long) (fun j _ => wires j)] at at_
    refine ⟨aux base, ?_⟩
    rw [← Option.some.inj at_]
    exact entry20_of_fresh block block₁ block₂ block₃ block₄ block₅ blockL layout accepted
      decoded fits last unique fresh queried table aux (entries bits) below events

end Sound

end NightstreamFPrime.Spec.Compression.Akita.Replay
