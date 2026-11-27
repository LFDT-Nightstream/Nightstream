import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Chain
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Count
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Reads

/-! The soundness bound of the claims for `key20.aks` on the byte verifier `verifyProof`, under
the ideal sponge and the ideal digest only (Akita revision `d50f700`; ePrint 2026/1983, revision
of 2026-11-23, Theorem 10.17, Lemma 10.47 with (213), and Section 10.3;
`docs/akita-work/w22/design.md`; `SECURITY_MODEL.md`, `[KECCAK]`; `Replay/Sound20Reads.lean`,
`Replay/Sound20Game.lean`, `Replay/Sound20Positions.lean`, `Replay/Sound20Count.lean`,
`Replay/Sound20Chain.lean`, `Replay/JoinStagePage.lean`).

- `scripted20`, `bad20_closed_le`: a bad event of the experiment has probability at most
  `(C(Q_dig + 2, 2) + 2·(Q + 473)·(Q_dig + 2))/|Bytes 32|`.
- `closedProver20_spread`: the recording prover on a closed run has at most `Q` different keys of
  each stage, `Q` the count of the run before `closed`.
- `verifyProof_bound20_guarded`: `verifyProof_bound20_of_staged` with a guard in the event and a
  hash for each table. `guarded20`: this bound for the recording prover on the experiment.
- `verifyProof_sound20`: for each adversary with at most `Q` sponge queries and `Q_dig` digest
  queries, the probability that the verifier accepts and the extractor returns no output is at
  most `(Q + 474)·(317479/(1024·|E|) + 135/q²) + (Q + 474)·142/budget
  + (C(Q_dig + 2, 2) + 2·(Q + 473)·(Q_dig + 2))/|Bytes 32|`.
- `verifyProof_sound20_real`: the same for the experiment with a uniform sponge. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Prefix Address)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)
open Sponge (split answerOf spongeOf)
open Replayer

attribute [local instance low] Classical.propDecidable

section Rare

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {L B C S D total : ℕ}
  (block : Key20.root.Block → Coord) (block₁ : Fin Key20.level₁.shape.blocks → Coord)
  (block₂ : Fin Key20.level₂.shape.blocks → Coord) (block₃ : Fin Key20.level₃.shape.blocks → Coord)
  (block₄ : Fin Key20.level₄.shape.blocks → Coord) (block₅ : Fin Key20.level₅.shape.blocks → Coord)
  (blockL : Fin Key20.terminal.blocks → Coord) (pubV : Public)
  (adversary : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) ByteArray)

local notation "Reply" => (Option Coord → CodeSlot E)

local notation "Addr" => Address (Key20.Wire E) total

/-- For a public input of `layout20` whose layout takes its plan, the public input of `resolveAt`
in the assembly has the script of `layout20` and the work bits of the model. -/
theorem scripted20 {pubV : Public} {zMax rice : ℕ} (layout : pubV.layout = layout20 zMax rice)
    (scripted : script pubV.layout pubV.plan = .ok pubV.ops) :
    (pubV.withSponge Duplex.empty).sealed.ops.toList.map Op.erase = skeleton (layout20 0 0) ∧
      Bits20 (pubV.withSponge Duplex.empty).sealed := by
  have scripted' := scripted
  rw [layout] at scripted'
  exact ⟨(script_erase scripted').trans (skeleton20 zMax rice),
    bits20_of_scripted (pub := (pubV.withSponge Duplex.empty).sealed) layout scripted⟩

/-- The bound `ε_dig` of Akita (213) for the experiment, with the count `Q + 473` of the run
before `closed`: for a uniform `table`, a uniform `aux` and a uniform digest function, a bad
event `bad20` has probability at most `(C(Q_dig + 2, 2) + 2·(Q + 473)·(Q_dig + 2))/|Bytes 32|`,
for an adversary with at most `Q` sponge queries and `Q_dig` digest queries. -/
theorem bad20_closed_le {zMax rice : ℕ} (layout : pubV.layout = layout20 zMax rice)
    (scripted : script pubV.layout pubV.plan = .ok pubV.ops) (long : 472 ≤ total)
    {queries digests : ℕ} (bounded : adversary.Bound queries digests) :
    𝔼 table : Addr → Reply, 𝔼 aux : Sponge.Base L B C → Sponge.Y S C,
        𝔼 hash : Sponge.Str D → Bytes 32,
        (if bad20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux hash
          then (1 : ℝ) else 0) ≤
      (((digests + 2).choose 2 + 2 * (queries + 473) * (digests + 2) : ℕ) : ℝ) /
        Fintype.card (Bytes 32) := by
  obtain ⟨erased, bits⟩ := scripted20 layout scripted
  have core := Simulate.oracleDeps_closed_bad_le
    (resolveAt (E := E) total (pubV.withSponge Duplex.empty).sealed) (deps20 S)
    (phiAt (E := E) block block₁ block₂ block₃ block₄ block₅ blockL)
    (back := backAt (pubV.withSponge Duplex.empty).sealed) (resolveAt_local bits total)
    (deps20_closed S)
    (fun fs record point address resolved =>
      resolveAt_back erased long fs record point address resolved)
    (card_phiAt block block₁ block₂ block₃ block₄ block₅ blockL) (game20 pubV adversary)
    (game20_bound layout bounded) (positions20 (pubV.withSponge Duplex.empty).sealed) (width := 2)
    (fun fs point => positions20_chain erased bits (fresh20 pubV) fs point) (fun _ _ => rfl)
    (positions20_alike bits (game20 pubV adversary))
  refine le_trans (le_of_eq ?_) core
  exact Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun aux _ =>
    Finset.expect_congr rfl fun hash _ => indicator_congr Iff.rfl

end Rare

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

/-- The recording prover with no named key on a closed run has at most `Q` different keys of each
stage, `Q` the count of sponge queries of the run before `closed`: one sponge query settles the
keys of one chain, and their addresses have the same key of each stage (`resolveAt_chain`).
`pub0` is the public input of `resolveAt`, a script of `layout20` with the work bits of the model
and a sponge with no event. -/
theorem closedProver20_spread {pub0 : Public} (pages : Public)
    (erased : pub0.ops.toList.map Op.erase = skeleton (layout20 0 0)) (bits : Bits20 pub0)
    (fresh : pub0.sponge.events = []) (stage : ℕ) (aux : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32) {Result : Type}
    {run : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32)
      (ByteArray × Result)} {queries digests : ℕ} (bounded : run.Bound queries digests) :
    (tower20 (E := E) pages block block₁ block₂ block₃ block₄ block₅ blockL total
      stage).body.protocol.Spread
      (closedProver20 block block₁ block₂ block₃ block₄ block₅ blockL pub0 aux hash run)
      queries := by
  rw [tower20_protocol]
  refine Protocol.Spread.of_card _ _ fun table level => ?_
  unfold closedProver20
  rw [OracleComp.queries_map]
  exact Simulate.compDeps_closed_classes (resolveAt total pub0) (deps20 S)
    (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux hash
    (resolveAt_local bits total) (deps20_closed S) (Address.trim level)
    (fun fs record point _ _ _ _ _ _ first second resolved resolved' trimmed trimmed' =>
      resolveAt_chain erased bits fresh total fs record point first second resolved resolved'
        trimmed trimmed')
    bounded

section Bound

variable [Nonempty Coord]

/-- `verifyProof_bound20_of_staged` with a guard and with a hash for each table: the verifier
runs with the sponge `sponge table` and the hash `digest table`, the event has the property
`good` of the table, and `reads` is a hypothesis only for a table with this property. The
probability that `verifyProof` accepts, the table is good and the extractor returns no output is
at most `(Q + 1)·(317479/(1024·|E|) + 135/q²) + (Q + 1)·142/budget`, for a prover with at most `Q`
different queried keys of each stage (`Checkpoint.Protocol.Spread`). -/
theorem verifyProof_bound20_guarded (pub : Public) {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice) {full : Array Fq}
    (terminalSetup : pub.sealed.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.sealed.setupFor (fold20 ℓ).setupCount).isSome)
    (root : pub.root = root20)
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord)
    (inside : 472 < total) (kept : Key20.Kept)
    (sponge : (Addr → Reply) → Duplex) (digest : (Addr → Reply) → ByteArray → ByteArray)
    (bytes : (Addr → Reply) → ByteArray)
    (prover : OracleComp Addr Reply (ℕ → Key20.Wire E)) {queries : ℕ}
    (spread : (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
      472).body.protocol.Spread prover queries) {budget : ℕ} (positive : 0 < budget)
    (good : (Addr → Reply) → Prop)
    (reads : ∀ table,
      verifyProof ((pub.withSponge (sponge table)).withDigest (digest table)) (bytes table) =
        .verdict .accept → good table →
      ∃ (messages : Array Message) (rest : ℕ) (tLast : ℕ → Rq Key20.terminal.innerDegree)
        (eLast : Rq Key20.terminal.innerDegree) (answer : List Duplex.Event → ℕ → ByteArray),
        decode pub.ops (bytes table) = .ok (messages, rest) ∧
        (sponge table).ideal = some answer ∧
        Last20 (E := E) ((pub.withSponge (sponge table)).withDigest (digest table)).sealed
          (bytes table) messages tLast eLast ∧
        473 ≤ ((tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
          472).body.protocol.final prover table).rest.length ∧
        (∀ k ≤ 473, ((tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL
            total 472).body.protocol.final prover table).wire k =
          stagePage20 ((pub.withSponge (sponge table)).withDigest (digest table)).sealed
            (bytes table) messages tLast eLast k) ∧
        Readings20 ((pub.withSponge (sponge table)).withDigest (digest table)).sealed
          (bytes table) messages answer block block₁ block₂ block₃ block₄ block₅ blockL
          ((tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
            472).body.protocol.final prover table)) :
    𝔼 table, 𝔼 tape : Tape Addr (Option Coord) (CodeSlot E) budget
          (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
            472).body.protocol.stages,
        (if verifyProof ((pub.withSponge (sponge table)).withDigest (digest table))
            (bytes table) = .verdict .accept ∧
          good table ∧
          Checkpoint.extract (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅
            blockL total 472).body prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (317479 / (1024 * Fintype.card E) + 135 * (1 / (fieldModulus : ℝ) ^ 2)) +
        (queries + 1) * (142 : ℕ) / budget := by
  refine le_trans (Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun tape _ => ?_)
    (Key20.Tower.extraction_spread_le
      (tower20_drawn pub.sealed injective injective₁ injective₂ injective₃ injective₄ injective₅
        injectiveL inside kept) slots prover spread positive)
  by_cases event :
      verifyProof ((pub.withSponge (sponge table)).withDigest (digest table)) (bytes table) =
        .verdict .accept ∧
      good table ∧
      Checkpoint.extract (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅
        blockL total 472).body prover budget table tape = none
  · obtain ⟨messages, rest, tLast, eLast, answer, decoded, ideal, last, long, wires, readings⟩ :=
      reads table event.1 event.2.1
    obtain ⟨points, values, -⟩ := shaped20 (pub := pub.withSponge (sponge table)) root
      ((verifyProof_accept_iff _ _).mp event.1).1
    have ran := ((verifyProof_accept_iff _ _).mp event.1).2
    have accepts := presents20_accepts
      ((pub.withSponge (sponge table)).withDigest (digest table)).sealed (bytes table)
      messages tLast eLast layout ran terminalSetup expanded decoded points values last
      (presents20_of_agrees_staged
        ((pub.withSponge (sponge table)).withDigest (digest table)).sealed (bytes table) messages
        tLast eLast layout ran terminalSetup decoded long wires
        (agrees20_of_reads_staged
          ((pub.withSponge (sponge table)).withDigest (digest table)).sealed (bytes table)
          messages tLast eLast layout ran terminalSetup expanded decoded points values ideal long
          wires readings))
    have model : (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
        472).accepts ((tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL
          total 472).body.protocol.final prover table) ∧
        Checkpoint.extract (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅
          blockL total 472).body prover budget table tape = none := ⟨accepts, event.2.2⟩
    rw [ite_eq_left event, ite_eq_left model]
  · rw [ite_eq_right event]
    split_ifs <;> norm_num

/-- `verifyProof_bound20_guarded` for the recording prover on the experiment, for each `aux`
and each digest function: the probability over the table and the tape that the verifier accepts
the bytes of the adversary, no bad event occurs and the extractor returns no output from
`closedProver20` is at most the bound of `Key20.Tower.extraction_spread_le` for `Q + 473`
different keys of each stage. `scripted`: the layout takes the plan. -/
theorem guarded20 {zMax rice : ℕ} (layout : pubV.layout = layout20 zMax rice)
    (scripted : script pubV.layout pubV.plan = .ok pubV.ops) {full : Array Fq}
    (terminalSetup : pubV.sealed.setupFor pubV.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pubV.sealed.setupFor (fold20 ℓ).setupCount).isSome)
    (root : pubV.root = root20)
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord)
    (inside : 472 < total) (kept : Key20.Kept) (sizes : drawBytes20 ≤ C) (three : 3 ≤ S)
    (fits : 4128 ≤ D) {queries digests : ℕ} (bounded : adversary.Bound queries digests)
    {budget : ℕ} (positive : 0 < budget) (aux : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32) :
    𝔼 table : Addr → Reply,
        𝔼 tape : Tape Addr (Option Coord) (CodeSlot E) budget
          (tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
            472).body.protocol.stages,
        (if accepted20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux
            hash ∧
          ¬ bad20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux hash ∧
          Checkpoint.extract
            (tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
              472).body
            (closedProver20 block block₁ block₂ block₃ block₄ block₅ blockL
              (pubV.withSponge Duplex.empty).sealed aux hash (game20 pubV adversary))
            budget table tape = none then (1 : ℝ) else 0) ≤
      ((queries + 473 : ℕ) + 1) *
          (317479 / (1024 * Fintype.card E) + 135 * (1 / (fieldModulus : ℝ) ^ 2)) +
        ((queries + 473 : ℕ) + 1) * (142 : ℕ) / budget := by
  obtain ⟨erased, bits⟩ := scripted20 layout scripted
  have core := verifyProof_bound20_guarded (E := E) (total := total) block block₁ block₂ block₃
    block₄ block₅ blockL pubV layout terminalSetup expanded root injective injective₁ injective₂
    injective₃ injective₄ injective₅ injectiveL slots inside kept
    (fun table => spongeOf (seen table aux hash)) (fun _ => digestOf hash)
    (fun table => adversary.eval (seen table aux hash) hash)
    (closedProver20 block block₁ block₂ block₃ block₄ block₅ blockL
      (pubV.withSponge Duplex.empty).sealed aux hash (game20 pubV adversary))
    (closedProver20_spread block block₁ block₂ block₃ block₄ block₅ blockL pubV.sealed erased
      bits (fresh20 pubV) 472 aux hash (game20_bound layout bounded))
    positive
    (fun table =>
      ¬ bad20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux hash)
    (fun table accepted clean =>
      reads20_of_good block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary layout
        root terminalSetup expanded injective injective₁ injective₂ injective₃ injective₄
        injective₅ injectiveL slots inside sizes three fits table aux hash accepted
        (not_or.mp clean).1 (not_or.mp clean).2)
  refine le_trans (le_of_eq ?_) core
  exact Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun tape _ =>
    indicator_congr Iff.rfl

/-- The soundness bound of the claims for `key20.aks` on the byte verifier, under the ideal
sponge and the ideal digest (Akita Theorem 10.17 with Lemma 10.47 and (213)). For `layout20`,
the statement shape `root20`, the setup facts of the seed, and each adversary with at most `Q`
queries to the sponge and `Q_dig` queries to the digest oracle: in the recording prover with a
uniform `table`, a uniform `aux` and a uniform digest function, the probability that the
verifier accepts the bytes of the adversary and the extractor of `Key20.Tower.body` returns no
output from `closedProver20` is at most
`(Q + 474)·(317479/(1024·|E|) + 135/q²) + (Q + 474)·142/budget +
(C(Q_dig + 2, 2) + 2·(Q + 473)·(Q_dig + 2))/|Bytes 32|`. The cap `L` of the events of a sponge
key is in no number of the bound. No hypothesis is about the prover, the transcript, the readings
or the work bits. -/
theorem verifyProof_sound20 {zMax rice : ℕ} (layout : pubV.layout = layout20 zMax rice)
    {full : Array Fq}
    (terminalSetup : pubV.sealed.setupFor pubV.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pubV.sealed.setupFor (fold20 ℓ).setupCount).isSome)
    (root : pubV.root = root20)
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord)
    (inside : 472 < total) (kept : Key20.Kept) (sizes : drawBytes20 ≤ C) (three : 3 ≤ S)
    (fits : 4128 ≤ D) {queries digests : ℕ} (bounded : adversary.Bound queries digests)
    {budget : ℕ} (positive : 0 < budget) :
    𝔼 table : Addr → Reply, 𝔼 aux : Sponge.Base L B C → Sponge.Y S C,
        𝔼 hash : Sponge.Str D → Bytes 32,
        𝔼 tape : Tape Addr (Option Coord) (CodeSlot E) budget
          (tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
            472).body.protocol.stages,
        (if accepted20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux
            hash ∧
          Checkpoint.extract
            (tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
              472).body
            (closedProver20 block block₁ block₂ block₃ block₄ block₅ blockL
              (pubV.withSponge Duplex.empty).sealed aux hash (game20 pubV adversary))
            budget table tape = none then (1 : ℝ) else 0) ≤
      ((queries + 473 : ℕ) + 1) *
          (317479 / (1024 * Fintype.card E) + 135 * (1 / (fieldModulus : ℝ) ^ 2)) +
        ((queries + 473 : ℕ) + 1) * (142 : ℕ) / budget +
        (((digests + 2).choose 2 + 2 * (queries + 473) * (digests + 2) : ℕ) : ℝ) /
          Fintype.card (Bytes 32) := by
  by_cases scripted : script pubV.layout pubV.plan = .ok pubV.ops
  · have rare : 𝔼 table : Addr → Reply, 𝔼 aux : Sponge.Base L B C → Sponge.Y S C,
        𝔼 hash : Sponge.Str D → Bytes 32,
        (if bad20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux hash
          then (1 : ℝ) else 0) ≤
        (((digests + 2).choose 2 + 2 * (queries + 473) * (digests + 2) : ℕ) : ℝ) /
          Fintype.card (Bytes 32) := by
      have core := bad20_closed_le (E := E) (total := total) block block₁ block₂ block₃ block₄
        block₅ blockL pubV adversary layout scripted (Nat.le_of_lt inside) bounded
      refine le_trans (le_of_eq ?_) core
      refine Eq.trans ?_ (expect_fintype (T := Addr → Reply) inferInstance _ _)
      exact Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun aux _ =>
        Finset.expect_congr rfl fun hash _ => indicator_congr Iff.rfl
    exact expect_union_le (T := Addr → Reply) (A := Sponge.Base L B C → Sponge.Y S C)
      (H := Sponge.Str D → Bytes 32)
      (P := Tape Addr (Option Coord) (CodeSlot E) budget
        (tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
          472).body.protocol.stages)
      (guarded := fun table aux hash tape =>
        accepted20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux
            hash ∧
          ¬ bad20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux hash ∧
          Checkpoint.extract
            (tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
              472).body
            (closedProver20 block block₁ block₂ block₃ block₄ block₅ blockL
              (pubV.withSponge Duplex.empty).sealed aux hash (game20 pubV adversary))
            budget table tape = none)
      (bad := fun table aux hash =>
        bad20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux hash)
      (fun aux hash => guarded20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary
        layout scripted terminalSetup expanded root injective injective₁ injective₂ injective₃
        injective₄ injective₅ injectiveL slots inside kept sizes three fits bounded positive aux
        hash)
      rare fun _ _ _ _ event clean => ⟨event.1, clean, event.2⟩
  · have never : ∀ (table : Addr → Reply) (aux : Sponge.Base L B C → Sponge.Y S C)
        (hash : Sponge.Str D → Bytes 32),
        ¬ accepted20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux
          hash :=
      fun _ _ _ accepted => by
        have taken := scripted_of_accepted accepted
        exact scripted taken
    refine le_trans (Simulate.expect_le_const _ 0 fun table => Simulate.expect_le_const _ 0
      fun aux => Simulate.expect_le_const _ 0 fun hash => Simulate.expect_le_const _ 0
        fun tape => le_of_eq (ite_eq_right fun both => never table aux hash both.1)) ?_
    positivity

/-- `verifyProof_sound20` for the experiment with a uniform table of the sponge and a uniform
digest function: the probability that `verifyProof` accepts the bytes of the adversary is at most
the probability, in the recording prover, that the verifier accepts and the extractor of
`Key20.Tower.body` returns an output from `closedProver20`, plus the bound of
`verifyProof_sound20`. -/
theorem verifyProof_sound20_real {zMax rice : ℕ} (layout : pubV.layout = layout20 zMax rice)
    {full : Array Fq}
    (terminalSetup : pubV.sealed.setupFor pubV.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pubV.sealed.setupFor (fold20 ℓ).setupCount).isSome)
    (root : pubV.root = root20)
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord)
    (inside : 472 < total) (kept : Key20.Kept) (sizes : drawBytes20 ≤ C) (three : 3 ≤ S)
    (fits : 4128 ≤ D) {queries digests : ℕ} (bounded : adversary.Bound queries digests)
    {budget : ℕ} (positive : 0 < budget) :
    𝔼 fs : Sponge.Base L B C → Sponge.Y S C, 𝔼 hash : Sponge.Str D → Bytes 32,
        (if verifyProof (pubV.withOracles fs hash) (adversary.eval fs hash) = .verdict .accept
          then (1 : ℝ) else 0) ≤
      (𝔼 table : Addr → Reply, 𝔼 aux : Sponge.Base L B C → Sponge.Y S C,
        𝔼 hash : Sponge.Str D → Bytes 32,
        𝔼 tape : Tape Addr (Option Coord) (CodeSlot E) budget
          (tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
            472).body.protocol.stages,
        (if accepted20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux
            hash ∧
          Checkpoint.extract
            (tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
              472).body
            (closedProver20 block block₁ block₂ block₃ block₄ block₅ blockL
              (pubV.withSponge Duplex.empty).sealed aux hash (game20 pubV adversary))
            budget table tape ≠ none then (1 : ℝ) else 0)) +
        (((queries + 473 : ℕ) + 1) *
            (317479 / (1024 * Fintype.card E) + 135 * (1 / (fieldModulus : ℝ) ^ 2)) +
          ((queries + 473 : ℕ) + 1) * (142 : ℕ) / budget +
          (((digests + 2).choose 2 + 2 * (queries + 473) * (digests + 2) : ℕ) : ℝ) /
            Fintype.card (Bytes 32)) := by
  refine le_trans (le_of_eq (closedGame20_law (E := E) (total := total) block block₁ block₂
    block₃ block₄ block₅ blockL pubV adversary layout (Nat.le_of_lt inside)).symm) ?_
  refine le_trans (le_of_eq (expect_fintype _ (inferInstance : Fintype (Addr → Reply)) _)) ?_
  exact expect_cover_le (T := Addr → Reply) (A := Sponge.Base L B C → Sponge.Y S C)
    (H := Sponge.Str D → Bytes 32)
    (P := Tape Addr (Option Coord) (CodeSlot E) budget
      (tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
        472).body.protocol.stages)
    (accept := fun table aux hash =>
      accepted20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary table aux hash)
    (verifyProof_sound20 block block₁ block₂ block₃ block₄ block₅ blockL pubV adversary layout
      terminalSetup expanded root injective injective₁ injective₂ injective₃ injective₄
      injective₅ injectiveL slots inside kept sizes three fits bounded positive)
    fun table aux hash tape accepted => by
      by_cases lost : Checkpoint.extract
          (tower20 (E := E) pubV.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
            472).body
          (closedProver20 block block₁ block₂ block₃ block₄ block₅ blockL
            (pubV.withSponge Duplex.empty).sealed aux hash (game20 pubV adversary))
          budget table tape = none
      · exact Or.inr ⟨accepted, lost⟩
      · exact Or.inl ⟨accepted, lost⟩

end Bound

end Sound

section Guard

/-- The hypotheses `terminalSetup` and `expanded` of `verifyProof_sound20` are facts of the seed:
the public input without the setup prefix of the caller has the prefix that `Setup.expand` gives
from the seed. -/
theorem sealed_setupFor (pub : Public) {count : ℕ} {full : Array Fq}
    (expanded : Setup.expand pub.seed count = some full) (positive : 0 < count) :
    pub.sealed.setupFor count = some full := by
  unfold Public.setupFor
  exact (ite_eq_right (Nat.not_le.mpr positive)).trans expanded

/-- A type of coordinates for the answers of the model: one coordinate for each fold block of
the root, of the levels 1 to 5 and of the terminal of `key20.aks`, and 22 more coordinates. -/
abbrev Coord20 : Type :=
  (Key20.root.Block ⊕ Fin Key20.level₁.shape.blocks ⊕ Fin Key20.level₂.shape.blocks ⊕
    Fin Key20.level₃.shape.blocks ⊕ Fin Key20.level₄.shape.blocks ⊕
    Fin Key20.level₅.shape.blocks ⊕ Fin Key20.terminal.blocks) ⊕ Fin 22

/-- The hypotheses of `verifyProof_sound20` on the block maps, on the coordinates, on the stages
and on the sizes of the oracles hold for `Coord20` with its injections, 473 stages, squeezes of
5608 bytes, 3 squeezes after an absorb and digest inputs of 4128 bytes; and the adversary that
gives fixed bytes has no query: the guard against a vacuous statement. -/
theorem sound20_guard {L B C S D : ℕ} :
    Function.Injective (fun b : Key20.root.Block => (.inl (.inl b) : Coord20)) ∧
      Function.Injective
        (fun b : Fin Key20.level₁.shape.blocks => (.inl (.inr (.inl b)) : Coord20)) ∧
      Function.Injective
        (fun b : Fin Key20.level₂.shape.blocks => (.inl (.inr (.inr (.inl b))) : Coord20)) ∧
      Function.Injective
        (fun b : Fin Key20.level₃.shape.blocks =>
          (.inl (.inr (.inr (.inr (.inl b)))) : Coord20)) ∧
      Function.Injective
        (fun b : Fin Key20.level₄.shape.blocks =>
          (.inl (.inr (.inr (.inr (.inr (.inl b))))) : Coord20)) ∧
      Function.Injective
        (fun b : Fin Key20.level₅.shape.blocks =>
          (.inl (.inr (.inr (.inr (.inr (.inr (.inl b)))))) : Coord20)) ∧
      Function.Injective
        (fun b : Fin Key20.terminal.blocks =>
          (.inl (.inr (.inr (.inr (.inr (.inr (.inr b)))))) : Coord20)) ∧
      22 ≤ Fintype.card Coord20 ∧ 472 < 473 ∧ drawBytes20 ≤ 5608 ∧ 3 ≤ 3 ∧ 4128 ≤ 4128 ∧
      ∀ proof : ByteArray,
        (Digest.Run.done proof : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D)
          (Bytes 32) ByteArray).Bound 0 0 := by
  refine ⟨fun _ _ same => Sum.inl.inj (Sum.inl.inj same),
    fun _ _ same => Sum.inl.inj (Sum.inr.inj (Sum.inl.inj same)),
    fun _ _ same => Sum.inl.inj (Sum.inr.inj (Sum.inr.inj (Sum.inl.inj same))),
    fun _ _ same => Sum.inl.inj (Sum.inr.inj (Sum.inr.inj (Sum.inr.inj (Sum.inl.inj same)))),
    fun _ _ same =>
      Sum.inl.inj (Sum.inr.inj (Sum.inr.inj (Sum.inr.inj (Sum.inr.inj (Sum.inl.inj same))))),
    fun _ _ same =>
      Sum.inl.inj (Sum.inr.inj (Sum.inr.inj (Sum.inr.inj (Sum.inr.inj (Sum.inr.inj
        (Sum.inl.inj same)))))),
    fun _ _ same =>
      Sum.inr.inj (Sum.inr.inj (Sum.inr.inj (Sum.inr.inj (Sum.inr.inj (Sum.inr.inj
        (Sum.inl.inj same)))))),
    ?_, Nat.lt_succ_self 472, Nat.le_refl 5608, Nat.le_refl 3, Nat.le_refl 4128,
    fun proof => .done proof 0 0⟩
  rw [Fintype.card_sum, Fintype.card_fin]
  exact Nat.le_add_left 22 _

end Guard

end NightstreamFPrime.Spec.Compression.Akita.Replay
