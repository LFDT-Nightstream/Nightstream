import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinStage
import NightstreamFPrime.Spec.Compression.Akita.Replay.Statement

/-! The bound of the claims for `key20.aks` on the byte verifier with an ideal sponge (Akita
revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23, Theorem 10.17 and Section 10.3;
`Key20.Tower.extraction_le` of `Key20Opening.lean`; `SECURITY_MODEL.md`, `[KECCAK]`;
`docs/akita-work/w17/design.md` section 3).

- `verifyProof_accepts20`: the join for the verifier `verifyProof` with its checks of the
  statement: for the layout and the statement shape `root20` of `key20.aks`, an accepted proof has
  last blocks of the code and a transcript of `tower20` that presents its bytes, whose wire
  message `k ≤ 473` has the `Note` of the state before the work of the stage `k`, and that
  `Key20.Tower.accepts` accepts. The shape of the statement is a result of the acceptance.
- `Public.withSponge pub sponge` is the public input with the sponge `sponge` before the channel.
  With `Duplex.ideal = some answer`, `verifyProof (pub.withSponge sponge)` is the verifier of the
  code with each squeeze replaced by the answer of `answer` for the events before it.
- `verifyProof_bound20`: a prover of the model, a sponge for each table of the oracle and proof
  bytes for each table. Hypothesis `carries`: for each table, when `verifyProof` with the sponge
  of the table accepts the bytes of the table, the transcript of the prover with the answers of
  the table presents these bytes (`Presents20`, with last blocks `Last20`). Then the probability
  over the uniform table and the tape that `verifyProof` accepts and the extractor of
  `Key20.Tower.body` returns no output is at most the bound of `Key20.Tower.extraction_le` for the
  query count of the prover.
- `carries` is a hypothesis, not a theorem: Lean does not construct, for an adversary with access
  to an ideal sponge, the prover and the sponge of each table. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Prefix Address)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

attribute [local instance low] Classical.propDecidable

/-- The public input with the sponge `sponge` before the channel absorbs the session. -/
def Public.withSponge (pub : Public) (sponge : Duplex) : Public := { pub with sponge }

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] [Nonempty Coord] {total : ℕ}
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

omit [Nonempty Coord] in
/-- The join for `verifyProof` and `key20.aks`: with the layout and the statement shape of the
schedule row, an accepted proof has last blocks `Last20` and a transcript of `tower20` that
presents its bytes, with the `Note` `noteAt20` of the stage in each wire message `k ≤ 473`, and
`Key20.Tower.accepts` accepts it. `terminalSetup` and `expanded` are facts of the seed. -/
theorem verifyProof_accepts20 {pub : Public} {proof : ByteArray} {messages : Array Message}
    {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice) (root : pub.root = root20)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.sealed.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.sealed.setupFor (fold20 ℓ).setupCount).isSome) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord)
    (inside : 473 ≤ total) :
    ∃ (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
      (point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total),
      Last20 (E := E) pub.sealed proof messages tLast eLast ∧
        Presents20 pub.sealed proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅
          blockL point ∧
        (∀ k ≤ 473, (point.wire k).data = noteAt20 pub.sealed proof messages tLast eLast k) ∧
        (tower20 pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total 472).accepts
          point := by
  obtain ⟨shaped, ran⟩ := (verifyProof_accept_iff pub proof).mp accepted
  obtain ⟨points, values, -⟩ := shaped20 root shaped
  exact verifyProof_exists_accepts20_staged pub.sealed proof messages layout ran terminalSetup
    expanded decoded points values injective injective₁ injective₂ injective₃ injective₄
    injective₅ injectiveL slots inside

/-- The bound of `Key20.Tower.extraction_le` for the byte verifier with the sponge of the table
(Akita Theorem 10.17 in the form (161)). `carries`: when `verifyProof` with the sponge of a table
accepts the bytes of the table, the transcript of `prover` with the answers of the table presents
these bytes with last blocks of the code. The probability that `verifyProof` accepts and the
extractor returns no output is at most
`(Q + 1)·(317479/(1024·|E|) + 135/q²) + (Q + 1)·142/budget`. -/
theorem verifyProof_bound20 (pub : Public) {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice) {full : Array Fq}
    (terminalSetup : pub.sealed.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.sealed.setupFor (fold20 ℓ).setupCount).isSome)
    (root : pub.root = root20)
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord)
    (inside : 472 < total) (kept : Key20.Kept)
    (sponge : (Address (Key20.Wire E) total → Option Coord → CodeSlot E) → Duplex)
    (bytes : (Address (Key20.Wire E) total → Option Coord → CodeSlot E) → ByteArray)
    (prover : OracleComp (Address (Key20.Wire E) total) (Option Coord → CodeSlot E)
      (ℕ → Key20.Wire E)) {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ}
    (positive : 0 < budget)
    (carries : ∀ table, verifyProof (pub.withSponge (sponge table)) (bytes table) =
        .verdict .accept →
      ∃ (messages : Array Message) (rest : ℕ) (tLast : ℕ → Rq Key20.terminal.innerDegree)
        (eLast : Rq Key20.terminal.innerDegree),
        decode pub.ops (bytes table) = .ok (messages, rest) ∧
        Last20 (E := E) (pub.withSponge (sponge table)).sealed (bytes table) messages tLast
          eLast ∧
        Presents20 (pub.withSponge (sponge table)).sealed (bytes table) messages tLast eLast block
          block₁ block₂ block₃ block₄ block₅ blockL
          ((tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
            472).body.protocol.final prover table)) :
    𝔼 table, 𝔼 tape : Tape (Address (Key20.Wire E) total) (Option Coord) (CodeSlot E) budget
          (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
            472).body.protocol.stages,
        (if verifyProof (pub.withSponge (sponge table)) (bytes table) = .verdict .accept ∧
          Checkpoint.extract (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅
            blockL total 472).body prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (317479 / (1024 * Fintype.card E) + 135 * (1 / (fieldModulus : ℝ) ^ 2)) +
        (queries + 1) * (142 : ℕ) / budget := by
  refine le_trans (Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun tape _ => ?_)
    (Key20.Tower.extraction_le
      (tower20_drawn pub.sealed injective injective₁ injective₂ injective₃ injective₄ injective₅
        injectiveL inside kept) slots prover bounded positive)
  by_cases event : verifyProof (pub.withSponge (sponge table)) (bytes table) = .verdict .accept ∧
      Checkpoint.extract (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅
        blockL total 472).body prover budget table tape = none
  · obtain ⟨messages, rest, tLast, eLast, decoded, last, presents⟩ := carries table event.1
    obtain ⟨points, values, -⟩ := shaped20 (pub := pub.withSponge (sponge table)) root
      ((verifyProof_accept_iff _ _).mp event.1).1
    have accepts := presents20_accepts (pub.withSponge (sponge table)).sealed (bytes table)
      messages tLast eLast layout ((verifyProof_accept_iff _ _).mp event.1).2 terminalSetup
      expanded decoded points values last presents
    have model : (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
        472).accepts ((tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL
          total 472).body.protocol.final prover table) ∧
        Checkpoint.extract (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅
          blockL total 472).body prover budget table tape = none := ⟨accepts, event.2⟩
    simp only [event, model, and_self, ↓reduceIte, le_refl]
  · simp only [event, ↓reduceIte]
    split_ifs <;> norm_num

end NightstreamFPrime.Spec.Compression.Akita.Replay
