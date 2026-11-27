import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Closed
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Digest
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinStagePage
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageDigest

/-! The wire messages of the recording prover of `key20.aks` and the transcript of a prover
(Akita revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23, Lemma 10.47, the paragraph
"The prover P", and Section 10.3; `docs/akita-work/w22/design.md`, section "The assembly";
`Replay/Sound20Game.lean`, `Replay/Resolve.lean`, `Replay/SpongeFreeBases.lean`,
`Replay/JoinStagePage.lean`).

- `OracleComp.map`, with `run_map` and `queries_map`.
  `Digest.Run.answered_of_trace`: each query to `H_FS` is an answered query.
- `wireAddress proof stages k`: the messages `0, …, k` of a proof as a key. `path_parts`,
  `final_length`, `final_wire`, `final_reply`: the transcript of a prover has each stage, its
  wire message `k` is the message `k` of the proof, and its answer `k` is the answer of the
  table at the key of the messages `0, …, k`. `final_length20`, `final_wire20`, `final_reply20`:
  the same for `tower20`, with the key `addressAt20` for the messages `stagePage20`.
- `logTable`: the table of the answered queries of a run. `messagesOf`: the decoded messages.
  `wires20`: the wire messages of an output of the recording prover. `closedProver20`: the
  recording prover on a closed run with these wire messages as its output.
- `lastEvents20`, `squeeze_last20`, `agreeOn_last20`: each squeeze before the terminal is a
  squeeze of a stage `j < 473`. `stagePage20_tables`: each page and the two digests are the same
  for two tables with the same answers at the base of each stage. `stagePage20_found`: the
  pages with the blocks of a record are the pages with the last blocks of the code.
- `wires20_eq`: for an accepted run, the wire messages `wires20` are the pages `stagePage20` of
  the run with the last blocks of the code. `wires20_guard`: the hypotheses of `wires20_eq` on
  the answered queries and on the two digest queries hold for the queries of the verifier. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.RandomOracle.OracleComp

variable {Point Answer Output Other : Type}

/-- The algorithm with the function `f` on its output. -/
def map (f : Output → Other) : OracleComp Point Answer Output → OracleComp Point Answer Other
  | .done output => .done (f output)
  | .query point next => .query point fun answer => map f (next answer)

theorem run_map (f : Output → Other) (oracle : Point → Answer)
    (computation : OracleComp Point Answer Output) :
    (computation.map f).run oracle = f (computation.run oracle) := by
  induction computation with
  | done output => rfl
  | query point next induction => exact induction (oracle point)

theorem queries_map (f : Output → Other) (oracle : Point → Answer)
    (computation : OracleComp Point Answer Output) :
    (computation.map f).queries oracle = computation.queries oracle := by
  induction computation with
  | done output => rfl
  | query point next induction => exact congrArg (point :: ·) (induction (oracle point))

end NightstreamFPrime.Spec.RandomOracle.OracleComp

namespace NightstreamFPrime.Spec.Compression.Akita.Digest.Run

variable {Fs Challenge Msg Digest Output : Type}

/-- Each query of a run to `H_FS` is in its answered queries with its answer. -/
theorem answered_of_trace {fs : Fs → Challenge} {hash : Msg → Digest}
    {run : Run Fs Challenge Msg Digest Output} {key : Fs}
    (queried : Sum.inl key ∈ run.trace fs hash) :
    ∃ value, Sum.inl (key, value) ∈ run.answered fs hash := by
  rw [← map_answered] at queried
  obtain ⟨entry, inside, mapped⟩ := List.mem_map.mp queried
  rcases entry with ⟨point, value⟩ | pair
  · obtain rfl : point = key := Sum.inl.inj mapped
    exact ⟨value, inside⟩
  · cases mapped

end NightstreamFPrime.Spec.Compression.Akita.Digest.Run

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

variable {Wire Reply : Type} {stages : ℕ}

/-- The key of the stage `k` of a proof: its messages `0, …, k`. -/
def wireAddress (proof : ℕ → Wire) (stages k : ℕ) : Address Wire stages :=
  ⟨proof 0, (List.range (min k stages)).map fun j => proof (j + 1), by
    rw [List.length_map, List.length_range]
    exact Nat.min_le_right k stages⟩

theorem wireAddress_congr {proof other : ℕ → Wire} {k : ℕ}
    (same : ∀ j ≤ k, proof j = other j) (stages : ℕ) :
    wireAddress proof stages k = wireAddress other stages k := by
  refine Address.ext (same 0 (Nat.zero_le k)) ?_
  show (List.range (min k stages)).map _ = (List.range (min k stages)).map _
  refine List.map_congr_left fun j member => same (j + 1) ?_
  have below := List.mem_range.mp member
  omega

variable [Nonempty Reply] {accept : Prefix Wire Reply stages → Prop}

/-- The stage input of the stage `level` of a proof for a key table: the first message, then for
each earlier stage the answer of the table at its key and the next message. -/
theorem path_parts (proof : ℕ → Wire) (table : Address Wire stages → Reply) :
    ∀ level, level ≤ stages →
      ((transcript stages accept).path proof table level).first = proof 0 ∧
        ((transcript stages accept).path proof table level).rest =
          (List.range level).map fun j => (table (wireAddress proof stages j), proof (j + 1))
  | 0, _ => ⟨rfl, rfl⟩
  | level + 1, inside => by
      obtain ⟨first, rest⟩ := path_parts proof table level (Nat.le_of_succ_le inside)
      have length : ((transcript stages accept).path proof table level).rest.length = level := by
        rw [rest, List.length_map, List.length_range]
      have room : ((transcript stages accept).path proof table level).rest.length < stages :=
        length.trans_lt inside
      have address : ((transcript stages accept).path proof table level).address =
          wireAddress proof stages level := by
        refine Address.ext first ?_
        show ((transcript stages accept).path proof table level).rest.map Prod.snd =
          (List.range (min level stages)).map fun j => proof (j + 1)
        rw [rest, List.map_map, Nat.min_eq_left (Nat.le_of_succ_le inside)]
        rfl
      have step : (transcript stages accept).path proof table (level + 1) =
          ((transcript stages accept).path proof table level).snoc
            (table ((transcript stages accept).path proof table level).address,
              proof (((transcript stages accept).path proof table level).rest.length + 1)) room :=
        transcript_step proof _ _ room
      rw [step]
      refine ⟨first, ?_⟩
      show ((transcript stages accept).path proof table level).rest ++
          [(table ((transcript stages accept).path proof table level).address,
            proof (((transcript stages accept).path proof table level).rest.length + 1))] = _
      rw [length, address, rest, List.range_succ, List.map_append]
      rfl

variable (prover : OracleComp (Address Wire stages) Reply (ℕ → Wire))
  (table : Address Wire stages → Reply)

/-- The transcript of a prover has each stage. -/
theorem final_length : ((transcript stages accept).final prover table).rest.length = stages :=
  (congrArg List.length
    (path_parts (accept := accept) (prover.run table) table stages (Nat.le_refl _)).2).trans
    (by rw [List.length_map, List.length_range])

/-- The wire message `k` of the transcript of a prover is the message `k` of its proof. -/
theorem final_wire {k : ℕ} (le : k ≤ stages) :
    ((transcript stages accept).final prover table).wire k = prover.run table k := by
  obtain ⟨first, rest⟩ :=
    path_parts (accept := accept) (prover.run table) table stages (Nat.le_refl _)
  cases k with
  | zero => exact first
  | succ k =>
      show (((transcript stages accept).path (prover.run table) table stages).rest[k]?.map
        Prod.snd).getD _ = _
      rw [rest, List.getElem?_map, List.getElem?_range le]
      rfl

/-- The answer `k` of the transcript of a prover is the answer of the table at the key of the
messages `0, …, k` of its proof. -/
theorem final_reply {k : ℕ} (below : k < stages) :
    ((transcript stages accept).final prover table).reply k =
      some (table (wireAddress (prover.run table) stages k)) := by
  show ((transcript stages accept).path (prover.run table) table stages).rest[k]?.map Prod.fst = _
  rw [(path_parts (accept := accept) (prover.run table) table stages (Nat.le_refl _)).2,
    List.getElem?_map, List.getElem?_range below]
  rfl

end NightstreamFPrime.Spec.Compression.Akita.Extract

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Prefix Address wireAddress)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)
open Sponge (split answerOf spongeOf)
open Replayer

section Transcript

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  (pub : Public) (block : Key20.root.Block → Coord)
  (block₁ : Fin Key20.level₁.shape.blocks → Coord) (block₂ : Fin Key20.level₂.shape.blocks → Coord)
  (block₃ : Fin Key20.level₃.shape.blocks → Coord) (block₄ : Fin Key20.level₄.shape.blocks → Coord)
  (block₅ : Fin Key20.level₅.shape.blocks → Coord) (blockL : Fin Key20.terminal.blocks → Coord)
  (stage : ℕ)
  (prover : OracleComp (Address (Key20.Wire E) total) (Option Coord → CodeSlot E)
    (ℕ → Key20.Wire E))
  (table : Address (Key20.Wire E) total → Option Coord → CodeSlot E)

/-- The protocol of `tower20` is the protocol of the stage inputs `Prefix`. -/
theorem tower20_protocol :
    (tower20 (E := E) pub block block₁ block₂ block₃ block₄ block₅ blockL total stage).body.protocol =
      Extract.transcript total
        (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total stage).accepts :=
  rfl

/-- The transcript of a prover for `tower20` has each stage. -/
theorem final_length20 :
    ((tower20 (E := E) pub block block₁ block₂ block₃ block₄ block₅ blockL total
      stage).body.protocol.final prover table).rest.length = total :=
  Extract.final_length
    (accept := (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total stage).accepts)
    prover table

/-- The wire message `k` of the transcript of a prover for `tower20` is the message `k` of its
proof. -/
theorem final_wire20 {k : ℕ} (le : k ≤ total) :
    ((tower20 (E := E) pub block block₁ block₂ block₃ block₄ block₅ blockL total
      stage).body.protocol.final prover table).wire k = prover.run table k :=
  Extract.final_wire
    (accept := (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total stage).accepts)
    prover table le

/-- The answer `k` of the transcript of a prover for `tower20`, whose proof has the pages
`stagePage20 … j` as its messages `j ≤ k`, is the answer of the table at the key
`addressAt20 … k`. -/
theorem final_reply20 {pages : Public} {proof : ByteArray} {messages : Array Message}
    {tLast : ℕ → Rq Key20.terminal.innerDegree} {eLast : Rq Key20.terminal.innerDegree} {k : ℕ}
    (below : k < total)
    (wires : ∀ j ≤ k, prover.run table j = stagePage20 pages proof messages tLast eLast j) :
    ((tower20 (E := E) pub block block₁ block₂ block₃ block₄ block₅ blockL total
      stage).body.protocol.final prover table).reply k =
      some (table (addressAt20 pages proof messages tLast eLast total k)) :=
  (Extract.final_reply
    (accept := (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total stage).accepts)
    prover table below).trans
    (congrArg (fun key => some (table key))
      (Extract.wireAddress_congr (other := stagePage20 pages proof messages tLast eLast) wires
        total))

end Transcript

section Wires

variable {L B C S D : ℕ}

/-- The table of the answered queries of a run: the answer of the first query at a base, and the
strings of zeros for a base with no query. -/
def logTable : List ((Sponge.Base L B C × Sponge.Y S C) ⊕ (Sponge.Str D × Bytes 32)) →
    Sponge.Base L B C → Sponge.Y S C
  | [], _ => fun _ _ => 0
  | .inl (key, value) :: rest, base => if key = base then value else logTable rest base
  | .inr _ :: rest, base => logTable rest base

/-- The table of answered queries with the answers of `fs` has the answer of `fs` at each base
with a query. -/
theorem logTable_eq {fs : Sponge.Base L B C → Sponge.Y S C} {base : Sponge.Base L B C} :
    ∀ {answered : List ((Sponge.Base L B C × Sponge.Y S C) ⊕ (Sponge.Str D × Bytes 32))},
      (∀ key value, Sum.inl (key, value) ∈ answered → value = fs key) →
      (∃ value, Sum.inl (base, value) ∈ answered) → logTable answered base = fs base
  | [], _, ⟨_, member⟩ => absurd member List.not_mem_nil
  | .inl (key, value) :: rest, answers, ⟨found, member⟩ => by
      by_cases same : key = base
      · show (if key = base then value else logTable rest base) = fs base
        rw [ite_eq_left same, ← same]
        exact answers key value List.mem_cons_self
      · show (if key = base then value else logTable rest base) = fs base
        rw [ite_eq_right same]
        refine logTable_eq (fun k v inside => answers k v (List.mem_cons_of_mem _ inside))
          ⟨found, ?_⟩
        rcases List.mem_cons.mp member with head | tail
        · exact absurd (congrArg Prod.fst (Sum.inl.inj head)).symm same
        · exact tail
  | .inr _ :: rest, answers, ⟨found, member⟩ => by
      show logTable rest base = fs base
      refine logTable_eq (fun k v inside => answers k v (List.mem_cons_of_mem _ inside))
        ⟨found, ?_⟩
      rcases List.mem_cons.mp member with head | tail
      · cases head
      · exact tail

/-- The decoded messages of the bytes of a proof, and no message when the decoding fails. -/
def messagesOf (ops : Array Op) (proof : ByteArray) : Array Message :=
  match decode ops proof with
  | .ok (messages, _) => messages
  | .error _ => #[]

theorem messagesOf_decoded {ops : Array Op} {proof : ByteArray} {messages : Array Message}
    {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest)) :
    messagesOf ops proof = messages := by
  unfold messagesOf
  rw [decoded]

/-- One digest query of a record with one input for each digest: the record as byte arrays has
the bytes with this digest, and no other bytes with it. -/
theorem recordOf_listed {record : List (Sponge.Str D × Bytes 32)} {bytes : ByteArray}
    (short : bytes.size ≤ D) {digest : Bytes 32} (listed : (strOf bytes short, digest) ∈ record)
    (single : ∀ a b d, (a, d) ∈ record → (b, d) ∈ record → a = b) :
    (bytes, Bytes.array digest) ∈ recordOf record ∧
      ∀ input, (input, Bytes.array digest) ∈ recordOf record → input = bytes := by
  refine ⟨List.mem_map.mpr ⟨_, listed, Prod.ext (array_strOf bytes short) rfl⟩, ?_⟩
  intro input found
  obtain ⟨⟨message, value⟩, inside, same⟩ := List.mem_map.mp found
  obtain ⟨first, second⟩ := Prod.mk.inj same
  obtain rfl : value = digest := Bytes.array_injective second
  rw [← first, single message _ value inside listed]
  exact array_strOf bytes short

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {Result Store : Type}

/-- The wire messages of an output of `prover20` (the bytes of a proof with a result, the
answered queries, a cache and the record of the digest queries): the pages `stagePage20` of the
bytes and their decoded messages, with the answers of the answered queries in the sponge of
`pub0`, and with the last blocks of the record at the two digests of the stage 472. -/
noncomputable def wires20 (pub0 : Public)
    (out : ((ByteArray × Result) ×
      List ((Sponge.Base L B C × Sponge.Y S C) ⊕ (Sponge.Str D × Bytes 32))) ×
        Store × List (Sponge.Str D × Bytes 32)) (k : ℕ) : Key20.Wire E :=
  stagePage20 (pub0.reideal (some (answerOf (logTable out.1.2)))) out.1.1.1
    (messagesOf pub0.ops out.1.1.1)
    (tFound (recordOf out.2.2) (tDigestAt (pub0.reideal (some (answerOf (logTable out.1.2))))
      out.1.1.1 (messagesOf pub0.ops out.1.1.1) 472))
    (eFound (recordOf out.2.2) (eDigestAt (pub0.reideal (some (answerOf (logTable out.1.2))))
      out.1.1.1 (messagesOf pub0.ops out.1.1.1) 472)) k

end Wires

section Prover

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {L B C S D total : ℕ}
  (block : Key20.root.Block → Coord) (block₁ : Fin Key20.level₁.shape.blocks → Coord)
  (block₂ : Fin Key20.level₂.shape.blocks → Coord) (block₃ : Fin Key20.level₃.shape.blocks → Coord)
  (block₄ : Fin Key20.level₄.shape.blocks → Coord) (block₅ : Fin Key20.level₅.shape.blocks → Coord)
  (blockL : Fin Key20.terminal.blocks → Coord) (pub0 : Public) {Result : Type}
  (aux : Sponge.Base L B C → Sponge.Y S C) (hash : Sponge.Str D → Bytes 32)

local notation "Reply" => (Option Coord → CodeSlot E)

local notation "Addr" => Address (Key20.Wire E) total

/-- The recording prover of `key20.aks` with no named key on the closed run of `run`, with its
wire messages `wires20` as the output. -/
noncomputable def closedProver20
    (run : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32)
      (ByteArray × Result)) : OracleComp Addr Reply (ℕ → Key20.Wire E) :=
  (Simulate.compDeps (resolveAt total pub0) (fun _ => [])
    (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) aux hash
    (run.closed (deps20 S)).logged (fun _ => none) [] aux).map (wires20 pub0)

/-- The proof of `closedProver20` with the oracle `table` has the wire messages `wires20` of the
simulation, with the named keys `deps20`, of the closed run with its answered queries. -/
theorem closedProver20_run
    (run : Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32)
      (ByteArray × Result)) (table : Addr → Reply) :
    (closedProver20 (total := total) block block₁ block₂ block₃ block₄ block₅ blockL pub0 aux hash
      run).run table =
      wires20 pub0 (Simulate.simulateDeps (resolveAt total pub0) (deps20 S)
        (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) table aux hash
        (run.closed (deps20 S)).logged (fun _ => none) []) :=
  (OracleComp.run_map _ table _).trans
    (congrArg (wires20 pub0)
      (Simulate.compDeps_closed_run_empty (resolveAt total pub0) (deps20 S)
        (phiAt block block₁ block₂ block₃ block₄ block₅ blockL) (deps20_closed S) table aux hash
        run))

end Prover

section Last

/-- In the skeleton of `layout20`, the draws of the stage 472 end before the terminal, and no
operation after them and before the terminal is a work or a draw. -/
theorem skeleton20_quietLast :
    (decide (workIndices20.getD 472 0 + 1 + drawCount20 472 ≤ 1521) &&
      ((sk20.take 1521).drop (workIndices20.getD 472 0 + 1 + drawCount20 472)).all
        fun x => !isWork x.1 && !isDraw x.1) = true := by
  decide +kernel

variable {pub : Public} {proof : ByteArray} {messages : Array Message}
  {a : List Duplex.Event → ℕ → ByteArray}

/-- On a script of `layout20`, the wire message 473 is on the state before the operation
1521. -/
theorem stageIndex20_last (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) :
    stageIndex20 pub 473 = 1521 := by
  have size : pub.ops.size = 1522 := by rw [size_skeleton erased, skeleton20_length]
  unfold stageIndex20
  rw [ite_eq_right (by decide), size]

/-- On a script of `layout20`, the draws of the stage 472 end before the terminal, and no
operation after them and before the terminal is a work or a draw. -/
theorem quietLast20_of (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {k : ℕ}
    (top : k = 472) :
    stageIndex20 pub k + 1 + drawCount20 k ≤ stageIndex20 pub 473 ∧
      ∀ x ∈ (pub.ops.zipIdx.toList.take (stageIndex20 pub 473)).drop
        (stageIndex20 pub k + 1 + drawCount20 k), isWork x.1 = false ∧ isDraw x.1 = false := by
  subst top
  have fit := skeleton20_quietLast
  rw [Bool.and_eq_true] at fit
  rw [stageIndex20_lt pub (by decide : 472 < 473), stageIndex20_last erased]
  refine ⟨of_decide_eq_true fit.1, fun x member => ?_⟩
  have erasedMember : eraseAt x ∈ (sk20.take 1521).drop
      (workIndices20.getD 472 0 + 1 + drawCount20 472) := by
    rw [← take_erase erased, ← List.map_drop]
    exact List.mem_map_of_mem member
  have both := List.all_eq_true.mp fit.2 _ erasedMember
  rw [Bool.and_eq_true] at both
  have works : isWork x.1.erase = isWork x.1 := by cases x.1 <;> rfl
  exact ⟨works.symm.trans (by simpa using both.1),
    (isDraw_erase x.1).symm.trans (by simpa using both.2)⟩

/-- On a script of `layout20`, the squeeze of the work of each stage is before the terminal. -/
theorem stageIndex20_succ_le_last (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    {j : ℕ} (inside : j < 473) : stageIndex20 pub j + 1 ≤ stageIndex20 pub 473 := by
  have last := (quietLast20_of erased rfl).1
  rcases Nat.lt_or_ge j 472 with lt | ge
  · exact Nat.le_trans (stageIndex20_succ_le pub lt (by decide))
      (Nat.le_trans (Nat.le_add_right _ _) (Nat.le_trans (Nat.le_add_right _ _) last))
  · obtain rfl : j = 472 := Nat.le_antisymm (Nat.le_of_lt_succ inside) ge
    exact Nat.le_trans (Nat.le_add_right _ _) last

/-- On a script of `layout20`, the events of the state before the work of each stage, and before
the terminal for `473 ≤ k`, are a suffix of the events before the terminal. -/
theorem stateBefore_suffixLast20 (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub 473))
      (channelStart pub)).1 = .ok ()) (k : ℕ) :
    (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events <:+
      (stateBefore pub proof messages pub.ops (stageIndex20 pub 473)).sponge.events := by
  have le : stageIndex20 pub k ≤ stageIndex20 pub 473 :=
    Nat.le_trans (stageIndex20_le_last pub erased k) (Nat.le_of_eq (ite_eq_right (by decide)).symm)
  obtain ⟨-, -, cut⟩ := stateBefore_le pub proof messages pub.ops le ran
  obtain ⟨grown, later⟩ := runFrom_suffix pub proof messages
    ((pub.ops.zipIdx.toList.take (stageIndex20 pub 473)).drop (stageIndex20 pub k))
    (stateBefore pub proof messages pub.ops (stageIndex20 pub k))
  rw [← cut] at later
  exact ⟨grown, later.symm⟩

/-- On a script of `layout20` with the work bits of the model and an ideal sponge, the events
before the terminal of a run with no stop before the terminal are absorbs, the squeezes of the
draws and of the work of the stage 472, and the events before the squeeze of this work. -/
theorem lastEvents20 (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub 473))
      (channelStart pub)).1 = .ok ()) {k : ℕ} (top : k = 472) :
    ∃ (new : List ByteArray) (counts : List ℕ),
      (stateBefore pub proof messages pub.ops (stageIndex20 pub 473)).sponge.events =
        new.map Duplex.Event.absorb ++
          (counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages k) := by
  have inside : k < 473 := top.trans_lt (by decide)
  obtain ⟨le, quiet⟩ := quietLast20_of erased top
  obtain ⟨site, level, at_⟩ := bits k inside
  obtain ⟨first, -, cut⟩ := stateBefore_le pub proof messages pub.ops le ran
  have drawn := (draw_squeeze20 at_ (bits20_ne_zero inside) (Nat.le_add_right _ _)
    (draws20_of erased inside (Nat.le_refl _)) first ideal).1
  obtain ⟨counts, shape⟩ : ∃ counts : List ℕ,
      (stateBefore pub proof messages pub.ops
        (stageIndex20 pub k + 1 + drawCount20 k)).sponge.events =
        counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages k :=
    ⟨(((pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1 + drawCount20 k)).drop
        (stageIndex20 pub k + 1)).reverse.map fun x => x.1.drawBytes) ++ [8], by
      rw [drawn, List.map_append, List.map_map, List.append_assoc]
      rfl⟩
  obtain ⟨new, events⟩ := runFrom_absorbs pub proof messages
    ((pub.ops.zipIdx.toList.take (stageIndex20 pub 473)).drop
      (stageIndex20 pub k + 1 + drawCount20 k))
    (stateBefore pub proof messages pub.ops (stageIndex20 pub k + 1 + drawCount20 k)) quiet
  rw [← cut, shape] at events
  exact ⟨new, counts, events⟩

/-- On a script of `layout20` with the work bits of the model and an ideal sponge with no event
before the channel: the events before a squeeze inside the events before the terminal are
squeezes and the events before the squeeze of the work of a stage `j < 473`. -/
theorem squeeze_last20 (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a) (fresh : pub.sponge.events = [])
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub 473))
      (channelStart pub)).1 = .ok ())
    {before : List Duplex.Event} {count : ℕ}
    (inner : (Duplex.Event.squeeze count :: before) <:+
      (stateBefore pub proof messages pub.ops (stageIndex20 pub 473)).sponge.events) :
    ∃ j, j < 473 ∧ ∃ counts : List ℕ,
      before = counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages j := by
  obtain ⟨new, counts, events⟩ := lastEvents20 erased bits ideal ran rfl
  rw [events] at inner
  have early : stageIndex20 pub 472 ≤ stageIndex20 pub 473 :=
    Nat.le_trans (Nat.le_succ _) (stageIndex20_succ_le_last erased (by decide))
  rcases squeeze_squeezes counts (squeeze_absorbs new inner) with ⟨later, same⟩ | deeper
  · exact ⟨472, by decide, later, same⟩
  · obtain ⟨j, lt, found⟩ := squeeze_stage20 erased bits ideal fresh 472 (by decide)
      (stateBefore_le pub proof messages pub.ops early ran).1 deeper
    exact ⟨j, Nat.lt_trans lt (by decide), found⟩

/-- Two tables of the sponge with the same answers at the base of each stage `j < 473` give
answer functions that agree on the events before the terminal. -/
theorem agreeOn_last20 {L B C S : ℕ} {fs other : Sponge.Base L B C → Sponge.Y S C}
    (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) (bits : Bits20 pub)
    (ideal : pub.sponge.ideal = some (answerOf fs)) (fresh : pub.sponge.events = [])
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub 473))
      (channelStart pub)).1 = .ok ())
    (same : ∀ j, j < 473 → ∀ base : Sponge.Base L B C,
      base.events = eventsAt20 pub proof messages j → fs base = other base) :
    AgreeOn (stateBefore pub proof messages pub.ops (stageIndex20 pub 473)).sponge.events
      (answerOf fs) (answerOf other) := by
  refine agreeOn_answerOf fun before count base sub inner found => ?_
  obtain ⟨j, lt, counts, shape⟩ := squeeze_last20 erased bits ideal fresh ran inner
  obtain ⟨parts, -⟩ := Sponge.split_eq_some.mp found
  obtain ⟨site, level, at_⟩ := bits j lt
  obtain ⟨m, -, events, -, -⟩ := work_squeeze20 at_ (bits20_ne_zero lt)
    (stateBefore_le pub proof messages pub.ops (stageIndex20_succ_le_last erased lt) ran).1 ideal
  have head : ∀ c rest, eventsAt20 pub proof messages j ≠ Duplex.Event.squeeze c :: rest := by
    intro c rest equal
    rw [events] at equal
    exact Duplex.Event.noConfusion (List.cons.inj equal).1
  exact same j lt base (Sponge.squeezes_append_inj base.events_ne head
    (show sub.counts.map Duplex.Event.squeeze ++ base.events =
      counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages j from
      parts.symm.trans shape)).2

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- For two tables of the sponge with the same answers at the base of each stage `j < 473` of a
run with no stop before the terminal: each page `k`, also the page 473, and the two digests in
the state of the page `k` are the same for the two tables. -/
theorem stagePage20_tables {L B C S : ℕ} {fs other : Sponge.Base L B C → Sponge.Y S C}
    (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) (bits : Bits20 pub)
    (ideal : pub.sponge.ideal = some (answerOf fs)) (fresh : pub.sponge.events = [])
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub 473))
      (channelStart pub)).1 = .ok ())
    (same : ∀ j, j < 473 → ∀ base : Sponge.Base L B C,
      base.events = eventsAt20 pub proof messages j → fs base = other base)
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree) (k : ℕ) :
    stagePage20 (E := E) (pub.reideal (some (answerOf other))) proof messages tLast eLast k =
        stagePage20 pub proof messages tLast eLast k ∧
      tDigestAt (pub.reideal (some (answerOf other))) proof messages k =
        tDigestAt pub proof messages k ∧
      eDigestAt (pub.reideal (some (answerOf other))) proof messages k =
        eDigestAt pub proof messages k := by
  have le : stageIndex20 pub k ≤ stageIndex20 pub 473 :=
    Nat.le_trans (stageIndex20_le_last pub erased k) (Nat.le_of_eq (ite_eq_right (by decide)).symm)
  have before := (stateBefore_le pub proof messages pub.ops le ran).1
  have agree := (agreeOn_last20 erased bits ideal fresh ran same).suffix
    (stateBefore_suffixLast20 erased ran k)
  exact ⟨pageAt20_reideal ideal before agree _ _, digestAt_reideal ideal before agree⟩

end Last

section Equal

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {L B C S D : ℕ}
  {Result Store : Type}

/-- The public input of the verifier of the experiment without the setup prefix of the caller:
the public input of `resolveAt` with the answers of the table and the digest of `hash`. -/
theorem sealed_withOracles (pub : Public) (fs : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32) :
    (pub.withOracles fs hash).sealed =
      ((pub.withSponge Duplex.empty).sealed.reideal (some (answerOf fs))).withDigest
        (digestOf hash) :=
  rfl

theorem Public.reideal_reideal (pub : Public)
    (first second : Option (List Duplex.Event → ℕ → ByteArray)) :
    (pub.reideal first).reideal second = pub.reideal second :=
  rfl

/-- The pages with the answers of a second table and with the last blocks of a record are the
pages of an accepted run. For a public input of `layout20` with the answers of the table `fs`
and a hash `digest`, an accepted proof with last blocks `Last20`, a table `other` with the
answers of `fs` at the base of each stage, and a record with the digest input of each last
block at its digest and no other input at this digest: each page `stagePage20 … k` with the
answers of `other` and the blocks of the record at the two digests of the stage 472 is the page
of the run with the last blocks. -/
theorem stagePage20_found {pub : Public} {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice) (digest : ByteArray → ByteArray)
    {proof : ByteArray} {messages : Array Message}
    (accepted : runProof (pub.withDigest digest) proof = .verdict .accept)
    (whole : (runFrom (pub.withDigest digest) proof messages
      (pub.withDigest digest).ops.zipIdx.toList (channelStart (pub.withDigest digest))).1 =
        .ok ())
    {fs other : Sponge.Base L B C → Sponge.Y S C}
    (ideal : pub.sponge.ideal = some (answerOf fs)) (fresh : pub.sponge.events = [])
    {tLast : ℕ → Rq Key20.terminal.innerDegree} {eLast : Rq Key20.terminal.innerDegree}
    (last : Last20 (E := E) (pub.withDigest digest) proof messages tLast eLast)
    (same : ∀ j, j < 473 → ∀ base : Sponge.Base L B C,
      base.events = eventsAt20 (pub.withDigest digest) proof messages j → fs base = other base)
    {record : List (ByteArray × ByteArray)}
    (tIn : (tInput Key20.terminal tLast, digest (tInput Key20.terminal tLast)) ∈ record)
    (tOne : ∀ input, (input, digest (tInput Key20.terminal tLast)) ∈ record →
      input = tInput Key20.terminal tLast)
    (eIn : (eInput eLast, digest (eInput eLast)) ∈ record)
    (eOne : ∀ input, (input, digest (eInput eLast)) ∈ record → input = eInput eLast) (k : ℕ) :
    stagePage20 (E := E) (pub.reideal (some (answerOf other))) proof messages
        (tFound record (tDigestAt (pub.reideal (some (answerOf other))) proof messages 472))
        (eFound record (eDigestAt (pub.reideal (some (answerOf other))) proof messages 472)) k =
      stagePage20 (pub.withDigest digest) proof messages tLast eLast k := by
  have scriptedD := verifyProof_scripted accepted
  have scripted : script pub.layout pub.plan = .ok pub.ops := scriptedD
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    have scripted' := scripted
    rw [layout] at scripted'
    exact (script_erase scripted').trans (skeleton20 zMax rice)
  have bits : Bits20 pub := bits20_of_scripted layout scripted
  have ran := (ran20_withDigest pub digest proof messages layout 473).mp
    (runFrom_take_ok (pub.withDigest digest) proof messages _ _ _ whole)
  have same' : ∀ j, j < 473 → ∀ base : Sponge.Base L B C,
      base.events = eventsAt20 pub proof messages j → fs base = other base :=
    fun j lt base events => same j lt base
      (events.trans (eventsAt20_withDigest pub digest proof messages layout j).symm)
  obtain ⟨tHash, eHash⟩ := digestAt_of_last (pub := pub.withDigest digest) layout accepted last
  obtain ⟨-, tMoved, eMoved⟩ :=
    stagePage20_tables (E := E) erased bits ideal fresh ran same' tLast eLast 472
  have tDigest : tDigestAt (pub.reideal (some (answerOf other))) proof messages 472 =
      digest (tInput Key20.terminal tLast) :=
    tMoved.trans ((tDigestAt_withDigest pub digest proof messages layout 472).symm.trans
      (tHash 472 (by decide)).symm)
  have eDigest : eDigestAt (pub.reideal (some (answerOf other))) proof messages 472 =
      digest (eInput eLast) :=
    eMoved.trans ((eDigestAt_withDigest pub digest proof messages layout 472).symm.trans
      (eHash 472 (Nat.le_refl _)).symm)
  rw [tDigest, eDigest, eFound_eq eIn eOne]
  have rows : rowsOf Key20.terminal (tFound record (digest (tInput Key20.terminal tLast))) =
      rowsOf Key20.terminal tLast :=
    (congrArg (rowsOf Key20.terminal) (tFound_eq tIn tOne)).trans
      (rowsOf_rowsOf Key20.terminal tLast)
  have blocks : stagePage20 (E := E) pub proof messages
      (tFound record (digest (tInput Key20.terminal tLast))) eLast k =
      stagePage20 pub proof messages tLast eLast k :=
    stagePage20_congr (fun _ => rows) fun _ => rfl
  exact ((stagePage20_tables (E := E) erased bits ideal fresh ran same' _ eLast k).1.trans
    blocks).trans
    (stagePage20_withDigest pub digest proof messages tLast eLast layout k).symm

/-- The wire messages of the recording prover are the pages of the accepted run. For an accepted
proof of `layout20` with the tables `fs` and `hash`, its decoded messages and last blocks
`Last20`; answered queries with the answers of `fs` and one query or more at the base of each
stage; a record of digest queries with the query of the digest input of each last block, and
one input for each digest: each wire message `wires20 … k` is the page `stagePage20 … k` of the
run with the last blocks. -/
theorem wires20_eq {pubV : Public} {zMax rice : ℕ} (layout : pubV.layout = layout20 zMax rice)
    {fs : Sponge.Base L B C → Sponge.Y S C} {hash : Sponge.Str D → Bytes 32} {proof : ByteArray}
    (accepted : verifyProof (pubV.withOracles fs hash) proof = .verdict .accept)
    {messages : Array Message} {rest : ℕ}
    (decoded : decode (pubV.withOracles fs hash).sealed.ops proof = .ok (messages, rest))
    (fits : 4128 ≤ D) {tLast : ℕ → Rq Key20.terminal.innerDegree}
    {eLast : Rq Key20.terminal.innerDegree}
    (last : Last20 (E := E) (pubV.withOracles fs hash).sealed proof messages tLast eLast)
    {answered : List ((Sponge.Base L B C × Sponge.Y S C) ⊕ (Sponge.Str D × Bytes 32))}
    (answers : ∀ base value, Sum.inl (base, value) ∈ answered → value = fs base)
    (asked : ∀ j, j < 473 → ∀ base : Sponge.Base L B C,
      base.events = eventsAt20 (pubV.withOracles fs hash).sealed proof messages j →
        ∃ value, Sum.inl (base, value) ∈ answered)
    {record : List (Sponge.Str D × Bytes 32)}
    (tListed : (strOf (tInput Key20.terminal tLast) ((size_tInput20 tLast).trans_le fits),
      hash (strOf (tInput Key20.terminal tLast) ((size_tInput20 tLast).trans_le fits))) ∈ record)
    (eListed : (strOf (eInput eLast)
        ((size_eInput20 eLast).trans_le (Nat.le_trans (by decide) fits)),
      hash (strOf (eInput eLast)
        ((size_eInput20 eLast).trans_le (Nat.le_trans (by decide) fits)))) ∈ record)
    (single : ∀ a b d, (a, d) ∈ record → (b, d) ∈ record → a = b)
    (result : Result) (cache : Store) (k : ℕ) :
    wires20 (E := E) (pubV.withSponge Duplex.empty).sealed
        (((proof, result), answered), cache, record) k =
      stagePage20 (pubV.withOracles fs hash).sealed proof messages tLast eLast k := by
  obtain ⟨-, -, none_⟩ := replay_of_accepted pubV proof fs hash accepted decoded
  have whole := (replay_none_iff _ _ _ _).mp none_
  have ranV := ((verifyProof_accept_iff _ _).mp accepted).2
  have same : ∀ j, j < 473 → ∀ base : Sponge.Base L B C,
      base.events = eventsAt20 (pubV.withOracles fs hash).sealed proof messages j →
        fs base = logTable answered base :=
    fun j lt base events => (logTable_eq answers (asked j lt base events)).symm
  have tPair := recordOf_listed _ tListed single
  have ePair := recordOf_listed _ eListed single
  rw [← digestOf_short hash ((size_tInput20 tLast).trans_le fits)] at tPair
  rw [← digestOf_short hash ((size_eInput20 eLast).trans_le (Nat.le_trans (by decide) fits))]
    at ePair
  have names : messagesOf (pubV.withSponge Duplex.empty).sealed.ops proof = messages :=
    messagesOf_decoded decoded
  rw [sealed_withOracles] at whole ranV same last ⊢
  have core := stagePage20_found (E := E) (pub := (pubV.withSponge Duplex.empty).sealed.reideal
    (some (answerOf fs))) layout (digestOf hash) ranV whole rfl rfl last same tPair.1 tPair.2
    ePair.1 ePair.2 k
  rw [Public.reideal_reideal] at core
  show stagePage20 ((pubV.withSponge Duplex.empty).sealed.reideal
      (some (answerOf (logTable answered)))) proof
      (messagesOf (pubV.withSponge Duplex.empty).sealed.ops proof)
      (tFound (recordOf record) (tDigestAt ((pubV.withSponge Duplex.empty).sealed.reideal
        (some (answerOf (logTable answered)))) proof
        (messagesOf (pubV.withSponge Duplex.empty).sealed.ops proof) 472))
      (eFound (recordOf record) (eDigestAt ((pubV.withSponge Duplex.empty).sealed.reideal
        (some (answerOf (logTable answered)))) proof
        (messagesOf (pubV.withSponge Duplex.empty).sealed.ops proof) 472)) k = _
  rw [names]
  exact core

/-- The hypotheses of `wires20_eq` on the answered queries and on the digest queries hold for
the queries of the verifier: for an accepted proof of `layout20` with last blocks `Last20`, the
base of each stage is a query of the verifier, and the digest inputs of the two last blocks are
queries of the verifier. -/
theorem wires20_guard {pubV : Public} {zMax rice : ℕ} (layout : pubV.layout = layout20 zMax rice)
    {fs : Sponge.Base L B C → Sponge.Y S C} {hash : Sponge.Str D → Bytes 32} {proof : ByteArray}
    (accepted : verifyProof (pubV.withOracles fs hash) proof = .verdict .accept)
    {messages : Array Message} {rest : ℕ}
    (decoded : decode (pubV.withOracles fs hash).sealed.ops proof = .ok (messages, rest))
    (fits : 4128 ≤ D) {tLast : ℕ → Rq Key20.terminal.innerDegree}
    {eLast : Rq Key20.terminal.innerDegree}
    (last : Last20 (E := E) (pubV.withOracles fs hash).sealed proof messages tLast eLast) :
    (∀ j, j < 473 → ∀ base : Sponge.Base L B C,
      base.events = eventsAt20 (pubV.withOracles fs hash).sealed proof messages j →
        Sum.inl base ∈ (verifierRun pubV proof).trace fs hash) ∧
      Sum.inr (strOf (tInput Key20.terminal tLast) ((size_tInput20 tLast).trans_le fits)) ∈
        (verifierRun pubV proof).trace fs hash ∧
      Sum.inr (strOf (eInput eLast)
        ((size_eInput20 eLast).trans_le (Nat.le_trans (by decide) fits))) ∈
        (verifierRun pubV proof).trace fs hash := by
  obtain ⟨eAsked, tAsked⟩ := verifierRun_digests20 pubV proof fs hash layout accepted decoded fits
    last
  refine ⟨fun j lt base events => ?_, tAsked, eAsked⟩
  obtain ⟨found, same, queried⟩ := verifierRun_asked20 pubV proof fs hash layout accepted decoded lt
  rwa [Sponge.Base.events_injective (events.trans same.symm)]

end Equal

end NightstreamFPrime.Spec.Compression.Akita.Replay
