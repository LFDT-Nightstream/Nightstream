import NightstreamFPrime.Spec.Compression.Akita.Replay.PageAt
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageReadings

/-! The address of the model for the events of the sponge before the work of a stage, for
`key20.aks` (Akita revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23, Lemma 10.47,
the paragraph "The prover P"; `docs/akita-work/w21/design.md`, section 3; `Replay/Simulate.lean`,
the functions `resolve` and `back`; `Replay/PageAt.lean`).

- `digestInput`, `eInput`, `tInput`: the bytes that `fieldDigest`, `ringDigest` and `rowsDigestOf`
  hash. `eInput_injective` and `tInput_inj`: they give the block (for `t`, its rows
  `0, …, n_A − 1`; `rowsOf` puts 0 in each other row, and `pageAt20_rows`: a page reads no other
  row).
- `stagePage20 … j` is `pageAt20 … j` with the last block of `t` only for `379 ≤ j` and the last
  block of `e` only for `472 ≤ j`, and zero blocks before (`stagePage20_early`,
  `stagePage20_middle`, `stagePage20_last`): the events before the work of a stage `j < 379` do
  not have the digest of the last block of `t`, and those of a stage `j < 472` do not have the
  digest of the last block of `e`. `stagePage20_reads`: `Key20.Tower.accepts` reads the blocks
  only in the pages 379, 472 and 473.
- `addressAt20 … total k`: the pages `stagePage20 … 0, …, k` as a key of the model.
- `Stage20 pub events k proof messages`: `events` are `eventsAt20 … k` of a proof with its decoded
  messages whose run has no stop before the work of the stage `k < 473`. `stage20_unique`: with
  a work bit or more in each stage, the events give the stage.
- `tDigestAt`, `eDigestAt`: the digests of the last blocks in the state before the work of a
  stage. `tFound`, `eFound`: the block of a digest in a record of digest queries, and zero when
  the record has no such block.
- `resolve20`, with `resolve20_stage`, `resolve20_eq` and `resolve20_mono`; `back20`, with
  `resolves20`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Address)

open Replayer

section Bytes

theorem bytesOf_append (a b : ByteArray) : bytesOf (a ++ b) = bytesOf a ++ bytesOf b := by
  unfold bytesOf
  rw [ByteArray.data_append, Array.toList_append, List.map_append]

/-- Two lists of the same count of strings of one length with the same bytes are equal. -/
theorem flatten_inj {β : Type} (n : ℕ) : ∀ (L L' : List (List β)), L.length = L'.length →
    (∀ x ∈ L, x.length = n) → (∀ x ∈ L', x.length = n) → L.flatten = L'.flatten → L = L'
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, length, _, _, _ => by simp at length
  | _ :: _, [], length, _, _, _ => by simp at length
  | a :: L, b :: L', length, sized, sized', same => by
      rw [List.flatten_cons, List.flatten_cons] at same
      obtain ⟨head, tail⟩ := List.append_inj same
        ((sized a List.mem_cons_self).trans (sized' b List.mem_cons_self).symm)
      rw [head, flatten_inj n L L' (by simpa using length)
        (fun x member => sized x (List.mem_cons_of_mem a member))
        (fun x member => sized' x (List.mem_cons_of_mem b member)) tail]

theorem bytesOf_foldl {α : Type} (g : α → ByteArray) : ∀ (l : List α) (init : ByteArray),
    bytesOf (l.foldl (fun out a => out ++ g a) init) =
      bytesOf init ++ (l.map fun a => bytesOf (g a)).flatten
  | [], init => by simp
  | a :: l, init => by
      rw [List.foldl_cons, bytesOf_foldl g l, bytesOf_append, List.map_cons, List.flatten_cons,
        List.append_assoc]

theorem toList_foldl {α β : Type} (g : α → Array β) : ∀ (l : List α) (init : Array β),
    (l.foldl (fun out a => out ++ g a) init).toList =
      init.toList ++ (l.map fun a => (g a).toList).flatten
  | [], init => by simp
  | a :: l, init => by
      rw [List.foldl_cons, toList_foldl g l, Array.toList_append, List.map_cons, List.flatten_cons,
        List.append_assoc]

theorem size_foldl {α β : Type} (g g' : α → Array β) : ∀ (l : List α) (init init' : Array β),
    init.size = init'.size → (∀ a ∈ l, (g a).size = (g' a).size) →
      (l.foldl (fun out a => out ++ g a) init).size =
        (l.foldl (fun out a => out ++ g' a) init').size
  | [], _, _, size, _ => size
  | a :: l, init, init', size, sized => by
      rw [List.foldl_cons, List.foldl_cons]
      exact size_foldl g g' l _ _
        (by rw [Array.size_append, Array.size_append, size, sized a List.mem_cons_self])
        fun x member => sized x (List.mem_cons_of_mem a member)

end Bytes

section Inputs

/-- The bytes that `fieldDigest` hashes: the tag `akita-pcs/field-digest/v8` in 32 bytes, then
the 8 bytes of each word. -/
def digestInput (values : Array Fq) : ByteArray :=
  values.foldl (fun out a => out ++ Setup.littleEndian 8 a.word.toNat)
    ("akita-pcs/field-digest/v8".toUTF8 ++ ByteArray.mk (Array.replicate 7 0))

theorem fieldDigest_input (digest : ByteArray → ByteArray) (values : Array Fq) :
    fieldDigest digest values = digest (digestInput values) := rfl

/-- The bytes that `fieldDigest` hashes give the words, for two arrays of one size. -/
theorem digestInput_inj {a b : Array Fq} (same : digestInput a = digestInput b)
    (size : a.size = b.size) : a = b := by
  have bytes := congrArg bytesOf same
  unfold digestInput at bytes
  rw [← Array.foldl_toList, ← Array.foldl_toList, bytesOf_foldl, bytesOf_foldl] at bytes
  have sized : ∀ (c : Array Fq),
      ∀ x ∈ c.toList.map (fun v : Fq => bytesOf (Setup.littleEndian 8 v.word.toNat)),
        x.length = 8 := by
    intro c x member
    obtain ⟨v, -, rfl⟩ := List.mem_map.mp member
    rw [length_bytesOf]
    exact le64_size _
  have chunks := flatten_inj 8 _ _
    (by rw [List.length_map, List.length_map, Array.length_toList, Array.length_toList, size])
    (sized a) (sized b) (List.append_cancel_left bytes)
  have injective :
      Function.Injective fun v : Fq => bytesOf (Setup.littleEndian 8 v.word.toNat) := by
    intro v w equal
    have nat : v.word.toNat = w.word.toNat :=
      le64_inj v.word.toNat_lt w.word.toNat_lt (bytesOf_injective equal)
    cases v
    cases w
    exact congrArg Fq.mk (UInt64.toNat_inj.mp nat)
  exact Array.toList_inj.mp (List.map_injective_iff.mpr injective chunks)

/-- The canonical coefficients of a ring element give the element. -/
theorem canonical_inj {d : ℕ} [NeZero d] {y y' : Rq d}
    (same : Poly.ofFn d (fun k => Fq.ofNat (coeff y k).val) =
      Poly.ofFn d fun k => Fq.ofNat (coeff y' k).val) : y = y' := by
  refine ext_coeff fun k small => ?_
  have entry := congrArg (fun a => (Poly.get a k).toZq) same
  simp only [Poly.get_ofFn _ small, Fq.toZq_ofNat] at entry
  rwa [ZMod.natCast_zmod_val, ZMod.natCast_zmod_val] at entry

/-- The bytes that `ringDigest` hashes. -/
noncomputable def eInput {d : ℕ} [NeZero d] (y : Rq d) : ByteArray :=
  digestInput (Poly.ofFn d fun k => Fq.ofNat (coeff y k).val)

theorem ringDigest_input (digest : ByteArray → ByteArray) {d : ℕ} [NeZero d] (y : Rq d) :
    ringDigest digest y = digest (eInput y) := rfl

/-- The bytes that `ringDigest` hashes give the ring element. -/
theorem eInput_injective {d : ℕ} [NeZero d] : Function.Injective (eInput (d := d)) :=
  fun _ _ same => canonical_inj (digestInput_inj same (by rw [Poly.size_ofFn, Poly.size_ofFn]))

variable (S : Shape)

/-- The bytes that `rowsDigestOf` hashes. -/
noncomputable def tInput (rows : ℕ → Rq S.innerDegree) : ByteArray :=
  digestInput ((List.range S.innerRank).foldl (fun out r => out ++ Poly.ofFn S.innerDegree
    fun k => Fq.ofNat (coeff (rows r) k).val) #[])

theorem rowsDigestOf_input (digest : ByteArray → ByteArray) (rows : ℕ → Rq S.innerDegree) :
    rowsDigestOf S digest rows = digest (tInput S rows) := rfl

/-- The bytes that `rowsDigestOf` hashes give the rows `0, …, n_A − 1`. -/
theorem tInput_inj {rows rows' : ℕ → Rq S.innerDegree} (same : tInput S rows = tInput S rows') :
    ∀ r < S.innerRank, rows r = rows' r := by
  unfold tInput at same
  have lists := congrArg Array.toList (digestInput_inj same
    (size_foldl _ _ _ _ _ rfl fun r _ => by rw [Poly.size_ofFn, Poly.size_ofFn]))
  rw [toList_foldl, toList_foldl] at lists
  have sized : ∀ (rows : ℕ → Rq S.innerDegree),
      ∀ x ∈ (List.range S.innerRank).map (fun r => (Poly.ofFn S.innerDegree
        fun k => Fq.ofNat (coeff (rows r) k).val).toList), x.length = S.innerDegree := by
    intro rows x member
    obtain ⟨r, -, rfl⟩ := List.mem_map.mp member
    rw [Array.length_toList, Poly.size_ofFn]
  have chunks := flatten_inj S.innerDegree _ _ (by rw [List.length_map, List.length_map])
    (sized rows) (sized rows') (List.append_cancel_left lists)
  intro r below
  exact canonical_inj (Array.toList_inj.mp
    (List.map_inj_left.mp chunks r (List.mem_range.mpr below)))

/-- The rows `0, …, n_A − 1` of a block, and 0 in each other row. -/
noncomputable def rowsOf (rows : ℕ → Rq S.innerDegree) : ℕ → Rq S.innerDegree :=
  fun r => if r < S.innerRank then rows r else 0

theorem rowsOf_rowsOf (rows : ℕ → Rq S.innerDegree) : rowsOf S (rowsOf S rows) = rowsOf S rows := by
  funext r
  by_cases below : r < S.innerRank <;> simp [rowsOf, below]

variable {S}

theorem rowsOf_of_input {rows rows' : ℕ → Rq S.innerDegree}
    (same : tInput S rows = tInput S rows') : rowsOf S rows = rowsOf S rows' := by
  funext r
  by_cases below : r < S.innerRank
  · simp only [rowsOf, below, ↓reduceIte]
    exact tInput_inj S same r below
  · simp only [rowsOf, below, ↓reduceIte]

theorem stateOf_rows (msgs : TerminalMessages) (tLast : ℕ → Rq S.innerDegree) :
    stateOf S msgs (rowsOf S tLast) = stateOf S msgs tLast := by
  funext i r
  simp only [stateOf, rowsOf, r.isLt, ↓reduceIte]

end Inputs

section Pages

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]
  (pub : Public) (proof proof' : ByteArray) (messages messages' : Array Message)
  (tLast tLast' : ℕ → Rq Key20.terminal.innerDegree)
  (eLast eLast' : Rq Key20.terminal.innerDegree)

/-- A page reads only the rows `0, …, n_A − 1` of the last block of `t`. -/
theorem pageAt20_rows (k : ℕ) :
    pageAt20 (E := E) pub proof messages (rowsOf Key20.terminal tLast) eLast k =
      pageAt20 pub proof messages tLast eLast k := by
  simp only [pageAt20, noteAt20, noteOf, terminalStatementIn, stateOf_rows]

/-- The last block of `t` in the page `j`: the block for `379 ≤ j`, and zero before. -/
noncomputable def tBlockAt (j : ℕ) : ℕ → Rq Key20.terminal.innerDegree :=
  if 379 ≤ j then tLast else fun _ => 0

/-- The last block of `e` in the page `j`: the block for `472 ≤ j`, and zero before. -/
noncomputable def eBlockAt (j : ℕ) : Rq Key20.terminal.innerDegree :=
  if 472 ≤ j then eLast else 0

/-- The page `j` of the key of a stage: `pageAt20 … j` with the last blocks that the events
before the work of the stage `j` fix through their digests. -/
noncomputable def stagePage20 (j : ℕ) : Key20.Wire E :=
  pageAt20 pub proof messages (tBlockAt tLast j) (eBlockAt eLast j) j

/-- The key of the model for the stage `k`: the pages `stagePage20 … 0, …, k`. -/
noncomputable def addressAt20 (total k : ℕ) : Address (Key20.Wire E) total :=
  ⟨stagePage20 pub proof messages tLast eLast 0,
    (List.range (min k total)).map fun j => stagePage20 pub proof messages tLast eLast (j + 1), by
      rw [List.length_map, List.length_range]
      exact Nat.min_le_right k total⟩

variable {pub proof proof' messages messages'}

theorem stagePage20_early {j : ℕ} (early : j < 379) :
    stagePage20 (E := E) pub proof messages tLast eLast j =
      pageAt20 pub proof messages (fun _ => 0) 0 j := by
  unfold stagePage20 tBlockAt eBlockAt
  rw [ite_eq_right (by omega), ite_eq_right (by omega)]

theorem stagePage20_middle {j : ℕ} (late : 379 ≤ j) (early : j < 472) :
    stagePage20 (E := E) pub proof messages tLast eLast j =
      pageAt20 pub proof messages tLast 0 j := by
  unfold stagePage20 tBlockAt eBlockAt
  rw [ite_eq_left late, ite_eq_right (by omega)]

theorem stagePage20_last {j : ℕ} (late : 472 ≤ j) :
    stagePage20 (E := E) pub proof messages tLast eLast j =
      pageAt20 pub proof messages tLast eLast j := by
  unfold stagePage20 tBlockAt eBlockAt
  rw [ite_eq_left (by omega), ite_eq_left late]

variable {tLast tLast' eLast eLast'}

/-- The page `j` reads the rows `0, …, n_A − 1` of the last block of `t` only for `379 ≤ j`, and
the last block of `e` only for `472 ≤ j`. -/
theorem stagePage20_congr {j : ℕ}
    (rows : 379 ≤ j → rowsOf Key20.terminal tLast = rowsOf Key20.terminal tLast')
    (block : 472 ≤ j → eLast = eLast') :
    stagePage20 (E := E) pub proof messages tLast eLast j =
      stagePage20 pub proof messages tLast' eLast' j := by
  have partials : eBlockAt eLast j = eBlockAt eLast' j := by
    unfold eBlockAt
    by_cases late : 472 ≤ j
    · rw [ite_eq_left late, ite_eq_left late, block late]
    · rw [ite_eq_right late, ite_eq_right late]
  unfold stagePage20
  rw [partials]
  by_cases late : 379 ≤ j
  · unfold tBlockAt
    rw [ite_eq_left late, ite_eq_left late]
    exact (pageAt20_rows pub proof messages tLast _ j).symm.trans
      ((congrArg (fun t => pageAt20 pub proof messages t (eBlockAt eLast' j) j) (rows late)).trans
        (pageAt20_rows pub proof messages tLast' _ j))
  · unfold tBlockAt
    rw [ite_eq_right late, ite_eq_right late]

variable (tLast eLast)

theorem stagePage20_trace {j : ℕ} (inside : j < 473) :
    (stagePage20 (E := E) pub proof messages tLast eLast j).data.trace =
      traceAt20 pub proof messages j :=
  pageAt20_trace _ _ inside

theorem addressAt20_first (total k : ℕ) :
    (addressAt20 (E := E) pub proof messages tLast eLast total k).first =
      stagePage20 pub proof messages tLast eLast 0 := rfl

theorem length_addressAt20 {total k : ℕ} (le : k ≤ total) :
    (addressAt20 (E := E) pub proof messages tLast eLast total k).rest.length = k := by
  show ((List.range (min k total)).map _).length = k
  rw [List.length_map, List.length_range, Nat.min_eq_left le]

theorem addressAt20_rest {total k j : ℕ} (le : k ≤ total) (below : j < k) :
    (addressAt20 (E := E) pub proof messages tLast eLast total k).rest[j]? =
      some (stagePage20 pub proof messages tLast eLast (j + 1)) := by
  show ((List.range (min k total)).map _)[j]? = _
  rw [List.getElem?_map, List.getElem?_range (by rw [Nat.min_eq_left le]; exact below)]
  rfl

/-- The key of a stage `j ≤ k` is the key of the stage `k` up to the page `j`. -/
theorem addressAt20_take {k j : ℕ} (le : j ≤ k) (total : ℕ) :
    (addressAt20 (E := E) pub proof messages tLast eLast total k).take j =
      addressAt20 pub proof messages tLast eLast total j := by
  refine Address.ext rfl ?_
  show ((List.range (min k total)).map _).take j = (List.range (min j total)).map _
  have less : min j (min k total) = min j total := by omega
  rw [← List.map_take, List.take_range, less]

variable {tLast eLast}

/-- The key of the stage `k` reads the rows `0, …, n_A − 1` of the last block of `t` only for
`379 ≤ k`, and the last block of `e` only for `472 ≤ k`. -/
theorem addressAt20_congr {k : ℕ}
    (rows : 379 ≤ k → rowsOf Key20.terminal tLast = rowsOf Key20.terminal tLast')
    (block : 472 ≤ k → eLast = eLast') (total : ℕ) :
    addressAt20 (E := E) pub proof messages tLast eLast total k =
      addressAt20 pub proof messages tLast' eLast' total k := by
  refine Address.ext (stagePage20_congr (fun late => absurd late (by decide))
    fun late => absurd late (by decide)) ?_
  show (List.range (min k total)).map _ = (List.range (min k total)).map _
  refine List.map_congr_left fun j member => ?_
  have below : j < min k total := List.mem_range.mp member
  exact stagePage20_congr (fun late => rows (by omega)) fun late => block (by omega)

end Pages

section Stage

variable {pub : Public} {proof proof' : ByteArray} {messages messages' : Array Message}

/-- `events` are the events of the sponge before the squeeze of the work of the stage `k < 473`
of a proof with its decoded messages, whose run has no stop before this work. -/
def Stage20 (pub : Public) (events : List Duplex.Event) (k : ℕ) (proof : ByteArray)
    (messages : Array Message) : Prop :=
  k < 473 ∧ (∃ rest, decode pub.ops proof = .ok (messages, rest)) ∧
    (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok () ∧
    eventsAt20 pub proof messages k = events

theorem stageIndex20_succ_le (pub : Public) {j k : ℕ} (lt : j < k) (inside : k < 473) :
    stageIndex20 pub j + 1 ≤ stageIndex20 pub k := by
  rw [← stageStart20_succ]
  exact Nat.le_trans (stageStart20_le pub (by omega)) (stageIndex20_mono pub (by omega) inside)

/-- For a work with bits, the events before the squeeze of the work are one absorb and the events
before the work. -/
theorem eventsAt20_split (bits : Bits20 pub) {k : ℕ} (inside : k < 473) (proof : ByteArray)
    (messages : Array Message) :
    ∃ nonce : List Duplex.Event, nonce.length = 1 ∧
      eventsAt20 pub proof messages k =
        nonce ++ (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events := by
  obtain ⟨site, level, at_⟩ := bits k inside
  have positive := bits20_ne_zero inside
  generalize bits20.flatten.getD k 0 = count at at_ positive
  refine ⟨(inputsFrom proof messages
    ((pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)).drop (stageIndex20 pub k))
    (readCount (pub.ops.zipIdx.toList.take (stageIndex20 pub k)))).map Duplex.Event.absorb,
    ?_, rfl⟩
  have short : (pub.ops.zipIdx.toList.take (stageIndex20 pub k)).length = stageIndex20 pub k := by
    rw [List.length_take, Array.length_toList, Array.size_zipIdx]
    exact Nat.min_eq_left (Nat.le_of_lt (Array.getElem?_eq_some_iff.mp at_).1)
  rw [List.length_map, length_inputsFrom, take_succ_zipIdx pub.ops at_, List.drop_left' short]
  simp [readCount, Op.reads, positive]

/-- With a work bit or more in each stage, a later stage of a proof has more events than an
earlier stage of a second proof. -/
theorem stage20_not_lt (bits : Bits20 pub) {k k' : ℕ} (lt : k' < k) (inside : k < 473)
    {rest rest' : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k'))
      (channelStart pub)).1 = .ok ()) :
    eventsAt20 pub proof messages k ≠ eventsAt20 pub proof' messages' k' := by
  intro events
  have inside' : k' < 473 := by omega
  have above := stageIndex20_succ_le pub lt inside
  obtain ⟨N, one, split⟩ := eventsAt20_split bits inside proof messages
  obtain ⟨N', one', split'⟩ := eventsAt20_split bits inside' proof' messages'
  obtain ⟨okNext, -, cutNext⟩ := stateBefore_le pub proof messages pub.ops above ran
  obtain ⟨V, grown⟩ := runFrom_suffix pub proof messages
    ((pub.ops.zipIdx.toList.take (stageIndex20 pub k)).drop (stageIndex20 pub k' + 1))
    (stateBefore pub proof messages pub.ops (stageIndex20 pub k' + 1))
  rw [← cutNext] at grown
  obtain ⟨site, level, at_⟩ := bits k' inside'
  obtain ⟨stepped, after⟩ := stateBefore_succ pub proof messages pub.ops at_ okNext
  obtain ⟨m, -, sponge⟩ := step_work pub proof messages _ site level _ _
    (bits20_ne_zero inside') stepped
  have worked : (stateBefore pub proof messages pub.ops (stageIndex20 pub k' + 1)).sponge.events =
      .squeeze 8 :: .absorb ((Op.work site level (bits20.flatten.getD k' 0)).input proof m) ::
        (stateBefore pub proof messages pub.ops (stageIndex20 pub k')).sponge.events := by
    rw [after, sponge, Duplex.squeeze_events, Duplex.absorb_events]
  have okHere := runFrom_take_ok pub proof messages _ (stageIndex20 pub k') _ okNext
  rw [List.take_take, Nat.min_eq_left (Nat.le_succ _)] at okHere
  have chain : (N ++ (V ++ [Duplex.Event.squeeze 8,
        .absorb ((Op.work site level (bits20.flatten.getD k' 0)).input proof m)])) ++
        (stateBefore pub proof messages pub.ops (stageIndex20 pub k')).sponge.events =
      N' ++ (stateBefore pub proof' messages' pub.ops (stageIndex20 pub k')).sponge.events := by
    rw [← split', ← events, split, grown, worked]
    simp
  have alike : Alike (stateBefore pub proof messages pub.ops (stageIndex20 pub k'))
      (stateBefore pub proof' messages' pub.ops (stageIndex20 pub k')) :=
    run_alike pub proof proof' messages messages' _ _ _ (Alike.refl _) rfl okHere ran'
      (Faithful.take _ 0 _ (decode_faithful decoded decoded')) ⟨_, _, chain⟩
  have lengths := congrArg List.length chain
  rw [congrArg Duplex.events alike.1] at lengths
  simp only [List.length_append, List.length_cons, List.length_nil] at lengths
  omega

/-- With a work bit or more in each stage, the events before the squeeze of a work give the
stage. -/
theorem stage20_unique (bits : Bits20 pub) {events : List Duplex.Event} {k k' : ℕ}
    (stage : Stage20 pub events k proof messages)
    (stage' : Stage20 pub events k' proof' messages') : k = k' := by
  obtain ⟨inside, ⟨rest, decoded⟩, ran, same⟩ := stage
  obtain ⟨inside', ⟨rest', decoded'⟩, ran', same'⟩ := stage'
  rcases Nat.lt_trichotomy k k' with lt | eq | gt
  · exact absurd (same'.trans same.symm) (stage20_not_lt bits lt inside' decoded' decoded ran' ran)
  · exact eq
  · exact absurd (same.trans same'.symm) (stage20_not_lt bits gt inside decoded decoded' ran ran')

/-- The digest of the last block of `t` in the state before the work of the stage `k`: the bytes
of the operation `terminal_t_last_block_digest`, and no byte before this operation. -/
def tDigestAt (pub : Public) (proof : ByteArray) (messages : Array Message) (k : ℕ) : ByteArray :=
  (terminalMessagesIn (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) 6).tDigest

/-- The digest of the last block of `e` in the state before the work of the stage `k`: the bytes
of the operation `terminal.e_last_block_digest`, and no byte before this operation. -/
def eDigestAt (pub : Public) (proof : ByteArray) (messages : Array Message) (k : ℕ) : ByteArray :=
  (terminalMessagesIn (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) 6).eDigest

/-- The digests in the state before the work of a stage are a function of the events. -/
theorem digestAt_causal {events : List Duplex.Event} {k : ℕ}
    (stage : Stage20 pub events k proof messages)
    (stage' : Stage20 pub events k proof' messages') :
    tDigestAt pub proof messages k = tDigestAt pub proof' messages' k ∧
      eDigestAt pub proof messages k = eDigestAt pub proof' messages' k := by
  obtain ⟨inside, ⟨rest, decoded⟩, ran, same⟩ := stage
  obtain ⟨-, ⟨rest', decoded'⟩, ran', same'⟩ := stage'
  have core := core_causal pub proof proof' messages messages' _ decoded decoded' ran ran'
    (eventsAt20_parts (same.trans same'.symm)).2
  have bytes : ∀ key,
      (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).derived.bytes.get? key =
        (stateBefore pub proof' messages' pub.ops (stageIndex20 pub k)).derived.bytes.get? key :=
    fun key => congrFun (congrArg Core.bytes core) key
  refine ⟨?_, ?_⟩ <;> simp only [tDigestAt, eDigestAt, terminalMessagesIn, bytesIn, bytes]

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- The key of a stage is a function of the events before the squeeze of its work and of the last
blocks. -/
theorem addressAt20_causal {events : List Duplex.Event} {k : ℕ}
    (stage : Stage20 pub events k proof messages)
    (stage' : Stage20 pub events k proof' messages')
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
    (total : ℕ) :
    addressAt20 (E := E) pub proof messages tLast eLast total k =
      addressAt20 pub proof' messages' tLast eLast total k := by
  obtain ⟨inside, ⟨rest, decoded⟩, ran, same⟩ := stage
  obtain ⟨-, ⟨rest', decoded'⟩, ran', same'⟩ := stage'
  have causal := stage_causal inside decoded decoded' ran ran' (same.trans same'.symm)
  have page : ∀ j ≤ k, stagePage20 (E := E) pub proof messages tLast eLast j =
      stagePage20 pub proof' messages' tLast eLast j := fun j le =>
    pageAt20_causal _ _ (Nat.lt_of_le_of_lt le inside) decoded decoded'
      (stateBefore_le pub proof messages pub.ops (stageIndex20_mono pub le inside) ran).1
      (stateBefore_le pub proof' messages' pub.ops (stageIndex20_mono pub le inside) ran').1
      (causal j le).1
  refine Address.ext (page 0 (Nat.zero_le k)) ?_
  show (List.range (min k total)).map _ = (List.range (min k total)).map _
  refine List.map_congr_left fun j member => page (j + 1) ?_
  have below := List.mem_range.mp member
  omega

/-- On a script of `layout20`, the key of a stage gives the events before the squeeze of its
work, for each last blocks. -/
theorem addressAt20_events (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    {k : ℕ} (inside : k < 473) {total : ℕ} (long : k ≤ total) {rest rest' : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    {tLast tLast' : ℕ → Rq Key20.terminal.innerDegree}
    {eLast eLast' : Rq Key20.terminal.innerDegree}
    (same : addressAt20 (E := E) pub proof messages tLast eLast total k =
      addressAt20 pub proof' messages' tLast' eLast' total k) :
    eventsAt20 pub proof messages k = eventsAt20 pub proof' messages' k := by
  have page : ∀ j ≤ k, stagePage20 (E := E) pub proof messages tLast eLast j =
      stagePage20 pub proof' messages' tLast' eLast' j := by
    intro j le
    cases j with
    | zero => exact congrArg Address.first same
    | succ j =>
        have entry : (addressAt20 (E := E) pub proof messages tLast eLast total k).rest[j]? =
            (addressAt20 pub proof' messages' tLast' eLast' total k).rest[j]? := by rw [same]
        rw [addressAt20_rest tLast eLast long (by omega),
          addressAt20_rest tLast' eLast' long (by omega)] at entry
        exact Option.some.inj entry
  exact (stage_alike inside decoded decoded' ran ran' fun j le =>
    traceAt20_inj pub proof proof' messages messages' erased decoded decoded'
      (Nat.lt_of_le_of_lt le inside)
      ((stagePage20_trace (E := E) tLast eLast (Nat.lt_of_le_of_lt le inside)).symm.trans
        ((congrArg (fun w : Key20.Wire E => w.data.trace) (page j le)).trans
          (stagePage20_trace (E := E) tLast' eLast' (Nat.lt_of_le_of_lt le inside))))).2

end Stage

section Found

open scoped Classical in
/-- The last block of `t` of a digest in a record of digest queries (input, digest): the rows
`0, …, n_A − 1` of a block whose digest input has this digest in the record, and zero when the
record has no such block. -/
noncomputable def tFound (record : List (ByteArray × ByteArray)) (digest : ByteArray) :
    ℕ → Rq Key20.terminal.innerDegree :=
  if found : ∃ rows, (tInput Key20.terminal rows, digest) ∈ record then
    rowsOf Key20.terminal (Classical.choose found)
  else fun _ => 0

open scoped Classical in
/-- The last block of `e` of a digest in a record of digest queries (input, digest): a block whose
digest input has this digest in the record, and zero when the record has no such block. -/
noncomputable def eFound (record : List (ByteArray × ByteArray)) (digest : ByteArray) :
    Rq Key20.terminal.innerDegree :=
  if found : ∃ y : Rq Key20.terminal.innerDegree, (eInput y, digest) ∈ record then
    Classical.choose found
  else 0

variable {record record' : List (ByteArray × ByteArray)} {digest : ByteArray}

theorem tFound_eq {tLast : ℕ → Rq Key20.terminal.innerDegree}
    (listed : (tInput Key20.terminal tLast, digest) ∈ record)
    (single : ∀ input, (input, digest) ∈ record → input = tInput Key20.terminal tLast) :
    tFound record digest = rowsOf Key20.terminal tLast := by
  have found : ∃ rows, (tInput Key20.terminal rows, digest) ∈ record := ⟨tLast, listed⟩
  unfold tFound
  rw [dite_eq_left found]
  exact rowsOf_of_input (single _ (Classical.choose_spec found))

theorem eFound_eq {eLast : Rq Key20.terminal.innerDegree}
    (listed : (eInput eLast, digest) ∈ record)
    (single : ∀ input, (input, digest) ∈ record → input = eInput eLast) :
    eFound record digest = eLast := by
  have found : ∃ y : Rq Key20.terminal.innerDegree, (eInput y, digest) ∈ record :=
    ⟨eLast, listed⟩
  unfold eFound
  rw [dite_eq_left found]
  exact eInput_injective (single _ (Classical.choose_spec found))

theorem tFound_mono (sub : record ⊆ record')
    (single : ∀ a b d, (a, d) ∈ record' → (b, d) ∈ record' → a = b)
    (found : ∃ rows, (tInput Key20.terminal rows, digest) ∈ record) :
    tFound record' digest = tFound record digest := by
  obtain ⟨rows, listed⟩ := found
  rw [tFound_eq listed fun input other => single _ _ _ (sub other) (sub listed),
    tFound_eq (sub listed) fun input other => single _ _ _ other (sub listed)]

theorem eFound_mono (sub : record ⊆ record')
    (single : ∀ a b d, (a, d) ∈ record' → (b, d) ∈ record' → a = b)
    (found : ∃ y : Rq Key20.terminal.innerDegree, (eInput y, digest) ∈ record) :
    eFound record' digest = eFound record digest := by
  obtain ⟨y, listed⟩ := found
  rw [eFound_eq listed fun input other => single _ _ _ (sub other) (sub listed),
    eFound_eq (sub listed) fun input other => single _ _ _ other (sub listed)]

end Found

section Resolve

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]
  {pub : Public} {proof proof' : ByteArray} {messages messages' : Array Message}

/-- The key of the stage `k` of a proof with the last blocks of a record of digest queries: the
blocks of the digests in the state before the work of the stage `k`. -/
noncomputable def recordAddress20 (pub : Public) (record : List (ByteArray × ByteArray))
    (total k : ℕ) (proof : ByteArray) (messages : Array Message) :
    Address (Key20.Wire E) total :=
  addressAt20 pub proof messages (tFound record (tDigestAt pub proof messages k))
    (eFound record (eDigestAt pub proof messages k)) total k

theorem recordAddress20_causal {events : List Duplex.Event} {k : ℕ}
    (stage : Stage20 pub events k proof messages)
    (stage' : Stage20 pub events k proof' messages')
    (record : List (ByteArray × ByteArray)) (total : ℕ) :
    recordAddress20 (E := E) pub record total k proof messages =
      recordAddress20 pub record total k proof' messages' := by
  unfold recordAddress20
  rw [(digestAt_causal stage stage').1, (digestAt_causal stage stage').2]
  exact addressAt20_causal stage stage' _ _ total

open scoped Classical in
/-- `resolve` of Akita Lemma 10.47 for `key20.aks`: the key of the model for the events of a
stage. When `events` are the events before the squeeze of the work of a stage `k < 473` of a
proof, it is the key `recordAddress20` of a chosen such stage and proof, with the last blocks
from the record of the digest queries; it is `none` when no such stage exists. -/
noncomputable def resolve20 (total : ℕ) (pub : Public) (record : List (ByteArray × ByteArray))
    (events : List Duplex.Event) : Option (Address (Key20.Wire E) total) :=
  if found : ∃ stage : ℕ × ByteArray × Array Message,
      Stage20 pub events stage.1 stage.2.1 stage.2.2 then
    some (recordAddress20 pub record total (Classical.choose found).1 (Classical.choose found).2.1
      (Classical.choose found).2.2)
  else none

/-- With a work bit or more in each stage, `resolve20` on the events of a stage of a proof is the
key of this stage and this proof with the last blocks of the record. -/
theorem resolve20_stage (bits : Bits20 pub) {events : List Duplex.Event} {k : ℕ}
    (stage : Stage20 pub events k proof messages) (record : List (ByteArray × ByteArray))
    (total : ℕ) :
    resolve20 (E := E) total pub record events =
      some (recordAddress20 pub record total k proof messages) := by
  have found : ∃ stage : ℕ × ByteArray × Array Message,
      Stage20 pub events stage.1 stage.2.1 stage.2.2 := ⟨(k, proof, messages), stage⟩
  have chosen := Classical.choose_spec found
  have index : (Classical.choose found).1 = k := stage20_unique bits chosen stage
  unfold resolve20
  rw [dite_eq_left found]
  generalize Classical.choose found = w at chosen index ⊢
  obtain ⟨k', proof', messages'⟩ := w
  dsimp only at chosen index ⊢
  subst index
  exact congrArg some (recordAddress20_causal chosen stage record total)

/-- `resolve20` on the events of a stage `k` of a proof is the key with the last blocks `tLast`
and `eLast`, when the record of the digest queries has the digest input of each block that the
pages `0, …, k` have (the block of `t` for `379 ≤ k`, the block of `e` for `472 ≤ k`) with the
digest of the proof, and no other input with this digest. -/
theorem resolve20_eq (bits : Bits20 pub) {k : ℕ} (inside : k < 473) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    {record : List (ByteArray × ByteArray)} {tLast : ℕ → Rq Key20.terminal.innerDegree}
    {eLast : Rq Key20.terminal.innerDegree}
    (tListed : 379 ≤ k → (tInput Key20.terminal tLast, tDigestAt pub proof messages k) ∈ record ∧
      ∀ input, (input, tDigestAt pub proof messages k) ∈ record →
        input = tInput Key20.terminal tLast)
    (eListed : 472 ≤ k → (eInput eLast, eDigestAt pub proof messages k) ∈ record ∧
      ∀ input, (input, eDigestAt pub proof messages k) ∈ record → input = eInput eLast)
    (total : ℕ) :
    resolve20 (E := E) total pub record (eventsAt20 pub proof messages k) =
      some (addressAt20 pub proof messages tLast eLast total k) := by
  rw [resolve20_stage bits ⟨inside, ⟨rest, decoded⟩, ran, rfl⟩ record total]
  unfold recordAddress20
  refine congrArg some (addressAt20_congr (fun late => ?_) (fun late => ?_) total)
  · rw [tFound_eq (tListed late).1 (tListed late).2, rowsOf_rowsOf]
  · exact eFound_eq (eListed late).1 (eListed late).2

/-- A larger record with one input for each digest gives the same key, when the first record has
a block for each digest that the pages of the stage have. -/
theorem resolve20_mono (bits : Bits20 pub) {events : List Duplex.Event} {k : ℕ}
    (stage : Stage20 pub events k proof messages) {record record' : List (ByteArray × ByteArray)}
    (sub : record ⊆ record')
    (single : ∀ a b d, (a, d) ∈ record' → (b, d) ∈ record' → a = b)
    (tListed : 379 ≤ k →
      ∃ rows, (tInput Key20.terminal rows, tDigestAt pub proof messages k) ∈ record)
    (eListed : 472 ≤ k → ∃ y : Rq Key20.terminal.innerDegree,
      (eInput y, eDigestAt pub proof messages k) ∈ record)
    (total : ℕ) :
    resolve20 (E := E) total pub record' events = resolve20 total pub record events := by
  rw [resolve20_stage bits stage record' total, resolve20_stage bits stage record total]
  unfold recordAddress20
  exact congrArg some (addressAt20_congr
    (fun late => congrArg (rowsOf Key20.terminal) (tFound_mono sub single (tListed late)))
    (fun late => eFound_mono sub single (eListed late)) total)

/-- The key `address` is the key of the stage `address.rest.length < 473` of the proof with its
decoded messages, whose run has no stop before the work of this stage, for the last blocks
`tLast` and `eLast`. -/
def Presented20 {total : ℕ} (pub : Public) (address : Address (Key20.Wire E) total)
    (proof : ByteArray) (messages : Array Message) (tLast : ℕ → Rq Key20.terminal.innerDegree)
    (eLast : Rq Key20.terminal.innerDegree) : Prop :=
  address.rest.length < 473 ∧ (∃ rest, decode pub.ops proof = .ok (messages, rest)) ∧
    (runFrom pub proof messages
      (pub.ops.zipIdx.toList.take (stageIndex20 pub address.rest.length))
      (channelStart pub)).1 = .ok () ∧
    addressAt20 pub proof messages tLast eLast total address.rest.length = address

open scoped Classical in
/-- `back` of `Replay/Simulate.lean` for `key20.aks`: the events before the squeeze of the work of
the stage `address.rest.length` of a chosen proof that has the key `address` for some last
blocks, and no event when no such proof exists. -/
noncomputable def back20 {total : ℕ} (pub : Public) (address : Address (Key20.Wire E) total) :
    List Duplex.Event :=
  if found : ∃ run : ByteArray × Array Message × (ℕ → Rq Key20.terminal.innerDegree) ×
      Rq Key20.terminal.innerDegree,
      Presented20 pub address run.1 run.2.1 run.2.2.1 run.2.2.2 then
    eventsAt20 pub (Classical.choose found).1 (Classical.choose found).2.1 address.rest.length
  else []

/-- On a script of `layout20`, for each record: a key that `resolve20` gives for `events` has
`back20` equal to `events`. -/
theorem resolves20 (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {total : ℕ}
    (long : 472 ≤ total) (record : List (ByteArray × ByteArray)) (events : List Duplex.Event)
    (address : Address (Key20.Wire E) total)
    (resolved : resolve20 total pub record events = some address) :
    back20 pub address = events := by
  unfold resolve20 at resolved
  split at resolved
  · rename_i found
    have chosen := Classical.choose_spec found
    generalize Classical.choose found = w at resolved chosen
    obtain ⟨k, proof, messages⟩ := w
    dsimp only at resolved chosen
    obtain ⟨inside, ⟨rest, decoded⟩, ran, same⟩ := chosen
    have built := Option.some.inj resolved
    subst built
    have length : (recordAddress20 (E := E) pub record total k proof messages).rest.length = k :=
      length_addressAt20 _ _ (by omega)
    have presented : ∃ run : ByteArray × Array Message × (ℕ → Rq Key20.terminal.innerDegree) ×
        Rq Key20.terminal.innerDegree,
        Presented20 pub (recordAddress20 (E := E) pub record total k proof messages)
          run.1 run.2.1 run.2.2.1 run.2.2.2 := by
      refine ⟨(proof, messages, tFound record (tDigestAt pub proof messages k),
        eFound record (eDigestAt pub proof messages k)), ?_⟩
      unfold Presented20
      rw [length]
      exact ⟨inside, ⟨rest, decoded⟩, ran, rfl⟩
    unfold back20
    rw [dite_eq_left presented]
    have spec := Classical.choose_spec presented
    generalize Classical.choose presented = v at spec ⊢
    obtain ⟨proof', messages', tLast', eLast'⟩ := v
    unfold Presented20 at spec
    dsimp only at spec ⊢
    rw [length] at spec ⊢
    obtain ⟨-, ⟨rest', decoded'⟩, ran', same'⟩ := spec
    rw [← same]
    exact addressAt20_events erased inside (by omega) decoded' decoded ran' ran same'
  · cases resolved

end Resolve

section Reads

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]
  {pub : Public} {proof : ByteArray} {messages : Array Message}

/-- The readings of `Key20.Tower.accepts` on the pages `stagePage20` of an accepted proof of
`layout20` are those of `note20`: the tower reads the last block of `t` only in the pages 379 and
472, and the last block of `e` only in the page 472. -/
theorem stagePage20_reads {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept)
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree) :
    Reads20 pub proof messages tLast eLast fun k =>
      (stagePage20 (E := E) pub proof messages tLast eLast k).data := by
  have reads := noteAt20_reads (E := E) pub proof messages tLast eLast layout accepted
  refine ⟨reads.commit, reads.value, reads.place, reads.claim, reads.commit₁, reads.claim₁,
    reads.commit₂, reads.claim₂, reads.commit₃, reads.claim₃, reads.commit₄, reads.claim₄,
    reads.commit₅, reads.claim₅, ?_, ?_, ?_, ?_⟩
  · show (stagePage20 (E := E) pub proof messages tLast eLast 379).data.state = _
    rw [stagePage20_middle tLast eLast (by decide) (by decide)]
    exact reads.state
  · show (stagePage20 (E := E) pub proof messages tLast eLast 472).data.statement = _
    rw [stagePage20_last tLast eLast (by decide)]
    exact reads.statement
  · show (stagePage20 (E := E) pub proof messages tLast eLast 472).data.partials = _
    rw [stagePage20_last tLast eLast (by decide)]
    exact reads.partials
  · show (stagePage20 (E := E) pub proof messages tLast eLast 473).data.response = _
    rw [stagePage20_last tLast eLast (by decide)]
    exact reads.response

end Reads

end NightstreamFPrime.Spec.Compression.Akita.Replay
