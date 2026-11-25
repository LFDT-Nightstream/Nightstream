import NightstreamFPrime.Spec.Compression.Akita.Replay.Causal
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeRange

/-! The events of the sponge before the squeeze of the work of a stage are an injective and a
causal image of the byte strings of the stages, for `key20.aks` (Akita revision `d50f700`;
`docs/akita-work/w20/design.md`, decisions D2 and D3; `Replay/StageTrace.lean`,
`Replay/Causal.lean`).

- `inputsFrom_append`, `length_inputsFrom`, `inputsFrom_slice`: the absorbed bytes of a list of
  operations; their count is the count of the reading operations.
- `run_alike_of_inputs`: two runs without a stop of the same operations from alike states, with
  faithful messages and the same absorbed bytes, end in alike states. `run_inputs`: with the same
  events at the end, they have the same absorbed bytes.
- `traceOf_inj`: `traceOf` is injective on lists that fit its caps.
- `skeleton20_fits`: each stage of the skeleton of `layout20` reads at most 5 messages, no payload
  of `z`, and each message has at most 16384 absorbed bytes. `skeleton20_draws`: the draws of a
  stage are directly after its work: one `.challenge` of at most 22 values, or `.squeeze`
  operations. `workIndices20_mono`: the indices of the works grow.
- `inputs20_fit` and `traceAt20_inj`: on a script of `layout20`, `inputs20 … k` fits the caps of
  the trace, and `traceAt20 … k` gives it.
- `eventsAt20 … k` are the events of the sponge before the squeeze of the work of the stage `k`:
  the absorb of the nonce of the work, then the events before the work.
- `stage_alike` (pages give events): two proofs with the same `inputs20 … j` for each `j ≤ k`
  have alike states before the work of the stage `k` and the same `eventsAt20 … k`.
- `stage_causal` (events give pages): two proofs with the same `eventsAt20 … k` have the same
  `eventsAt20 … j` and `inputs20 … j` for each `j ≤ k`. `noteAt20_causal`: they have the same
  `noteAt20 … k`.
- `SameIdeal`, `step_kept`, `stateBefore_ideal`: the replay keeps the answers of an ideal sponge.
  `work_squeeze20`: the work of the stage `k` squeezes the answer for `eventsAt20 … k`, and
  `work_passed20`: this answer has the work.
  `draw_squeeze20`: an operation after draws that follow the work squeezes the answer for the
  squeezes before it and `eventsAt20 … k`. `work20_of` and `draws20_of`: on a script of
  `layout20`, the operation at `stageIndex20 pub k` is a work, and the `drawCount20 k` operations
  directly after it are draws.

The bits of a work are those of the plan of the public input. A work with no bits reads no
message and squeezes no byte, so `work_squeeze20` and `draw_squeeze20` take a work with bits. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita

section Lists

theorem take_cut {α : Type} (l : List α) {a b : ℕ} (le : a ≤ b) :
    l.take b = l.take a ++ (l.take b).drop a := by
  conv_lhs => rw [← List.take_append_drop a (l.take b)]
  rw [List.take_take, Nat.min_eq_left le]

theorem mem_slice {α : Type} (l : List α) {a c c' : ℕ} (le : c ≤ c') {x : α}
    (member : x ∈ (l.take c).drop a) : x ∈ (l.take c').drop a := by
  have smaller : l.take c = (l.take c').take c := by rw [List.take_take, Nat.min_eq_left le]
  rw [smaller, List.drop_take] at member
  exact List.mem_of_mem_take member

theorem readCount_cons (x : Op × ℕ) (l : List (Op × ℕ)) :
    readCount (x :: l) = (if x.1.reads then 1 else 0) + readCount l := by
  unfold readCount
  rw [List.filter_cons]
  by_cases reads : x.1.reads = true
  · rw [ite_eq_left reads, ite_eq_left reads, List.length_cons, Nat.add_comm]
  · rw [ite_eq_right reads, ite_eq_right reads, Nat.zero_add]

theorem readCount_append (a b : List (Op × ℕ)) :
    readCount (a ++ b) = readCount a + readCount b := by
  unfold readCount
  rw [List.filter_append, List.length_append]

variable (proof proof' : ByteArray) (messages messages' : Array Message)

theorem inputsFrom_append : ∀ (a b : List (Op × ℕ)) (n : ℕ),
    inputsFrom proof messages (a ++ b) n =
      inputsFrom proof messages a n ++ inputsFrom proof messages b (n + readCount a)
  | [], _, _ => rfl
  | x :: a, b, n => by
      rw [List.cons_append, inputsFrom, inputsFrom, inputsFrom_append a b, readCount_cons,
        List.append_assoc, Nat.add_assoc]

/-- The count of the absorbed strings is the count of the reading operations. -/
theorem length_inputsFrom : ∀ (l : List (Op × ℕ)) (n : ℕ),
    (inputsFrom proof messages l n).length = readCount l
  | [], _ => rfl
  | x :: l, n => by
      rw [inputsFrom, List.length_append, length_inputsFrom l, readCount_cons]
      by_cases reads : x.1.reads = true
      · rw [ite_eq_left reads, ite_eq_left reads]
        rfl
      · rw [ite_eq_right reads, ite_eq_right reads]
        rfl

variable {proof proof' messages messages'}

theorem inputsFrom_mid (u v w : List (Op × ℕ))
    (same : inputsFrom proof messages (u ++ v ++ w) 0 =
      inputsFrom proof' messages' (u ++ v ++ w) 0) :
    inputsFrom proof messages v (readCount u) = inputsFrom proof' messages' v (readCount u) := by
  rw [inputsFrom_append, inputsFrom_append, inputsFrom_append, inputsFrom_append] at same
  have front := (List.append_inj same (by
    rw [List.length_append, List.length_append, length_inputsFrom, length_inputsFrom,
      length_inputsFrom, length_inputsFrom])).1
  have mid := (List.append_inj front (by rw [length_inputsFrom, length_inputsFrom])).2
  rw [Nat.zero_add] at mid
  exact mid

/-- Two proofs with the same absorbed bytes of the first `c` operations have the same absorbed
bytes of the operations from `a` to `b ≤ c`. -/
theorem inputsFrom_slice (l : List (Op × ℕ)) {a b c : ℕ} (le : a ≤ b) (le' : b ≤ c)
    (same : inputsFrom proof messages (l.take c) 0 = inputsFrom proof' messages' (l.take c) 0) :
    inputsFrom proof messages ((l.take b).drop a) (readCount (l.take a)) =
      inputsFrom proof' messages' ((l.take b).drop a) (readCount (l.take a)) := by
  have cut : l.take c = l.take a ++ (l.take b).drop a ++ (l.take c).drop b := by
    rw [← take_cut l le]
    exact take_cut l le'
  rw [cut] at same
  exact inputsFrom_mid _ _ _ same

theorem ReadsFrom.take : ∀ (l : List (Op × ℕ)) (n c : ℕ), ReadsFrom proof messages l n →
    ReadsFrom proof messages (l.take c) n
  | [], _, _, _ => by
      rw [List.take_nil]
      trivial
  | _ :: _, _, 0, _ => trivial
  | _ :: l, _, c + 1, reads => ⟨reads.1, ReadsFrom.take l _ c reads.2⟩

theorem ReadsFrom.append : ∀ (a b : List (Op × ℕ)) (n : ℕ),
    ReadsFrom proof messages (a ++ b) n → ReadsFrom proof messages b (n + readCount a)
  | [], _, _, reads => reads
  | x :: a, b, n, reads => by
      rw [readCount_cons, ← Nat.add_assoc]
      exact ReadsFrom.append a b _ reads.2

end Lists

namespace Replayer

section Runs

variable (pub : Public) (proof proof' : ByteArray) (messages messages' : Array Message)

/-- Two steps without a stop of one reading operation from alike states, with faithful messages
and the same absorbed bytes, end in alike states. -/
theorem read_alike_of_input (i : ℕ) (op : Op) (reads : op.reads = true) {s s' : Replayer}
    (alike : Alike s s') (next : s.next = s'.next)
    (ran : (((step pub proof messages i op).run).run s).1 = .ok ())
    (ran' : (((step pub proof' messages' i op).run).run s').1 = .ok ())
    (faithful : op.stores = true → ∀ m m', messages[s.next]? = some m →
      messages'[s.next]? = some m' → op.input proof m = op.input proof' m' → m.data = m'.data)
    (input : inputAt proof messages op s.next = inputAt proof' messages' op s.next) :
    Alike (((step pub proof messages i op).run).run s).2
      (((step pub proof' messages' i op).run).run s').2 := by
  obtain ⟨m, found, sponge, cored⟩ := step_read pub proof messages i op s reads ran
  obtain ⟨m', found', sponge', cored'⟩ := step_read pub proof' messages' i op s' reads ran'
  rw [← next] at found'
  simp only [inputAt, found, found'] at input
  refine ⟨by rw [sponge, sponge', alike.1, input], ?_⟩
  rw [cored, cored', alike.2]
  by_cases stores : op.stores = true
  · rw [faithful stores m m' found found' input]
  · rw [store_quiet (Bool.eq_false_iff.mpr stores), store_quiet (Bool.eq_false_iff.mpr stores)]

/-- Two runs without a stop of the same operations from alike states, with faithful messages and
the same absorbed bytes, end in alike states. -/
theorem run_alike_of_inputs : ∀ (l : List (Op × ℕ)) (s s' : Replayer), Alike s s' →
    s.next = s'.next →
    (runFrom pub proof messages l s).1 = .ok () → (runFrom pub proof' messages' l s').1 = .ok () →
    Faithful proof proof' messages messages' l s.next →
    inputsFrom proof messages l s.next = inputsFrom proof' messages' l s.next →
    Alike (runFrom pub proof messages l s).2 (runFrom pub proof' messages' l s').2
  | [], _, _, alike, _, _, _, _, _ => alike
  | x :: l, s, s', alike, next, ran, ran', faithful, inputs => by
      obtain ⟨stepped, tail⟩ := (runFrom_cons_fst pub proof messages _ _ _).mp ran
      obtain ⟨stepped', tail'⟩ := (runFrom_cons_fst pub proof' messages' _ _ _).mp ran'
      rw [runFrom_cons_snd pub proof messages _ _ _ stepped,
        runFrom_cons_snd pub proof' messages' _ _ _ stepped']
      rw [inputsFrom, inputsFrom] at inputs
      obtain ⟨head, rest⟩ := List.append_inj inputs (by split <;> rfl)
      have after : Alike (((step pub proof messages x.2 x.1).run).run s).2
          (((step pub proof' messages' x.2 x.1).run).run s').2 := by
        by_cases reads : x.1.reads = true
        · refine read_alike_of_input pub proof proof' messages messages' x.2 x.1 reads alike next
            stepped stepped' faithful.1 ?_
          simp only [ite_eq_left reads] at head
          exact List.head_eq_of_cons_eq head
        · by_cases still : (∃ key check, x.1 = .check key check) ∨ ∃ level, x.1 = .terminal level
          · exact (step_inert pub proof messages x.2 x.1 s still).symm.trans
              (alike.trans (step_inert pub proof' messages' x.2 x.1 s' still))
          · exact (step_sim pub proof proof' messages messages' x.2 x.1
              (Bool.eq_false_iff.mpr reads) (fun key check same => still (Or.inl ⟨key, check, same⟩))
              (fun level same => still (Or.inr ⟨level, same⟩)) s s' alike).2
      have moved := step_next pub proof messages x.2 x.1 s stepped
      have moved' := step_next pub proof' messages' x.2 x.1 s' stepped'
      refine run_alike_of_inputs l _ _ after (by rw [moved, moved', next]) tail tail' ?_ ?_
      · rw [moved]
        exact faithful.2
      · rw [moved]
        exact rest

/-- Two steps without a stop of one reading operation from alike states, with events after them
that agree up to the events of the later operations, absorb the same bytes. -/
theorem read_input (i : ℕ) (op : Op) (reads : op.reads = true) {s s' : Replayer}
    (alike : Alike s s') (next : s.next = s'.next)
    (ran : (((step pub proof messages i op).run).run s).1 = .ok ())
    (ran' : (((step pub proof' messages' i op).run).run s').1 = .ok ())
    (later : ∃ T T', T ++ (((step pub proof messages i op).run).run s).2.sponge.events =
      T' ++ (((step pub proof' messages' i op).run).run s').2.sponge.events) :
    inputAt proof messages op s.next = inputAt proof' messages' op s.next := by
  obtain ⟨m, found, sponge, -⟩ := step_read pub proof messages i op s reads ran
  obtain ⟨m', found', sponge', -⟩ := step_read pub proof' messages' i op s' reads ran'
  rw [← next] at found'
  obtain ⟨pre, tail⟩ := op.tail_events
  obtain ⟨T, T', agree⟩ := later
  rw [sponge, sponge', tail, tail, Duplex.absorb_events, Duplex.absorb_events, ← alike.1,
    ← List.append_assoc, ← List.append_assoc] at agree
  have cut := (List.append_inj' agree rfl).2
  simp only [inputAt, found, found']
  injection cut with head
  injection head

/-- Two runs without a stop of the same operations from alike states, with faithful messages and
events at the end that agree up to a prefix, absorb the same bytes. -/
theorem run_inputs : ∀ (l : List (Op × ℕ)) (s s' : Replayer), Alike s s' → s.next = s'.next →
    (runFrom pub proof messages l s).1 = .ok () → (runFrom pub proof' messages' l s').1 = .ok () →
    Faithful proof proof' messages messages' l s.next →
    (∃ T T', T ++ (runFrom pub proof messages l s).2.sponge.events =
      T' ++ (runFrom pub proof' messages' l s').2.sponge.events) →
    inputsFrom proof messages l s.next = inputsFrom proof' messages' l s.next
  | [], _, _, _, _, _, _, _, _ => rfl
  | x :: l, s, s', alike, next, ran, ran', faithful, later => by
      obtain ⟨stepped, tail⟩ := (runFrom_cons_fst pub proof messages _ _ _).mp ran
      obtain ⟨stepped', tail'⟩ := (runFrom_cons_fst pub proof' messages' _ _ _).mp ran'
      rw [runFrom_cons_snd pub proof messages _ _ _ stepped,
        runFrom_cons_snd pub proof' messages' _ _ _ stepped'] at later
      obtain ⟨T, T', agree⟩ := later
      obtain ⟨U, grown⟩ := runFrom_suffix pub proof messages l
        (((step pub proof messages x.2 x.1).run).run s).2
      obtain ⟨U', grown'⟩ := runFrom_suffix pub proof' messages' l
        (((step pub proof' messages' x.2 x.1).run).run s').2
      have here : ∃ T T', T ++ (((step pub proof messages x.2 x.1).run).run s).2.sponge.events =
          T' ++ (((step pub proof' messages' x.2 x.1).run).run s').2.sponge.events := by
        refine ⟨T ++ U, T' ++ U', ?_⟩
        rw [List.append_assoc, List.append_assoc, ← grown, ← grown']
        exact agree
      have after : Alike (((step pub proof messages x.2 x.1).run).run s).2
          (((step pub proof' messages' x.2 x.1).run).run s').2 := by
        by_cases reads : x.1.reads = true
        · exact read_alike pub proof proof' messages messages' x.2 x.1 reads alike next stepped
            stepped' faithful.1 here
        · by_cases still : (∃ key check, x.1 = .check key check) ∨ ∃ level, x.1 = .terminal level
          · exact (step_inert pub proof messages x.2 x.1 s still).symm.trans
              (alike.trans (step_inert pub proof' messages' x.2 x.1 s' still))
          · exact (step_sim pub proof proof' messages messages' x.2 x.1
              (Bool.eq_false_iff.mpr reads) (fun key check same => still (Or.inl ⟨key, check, same⟩))
              (fun level same => still (Or.inr ⟨level, same⟩)) s s' alike).2
      have moved := step_next pub proof messages x.2 x.1 s stepped
      have moved' := step_next pub proof' messages' x.2 x.1 s' stepped'
      have rest := run_inputs l _ _ after (by rw [moved, moved', next]) tail tail'
        (by rw [moved]; exact faithful.2) ⟨T, T', agree⟩
      rw [moved] at rest
      rw [inputsFrom, inputsFrom, rest]
      by_cases reads : x.1.reads = true
      · simp only [ite_eq_left reads]
        rw [read_input pub proof proof' messages messages' x.2 x.1 reads alike next stepped stepped'
          here]
      · simp only [ite_eq_right reads]

end Runs

end Replayer

open Replayer

section Trace

theorem length_bytesOf (bytes : ByteArray) : (bytesOf bytes).length = bytes.size := by
  unfold bytesOf
  rw [List.length_map, Array.length_toList, ByteArray.size_data]

theorem bytesOf_injective : Function.Injective bytesOf := by
  intro a b same
  have lists : a.data.toList = b.data.toList :=
    (List.map_injective_iff (f := fun byte : UInt8 => (⟨byte.toNat, byte.toNat_lt⟩ : Fin 256))).mpr
      (fun (x y : UInt8) (h : (⟨x.toNat, x.toNat_lt⟩ : Fin 256) = ⟨y.toNat, y.toNat_lt⟩) =>
        UInt8.toNat_inj.mp (congrArg Fin.val h)) same
  exact ByteArray.ext (Array.toList_inj.mp lists)

/-- The strings of the trace of a list that fits the caps are the strings of the list. -/
theorem traceOf_strings {count bytes : ℕ} {strings : List ByteArray}
    (short : strings.length ≤ count) (fits : ∀ string ∈ strings, string.size ≤ bytes) :
    (traceOf count bytes strings).1.map Subtype.val = strings.map bytesOf := by
  show List.map Subtype.val ((strings.take count).map _) = strings.map bytesOf
  rw [List.take_of_length_le short, List.map_map]
  refine List.map_congr_left fun string member => ?_
  exact List.take_of_length_le (by rw [length_bytesOf]; exact fits string member)

/-- `traceOf` is injective on the lists of at most `count` strings of at most `bytes` bytes. -/
theorem traceOf_inj {count bytes : ℕ} {a b : List ByteArray} (short : a.length ≤ count)
    (short' : b.length ≤ count) (fits : ∀ string ∈ a, string.size ≤ bytes)
    (fits' : ∀ string ∈ b, string.size ≤ bytes)
    (same : traceOf count bytes a = traceOf count bytes b) : a = b := by
  have lists : a.map bytesOf = b.map bytesOf := by
    rw [← traceOf_strings short fits, ← traceOf_strings short' fits', same]
  exact List.map_injective_iff.mpr bytesOf_injective lists

end Trace

section Skeleton

/-- The operation has a message in a script with its work bits: it reads a message, or it is a
work. -/
def Op.sends : Op → Bool
  | .fields .. | .exts .. | .integer .. | .digest .. | .bounded .. | .foldNonce .. | .work .. => true
  | _ => false

/-- The most bytes that the step of the operation absorbs for its message; 0 for the payload of
`z`, which has no cap in the skeleton. -/
def Op.cap : Op → ℕ
  | .fields _ count => 8 * count
  | .exts _ count => 16 * count
  | .integer _ => 16
  | .digest _ => 32
  | .work .. | .foldNonce .. => 8
  | _ => 0

/-- The operation reads the payload of `z`. -/
def isBounded : Op → Bool
  | .bounded .. => true
  | _ => false

theorem Op.sends_erase {op : Op} (reads : op.reads = true) : op.erase.sends = true := by
  cases op <;> first | rfl | simp [Op.reads] at reads

theorem Op.cap_erase (op : Op) : op.erase.cap = op.cap := by cases op <;> rfl

theorem isBounded_erase (op : Op) : isBounded op.erase = isBounded op := by cases op <;> rfl

theorem isDraw_erase (op : Op) : isDraw op.erase = isDraw op := by cases op <;> rfl

/-- The operations of the skeleton of `layout20` after the work of the stage `k - 1` up to and
with the work of the stage `k`. -/
def stageOps20 (k : ℕ) : List (Op × ℕ) :=
  (sk20.take (workIndices20.getD k 0 + 1)).drop
    (if k = 0 then 0 else workIndices20.getD (k - 1) 0 + 1)

/-- The stage `k` has at most 5 messages, no payload of `z`, and each message has at most 16384
absorbed bytes. -/
def fits20 (k : ℕ) : Bool :=
  decide (((stageOps20 k).filter fun x => x.1.sends).length ≤ 5) &&
    (stageOps20 k).all fun x => !isBounded x.1 && decide (x.1.cap ≤ 16384)

/-- Each stage of the skeleton of `layout20` has at most 5 messages of at most 16384 absorbed
bytes, and no payload of `z`. -/
theorem skeleton20_fits : (List.range 473).all fits20 = true := by decide +kernel

/-- The indices of the works of the skeleton of `layout20` grow. -/
theorem workIndices20_step : (List.range 472).all (fun k =>
    decide (workIndices20.getD k 0 < workIndices20.getD (k + 1) 0)) = true := by decide +kernel

theorem workIndices20_lt {k : ℕ} (inside : k + 1 < 473) :
    workIndices20.getD k 0 < workIndices20.getD (k + 1) 0 :=
  of_decide_eq_true (List.all_eq_true.mp workIndices20_step k (List.mem_range.mpr (by omega)))

theorem workIndices20_mono {j k : ℕ} (le : j ≤ k) (inside : k < 473) :
    workIndices20.getD j 0 ≤ workIndices20.getD k 0 := by
  induction k with
  | zero =>
      have zero : j = 0 := by omega
      subst zero
      exact Nat.le_refl _
  | succ k ih =>
      rcases Nat.lt_or_ge j (k + 1) with lt | ge
      · exact Nat.le_trans (ih (by omega) (by omega)) (Nat.le_of_lt (workIndices20_lt inside))
      · have last : j = k + 1 := by omega
        subst last
        exact Nat.le_refl _

/-- The count of the draws directly after the work of the stage `k`. -/
def drawCount20 (k : ℕ) : ℕ :=
  ((sk20.drop (workIndices20.getD k 0 + 1)).takeWhile fun x => isDraw x.1).length

/-- The draws of the stage `k`: the operations directly after its work. -/
def stageDraws20 (k : ℕ) : List (Op × ℕ) :=
  (sk20.take (workIndices20.getD k 0 + 1 + drawCount20 k)).drop (workIndices20.getD k 0 + 1)

/-- The draws of the stage `k` are one `.challenge` of at most 22 values, or `.squeeze`
operations and one or more of them, and no operation after them and before the next work is a
draw. -/
def drawsFit20 (k : ℕ) : Bool :=
  (match stageDraws20 k with
    | [(.challenge _ count, _)] => decide (count ≤ 22)
    | draws => !draws.isEmpty && draws.all fun x =>
        match x.1 with
        | .squeeze .. => true
        | _ => false) &&
    (stageDraws20 k).all (fun x => isDraw x.1) &&
    ((sk20.drop (workIndices20.getD k 0 + 1 + drawCount20 k)).takeWhile fun x => !isWork x.1).all
      fun x => !isDraw x.1

/-- The draws of each stage of the skeleton of `layout20` are directly after its work: one
`.challenge` of at most 22 values, or `.squeeze` operations. -/
theorem skeleton20_draws : (List.range 473).all drawsFit20 = true := by decide +kernel

end Skeleton

section Fit

/-- The absorbed bytes of a message that `Wire.read` gives, other than the payload of `z`, are at
most `Op.cap` bytes. -/
theorem input_size {proof : ByteArray} {at_ : ℕ} {op : Op} {m : Message}
    (read : Wire.read proof at_ op = some (some m)) (open_ : isBounded op = false) :
    (op.input proof m).size ≤ op.cap := by
  cases op with
  | fields key count =>
      simp only [Wire.read, Option.map_eq_some_iff] at read
      obtain ⟨v, -, made⟩ := read
      cases made
      show (proof.extract at_ (at_ + 8 * count)).size ≤ 8 * count
      rw [ByteArray.size_extract]
      omega
  | exts key count =>
      simp only [Wire.read, Option.map_eq_some_iff] at read
      obtain ⟨v, -, made⟩ := read
      cases made
      show (proof.extract at_ (at_ + 16 * count)).size ≤ 16 * count
      rw [ByteArray.size_extract]
      omega
  | integer key =>
      simp only [Wire.read] at read
      split at read
      · cases read
      · cases read
        show (proof.extract at_ (at_ + 16)).size ≤ 16
        rw [ByteArray.size_extract]
        omega
  | digest key =>
      simp only [Wire.read] at read
      split at read
      · cases read
      · cases read
        show (proof.extract at_ (at_ + 32)).size ≤ 32
        rw [ByteArray.size_extract]
        omega
  | bounded key most => cases open_
  | work site level bits =>
      simp only [Wire.read] at read
      split at read
      · cases read
      · simp only [Option.map_eq_some_iff] at read
        obtain ⟨⟨value, size, canonical⟩, -, made⟩ := read
        cases made
        exact Nat.le_of_eq (le64_size value)
  | foldNonce level width =>
      simp only [Wire.read, Option.bind_eq_some_iff] at read
      obtain ⟨⟨value, size, canonical⟩, -, made⟩ := read
      dsimp only at made
      split at made
      · cases made
        exact Nat.le_of_eq (le64_size value)
      · cases made
  | _ => simp [Wire.read] at read

theorem inputsFrom_fit {proof : ByteArray} {messages : Array Message} (most : ℕ) :
    ∀ (l : List (Op × ℕ)) (n : ℕ), ReadsFrom proof messages l n →
      (∀ x ∈ l, isBounded x.1 = false ∧ x.1.cap ≤ most) →
      ∀ string ∈ inputsFrom proof messages l n, string.size ≤ most
  | [], _, _, _, _, member => by cases member
  | x :: l, n, reads, caps, string, member => by
      rw [inputsFrom, List.mem_append] at member
      rcases member with head | tail
      · by_cases reading : x.1.reads = true
        · rw [ite_eq_left reading, List.mem_singleton] at head
          obtain ⟨m, at_, found, read⟩ := reads.1 reading
          have cap := caps x List.mem_cons_self
          rw [head]
          simp only [inputAt, found]
          exact Nat.le_trans (input_size read cap.1) cap.2
        · rw [ite_eq_right reading] at head
          cases head
      · exact inputsFrom_fit most l _ reads.2
          (fun y inside => caps y (List.mem_cons_of_mem _ inside)) string tail

theorem readCount_le : ∀ l : List (Op × ℕ),
    readCount l ≤ ((l.map eraseAt).filter fun x => x.1.sends).length
  | [] => Nat.le_refl _
  | x :: l => by
      have ih := readCount_le l
      rw [readCount_cons, List.map_cons, List.filter_cons]
      by_cases reads : x.1.reads = true
      · have sends : (eraseAt x).1.sends = true := Op.sends_erase reads
        rw [ite_eq_left reads, ite_eq_left sends, List.length_cons]
        omega
      · rw [ite_eq_right reads]
        split
        · rw [List.length_cons]
          omega
        · omega

variable (pub : Public) (proof proof' : ByteArray) (messages messages' : Array Message)

theorem stageStart20_lt {k : ℕ} (inside : k < 473) :
    stageStart20 pub k = if k = 0 then 0 else workIndices20.getD (k - 1) 0 + 1 := by
  unfold stageStart20
  by_cases zero : k = 0
  · rw [ite_eq_left zero, ite_eq_left zero]
  · rw [ite_eq_right zero, ite_eq_right zero, stageIndex20_lt pub (by omega)]

theorem stageStart20_succ (k : ℕ) : stageStart20 pub (k + 1) = stageIndex20 pub k + 1 := by
  unfold stageStart20
  rw [ite_eq_right (Nat.succ_ne_zero k), Nat.add_sub_cancel]

theorem stageStart20_le {k : ℕ} (inside : k < 473) : stageStart20 pub k ≤ stageIndex20 pub k := by
  rw [stageStart20_lt pub inside, stageIndex20_lt pub inside]
  by_cases zero : k = 0
  · rw [ite_eq_left zero]
    exact Nat.zero_le _
  · rw [ite_eq_right zero]
    have step := workIndices20_lt (k := k - 1) (by omega)
    rw [show k - 1 + 1 = k by omega] at step
    exact step

theorem stageIndex20_mono {j k : ℕ} (le : j ≤ k) (inside : k < 473) :
    stageIndex20 pub j ≤ stageIndex20 pub k := by
  rw [stageIndex20_lt pub inside, stageIndex20_lt pub (by omega)]
  exact workIndices20_mono le inside

/-- On a script of `layout20`, the byte strings of a stage `k < 473` fit the caps of the trace:
at most 5 strings of at most 16384 bytes. -/
theorem inputs20_fit (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) {k : ℕ} (inside : k < 473) :
    (inputs20 pub proof messages k).length ≤ 5 ∧
      ∀ string ∈ inputs20 pub proof messages k, string.size ≤ 16384 := by
  have fit := List.all_eq_true.mp skeleton20_fits k (List.mem_range.mpr inside)
  unfold fits20 at fit
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, Bool.not_eq_true'] at fit
  obtain ⟨count, each⟩ := fit
  have segment : ((pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)).drop
      (stageStart20 pub k)).map eraseAt = stageOps20 k := by
    rw [List.map_drop, take_erase erased, stageStart20_lt pub inside, stageIndex20_lt pub inside]
    rfl
  refine ⟨?_, ?_⟩
  · unfold inputs20
    rw [length_inputsFrom]
    refine Nat.le_trans (readCount_le _) ?_
    rw [segment]
    exact count
  · unfold inputs20
    have reads : ReadsFrom proof messages
        ((pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)).drop (stageStart20 pub k))
        (readCount (pub.ops.zipIdx.toList.take (stageStart20 pub k))) := by
      have whole := ReadsFrom.take _ 0 (stageIndex20 pub k + 1) (decode_from decoded)
      rw [take_cut pub.ops.zipIdx.toList
        (Nat.le_trans (stageStart20_le pub inside) (Nat.le_succ _))] at whole
      have slice := ReadsFrom.append _ _ 0 whole
      rw [Nat.zero_add] at slice
      exact slice
    refine inputsFrom_fit 16384 _ _ reads fun x member => ?_
    have inside' : eraseAt x ∈ stageOps20 k := by
      rw [← segment]
      exact List.mem_map_of_mem member
    have ok := each (eraseAt x) inside'
    exact ⟨by rw [← isBounded_erase]; exact ok.1, by rw [← Op.cap_erase]; exact ok.2⟩

/-- On a script of `layout20`, the trace of a stage `k < 473` gives the byte strings of the
stage. -/
theorem traceAt20_inj (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    {rest rest' : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest')) {k : ℕ} (inside : k < 473)
    (same : traceAt20 pub proof messages k = traceAt20 pub proof' messages' k) :
    inputs20 pub proof messages k = inputs20 pub proof' messages' k := by
  obtain ⟨short, fits⟩ := inputs20_fit pub proof messages erased decoded inside
  obtain ⟨short', fits'⟩ := inputs20_fit pub proof' messages' erased decoded' inside
  exact traceOf_inj short short' fits fits' same

end Fit

section Stages

variable (pub : Public) (proof proof' : ByteArray) (messages messages' : Array Message)

/-- The events of the sponge directly before the squeeze of the work of the stage `k`: the absorb
of the nonce of the work, then the events of the state before the work. -/
def eventsAt20 (k : ℕ) : List Duplex.Event :=
  (inputsFrom proof messages
      ((pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)).drop (stageIndex20 pub k))
      (readCount (pub.ops.zipIdx.toList.take (stageIndex20 pub k)))).map Duplex.Event.absorb ++
    (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events

theorem inputs20_zero :
    inputs20 pub proof messages 0 =
      inputsFrom proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub 0 + 1)) 0 := rfl

variable {pub proof proof' messages messages'}

/-- Two proofs with the same byte strings of the stages `0 … k` absorb the same bytes up to and
with the work of the stage `k`. -/
theorem inputs_upto : ∀ {k : ℕ}, k < 473 →
    (∀ j ≤ k, inputs20 pub proof messages j = inputs20 pub proof' messages' j) →
    inputsFrom proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)) 0 =
      inputsFrom proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)) 0
  | 0, _, same => by
      rw [← inputs20_zero, ← inputs20_zero]
      exact same 0 (Nat.le_refl 0)
  | k + 1, inside, same => by
      have before := inputs_upto (k := k) (by omega) fun j le => same j (Nat.le_succ_of_le le)
      have here := same (k + 1) (Nat.le_refl _)
      unfold inputs20 at here
      rw [stageStart20_succ] at here
      rw [take_cut pub.ops.zipIdx.toList (Nat.succ_le_succ
          (stageIndex20_mono pub (Nat.le_succ k) inside)),
        inputsFrom_append, inputsFrom_append, before, Nat.zero_add, here]

/-- Pages give events: two proofs with their decoded messages, with no stop before the work of
the stage `k` and with the same byte strings of the stages `0 … k`, have alike states before this
work and the same events before its squeeze. -/
theorem stage_alike {k : ℕ} (inside : k < 473) {rest rest' : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (inputs : ∀ j ≤ k, inputs20 pub proof messages j = inputs20 pub proof' messages' j) :
    Alike (stateBefore pub proof messages pub.ops (stageIndex20 pub k))
        (stateBefore pub proof' messages' pub.ops (stageIndex20 pub k)) ∧
      eventsAt20 pub proof messages k = eventsAt20 pub proof' messages' k := by
  have upto := inputs_upto inside inputs
  rw [take_cut pub.ops.zipIdx.toList (Nat.le_succ (stageIndex20 pub k)), inputsFrom_append,
    inputsFrom_append] at upto
  obtain ⟨front, back⟩ := List.append_inj upto (by rw [length_inputsFrom, length_inputsFrom])
  rw [Nat.zero_add] at back
  have alike : Alike (stateBefore pub proof messages pub.ops (stageIndex20 pub k))
      (stateBefore pub proof' messages' pub.ops (stageIndex20 pub k)) :=
    run_alike_of_inputs pub proof proof' messages messages' _ _ _ (Alike.refl _) rfl ran ran'
      (Faithful.take _ 0 _ (decode_faithful decoded decoded')) front
  refine ⟨alike, ?_⟩
  unfold eventsAt20
  rw [back, alike.1]

/-- The events before the squeeze of a work give the events before the work and the absorbed
bytes of the work. -/
theorem eventsAt20_parts {k : ℕ}
    (events : eventsAt20 pub proof messages k = eventsAt20 pub proof' messages' k) :
    inputsFrom proof messages
        ((pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)).drop (stageIndex20 pub k))
        (readCount (pub.ops.zipIdx.toList.take (stageIndex20 pub k))) =
      inputsFrom proof' messages'
        ((pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)).drop (stageIndex20 pub k))
        (readCount (pub.ops.zipIdx.toList.take (stageIndex20 pub k))) ∧
      (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events =
        (stateBefore pub proof' messages' pub.ops (stageIndex20 pub k)).sponge.events := by
  unfold eventsAt20 at events
  obtain ⟨nonce, before⟩ := List.append_inj events (by
    rw [List.length_map, List.length_map, length_inputsFrom, length_inputsFrom])
  exact ⟨(List.map_injective_iff (f := Duplex.Event.absorb)).mpr
    (fun _ _ h => Duplex.Event.absorb.inj h) nonce, before⟩

/-- Events give pages: two proofs with their decoded messages, with no stop before the work of
the stage `k` and with the same events before the squeeze of this work, have the same events
before the squeeze of the work of each stage `j ≤ k` and the same byte strings of each stage
`j ≤ k`. -/
theorem stage_causal {k : ℕ} (inside : k < 473) {rest rest' : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (events : eventsAt20 pub proof messages k = eventsAt20 pub proof' messages' k) :
    ∀ j ≤ k, eventsAt20 pub proof messages j = eventsAt20 pub proof' messages' j ∧
      inputs20 pub proof messages j = inputs20 pub proof' messages' j := by
  obtain ⟨nonce, before⟩ := eventsAt20_parts events
  have faithful := decode_faithful decoded decoded'
  have front : inputsFrom proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k)) 0 =
      inputsFrom proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k)) 0 :=
    run_inputs pub proof proof' messages messages' _ _ _ (Alike.refl _) rfl ran ran'
      (Faithful.take _ 0 _ faithful) ⟨[], [], before⟩
  have upto : inputsFrom proof messages
        (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)) 0 =
      inputsFrom proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)) 0 := by
    rw [take_cut pub.ops.zipIdx.toList (Nat.le_succ (stageIndex20 pub k)), inputsFrom_append,
      inputsFrom_append, front, Nat.zero_add, nonce]
  intro j le
  have below := stageIndex20_mono pub le inside
  refine ⟨?_, inputsFrom_slice _
    (Nat.le_trans (stageStart20_le pub (Nat.lt_of_le_of_lt le inside)) (Nat.le_succ _))
    (Nat.succ_le_succ below) upto⟩
  obtain ⟨ok, -, cut⟩ := stateBefore_le pub proof messages pub.ops below ran
  obtain ⟨ok', -, cut'⟩ := stateBefore_le pub proof' messages' pub.ops below ran'
  obtain ⟨U, grown⟩ := runFrom_suffix pub proof messages
    ((pub.ops.zipIdx.toList.take (stageIndex20 pub k)).drop (stageIndex20 pub j))
    (stateBefore pub proof messages pub.ops (stageIndex20 pub j))
  obtain ⟨U', grown'⟩ := runFrom_suffix pub proof' messages'
    ((pub.ops.zipIdx.toList.take (stageIndex20 pub k)).drop (stageIndex20 pub j))
    (stateBefore pub proof' messages' pub.ops (stageIndex20 pub j))
  rw [← cut] at grown
  rw [← cut'] at grown'
  have later : U ++ (stateBefore pub proof messages pub.ops (stageIndex20 pub j)).sponge.events =
      U' ++ (stateBefore pub proof' messages' pub.ops (stageIndex20 pub j)).sponge.events := by
    rw [← grown, ← grown']
    exact before
  have alike : Alike (stateBefore pub proof messages pub.ops (stageIndex20 pub j))
      (stateBefore pub proof' messages' pub.ops (stageIndex20 pub j)) :=
    run_alike pub proof proof' messages messages' _ _ _ (Alike.refl _) rfl ok ok'
      (Faithful.take _ 0 _ faithful) ⟨U, U', later⟩
  unfold eventsAt20
  rw [alike.1, inputsFrom_slice _ (Nat.le_succ (stageIndex20 pub j)) (Nat.succ_le_succ below) upto]

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- The `Note` of the wire message `k < 473` of the model is a function of the public input, the
events of the sponge before the squeeze of the work of the stage `k` and the last blocks: two
proofs with their decoded messages, whose runs have no stop before this work and have the same
events before its squeeze, have the same `noteAt20 … k`. -/
theorem noteAt20_causal (tLast : ℕ → Rq Key20.terminal.innerDegree)
    (eLast : Rq Key20.terminal.innerDegree) {k : ℕ} (inside : k < 473) {rest rest' : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (events : eventsAt20 pub proof messages k = eventsAt20 pub proof' messages' k) :
    noteAt20 (E := E) pub proof messages tLast eLast k =
      noteAt20 pub proof' messages' tLast eLast k := by
  have core := core_causal pub proof proof' messages messages' _ decoded decoded' ran ran'
    (eventsAt20_parts events).2
  have strings := (stage_causal inside decoded decoded' ran ran' events k (Nat.le_refl k)).2
  unfold noteAt20 traceAt20
  rw [noteOf_core pub core tLast eLast, strings]

/-- The `Note` of the wire message `k ≥ 473` of the model is a function of the events of the
sponge before the terminal. -/
theorem noteAt20_causal_last (tLast : ℕ → Rq Key20.terminal.innerDegree)
    (eLast : Rq Key20.terminal.innerDegree) {k : ℕ} (outside : 473 ≤ k) {rest rest' : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (events : (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events =
      (stateBefore pub proof' messages' pub.ops (stageIndex20 pub k)).sponge.events) :
    noteAt20 (E := E) pub proof messages tLast eLast k =
      noteAt20 pub proof' messages' tLast eLast k := by
  have core := core_causal pub proof proof' messages messages' _ decoded decoded' ran ran' events
  unfold noteAt20
  rw [noteOf_core pub core tLast eLast, ite_eq_right (Nat.not_lt.mpr outside),
    ite_eq_right (Nat.not_lt.mpr outside)]

end Stages

namespace Replayer

section Ideal

/-- The two sponges have the same answers. -/
def SameIdeal (a b : Duplex) : Prop := a.ideal = b.ideal

theorem kept_path : Path SameIdeal := ⟨fun _ => rfl, fun first second => first.trans second⟩

theorem kept_absorb (bytes : ByteArray) : Moves SameIdeal (absorb bytes) := fun s =>
  (Duplex.absorb_ideal s.sponge bytes).symm

theorem kept_squeeze (count : ℕ) : Moves SameIdeal (squeeze count) := fun s =>
  (Duplex.squeeze_ideal s.sponge count).symm

macro "replay_kept_step" : tactic => `(tactic| repeat' (first
  | with_reducible exact moves_pure kept_path _ | with_reducible exact moves_throw kept_path _
  | with_reducible exact moves_get kept_path
  | with_reducible exact kept_squeeze _ | with_reducible exact kept_absorb _
  | with_reducible exact moves_message kept_path _ _
  | with_reducible exact moves_recorded kept_path _ | with_reducible exact moves_first kept_path _
  | with_reducible exact moves_pointBefore kept_path _
  | with_reducible exact moves_record kept_path _ _
  | with_reducible exact moves_value kept_path _ _ _
  | with_reducible exact moves_draw kept_path _ _ _ _ _
  | with_reducible exact moves_terminalWith kept_path _ _ _
  | with_reducible exact moves_checkWith kept_path _ _ _ _ _
  | with_reducible exact moves_mapM kept_path (fun _ => moves_recorded kept_path _) _
  | with_reducible exact moves_decodeExts kept_path _
  | (with_reducible refine moves_modify kept_path fun _ => ?_; rfl)
  | refine moves_bind kept_path ?_ fun _ => ?_
  | split))

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- Each step of the replay keeps the answers of the sponge. -/
theorem step_kept (i : ℕ) (op : Op) : Moves SameIdeal (step pub proof messages i op) := by
  cases op <;> simp only [step] <;> replay_kept_step

/-- A run of the replay keeps the answers of the sponge. -/
theorem runFrom_kept : ∀ (l : List (Op × ℕ)) (s : Replayer),
    (runFrom pub proof messages l s).2.sponge.ideal = s.sponge.ideal
  | [], _ => rfl
  | x :: l, s => by
      have stepped := step_kept pub proof messages x.2 x.1 s
      rw [runFrom_cons]
      rcases found : ((step pub proof messages x.2 x.1).run).run s with ⟨_ | _, s₁⟩
      · rw [found] at stepped
        exact stepped.symm
      · rw [found] at stepped
        exact (runFrom_kept l s₁).trans stepped.symm

/-- The sponge of each state of the replay has the answers of the sponge of the public input. -/
theorem stateBefore_ideal (ops : Array Op) (c : ℕ) :
    (stateBefore pub proof messages ops c).sponge.ideal = pub.sponge.ideal := by
  unfold stateBefore
  rw [runFrom_kept]
  simp only [channelStart, channel, Duplex.absorb_ideal]

/-- The count of the bytes that a draw squeezes. -/
def _root_.NightstreamFPrime.Spec.Compression.Akita.Replay.Op.drawBytes : Op → ℕ
  | .challenge _ count => poolBytes (2 * count)
  | .squeeze _ _ G => G.messageBytes
  | _ => 0

theorem sponge_bind_squeeze {β : Type} (count : ℕ) (f : ByteArray → ReplayM β)
    (still : ∀ bytes, Moves (fun a b : Duplex => a = b) (f bytes)) (s : Replayer) :
    (((squeeze count >>= f).run).run s).2.sponge = (s.sponge.squeeze count).2 := by
  rw [run_bind, run_squeeze]
  exact (still _ _).symm

/-- The step of a draw squeezes `Op.drawBytes` bytes and does not change the sponge after
that. -/
theorem step_draw (i : ℕ) (op : Op) (draws : isDraw op = true) (s : Replayer) :
    (((step pub proof messages i op).run).run s).2.sponge = (s.sponge.squeeze op.drawBytes).2 := by
  cases op with
  | challenge key count =>
      simp only [step]
      exact sponge_bind_squeeze _ _ (fun _ => by replay_moves eq_path) s
  | squeeze level group G =>
      simp only [step]
      exact sponge_bind_squeeze _ _ (fun bytes => moves_draw eq_path i level group G bytes) s
  | _ => cases draws

/-- A run without a stop of draws adds one squeeze for each draw to the events of the sponge. -/
theorem runFrom_draws : ∀ (l : List (Op × ℕ)) (s : Replayer), (∀ x ∈ l, isDraw x.1 = true) →
    (runFrom pub proof messages l s).1 = .ok () →
    (runFrom pub proof messages l s).2.sponge.events =
      (l.reverse.map fun x => Duplex.Event.squeeze x.1.drawBytes) ++ s.sponge.events
  | [], _, _, _ => rfl
  | x :: l, s, draws, ran => by
      obtain ⟨stepped, tail⟩ := (runFrom_cons_fst pub proof messages _ _ _).mp ran
      rw [runFrom_cons_snd pub proof messages _ _ _ stepped,
        runFrom_draws l _ (fun y inside => draws y (List.mem_cons_of_mem _ inside)) tail,
        step_draw pub proof messages x.2 x.1 (draws x List.mem_cons_self), Duplex.squeeze_events,
        List.reverse_cons, List.map_append, List.append_assoc]
      rfl

/-- A step without a stop of a work with `bits > 0`: the 8 bytes that it squeezes after the
absorb of its nonce have the work. -/
theorem step_work_passed (i : ℕ) (site : Site) (level bits : ℕ) (s : Replayer)
    (positive : bits ≠ 0)
    (ran : (((step pub proof messages i (.work site level bits)).run).run s).1 = .ok ()) :
    ∃ m, messages[s.next]? = some m ∧
      worked ((s.sponge.absorb ((Op.work site level bits).input proof m)).squeeze 8).1 bits =
        true := by
  cases found : messages[s.next]? with
  | none =>
      simp only [step, positive, ↓reduceIte, run_bind, run_message, found] at ran
      cases ran
  | some m =>
      refine ⟨m, rfl, ?_⟩
      cases data : m.data with
      | nonce nonce canonical =>
          have input : (Op.work site level bits).input proof m = le64 nonce := by
            simp only [Op.input, data]
          rw [input]
          simp only [step, positive, ↓reduceIte, run_bind, run_message, found, data] at ran
          split at ran
          · simp only [run_bind, run_absorb, run_squeeze, run_modify] at ran
            split at ran
            · rename_i passed
              exact passed
            · cases ran
          · cases ran
      | _ =>
          simp only [step, positive, ↓reduceIte, run_bind, run_message, found, data] at ran
          cases ran

end Ideal

end Replayer

section Squeezes

variable {pub : Public} {proof : ByteArray} {messages : Array Message}

/-- On a script of `layout20`, the operation at `stageIndex20 pub k` is a work, for `k < 473`. -/
theorem work20_of (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {k : ℕ}
    (inside : k < 473) :
    ∃ site level bits, pub.ops[stageIndex20 pub k]? = some (.work site level bits) := by
  have member : workIndices20.getD k 0 ∈ workIndices20 := by
    have below : k < workIndices20.length := by
      rw [workIndices20_length]
      exact inside
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem below]
    exact List.getElem_mem below
  obtain ⟨⟨op', i⟩, kept, same⟩ := List.mem_map.mp member
  obtain ⟨listed, work⟩ := List.mem_filter.mp kept
  have index : i = workIndices20.getD k 0 := same
  have at_ : (skeleton (layout20 0 0))[i]? = some op' := List.mem_zipIdx_iff_getElem?.mp listed
  rw [← erased, List.getElem?_map] at at_
  obtain ⟨op, found, erasedOp⟩ := Option.map_eq_some_iff.mp at_
  subst erasedOp
  rw [stageIndex20_lt pub inside, ← index, ← Array.getElem?_toList, found]
  cases op <;> first | exact ⟨_, _, _, rfl⟩ | cases work

/-- The events before the squeeze of a work that reads the message `m`: the absorb of the bytes
of `m`, then the events before the work. -/
theorem eventsAt20_read {k : ℕ} {op : Op} (at_ : pub.ops[stageIndex20 pub k]? = some op)
    (reads : op.reads = true)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    {m : Message}
    (found : messages[(stateBefore pub proof messages pub.ops (stageIndex20 pub k)).next]? =
      some m) :
    eventsAt20 pub proof messages k = .absorb (op.input proof m) ::
      (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events := by
  have next : (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).next =
      readCount (pub.ops.zipIdx.toList.take (stageIndex20 pub k)) :=
    (runFrom_next pub proof messages _ _ ran).trans (Nat.zero_add _)
  rw [next] at found
  have short : (pub.ops.zipIdx.toList.take (stageIndex20 pub k)).length = stageIndex20 pub k := by
    rw [List.length_take, Array.length_toList, Array.size_zipIdx]
    exact Nat.min_eq_left (Nat.le_of_lt (Array.getElem?_eq_some_iff.mp at_).1)
  unfold eventsAt20
  rw [take_succ_zipIdx pub.ops at_, List.drop_left' short, inputsFrom, ite_eq_left reads]
  simp only [inputAt, found]
  rfl

/-- The squeeze of the work of the stage `k`, for a work with bits, a run with no stop up to and
with this work and an ideal sponge: the work reads a message `m`, the events before its squeeze
are `eventsAt20 … k`, its 8 bytes are the answer for these events, and the events after the work
are this squeeze and `eventsAt20 … k`. -/
theorem work_squeeze20 {k : ℕ} {site : Site} {level bits : ℕ}
    (at_ : pub.ops[stageIndex20 pub k]? = some (.work site level bits)) (positive : bits ≠ 0)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1))
      (channelStart pub)).1 = .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer) :
    ∃ m, messages[(stateBefore pub proof messages pub.ops (stageIndex20 pub k)).next]? = some m ∧
      eventsAt20 pub proof messages k = .absorb ((Op.work site level bits).input proof m) ::
        (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events ∧
      (((stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.absorb
        ((Op.work site level bits).input proof m)).squeeze 8).1 =
          answer (eventsAt20 pub proof messages k) 8 ∧
      (stateBefore pub proof messages pub.ops (stageIndex20 pub k + 1)).sponge.events =
        .squeeze 8 :: eventsAt20 pub proof messages k := by
  obtain ⟨stepped, after⟩ := stateBefore_succ pub proof messages pub.ops at_ ran
  obtain ⟨m, found, sponge⟩ := step_work pub proof messages _ site level bits _ positive stepped
  have before := runFrom_take_ok pub proof messages _ (stageIndex20 pub k) _ ran
  rw [List.take_take, Nat.min_eq_left (Nat.le_succ _)] at before
  have reads : (Op.work site level bits).reads = true := by simpa [Op.reads] using positive
  have events := eventsAt20_read at_ reads before found
  have live : ((stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.absorb
      ((Op.work site level bits).input proof m)).ideal = some answer := by
    rw [Duplex.absorb_ideal, stateBefore_ideal]
    exact ideal
  refine ⟨m, found, events, ?_, ?_⟩
  · rw [Duplex.squeeze_of_ideal _ (by decide) live, Duplex.absorb_events, events]
  · rw [after, sponge, Duplex.squeeze_events, Duplex.absorb_events, events]

/-- The work of the stage `k`, for a work with bits, a run with no stop up to and with this work
and an ideal sponge: the answer for `eventsAt20 … k` has the work. -/
theorem work_passed20 {k : ℕ} {site : Site} {level bits : ℕ}
    (at_ : pub.ops[stageIndex20 pub k]? = some (.work site level bits)) (positive : bits ≠ 0)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1))
      (channelStart pub)).1 = .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer) :
    worked (answer (eventsAt20 pub proof messages k) 8) bits = true := by
  obtain ⟨stepped, -⟩ := stateBefore_succ pub proof messages pub.ops at_ ran
  obtain ⟨m, found, passed⟩ :=
    step_work_passed pub proof messages _ site level bits _ positive stepped
  obtain ⟨m', found', -, squeezed, -⟩ := work_squeeze20 at_ positive ran ideal
  rw [found] at found'
  cases found'
  rw [← squeezed]
  exact passed

/-- The squeeze of an operation after draws that follow the work of the stage `k`, for a work
with bits, a run with no stop before the operation `c` and an ideal sponge: the events before `c`
are one squeeze for each draw, the squeeze of the work and `eventsAt20 … k`, and a squeeze at `c`
is the answer for these events. -/
theorem draw_squeeze20 {k c : ℕ} {site : Site} {level bits : ℕ}
    (at_ : pub.ops[stageIndex20 pub k]? = some (.work site level bits)) (positive : bits ≠ 0)
    (after : stageIndex20 pub k + 1 ≤ c)
    (draws : ∀ x ∈ (pub.ops.zipIdx.toList.take c).drop (stageIndex20 pub k + 1),
      isDraw x.1 = true)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer) :
    (stateBefore pub proof messages pub.ops c).sponge.events =
        (((pub.ops.zipIdx.toList.take c).drop (stageIndex20 pub k + 1)).reverse.map fun x =>
          Duplex.Event.squeeze x.1.drawBytes) ++ .squeeze 8 :: eventsAt20 pub proof messages k ∧
      ∀ count, count ≠ 0 →
        ((stateBefore pub proof messages pub.ops c).sponge.squeeze count).1 =
          answer ((((pub.ops.zipIdx.toList.take c).drop (stageIndex20 pub k + 1)).reverse.map
            fun x => Duplex.Event.squeeze x.1.drawBytes) ++
              .squeeze 8 :: eventsAt20 pub proof messages k) count := by
  obtain ⟨first, rest, cut⟩ := stateBefore_le pub proof messages pub.ops after ran
  obtain ⟨-, -, -, -, base⟩ := work_squeeze20 at_ positive first ideal
  have events : (stateBefore pub proof messages pub.ops c).sponge.events =
      (((pub.ops.zipIdx.toList.take c).drop (stageIndex20 pub k + 1)).reverse.map fun x =>
        Duplex.Event.squeeze x.1.drawBytes) ++ .squeeze 8 :: eventsAt20 pub proof messages k := by
    rw [cut, runFrom_draws pub proof messages _ _ draws rest, base]
  refine ⟨events, fun count positive' => ?_⟩
  rw [Duplex.squeeze_of_ideal _ positive'
    ((stateBefore_ideal pub proof messages pub.ops c).trans ideal), events]

/-- On a script of `layout20`, the `drawCount20 k` operations directly after the work of the
stage `k` are draws. -/
theorem draws20_of (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {k : ℕ}
    (inside : k < 473) {c : ℕ} (le : c ≤ stageIndex20 pub k + 1 + drawCount20 k) :
    ∀ x ∈ (pub.ops.zipIdx.toList.take c).drop (stageIndex20 pub k + 1), isDraw x.1 = true := by
  intro x member
  have fit := List.all_eq_true.mp skeleton20_draws k (List.mem_range.mpr inside)
  unfold drawsFit20 at fit
  simp only [Bool.and_eq_true, List.all_eq_true] at fit
  have wider := mem_slice _ le member
  have erasedMember : eraseAt x ∈ stageDraws20 k := by
    unfold stageDraws20
    rw [← stageIndex20_lt pub inside, ← take_erase erased, ← List.map_drop]
    exact List.mem_map_of_mem wider
  rw [← isDraw_erase]
  exact fit.1.2 _ erasedMember

end Squeezes

end NightstreamFPrime.Spec.Compression.Akita.Replay
