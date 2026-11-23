import NightstreamFPrime.Spec.Checkpoint.Protocol
import Mathlib.Data.Set.Finite.List

/-! A finite model of the Fiat–Shamir protocol shape of Akita Section 10.3 (ePrint 2026/1983,
revision of 2026-11-23, "Extraction with checkpoints", p. 76).

- `Prefix` is a stage input `ξ_r`: the first message of the proof, and for each earlier stage the
  answer at this stage and the next message of the proof. A message is a value of a finite type
  `Wire`.
- `Address` is the key `⟨ξ_r⟩` of a stage input: its messages without the answers.
- `transcript` is the `Checkpoint.Protocol` of these stage inputs and keys. A proof is the sequence
  of its messages. The stage input of a stage has the messages and the answers before the answer
  of this stage, and no later message. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Checkpoint

/-- A stage input of a protocol with `stages` stages: the message `first`, and the pairs of an
answer and the next message. -/
@[ext] structure Prefix (Wire Reply : Type) (stages : ℕ) where
  first : Wire
  rest : List (Reply × Wire)
  short : rest.length ≤ stages

variable {Wire Reply : Type} {stages : ℕ}

namespace Prefix

instance [Finite Wire] [Finite Reply] : Finite (Prefix Wire Reply stages) :=
  have : Finite {rest : List (Reply × Wire) // rest.length ≤ stages} :=
    (List.finite_length_le (Reply × Wire) stages).to_subtype
  Finite.of_injective
    (fun point : Prefix Wire Reply stages =>
      (point.first, (⟨point.rest, point.short⟩ : {rest : List (Reply × Wire) //
        rest.length ≤ stages})))
    fun point other same => by
      simp only [Prod.mk.injEq, Subtype.mk.injEq] at same
      exact Prefix.ext same.1 same.2

noncomputable instance [Finite Wire] [Finite Reply] : Fintype (Prefix Wire Reply stages) :=
  Fintype.ofFinite _

/-- The answer of the stage `index`, when the stage input has it. -/
def reply (point : Prefix Wire Reply stages) (index : ℕ) : Option Reply :=
  point.rest[index]?.map Prod.fst

/-- The message `index` of the proof. It is the first message when the stage input does not have
the message `index`. -/
def wire (point : Prefix Wire Reply stages) : ℕ → Wire
  | 0 => point.first
  | index + 1 => (point.rest[index]?.map Prod.snd).getD point.first

/-- The stage input of the stage `level` that a stage input extends. -/
def take (level : ℕ) (point : Prefix Wire Reply stages) : Prefix Wire Reply stages :=
  ⟨point.first, point.rest.take level, by
    rw [List.length_take]
    exact (Nat.min_le_right _ _).trans point.short⟩

theorem reply_take {level index : ℕ} (below : index < level) (point : Prefix Wire Reply stages) :
    (point.take level).reply index = point.reply index := by
  simp [take, reply, below]

theorem wire_take {level index : ℕ} (below : index ≤ level) (point : Prefix Wire Reply stages) :
    (point.take level).wire index = point.wire index := by
  rcases index with _ | index
  · rfl
  · simp [take, wire, Nat.lt_of_succ_le below]

end Prefix

/-- The stage input with one more answer and one more message. -/
def Prefix.snoc (point : Prefix Wire Reply stages) (move : Reply × Wire)
    (room : point.rest.length < stages) : Prefix Wire Reply stages :=
  ⟨point.first, point.rest ++ [move], by
    rw [List.length_append, List.length_singleton]
    exact room⟩

/-- A key of a protocol with `stages` stages: the message `first` and the later messages. -/
@[ext] structure Address (Wire : Type) (stages : ℕ) where
  first : Wire
  rest : List Wire
  short : rest.length ≤ stages

namespace Address

instance [Finite Wire] : Finite (Address Wire stages) :=
  have : Finite {rest : List Wire // rest.length ≤ stages} :=
    (List.finite_length_le Wire stages).to_subtype
  Finite.of_injective
    (fun key : Address Wire stages =>
      (key.first, (⟨key.rest, key.short⟩ : {rest : List Wire // rest.length ≤ stages})))
    fun key other same => by
      simp only [Prod.mk.injEq, Subtype.mk.injEq] at same
      exact Address.ext same.1 same.2

noncomputable instance [Finite Wire] : Fintype (Address Wire stages) :=
  Fintype.ofFinite _

/-- The key of the stage `level` that a key extends. -/
def take (level : ℕ) (key : Address Wire stages) : Address Wire stages :=
  ⟨key.first, key.rest.take level, by
    rw [List.length_take]
    exact (Nat.min_le_right _ _).trans key.short⟩

/-- `k_{≤ level}`: the key of the stage `level` that a key extends, when the key has that stage. -/
def trim (level : ℕ) (key : Address Wire stages) : Option (Address Wire stages) :=
  if level ≤ key.rest.length then some (key.take level) else none

/-- `T_J(k)`: the messages of `key`, each after the answer of `table` at the key of its stage. -/
def rebuild (table : Address Wire stages → Reply) (key : Address Wire stages) :
    Prefix Wire Reply stages :=
  ⟨key.first, key.rest.mapIdx fun index wire => (table (key.take index), wire), by
    rw [List.length_mapIdx]
    exact key.short⟩

theorem length_take {level : ℕ} {key : Address Wire stages} (below : level ≤ key.rest.length) :
    (key.take level).rest.length = level := by
  simp [take, below]

theorem trim_rank {level : ℕ} {key anchor : Address Wire stages}
    (found : key.trim level = some anchor) :
    anchor.rest.length = level ∧ level ≤ key.rest.length := by
  unfold trim at found
  split at found
  · rename_i below
    obtain rfl := Option.some.inj found
    exact ⟨length_take below, below⟩
  · exact absurd found (by simp)

theorem rebuild_congr (table other : Address Wire stages → Reply) (key : Address Wire stages)
    (agree : ∀ earlier : Address Wire stages, earlier.rest.length < key.rest.length →
      table earlier = other earlier) :
    key.rebuild table = key.rebuild other :=
  Prefix.ext rfl (List.mapIdx_eq_mapIdx_iff.mpr fun index inside =>
    congrArg (·, key.rest[index]) (agree _ (by simp only [take, List.length_take]; omega)))

end Address

/-- `⟨ξ⟩`: the messages of a stage input. -/
def Prefix.address (point : Prefix Wire Reply stages) : Address Wire stages :=
  ⟨point.first, point.rest.map Prod.snd, by
    rw [List.length_map]
    exact point.short⟩

theorem Prefix.length_address (point : Prefix Wire Reply stages) :
    point.address.rest.length = point.rest.length :=
  List.length_map _

theorem Prefix.address_take (level : ℕ) (point : Prefix Wire Reply stages) :
    (point.take level).address = point.address.take level :=
  Address.ext rfl List.map_take

theorem Prefix.trim_address (level : ℕ) (point : Prefix Wire Reply stages) :
    point.address.trim level =
      (if level ≤ point.rest.length then some (point.take level) else none).map Prefix.address := by
  unfold Address.trim
  rw [Prefix.length_address]
  split
  · rw [Option.map_some, Prefix.address_take]
  · rfl

theorem Address.address_rebuild (table : Address Wire stages → Reply) (key : Address Wire stages) :
    (key.rebuild table).address = key :=
  Address.ext rfl (List.ext_getElem (by simp [Address.rebuild, Prefix.address])
    fun index _ _ => by simp [Address.rebuild, Prefix.address])

theorem Prefix.address_snoc (point : Prefix Wire Reply stages) (move : Reply × Wire)
    (room : point.rest.length < stages) :
    (point.snoc move room).address.rest = point.address.rest ++ [move.2] := by
  simp [Prefix.snoc, Prefix.address]

theorem Address.rebuild_snoc (table : Address Wire stages → Reply)
    (point : Prefix Wire Reply stages) (wire : Wire) (room : point.rest.length < stages)
    (rebuilt : point.address.rebuild table = point) :
    (point.snoc (table point.address, wire) room).address.rebuild table =
      point.snoc (table point.address, wire) room := by
  refine Prefix.ext rfl ?_
  have before : (point.address.rest.mapIdx fun index message =>
      (table (point.address.take index), message)) = point.rest := congrArg Prefix.rest rebuilt
  have taken (index : ℕ) (below : index ≤ point.address.rest.length) :
      (point.snoc (table point.address, wire) room).address.take index =
        point.address.take index :=
    Address.ext rfl (by
      show (point.snoc (table point.address, wire) room).address.rest.take index =
        point.address.rest.take index
      rw [Prefix.address_snoc, List.take_append_of_le_length below])
  show (point.snoc (table point.address, wire) room).address.rest.mapIdx
      (fun index message =>
        (table ((point.snoc (table point.address, wire) room).address.take index), message)) =
      point.rest ++ [(table point.address, wire)]
  rw [Prefix.address_snoc, List.mapIdx_concat]
  congr 1
  · rw [← before]
    exact List.mapIdx_eq_mapIdx_iff.mpr fun index inside =>
      congrArg (·, point.address.rest[index]) (congrArg table (taken index (Nat.le_of_lt inside)))
  · have whole : point.address.take point.address.rest.length = point.address :=
      Address.ext rfl List.take_length
    rw [taken _ le_rfl, whole]

variable [Nonempty Reply]

/-- The protocol of the stage inputs `Prefix`, with the acceptance predicate `accept` of the
verifier. `step` adds the answer and the message of the proof after it. At a complete transcript
`step` has no effect. -/
noncomputable def transcript (stages : ℕ) (accept : Prefix Wire Reply stages → Prop) :
    Protocol (Address Wire stages) (Prefix Wire Reply stages) Reply (ℕ → Wire) where
  stages := stages
  root proof := ⟨proof 0, [], Nat.zero_le _⟩
  step proof point reply :=
    if room : point.rest.length < stages then
      ⟨point.first, point.rest ++ [(reply, proof (point.rest.length + 1))], by
        rw [List.length_append, List.length_singleton]
        exact room⟩
    else point
  stage point := point.rest.length
  cut level point := if level ≤ point.rest.length then some (point.take level) else none
  last point := (point.rest.getLast?.map Prod.fst).getD (Classical.arbitrary Reply)
  accept := accept
  address := Prefix.address
  rank key := key.rest.length
  trim := Address.trim
  rebuild := Address.rebuild
  stage_root _ := rfl
  stage_step proof point reply room := by
    simp only [dite_eq_left room, List.length_append, List.length_singleton]
  cut_self point := by
    simp only [le_refl, ite_true, Prefix.take, List.take_length]
  cut_step proof point reply level below := by
    by_cases room : point.rest.length < stages
    · simp only [dite_eq_left room, List.length_append, List.length_singleton, Prefix.take,
        ite_eq_left below, ite_eq_left (Nat.le_succ_of_le below), List.take_append_of_le_length below]
    · simp only [dite_eq_right room]
  cut_stage level point anchor found := by
    by_contra above
    simp only [ite_eq_right above] at found
    exact absurd found (by simp)
  last_step proof point reply room := by
    simp only [dite_eq_left room, List.getLast?_append, List.getLast?_singleton, Option.some_or,
      Option.map_some, Option.getD_some]
  rank_address := Prefix.length_address
  trim_address := Prefix.trim_address
  trim_rank _ _ _ found := Address.trim_rank found
  address_rebuild := Address.address_rebuild
  address_step proof point other reply answer same := by
    have length : point.rest.length = other.rest.length := by
      rw [← Prefix.length_address, same, Prefix.length_address]
    have first : point.first = other.first := congrArg Address.first same
    have rest : point.address.rest = other.address.rest := congrArg Address.rest same
    by_cases room : point.rest.length < stages
    · have room' : other.rest.length < stages := length ▸ room
      simp only [dite_eq_left room, dite_eq_left room']
      change (point.snoc _ room).address = (other.snoc _ room').address
      exact Address.ext first (by rw [Prefix.address_snoc, Prefix.address_snoc, rest, length])
    · have room' : ¬ other.rest.length < stages := length ▸ room
      simp only [dite_eq_right room, dite_eq_right room']
      exact same
  rebuild_root table proof := Prefix.ext rfl (by simp [Address.rebuild, Prefix.address])
  rebuild_step table proof point room rebuilt := by
    simp only [dite_eq_left room]
    exact Address.rebuild_snoc table point _ room rebuilt
  rebuild_congr := Address.rebuild_congr

omit [Nonempty Reply] in
theorem Prefix.take_length (point : Prefix Wire Reply stages) :
    point.take point.rest.length = point :=
  Prefix.ext rfl (List.take_length)

variable {accept : Prefix Wire Reply stages → Prop}

theorem transcript_cut {level : ℕ} {point anchor : Prefix Wire Reply stages}
    (found : (transcript stages accept).cut level point = some anchor) :
    level ≤ point.rest.length ∧ anchor = point.take level := by
  have below := (transcript stages accept).cut_stage level point anchor found
  refine ⟨below, ?_⟩
  have value : (transcript stages accept).cut level point = some (point.take level) :=
    ite_eq_left below
  exact (Option.some.inj (value.symm.trans found)).symm

omit [Nonempty Reply] in
theorem Prefix.take_take {level deeper : ℕ} (below : level ≤ deeper)
    (point : Prefix Wire Reply stages) : (point.take deeper).take level = point.take level :=
  Prefix.ext rfl (by
    show (point.rest.take deeper).take level = point.rest.take level
    rw [List.take_take, Nat.min_eq_left below])

omit [Nonempty Reply] in
theorem Prefix.reply_snoc (point : Prefix Wire Reply stages) (move : Reply × Wire)
    (room : point.rest.length < stages) {index : ℕ} (below : index < point.rest.length) :
    (point.snoc move room).reply index = point.reply index := by
  simp only [Prefix.snoc, Prefix.reply, List.getElem?_append_left below]

omit [Nonempty Reply] in
theorem Prefix.wire_snoc (point : Prefix Wire Reply stages) (move : Reply × Wire)
    (room : point.rest.length < stages) {index : ℕ} (below : index ≤ point.rest.length) :
    (point.snoc move room).wire index = point.wire index := by
  rcases index with _ | index
  · rfl
  · simp only [Prefix.snoc, Prefix.wire, List.getElem?_append_left (Nat.lt_of_succ_le below)]

omit [Nonempty Reply] in
theorem Prefix.take_snoc (point : Prefix Wire Reply stages) (move : Reply × Wire)
    (room : point.rest.length < stages) {level : ℕ} (below : level ≤ point.rest.length) :
    (point.snoc move room).take level = point.take level :=
  Prefix.ext rfl (List.take_append_of_le_length below)

theorem transcript_step (proof : ℕ → Wire) (point : Prefix Wire Reply stages) (reply : Reply)
    (room : point.rest.length < stages) :
    (transcript stages accept).step proof point reply =
      point.snoc (reply, proof (point.rest.length + 1)) room :=
  dite_eq_left room

/-- The last answer of the stage input after the stage `index`. -/
theorem last_take_succ {point : Prefix Wire Reply stages} {index : ℕ} {reply : Reply}
    (answered : point.reply index = some reply) :
    (transcript stages accept).last (point.take (index + 1)) = reply := by
  obtain ⟨move, found, rfl⟩ : ∃ move, point.rest[index]? = some move ∧ move.1 = reply :=
    Option.map_eq_some_iff.mp answered
  show (((point.rest.take (index + 1)).getLast?.map Prod.fst).getD _) = move.1
  rw [List.take_add_one, found]
  simp

/-- The stage inputs that a proof and a table on stage inputs give. -/
noncomputable def pointPath (stages : ℕ) (accept : Prefix Wire Reply stages → Prop)
    (proof : ℕ → Wire) (table : Prefix Wire Reply stages → Reply) : ℕ → Prefix Wire Reply stages
  | 0 => (transcript stages accept).root proof
  | level + 1 => (transcript stages accept).step proof (pointPath stages accept proof table level)
      (table (pointPath stages accept proof table level))

/-- The transcript of a key table `J` is the transcript of the table `ξ ↦ J(⟨ξ⟩)`. -/
theorem path_eq_pointPath (proof : ℕ → Wire) (table : Address Wire stages → Reply) (level : ℕ) :
    (transcript stages accept).path proof table level =
      pointPath stages accept proof (fun point => table point.address) level := by
  induction level with
  | zero => rfl
  | succ level induction =>
      simp only [Protocol.path, pointPath, induction]
      rfl

/-- The complete transcript of a proof for a key table `J` is the last stage input of the path of
the proof for the table `ξ ↦ J(⟨ξ⟩)`. -/
theorem final_done (proof : ℕ → Wire) (table : Address Wire stages → Reply) :
    (transcript stages accept).final (.done proof) table =
      pointPath stages accept proof (fun point => table point.address) stages :=
  path_eq_pointPath proof table stages

end NightstreamFPrime.Spec.Compression.Akita.Extract
