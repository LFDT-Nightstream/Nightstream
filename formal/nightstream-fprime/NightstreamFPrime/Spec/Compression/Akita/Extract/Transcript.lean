import NightstreamFPrime.Spec.Checkpoint.Protocol
import Mathlib.Data.Set.Finite.List

/-! A finite model of the Fiat–Shamir protocol shape of Akita Section 10.3 (ePrint 2026/1983,
revision of 2026-10-15, "Extraction with checkpoints: conventions", pp. 75–76).

- `Prefix` is a stage input `ξ_r`: the first message of the proof, and for each earlier stage the
  answer at this stage and the next message of the proof. A message is a value of a finite type
  `Wire`.
- `transcript` is the `Checkpoint.Protocol` of these stage inputs. A proof is the sequence of its
  messages. The stage input of a stage has the messages and the answers before the answer of this
  stage, and no later message. -/

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

variable [Nonempty Reply]

/-- The protocol of the stage inputs `Prefix`, with the acceptance predicate `accept` of the
verifier. `step` adds the answer and the message of the proof after it. At a complete transcript
`step` has no effect. -/
noncomputable def transcript (stages : ℕ) (accept : Prefix Wire Reply stages → Prop) :
    Protocol (Prefix Wire Reply stages) Reply (ℕ → Wire) where
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

/-- The stage input with one more answer and one more message. -/
def Prefix.snoc (point : Prefix Wire Reply stages) (move : Reply × Wire)
    (room : point.rest.length < stages) : Prefix Wire Reply stages :=
  ⟨point.first, point.rest ++ [move], by
    rw [List.length_append, List.length_singleton]
    exact room⟩

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

end NightstreamFPrime.Spec.Compression.Akita.Extract
