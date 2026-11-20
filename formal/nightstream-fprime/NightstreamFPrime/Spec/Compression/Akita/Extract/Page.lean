import NightstreamFPrime.Spec.Compression.Akita.Extract.CheckComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.Wire

/-! A type of wire messages for the honest prover of an opening of Akita (ePrint 2026/1983,
revision of 2026-10-15, Figure 4: the messages of the prover), with the conditions `Sends` and
`Carries` of `Extract/Play.lean` and `Extract/CheckComplete.lean`.

- `Cell` has the data of one message of the checks: `width` elements of `E` and a natural number
  up to `cap`. `Cell.message` reads a message of a type from a cell: the number for the type `ℕ`,
  and `unpack` of the elements for each other type.
- `Page` is a wire message: a value of the other data of the proof, and `slots` cells.
  `Page.read` reads message `index` after a draw from cell `index`.
- `Cell.Sendable` states that a message is the message of a cell. `Page.sends` is the condition
  `Sends` for it, and `Page.carries` is the condition `Carries` for each part of the data. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

variable {Data E : Type} {slots width cap : ℕ}

/-- The data of one message of the checks: `width` elements of `E` for an element, a pair or a
round polynomial, and a natural number up to `cap` for the energy of a direct norm check. -/
abbrev Cell (E : Type) (width cap : ℕ) : Type := (Fin width → E) × Fin (cap + 1)

open Classical in
/-- The message of the type `M` of a cell: the number of the cell for the natural numbers, and
`unpack` of its elements for each other type. -/
noncomputable def Cell.message (M : Type) [Nonempty M] (cell : Cell E width cap) : M :=
  if natural : M = ℕ then cast natural.symm (cell.2 : ℕ) else unpack width E M cell.1

variable (E width cap) in
/-- A message of the checks is the message of a cell. -/
def Cell.Sendable (M : Type) (message : M) : Prop :=
  ∀ inhabited : Nonempty M, ∃ cell : Cell E width cap,
    @Cell.message E width cap M inhabited cell = message

/-- A wire message: a value of the other data of the proof, and one cell for each of `slots`
messages of the checks after one draw. -/
structure Page (Data E : Type) (slots width cap : ℕ) where
  data : Data
  cells : Fin slots → Cell E width cap

/-- The reader of the messages of the checks: message `index` after a draw is the message of the
cell `index`. -/
noncomputable def Page.read : Reader (Page Data E slots width cap) :=
  fun M _ wire index =>
    if inside : index < slots then (wire.cells ⟨index, inside⟩).message M
    else Classical.arbitrary M

theorem ne_nat (M : Type) [Finite M] : M ≠ ℕ := by
  rintro rfl
  exact not_finite ℕ

theorem Cell.sendable_of_onto [Zero E] {M : Type} [Finite M]
    (onto : ∃ map : (Fin width → E) → M, Function.Surjective map) (message : M) :
    Cell.Sendable E width cap M message := fun _ => by
  obtain ⟨vector, sent⟩ := unpack_surjective onto message
  exact ⟨(vector, 0), (dite_eq_right (ne_nat M)).trans sent⟩

/-- The messages of the cells satisfy `Sends`: a cell has each element of `E`, each pair, each
vector of at most `width` elements, and each natural number up to `cap`. -/
theorem Page.sends [Finite E] [Zero E] (two : 2 ≤ width) :
    Sends (Cell.Sendable E width cap) E width cap where
  scalar := Cell.sendable_of_onto ⟨fun vector => vector ⟨0, by omega⟩,
    fun value => ⟨fun _ => value, rfl⟩⟩
  pair := Cell.sendable_of_onto ⟨fun vector => (vector ⟨0, by omega⟩, vector ⟨1, by omega⟩),
    fun value => ⟨fun index => if (index : ℕ) = 0 then value.1 else value.2, rfl⟩⟩
  vector n := fun message short => Cell.sendable_of_onto
    ⟨fun vector index => vector (Fin.castLE short index), fun value =>
      ⟨fun index => if inside : (index : ℕ) < n then value ⟨index, inside⟩ else 0,
        funext fun index => dite_eq_left index.isLt⟩⟩ message
  count message small := fun _ =>
    ⟨(fun _ => 0, ⟨message, Nat.lt_succ_of_le small⟩), dite_eq_left rfl⟩

/-- A wire message can have each value of a part `view` of the data with each list of at most
`slots` messages of the cells. -/
theorem Page.carries [Zero E] {Part : Type} (view : Data → Part)
    (onto : Function.Surjective view) :
    Carries (Cell.Sendable E width cap) slots Page.read
      fun wire : Page Data E slots width cap => view wire.data := by
  intro part sent short each
  obtain ⟨data, rfl⟩ := onto part
  refine ⟨⟨data, fun index => if inside : (index : ℕ) < sent.length then
    (each sent[index] (List.getElem_mem inside) sent[index].inhabited).choose
    else (fun _ => 0, 0)⟩, rfl, fun index inside => ?_⟩
  have chosen := (each sent[index] (List.getElem_mem inside) sent[index].inhabited).choose_spec
  refine (dite_eq_left (inside.trans_le short)).trans (Eq.trans ?_ chosen)
  exact congrArg (@Cell.message E width cap sent[index].M sent[index].inhabited)
    (dite_eq_left inside)

end NightstreamFPrime.Spec.Compression.Akita.Extract
