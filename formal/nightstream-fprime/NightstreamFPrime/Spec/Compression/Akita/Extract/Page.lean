import NightstreamFPrime.Spec.Compression.Akita.Extract.CheckComplete
import NightstreamFPrime.Spec.Compression.Akita.Extract.Wire

/-! A type of wire messages for the honest prover of an opening of Akita (ePrint 2026/1983,
revision of 2026-11-23, Figure 4: the messages of the prover), with the conditions `Sends` and
`Carries` of `Extract/Play.lean` and `Extract/CheckComplete.lean`.

- `Cell` has the data of one message of the checks: `width` elements of `E` and a natural number
  up to `cap`. `Cell.message` reads a message of a type from a cell: the number for the type `ℕ`,
  `Cell.values` for the type `E × (ℕ → E) × E` of the values of the compression argument
  (SPEC.md Algorithm 9, step 9), and `unpack` of the elements for each other type.
- `Page` is a wire message: a value of the other data of the proof, and `slots` cells.
  `Page.read` reads message `index` after a draw from cell `index`.
- `Cell.Sendable` states that a message is the message of a cell. `Page.sends` is the condition
  `Sends` for it, and `Page.carries` is the condition `Carries` for each part of the data.
- `Page.finite`: a page is finite when its data and `E` are, with the instances for the claims of
  the levels and the statement of the terminal. `ofFn_getD` reads a list back from a vector. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

variable {Data E : Type} {slots width cap : ℕ}

/-- The data of one message of the checks: `width` elements of `E` for an element, a pair or a
round polynomial, and a natural number up to `cap` for the energy of a direct norm check. -/
abbrev Cell (E : Type) (width cap : ℕ) : Type := (Fin width → E) × Fin (cap + 1)

/-- Element `index` of a cell, and 0 from `width`. -/
def Cell.entry [Zero E] (cell : Cell E width cap) (index : ℕ) : E :=
  if inside : index < width then cell.1 ⟨index, inside⟩ else 0

/-- The values `(w, column, k)` of a cell: `w` and `k` are the elements 0 and 1, and value `j` of
`column` is the element `j + 2`. -/
def Cell.values [Zero E] (cell : Cell E width cap) : E × (ℕ → E) × E :=
  (cell.entry 0, fun index => cell.entry (index + 2), cell.entry 1)

open Classical in
/-- The message of the type `M` of a cell: the number of the cell for the natural numbers, the
values `Cell.values` for the type `E × (ℕ → E) × E`, and `unpack` of its elements for each other
type. -/
noncomputable def Cell.message [Zero E] (M : Type) [Nonempty M] (cell : Cell E width cap) : M :=
  if natural : M = ℕ then cast natural.symm (cell.2 : ℕ)
  else if triple : M = (E × (ℕ → E) × E) then cast triple.symm cell.values
  else unpack width E M cell.1

variable (E width cap) in
/-- A message of the checks is the message of a cell. -/
def Cell.Sendable [Zero E] (M : Type) (message : M) : Prop :=
  ∀ inhabited : Nonempty M, ∃ cell : Cell E width cap,
    @Cell.message E width cap _ M inhabited cell = message

/-- A wire message: a value of the other data of the proof, one cell for each of `slots`
messages of the checks after one draw, and the nonce of the response search of a fold as a
number of 32 bits (`FoldResponse` of the Akita code). -/
structure Page (Data E : Type) (slots width cap : ℕ) where
  data : Data
  cells : Fin slots → Cell E width cap
  nonce : Fin (2 ^ 32)

/-- The reader of the messages of the checks: message `index` after a draw is the message of the
cell `index`. -/
noncomputable def Page.read [Zero E] : Reader (Page Data E slots width cap) where
  message M _ wire index :=
    if inside : index < slots then (wire.cells ⟨index, inside⟩).message M
    else Classical.arbitrary M
  nonce wire := wire.nonce

theorem ne_nat (M : Type) [Finite M] : M ≠ ℕ := by
  rintro rfl
  exact not_finite ℕ

theorem ne_values (E : Type) [Nontrivial E] (M : Type) [Finite M] : M ≠ (E × (ℕ → E) × E) := by
  rintro rfl
  exact not_finite (E × (ℕ → E) × E)

open Classical in
/-- The type of the values is not the type of the natural numbers: the sets of natural numbers
have an injective map to the values. -/
theorem values_ne_nat (E : Type) [Nontrivial E] : (E × (ℕ → E) × E) ≠ ℕ := by
  intro same
  obtain ⟨a, b, different⟩ := exists_pair_ne E
  refine Function.cantor_injective
    (fun members : Set ℕ => cast same (a, (fun n => if n ∈ members then a else b), a))
    fun left right equal => ?_
  have columns : (fun n => if n ∈ left then a else b) = fun n => if n ∈ right then a else b :=
    congrArg (fun values : E × (ℕ → E) × E => values.2.1) ((cast_bijective same).1 equal)
  ext n
  have here := congrFun columns n
  by_cases inLeft : n ∈ left <;> by_cases inRight : n ∈ right <;> simp_all

theorem Cell.sendable_of_onto [Zero E] [Nontrivial E] {M : Type} [Finite M]
    (onto : ∃ map : (Fin width → E) → M, Function.Surjective map) (message : M) :
    Cell.Sendable E width cap M message := fun _ => by
  obtain ⟨vector, sent⟩ := unpack_surjective onto message
  exact ⟨(vector, 0), (dite_eq_right (ne_nat M)).trans ((dite_eq_right (ne_values E M)).trans sent)⟩

/-- A cell has each value `(w, column, k)` whose column values are 0 from `width − 2`. -/
theorem Cell.sendable_values [Zero E] [Nontrivial E] (two : 2 ≤ width) {values : E × (ℕ → E) × E}
    (short : ∀ index, width ≤ index + 2 → values.2.1 index = 0) :
    Cell.Sendable E width cap (E × (ℕ → E) × E) values := fun _ => by
  refine ⟨(fun index => if (index : ℕ) = 0 then values.1 else if (index : ℕ) = 1 then values.2.2
    else values.2.1 (index - 2), 0), (dite_eq_right (values_ne_nat E)).trans
      ((dite_eq_left rfl).trans ?_)⟩
  refine Prod.ext ?_ (Prod.ext (funext fun index => ?_) ?_)
  · show Cell.entry _ 0 = _
    unfold Cell.entry
    rw [dite_eq_left (show 0 < width by omega)]
    rfl
  · show Cell.entry _ (index + 2) = _
    unfold Cell.entry
    by_cases inside : index + 2 < width
    · rw [dite_eq_left inside]
      show (if index + 2 = 0 then values.1 else if index + 2 = 1 then values.2.2
        else values.2.1 (index + 2 - 2)) = _
      rw [ite_eq_right (by omega), ite_eq_right (by omega), Nat.add_sub_cancel]
    · rw [dite_eq_right inside]
      exact (short index (by omega)).symm
  · show Cell.entry _ 1 = _
    unfold Cell.entry
    rw [dite_eq_left (show 1 < width by omega)]
    rfl

/-- The messages of the cells satisfy `Sends`: a cell has each element of `E`, each pair, each
vector of at most `width` elements, and each natural number up to `cap`. -/
theorem Page.sends [Finite E] [Zero E] [Nontrivial E] (two : 2 ≤ width) :
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
    else (fun _ => 0, 0), 0⟩, rfl, (by decide : 0 < responseTries), fun index inside => ?_⟩
  have chosen := (each sent[index] (List.getElem_mem inside) sent[index].inhabited).choose_spec
  refine (dite_eq_left (inside.trans_le short)).trans (Eq.trans ?_ chosen)
  exact congrArg (@Cell.message E width cap _ sent[index].M sent[index].inhabited)
    (dite_eq_left inside)

instance Page.finite [Finite Data] [Finite E] : Finite (Page Data E slots width cap) :=
  Finite.of_injective (fun page : Page Data E slots width cap =>
    (page.data, page.cells, page.nonce)) fun left right same => by
      cases left
      cases right
      simp_all

/-- A list of `n` elements is the list of its first `n` entries. -/
theorem ofFn_getD {list : List E} {n : ℕ} (long : list.length = n) (fallback : E) :
    List.ofFn (fun index : Fin n => list.getD index fallback) = list :=
  List.ext_getElem (by rw [List.length_ofFn, long]) fun _ _ inside => by
    rw [List.getElem_ofFn, List.getD_eq_getElem?_getD, Fin.val_mk, List.getElem?_eq_getElem inside,
      Option.getD_some]

end NightstreamFPrime.Spec.Compression.Akita.Extract

namespace NightstreamFPrime.Spec.Compression.Akita

variable {E : Type}

instance Batch.Claim.finite {G : Type} [Finite G] {B : Batch G} [Finite E] :
    Finite (Batch.Claim B E) :=
  Finite.of_injective (fun claim : Batch.Claim B E =>
    (claim.payload, claim.openPayload, claim.point, claim.weight, claim.value))
    fun left right same => by
      cases left
      cases right
      simp_all

variable [CommRing E] [Module Zq E] [Finite E] {S : Shape}

instance TerminalStatement.finite : Finite (TerminalStatement S E) :=
  have : Finite (E →+* Rq S.innerDegree) := Finite.of_injective _ DFunLike.coe_injective
  have : Finite (Rq S.innerDegree →ₗ[Zq] E) := Finite.of_injective _ DFunLike.coe_injective
  Finite.of_injective (fun statement : TerminalStatement S E =>
    (statement.matrix, statement.state, statement.subfield, statement.block, statement.position,
      statement.functional, statement.scale, statement.value)) fun left right same => by
      cases left
      cases right
      simp_all

instance Extract.TraceClaim.finite {O : OpeningShape S} : Finite (Extract.TraceClaim O E) :=
  have : Finite (E →+* Rq S.innerDegree) := Finite.of_injective _ DFunLike.coe_injective
  have : Finite (Rq S.innerDegree →ₗ[Zq] E) := Finite.of_injective _ DFunLike.coe_injective
  Finite.of_injective (fun claim : Extract.TraceClaim O E =>
    (claim.payload, claim.openPayload, claim.subfield, claim.block, claim.position,
      claim.functional, claim.scale, claim.value)) fun left right same => by
      cases left
      cases right
      simp_all

end NightstreamFPrime.Spec.Compression.Akita
