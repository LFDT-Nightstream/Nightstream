import NightstreamFPrime.Export.BoundedDepth.PackedBooleanCompleteness
import NightstreamFPrime.Export.Generic.PackedProduct

set_option autoImplicit false

/-! The width budget of the packed Boolean lowering (SPEC.md section 7.9). `choose`, `pairs`,
`sumColumns` and `keep` mirror `PackedBooleans::lower` in
`crates/nightstream-fprime/src/bounded_depth/packed_boolean.rs`. -/

namespace NightstreamFPrime.Export.BoundedDepth.PackedBudget

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Export NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.RelationCompleteness
open NightstreamFPrime.Export.Generic.PackedBoolean (Pair companions Covers)
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix

variable {α β γ : Type}

/-- The packed sources, zero products and products. -/
structure Choice (α β γ : Type) where
  sources : List α
  zeros : List β
  products : List γ

/-- The added columns: one per source, two per product and one per zero product. -/
def Choice.columns (choice : Choice α β γ) : Nat :=
  choice.sources.length + 2 * choice.products.length + choice.zeros.length

/-- Rust mirror: the truncations of `PackedBooleans::lower`, with `room = width - start`. -/
def choose (room : Nat) (sources : List α) (zeros : List β) (products : List γ) : Choice α β γ :=
  let chosenSources := sources.take room
  let chosenZeros := zeros.take (room - chosenSources.length)
  ⟨chosenSources, chosenZeros,
    products.take ((room - chosenSources.length - chosenZeros.length) / 2)⟩

theorem choose_lengths (room : Nat) (sources : List α) (zeros : List β) (products : List γ) :
    let choice := choose room sources zeros products
    choice.sources.length = min room sources.length ∧
      choice.zeros.length = min (room - choice.sources.length) zeros.length ∧
      choice.products.length =
        min ((room - choice.sources.length - choice.zeros.length) / 2) products.length := by
  simp only [choose, List.length_take, and_self]

theorem choose_fits (room : Nat) (sources : List α) (zeros : List β) (products : List γ) :
    (choose room sources zeros products).columns ≤ room := by
  simp only [choose, Choice.columns, List.length_take]
  omega

/-- The end of the added columns is at most `width`, the check of `PackedBooleans::validate`. -/
theorem choose_end {start width : Nat} (sources : List α) (zeros : List β) (products : List γ)
    (inside : start ≤ width) :
    start + (choose (width - start) sources zeros products).columns ≤ width := by
  have fit := choose_fits (width - start) sources zeros products
  omega

theorem choose_prefix (room : Nat) (sources : List α) (zeros : List β) (products : List γ) :
    let choice := choose room sources zeros products
    choice.sources <+: sources ∧ choice.zeros <+: zeros ∧ choice.products <+: products :=
  ⟨List.take_prefix _ _, List.take_prefix _ _, List.take_prefix _ _⟩

/-- When every added column fits, nothing is dropped, so the packages are unchanged. -/
theorem choose_of_fits {room : Nat} {sources : List α} {zeros : List β} {products : List γ}
    (fit : sources.length + 2 * products.length + zeros.length ≤ room) :
    choose room sources zeros products = ⟨sources, zeros, products⟩ := by
  have allSources : sources.take room = sources := List.take_of_length_le (by omega)
  have allZeros : zeros.take (room - sources.length) = zeros := List.take_of_length_le (by omega)
  have allProducts : products.take ((room - sources.length - zeros.length) / 2) = products :=
    List.take_of_length_le (by omega)
  simp only [choose, allSources, allZeros, allProducts]

/-- Rust mirror: the parity pairs of `PackedBooleans::pairs`, two columns per product after the
source companions. -/
def parityPairs (productStart count : Nat) : List Pair :=
  List.ofFn fun index : Fin count =>
    (productStart + 2 * index.val, productStart + 2 * index.val + 1)

/-- Rust mirror: `PackedBooleans::pairs`. -/
def pairs (start : Nat) (sources : List Nat) (products : Nat) : List Pair :=
  companions start sources ++ parityPairs (start + sources.length) products

/-- Rust mirror: the sum columns of `packed_zero::rows`, one per zero product after the parity
pairs. -/
def sumColumns (start : Nat) (sources : List Nat) (products zeros : Nat) : List Nat :=
  List.range' (start + sources.length + 2 * products) zeros

theorem pairs_columns {start : Nat} {sources : List Nat} {products : Nat}
    (before : ∀ source ∈ sources, source < start) {pair : Pair}
    (member : pair ∈ pairs start sources products) :
    pair.1 < start + sources.length + 2 * products ∧ start ≤ pair.2 ∧
      pair.2 < start + sources.length + 2 * products := by
  rcases List.mem_append.mp member with source | parity
  · obtain ⟨index, rfl⟩ := List.mem_ofFn.mp source
    have below := before _ (List.get_mem sources index)
    have bound := index.isLt
    dsimp only
    omega
  · obtain ⟨index, rfl⟩ := List.mem_ofFn.mp parity
    have bound := index.isLt
    dsimp only
    omega

/-- With `room = width - start`, the chosen pairs and sum columns lie below `width`, and each
added column is at or after `start`. -/
theorem chosen_inside (start width : Nat) (sources : List Nat) (zeros : List β) (products : List γ)
    (before : ∀ source ∈ sources, source < start) :
    let choice := choose (width - start) sources zeros products
    (∀ pair ∈ pairs start choice.sources choice.products.length,
      pair.1 < width ∧ start ≤ pair.2 ∧ pair.2 < width) ∧
    ∀ column ∈ sumColumns start choice.sources choice.products.length choice.zeros.length,
      start ≤ column ∧ column < width := by
  intro choice
  have fit := choose_fits (width - start) sources zeros products
  change choice.columns ≤ width - start at fit
  unfold Choice.columns at fit
  have chosenBefore : ∀ source ∈ choice.sources, source < start :=
    fun source member => before source (List.mem_of_mem_take member)
  refine ⟨fun pair member => ?_, fun column member => ?_⟩
  · have bounds := pairs_columns chosenBefore member
    omega
  · have bounds := List.mem_range'_1.mp member
    omega

theorem row_vars_below {width : Nat} (group : List Pair)
    (inside : ∀ pair ∈ group, pair.1 < width ∧ pair.2 < width) :
    (PackedBoolean.row group).VarsSatisfy (· < width) := by
  have linear : (PackedBoolean.linear group).VarsSatisfy (· < width) := by
    induction group with
    | nil => intro term member; simp [PackedBoolean.linear, R1CS.LinearCombination.zero] at member
    | cons pair rest ih =>
      have head := inside pair List.mem_cons_self
      have tail := ih (fun p member => inside p (List.mem_cons_of_mem _ member))
      refine R1CS.LinearCombination.VarsSatisfy.add _ _ _ ?_ ?_
      · refine R1CS.LinearCombination.VarsSatisfy.add _ _ _ ?_ ?_
        · exact R1CS.LinearCombination.VarsSatisfy.add _ _ _
            (R1CS.LinearCombination.VarsSatisfy.ofVar _ _ head.1)
            (R1CS.LinearCombination.VarsSatisfy.ofVar _ _ head.2)
        · intro term member; simp [R1CS.LinearCombination.const] at member
      · exact R1CS.LinearCombination.VarsSatisfy.scale _ _ _ tail
  refine ⟨linear, ?_, ?_⟩ <;> intro term member <;>
    simp [PackedBoolean.row, R1CS.LinearCombination.one, R1CS.LinearCombination.zero] at member

private theorem foldl_max_lt {δ : Type} (measure : δ → Nat) {bound : Nat} :
    ∀ (values : List δ) (initial : Nat), initial < bound →
      (∀ value ∈ values, measure value < bound) →
        values.foldl (fun current value => max current (measure value)) initial < bound
  | [], _, start, _ => start
  | value :: rest, _, start, all =>
      foldl_max_lt measure rest _ (max_lt start (all value List.mem_cons_self))
        (fun other member => all other (List.mem_cons_of_mem _ member))

/-- The packed pair rows of the chosen pairs read only columns below `width`. This gives the
`column < width` part of the `BitSource` premises of `PackedBoolean.restore_logical_rows` and of
the `covered` premise of `PackedBooleanCompleteness.native_complete`. -/
theorem groups_inside (start width : Nat) (sources : List Nat) (zeros : List β) (products : List γ)
    (before : ∀ source ∈ sources, source < start) (groups : List (List Pair))
    (grouped : let choice := choose (width - start) sources zeros products
      groups.flatten = pairs start choice.sources choice.products.length) :
    ∀ group ∈ groups, (∀ pair ∈ group, pair.1 < width ∧ pair.2 < width) ∧
      (PackedBoolean.row group).VarsSatisfy (· < width) := by
  intro group groupMember
  have inside : ∀ pair ∈ group, pair.1 < width ∧ pair.2 < width := by
    intro pair pairMember
    have chosen := List.mem_flatten.mpr ⟨group, groupMember, pairMember⟩
    rw [grouped] at chosen
    have bounds := (chosen_inside start width sources zeros products before).1 pair chosen
    exact ⟨bounds.1, bounds.2.2⟩
  exact ⟨inside, row_vars_below group inside⟩

/-- Rows that read only columns below a positive `width` meet `LogicalPlan.Fits.columns`. -/
theorem rowsMax_lt {rows : List R1CS.Row} {width : Nat} (positive : 0 < width)
    (inside : ∀ row ∈ rows, row.VarsSatisfy (· < width)) : LogicalPlan.rowsMax rows < width := by
  have combination : ∀ side : R1CS.LinearCombination, side.VarsSatisfy (· < width) →
      LogicalPlan.combinationMax side < width :=
    fun side scope => foldl_max_lt Prod.fst side.terms 0 positive scope
  exact foldl_max_lt LogicalPlan.rowMax rows 0 positive fun row member =>
    max_lt (combination _ (inside row member).1)
      (max_lt (combination _ (inside row member).2.1) (combination _ (inside row member).2.2))

/-- Rust mirror: the row filter of `PackedBooleans::lower`. A row leaves when it forces a chosen
source or when its index is the row of a chosen product or zero product. -/
def removed (sources gates : List Nat) (index : Nat) (row : R1CS.Row) : Bool :=
  (SlotLayout.forcedColumn? row).any (fun column => sources.contains column) || gates.contains index

/-- Rust mirror: the rows that `PackedBooleans::lower` keeps, in their order. -/
def keep (sources gates : List Nat) (original : List R1CS.Row) : List R1CS.Row :=
  (original.zipIdx.filter fun entry => !removed sources gates entry.2 entry.1).map Prod.fst

theorem keep_sublist (sources gates : List Nat) (original : List R1CS.Row) :
    (keep sources gates original).Sublist original := by
  have whole := (List.filter_sublist (l := original.zipIdx)
    (p := fun entry => !removed sources gates entry.2 entry.1)).map Prod.fst
  rwa [List.zipIdx_map_fst] at whole

theorem mem_keep {sources gates : List Nat} {original : List R1CS.Row} {row : R1CS.Row} :
    row ∈ keep sources gates original ↔
      ∃ index, original[index]? = some row ∧ removed sources gates index row = false := by
  simp [keep, List.mem_zipIdx_iff_getElem?]

theorem keep_nil (original : List R1CS.Row) : keep [] [] original = original := by
  have none : ∀ entry : R1CS.Row × Nat, (!removed [] [] entry.2 entry.1) = true := by
    intro entry
    cases found : SlotLayout.forcedColumn? entry.1 <;> simp [removed, found]
  simp only [keep, List.filter_eq_self.mpr fun entry _ => none entry, List.zipIdx_map_fst]

/-- The kept rows and the chosen sources cover every row that is not a gate row. So
`PackedBoolean.restore_rows` applies to any chosen sources, in particular to a prefix. -/
theorem covers_keep (start : Nat) (sources gates : List Nat) (products : Nat)
    (original : List R1CS.Row) (groups : List (List Pair))
    (grouped : groups.flatten = pairs start sources products) :
    Covers (keep [] gates original) (keep sources gates original) groups := by
  intro row member
  obtain ⟨index, found, outside⟩ := mem_keep.mp member
  have notGate : index ∉ gates := by simpa [removed] using outside
  cases leaves : removed sources gates index row
  · exact Or.inl (mem_keep.mpr ⟨index, found, leaves⟩)
  · right
    cases forced : SlotLayout.forcedColumn? row with
    | none =>
      simp only [removed, forced, Option.any_none, Bool.false_or, List.contains_iff_mem] at leaves
      exact absurd leaves notGate
    | some column =>
      have inside : column ∈ sources := by
        simp only [removed, forced, Option.any_some, Bool.or_eq_true, List.contains_iff_mem] at leaves
        exact leaves.resolve_right notGate
      obtain ⟨position, rfl⟩ := List.get_of_mem inside
      have pairMember : (sources.get position, start + position.val) ∈
          pairs start sources products :=
        List.mem_append_left _ (List.mem_ofFn.mpr ⟨position, rfl⟩)
      rw [← grouped] at pairMember
      obtain ⟨group, groupMember, inGroup⟩ := List.mem_flatten.mp pairMember
      exact ⟨group, groupMember, _, inGroup, rfl⟩

/-- A prefix of the checked products keeps both product premises of
`PackedProduct.restore_rows`. A dropped product stays in the kept rows as its original row.
`Gate.Ready` reads only the original rows, so it does not depend on the other packed gates. -/
theorem prefix_products (original seeds : List R1CS.Row) (gates : List PackedProduct.Gate)
    (count : Nat)
    (matched : ∀ gate ∈ gates, ∀ inside : gate.row < original.length,
      gate.Matches original[gate.row])
    (ready : ∀ gate ∈ gates, gate.Ready original seeds) :
    (∀ i (inside : i < original.length),
      original[i] ∈ keep [] ((gates.take count).map PackedProduct.Gate.row) original ∨
        ∃ gate ∈ gates.take count, gate.row = i ∧ gate.Matches original[i]) ∧
    ∀ gate ∈ gates.take count, gate.Ready original seeds := by
  refine ⟨fun i inside => ?_, fun gate member => ready gate (List.mem_of_mem_take member)⟩
  by_cases chosen : i ∈ (gates.take count).map PackedProduct.Gate.row
  · obtain ⟨gate, member, position⟩ := List.mem_map.mp chosen
    subst position
    exact Or.inr ⟨gate, member, rfl, matched gate (List.mem_of_mem_take member) inside⟩
  · refine Or.inl (mem_keep.mpr ⟨i, List.getElem?_eq_getElem inside, ?_⟩)
    simp only [List.map_take] at chosen
    cases forced : SlotLayout.forcedColumn? original[i] <;> simp [removed, forced, chosen]

/-- Soundness of a partial packing: the restore theorems compose for any chosen sources and
products. The zero-product rows are a premise, since Lean has no zero-product row map. -/
theorem restore_chosen_rows (original seeds : List R1CS.Row) (start : Nat) (sources : List Nat)
    (gates : List PackedProduct.Gate) (zeroRows : List Nat) (groups : List (List Pair))
    (grouped : groups.flatten = pairs start sources gates.length)
    (counts : ∀ group ∈ groups, group.length ≤ PackedBoolean.capacity)
    (matched : ∀ gate ∈ gates, ∀ inside : gate.row < original.length,
      gate.Matches original[gate.row])
    (ready : ∀ gate ∈ gates, gate.Ready original seeds)
    {env : Env} (one : env 0 = 1)
    (pairNorm : ∀ group ∈ groups, ∀ pair ∈ group,
      centeredMagnitude (env pair.1) < 2 ∧ centeredMagnitude (env pair.2) < 2)
    (holds : R1CS.RowsHold env
      (keep sources (gates.map PackedProduct.Gate.row ++ zeroRows) original ++
        groups.map PackedBoolean.row))
    (zeroHold : ∀ index (inside : index < original.length), index ∈ zeroRows →
      original[index].Holds env)
    (seedsHold : R1CS.RowsHold env seeds)
    (gateNorm : ∀ gate ∈ gates, (gate.values env).Norm)
    (residuals : ∀ gate ∈ gates, (gate.values env).residual = 0) :
    R1CS.RowsHold env original := by
  have plain := PackedBoolean.restore_rows _ _ groups
    (covers_keep start sources (gates.map PackedProduct.Gate.row ++ zeroRows) gates.length
      original groups grouped) counts one pairNorm holds
  obtain ⟨cover, _⟩ := prefix_products original seeds gates gates.length matched ready
  rw [List.take_length] at cover
  have keptHold : R1CS.RowsHold env (keep [] (gates.map PackedProduct.Gate.row) original) := by
    intro row member
    obtain ⟨index, found, outside⟩ := mem_keep.mp member
    obtain ⟨inside, rfl⟩ := List.getElem?_eq_some_iff.mp found
    by_cases zero : index ∈ zeroRows
    · exact zeroHold index inside zero
    · apply plain
      refine mem_keep.mpr ⟨index, found, ?_⟩
      cases forced : SlotLayout.forcedColumn? original[index] <;>
        simp_all [removed]
  exact PackedProduct.restore_rows original _ seeds gates cover ready one seedsHold keptHold
    gateNorm residuals

open PackedBooleanCompleteness in
/-- The chosen sources meet the source, row and pair premises of
`PackedBooleanCompleteness.native_complete`, and its packed rows read only columns below
`width`. -/
theorem native_premises {pieces : List Pieces.Piece} {next width : Nat}
    (nativeRows : List R1CS.Row) (sources : List Nat) (zeros : List β) (products : List γ)
    (groups : List (List Pair))
    (before : ∀ source ∈ sources, source < slotEnd pieces next)
    (forced : ∀ source ∈ sources, PackedBoolean.SourceForced nativeRows source)
    (grouped : groups.flatten = companions (slotEnd pieces next)
      (choose (width - slotEnd pieces next) sources zeros products).sources) :
    let chosen := (choose (width - slotEnd pieces next) sources zeros products).sources
    (∀ source ∈ chosen, source < slotEnd pieces next) ∧
      (∀ source ∈ chosen, PackedBoolean.SourceForced nativeRows source) ∧
      (∀ r ∈ keep chosen [] (nativeRows.map
          (Substitute.row (outputForms width (blocksOf pieces) (sboxStart pieces next)))),
        r ∈ originalRows nativeRows pieces next width) ∧
      (∀ group ∈ groups, ∀ pair ∈ group, pair ∈ companions (slotEnd pieces next) chosen) ∧
      ∀ group ∈ groups, (PackedBoolean.row group).VarsSatisfy (· < width) := by
  intro chosen
  have paired : ∀ group ∈ groups, ∀ pair ∈ group,
      pair ∈ companions (slotEnd pieces next) chosen := by
    intro group groupMember pair pairMember
    rw [← grouped]
    exact List.mem_flatten.mpr ⟨group, groupMember, pairMember⟩
  have inside := (chosen_inside (slotEnd pieces next) width sources zeros products before).1
  refine ⟨fun source member => before source (List.mem_of_mem_take member),
    fun source member => forced source (List.mem_of_mem_take member), ?_, paired, ?_⟩
  · intro r member
    have native := (keep_sublist _ _ _).subset member
    unfold originalRows
    rw [List.map_append]
    exact List.mem_append_left _ native
  · intro group groupMember
    apply row_vars_below
    intro pair pairMember
    have bounds := inside pair (List.mem_append_left _ (paired group groupMember pair pairMember))
    exact ⟨bounds.1, bounds.2.2⟩

end NightstreamFPrime.Export.BoundedDepth.PackedBudget
