import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Ring.Defs
import SuperNeo.Primitives.Field
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Parameters

/-!
Owns: exact algebraic refinement from product-column selection rows to the
three aggregate equations emitted for one output position, plus the optional
pointwise guarded formulation.

Does not own: transcript or sampler semantics, Rust trace validation, or any
row family outside selection.

Emits constraints: no. It proves equivalence between two row relations.

Authority boundary: the theorem relies on the separately emitted bitness and
selector-sum rows to establish exact one-hotness.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `FixedOneHotPrerequisiteRows`, `fixedOneHotPrerequisiteRows_exactOneHot` | `challenge.sampler.selection.one_hot` | Eleven bitness rows plus one sum row imply exact one-hotness | Goldilocks field and fixed window 11 | No — Rust refinement open |
| `CurrentSelectionBlock`, `currentSelectionBlock_iff_aggregate` | `challenge.sampler.selection.bind.{accept,prefix,symbol}` | Product columns are exactly substitutable into the three emitted aggregate equations | Product definitions | No — concrete Rust row refinement open |
| `GuardedSelectionBlock`, `aggregateSelectionBlock_iff_guarded` | optional pointwise formulation | Aggregate and pointwise-guarded equations agree under exact one-hotness | Exact one-hot selector | No |
| `fixedOneHotRows_currentSelectionBlock_iff_guarded` | `challenge.sampler.selection` | Full source and guarded blocks are model-equivalent | Prerequisite rows and abstract source values | No — concrete Rust row refinement open |

The fixed refinement theorem derives exact one-hotness from the prerequisite
rows. Showing that a concrete emitter produces these rows and either selection
block remains a separate Rust/R1CS conformance obligation.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

open scoped BigOperators

universe u v

/-- The three families of multiplication temporaries in the current block. -/
structure SelectionProducts (ι : Type u) (K : Type v) where
  accepted : ι → K
  prefixWeighted : ι → K
  symbol : ι → K

/-- A selector is exactly one-hot at `selected`. -/
def ExactOneHotAt {ι : Type u} {K : Type v} [One K] [Zero K]
    (oneHot : ι → K) (selected : ι) : Prop :=
  oneHot selected = 1 ∧
    ∀ i, i ≠ selected → oneHot i = 0

/-- A selector is exactly one-hot at some index. -/
def ExactOneHot {ι : Type u} {K : Type v} [One K] [Zero K]
    (oneHot : ι → K) : Prop :=
  ∃ selected, ExactOneHotAt oneHot selected

/--
Exact prerequisite rows emitted for one fixed 11-way selector: one bitness row
per entry and one row fixing the selector sum to one.
-/
def FixedOneHotPrerequisiteRows (oneHot : Fin selectionWindow → SuperNeo.F) : Prop :=
  (∀ i, oneHot i * (oneHot i - 1) = 0) ∧
    (∑ i, oneHot i) = 1

private theorem value_eq_zero_or_one_of_bitness
    {value : SuperNeo.F}
    (hBitness : value * (value - 1) = 0) :
    value = 0 ∨ value = 1 := by
  rcases mul_eq_zero.mp hBitness with hZero | hOne
  · exact Or.inl hZero
  · exact Or.inr (sub_eq_zero.mp hOne)

/-- The exact fixed prerequisite rows force a unique selected index. -/
theorem fixedOneHotPrerequisiteRows_exactOneHot
    {oneHot : Fin selectionWindow → SuperNeo.F}
    (hRows : FixedOneHotPrerequisiteRows oneHot) :
    ExactOneHot oneHot := by
  classical
  let selected := Finset.univ.filter fun i => oneHot i = 1
  have hValue (i : Fin selectionWindow) :
      oneHot i = if i ∈ selected then 1 else 0 := by
    rcases value_eq_zero_or_one_of_bitness (hRows.1 i) with hZero | hOne
    · simp [selected, hZero]
    · simp [selected, hOne]
  have hCardCast : (selected.card : SuperNeo.F) = 1 := by
    calc
      (selected.card : SuperNeo.F) =
          ∑ i : Fin selectionWindow, if i ∈ selected then (1 : SuperNeo.F) else 0 := by
        simp
      _ = ∑ i : Fin selectionWindow, oneHot i := by
        apply Finset.sum_congr rfl
        intro i _
        exact (hValue i).symm
      _ = 1 := hRows.2
  have hCardLt : selected.card < SuperNeo.Goldilocks.q := by
    calc
      selected.card ≤ Fintype.card (Fin selectionWindow) := Finset.card_le_univ selected
      _ = selectionWindow := Fintype.card_fin selectionWindow
      _ < SuperNeo.Goldilocks.q := by decide
  have hCardRep :
      SuperNeo.F.canonicalRep (selected.card : SuperNeo.F) = selected.card := by
    change SuperNeo.F.canonicalRep (SuperNeo.F.ofNat selected.card) = selected.card
    exact SuperNeo.F.canonicalRep_ofNat_eq_of_lt hCardLt
  have hCardEq : selected.card = 1 := by
    calc
      selected.card = SuperNeo.F.canonicalRep (selected.card : SuperNeo.F) := hCardRep.symm
      _ = SuperNeo.F.canonicalRep (1 : SuperNeo.F) :=
        congrArg SuperNeo.F.canonicalRep hCardCast
      _ = 1 := SuperNeo.F.canonicalRep_one
  rcases Finset.card_eq_one.mp hCardEq with ⟨selectedIndex, hSelected⟩
  refine ⟨selectedIndex, ?_, ?_⟩
  · have hMember : selectedIndex ∈ selected := by simp [hSelected]
    exact (Finset.mem_filter.mp hMember).2
  · intro i hNe
    have hNotMember : i ∉ selected := by simp [hSelected, hNe]
    simpa [hNotMember] using hValue i

/-- Each existential product column is constrained to its defining product. -/
def SelectionProductDefinitions {ι : Type u} {K : Type v} [Mul K]
    (oneHot accepts prefixes symbols : ι → K)
    (products : SelectionProducts ι K) : Prop :=
  ∀ i,
    products.accepted i = oneHot i * accepts i ∧
      products.prefixWeighted i = oneHot i * prefixes i ∧
      products.symbol i = oneHot i * symbols i

/-- Current block: existential products followed by three aggregate equations. -/
def CurrentSelectionBlock {ι : Type u} {K : Type v}
    [CommRing K] [Fintype ι]
    (oneHot accepts prefixes symbols : ι → K)
    (position output : K) : Prop :=
  ∃ products : SelectionProducts ι K,
    SelectionProductDefinitions oneHot accepts prefixes symbols products ∧
      (∑ i, products.accepted i) = 1 ∧
      (∑ i, products.prefixWeighted i) = position ∧
      output = ∑ i, products.symbol i

/-- The current aggregate equations after substituting product definitions. -/
def AggregateSelectionBlock {ι : Type u} {K : Type v}
    [CommRing K] [Fintype ι]
    (oneHot accepts prefixes symbols : ι → K)
    (position output : K) : Prop :=
  (∑ i, oneHot i * accepts i) = 1 ∧
    (∑ i, oneHot i * prefixes i) = position ∧
    output = ∑ i, oneHot i * symbols i

/-- Cheaper pointwise equations guarded by each one-hot selector entry. -/
def GuardedSelectionBlock {ι : Type u} {K : Type v} [CommRing K]
    (oneHot accepts prefixes symbols : ι → K)
    (position output : K) : Prop :=
  ∀ i,
    oneHot i * (accepts i - 1) = 0 ∧
      oneHot i * (prefixes i - position) = 0 ∧
      oneHot i * (symbols i - output) = 0

private theorem eq_of_difference_eq_zero
    {K : Type v} [AddCommGroup K] {left right : K}
    (hDifference : left - right = 0) : left = right := by
  calc
    left = (left - right) + right := by
      simp [sub_eq_add_neg, add_assoc]
    _ = 0 + right := by rw [hDifference]
    _ = right := zero_add right

theorem currentSelectionBlock_iff_aggregate
    {ι : Type u} {K : Type v} [CommRing K] [Fintype ι]
    (oneHot accepts prefixes symbols : ι → K)
    (position output : K) :
    CurrentSelectionBlock oneHot accepts prefixes symbols position output ↔
      AggregateSelectionBlock oneHot accepts prefixes symbols position output := by
  classical
  constructor
  · rintro ⟨products, hProducts, hAccepted, hPrefix, hSymbol⟩
    refine ⟨?_, ?_, ?_⟩
    · calc
        (∑ i, oneHot i * accepts i) = ∑ i, products.accepted i := by
          apply Finset.sum_congr rfl
          intro i _
          exact (hProducts i).1.symm
        _ = 1 := hAccepted
    · calc
        (∑ i, oneHot i * prefixes i) =
            ∑ i, products.prefixWeighted i := by
          apply Finset.sum_congr rfl
          intro i _
          exact (hProducts i).2.1.symm
        _ = position := hPrefix
    · calc
        output = ∑ i, products.symbol i := hSymbol
        _ = ∑ i, oneHot i * symbols i := by
          apply Finset.sum_congr rfl
          intro i _
          exact (hProducts i).2.2
  · rintro ⟨hAccepted, hPrefix, hSymbol⟩
    let products : SelectionProducts ι K :=
      { accepted := fun i => oneHot i * accepts i
        prefixWeighted := fun i => oneHot i * prefixes i
        symbol := fun i => oneHot i * symbols i }
    refine ⟨products, ?_, ?_, ?_, ?_⟩
    · intro i
      exact ⟨rfl, rfl, rfl⟩
    · simpa [products] using hAccepted
    · simpa [products] using hPrefix
    · simpa [products] using hSymbol

theorem exactOneHotAt_weighted_sum
    {ι : Type u} {K : Type v}
    [CommRing K] [Fintype ι] [DecidableEq ι]
    {oneHot : ι → K} {selected : ι}
    (hOneHot : ExactOneHotAt oneHot selected)
    (values : ι → K) :
    (∑ i, oneHot i * values i) = values selected := by
  calc
    (∑ i, oneHot i * values i) = oneHot selected * values selected := by
      apply Finset.sum_eq_single selected
      · intro i _ hNe
        simp [hOneHot.2 i hNe]
      · simp
    _ = values selected := by simp [hOneHot.1]

theorem aggregateSelectionBlock_iff_guarded
    {ι : Type u} {K : Type v}
    [CommRing K] [Fintype ι] [DecidableEq ι]
    {oneHot accepts prefixes symbols : ι → K}
    {position output : K}
    (hOneHot : ExactOneHot oneHot) :
    AggregateSelectionBlock oneHot accepts prefixes symbols position output ↔
      GuardedSelectionBlock oneHot accepts prefixes symbols position output := by
  rcases hOneHot with ⟨selected, hAt⟩
  have hAcceptSum := exactOneHotAt_weighted_sum hAt accepts
  have hPrefixSum := exactOneHotAt_weighted_sum hAt prefixes
  have hSymbolSum := exactOneHotAt_weighted_sum hAt symbols
  constructor
  · rintro ⟨hAccept, hPrefix, hSymbol⟩ i
    have hSelectedAccept : accepts selected = 1 := by
      rw [← hAcceptSum]
      exact hAccept
    have hSelectedPrefix : prefixes selected = position := by
      rw [← hPrefixSum]
      exact hPrefix
    have hSelectedSymbol : symbols selected = output :=
      hSymbolSum.symm.trans hSymbol.symm
    by_cases hIndex : i = selected
    · subst i
      simp [hAt.1, hSelectedAccept, hSelectedPrefix, hSelectedSymbol]
    · simp [hAt.2 i hIndex]
  · intro hGuarded
    rcases hGuarded selected with ⟨hAccept, hPrefix, hSymbol⟩
    have hSelectedAccept : accepts selected = 1 := by
      apply eq_of_difference_eq_zero
      simpa [hAt.1] using hAccept
    have hSelectedPrefix : prefixes selected = position := by
      apply eq_of_difference_eq_zero
      simpa [hAt.1] using hPrefix
    have hSelectedSymbol : symbols selected = output := by
      apply eq_of_difference_eq_zero
      simpa [hAt.1] using hSymbol
    refine ⟨?_, ?_, ?_⟩
    · rw [hAcceptSum]
      exact hSelectedAccept
    · rw [hPrefixSum]
      exact hSelectedPrefix
    · rw [hSymbolSum]
      exact hSelectedSymbol.symm

/-- Exact proof-backed replacement theorem for one selection output position. -/
theorem currentSelectionBlock_iff_guarded
    {ι : Type u} {K : Type v}
    [CommRing K] [Fintype ι] [DecidableEq ι]
    {oneHot accepts prefixes symbols : ι → K}
    {position output : K}
    (hOneHot : ExactOneHot oneHot) :
    CurrentSelectionBlock oneHot accepts prefixes symbols position output ↔
      GuardedSelectionBlock oneHot accepts prefixes symbols position output := by
  rw [currentSelectionBlock_iff_aggregate]
  exact aggregateSelectionBlock_iff_guarded hOneHot

/--
Concrete fixed-selector refinement: the emitted prerequisite rows discharge the
one-hot premise required to replace the current aggregate selection block.
-/
theorem fixedOneHotRows_currentSelectionBlock_iff_guarded
    {oneHot accepts prefixes symbols : Fin selectionWindow → SuperNeo.F}
    {position output : SuperNeo.F}
    (hRows : FixedOneHotPrerequisiteRows oneHot) :
    CurrentSelectionBlock oneHot accepts prefixes symbols position output ↔
      GuardedSelectionBlock oneHot accepts prefixes symbols position output :=
  currentSelectionBlock_iff_guarded
    (fixedOneHotPrerequisiteRows_exactOneHot hRows)

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
