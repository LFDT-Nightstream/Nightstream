import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.OneHot

/-!
Semantic refinement of one complete `Pi_RLC` 54-of-64 output-position family.

Owns: the 33 selector/source product rows, the accepted/prefix/symbol binding
rows, and the proof that a selected routing witness names an accepted candidate
whose prior accepted count is the requested output position and whose symbol
is copied to the output.

Does not own: proof that such a candidate is the mathematical first-accepted
candidate, aggregation across 54 positions, production column placement,
coefficient assembly, Rust conformance, row removal, or costs.

Emits constraints: no.

Authority boundary: selectors and product columns are non-authoritative routing
witnesses. The three binding equations are meaningful only after the source
accept/prefix/symbol columns are independently tied to verifier-owned lane
semantics.

| Protocol | Phase | Constraint family | Multiplicity | Lean guarantee |
|---|---|---|---:|---|
| `Pi_RLC` | sampler/selection position | selector × symbol | 11 | selected symbol product equals the source symbol; all others are zero |
| `Pi_RLC` | sampler/selection position | selector × accept | 11 | selected accept product equals the source accept bit; all others are zero |
| `Pi_RLC` | sampler/selection position | selector × prefix | 11 | selected prefix product equals the source prefix count; all others are zero |
| `Pi_RLC` | sampler/selection position | accept/prefix/symbol bindings | 3 | selected candidate is accepted at this prefix and its symbol is the output |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Position

open Nightstream.Implementation.R1CS

private theorem range11 :
    List.range SelectionRows.selectionWindow =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] := by
  decide

private theorem satisfies_selectionRows
    {assignment : Nat -> Nat} {position : Nat}
    (positionLt : position < SelectionRows.outputCount)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    Satisfies (SelectionRows.selectionRows position) assignment := by
  intro row member
  apply satisfies row
  rw [SelectionRows.rows]
  apply List.mem_append_right
  exact List.mem_flatMap.mpr
    ⟨position, List.mem_range.mpr positionLt, member⟩

private theorem satisfies_productRowsAt
    {assignment : Nat -> Nat} {position offset : Nat}
    (positionLt : position < SelectionRows.outputCount)
    (offsetLt : offset < SelectionRows.selectionWindow)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    Satisfies (SelectionRows.productRowsAt position offset) assignment := by
  intro row member
  apply satisfies_selectionRows positionLt satisfies row
  rw [SelectionRows.selectionRows]
  apply List.mem_append_left
  apply List.mem_append_right
  rw [SelectionRows.productRows]
  exact List.mem_flatMap.mpr
    ⟨offset, List.mem_range.mpr offsetLt, member⟩

private theorem satisfies_bindingRows
    {assignment : Nat -> Nat} {position : Nat}
    (positionLt : position < SelectionRows.outputCount)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    Satisfies (SelectionRows.bindingRows position) assignment := by
  intro row member
  apply satisfies_selectionRows positionLt satisfies row
  rw [SelectionRows.selectionRows]
  exact List.mem_append_right _ member

private theorem product_eq_source_of_selector_one
    {assignment : Nat -> Nat} {selector source product : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (selectorOne : assignment selector = 1)
    (holds : RowHolds assignment
      ⟨[(selector, 1)], [(source, 1)], [(product, 1)]⟩) :
    assignment product = assignment source := by
  have selectorCanonical := canonical selector
  have sourceCanonical := canonical source
  have productCanonical := canonical product
  simpa [RowHolds, lcEval, selectorOne,
    Nat.mod_eq_of_lt selectorCanonical,
    Nat.mod_eq_of_lt sourceCanonical,
    Nat.mod_eq_of_lt productCanonical] using holds.symm

private theorem product_eq_zero_of_selector_zero
    {assignment : Nat -> Nat} {selector source product : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (selectorZero : assignment selector = 0)
    (holds : RowHolds assignment
      ⟨[(selector, 1)], [(source, 1)], [(product, 1)]⟩) :
    assignment product = 0 := by
  have selectorCanonical := canonical selector
  have productCanonical := canonical product
  simpa [RowHolds, lcEval, selectorZero,
    Nat.mod_eq_of_lt selectorCanonical,
    Nat.mod_eq_of_lt productCanonical] using holds.symm

theorem symbolProduct_eq_if
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat} {position selected offset : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (positionLt : position < SelectionRows.outputCount)
    (selectedLt : selected < SelectionRows.selectionWindow)
    (offsetLt : offset < SelectionRows.selectionWindow)
    (selectedOne :
      assignment (SelectionRows.selectorCol position selected) = 1)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    assignment (SelectionRows.symbolProductCol position offset) =
      if offset = selected then
        assignment (SelectionRows.symbolCol (position + offset)) else 0 := by
  have holds := satisfies_productRowsAt positionLt offsetLt satisfies
    ⟨[(SelectionRows.selectorCol position offset, 1)],
      [(SelectionRows.symbolCol (position + offset), 1)],
      [(SelectionRows.symbolProductCol position offset, 1)]⟩
    (by simp [SelectionRows.productRowsAt])
  change RowHolds assignment
    ⟨[(SelectionRows.selectorCol position offset, 1)],
      [(SelectionRows.symbolCol (position + offset), 1)],
      [(SelectionRows.symbolProductCol position offset, 1)]⟩ at holds
  by_cases same : offset = selected
  · subst offset
    simp only [↓reduceIte]
    exact product_eq_source_of_selector_one canonical selectedOne holds
  · rw [if_neg same]
    have selectorZero := OneHot.selector_eq_zero_of_ne prime canonical one
      positionLt selectedLt offsetLt selectedOne same satisfies
    exact product_eq_zero_of_selector_zero canonical selectorZero holds

theorem acceptProduct_eq_if
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat} {position selected offset : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (positionLt : position < SelectionRows.outputCount)
    (selectedLt : selected < SelectionRows.selectionWindow)
    (offsetLt : offset < SelectionRows.selectionWindow)
    (selectedOne :
      assignment (SelectionRows.selectorCol position selected) = 1)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    assignment (SelectionRows.acceptProductCol position offset) =
      if offset = selected then
        assignment (SelectionRows.acceptCol (position + offset)) else 0 := by
  have holds := satisfies_productRowsAt positionLt offsetLt satisfies
    ⟨[(SelectionRows.selectorCol position offset, 1)],
      [(SelectionRows.acceptCol (position + offset), 1)],
      [(SelectionRows.acceptProductCol position offset, 1)]⟩
    (by simp [SelectionRows.productRowsAt])
  change RowHolds assignment
    ⟨[(SelectionRows.selectorCol position offset, 1)],
      [(SelectionRows.acceptCol (position + offset), 1)],
      [(SelectionRows.acceptProductCol position offset, 1)]⟩ at holds
  by_cases same : offset = selected
  · subst offset
    simp only [↓reduceIte]
    exact product_eq_source_of_selector_one canonical selectedOne holds
  · rw [if_neg same]
    have selectorZero := OneHot.selector_eq_zero_of_ne prime canonical one
      positionLt selectedLt offsetLt selectedOne same satisfies
    exact product_eq_zero_of_selector_zero canonical selectorZero holds

theorem prefixProduct_eq_if
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat} {position selected offset : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (positionLt : position < SelectionRows.outputCount)
    (selectedLt : selected < SelectionRows.selectionWindow)
    (offsetLt : offset < SelectionRows.selectionWindow)
    (selectedOne :
      assignment (SelectionRows.selectorCol position selected) = 1)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    assignment (SelectionRows.prefixProductCol position offset) =
      if offset = selected then
        assignment (SelectionRows.prefixCol (position + offset)) else 0 := by
  have holds := satisfies_productRowsAt positionLt offsetLt satisfies
    ⟨[(SelectionRows.selectorCol position offset, 1)],
      [(SelectionRows.prefixCol (position + offset), 1)],
      [(SelectionRows.prefixProductCol position offset, 1)]⟩
    (by simp [SelectionRows.productRowsAt])
  change RowHolds assignment
    ⟨[(SelectionRows.selectorCol position offset, 1)],
      [(SelectionRows.prefixCol (position + offset), 1)],
      [(SelectionRows.prefixProductCol position offset, 1)]⟩ at holds
  by_cases same : offset = selected
  · subst offset
    simp only [↓reduceIte]
    exact product_eq_source_of_selector_one canonical selectedOne holds
  · rw [if_neg same]
    have selectorZero := OneHot.selector_eq_zero_of_ne prime canonical one
      positionLt selectedLt offsetLt selectedOne same satisfies
    exact product_eq_zero_of_selector_zero canonical selectorZero holds

def symbolProductTerms (position : Nat) : List (Nat × Nat) :=
  (List.range SelectionRows.selectionWindow).map
    (fun offset => (SelectionRows.symbolProductCol position offset, 1))

def acceptProductTerms (position : Nat) : List (Nat × Nat) :=
  (List.range SelectionRows.selectionWindow).map
    (fun offset => (SelectionRows.acceptProductCol position offset, 1))

def prefixProductTerms (position : Nat) : List (Nat × Nat) :=
  (List.range SelectionRows.selectionWindow).map
    (fun offset => (SelectionRows.prefixProductCol position offset, 1))

private theorem productTerms_canonical
    (columns : Nat -> Nat) :
    Program.CanonicalTerms
      ((List.range SelectionRows.selectionWindow).map
        (fun offset => (columns offset, 1))) := by
  intro term member
  rcases List.mem_map.mp member with ⟨offset, _, rfl⟩
  change 0 < 1 ∧ 1 < goldilocksP
  decide

private theorem lcEval_selectedProducts
    {assignment : Nat -> Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (product source : Nat -> Nat) (selected : Nat)
    (selectedLt : selected < SelectionRows.selectionWindow)
    (productEq : ∀ offset, offset < SelectionRows.selectionWindow ->
      assignment (product offset) =
        if offset = selected then assignment (source offset) else 0) :
    lcEval assignment
        ((List.range SelectionRows.selectionWindow).map
          (fun offset => (product offset, 1))) =
      assignment (source selected) := by
  have product0 := productEq 0 (by decide)
  have product1 := productEq 1 (by decide)
  have product2 := productEq 2 (by decide)
  have product3 := productEq 3 (by decide)
  have product4 := productEq 4 (by decide)
  have product5 := productEq 5 (by decide)
  have product6 := productEq 6 (by decide)
  have product7 := productEq 7 (by decide)
  have product8 := productEq 8 (by decide)
  have product9 := productEq 9 (by decide)
  have product10 := productEq 10 (by decide)
  have selectedCases : selected = 0 ∨ selected = 1 ∨ selected = 2 ∨
      selected = 3 ∨ selected = 4 ∨ selected = 5 ∨ selected = 6 ∨
      selected = 7 ∨ selected = 8 ∨ selected = 9 ∨ selected = 10 := by
    simp only [SelectionRows.selectionWindow, SelectionRows.candidateCount,
      SelectionRows.outputCount] at selectedLt
    omega
  rcases selectedCases with rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl
  all_goals
    simp at product0 product1 product2 product3 product4 product5 product6 product7 product8 product9 product10
    rw [range11]
    simp [lcEval, product0, product1, product2, product3, product4,
      product5, product6, product7, product8, product9, product10,
      Nat.mod_eq_of_lt (canonical _)]

private theorem acceptBinding_decoded
    {assignment : Nat -> Nat} {position : Nat}
    (one : assignment 0 = 1)
    (positionLt : position < SelectionRows.outputCount)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    lcEval assignment (acceptProductTerms position) = 1 := by
  have holds := satisfies_bindingRows positionLt satisfies
    (SelectionRows.acceptBindingRow position)
    (by simp [SelectionRows.bindingRows])
  have subtractionHolds : RowHolds assignment
      ⟨acceptProductTerms position ++ Program.negateTerms [(0, 1)],
        [(0, 1)], []⟩ := by
    simpa [SelectionRows.bindingRows, SelectionRows.acceptBindingRow,
      SelectionRows.zeroEqualityRow, acceptProductTerms,
      Program.negateTerms, Program.negCoeff] using holds
  have decoded := LinearEquality.sound one (acceptProductTerms position)
    [(0, 1)] (by intro term member; simp at member; rcases member with rfl; decide)
    subtractionHolds
  simpa [lcEval, one, goldilocksP] using decoded

def prefixRightTerms (position : Nat) : List (Nat × Nat) :=
  if position = 0 then [] else [(0, position)]

private theorem prefixRightTerms_canonical
    {position : Nat} (positionLt : position < SelectionRows.outputCount) :
    Program.CanonicalTerms (prefixRightTerms position) := by
  by_cases zero : position = 0
  · intro term member
    simp [prefixRightTerms, zero] at member
  · intro term member
    simp [prefixRightTerms, zero] at member
    rcases member with rfl
    constructor
    · omega
    · simp only [SelectionRows.outputCount] at positionLt
      have bound : 54 < goldilocksP := by decide
      omega

private theorem prefixBinding_decoded
    {assignment : Nat -> Nat} {position : Nat}
    (one : assignment 0 = 1)
    (positionLt : position < SelectionRows.outputCount)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    lcEval assignment (prefixProductTerms position) = position := by
  have holds := satisfies_bindingRows positionLt satisfies
    (SelectionRows.prefixBindingRow position)
    (by simp [SelectionRows.bindingRows])
  have subtractionHolds : RowHolds assignment
      ⟨prefixProductTerms position ++
          Program.negateTerms (prefixRightTerms position),
        [(0, 1)], []⟩ := by
    by_cases zero : position = 0
    · simpa [SelectionRows.bindingRows, SelectionRows.prefixBindingRow,
        SelectionRows.zeroEqualityRow, prefixProductTerms,
        prefixRightTerms, zero] using holds
    · simpa [SelectionRows.bindingRows, SelectionRows.prefixBindingRow,
        SelectionRows.zeroEqualityRow, prefixProductTerms,
        prefixRightTerms, zero, Program.negateTerms,
        Program.negCoeff] using holds
  have decoded := LinearEquality.sound one (prefixProductTerms position)
    (prefixRightTerms position) (prefixRightTerms_canonical positionLt)
    subtractionHolds
  by_cases zero : position = 0
  · subst position
    simpa [prefixRightTerms, lcEval] using decoded
  · have positionLtGoldilocks : position < goldilocksP := by
      simp only [SelectionRows.outputCount] at positionLt
      have bound : 54 < goldilocksP := by decide
      omega
    simpa [prefixRightTerms, zero, lcEval, one,
      Nat.mod_eq_of_lt positionLtGoldilocks] using decoded

structure Refines
    (assignment : Nat -> Nat) (position selected : Nat) : Prop where
  selectedLt : selected < SelectionRows.selectionWindow
  selectorOne : assignment (SelectionRows.selectorCol position selected) = 1
  accepted : assignment (SelectionRows.acceptCol (position + selected)) = 1
  priorCount : assignment (SelectionRows.prefixCol (position + selected)) = position
  output : assignment (SelectionRows.outputCol position) =
    assignment (SelectionRows.symbolCol (position + selected))

theorem refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat} {position selected : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (positionLt : position < SelectionRows.outputCount)
    (selectedLt : selected < SelectionRows.selectionWindow)
    (selectedOne :
      assignment (SelectionRows.selectorCol position selected) = 1)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    Refines assignment position selected := by
  have acceptDecoded := acceptBinding_decoded one positionLt satisfies
  have acceptProducts : ∀ offset, offset < SelectionRows.selectionWindow ->
      assignment (SelectionRows.acceptProductCol position offset) =
        if offset = selected then
          assignment (SelectionRows.acceptCol (position + offset)) else 0 := by
    intro offset offsetLt
    exact acceptProduct_eq_if prime canonical one positionLt selectedLt
      offsetLt selectedOne satisfies
  have acceptSelected := lcEval_selectedProducts canonical
    (SelectionRows.acceptProductCol position)
    (fun offset => SelectionRows.acceptCol (position + offset))
    selected selectedLt acceptProducts
  change lcEval assignment
    ((List.range SelectionRows.selectionWindow).map
      (fun offset => (SelectionRows.acceptProductCol position offset, 1))) = 1
    at acceptDecoded
  rw [acceptSelected] at acceptDecoded

  have prefixDecoded := prefixBinding_decoded one positionLt satisfies
  have prefixProducts : ∀ offset, offset < SelectionRows.selectionWindow ->
      assignment (SelectionRows.prefixProductCol position offset) =
        if offset = selected then
          assignment (SelectionRows.prefixCol (position + offset)) else 0 := by
    intro offset offsetLt
    exact prefixProduct_eq_if prime canonical one positionLt selectedLt
      offsetLt selectedOne satisfies
  have prefixSelected := lcEval_selectedProducts canonical
    (SelectionRows.prefixProductCol position)
    (fun offset => SelectionRows.prefixCol (position + offset))
    selected selectedLt prefixProducts
  change lcEval assignment
    ((List.range SelectionRows.selectionWindow).map
      (fun offset => (SelectionRows.prefixProductCol position offset, 1))) =
        position at prefixDecoded
  rw [prefixSelected] at prefixDecoded

  have symbolHolds := satisfies_bindingRows positionLt satisfies
    (SelectionRows.symbolBindingRow position)
    (by simp [SelectionRows.bindingRows])
  have symbolBuilder : RowHolds assignment
      (Program.builderLinearRow (SelectionRows.outputCol position)
        (symbolProductTerms position)) := by
    simpa [SelectionRows.bindingRows, SelectionRows.symbolBindingRow,
      SelectionRows.zeroEqualityRow, symbolProductTerms,
      Program.builderLinearRow, Program.negateTerms,
      Program.negCoeff] using symbolHolds
  have symbolDecoded := Program.builderLinearRow_sound canonical one
    (SelectionRows.outputCol position) (symbolProductTerms position)
    (productTerms_canonical (SelectionRows.symbolProductCol position))
    symbolBuilder
  have symbolProducts : ∀ offset, offset < SelectionRows.selectionWindow ->
      assignment (SelectionRows.symbolProductCol position offset) =
        if offset = selected then
          assignment (SelectionRows.symbolCol (position + offset)) else 0 := by
    intro offset offsetLt
    exact symbolProduct_eq_if prime canonical one positionLt selectedLt
      offsetLt selectedOne satisfies
  have symbolSelected := lcEval_selectedProducts canonical
    (SelectionRows.symbolProductCol position)
    (fun offset => SelectionRows.symbolCol (position + offset))
    selected selectedLt symbolProducts
  change assignment (SelectionRows.outputCol position) =
    lcEval assignment
      ((List.range SelectionRows.selectionWindow).map
        (fun offset => (SelectionRows.symbolProductCol position offset, 1)))
    at symbolDecoded
  rw [symbolSelected] at symbolDecoded

  exact {
    selectedLt := selectedLt
    selectorOne := selectedOne
    accepted := acceptDecoded
    priorCount := prefixDecoded
    output := symbolDecoded
  }

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.Position
