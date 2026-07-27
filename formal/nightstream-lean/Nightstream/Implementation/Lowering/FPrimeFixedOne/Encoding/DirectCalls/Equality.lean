import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common

/-!
Contract: an exact, activation-aware equality test for two equally sized
Goldilocks coordinate strings.

Each coordinate uses an inverse witness and an equality flag:

```text
(left - right) * inverse = 1 - equal
(left - right) * equal   = 0
```

The equality flags are multiplied through an ordered product chain.  A final
gated row binds that product to the visible Boolean output only when the call
is active.

Owns: the equations above, their exact row/temporary formulas, coordinate
equality soundness, and honest witness construction from a field inverse.

Does not own: semantic codecs, protocol call dispatch, Rust columns, or
generated artifacts.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed

/-- Executable field inversion used only to construct honest equality
witnesses.  Soundness uses `FieldLaws`; this structure is needed solely for
completeness. -/
structure InverseLaw where
  inverse : Field -> Field
  inverse_zero : inverse 0 = 0
  mul_inverse_of_ne_zero :
    ∀ value, value ≠ 0 -> value * inverse value = 1

def coordinateEqualValue (left right : Field) : Field :=
  if left = right then 1 else 0

def coordinateInverseValue
    (inverseLaw : InverseLaw)
    (left right : Field) : Field :=
  if left = right then 0 else inverseLaw.inverse (left - right)

private def inverseRow
    (one left right inverse equal : ColumnId) : Row where
  a := difference left right
  b := singleton inverse 1
  c := oneMinus one equal

private def annihilatorRow
    (left right equal : ColumnId) : Row where
  a := difference left right
  b := singleton equal 1
  c := []

private theorem singleton_eval
    (assignment : ColumnId -> Field)
    (column : ColumnId) :
    (Goldilocks.singleton column 1).eval assignment =
      assignment column := by
  simp only [Goldilocks.singleton, LinearCombination.eval, Fin.one_mul,
    Fin.add_zero]

private theorem difference_eval
    (assignment : ColumnId -> Field)
    (left right : ColumnId) :
    (difference left right).eval assignment =
      assignment left - assignment right := by
  simp only [difference, LinearCombination.eval, Fin.one_mul,
    Fin.add_zero, Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg]

private theorem oneMinus_eval
    (assignment : ColumnId -> Field)
    (one value : ColumnId)
    (constantOne : assignment one = 1) :
    (oneMinus one value).eval assignment =
      1 - assignment value := by
  simp only [oneMinus, LinearCombination.eval, constantOne, Fin.one_mul,
    Fin.add_zero, Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg]

private theorem inverseRow_iff
    (assignment : ColumnId -> Field)
    (one left right inverse equal : ColumnId)
    (constantOne : assignment one = 1) :
    (inverseRow one left right inverse equal).Holds assignment ↔
      (assignment left - assignment right) * assignment inverse =
        1 - assignment equal := by
  unfold inverseRow Row.Holds
  rw [difference_eval, singleton_eval,
    oneMinus_eval assignment one equal constantOne]

private theorem inverseRow_equation
    (assignment : ColumnId -> Field)
    (one left right inverse equal : ColumnId)
    (constantOne : assignment one = 1)
    (holds : (inverseRow one left right inverse equal).Holds assignment) :
    (assignment left - assignment right) * assignment inverse =
      1 - assignment equal := by
  exact
    (inverseRow_iff assignment one left right inverse equal constantOne).mp
      holds

private theorem annihilatorRow_iff
    (assignment : ColumnId -> Field)
    (left right equal : ColumnId) :
    (annihilatorRow left right equal).Holds assignment ↔
      (assignment left - assignment right) * assignment equal = 0 := by
  unfold annihilatorRow Row.Holds
  rw [difference_eval, singleton_eval, LinearCombination.eval_nil]

private theorem annihilatorRow_equation
    (assignment : ColumnId -> Field)
    (left right equal : ColumnId)
    (holds : (annihilatorRow left right equal).Holds assignment) :
    (assignment left - assignment right) * assignment equal = 0 := by
  exact (annihilatorRow_iff assignment left right equal).mp holds

theorem equalityFlag_sound
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (one left right inverse equal : ColumnId)
    (constantOne : assignment one = 1)
    (inverseHolds :
      (inverseRow one left right inverse equal).Holds assignment)
    (annihilatorHolds :
      (annihilatorRow left right equal).Holds assignment) :
    assignment equal =
      coordinateEqualValue (assignment left) (assignment right) := by
  by_cases same : assignment left = assignment right
  · have differenceZero :
        assignment left - assignment right = 0 :=
      Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr same
    have equation :=
      inverseRow_equation assignment one left right inverse equal
        constantOne inverseHolds
    rw [differenceZero, Fin.zero_mul] at equation
    have equalOne : assignment equal = 1 := by
      have subZero : (1 : Field) - assignment equal = 0 := equation.symm
      exact
        (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp subZero).symm
    simp [coordinateEqualValue, same, equalOne]
  · have differenceNe :
        assignment left - assignment right ≠ 0 := by
      intro differenceZero
      exact same
        (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp differenceZero)
    have equation :=
      annihilatorRow_equation assignment left right equal annihilatorHolds
    rcases laws.noZeroDivisors _ _ equation with impossible | equalZero
    · exact False.elim (differenceNe impossible)
    · simp [coordinateEqualValue, same, equalZero]

theorem equalityFlag_complete
    (inverseLaw : InverseLaw)
    (assignment : ColumnId -> Field)
    (one left right inverse equal : ColumnId)
    (constantOne : assignment one = 1)
    (inverseValue :
      assignment inverse =
        coordinateInverseValue inverseLaw
          (assignment left) (assignment right))
    (equalValue :
      assignment equal =
        coordinateEqualValue (assignment left) (assignment right)) :
    (inverseRow one left right inverse equal).Holds assignment ∧
      (annihilatorRow left right equal).Holds assignment := by
  by_cases same : assignment left = assignment right
  · have differenceZero :
        assignment left - assignment right = 0 :=
      Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr same
    have inverseZero : assignment inverse = 0 := by
      rw [inverseValue]
      simp [coordinateInverseValue, same]
    have equalOne : assignment equal = 1 := by
      rw [equalValue]
      simp [coordinateEqualValue, same]
    constructor
    · apply
        (inverseRow_iff assignment one left right inverse equal
          constantOne).mpr
      rw [differenceZero, inverseZero, equalOne, Fin.zero_mul]
      exact Lean.Grind.AddCommGroup.sub_self (1 : Field)
    · apply (annihilatorRow_iff assignment left right equal).mpr
      rw [differenceZero, Fin.zero_mul]
  · have differenceNe :
        assignment left - assignment right ≠ 0 := by
      intro differenceZero
      exact same
        (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp differenceZero)
    have inverseExact :
        assignment inverse =
          inverseLaw.inverse
            (assignment left - assignment right) := by
      rw [inverseValue]
      simp [coordinateInverseValue, same]
    have equalZero : assignment equal = 0 := by
      rw [equalValue]
      simp [coordinateEqualValue, same]
    have inverseProduct :
        (assignment left - assignment right) * assignment inverse = 1 := by
      rw [inverseExact]
      exact inverseLaw.mul_inverse_of_ne_zero _ differenceNe
    constructor
    · apply
        (inverseRow_iff assignment one left right inverse equal
          constantOne).mpr
      rw [equalZero]
      simpa only [Fin.sub_eq_add_neg, Lean.Grind.AddCommGroup.neg_zero,
        Fin.add_zero] using inverseProduct
    · apply (annihilatorRow_iff assignment left right equal).mpr
      rw [equalZero, Fin.mul_zero]

private def coordinateRows :
    ColumnId ->
    List OwnedColumn ->
    List OwnedColumn ->
    List OwnedColumn ->
    List OwnedColumn ->
    List Row
  | one, left :: lefts, right :: rights, inverse :: inverses, equal :: equals =>
      inverseRow one left.id right.id inverse.id equal.id ::
        annihilatorRow left.id right.id equal.id ::
        coordinateRows one lefts rights inverses equals
  | _, _, _, _, _ => []

private theorem coordinateRows_length
    (one : ColumnId)
    (lefts rights inverses equals : List OwnedColumn)
    (rightLength : rights.length = lefts.length)
    (inverseLength : inverses.length = lefts.length)
    (equalLength : equals.length = lefts.length) :
    (coordinateRows one lefts rights inverses equals).length =
      2 * lefts.length := by
  induction lefts generalizing rights inverses equals with
  | nil =>
      have rightsNil : rights = [] :=
        List.eq_nil_of_length_eq_zero rightLength
      have inversesNil : inverses = [] :=
        List.eq_nil_of_length_eq_zero inverseLength
      have equalsNil : equals = [] :=
        List.eq_nil_of_length_eq_zero equalLength
      subst rights
      subst inverses
      subst equals
      rfl
  | cons left lefts inductionHypothesis =>
      cases rights with
      | nil => simp at rightLength
      | cons right rights =>
          cases inverses with
          | nil => simp at inverseLength
          | cons inverse inverses =>
              cases equals with
              | nil => simp at equalLength
              | cons equal equals =>
                  simp only [List.length_cons, Nat.succ.injEq] at rightLength
                  simp only [List.length_cons, Nat.succ.injEq] at inverseLength
                  simp only [List.length_cons, Nat.succ.injEq] at equalLength
                  simp only [coordinateRows, List.length_cons,
                    inductionHypothesis rights inverses equals rightLength
                      inverseLength equalLength]
                  omega

private theorem coordinateRows_supported
    (one : ColumnId)
    (lefts rights inverses equals : List OwnedColumn)
    (row : Row)
    (member : row ∈ coordinateRows one lefts rights inverses equals)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column = one ∨
      column ∈ lefts.map (fun item => item.id) ∨
      column ∈ rights.map (fun item => item.id) ∨
      column ∈ inverses.map (fun item => item.id) ∨
      column ∈ equals.map (fun item => item.id) := by
  induction lefts generalizing rights inverses equals row with
  | nil =>
      simp [coordinateRows] at member
  | cons left lefts inductionHypothesis =>
      cases rights with
      | nil =>
          simp [coordinateRows] at member
      | cons right rights =>
          cases inverses with
          | nil =>
              simp [coordinateRows] at member
          | cons inverse inverses =>
              cases equals with
              | nil =>
                  simp [coordinateRows] at member
              | cons equal equals =>
                  simp only [coordinateRows, List.mem_cons] at member
                  rcases member with inverseRowMember |
                      annihilatorRowMember | tailMember
                  · subst row
                    simp only [List.map_cons, List.mem_cons]
                    simp [Row.columnIds, inverseRow, difference,
                      Goldilocks.singleton, oneMinus] at columnMember
                    rcases columnMember with
                        leftMember | rightMember | inverseMember |
                          oneMember | equalMember
                    · exact Or.inr <| Or.inl <| Or.inl leftMember
                    · exact Or.inr <| Or.inr <| Or.inl <|
                        Or.inl rightMember
                    · exact Or.inr <| Or.inr <| Or.inr <|
                        Or.inl <| Or.inl inverseMember
                    · exact Or.inl oneMember
                    · exact Or.inr <| Or.inr <| Or.inr <|
                        Or.inr <| Or.inl equalMember
                  · subst row
                    simp only [List.map_cons, List.mem_cons]
                    simp [Row.columnIds, annihilatorRow, difference,
                      Goldilocks.singleton] at columnMember
                    rcases columnMember with
                        leftMember | rightMember | equalMember
                    · exact Or.inr <| Or.inl <| Or.inl leftMember
                    · exact Or.inr <| Or.inr <| Or.inl <|
                        Or.inl rightMember
                    · exact Or.inr <| Or.inr <| Or.inr <|
                        Or.inr <| Or.inl equalMember
                  · have tail :=
                      inductionHypothesis rights inverses equals row
                        tailMember columnMember
                    simp only [List.map_cons, List.mem_cons]
                    rcases tail with
                        oneMember | leftMember | rightMember |
                          inverseMember | equalMember
                    · exact Or.inl oneMember
                    · exact Or.inr <| Or.inl <| Or.inr leftMember
                    · exact Or.inr <| Or.inr <| Or.inl <|
                        Or.inr rightMember
                    · exact Or.inr <| Or.inr <| Or.inr <|
                        Or.inl <| Or.inr inverseMember
                    · exact Or.inr <| Or.inr <| Or.inr <|
                        Or.inr <| Or.inr equalMember

def coordinateEqualValues : List Field -> List Field -> List Field
  | left :: lefts, right :: rights =>
      coordinateEqualValue left right ::
        coordinateEqualValues lefts rights
  | _, _ => []

def coordinateInverseValues
    (inverseLaw : InverseLaw) : List Field -> List Field -> List Field
  | left :: lefts, right :: rights =>
      coordinateInverseValue inverseLaw left right ::
        coordinateInverseValues inverseLaw lefts rights
  | _, _ => []

theorem coordinateEqualValues_length
    (lefts rights : List Field)
    (lengthEqual : rights.length = lefts.length) :
    (coordinateEqualValues lefts rights).length = lefts.length := by
  induction lefts generalizing rights with
  | nil =>
      have rightsNil : rights = [] :=
        List.eq_nil_of_length_eq_zero lengthEqual
      subst rights
      rfl
  | cons left lefts inductionHypothesis =>
      cases rights with
      | nil =>
          simp at lengthEqual
      | cons right rights =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          simp only [coordinateEqualValues, List.length_cons,
            inductionHypothesis rights lengthEqual]

theorem coordinateInverseValues_length
    (inverseLaw : InverseLaw)
    (lefts rights : List Field)
    (lengthEqual : rights.length = lefts.length) :
    (coordinateInverseValues inverseLaw lefts rights).length =
      lefts.length := by
  induction lefts generalizing rights with
  | nil =>
      have rightsNil : rights = [] :=
        List.eq_nil_of_length_eq_zero lengthEqual
      subst rights
      rfl
  | cons left lefts inductionHypothesis =>
      cases rights with
      | nil =>
          simp at lengthEqual
      | cons right rights =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          simp only [coordinateInverseValues, List.length_cons,
            inductionHypothesis rights lengthEqual]

private theorem coordinateRows_sound
    (laws : FieldLaws)
    (one : ColumnId)
    (lefts rights inverses equals : List OwnedColumn)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (rightLength : rights.length = lefts.length)
    (inverseLength : inverses.length = lefts.length)
    (equalLength : equals.length = lefts.length)
    (holds :
      RawSatisfies
        (coordinateRows one lefts rights inverses equals) assignment) :
    equals.map (fun column => assignment column.id) =
      coordinateEqualValues
        (lefts.map fun column => assignment column.id)
        (rights.map fun column => assignment column.id) := by
  induction lefts generalizing rights inverses equals with
  | nil =>
      have rightsNil : rights = [] := by
        apply List.eq_nil_of_length_eq_zero
        simpa using rightLength
      have inversesNil : inverses = [] := by
        apply List.eq_nil_of_length_eq_zero
        simpa using inverseLength
      have equalsNil : equals = [] := by
        apply List.eq_nil_of_length_eq_zero
        simpa using equalLength
      subst rights
      subst inverses
      subst equals
      rfl
  | cons left lefts inductionHypothesis =>
      cases rights with
      | nil =>
          simp at rightLength
      | cons right rights =>
          cases inverses with
          | nil =>
              simp at inverseLength
          | cons inverse inverses =>
              cases equals with
              | nil =>
                  simp at equalLength
              | cons equal equals =>
                  simp only [List.length_cons, Nat.succ.injEq] at rightLength
                  simp only [List.length_cons, Nat.succ.injEq] at inverseLength
                  simp only [List.length_cons, Nat.succ.injEq] at equalLength
                  have head :=
                    equalityFlag_sound laws assignment one left.id right.id
                      inverse.id equal.id constantOne holds.1 holds.2.1
                  have tail :=
                    inductionHypothesis rights inverses equals
                      rightLength inverseLength equalLength holds.2.2
                  simpa only [List.map_cons, coordinateEqualValues,
                    List.cons.injEq] using And.intro head tail

private theorem coordinateRows_complete
    (inverseLaw : InverseLaw)
    (one : ColumnId)
    (lefts rights inverses equals : List OwnedColumn)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (rightLength : rights.length = lefts.length)
    (inverseLength : inverses.length = lefts.length)
    (equalLength : equals.length = lefts.length)
    (inverseValues :
      inverses.map (fun column => assignment column.id) =
        coordinateInverseValues inverseLaw
          (lefts.map fun column => assignment column.id)
          (rights.map fun column => assignment column.id))
    (equalValues :
      equals.map (fun column => assignment column.id) =
        coordinateEqualValues
          (lefts.map fun column => assignment column.id)
          (rights.map fun column => assignment column.id)) :
    RawSatisfies
      (coordinateRows one lefts rights inverses equals) assignment := by
  induction lefts generalizing rights inverses equals with
  | nil =>
      have rightsNil : rights = [] := by
        apply List.eq_nil_of_length_eq_zero
        simpa using rightLength
      have inversesNil : inverses = [] := by
        apply List.eq_nil_of_length_eq_zero
        simpa using inverseLength
      have equalsNil : equals = [] := by
        apply List.eq_nil_of_length_eq_zero
        simpa using equalLength
      subst rights
      subst inverses
      subst equals
      trivial
  | cons left lefts inductionHypothesis =>
      cases rights with
      | nil =>
          simp at rightLength
      | cons right rights =>
          cases inverses with
          | nil =>
              simp at inverseLength
          | cons inverse inverses =>
              cases equals with
              | nil =>
                  simp at equalLength
              | cons equal equals =>
                  simp only [List.length_cons, Nat.succ.injEq] at rightLength
                  simp only [List.length_cons, Nat.succ.injEq] at inverseLength
                  simp only [List.length_cons, Nat.succ.injEq] at equalLength
                  have inverseSplit :
                      assignment inverse.id =
                          coordinateInverseValue inverseLaw
                            (assignment left.id) (assignment right.id) ∧
                        inverses.map (fun column => assignment column.id) =
                          coordinateInverseValues inverseLaw
                            (lefts.map fun column => assignment column.id)
                            (rights.map fun column => assignment column.id) := by
                    simpa only [List.map_cons, coordinateInverseValues,
                      List.cons.injEq] using inverseValues
                  have equalSplit :
                      assignment equal.id =
                          coordinateEqualValue
                            (assignment left.id) (assignment right.id) ∧
                        equals.map (fun column => assignment column.id) =
                          coordinateEqualValues
                            (lefts.map fun column => assignment column.id)
                            (rights.map fun column => assignment column.id) := by
                    simpa only [List.map_cons, coordinateEqualValues,
                      List.cons.injEq] using equalValues
                  have head :=
                    equalityFlag_complete inverseLaw assignment one left.id
                      right.id inverse.id equal.id constantOne
                      inverseSplit.1 equalSplit.1
                  exact ⟨head.1, head.2,
                    inductionHypothesis rights inverses equals
                      rightLength inverseLength equalLength
                      inverseSplit.2 equalSplit.2⟩

/-! ## Exact conjunction of coordinate flags -/

def fieldProduct : List Field -> Field
  | [] => 1
  | value :: values => value * fieldProduct values

private def productTailRows :
    OwnedColumn ->
    List OwnedColumn ->
    List OwnedColumn ->
    List Row
  | _, [], _ => []
  | current, flag :: flags, product :: products =>
      (CanonicalRow.product product.id current.id flag.id).row ::
        productTailRows product flags products
  | _, _ :: _, [] => []

private def productRows :
    List OwnedColumn -> List OwnedColumn -> List Row
  | [], _ => []
  | first :: flags, products =>
      productTailRows first flags products

private def productTailResult :
    OwnedColumn ->
    List OwnedColumn ->
    List OwnedColumn ->
    OwnedColumn
  | current, [], _ => current
  | current, _ :: _, [] => current
  | _, _ :: flags, product :: products =>
      productTailResult product flags products

private def productResult
    (one : ColumnId) :
    List OwnedColumn -> List OwnedColumn -> ColumnId
  | [], _ => one
  | first :: flags, products =>
      (productTailResult first flags products).id

private def productTailValues :
    Field -> List Field -> List Field
  | _, [] => []
  | current, flag :: flags =>
      let product := current * flag
      product :: productTailValues product flags

def productValues : List Field -> List Field
  | [] => []
  | first :: flags => productTailValues first flags

private theorem productTailRows_length
    (current : OwnedColumn)
    (flags products : List OwnedColumn)
    (lengthEqual : products.length = flags.length) :
    (productTailRows current flags products).length = flags.length := by
  induction flags generalizing current products with
  | nil =>
      have productsNil : products = [] :=
        List.eq_nil_of_length_eq_zero lengthEqual
      subst products
      rfl
  | cons flag flags inductionHypothesis =>
      cases products with
      | nil =>
          simp at lengthEqual
      | cons product products =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          simp only [productTailRows, List.length_cons,
            inductionHypothesis product products lengthEqual]

private theorem productRows_length
    (flags products : List OwnedColumn)
    (lengthEqual : products.length = flags.length.pred) :
    (productRows flags products).length = flags.length.pred := by
  cases flags with
  | nil =>
      have productsNil : products = [] :=
        List.eq_nil_of_length_eq_zero lengthEqual
      subst products
      rfl
  | cons first flags =>
      simp only [List.length_cons, Nat.pred_succ] at lengthEqual ⊢
      exact productTailRows_length first flags products lengthEqual

private theorem productTailRows_supported
    (current : OwnedColumn)
    (flags products : List OwnedColumn)
    (row : Row)
    (member : row ∈ productTailRows current flags products)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column = current.id ∨
      column ∈ flags.map (fun item => item.id) ∨
      column ∈ products.map (fun item => item.id) := by
  induction flags generalizing current products row with
  | nil =>
      simp [productTailRows] at member
  | cons flag flags inductionHypothesis =>
      cases products with
      | nil =>
          simp [productTailRows] at member
      | cons product products =>
          simp only [productTailRows, List.mem_cons] at member
          rcases member with headMember | tailMember
          · subst row
            simp only [List.map_cons, List.mem_cons]
            simp [Row.columnIds, CanonicalRow.row,
              Goldilocks.singleton] at columnMember
            rcases columnMember with
                currentMember | flagMember | productMember
            · exact Or.inl currentMember
            · exact Or.inr <| Or.inl <| Or.inl flagMember
            · exact Or.inr <| Or.inr <| Or.inl productMember
          · have tail :=
              inductionHypothesis product products row tailMember
                columnMember
            simp only [List.map_cons, List.mem_cons]
            rcases tail with
                productMember | flagMember | productsMember
            · exact Or.inr <| Or.inr <| Or.inl productMember
            · exact Or.inr <| Or.inl <| Or.inr flagMember
            · exact Or.inr <| Or.inr <| Or.inr productsMember

private theorem productRows_supported
    (flags products : List OwnedColumn)
    (row : Row)
    (member : row ∈ productRows flags products)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ flags.map (fun item => item.id) ∨
      column ∈ products.map (fun item => item.id) := by
  cases flags with
  | nil =>
      simp [productRows] at member
  | cons first flags =>
      have supported :=
        productTailRows_supported first flags products row member
          column columnMember
      simp only [List.map_cons, List.mem_cons]
      rcases supported with firstMember | flagMember | productMember
      · exact Or.inl <| Or.inl firstMember
      · exact Or.inl <| Or.inr flagMember
      · exact Or.inr productMember

private theorem productTailResult_supported
    (current : OwnedColumn)
    (flags products : List OwnedColumn) :
    (productTailResult current flags products).id = current.id ∨
      (productTailResult current flags products).id ∈
        products.map (fun item => item.id) := by
  induction flags generalizing current products with
  | nil =>
      exact Or.inl rfl
  | cons flag flags inductionHypothesis =>
      cases products with
      | nil =>
          exact Or.inl rfl
      | cons product products =>
          have tail := inductionHypothesis product products
          rcases tail with productExact | tailMember
          · exact Or.inr <| by
              simp only [productTailResult, List.map_cons, List.mem_cons]
              exact Or.inl productExact
          · exact Or.inr <| by
              simp only [productTailResult, List.map_cons, List.mem_cons]
              exact Or.inr tailMember

private theorem productResult_supported
    (one : ColumnId)
    (flags products : List OwnedColumn) :
    productResult one flags products = one ∨
      productResult one flags products ∈
        flags.map (fun item => item.id) ∨
      productResult one flags products ∈
        products.map (fun item => item.id) := by
  cases flags with
  | nil =>
      exact Or.inl rfl
  | cons first flags =>
      have supported :=
        productTailResult_supported first flags products
      rcases supported with firstExact | productMember
      · exact Or.inr <| Or.inl <| by
          simp only [productResult, List.map_cons, List.mem_cons]
          exact Or.inl firstExact
      · exact Or.inr <| Or.inr <| by
          simpa only [productResult] using productMember

private theorem productTailRows_sound
    (assignment : ColumnId -> Field)
    (current : OwnedColumn)
    (flags products : List OwnedColumn)
    (lengthEqual : products.length = flags.length)
    (holds :
      RawSatisfies (productTailRows current flags products) assignment) :
    assignment (productTailResult current flags products).id =
      assignment current.id *
        fieldProduct (flags.map fun flag => assignment flag.id) := by
  induction flags generalizing current products with
  | nil =>
      have productsNil : products = [] :=
        List.eq_nil_of_length_eq_zero lengthEqual
      subst products
      simp [productTailResult, fieldProduct, Fin.mul_one]
  | cons flag flags inductionHypothesis =>
      cases products with
      | nil =>
          simp at lengthEqual
      | cons product products =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          have head :
              assignment current.id * assignment flag.id =
                assignment product.id :=
            (CanonicalRow.product_iff assignment product.id current.id
              flag.id).mp holds.1
          have tail :=
            inductionHypothesis product products lengthEqual holds.2
          simp only [productTailResult, List.map_cons, fieldProduct]
          calc
            assignment
                (productTailResult product flags products).id =
              assignment product.id *
                fieldProduct (flags.map fun item => assignment item.id) :=
              tail
            _ = (assignment current.id * assignment flag.id) *
                fieldProduct (flags.map fun item => assignment item.id) := by
              rw [head]
            _ = assignment current.id *
                (assignment flag.id *
                  fieldProduct
                    (flags.map fun item => assignment item.id)) :=
              Lean.Grind.Fin.mul_assoc _ _ _

private theorem productRows_sound
    (one : ColumnId)
    (flags products : List OwnedColumn)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (lengthEqual : products.length = flags.length.pred)
    (holds : RawSatisfies (productRows flags products) assignment) :
    assignment (productResult one flags products) =
      fieldProduct (flags.map fun flag => assignment flag.id) := by
  cases flags with
  | nil =>
      simp [productResult, fieldProduct, constantOne]
  | cons first flags =>
      simp only [List.length_cons, Nat.pred_succ] at lengthEqual
      exact
        productTailRows_sound assignment first flags products lengthEqual
          holds

private theorem productTailValues_length
    (current : Field)
    (flags : List Field) :
    (productTailValues current flags).length = flags.length := by
  induction flags generalizing current with
  | nil => rfl
  | cons flag flags inductionHypothesis =>
      simp only [productTailValues, List.length_cons,
        inductionHypothesis (current * flag)]

theorem productValues_length (values : List Field) :
    (productValues values).length = values.length.pred := by
  cases values with
  | nil => rfl
  | cons first values =>
      simp only [productValues, List.length_cons, Nat.pred_succ]
      exact productTailValues_length first values

private theorem productTailRows_complete
    (assignment : ColumnId -> Field)
    (current : OwnedColumn)
    (flags products : List OwnedColumn)
    (lengthEqual : products.length = flags.length)
    (values :
      products.map (fun product => assignment product.id) =
        productTailValues (assignment current.id)
          (flags.map fun flag => assignment flag.id)) :
    RawSatisfies (productTailRows current flags products) assignment := by
  induction flags generalizing current products with
  | nil =>
      have productsNil : products = [] :=
        List.eq_nil_of_length_eq_zero lengthEqual
      subst products
      simp [productTailRows]
  | cons flag flags inductionHypothesis =>
      cases products with
      | nil =>
          simp at lengthEqual
      | cons product products =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          have rawSplit :
              assignment product.id =
                    assignment current.id * assignment flag.id ∧
                products.map (fun item => assignment item.id) =
                  productTailValues
                    (assignment current.id * assignment flag.id)
                    (flags.map fun item => assignment item.id) := by
            simpa only [List.map_cons, productTailValues, List.cons.injEq]
              using values
          have tail :
              products.map (fun item => assignment item.id) =
                productTailValues (assignment product.id)
                  (flags.map fun item => assignment item.id) := by
            rw [rawSplit.1]
            exact rawSplit.2
          exact ⟨
            (CanonicalRow.product_iff assignment product.id current.id
              flag.id).mpr rawSplit.1.symm,
            inductionHypothesis product products lengthEqual tail⟩

private theorem productRows_complete
    (flags products : List OwnedColumn)
    (assignment : ColumnId -> Field)
    (lengthEqual : products.length = flags.length.pred)
    (values :
      products.map (fun product => assignment product.id) =
        productValues (flags.map fun flag => assignment flag.id)) :
    RawSatisfies (productRows flags products) assignment := by
  cases flags with
  | nil =>
      have productsNil : products = [] :=
        List.eq_nil_of_length_eq_zero (by simpa using lengthEqual)
      subst products
      simp [productRows]
  | cons first flags =>
      simp only [List.length_cons, Nat.pred_succ] at lengthEqual
      exact productTailRows_complete assignment first flags products
        lengthEqual values

private theorem fieldProduct_coordinateEqualValues
    (lefts rights : List Field)
    (lengthEqual : lefts.length = rights.length) :
    fieldProduct (coordinateEqualValues lefts rights) =
      (if lefts = rights then 1 else 0) := by
  induction lefts generalizing rights with
  | nil =>
      have rightsNil : rights = [] :=
        List.eq_nil_of_length_eq_zero lengthEqual.symm
      subst rights
      rfl
  | cons left lefts inductionHypothesis =>
      cases rights with
      | nil =>
          simp at lengthEqual
      | cons right rights =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          by_cases headEqual : left = right
          · subst right
            simp only [coordinateEqualValues, fieldProduct]
            rw [show coordinateEqualValue left left = 1 by
              simp [coordinateEqualValue], Fin.one_mul]
            simp only [List.cons.injEq, true_and]
            exact inductionHypothesis rights lengthEqual
          · have listsDifferent : left :: lefts ≠ right :: rights := by
              intro equal
              exact headEqual (List.cons.inj equal).1
            simp only [coordinateEqualValues, fieldProduct]
            rw [show coordinateEqualValue left right = 0 by
              simp [coordinateEqualValue, headEqual], Fin.zero_mul]
            simp [listsDifferent]

private def finalRow
    (active result output : ColumnId) : Row where
  a := singleton active 1
  b := difference result output
  c := []

private theorem finalRow_active_iff
    (assignment : ColumnId -> Field)
    (active result output : ColumnId)
    (activeOne : assignment active = 1) :
    (finalRow active result output).Holds assignment ↔
      assignment output = assignment result := by
  unfold finalRow Row.Holds
  rw [singleton_eval, difference_eval, LinearCombination.eval_nil,
    activeOne, Fin.one_mul]
  constructor
  · intro differenceZero
    exact (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp differenceZero).symm
  · intro equal
    exact
      Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr equal.symm

private theorem finalRow_inactive
    (assignment : ColumnId -> Field)
    (active result output : ColumnId)
    (activeZero : assignment active = 0) :
    (finalRow active result output).Holds assignment := by
  unfold finalRow Row.Holds
  rw [singleton_eval, difference_eval, LinearCombination.eval_nil,
    activeZero, Fin.zero_mul]

/-! ## Complete vector-equality occurrence -/

/-- One concrete equality occurrence.  The three temporary lists are,
respectively, coordinate inverses, coordinate equality flags, and the
`width - 1` product chain. -/
structure EqualityRecipe where
  owner : PhysicalOwner
  one : ColumnId
  active : ColumnId
  left : List OwnedColumn
  right : List OwnedColumn
  output : OwnedColumn
  inverses : List OwnedColumn
  equals : List OwnedColumn
  products : List OwnedColumn
  rightLength : right.length = left.length
  inverseLength : inverses.length = left.length
  equalLength : equals.length = left.length
  productLength : products.length = left.length.pred

namespace EqualityRecipe

def rawRows (recipe : EqualityRecipe) : List Row :=
  coordinateRows recipe.one recipe.left recipe.right
      recipe.inverses recipe.equals ++
    productRows recipe.equals recipe.products ++
    [finalRow recipe.active
      (productResult recipe.one recipe.equals recipe.products)
      recipe.output.id]

def rows (recipe : EqualityRecipe) : List OwnedRow :=
  ownRows recipe.owner recipe.rawRows

theorem raw_row_count (recipe : EqualityRecipe) :
    recipe.rawRows.length =
      2 * recipe.left.length + recipe.left.length.pred + 1 := by
  unfold rawRows
  rw [List.length_append, List.length_append,
    coordinateRows_length recipe.one recipe.left recipe.right
      recipe.inverses recipe.equals recipe.rightLength
      recipe.inverseLength recipe.equalLength,
    productRows_length recipe.equals recipe.products (by
      rw [recipe.equalLength, recipe.productLength])]
  rw [recipe.equalLength]
  simp only [List.length_singleton]

theorem row_count (recipe : EqualityRecipe) :
    recipe.rows.length =
      2 * recipe.left.length + recipe.left.length.pred + 1 := by
  rw [rows, ownRows_length, recipe.raw_row_count]

theorem temporary_count (recipe : EqualityRecipe) :
    (recipe.inverses ++ recipe.equals ++ recipe.products).length =
      2 * recipe.left.length + recipe.left.length.pred := by
  simp only [List.length_append, recipe.inverseLength,
    recipe.equalLength, recipe.productLength]
  omega

theorem rows_owned
    (recipe : EqualityRecipe)
    (row : OwnedRow)
    (member : row ∈ recipe.rows) :
    row.id.owner = recipe.owner :=
  ownRows_owner recipe.owner recipe.rawRows row member

theorem row_ids_nodup (recipe : EqualityRecipe) :
    ((recipe.rows.map fun row => row.id)).Nodup :=
  ownRows_ids_nodup recipe.owner recipe.rawRows

theorem rawRows_supported
    (recipe : EqualityRecipe)
    (row : Row)
    (member : row ∈ recipe.rawRows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column = recipe.one ∨
      column = recipe.active ∨
      column ∈ recipe.left.map (fun item => item.id) ∨
      column ∈ recipe.right.map (fun item => item.id) ∨
      column = recipe.output.id ∨
      column ∈ recipe.inverses.map (fun item => item.id) ∨
      column ∈ recipe.equals.map (fun item => item.id) ∨
      column ∈ recipe.products.map (fun item => item.id) := by
  have reassociated :
      row ∈
        coordinateRows recipe.one recipe.left recipe.right
            recipe.inverses recipe.equals ++
          (productRows recipe.equals recipe.products ++
            [finalRow recipe.active
              (productResult recipe.one recipe.equals recipe.products)
              recipe.output.id]) := by
    simpa only [rawRows, List.append_assoc] using member
  rw [List.mem_append] at reassociated
  rcases reassociated with coordinateMember | tailMember
  · have supported :=
      coordinateRows_supported recipe.one recipe.left recipe.right
        recipe.inverses recipe.equals row coordinateMember
        column columnMember
    rcases supported with
        oneMember | leftMember | rightMember |
          inverseMember | equalMember
    · exact Or.inl oneMember
    · exact Or.inr <| Or.inr <| Or.inl leftMember
    · exact Or.inr <| Or.inr <| Or.inr <| Or.inl rightMember
    · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <|
        Or.inr <| Or.inl inverseMember
    · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <|
        Or.inr <| Or.inr <| Or.inl equalMember
  · rw [List.mem_append] at tailMember
    rcases tailMember with productMember | finalMember
    · have supported :=
        productRows_supported recipe.equals recipe.products row
          productMember column columnMember
      rcases supported with equalMember | productMember
      · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <|
          Or.inr <| Or.inr <| Or.inl equalMember
      · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <|
          Or.inr <| Or.inr <| Or.inr productMember
    · simp only [List.mem_singleton] at finalMember
      rw [finalMember] at columnMember
      simp [Row.columnIds, finalRow, Goldilocks.singleton,
        difference] at columnMember
      rcases columnMember with activeMember | resultMember | outputMember
      · exact Or.inr <| Or.inl activeMember
      · rcases
            productResult_supported recipe.one recipe.equals
              recipe.products with
          oneMember | equalMember | productMember
        · exact Or.inl (resultMember.trans oneMember)
        · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <|
            Or.inr <| Or.inr <| Or.inl <| by
              simpa [resultMember] using equalMember
        · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <|
            Or.inr <| Or.inr <| Or.inr <| by
              simpa [resultMember] using productMember
      · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <|
          Or.inl outputMember

theorem rows_supported
    (recipe : EqualityRecipe)
    (row : OwnedRow)
    (member : row ∈ recipe.rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column = recipe.one ∨
      column = recipe.active ∨
      column ∈ recipe.left.map (fun item => item.id) ∨
      column ∈ recipe.right.map (fun item => item.id) ∨
      column = recipe.output.id ∨
      column ∈ recipe.inverses.map (fun item => item.id) ∨
      column ∈ recipe.equals.map (fun item => item.id) ∨
      column ∈ recipe.products.map (fun item => item.id) := by
  apply recipe.rawRows_supported row.row
    (ownRows_row_mem recipe.owner recipe.rawRows row member)
    column
  exact columnMember

theorem active_sound
    (laws : FieldLaws)
    (recipe : EqualityRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (activeOne : assignment recipe.active = 1)
    (holds : Satisfies recipe.rows assignment) :
    assignment recipe.output.id =
      (if recipe.left.map (fun column => assignment column.id) =
          recipe.right.map (fun column => assignment column.id)
        then 1 else 0) := by
  have rawHolds :
      RawSatisfies recipe.rawRows assignment :=
    (satisfies_ownRows_iff recipe.owner recipe.rawRows assignment).mp holds
  have reassociated :
      RawSatisfies
        (coordinateRows recipe.one recipe.left recipe.right
            recipe.inverses recipe.equals ++
          (productRows recipe.equals recipe.products ++
            [finalRow recipe.active
              (productResult recipe.one recipe.equals recipe.products)
              recipe.output.id]))
        assignment := by
    simpa only [rawRows, List.append_assoc] using rawHolds
  have split :
      RawSatisfies
          (coordinateRows recipe.one recipe.left recipe.right
            recipe.inverses recipe.equals) assignment ∧
        RawSatisfies
          (productRows recipe.equals recipe.products) assignment ∧
        (finalRow recipe.active
          (productResult recipe.one recipe.equals recipe.products)
          recipe.output.id).Holds assignment := by
    have outer :=
      (rawSatisfies_append_iff
        (coordinateRows recipe.one recipe.left recipe.right
          recipe.inverses recipe.equals)
        (productRows recipe.equals recipe.products ++
          [finalRow recipe.active
            (productResult recipe.one recipe.equals recipe.products)
            recipe.output.id])
        assignment).mp reassociated
    have inner :=
      (rawSatisfies_append_iff
        (productRows recipe.equals recipe.products)
        [finalRow recipe.active
          (productResult recipe.one recipe.equals recipe.products)
          recipe.output.id]
        assignment).mp outer.2
    exact ⟨outer.1, inner.1, inner.2.1⟩
  have equalCoordinates :=
    coordinateRows_sound laws recipe.one recipe.left recipe.right
      recipe.inverses recipe.equals assignment constantOne
      recipe.rightLength recipe.inverseLength recipe.equalLength split.1
  have product :
      assignment
          (productResult recipe.one recipe.equals recipe.products) =
        fieldProduct
          (recipe.equals.map fun column => assignment column.id) :=
    productRows_sound recipe.one recipe.equals recipe.products assignment
      constantOne (by rw [recipe.productLength, recipe.equalLength])
      split.2.1
  have output :
      assignment recipe.output.id =
        assignment
          (productResult recipe.one recipe.equals recipe.products) :=
    (finalRow_active_iff assignment recipe.active
      (productResult recipe.one recipe.equals recipe.products)
      recipe.output.id activeOne).mp split.2.2
  rw [output, product, equalCoordinates]
  exact
    fieldProduct_coordinateEqualValues
      (recipe.left.map fun column => assignment column.id)
      (recipe.right.map fun column => assignment column.id)
      (by rw [List.length_map, List.length_map, recipe.rightLength])

theorem active_complete
    (inverseLaw : InverseLaw)
    (recipe : EqualityRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (activeOne : assignment recipe.active = 1)
    (inverseValues :
      recipe.inverses.map (fun column => assignment column.id) =
        coordinateInverseValues inverseLaw
          (recipe.left.map fun column => assignment column.id)
          (recipe.right.map fun column => assignment column.id))
    (equalValues :
      recipe.equals.map (fun column => assignment column.id) =
        coordinateEqualValues
          (recipe.left.map fun column => assignment column.id)
          (recipe.right.map fun column => assignment column.id))
    (productCoordinates :
      recipe.products.map (fun column => assignment column.id) =
        productValues
          (recipe.equals.map fun column => assignment column.id))
    (outputValue :
      assignment recipe.output.id =
        (if recipe.left.map (fun column => assignment column.id) =
            recipe.right.map (fun column => assignment column.id)
          then 1 else 0)) :
    Satisfies recipe.rows assignment := by
  apply
    (satisfies_ownRows_iff recipe.owner recipe.rawRows assignment).mpr
  rw [rawRows, List.append_assoc]
  apply
    (rawSatisfies_append_iff
      (coordinateRows recipe.one recipe.left recipe.right
        recipe.inverses recipe.equals)
      (productRows recipe.equals recipe.products ++
        [finalRow recipe.active
          (productResult recipe.one recipe.equals recipe.products)
          recipe.output.id])
      assignment).mpr
  constructor
  · exact coordinateRows_complete inverseLaw recipe.one recipe.left
      recipe.right recipe.inverses recipe.equals assignment constantOne
      recipe.rightLength recipe.inverseLength recipe.equalLength
      inverseValues equalValues
  · apply
      (rawSatisfies_append_iff
        (productRows recipe.equals recipe.products)
        [finalRow recipe.active
          (productResult recipe.one recipe.equals recipe.products)
          recipe.output.id]
        assignment).mpr
    constructor
    · exact productRows_complete recipe.equals recipe.products assignment
        (by rw [recipe.productLength, recipe.equalLength])
        productCoordinates
    · constructor
      · apply
          (finalRow_active_iff assignment recipe.active
            (productResult recipe.one recipe.equals recipe.products)
            recipe.output.id activeOne).mpr
        rw [productRows_sound recipe.one recipe.equals recipe.products
          assignment constantOne
          (by rw [recipe.productLength, recipe.equalLength])
          (productRows_complete recipe.equals recipe.products assignment
            (by rw [recipe.productLength, recipe.equalLength])
            productCoordinates)]
        rw [equalValues,
          fieldProduct_coordinateEqualValues
            (recipe.left.map fun column => assignment column.id)
            (recipe.right.map fun column => assignment column.id)
            (by rw [List.length_map, List.length_map,
              recipe.rightLength])]
        exact outputValue
      · trivial

theorem inactive_complete
    (inverseLaw : InverseLaw)
    (recipe : EqualityRecipe)
    (assignment : ColumnId -> Field)
    (constantOne : assignment recipe.one = 1)
    (activeZero : assignment recipe.active = 0)
    (inverseValues :
      recipe.inverses.map (fun column => assignment column.id) =
        coordinateInverseValues inverseLaw
          (recipe.left.map fun column => assignment column.id)
          (recipe.right.map fun column => assignment column.id))
    (equalValues :
      recipe.equals.map (fun column => assignment column.id) =
        coordinateEqualValues
          (recipe.left.map fun column => assignment column.id)
          (recipe.right.map fun column => assignment column.id))
    (productCoordinates :
      recipe.products.map (fun column => assignment column.id) =
        productValues
          (recipe.equals.map fun column => assignment column.id)) :
    Satisfies recipe.rows assignment := by
  apply
    (satisfies_ownRows_iff recipe.owner recipe.rawRows assignment).mpr
  rw [rawRows, List.append_assoc]
  apply
    (rawSatisfies_append_iff
      (coordinateRows recipe.one recipe.left recipe.right
        recipe.inverses recipe.equals)
      (productRows recipe.equals recipe.products ++
        [finalRow recipe.active
          (productResult recipe.one recipe.equals recipe.products)
          recipe.output.id])
      assignment).mpr
  constructor
  · exact coordinateRows_complete inverseLaw recipe.one recipe.left
      recipe.right recipe.inverses recipe.equals assignment constantOne
      recipe.rightLength recipe.inverseLength recipe.equalLength
      inverseValues equalValues
  · apply
      (rawSatisfies_append_iff
        (productRows recipe.equals recipe.products)
        [finalRow recipe.active
          (productResult recipe.one recipe.equals recipe.products)
          recipe.output.id]
        assignment).mpr
    exact ⟨
      productRows_complete recipe.equals recipe.products assignment
        (by rw [recipe.productLength, recipe.equalLength])
        productCoordinates,
      ⟨finalRow_inactive assignment recipe.active
        (productResult recipe.one recipe.equals recipe.products)
        recipe.output.id activeZero, trivial⟩⟩

def witnessValues
    (inverseLaw : InverseLaw)
    (recipe : EqualityRecipe)
    (assignment : ColumnId -> Field) : List Field :=
  let leftValues :=
    recipe.left.map (fun column => assignment column.id)
  let rightValues :=
    recipe.right.map (fun column => assignment column.id)
  let equalValues := coordinateEqualValues leftValues rightValues
  coordinateInverseValues inverseLaw leftValues rightValues ++
    (equalValues ++ productValues equalValues)

def completion
    (inverseLaw : InverseLaw)
    (recipe : EqualityRecipe)
    (temporaryIds : List ColumnId)
    (assignment : ColumnId -> Field) : ColumnId -> Field :=
  writeColumns assignment temporaryIds
    (witnessValues inverseLaw recipe assignment)

theorem completion_changesOnly
    (inverseLaw : InverseLaw)
    (recipe : EqualityRecipe)
    (temporaryIds : List ColumnId)
    (assignment : ColumnId -> Field) :
    ChangesOnly temporaryIds assignment
      (completion inverseLaw recipe temporaryIds assignment) :=
  writeColumns_changesOnly assignment temporaryIds
    (witnessValues inverseLaw recipe assignment)

theorem completion_values
    (inverseLaw : InverseLaw)
    (recipe : EqualityRecipe)
    (temporaryIds : List ColumnId)
    (assignment : ColumnId -> Field)
    (idsExact :
      temporaryIds =
        recipe.inverses.map (fun column => column.id) ++
          (recipe.equals.map (fun column => column.id) ++
            recipe.products.map (fun column => column.id)))
    (idsNodup : temporaryIds.Nodup) :
    recipe.inverses.map
          (fun column =>
            completion inverseLaw recipe temporaryIds assignment
              column.id) =
        coordinateInverseValues inverseLaw
          (recipe.left.map (fun column => assignment column.id))
          (recipe.right.map (fun column => assignment column.id)) ∧
      recipe.equals.map
          (fun column =>
            completion inverseLaw recipe temporaryIds assignment
              column.id) =
        coordinateEqualValues
          (recipe.left.map (fun column => assignment column.id))
          (recipe.right.map (fun column => assignment column.id)) ∧
      recipe.products.map
          (fun column =>
            completion inverseLaw recipe temporaryIds assignment
              column.id) =
        productValues
          (coordinateEqualValues
            (recipe.left.map (fun column => assignment column.id))
            (recipe.right.map (fun column => assignment column.id))) := by
  let leftValues :=
    recipe.left.map (fun column => assignment column.id)
  let rightValues :=
    recipe.right.map (fun column => assignment column.id)
  let inverseValues :=
    coordinateInverseValues inverseLaw leftValues rightValues
  let equalValues := coordinateEqualValues leftValues rightValues
  let productCoordinates := productValues equalValues
  have rightLength : rightValues.length = leftValues.length := by
    simp only [rightValues, leftValues, List.length_map,
      recipe.rightLength]
  have inverseLength :
      inverseValues.length = recipe.inverses.length := by
    rw [coordinateInverseValues_length inverseLaw leftValues rightValues
      rightLength, recipe.inverseLength]
    simp [leftValues]
  have equalLength :
      equalValues.length = recipe.equals.length := by
    rw [coordinateEqualValues_length leftValues rightValues rightLength,
      recipe.equalLength]
    simp [leftValues]
  have productLength :
      productCoordinates.length = recipe.products.length := by
    rw [productValues_length, recipe.productLength]
    simp only [equalValues,
      coordinateEqualValues_length leftValues rightValues rightLength]
    simp [leftValues]
  have valuesLength :
      temporaryIds.length =
        (inverseValues ++ (equalValues ++ productCoordinates)).length := by
    rw [idsExact]
    simp only [List.length_append, List.length_map]
    omega
  have recovered :=
    writeColumns_map_eq assignment temporaryIds
      (inverseValues ++ (equalValues ++ productCoordinates))
      valuesLength idsNodup
  have splitExact :
      recipe.inverses.map
            (fun column =>
              completion inverseLaw recipe temporaryIds assignment
                column.id) ++
          (recipe.equals.map
              (fun column =>
                completion inverseLaw recipe temporaryIds assignment
                  column.id) ++
            recipe.products.map
              (fun column =>
                completion inverseLaw recipe temporaryIds assignment
                  column.id)) =
        inverseValues ++ (equalValues ++ productCoordinates) := by
    rw [idsExact]
    have recoveredExact := recovered
    rw [idsExact, List.map_append, List.map_append] at recoveredExact
    simpa [completion, witnessValues, leftValues, rightValues,
      inverseValues, equalValues, productCoordinates, List.map_map,
      Function.comp_def] using recoveredExact
  have inverseSplit :=
    List.append_inj splitExact (by
      simpa only [List.length_map] using inverseLength.symm)
  have equalSplit :=
    List.append_inj inverseSplit.2 (by
      simpa only [List.length_map] using equalLength.symm)
  exact ⟨
    by simpa [inverseValues, leftValues, rightValues] using inverseSplit.1,
    by simpa [equalValues, leftValues, rightValues] using equalSplit.1,
    by simpa [productCoordinates, equalValues, leftValues, rightValues]
      using equalSplit.2⟩

end EqualityRecipe

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
