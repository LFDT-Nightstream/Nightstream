import Nightstream.Implementation.NebulaV2.ProductPiDecLinearCombination
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: exact verifier-computed public-input split rows for production PiDEC.

For each of the 540 parent public coordinates, this block owns one common sign
bit, fourteen magnitude bits, fourteen sign products, fourteen signed child
coordinates, and one radix recomposition check. The common sign is necessary:
centered-unit digits plus recomposition alone allow mixed-sign alternate
representations.

The soundness theorem derives the exact deterministic `splitScalar` child for
every coordinate. It does not assume the child values, the parent bound, or a
PiDEC acceptance result.

Assurance tier: generated-row semantic model.

Does not own absolute placement in the recursive manifest, NIFS output-carrier
serialization, Rust refinement, or cryptographic soundness.

Emits constraints: 23,760 R1CS rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.NebulaV2.ProductPiDecPublicSplitRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.PiDecStrictSound
open Nightstream.Implementation.NebulaV2.ProductPiDecLinearCombination
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev PublicCoordinate := Fin 540
abbrev ChildIndex := PiDECAlgebra.Radix.ChildIndex

/-- Physical columns for the complete verifier-computed public split. -/
structure Layout where
  parentColumn : PublicCoordinate -> Nat
  childColumn : ChildIndex -> PublicCoordinate -> Nat
  signColumn : PublicCoordinate -> Nat
  magnitudeBitColumn : PublicCoordinate -> ChildIndex -> Nat
  signProductColumn : PublicCoordinate -> ChildIndex -> Nat

def signRow (layout : Layout) (coordinate : PublicCoordinate) : Row :=
  bitRow (layout.signColumn coordinate)

def magnitudeBitRow (layout : Layout) (coordinate : PublicCoordinate)
    (child : ChildIndex) : Row :=
  bitRow (layout.magnitudeBitColumn coordinate child)

def signProductDefinition (layout : Layout) (coordinate : PublicCoordinate)
    (child : ChildIndex) : Definition where
  output := layout.signProductColumn coordinate child
  rhs := .product
    [(layout.signColumn coordinate, 1)]
    [(layout.magnitudeBitColumn coordinate child, 1)]

def signProductRow (layout : Layout) (coordinate : PublicCoordinate)
    (child : ChildIndex) : Row :=
  (signProductDefinition layout coordinate child).builderRow

/-- `child = magnitudeBit - 2 * sign * magnitudeBit` in Goldilocks. -/
def childDefinition (layout : Layout) (coordinate : PublicCoordinate)
    (child : ChildIndex) : Definition where
  output := layout.childColumn child coordinate
  rhs := .linear
    [(layout.magnitudeBitColumn coordinate child, 1),
     (layout.signProductColumn coordinate child, goldilocksP - 2)]

def childValueRow (layout : Layout) (coordinate : PublicCoordinate)
    (child : ChildIndex) : Row :=
  (childDefinition layout coordinate child).builderRow

def childRows (layout : Layout) (coordinate : PublicCoordinate)
    (child : ChildIndex) : List Row :=
  [magnitudeBitRow layout coordinate child,
   signProductRow layout coordinate child,
   childValueRow layout coordinate child]

def recompositionRow (layout : Layout)
    (coordinate : PublicCoordinate) : Row :=
  (recompositionCheck
    (layout.parentColumn coordinate)
    (List.ofFn fun child : ChildIndex =>
      layout.childColumn child coordinate)
    ProductPiDecRows.radixPowers).row

def coordinateRows (layout : Layout)
    (coordinate : PublicCoordinate) : List Row :=
  [signRow layout coordinate] ++
    (List.ofFn fun child : ChildIndex =>
      childRows layout coordinate child).flatten ++
    [recompositionRow layout coordinate]

def rows (layout : Layout) : List Row :=
  (ProductPiDecRows.indices 540).flatMap fun coordinate =>
    coordinateRows layout coordinate

private theorem length_flatMap_uniform
    {Alpha Beta : Type} (items : List Alpha) (values : Alpha -> List Beta)
    (count : Nat) (uniform : forall item, (values item).length = count) :
    (items.flatMap values).length = items.length * count := by
  induction items with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp [uniform, inductionHypothesis, Nat.add_mul, Nat.add_comm]

@[simp] theorem childRows_length (layout : Layout)
    (coordinate : PublicCoordinate) (child : ChildIndex) :
    (childRows layout coordinate child).length = 3 := rfl

theorem coordinateRows_length (layout : Layout)
    (coordinate : PublicCoordinate) :
    (coordinateRows layout coordinate).length = 44 := by
  simp [coordinateRows]

theorem rows_length (layout : Layout) : (rows layout).length = 23760 := by
  rw [rows, length_flatMap_uniform _ _ 44 (coordinateRows_length layout),
    ProductPiDecRows.indices_length]

private theorem coordinate_rows_hold
    {layout : Layout} {assignment : Nat -> Nat}
    (satisfies : Satisfies (rows layout) assignment)
    (coordinate : PublicCoordinate) :
    Satisfies (coordinateRows layout coordinate) assignment := by
  intro row member
  apply satisfies row
  unfold rows
  exact List.mem_flatMap.mpr
    ⟨coordinate, ProductPiDecRows.index_mem coordinate, member⟩

private theorem sign_row_mem (layout : Layout)
    (coordinate : PublicCoordinate) :
    signRow layout coordinate ∈ coordinateRows layout coordinate := by
  simp [coordinateRows]

private theorem child_row_mem (layout : Layout)
    (coordinate : PublicCoordinate) (child : ChildIndex)
    {row : Row} (member : row ∈ childRows layout coordinate child) :
    row ∈ coordinateRows layout coordinate := by
  apply List.mem_append_left
  apply List.mem_append_right
  exact List.mem_flatten.mpr
    ⟨childRows layout coordinate child,
      List.mem_ofFn.mpr ⟨child, rfl⟩, member⟩

private theorem recomposition_row_mem (layout : Layout)
    (coordinate : PublicCoordinate) :
    recompositionRow layout coordinate ∈ coordinateRows layout coordinate := by
  simp [coordinateRows]

private theorem sign_le_one
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (coordinate : PublicCoordinate) :
    assignment (layout.signColumn coordinate) <= 1 := by
  apply bitRow_le_one
    Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime
    (canonical _) one
  exact coordinate_rows_hold satisfies coordinate _
    (sign_row_mem layout coordinate)

private theorem magnitude_le_one
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (coordinate : PublicCoordinate) (child : ChildIndex) :
    assignment (layout.magnitudeBitColumn coordinate child) <= 1 := by
  apply bitRow_le_one
    Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime
    (canonical _) one
  exact coordinate_rows_hold satisfies coordinate _
    (child_row_mem layout coordinate child (by
      simp [childRows, magnitudeBitRow]))

private theorem sign_product_exact
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (coordinate : PublicCoordinate) (child : ChildIndex) :
    assignment (layout.signProductColumn coordinate child) =
      assignment (layout.signColumn coordinate) *
        assignment (layout.magnitudeBitColumn coordinate child) %
          goldilocksP := by
  have holds : RowHolds assignment
      (signProductRow layout coordinate child) :=
    coordinate_rows_hold satisfies coordinate _
      (child_row_mem layout coordinate child (by simp [childRows]))
  have semantic := definition_sound canonical one
    (signProductDefinition layout coordinate child) (by
      simpa [signProductRow] using holds)
  simpa [Definition.Holds, signProductDefinition, Rhs.eval, lcEval,
    Nat.mod_eq_of_lt (canonical (layout.signColumn coordinate)),
    Nat.mod_eq_of_lt
      (canonical (layout.magnitudeBitColumn coordinate child))] using semantic

private theorem child_linear_exact
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (coordinate : PublicCoordinate) (child : ChildIndex) :
    assignment (layout.childColumn child coordinate) =
      lcEval assignment
        [(layout.magnitudeBitColumn coordinate child, 1),
         (layout.signProductColumn coordinate child, goldilocksP - 2)] := by
  apply builderLinearRow_sound canonical one _ _ (by
    intro term member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> norm_num [goldilocksP])
  exact coordinate_rows_hold satisfies coordinate _
    (child_row_mem layout coordinate child (by
      simp [childRows, childValueRow, childDefinition,
        Definition.builderRow]))

def negative (layout : Layout) (assignment : Nat -> Nat)
    (coordinate : PublicCoordinate) : Bool :=
  assignment (layout.signColumn coordinate) == 1

def magnitudeDigit (layout : Layout) (assignment : Nat -> Nat)
    (coordinate : PublicCoordinate) (child : ChildIndex) : Nat :=
  assignment (layout.magnitudeBitColumn coordinate child)

private theorem child_field_exact
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (coordinate : PublicCoordinate) (child : ChildIndex) :
    fieldAt assignment canonical (layout.childColumn child coordinate) =
      PiDECAlgebra.Radix.signedBinaryDigit
        (negative layout assignment coordinate)
        (magnitudeDigit layout assignment coordinate child) := by
  have signBound := sign_le_one canonical one satisfies coordinate
  have digitBound := magnitude_le_one canonical one satisfies coordinate child
  have productExact := sign_product_exact canonical one satisfies coordinate child
  have childExact := child_linear_exact canonical one satisfies coordinate child
  have signCases : assignment (layout.signColumn coordinate) = 0 ∨
      assignment (layout.signColumn coordinate) = 1 := by omega
  have digitCases : assignment (layout.magnitudeBitColumn coordinate child) = 0 ∨
      assignment (layout.magnitudeBitColumn coordinate child) = 1 := by omega
  rcases signCases with signZero | signOne
  · rcases digitCases with digitZero | digitOne
    · apply Fin.ext
      norm_num [fieldAt, PiDECAlgebra.Radix.signedBinaryDigit, negative,
        magnitudeDigit, signZero, digitZero, productExact, childExact, lcEval,
        goldilocksP, PiDECAlgebra.Radix.fieldOfNat, goldilocksModulus]
    · apply Fin.ext
      norm_num [fieldAt, PiDECAlgebra.Radix.signedBinaryDigit, negative,
        magnitudeDigit, signZero, digitOne, productExact, childExact, lcEval,
        goldilocksP, PiDECAlgebra.Radix.fieldOfNat, goldilocksModulus]
  · rcases digitCases with digitZero | digitOne
    · apply Fin.ext
      norm_num [fieldAt, PiDECAlgebra.Radix.signedBinaryDigit, negative,
        magnitudeDigit, signOne, digitZero, productExact, childExact, lcEval,
        goldilocksP, PiDECAlgebra.Radix.fieldOfNat, goldilocksModulus,
        Fin.val_neg]
    · apply Fin.ext
      norm_num [fieldAt, PiDECAlgebra.Radix.signedBinaryDigit, negative,
        magnitudeDigit, signOne, digitOne, productExact, childExact, lcEval,
        goldilocksP, PiDECAlgebra.Radix.fieldOfNat, goldilocksModulus,
        Fin.val_neg]

private theorem recomposes
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (coordinate : PublicCoordinate) :
    Recomposes assignment (layout.parentColumn coordinate)
      (List.ofFn fun child : ChildIndex =>
        layout.childColumn child coordinate)
      ProductPiDecRows.radixPowers := by
  apply recompositionCheck_sound canonical one _ _ _
    ProductPiDecRows.radixPowers_canonical
  exact coordinate_rows_hold satisfies coordinate _
    (recomposition_row_mem layout coordinate)

/-- Exact rows force every physical child public coordinate to equal the
verifier-computed deterministic public split of the physical parent. -/
theorem rows_sound
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    forall (coordinate : PublicCoordinate) (child : ChildIndex),
      fieldAt assignment canonical (layout.childColumn child coordinate) =
        PiDECAlgebra.Radix.splitScalar
          (fieldAt assignment canonical (layout.parentColumn coordinate))
          child := by
  intro coordinate child
  let digits : ChildIndex -> Nat :=
    fun index => magnitudeDigit layout assignment coordinate index
  let commonSign := negative layout assignment coordinate
  have binary : forall index, digits index < 2 := by
    intro index
    exact Nat.lt_succ_iff.mpr
      (magnitude_le_one canonical one satisfies coordinate index)
  have childExact : forall index,
      fieldAt assignment canonical (layout.childColumn index coordinate) =
        PiDECAlgebra.Radix.signedBinaryDigit commonSign (digits index) := by
    intro index
    exact child_field_exact canonical one satisfies coordinate index
  have parentEquation := recomposes_field canonical
    (layout.parentColumn coordinate)
    (fun index : ChildIndex => layout.childColumn index coordinate)
    EvaluationHomomorphism.PiDEC.radixWeight
    (recomposes canonical one satisfies coordinate)
  have exactRecomposition :
      PiDECAlgebra.Radix.recomposeScalar
          (fun index => PiDECAlgebra.Radix.signedBinaryDigit
            commonSign (digits index)) =
        fieldAt assignment canonical (layout.parentColumn coordinate) := by
    change ProductPiDecLinearCombination.combineFields
      EvaluationHomomorphism.PiDEC.radixWeight
        (fun index => PiDECAlgebra.Radix.signedBinaryDigit
          commonSign (digits index)) = _
    rw [parentEquation]
    apply congrArg
      (ProductPiDecLinearCombination.combineFields
        EvaluationHomomorphism.PiDEC.radixWeight)
    funext index
    exact (childExact index).symm
  have splitExact :=
    PiDECAlgebra.Radix.splitScalar_eq_signedBinary_of_recompose
      (fieldAt assignment canonical (layout.parentColumn coordinate))
      commonSign digits binary exactRecomposition child
  exact (childExact child).trans splitExact.symm

end Nightstream.Implementation.NebulaV2.ProductPiDecPublicSplitRows
