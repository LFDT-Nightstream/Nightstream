import Nightstream.Implementation.Lowering.Nebula.Compiler

/-!
Contract: exact R1CS lowering of one Lean-owned Nebula CCS row.

Assurance tier: model-level.

Owns: the four row kinds selected by the Nebula family tag, five named
extension intermediates, exact one-row and six-row lowering, and local
soundness and honest completeness.

Does not own: terminal Ajtai checks, Spartan, WHIR, JSON, Rust, or the
placement of this relation beside the native F-prime relation.

Emits constraints: one R1CS row for bit, product, and linear source rows;
six R1CS rows and five auxiliary columns for each extension source row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Nebula.TerminalR1cs

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula

inductive Kind where
  | bit
  | product
  | linear
  | extension
deriving DecidableEq, Repr

def familyKind : Rows.Family -> Kind
  | .operationBit | .initialScanBit | .finalScanBit => .bit
  | .readWrite | .timestampOrder | .romWrite | .romRange | .padding =>
      .product
  | .filler | .operationCount | .boundaryTimestamp | .boundaryProduct =>
      .linear
  | .readProduct | .writeProduct | .initialScanProduct |
      .finalScanProduct => .extension

inductive Auxiliary where
  | valueAProduct
  | valueBProduct
  | extensionAContribution
  | extensionBContribution
  | activeContribution
deriving DecidableEq, Repr

inductive Column where
  | source (column : Nat)
  | auxiliary (position : Nat) (kind : Auxiliary)
deriving DecidableEq, Repr

structure Term where
  column : Column
  coefficient : F
deriving DecidableEq, Repr

abbrev LinearCombination := List Term

namespace LinearCombination

def eval (assignment : Column -> F) : LinearCombination -> F
  | [] => 0
  | term :: rest =>
      term.coefficient * assignment term.column + eval assignment rest

def source (combination : Rows.LinearCombination) : LinearCombination :=
  combination.map fun term =>
    { column := .source term.column, coefficient := term.coefficient }

def singleton (column : Column) : LinearCombination :=
  [{ column := column, coefficient := 1 }]

def one : LinearCombination := singleton (.source 0)

def add (left right : LinearCombination) : LinearCombination := left ++ right

def scale (coefficient : F)
    (combination : LinearCombination) : LinearCombination :=
  combination.map fun term =>
    { term with coefficient := coefficient * term.coefficient }

def neg (combination : LinearCombination) : LinearCombination :=
  scale (-1) combination

def sub (left right : LinearCombination) : LinearCombination :=
  add left (neg right)

@[simp] theorem eval_source (assignment : Column -> F)
    (combination : Rows.LinearCombination) :
    eval assignment (source combination) =
      Rows.LinearCombination.eval (fun column => assignment (.source column))
        combination := by
  induction combination with
  | nil => rfl
  | cons term rest inductionHypothesis =>
      change
        term.coefficient * assignment (.source term.column) +
            eval assignment (source rest) =
          term.coefficient * assignment (.source term.column) +
            Rows.LinearCombination.eval
              (fun column => assignment (.source column)) rest
      rw [inductionHypothesis]

@[simp] theorem eval_singleton (assignment : Column -> F)
    (column : Column) :
    eval assignment (singleton column) = assignment column := by
  simp [singleton, eval, Fin.one_mul]

@[simp] theorem eval_one (assignment : Column -> F) :
    eval assignment one = assignment (.source 0) := by
  simp [one]

theorem eval_add (assignment : Column -> F)
    (left right : LinearCombination) :
    eval assignment (add left right) =
      eval assignment left + eval assignment right := by
  induction left with
  | nil => simp [add, eval]
  | cons term rest inductionHypothesis =>
      unfold add at inductionHypothesis ⊢
      simp only [List.cons_append, eval]
      rw [inductionHypothesis, Lean.Grind.Fin.add_assoc]

private theorem mul_assoc (left middle right : F) :
    (left * middle) * right = left * (middle * right) :=
  Fin.mul_assoc _ _ _

private theorem mul_add (left middle right : F) :
    left * (middle + right) = left * middle + left * right :=
  Lean.Grind.Fin.left_distrib _ _ _

theorem eval_scale (assignment : Column -> F) (coefficient : F)
    (combination : LinearCombination) :
    eval assignment (scale coefficient combination) =
      coefficient * eval assignment combination := by
  induction combination with
  | nil => simp [scale, eval, Fin.mul_zero]
  | cons term rest inductionHypothesis =>
      change
        (coefficient * term.coefficient) * assignment term.column +
            eval assignment (scale coefficient rest) =
          coefficient *
            (term.coefficient * assignment term.column + eval assignment rest)
      rw [inductionHypothesis]
      rw [mul_assoc, ← mul_add]

theorem eval_neg (assignment : Column -> F)
    (combination : LinearCombination) :
    eval assignment (neg combination) = -eval assignment combination := by
  rw [show neg combination = scale (-1) combination from rfl, eval_scale]
  calc
    (-1 : F) * eval assignment combination =
        -(1 * eval assignment combination) :=
      Lean.Grind.Fin.neg_mul _ _
    _ = -eval assignment combination := by rw [Fin.one_mul]

theorem eval_sub (assignment : Column -> F)
    (left right : LinearCombination) :
    eval assignment (sub left right) =
      eval assignment left - eval assignment right := by
  rw [show sub left right = add left (neg right) from rfl,
    eval_add, eval_neg]
  rw [Fin.sub_eq_add_neg]

end LinearCombination

structure Row where
  a : LinearCombination
  b : LinearCombination
  c : LinearCombination
deriving DecidableEq, Repr

def Row.Holds (assignment : Column -> F) (row : Row) : Prop :=
  row.a.eval assignment * row.b.eval assignment = row.c.eval assignment

def Satisfies : List Row -> (Column -> F) -> Prop
  | [], _ => True
  | row :: rest, assignment =>
      row.Holds assignment ∧ Satisfies rest assignment

@[simp] theorem satisfies_append_iff (left right : List Row)
    (assignment : Column -> F) :
    Satisfies (left ++ right) assignment ↔
      Satisfies left assignment ∧ Satisfies right assignment := by
  induction left with
  | nil => simp [Satisfies]
  | cons row rest inductionHypothesis =>
      simp only [List.cons_append, Satisfies, inductionHypothesis]
      constructor
      · rintro ⟨rowHolds, restHolds, rightHolds⟩
        exact ⟨⟨rowHolds, restHolds⟩, rightHolds⟩
      · rintro ⟨⟨rowHolds, restHolds⟩, rightHolds⟩
        exact ⟨rowHolds, restHolds, rightHolds⟩

theorem satisfies_iff_forall (rows : List Row) (assignment : Column -> F) :
    Satisfies rows assignment ↔
      ∀ row, row ∈ rows -> row.Holds assignment := by
  induction rows with
  | nil => simp [Satisfies]
  | cons head rest inductionHypothesis =>
      simp only [Satisfies, inductionHypothesis, List.mem_cons]
      constructor
      · rintro ⟨headHolds, restHolds⟩ row (rfl | member)
        · exact headHolds
        · exact restHolds row member
      · intro every
        exact ⟨every head (Or.inl rfl), fun row member =>
          every row (Or.inr member)⟩

def auxiliary (row : Rows.Row) (kind : Auxiliary) : LinearCombination :=
  LinearCombination.singleton (.auxiliary row.id.position kind)

def bitRows (row : Rows.Row) : List Row :=
  [{ a := .source row.images.bit
     b := .sub (.source row.images.bit) .one
     c := [] }]

def productRows (row : Rows.Row) : List Row :=
  [{ a := .source row.images.productLeft
     b := .source row.images.productRight
     c := [] }]

def linearRows (row : Rows.Row) : List Row :=
  [{ a := .one
     b := .source row.images.linearLeft
     c := .source row.images.linearRight }]

def extensionRows (row : Rows.Row) : List Row :=
  let valueAProduct := auxiliary row .valueAProduct
  let valueBProduct := auxiliary row .valueBProduct
  let extensionAContribution := auxiliary row .extensionAContribution
  let extensionBContribution := auxiliary row .extensionBContribution
  let activeContribution := auxiliary row .activeContribution
  [ { a := .source row.images.valueA
      b := .source row.images.value
      c := valueAProduct }
  , { a := .source row.images.valueB
      b := .source row.images.value
      c := valueBProduct }
  , { a := .source row.images.extensionA
      b := .sub (.source row.images.fingerprintA) valueAProduct
      c := extensionAContribution }
  , { a := .source row.images.extensionB
      b := .sub (.source row.images.fingerprintB) valueBProduct
      c := extensionBContribution }
  , { a := .source row.images.active
      b := .add extensionAContribution extensionBContribution
      c := activeContribution }
  , { a := .source row.images.extensionA
      b := .source row.images.pad
      c := .sub (.source row.images.output) activeContribution } ]

def lowerRow (row : Rows.Row) : List Row :=
  match familyKind row.id.family with
  | .bit => bitRows row
  | .product => productRows row
  | .linear => linearRows row
  | .extension => extensionRows row

def auxiliaryColumns (row : Rows.Row) : List Column :=
  match familyKind row.id.family with
  | .extension =>
      [ .auxiliary row.id.position .valueAProduct
      , .auxiliary row.id.position .valueBProduct
      , .auxiliary row.id.position .extensionAContribution
      , .auxiliary row.id.position .extensionBContribution
      , .auxiliary row.id.position .activeContribution ]
  | _ => []

@[simp] theorem bitRows_length (row : Rows.Row) :
    (bitRows row).length = 1 := rfl

@[simp] theorem productRows_length (row : Rows.Row) :
    (productRows row).length = 1 := rfl

@[simp] theorem linearRows_length (row : Rows.Row) :
    (linearRows row).length = 1 := rfl

@[simp] theorem extensionRows_length (row : Rows.Row) :
    (extensionRows row).length = 6 := rfl

@[simp] theorem auxiliaryColumns_extension_length (row : Rows.Row)
    (kind : familyKind row.id.family = .extension) :
    (auxiliaryColumns row).length = 5 := by
  simp [auxiliaryColumns, kind]

@[simp] theorem auxiliaryColumns_nonextension_length (row : Rows.Row)
    (kind : familyKind row.id.family ≠ .extension) :
    (auxiliaryColumns row).length = 0 := by
  cases familyKindEq : familyKind row.id.family <;>
    simp_all [auxiliaryColumns]

/-- The family tag selects the exact sparse constructor that produced the
source row. This is the fail-closed boundary used by the terminal lowering. -/
inductive Shape : Rows.Row -> Prop where
  | bit (id : Rows.RowId) (column : Nat)
      (kind : familyKind id.family = .bit) :
      Shape (Rows.bitRow id column)
  | product (id : Rows.RowId)
      (left right : Rows.LinearCombination)
      (kind : familyKind id.family = .product) :
      Shape (Rows.productRow id left right)
  | linear (id : Rows.RowId)
      (left right : Rows.LinearCombination)
      (kind : familyKind id.family = .linear) :
      Shape (Rows.linearRow id left right)
  | extension (id : Rows.RowId)
      (output extensionA extensionB pad active fingerprintA fingerprintB
        valueA valueB value : Rows.LinearCombination)
      (kind : familyKind id.family = .extension) :
      Shape (Rows.extensionUpdateRow id output extensionA extensionB pad
        active fingerprintA fingerprintB valueA valueB value)

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨Lean.Grind.Fin.add_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨Lean.Grind.Fin.add_comm⟩

private theorem mul_neg (left right : F) :
    left * -right = -(left * right) := by
  calc
    left * -right = -right * left := Fin.mul_comm _ _
    _ = -(right * left) := Lean.Grind.Fin.neg_mul _ _
    _ = -(left * right) := by rw [Fin.mul_comm right left]

private theorem mul_sub_expanded
    (left factor coefficient value : F) :
    left * (factor - coefficient * value) =
      left * factor + -(left * coefficient * value) := by
  rw [Fin.sub_eq_add_neg, Lean.Grind.Fin.left_distrib, mul_neg,
    ← Fin.mul_assoc]

private theorem update_component
    (left right pad active factorA coefficientA factorB coefficientB
      value : F) :
    left * (pad + active * (factorA - coefficientA * value)) +
        right * (active * (factorB - coefficientB * value)) =
      left * pad + left * active * factorA +
        -(left * active * coefficientA * value) +
        right * active * factorB +
        -(right * active * coefficientB * value) := by
  rw [Lean.Grind.Fin.left_distrib,
    ← Fin.mul_assoc left active,
    mul_sub_expanded,
    ← Fin.mul_assoc right active,
    mul_sub_expanded]
  simp only [Lean.Grind.Fin.add_assoc]

private theorem swap_mul_left (active left factor : F) :
    active * (left * factor) = left * (active * factor) := by
  rw [← Fin.mul_assoc, Fin.mul_comm active left, Fin.mul_assoc]

private theorem grouped_component
    (left right pad active factorA coefficientA factorB coefficientB
      value : F) :
    left * pad +
        active *
          (left * (factorA - coefficientA * value) +
            right * (factorB - coefficientB * value)) =
      left * (pad + active * (factorA - coefficientA * value)) +
        right * (active * (factorB - coefficientB * value)) := by
  simp only [Lean.Grind.Fin.left_distrib]
  rw [swap_mul_left active left (factorA - coefficientA * value),
    swap_mul_left active right (factorB - coefficientB * value)]
  rw [← Lean.Grind.Fin.add_assoc,
    Lean.Grind.Fin.add_comm
      (left * pad + left * (active * (factorA - coefficientA * value)))
      (right * (active * (factorB - coefficientB * value)))]

private theorem source_terms_grouped
    (left right pad active factorA coefficientA factorB coefficientB
      value : F) :
    left * pad + left * active * factorA +
        -(left * active * coefficientA * value) +
        right * active * factorB +
        -(right * active * coefficientB * value) =
      left * pad +
        active *
          (left * (factorA - coefficientA * value) +
            right * (factorB - coefficientB * value)) := by
  exact (update_component left right pad active factorA coefficientA
    factorB coefficientB value).symm.trans
      (grouped_component left right pad active factorA coefficientA
        factorB coefficientB value).symm

private theorem extension_algebra
    (output extensionA extensionB pad active fingerprintA fingerprintB
      valueA valueB value valueAProduct valueBProduct
      extensionAContribution extensionBContribution activeContribution : F)
    (valueAProductEq : valueA * value = valueAProduct)
    (valueBProductEq : valueB * value = valueBProduct)
    (extensionAContributionEq :
      extensionA * (fingerprintA - valueAProduct) =
        extensionAContribution)
    (extensionBContributionEq :
      extensionB * (fingerprintB - valueBProduct) =
        extensionBContribution)
    (activeContributionEq :
      active * (extensionAContribution + extensionBContribution) =
        activeContribution)
    (outputEq : extensionA * pad = output - activeContribution) :
    -output + extensionA * pad +
        extensionA * active * fingerprintA +
        -(extensionA * active * valueA * value) +
        extensionB * active * fingerprintB +
        -(extensionB * active * valueB * value) = 0 := by
  calc
    -output + extensionA * pad +
          extensionA * active * fingerprintA +
          -(extensionA * active * valueA * value) +
          extensionB * active * fingerprintB +
          -(extensionB * active * valueB * value) =
        -output +
          (extensionA * pad + extensionA * active * fingerprintA +
            -(extensionA * active * valueA * value) +
            extensionB * active * fingerprintB +
            -(extensionB * active * valueB * value)) := by
      simp only [Lean.Grind.Fin.add_assoc]
    _ = -output +
          (extensionA * pad +
            active *
              (extensionA * (fingerprintA - valueA * value) +
                extensionB * (fingerprintB - valueB * value))) := by
      rw [source_terms_grouped]
    _ = -output +
          (extensionA * pad +
            active *
              (extensionA * (fingerprintA - valueAProduct) +
                extensionB * (fingerprintB - valueBProduct))) := by
      rw [valueAProductEq, valueBProductEq]
    _ = -output +
          (extensionA * pad +
            active *
              (extensionAContribution + extensionBContribution)) := by
      rw [extensionAContributionEq, extensionBContributionEq]
    _ = -output + (extensionA * pad + activeContribution) := by
      rw [activeContributionEq]
    _ = -output + ((output - activeContribution) + activeContribution) := by
      rw [outputEq]
    _ = (-output + output) +
          (-activeContribution + activeContribution) := by
      rw [Fin.sub_eq_add_neg]
      ac_rfl
    _ = 0 := by
      rw [Lean.Grind.Fin.neg_add_cancel,
        Lean.Grind.Fin.neg_add_cancel, Fin.zero_add]

theorem bitRows_sound (id : Rows.RowId) (column : Nat)
    (assignment : Column -> F)
    (constantOne : assignment (.source 0) = 1)
    (holds : Satisfies (bitRows (Rows.bitRow id column)) assignment) :
    (Rows.bitRow id column).Holds
      (fun source => assignment (.source source)) := by
  rw [Rows.bitRow_holds_iff]
  have equation :
      assignment (.source column) *
          (assignment (.source column) - 1) = 0 := by
    simpa [bitRows, Satisfies, Row.Holds, Rows.bitRow,
      LinearCombination.eval_sub, constantOne] using holds.1
  have factored :
      assignment (.source column) *
          (assignment (.source column) - 1) =
        assignment (.source column) * assignment (.source column) +
          -assignment (.source column) := by
    rw [Fin.sub_eq_add_neg, Lean.Grind.Fin.left_distrib]
    have negOne :
        assignment (.source column) * (-1) =
          -assignment (.source column) := by
      calc
        assignment (.source column) * (-1) =
            (-1) * assignment (.source column) := Fin.mul_comm _ _
        _ = -(1 * assignment (.source column)) :=
          Lean.Grind.Fin.neg_mul _ _
        _ = -assignment (.source column) := by rw [Fin.one_mul]
    rw [negOne]
  calc
    assignment (.source column) * assignment (.source column) +
          -assignment (.source column) =
        assignment (.source column) *
          (assignment (.source column) - 1) := factored.symm
    _ = 0 := equation

theorem productRows_sound (id : Rows.RowId)
    (left right : Rows.LinearCombination) (assignment : Column -> F)
    (holds :
      Satisfies (productRows (Rows.productRow id left right)) assignment) :
    (Rows.productRow id left right).Holds
      (fun source => assignment (.source source)) := by
  rw [Rows.productRow_holds_iff]
  simpa [productRows, Satisfies, Row.Holds] using holds.1

theorem linearRows_sound (id : Rows.RowId)
    (left right : Rows.LinearCombination) (assignment : Column -> F)
    (constantOne : assignment (.source 0) = 1)
    (holds : Satisfies (linearRows (Rows.linearRow id left right))
      assignment) :
    (Rows.linearRow id left right).Holds
      (fun source => assignment (.source source)) := by
  rw [Rows.linearRow_holds_iff]
  have equation :
      Rows.LinearCombination.eval
          (fun source => assignment (.source source)) left =
        Rows.LinearCombination.eval
          (fun source => assignment (.source source)) right := by
    simpa [linearRows, Satisfies, Row.Holds, Rows.linearRow,
      constantOne, Fin.one_mul] using holds.1
  rw [equation]
  have cancel :
      Rows.LinearCombination.eval
          (fun source => assignment (.source source)) right -
        Rows.LinearCombination.eval
          (fun source => assignment (.source source)) right = 0 :=
    Fin.sub_self
  simpa only [Fin.sub_eq_add_neg] using cancel

theorem extensionRows_sound (id : Rows.RowId)
    (output extensionA extensionB pad active fingerprintA fingerprintB
      valueA valueB value : Rows.LinearCombination)
    (assignment : Column -> F)
    (holds :
      Satisfies
        (extensionRows
          (Rows.extensionUpdateRow id output extensionA extensionB pad active
            fingerprintA fingerprintB valueA valueB value)) assignment) :
    (Rows.extensionUpdateRow id output extensionA extensionB pad active
      fingerprintA fingerprintB valueA valueB value).Holds
      (fun source => assignment (.source source)) := by
  rcases holds with
    ⟨valueAEq, valueBEq, extensionAEq, extensionBEq, activeEq,
      outputEq, _⟩
  simp only [Row.Holds, LinearCombination.eval_source,
    LinearCombination.eval_singleton, LinearCombination.eval_sub,
    LinearCombination.eval_add, auxiliary] at valueAEq valueBEq extensionAEq extensionBEq activeEq outputEq
  rw [Rows.extensionUpdateRow_holds_iff]
  exact extension_algebra
    (Rows.LinearCombination.eval
      (fun source => assignment (.source source)) output)
    (Rows.LinearCombination.eval
      (fun source => assignment (.source source)) extensionA)
    (Rows.LinearCombination.eval
      (fun source => assignment (.source source)) extensionB)
    (Rows.LinearCombination.eval
      (fun source => assignment (.source source)) pad)
    (Rows.LinearCombination.eval
      (fun source => assignment (.source source)) active)
    (Rows.LinearCombination.eval
      (fun source => assignment (.source source)) fingerprintA)
    (Rows.LinearCombination.eval
      (fun source => assignment (.source source)) fingerprintB)
    (Rows.LinearCombination.eval
      (fun source => assignment (.source source)) valueA)
    (Rows.LinearCombination.eval
      (fun source => assignment (.source source)) valueB)
    (Rows.LinearCombination.eval
      (fun source => assignment (.source source)) value)
    (assignment (.auxiliary id.position .valueAProduct))
    (assignment (.auxiliary id.position .valueBProduct))
    (assignment (.auxiliary id.position .extensionAContribution))
    (assignment (.auxiliary id.position .extensionBContribution))
    (assignment (.auxiliary id.position .activeContribution))
    valueAEq valueBEq extensionAEq extensionBEq activeEq outputEq

theorem lowerRow_sound (row : Rows.Row) (assignment : Column -> F)
    (constantOne : assignment (.source 0) = 1)
    (shape : Shape row) (holds : Satisfies (lowerRow row) assignment) :
    row.Holds (fun source => assignment (.source source)) := by
  cases shape with
  | bit id column kind =>
      rw [show lowerRow (Rows.bitRow id column) =
        bitRows (Rows.bitRow id column) by
          simp [lowerRow, Rows.bitRow, kind]] at holds
      exact bitRows_sound id column assignment constantOne holds
  | product id left right kind =>
      rw [show lowerRow (Rows.productRow id left right) =
        productRows (Rows.productRow id left right) by
          simp [lowerRow, Rows.productRow, kind]] at holds
      exact productRows_sound id left right assignment holds
  | linear id left right kind =>
      rw [show lowerRow (Rows.linearRow id left right) =
        linearRows (Rows.linearRow id left right) by
          simp [lowerRow, Rows.linearRow, kind]] at holds
      exact linearRows_sound id left right assignment constantOne holds
  | extension id output extensionA extensionB pad active fingerprintA
      fingerprintB valueA valueB value kind =>
      rw [show lowerRow
        (Rows.extensionUpdateRow id output extensionA extensionB pad active
          fingerprintA fingerprintB valueA valueB value) =
        extensionRows
          (Rows.extensionUpdateRow id output extensionA extensionB pad active
            fingerprintA fingerprintB valueA valueB value) by
          simp [lowerRow, Rows.extensionUpdateRow, kind]] at holds
      exact extensionRows_sound id output extensionA extensionB pad active
        fingerprintA fingerprintB valueA valueB value assignment holds

def sourceValue (source : Nat -> F)
    (combination : Rows.LinearCombination) : F :=
  Rows.LinearCombination.eval source combination

def auxiliaryValue (row : Rows.Row) (source : Nat -> F) : Auxiliary -> F
  | .valueAProduct =>
      sourceValue source row.images.valueA *
        sourceValue source row.images.value
  | .valueBProduct =>
      sourceValue source row.images.valueB *
        sourceValue source row.images.value
  | .extensionAContribution =>
      sourceValue source row.images.extensionA *
        (sourceValue source row.images.fingerprintA -
          sourceValue source row.images.valueA *
            sourceValue source row.images.value)
  | .extensionBContribution =>
      sourceValue source row.images.extensionB *
        (sourceValue source row.images.fingerprintB -
          sourceValue source row.images.valueB *
            sourceValue source row.images.value)
  | .activeContribution =>
      sourceValue source row.images.active *
        (sourceValue source row.images.extensionA *
            (sourceValue source row.images.fingerprintA -
              sourceValue source row.images.valueA *
                sourceValue source row.images.value) +
          sourceValue source row.images.extensionB *
            (sourceValue source row.images.fingerprintB -
              sourceValue source row.images.valueB *
                sourceValue source row.images.value))

/-- Honest local assignment. Auxiliary values outside this row position are
zero because a whole-program completion supplies them from their own row. -/
def complete (row : Rows.Row) (source : Nat -> F) : Column -> F
  | .source column => source column
  | .auxiliary position kind =>
      if position = row.id.position then auxiliaryValue row source kind else 0

@[simp] theorem complete_source (row : Rows.Row) (source : Nat -> F)
    (column : Nat) :
    complete row source (.source column) = source column :=
  rfl

@[simp] theorem complete_auxiliary (row : Rows.Row) (source : Nat -> F)
    (kind : Auxiliary) :
    complete row source (.auxiliary row.id.position kind) =
      auxiliaryValue row source kind := by
  simp [complete]

private theorem output_eq_of_residual
    (output base active : F)
    (residual : -output + (base + active) = 0) :
    base = output - active := by
  apply Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp
  calc
    base - (output - active) = -output + (base + active) := by
      rw [Fin.sub_eq_add_neg, Fin.sub_eq_add_neg,
        Lean.Grind.AddCommGroup.neg_add,
        Lean.Grind.AddCommGroup.neg_neg]
      ac_rfl
    _ = 0 := residual

theorem bitRows_complete (id : Rows.RowId) (column : Nat)
    (source : Nat -> F) (constantOne : source 0 = 1)
    (holds : (Rows.bitRow id column).Holds source) :
    Satisfies (bitRows (Rows.bitRow id column))
      (complete (Rows.bitRow id column) source) := by
  rw [Rows.bitRow_holds_iff] at holds
  constructor
  · simp only [Row.Holds, Rows.bitRow, LinearCombination.eval_source,
      LinearCombination.eval_sub, LinearCombination.eval_one,
      LinearCombination.eval, Rows.LinearCombination.eval_bit,
      complete_source]
    rw [constantOne]
    have factored :
        source column * (source column - 1) =
          source column * source column + -source column := by
      rw [Fin.sub_eq_add_neg, Lean.Grind.Fin.left_distrib]
      have negOne : source column * (-1) = -source column := by
        calc
          source column * (-1) = (-1) * source column := Fin.mul_comm _ _
          _ = -(1 * source column) := Lean.Grind.Fin.neg_mul _ _
          _ = -source column := by rw [Fin.one_mul]
      rw [negOne]
    rw [factored]
    exact holds
  · trivial

theorem productRows_complete (id : Rows.RowId)
    (left right : Rows.LinearCombination) (source : Nat -> F)
    (holds : (Rows.productRow id left right).Holds source) :
    Satisfies (productRows (Rows.productRow id left right))
      (complete (Rows.productRow id left right) source) := by
  rw [Rows.productRow_holds_iff] at holds
  constructor
  · simpa [Row.Holds, Rows.productRow] using holds
  · trivial

theorem linearRows_complete (id : Rows.RowId)
    (left right : Rows.LinearCombination) (source : Nat -> F)
    (constantOne : source 0 = 1)
    (holds : (Rows.linearRow id left right).Holds source) :
    Satisfies (linearRows (Rows.linearRow id left right))
      (complete (Rows.linearRow id left right) source) := by
  rw [Rows.linearRow_holds_iff] at holds
  have equal :
      sourceValue source left = sourceValue source right :=
    Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp (by
      simpa only [sourceValue, Fin.sub_eq_add_neg] using holds)
  constructor
  · simp only [Row.Holds, Rows.linearRow, LinearCombination.eval_one,
      LinearCombination.eval_source]
    change source 0 * sourceValue source left = sourceValue source right
    rw [constantOne, Fin.one_mul, equal]
  · trivial

theorem extensionRows_complete (id : Rows.RowId)
    (output extensionA extensionB pad active fingerprintA fingerprintB
      valueA valueB value : Rows.LinearCombination)
    (source : Nat -> F)
    (holds :
      (Rows.extensionUpdateRow id output extensionA extensionB pad active
        fingerprintA fingerprintB valueA valueB value).Holds source) :
    Satisfies
      (extensionRows
        (Rows.extensionUpdateRow id output extensionA extensionB pad active
          fingerprintA fingerprintB valueA valueB value))
      (complete
        (Rows.extensionUpdateRow id output extensionA extensionB pad active
          fingerprintA fingerprintB valueA valueB value) source) := by
  rw [Rows.extensionUpdateRow_holds_iff] at holds
  let outputValue := sourceValue source output
  let extensionAValue := sourceValue source extensionA
  let extensionBValue := sourceValue source extensionB
  let padValue := sourceValue source pad
  let activeValue := sourceValue source active
  let fingerprintAValue := sourceValue source fingerprintA
  let fingerprintBValue := sourceValue source fingerprintB
  let valueAValue := sourceValue source valueA
  let valueBValue := sourceValue source valueB
  let valueValue := sourceValue source value
  let contributionA :=
    extensionAValue * (fingerprintAValue - valueAValue * valueValue)
  let contributionB :=
    extensionBValue * (fingerprintBValue - valueBValue * valueValue)
  let activeContribution := activeValue * (contributionA + contributionB)
  have groupedResidual :
      -outputValue +
        (extensionAValue * padValue + activeContribution) = 0 := by
    calc
      -outputValue +
          (extensionAValue * padValue + activeContribution) =
        -outputValue +
          (extensionAValue * padValue +
            activeValue *
              (extensionAValue *
                  (fingerprintAValue - valueAValue * valueValue) +
                extensionBValue *
                  (fingerprintBValue - valueBValue * valueValue))) := rfl
      _ = -outputValue +
          (extensionAValue * padValue +
            extensionAValue * activeValue * fingerprintAValue +
            -(extensionAValue * activeValue * valueAValue * valueValue) +
            extensionBValue * activeValue * fingerprintBValue +
            -(extensionBValue * activeValue * valueBValue * valueValue)) := by
        rw [← source_terms_grouped]
      _ = -outputValue + extensionAValue * padValue +
          extensionAValue * activeValue * fingerprintAValue +
          -(extensionAValue * activeValue * valueAValue * valueValue) +
          extensionBValue * activeValue * fingerprintBValue +
          -(extensionBValue * activeValue * valueBValue * valueValue) := by
        simp only [Lean.Grind.Fin.add_assoc]
      _ = 0 := by
        simpa [outputValue, extensionAValue, extensionBValue, padValue,
          activeValue, fingerprintAValue, fingerprintBValue, valueAValue,
          valueBValue, valueValue, sourceValue] using holds
  have outputEquation :
      extensionAValue * padValue = outputValue - activeContribution :=
    output_eq_of_residual outputValue
      (extensionAValue * padValue) activeContribution groupedResidual
  constructor
  · simp [Row.Holds, Rows.extensionUpdateRow, complete, auxiliary,
      auxiliaryValue, sourceValue]
  constructor
  · simp [Row.Holds, Rows.extensionUpdateRow, complete, auxiliary,
      auxiliaryValue, sourceValue]
  constructor
  · simp [Row.Holds, Rows.extensionUpdateRow, complete, auxiliary,
      auxiliaryValue, sourceValue, LinearCombination.eval_sub]
  constructor
  · simp [Row.Holds, Rows.extensionUpdateRow, complete, auxiliary,
      auxiliaryValue, sourceValue, LinearCombination.eval_sub]
  constructor
  · simp [Row.Holds, Rows.extensionUpdateRow, complete, auxiliary,
      auxiliaryValue, sourceValue, LinearCombination.eval_add]
  constructor
  · simpa [Row.Holds, Rows.extensionUpdateRow, complete, auxiliary,
      auxiliaryValue, sourceValue, outputValue, extensionAValue,
      extensionBValue, padValue,
      activeValue, fingerprintAValue, fingerprintBValue, valueAValue,
      valueBValue, valueValue, contributionA, contributionB,
      activeContribution, LinearCombination.eval_sub] using outputEquation
  · trivial

theorem lowerRow_complete (row : Rows.Row) (source : Nat -> F)
    (constantOne : source 0 = 1) (shape : Shape row)
    (holds : row.Holds source) :
    Satisfies (lowerRow row) (complete row source) := by
  cases shape with
  | bit id column kind =>
      rw [show lowerRow (Rows.bitRow id column) =
        bitRows (Rows.bitRow id column) by
          simp [lowerRow, Rows.bitRow, kind]]
      exact bitRows_complete id column source constantOne holds
  | product id left right kind =>
      rw [show lowerRow (Rows.productRow id left right) =
        productRows (Rows.productRow id left right) by
          simp [lowerRow, Rows.productRow, kind]]
      exact productRows_complete id left right source holds
  | linear id left right kind =>
      rw [show lowerRow (Rows.linearRow id left right) =
        linearRows (Rows.linearRow id left right) by
          simp [lowerRow, Rows.linearRow, kind]]
      exact linearRows_complete id left right source constantOne holds
  | extension id output extensionA extensionB pad active fingerprintA
      fingerprintB valueA valueB value kind =>
      rw [show lowerRow
        (Rows.extensionUpdateRow id output extensionA extensionB pad active
          fingerprintA fingerprintB valueA valueB value) =
        extensionRows
          (Rows.extensionUpdateRow id output extensionA extensionB pad active
            fingerprintA fingerprintB valueA valueB value) by
          simp [lowerRow, Rows.extensionUpdateRow, kind]]
      exact extensionRows_complete id output extensionA extensionB pad active
        fingerprintA fingerprintB valueA valueB value source holds

end Nightstream.Implementation.Lowering.Nebula.TerminalR1cs
