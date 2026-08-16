import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Boolean
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.Decoder

/-!
Contract: exact source-to-final linear images for grouped-product rewrites.

Assurance tier: model-level artifact interpreter.

Owns: low-norm slot images, recursive source-definition substitution, source
linear-combination images, derived-slot images, and the exact expected port
images of one five-product rewrite row.

Does not own: a concrete Rust artifact, production coverage, row necessity,
norm enforcement, lifecycle soundness, or permission to remove coordinates.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder

/-- A finite coefficient function with an explicit sparse support. The support
is proof data: coefficients outside it are zero. It keeps generated artifact
checks proportional to the used coordinates, not to the full assignment. -/
structure Form (columns : Nat) where
  coefficient : Fin columns → F
  support : List (Fin columns)
  zeroOutside : ∀ column, column ∉ support → coefficient column = 0

namespace Form

instance {columns : Nat} : CoeFun (Form columns) (fun _ => Fin columns → F) :=
  ⟨Form.coefficient⟩

def zero {columns : Nat} : Form columns where
  coefficient := fun _ => 0
  support := []
  zeroOutside := by simp

@[simp] theorem zero_apply {columns : Nat} (column : Fin columns) :
    (zero : Form columns) column = 0 := rfl

def add {columns : Nat} (left right : Form columns) : Form columns where
  coefficient := fun column => left column + right column
  support := left.support ++ right.support
  zeroOutside := by
    intro column absent
    have leftAbsent : column ∉ left.support := by
      intro member
      exact absent (List.mem_append.mpr (Or.inl member))
    have rightAbsent : column ∉ right.support := by
      intro member
      exact absent (List.mem_append.mpr (Or.inr member))
    rw [left.zeroOutside column leftAbsent,
      right.zeroOutside column rightAbsent]
    exact Fin.zero_add 0

@[simp] theorem add_apply {columns : Nat} (left right : Form columns)
    (column : Fin columns) :
    add left right column = left column + right column := rfl

def scale {columns : Nat} (coefficient : F) (form : Form columns) :
    Form columns where
  coefficient := fun column => coefficient * form column
  support := form.support
  zeroOutside := by
    intro column absent
    rw [form.zeroOutside column absent]
    exact baseLaws.mul_zero coefficient

@[simp] theorem scale_apply {columns : Nat} (coefficient : F)
    (form : Form columns) (column : Fin columns) :
    scale coefficient form column = coefficient * form column := rfl

def sub {columns : Nat} (left right : Form columns) : Form columns :=
  add left (scale (-1) right)

def single {columns : Nat} (selected : Fin columns) (coefficient : F) :
    Form columns where
  coefficient := fun column => if column = selected then coefficient else 0
  support := [selected]
  zeroOutside := by
    intro column absent
    have different : column ≠ selected := by
      intro equal
      apply absent
      simp [equal]
    simp [different]

@[simp] theorem single_apply {columns : Nat} (selected : Fin columns)
    (coefficient : F) (column : Fin columns) :
    single selected coefficient column =
      if column = selected then coefficient else 0 := rfl

private def coefficientsMatch {columns : Nat} (left right : Form columns) :
    List (Fin columns) → Prop
  | [] => True
  | column :: tail =>
      left column = right column ∧ coefficientsMatch left right tail

private def coefficientsMatchDecidable {columns : Nat}
    (left right : Form columns) :
    (indices : List (Fin columns)) →
      Decidable (coefficientsMatch left right indices)
  | [] => isTrue True.intro
  | head :: tail =>
      if equal : left head = right head then
        match coefficientsMatchDecidable left right tail with
        | isTrue tailEqual => isTrue ⟨equal, tailEqual⟩
        | isFalse tailDifferent =>
            isFalse (fun allEqual => tailDifferent allEqual.2)
      else
        isFalse (fun allEqual => equal allEqual.1)

def Equivalent {columns : Nat} (left right : Form columns) : Prop :=
  coefficientsMatch left right (left.support ++ right.support)

instance {columns : Nat} (left right : Form columns) :
    Decidable (Equivalent left right) :=
  coefficientsMatchDecidable left right (left.support ++ right.support)

private theorem coefficientsMatch_of_mem {columns : Nat}
    {left right : Form columns} {indices : List (Fin columns)}
    (agreement : coefficientsMatch left right indices)
    {column : Fin columns} (member : column ∈ indices) :
    left column = right column := by
  induction indices with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      simp only [coefficientsMatch] at agreement
      rw [List.mem_cons] at member
      cases member with
      | inl equal =>
          subst column
          exact agreement.1
      | inr tailMember =>
          exact inductionHypothesis agreement.2 tailMember

theorem pointwise_of_equivalent {columns : Nat} {left right : Form columns}
    (equal : Equivalent left right) (column : Fin columns) :
    left column = right column := by
  by_cases member : column ∈ left.support ++ right.support
  · exact coefficientsMatch_of_mem equal member
  · have leftAbsent : column ∉ left.support := by
      intro leftMember
      exact member (List.mem_append.mpr (Or.inl leftMember))
    have rightAbsent : column ∉ right.support := by
      intro rightMember
      exact member (List.mem_append.mpr (Or.inr rightMember))
    rw [left.zeroOutside column leftAbsent,
      right.zeroOutside column rightAbsent]

private def evaluateOn {columns : Nat} :
    List (Fin columns) → Form columns → (Fin columns → F) → F
  | [], _, _ => 0
  | column :: tail, form, assignment =>
      form column * assignment column + evaluateOn tail form assignment

def evaluate {columns : Nat} (form : Form columns)
    (assignment : Fin columns → F) : F :=
  evaluateOn (canonicalFinIndices columns) form assignment

theorem evaluate_congr {columns : Nat} {left right : Form columns}
    (equal : Equivalent left right) (assignment : Fin columns → F) :
    evaluate left assignment = evaluate right assignment := by
  unfold evaluate
  induction canonicalFinIndices columns with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [evaluateOn]
      rw [pointwise_of_equivalent equal head, inductionHypothesis]

@[simp] theorem evaluate_zero {columns : Nat}
    (assignment : Fin columns → F) :
    evaluate (zero : Form columns) assignment = 0 := by
  unfold evaluate
  induction canonicalFinIndices columns with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [evaluateOn, zero_apply, Fin.zero_mul, Fin.zero_add,
        inductionHypothesis]

@[simp] theorem evaluate_add {columns : Nat}
    (left right : Form columns) (assignment : Fin columns → F) :
    evaluate (add left right) assignment =
      evaluate left assignment + evaluate right assignment := by
  unfold evaluate
  induction canonicalFinIndices columns with
  | nil => exact (Fin.zero_add 0).symm
  | cons head tail inductionHypothesis =>
      simp only [evaluateOn, add_apply, inductionHypothesis]
      have distributeHead :
          (left head + right head) * assignment head =
            left head * assignment head + right head * assignment head := by
        calc
          (left head + right head) * assignment head =
              assignment head * (left head + right head) :=
            Fin.mul_comm _ _
          _ = assignment head * left head + assignment head * right head :=
            Lean.Grind.Fin.left_distrib _ _ _
          _ = left head * assignment head + right head * assignment head := by
            rw [Fin.mul_comm (assignment head) (left head),
              Fin.mul_comm (assignment head) (right head)]
      rw [distributeHead]
      letI : Std.Associative (fun (left right : F) => left + right) :=
        ⟨baseLaws.add_assoc⟩
      letI : Std.Commutative (fun (left right : F) => left + right) :=
        ⟨baseLaws.add_comm⟩
      ac_rfl

@[simp] theorem evaluate_scale {columns : Nat}
    (coefficient : F) (form : Form columns)
    (assignment : Fin columns → F) :
    evaluate (scale coefficient form) assignment =
      coefficient * evaluate form assignment := by
  unfold evaluate
  induction canonicalFinIndices columns with
  | nil => exact (baseLaws.mul_zero coefficient).symm
  | cons head tail inductionHypothesis =>
      simp only [evaluateOn, scale_apply, inductionHypothesis]
      calc
        coefficient * form head * assignment head +
            coefficient * evaluateOn tail form assignment =
          coefficient * (form head * assignment head) +
            coefficient * evaluateOn tail form assignment :=
          congrArg
            (fun value => value + coefficient * evaluateOn tail form assignment)
            (Fin.mul_assoc _ _ _)
        _ = coefficient *
            (form head * assignment head +
              evaluateOn tail form assignment) :=
          (Lean.Grind.Fin.left_distrib _ _ _).symm

private theorem evaluateOn_absent_single {columns : Nat}
    (indices : List (Fin columns)) (selected : Fin columns)
    (coefficient : F) (assignment : Fin columns → F)
    (absent : selected ∉ indices) :
    evaluateOn indices (single selected coefficient) assignment = 0 := by
  induction indices with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      have headNe : head ≠ selected := by
        intro equal
        exact absent (by simp [equal])
      have absentTail : selected ∉ tail := by
        intro member
        exact absent (by simp [member])
      simp only [evaluateOn, single_apply, if_neg headNe, Fin.zero_mul,
        Fin.zero_add]
      exact inductionHypothesis absentTail

private theorem evaluateOn_single {columns : Nat}
    (indices : List (Fin columns)) (selected : Fin columns)
    (coefficient : F) (assignment : Fin columns → F)
    (nodup : indices.Nodup) (member : selected ∈ indices) :
    evaluateOn indices (single selected coefficient) assignment =
      coefficient * assignment selected := by
  induction indices with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      simp only [evaluateOn]
      by_cases headEq : head = selected
      · subst head
        rw [show single selected coefficient selected = coefficient by
          simp [single_apply]]
        rw [evaluateOn_absent_single tail selected coefficient assignment
          (List.nodup_cons.mp nodup).1]
        exact baseLaws.add_zero _
      · have memberTail : selected ∈ tail := by
          simpa [Ne.symm headEq] using member
        rw [show single selected coefficient head = 0 by
          simp [single_apply, headEq]]
        rw [Fin.zero_mul, Fin.zero_add]
        exact inductionHypothesis (List.nodup_cons.mp nodup).2 memberTail

@[simp] theorem evaluate_single {columns : Nat}
    (selected : Fin columns) (coefficient : F)
    (assignment : Fin columns → F) :
    evaluate (single selected coefficient) assignment =
      coefficient * assignment selected := by
  unfold evaluate
  exact evaluateOn_single (canonicalFinIndices columns) selected coefficient
    assignment (canonicalFinIndices_nodup columns)
    (by simp [canonicalFinIndices])

def ofTerms {columns : Nat} : List (DecodedTerm columns) → Form columns
  | [] => zero
  | term :: tail => add (single term.column term.coefficient) (ofTerms tail)

def ofPort {columns : Nat} (port : DecodedPort columns) : Form columns :=
  ofTerms port.terms

private def termSum {columns : Nat}
    (assignment : Fin columns → F) : List (DecodedTerm columns) → F
  | [] => 0
  | term :: tail =>
      term.coefficient * assignment term.column + termSum assignment tail

private theorem foldl_action_eq_initial_add_termSum {columns : Nat}
    (assignment : Fin columns → F) (terms : List (DecodedTerm columns))
    (initial : F) :
    terms.foldl
        (fun total term => total + term.coefficient * assignment term.column)
        initial =
      initial + termSum assignment terms := by
  induction terms generalizing initial with
  | nil => exact (baseLaws.add_zero initial).symm
  | cons head tail inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      simp only [termSum]
      exact baseLaws.add_assoc _ _ _

theorem evaluate_ofPort {columns : Nat} (port : DecodedPort columns)
    (assignment : Fin columns → F) :
    evaluate (ofPort port) assignment = action port assignment := by
  unfold ofPort action
  rw [foldl_action_eq_initial_add_termSum]
  rw [Fin.zero_add]
  induction port.terms with
  | nil => simp [ofTerms, termSum]
  | cons head tail inductionHypothesis =>
      simp only [ofTerms, evaluate_add, evaluate_single, termSum,
        inductionHypothesis]

end Form

/-- The compiler uses 41 balanced ternary coordinates for an input field,
23 centered-septenary coordinates for a general field, and binary
coordinates for every other retained width. -/
def slotRadix (width : Nat) : F :=
  if width = 41 then 3 else if width = 23 then 7 else 2

private def sourceSlotColumn {sourceColumns finalColumns : Nat}
    (slot : DecodedSourceSlot sourceColumns finalColumns)
    (index : Fin slot.width) : Fin finalColumns :=
  ⟨slot.start + index.val,
    Nat.lt_of_lt_of_le (Nat.add_lt_add_left index.isLt slot.start)
      slot.columnsFit⟩

private def derivedSlotColumn {finalColumns : Nat}
    (slot : DecodedDerivedSlot finalColumns)
    (index : Fin slot.width) : Fin finalColumns :=
  ⟨slot.start + index.val,
    Nat.lt_of_lt_of_le (Nat.add_lt_add_left index.isLt slot.start)
      slot.columnsFit⟩

def sourceSlotForm {sourceColumns finalColumns : Nat}
    (slot : DecodedSourceSlot sourceColumns finalColumns) : Form finalColumns :=
  (canonicalFinIndices slot.width).foldr
    (fun index tail =>
      Form.add
        (Form.single (sourceSlotColumn slot index)
          (slotRadix slot.width ^ index.val)) tail)
    Form.zero

private def sourceSlotValueFrom {sourceColumns finalColumns : Nat}
    (slot : DecodedSourceSlot sourceColumns finalColumns)
    (assignment : Fin finalColumns → F) : List (Fin slot.width) → F
  | [] => 0
  | index :: tail =>
      slotRadix slot.width ^ index.val *
          assignment (sourceSlotColumn slot index) +
        sourceSlotValueFrom slot assignment tail

/-- Sparse evaluation of one source slot. It traverses only the retained
low-norm coordinates, not the complete final assignment. -/
def sourceSlotValue {sourceColumns finalColumns : Nat}
    (slot : DecodedSourceSlot sourceColumns finalColumns)
    (assignment : Fin finalColumns → F) : F :=
  sourceSlotValueFrom slot assignment (canonicalFinIndices slot.width)

/-- The sparse source-slot value is the canonical finite-index fold exposed
without its private recursive helper. -/
theorem sourceSlotValue_eq_foldr {sourceColumns finalColumns : Nat}
    (slot : DecodedSourceSlot sourceColumns finalColumns)
    (assignment : Fin finalColumns → F) :
    sourceSlotValue slot assignment =
      (canonicalFinIndices slot.width).foldr
        (fun index tail =>
          slotRadix slot.width ^ index.val *
              assignment
                ⟨slot.start + index.val,
                  Nat.lt_of_lt_of_le
                    (Nat.add_lt_add_left index.isLt slot.start)
                    slot.columnsFit⟩ +
            tail)
        0 := by
  unfold sourceSlotValue
  induction canonicalFinIndices slot.width with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [sourceSlotValueFrom, List.foldr_cons,
        inductionHypothesis]
      rfl

private theorem evaluate_sourceSlotFormFrom
    {sourceColumns finalColumns : Nat}
    (slot : DecodedSourceSlot sourceColumns finalColumns)
    (assignment : Fin finalColumns → F)
    (indices : List (Fin slot.width)) :
    Form.evaluate
        (indices.foldr
          (fun index tail =>
            Form.add
              (Form.single (sourceSlotColumn slot index)
                (slotRadix slot.width ^ index.val)) tail)
          Form.zero)
        assignment =
      sourceSlotValueFrom slot assignment indices := by
  induction indices with
  | nil =>
      exact Form.evaluate_zero assignment
  | cons index tail inductionHypothesis =>
      rw [List.foldr_cons, sourceSlotValueFrom, Form.evaluate_add,
        Form.evaluate_single, inductionHypothesis]

/-- The sparse slot evaluator is exactly the dense linear-form action. -/
theorem evaluate_sourceSlotForm {sourceColumns finalColumns : Nat}
    (slot : DecodedSourceSlot sourceColumns finalColumns)
    (assignment : Fin finalColumns → F) :
    Form.evaluate (sourceSlotForm slot) assignment =
      sourceSlotValue slot assignment := by
  unfold sourceSlotForm sourceSlotValue
  exact evaluate_sourceSlotFormFrom slot assignment _

def derivedSlotForm {finalColumns : Nat}
    (slot : DecodedDerivedSlot finalColumns) : Form finalColumns :=
  (canonicalFinIndices slot.width).foldr
    (fun index tail =>
      Form.add
        (Form.single (derivedSlotColumn slot index)
          (slotRadix slot.width ^ index.val)) tail)
    Form.zero

def constantForm {columns : Nat} (columnsPositive : 0 < columns) :
    Form columns :=
  Form.single ⟨0, columnsPositive⟩ 1

theorem evaluate_constantForm {columns : Nat}
    (columnsPositive : 0 < columns) (assignment : Fin columns → F) :
    Form.evaluate (constantForm columnsPositive) assignment =
      assignment ⟨0, columnsPositive⟩ := by
  unfold constantForm
  rw [Form.evaluate_single]
  exact Fin.one_mul _

def findSourceSlot {sourceColumns finalColumns : Nat}
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (column : Fin sourceColumns) :
    Option (DecodedSourceSlot sourceColumns finalColumns) :=
  slots.find? fun slot => slot.column = column

def findSourceDefinition {sourceColumns : Nat}
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (column : Fin sourceColumns) :
    Option (DecodedSourceDefinition sourceColumns) :=
  definitions.find? fun definition => definition.target = column

def findDerivedSlot {finalColumns : Nat}
    (slots : List (DecodedDerivedSlot finalColumns))
    (compilerIndex : Nat) : Option (DecodedDerivedSlot finalColumns) :=
  slots.find? fun slot => slot.compilerIndex = compilerIndex

/-- Expand one source column. Fuel makes malformed cyclic artifact definitions
total. A concrete refinement must prove that every referenced source resolves
before it uses this value. -/
def sourceForm {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns)) :
    Nat → Fin sourceColumns → Form finalColumns
  | 0, _ => Form.zero
  | fuel + 1, column =>
      if column.val = 0 then
        constantForm columnsPositive
      else
        match findSourceDefinition definitions column with
        | some definition =>
            definition.value.terms.foldl
              (fun total term =>
                Form.add total
                  (Form.scale term.coefficient
                    (sourceForm columnsPositive slots definitions fuel
                      term.column)))
              (Form.scale definition.value.constant
                (constantForm columnsPositive))
        | none =>
            match findSourceSlot slots column with
            | some slot => sourceSlotForm slot
            | none => Form.zero

def sourceLinearForm {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) (value : DecodedSourceLinearCombination sourceColumns) :
    Form finalColumns :=
  value.terms.foldl
    (fun total term =>
      Form.add total
        (Form.scale term.coefficient
          (sourceForm columnsPositive slots definitions fuel term.column)))
    (Form.scale value.constant (constantForm columnsPositive))

/-- Source assignment decoded from the exact final low-norm images. -/
def decodedSourceAssignment {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) (assignment : Fin finalColumns → F) :
    Fin sourceColumns → F :=
  fun column =>
    Form.evaluate
      (sourceForm columnsPositive slots definitions fuel column) assignment

/-- Direct source-linear-combination evaluation on the decoded assignment.
The constant uses the final constant-one coordinate explicitly. -/
def sourceLinearValue {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) (value : DecodedSourceLinearCombination sourceColumns)
    (assignment : Fin finalColumns → F) : F :=
  value.terms.foldl
    (fun total term =>
      total + term.coefficient *
        decodedSourceAssignment columnsPositive slots definitions fuel
          assignment term.column)
    (value.constant * assignment ⟨0, columnsPositive⟩)

/-- Direct evaluation of one decoded source linear combination on an
independent source assignment. The constant wire remains explicit because
the final and source assignments have different finite types. -/
def directSourceLinearValue {sourceColumns : Nat}
    (value : DecodedSourceLinearCombination sourceColumns)
    (sourceAssignment : Fin sourceColumns → F) (constantWire : F) : F :=
  value.terms.foldl
    (fun total term =>
      total + term.coefficient * sourceAssignment term.column)
    (value.constant * constantWire)

private theorem sourceTerms_foldl_congr
    {sourceColumns : Nat}
    (terms : List (DecodedSourceTerm sourceColumns))
    (left right : Fin sourceColumns → F)
    (agree : ∀ term ∈ terms, left term.column = right term.column)
    (initial : F) :
    terms.foldl
        (fun total term => total + term.coefficient * left term.column)
        initial =
      terms.foldl
        (fun total term => total + term.coefficient * right term.column)
        initial := by
  induction terms generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      rw [List.foldl_cons, List.foldl_cons]
      rw [agree head (by simp)]
      exact inductionHypothesis
        (fun term member => agree term (by simp [member])) _

/-- If an independent source assignment agrees with the decoded final image
on every referenced source column, both evaluations use the same values. This
is the assignment bridge required by a concrete encoder refinement. -/
theorem sourceLinearValue_eq_direct_of_agreement
    {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) (value : DecodedSourceLinearCombination sourceColumns)
    (finalAssignment : Fin finalColumns → F)
    (sourceAssignment : Fin sourceColumns → F)
    (agree : ∀ term ∈ value.terms,
      decodedSourceAssignment columnsPositive slots definitions fuel
          finalAssignment term.column = sourceAssignment term.column) :
    sourceLinearValue columnsPositive slots definitions fuel value
        finalAssignment =
      directSourceLinearValue value sourceAssignment
        (finalAssignment ⟨0, columnsPositive⟩) := by
  unfold sourceLinearValue directSourceLinearValue
  exact sourceTerms_foldl_congr value.terms
    (decodedSourceAssignment columnsPositive slots definitions fuel
      finalAssignment)
    sourceAssignment agree _

private theorem evaluate_sourceTerms_foldl
    {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) (assignment : Fin finalColumns → F)
    (terms : List (DecodedSourceTerm sourceColumns))
    (initial : Form finalColumns) :
    Form.evaluate
        (terms.foldl
          (fun total term =>
            Form.add total
              (Form.scale term.coefficient
                (sourceForm columnsPositive slots definitions fuel
                  term.column)))
          initial)
        assignment =
      terms.foldl
        (fun total term =>
          total + term.coefficient *
            decodedSourceAssignment columnsPositive slots definitions fuel
              assignment term.column)
        (Form.evaluate initial assignment) := by
  induction terms generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      rw [List.foldl_cons, List.foldl_cons, inductionHypothesis]
      simp [decodedSourceAssignment]

/-- Expanding a decoded source linear combination and evaluating its final
image gives exactly the same value as evaluating that source combination on
the decoded source assignment. -/
theorem evaluate_sourceLinearForm
    {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) (value : DecodedSourceLinearCombination sourceColumns)
    (assignment : Fin finalColumns → F) :
    Form.evaluate
        (sourceLinearForm columnsPositive slots definitions fuel value)
        assignment =
      sourceLinearValue columnsPositive slots definitions fuel value
        assignment := by
  unfold sourceLinearForm sourceLinearValue
  rw [evaluate_sourceTerms_foldl]
  simp only [Form.evaluate_scale, constantForm, Form.evaluate_single]
  exact
    congrArg
      (fun initial =>
        value.terms.foldl
          (fun total term =>
            total + term.coefficient *
              decodedSourceAssignment columnsPositive slots definitions fuel
                assignment term.column)
          initial)
      (congrArg (fun decoded => value.constant * decoded) (Fin.one_mul _))

def outputForm {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot finalColumns))
    (fuel : Nat) : DecodedOutput sourceColumns → Form finalColumns
  | .source value =>
      sourceLinearForm columnsPositive slots definitions fuel value
  | .derivedProductSum compilerIndex =>
      match findDerivedSlot derived compilerIndex with
      | some slot => derivedSlotForm slot
      | none => Form.zero

def previousForm {finalColumns : Nat}
    (derived : List (DecodedDerivedSlot finalColumns)) :
    Option Nat → Form finalColumns
  | none => Form.zero
  | some compilerIndex =>
      match findDerivedSlot derived compilerIndex with
      | some slot => derivedSlotForm slot
      | none => Form.zero

def factorLeftForm {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) (factor : DecodedFactor sourceColumns) : Form finalColumns :=
  Form.scale factor.coefficient
    (sourceLinearForm columnsPositive slots definitions fuel factor.left)

def factorRightForm {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) (factor : DecodedFactor sourceColumns) : Form finalColumns :=
  sourceLinearForm columnsPositive slots definitions fuel factor.right

def factorFormAt {rows sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel index : Nat) (left : Bool)
    (step : DecodedStep rows sourceColumns) : Form finalColumns :=
  match step.factors[index]? with
  | some factor =>
      if left then
        factorLeftForm columnsPositive slots definitions fuel factor
      else
        factorRightForm columnsPositive slots definitions fuel factor
  | none => Form.zero

def expectedCForm {rows sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot finalColumns))
    (fuel : Nat) (step : DecodedStep rows sourceColumns) : Form finalColumns :=
  Form.sub
    (Form.sub
      (outputForm columnsPositive slots definitions derived fuel step.output)
      (sourceLinearForm columnsPositive slots definitions fuel step.base))
    (previousForm derived step.previous)

def factorRolePairs : List (Role × Role) :=
  [(.bit, .a), (.b, .sboxInput), (.centeredUnit, .canonicalDigit),
    (.canonicalBorrow, .canonicalNextBorrow),
    (.canonicalBoundDigit, .evalTailRight)]

def factorRoles (index : Fin 5) : Role × Role :=
  factorRolePairs.get ⟨index.val, by simp [factorRolePairs]⟩

def PortImagesMatch {rows sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (ports : Fin 13 → DecodedPort finalColumns)
    (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot finalColumns))
    (fuel : Nat) : Prop :=
  Form.Equivalent
      (Form.ofPort (ports Role.c.index))
      (expectedCForm columnsPositive slots definitions derived fuel step) ∧
    ∀ index : Fin 5,
      Form.Equivalent
          (Form.ofPort (ports (factorRoles index).1.index))
          (factorFormAt columnsPositive slots definitions fuel index.val
            true step) ∧
        Form.Equivalent
          (Form.ofPort (ports (factorRoles index).2.index))
          (factorFormAt columnsPositive slots definitions fuel index.val
            false step)

instance {rows sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (ports : Fin 13 → DecodedPort finalColumns)
    (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot finalColumns))
    (fuel : Nat) :
    Decidable
      (PortImagesMatch columnsPositive ports step slots definitions derived
        fuel) := by
  unfold PortImagesMatch
  infer_instance

def StepImagesMatch {rows sourceColumns : Nat}
    (row : DecodedRow) (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns row.columns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot row.columns))
    (fuel : Nat) : Prop :=
  PortImagesMatch row.columnsPositive row.ports step slots definitions derived
    fuel

instance {rows sourceColumns : Nat}
    (row : DecodedRow) (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns row.columns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot row.columns))
    (fuel : Nat) :
    Decidable (StepImagesMatch row step slots definitions derived fuel) := by
  unfold StepImagesMatch
  infer_instance

theorem matched_port_c_action {rows sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (ports : Fin 13 → DecodedPort finalColumns)
    (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot finalColumns))
    (fuel : Nat)
    (imageMatch :
      PortImagesMatch columnsPositive ports step slots definitions derived
        fuel)
    (assignment : Fin finalColumns → F) :
    action (ports Role.c.index) assignment =
      Form.evaluate
        (expectedCForm columnsPositive slots definitions derived fuel step)
        assignment := by
  rw [← Form.evaluate_ofPort]
  exact Form.evaluate_congr imageMatch.1 assignment

theorem matched_port_factor_actions
    {rows sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (ports : Fin 13 → DecodedPort finalColumns)
    (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot finalColumns))
    (fuel : Nat)
    (imageMatch :
      PortImagesMatch columnsPositive ports step slots definitions derived
        fuel)
    (index : Fin 5) (assignment : Fin finalColumns → F) :
    action (ports (factorRoles index).1.index) assignment =
        Form.evaluate
          (factorFormAt columnsPositive slots definitions fuel index.val
            true step) assignment ∧
      action (ports (factorRoles index).2.index) assignment =
        Form.evaluate
          (factorFormAt columnsPositive slots definitions fuel index.val
            false step) assignment := by
  constructor
  · rw [← Form.evaluate_ofPort]
    exact Form.evaluate_congr (imageMatch.2 index).1 assignment
  · rw [← Form.evaluate_ofPort]
    exact Form.evaluate_congr (imageMatch.2 index).2 assignment

/-- Matched final factor ports evaluate the exact decoded source linear
combinations. This removes coefficient-image equality as an unproved step
between a generated row and its source recurrence. -/
theorem matched_port_factor_actions_eq_source_values
    {rows sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (ports : Fin 13 → DecodedPort finalColumns)
    (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot finalColumns))
    (fuel : Nat)
    (imageMatch :
      PortImagesMatch columnsPositive ports step slots definitions derived
        fuel)
    (index : Fin 5) (factor : DecodedFactor sourceColumns)
    (factorAt : step.factors[index.val]? = some factor)
    (assignment : Fin finalColumns → F) :
    action (ports (factorRoles index).1.index) assignment =
        factor.coefficient *
          sourceLinearValue columnsPositive slots definitions fuel factor.left
            assignment ∧
      action (ports (factorRoles index).2.index) assignment =
        sourceLinearValue columnsPositive slots definitions fuel factor.right
          assignment := by
  have actions := matched_port_factor_actions columnsPositive ports step slots
    definitions derived fuel imageMatch index assignment
  constructor
  · rw [actions.1]
    simp only [factorFormAt, factorAt, if_pos, factorLeftForm,
      Form.evaluate_scale]
    rw [evaluate_sourceLinearForm]
  · rw [actions.2]
    simp [factorFormAt, factorAt, factorRightForm,
      evaluate_sourceLinearForm]

theorem matched_c_action {rows sourceColumns : Nat}
    (row : DecodedRow) (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns row.columns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot row.columns))
    (fuel : Nat)
    (imageMatch : StepImagesMatch row step slots definitions derived fuel)
    (assignment : Fin row.columns → F) :
    action (row.port Role.c.index) assignment =
      Form.evaluate
        (expectedCForm row.columnsPositive slots definitions derived fuel step)
        assignment := by
  exact matched_port_c_action row.columnsPositive row.ports step slots
    definitions derived fuel imageMatch assignment

theorem matched_factor_actions {rows sourceColumns : Nat}
    (row : DecodedRow) (step : DecodedStep rows sourceColumns)
    (slots : List (DecodedSourceSlot sourceColumns row.columns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot row.columns))
    (fuel : Nat)
    (imageMatch : StepImagesMatch row step slots definitions derived fuel)
    (index : Fin 5) (assignment : Fin row.columns → F) :
    action (row.port (factorRoles index).1.index) assignment =
        Form.evaluate
          (factorFormAt row.columnsPositive slots definitions fuel index.val
            true step) assignment ∧
      action (row.port (factorRoles index).2.index) assignment =
        Form.evaluate
          (factorFormAt row.columnsPositive slots definitions fuel index.val
            false step) assignment := by
  exact matched_port_factor_actions row.columnsPositive row.ports step slots
    definitions derived fuel imageMatch index assignment

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
