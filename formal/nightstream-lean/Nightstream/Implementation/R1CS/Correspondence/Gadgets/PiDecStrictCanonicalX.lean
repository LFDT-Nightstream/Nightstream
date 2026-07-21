import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictSound
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits

/-!
Model-level R1CS compiler for canonical production `PiDEC` public digits.

Protocol: SuperNeo `Pi_DEC` at production radix two and fourteen children.
Phase: one verifier-owned parent public coordinate and its child digits.
Constraint family: the retained radix recomposition row plus a common-sign
canonicality block.

Owns: a column-parametric 17-row program; row soundness to the independent
`Radix.UniformSignedDigits.Accepted` predicate; honest completeness for
`Radix.splitScalar`; and exact model-level row counts.

Does not own: placement in the production layout, Rust emission, generated
artifact identity, the parent strict-bound constraint, whole-public-input
coverage, or authorization to delete production rows.

Emits constraints: seventeen rows per parent coordinate: one retained radix
recomposition row, two rows for the explicit `signColumn` and its explicit
`signOutputColumn` multiplication auxiliary, and fourteen digit-selector rows.

Assurance tier: model-level. Theorems consume exact row satisfaction and
canonical Goldilocks residues. No generated row or measured count is semantic
authority.

The canonicality block has sixteen rows: two for one centered-unit sign and
one quadratic selector row for each of fourteen digits. The unchanged radix
recomposition row makes seventeen total. The current independent centered-unit
encoding uses twenty-eight canonicality rows, or twenty-nine with the same
recomposition row.

| Stage path | Equation | Rows | Lean owner |
|---|---|---:|---|
| `nifs.pi_dec.public_x.recompose` | `parent = Σ 2^i digit_i` | 1 | `recompositionInstruction` |
| `nifs.pi_dec.public_x.sign` | `sign ∈ {0,1,-1}` | 2 | `centeredUnitInstructions` |
| `nifs.pi_dec.public_x.digits` | `digit_i(digit_i-sign)=0` | 14 | `digitInstructions` |
| `nifs.pi_dec.public_x.exact` | `digits = splitScalar parent` | 0 | `rows_force_splitScalar` |
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictCanonicalX

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.PiDecStrictCompiler

/-- Columns used by one public-coordinate canonicality gadget. The compiler is
column-parametric so later artifact refinement can supply production columns
without changing its semantics. -/
structure Layout where
  parentColumn : Nat
  signColumn : Nat
  signOutputColumn : Nat
  digitColumns : ChildIndex -> Nat

def childColumns (layout : Layout) : List Nat :=
  List.ofFn layout.digitColumns

def powers : List Nat :=
  radixPowers 2 productionGlobalParams.k

def recompositionInstruction (layout : Layout) : Instruction :=
  recompositionCheck layout.parentColumn (childColumns layout) powers

/-- One R1CS row imposing `digit * (digit - sign) = 0`. -/
def digitInstruction (layout : Layout) (index : ChildIndex) : Instruction :=
  .check <| {
    a := [(layout.digitColumns index, 1)]
    b := [(layout.digitColumns index, 1),
      (layout.signColumn, goldilocksP - 1)]
    c := []
  }

def digitInstructions (layout : Layout) : List Instruction :=
  List.ofFn (digitInstruction layout)

/-- The new sixteen-row canonicality family, excluding the retained
recomposition row. -/
def canonicalityInstructions (layout : Layout) : List Instruction :=
  centeredUnitInstructions layout.signColumn layout.signOutputColumn ++
    digitInstructions layout

/-- Complete one-coordinate program. -/
def instructions (layout : Layout) : List Instruction :=
  recompositionInstruction layout :: canonicalityInstructions layout

def rows (layout : Layout) : List Row :=
  CheckedProgram.rows (instructions layout)

def decodedParent (layout : Layout) (assignment : Nat -> Nat) : F :=
  fieldOfNat (assignment layout.parentColumn)

def decodedSign (layout : Layout) (assignment : Nat -> Nat) : F :=
  fieldOfNat (assignment layout.signColumn)

def decodedDigits (layout : Layout) (assignment : Nat -> Nat) :
    ChildIndex -> F :=
  fun index => fieldOfNat (assignment (layout.digitColumns index))

theorem powers_canonical :
    forall coefficient, coefficient ∈ powers ->
      0 < coefficient /\ coefficient < goldilocksP := by
  decide

private theorem satisfies_left
    {left right : List Instruction} {assignment : Nat -> Nat}
    (satisfies : Satisfies (CheckedProgram.rows (left ++ right)) assignment) :
    Satisfies (CheckedProgram.rows left) assignment := by
  intro row rowMember
  apply satisfies row
  simpa [CheckedProgram.rows] using
    List.mem_append_left (CheckedProgram.rows right) rowMember

private theorem satisfies_right
    {left right : List Instruction} {assignment : Nat -> Nat}
    (satisfies : Satisfies (CheckedProgram.rows (left ++ right)) assignment) :
    Satisfies (CheckedProgram.rows right) assignment := by
  intro row rowMember
  apply satisfies row
  simpa [CheckedProgram.rows] using
    List.mem_append_right (CheckedProgram.rows left) rowMember

/-! ## The common-sign rows -/

/-- The quadratic selector row forces one digit to be zero or the common sign.
Canonical representatives are essential when converting modular equality back
to equality of natural representatives. -/
theorem digitInstruction_sound
    (prime : EuclidPrime goldilocksP)
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (index : ChildIndex)
    (holds : RowHolds assignment (digitInstruction layout index).row) :
    assignment (layout.digitColumns index) = 0 \/
      assignment (layout.digitColumns index) =
        assignment layout.signColumn := by
  have equation :
      assignment (layout.digitColumns index) *
          ((assignment (layout.digitColumns index) +
              (goldilocksP - 1) * assignment layout.signColumn) %
            goldilocksP) % goldilocksP = 0 := by
    simpa [digitInstruction, Instruction.row, RowHolds, lcEval,
      Nat.mod_eq_of_lt (canonical (layout.digitColumns index))] using holds
  rcases prime _ _ equation with digitZero | differenceZero
  · left
    have digitLt := canonical (layout.digitColumns index)
    simp only [goldilocksP] at digitZero digitLt |-
    omega
  · right
    have digitLt := canonical (layout.digitColumns index)
    have signLt := canonical layout.signColumn
    simp only [goldilocksP] at differenceZero digitLt signLt |-
    omega

theorem digitInstruction_complete
    {layout : Layout} {assignment : Nat -> Nat}
    (index : ChildIndex)
    (accepted : assignment (layout.digitColumns index) = 0 \/
      assignment (layout.digitColumns index) =
        assignment layout.signColumn) :
    RowHolds assignment (digitInstruction layout index).row := by
  rcases accepted with digitZero | digitSign
  · simp [digitInstruction, Instruction.row, RowHolds, lcEval, digitZero]
  · have differenceZero :
        (assignment (layout.digitColumns index) +
            (goldilocksP - 1) * assignment layout.signColumn) %
          goldilocksP = 0 := by
      rw [digitSign]
      have factor :
          assignment layout.signColumn +
              (goldilocksP - 1) * assignment layout.signColumn =
            goldilocksP * assignment layout.signColumn := by
        change assignment layout.signColumn +
            18446744069414584320 * assignment layout.signColumn =
          (18446744069414584320 + 1) * assignment layout.signColumn
        rw [Nat.add_mul]
        omega
      rw [factor]
      simp
    simp [digitInstruction, Instruction.row, RowHolds, lcEval,
      differenceZero]

private theorem centeredInstruction_complete
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (definitionHolds : Definition.Holds assignment
      { output := layout.signOutputColumn
        rhs := .product
          [(layout.signColumn, 1), (0, 1)]
          [(layout.signColumn, 1)] })
    (centered : PiDecStrictCompiler.CenteredUnit
      (assignment layout.signColumn)) :
    Satisfies
      (CheckedProgram.rows (centeredUnitInstructions
        layout.signColumn layout.signOutputColumn)) assignment := by
  intro row rowMember
  simp only [centeredUnitInstructions, CheckedProgram.rows, List.map_cons,
    List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at rowMember
  rcases rowMember with rfl | rfl
  · exact builderDefinition_complete canonical one _ (by trivial)
      definitionHolds
  · exact Nightstream.Implementation.R1CS.PiDecStrictSound.centeredUnitCheck_complete
      one layout.signColumn layout.signOutputColumn definitionHolds centered

/-! ## Semantic decoding -/

private theorem fieldOfNat_mod (value : Nat) :
    fieldOfNat (value % goldilocksP) = fieldOfNat value := by
  apply Fin.ext
  simp [fieldOfNat, goldilocksP, goldilocksModulus]

private theorem fieldOfNat_rawLcEval (assignment : Nat -> Nat) :
    forall terms : List (Nat × Nat),
      fieldOfNat (rawLcEval assignment terms) =
        terms.foldr (fun term suffix =>
          fieldOfNat term.2 * fieldOfNat (assignment term.1) + suffix) 0 := by
  intro terms
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [rawLcEval, List.foldr_cons]
      rw [fieldOfNat_add, fieldOfNat_mul, inductionHypothesis]

private theorem fieldOfNat_lcEval
    (assignment : Nat -> Nat) (terms : List (Nat × Nat)) :
    fieldOfNat (lcEval assignment terms) =
      terms.foldr (fun term suffix =>
        fieldOfNat term.2 * fieldOfNat (assignment term.1) + suffix) 0 := by
  rw [lcEval_eq_raw_mod, fieldOfNat_mod]
  exact fieldOfNat_rawLcEval assignment terms

private theorem decoded_lcEval_eq_recomposeScalar
    (layout : Layout) (assignment : Nat -> Nat) :
    fieldOfNat (lcEval assignment ((childColumns layout).zip powers)) =
      recomposeScalar (decodedDigits layout assignment) := by
  rw [fieldOfNat_lcEval]
  have range14 : List.range 14 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] := by
    decide
  rw [← recomposeScalarList_eq]
  simp [recomposeScalarList, decodedDigits, childColumns, powers,
    productionGlobalParams, radixPowers, range14,
    Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight,
    fieldOfNat, goldilocksP, goldilocksModulus]

/-- The retained implementation recomposition equation decodes to the exact
independent radix equation. -/
theorem decodedRecomposition_of_recomposes
    {layout : Layout} {assignment : Nat -> Nat}
    (recomposes : Recomposes assignment layout.parentColumn
      (childColumns layout) powers) :
    recomposeScalar (decodedDigits layout assignment) =
      decodedParent layout assignment := by
  unfold Recomposes at recomposes
  unfold decodedParent
  rw [recomposes]
  symm
  exact decoded_lcEval_eq_recomposeScalar layout assignment

private theorem fieldOfNat_minus_one :
    fieldOfNat (goldilocksP - 1) = (-1 : F) := by
  decide

private theorem field_one_val : (1 : F).val = 1 := by
  decide

private theorem field_minus_one_val : (-1 : F).val = goldilocksP - 1 := by
  decide

private theorem centered_decodes
    {layout : Layout} {assignment : Nat -> Nat}
    (centered : PiDecStrictCompiler.CenteredUnit
      (assignment layout.signColumn)) :
    SignAllowed (decodedSign layout assignment) := by
  rcases centered with zero | one | minusOne
  · left
    simp [decodedSign, zero]
  · right; left
    simp [decodedSign, one]
  · right; right
    simpa [decodedSign, minusOne] using fieldOfNat_minus_one

private theorem digit_decodes
    {layout : Layout} {assignment : Nat -> Nat} (index : ChildIndex)
    (accepted : assignment (layout.digitColumns index) = 0 \/
      assignment (layout.digitColumns index) =
        assignment layout.signColumn) :
    decodedDigits layout assignment index = 0 \/
      decodedDigits layout assignment index = decodedSign layout assignment := by
  rcases accepted with zero | signed
  · left
    simp [decodedDigits, zero]
  · right
    simp [decodedDigits, decodedSign, signed]

/-- Exact canonicality rows imply the independent common-sign predicate. -/
theorem canonicality_sound
    (prime : EuclidPrime goldilocksP)
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      (CheckedProgram.rows (canonicalityInstructions layout)) assignment) :
    ConstraintPredicate (decodedSign layout assignment)
      (decodedDigits layout assignment) := by
  have signRows : Satisfies
      (CheckedProgram.rows (centeredUnitInstructions
        layout.signColumn layout.signOutputColumn)) assignment :=
    satisfies_left satisfies
  have digitRows : Satisfies
      (CheckedProgram.rows (digitInstructions layout)) assignment :=
    satisfies_right satisfies
  have centered :=
    Nightstream.Implementation.R1CS.PiDecStrictSound.centeredUnitInstructions_sound
      prime canonical one layout.signColumn layout.signOutputColumn signRows
  constructor
  · exact centered_decodes centered
  · intro index
    apply digit_decodes index
    apply digitInstruction_sound prime canonical index
    apply Nightstream.Implementation.R1CS.PiDecStrictSound.instruction_holds
      digitRows
    simp [digitInstructions]

/-- The full seventeen-row program implies the independent accepted predicate.
The parent strict bound is then derivable from the common-sign roots and
recomposition; it is not a separate row-soundness premise. -/
theorem rows_sound
    (prime : EuclidPrime goldilocksP)
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    Accepted (decodedParent layout assignment)
      (decodedSign layout assignment) (decodedDigits layout assignment) := by
  have recompositionRow : RowHolds assignment
      (recompositionInstruction layout).row := by
    apply satisfies _
    simp [rows, instructions, CheckedProgram.rows]
  have canonicalityRows : Satisfies
      (CheckedProgram.rows (canonicalityInstructions layout)) assignment := by
    intro row rowMember
    apply satisfies row
    simpa [rows, instructions, CheckedProgram.rows] using
      List.mem_cons_of_mem (recompositionInstruction layout).row rowMember
  have recomposes :=
    Nightstream.Implementation.R1CS.PiDecStrictSound.recompositionCheck_sound
      canonical one powers_canonical recompositionRow
  exact {
    constraint := canonicality_sound prime canonical one canonicalityRows
    recomposition := decodedRecomposition_of_recomposes recomposes
  }

/-- Row soundness plus the independent strict parent bound forces the exact
verifier-computed child digits. -/
theorem rows_force_splitScalar
    (prime : EuclidPrime goldilocksP)
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    decodedDigits layout assignment =
      splitScalar (decodedParent layout assignment) := by
  exact (rows_sound prime canonical one satisfies).digits_eq_splitScalar

/-! ## Honest materialization and completeness -/

/-- Sign written by the production honest witness materializer. The semantic
predicate permits zero, positive, and negative signs. For a nonzero canonical
split this agrees with `honestSign`; for the all-zero split it uses zero, as
the Rust materializer does when it finds no nonzero child digit. -/
def materializedSign (parent : F) : F :=
  if parent = 0 then 0 else honestSign parent

theorem materializedSign_zero : materializedSign (0 : F) = 0 := by
  simp [materializedSign]

theorem materializedSign_of_ne {parent : F} (nonzero : parent ≠ 0) :
    materializedSign parent = honestSign parent := by
  simp [materializedSign, nonzero]

/-- The Rust-facing honest sign convention is still accepted by the
row-independent common-sign semantics. This changes only witness
materialization; it does not weaken the accepted-root predicate. -/
theorem materializedSign_complete (parent : F)
    (bounded : centeredMagnitude parent < combinedBound) :
    Accepted parent (materializedSign parent) (splitScalar parent) := by
  by_cases zero : parent = 0
  · subst parent
    constructor
    · constructor
      · exact Or.inl materializedSign_zero
      · intro index
        left
        simp [splitScalar, combinedBound, productionGlobalParams,
          Nightstream.SuperNeo.GlobalParams.bigB, boundedDigit, isNonnegative,
          magnitudeDigit,
          natBit, centeredMagnitude]
    · exact splitScalar_recompose 0
  · simpa [materializedSign, zero] using honest_complete parent bounded

/-- Exact values that an honest witness materializer writes into this local
gadget. `signDefinition` is the one deterministic auxiliary multiplication
wire used by the existing centered-unit compiler. -/
structure HonestMaterialization
    (layout : Layout) (assignment : Nat -> Nat) (parent : F) : Prop where
  canonical : forall column, assignment column < goldilocksP
  one : assignment 0 = 1
  parentValue : assignment layout.parentColumn = parent.val
  signValue : assignment layout.signColumn = (materializedSign parent).val
  digitValues : forall index,
    assignment (layout.digitColumns index) = (splitScalar parent index).val
  signDefinition : Definition.Holds assignment
    { output := layout.signOutputColumn
      rhs := .product
        [(layout.signColumn, 1), (0, 1)]
        [(layout.signColumn, 1)] }

private theorem fieldOfNat_value (value : F) : fieldOfNat value.val = value := by
  apply Fin.ext
  simp [fieldOfNat, Nat.mod_eq_of_lt value.isLt]

private theorem materialized_parent
    {layout : Layout} {assignment : Nat -> Nat} {parent : F}
    (materialized : HonestMaterialization layout assignment parent) :
    decodedParent layout assignment = parent := by
  simp [decodedParent, materialized.parentValue, fieldOfNat_value]

private theorem materialized_sign
    {layout : Layout} {assignment : Nat -> Nat} {parent : F}
    (materialized : HonestMaterialization layout assignment parent) :
    decodedSign layout assignment = materializedSign parent := by
  simp [decodedSign, materialized.signValue, fieldOfNat_value]

private theorem materialized_digits
    {layout : Layout} {assignment : Nat -> Nat} {parent : F}
    (materialized : HonestMaterialization layout assignment parent) :
    decodedDigits layout assignment = splitScalar parent := by
  funext index
  simp [decodedDigits, materialized.digitValues, fieldOfNat_value]

private theorem centered_of_materialized
    {layout : Layout} {assignment : Nat -> Nat} {parent : F}
    (materialized : HonestMaterialization layout assignment parent) :
    PiDecStrictCompiler.CenteredUnit (assignment layout.signColumn) := by
  rw [materialized.signValue]
  by_cases zero : parent = 0
  · left
    simp [materializedSign, zero]
  · by_cases nonnegative : isNonnegative parent
    · right; left
      simpa [materializedSign, zero, honestSign, nonnegative] using
        field_one_val
    · right; right
      simpa [materializedSign, zero, honestSign, nonnegative] using
        field_minus_one_val

private theorem digit_of_materialized
    {layout : Layout} {assignment : Nat -> Nat} {parent : F}
    (bounded : centeredMagnitude parent < combinedBound)
    (materialized : HonestMaterialization layout assignment parent)
    (index : ChildIndex) :
    assignment (layout.digitColumns index) = 0 \/
      assignment (layout.digitColumns index) =
        assignment layout.signColumn := by
  have honest := materializedSign_complete parent bounded
  rcases honest.constraint.2 index with zero | signed
  · left
    rw [materialized.digitValues]
    exact congrArg Fin.val zero
  · right
    rw [materialized.digitValues, materialized.signValue]
    exact congrArg Fin.val signed

private theorem materialized_recomposes
    {layout : Layout} {assignment : Nat -> Nat} {parent : F}
    (materialized : HonestMaterialization layout assignment parent) :
    Recomposes assignment layout.parentColumn (childColumns layout) powers := by
  unfold Recomposes
  have fieldEquation :
      fieldOfNat (lcEval assignment ((childColumns layout).zip powers)) =
        fieldOfNat (assignment layout.parentColumn) := by
    calc
      fieldOfNat (lcEval assignment ((childColumns layout).zip powers)) =
          recomposeScalar (decodedDigits layout assignment) :=
        decoded_lcEval_eq_recomposeScalar layout assignment
      _ = parent := by
        rw [materialized_digits materialized]
        exact splitScalar_recompose parent
      _ = fieldOfNat (assignment layout.parentColumn) := by
        rw [materialized.parentValue, fieldOfNat_value]
  have values := congrArg Fin.val fieldEquation
  have lcLt :
      lcEval assignment ((childColumns layout).zip powers) < goldilocksP := by
    unfold lcEval
    exact Nat.mod_lt _ (by decide)
  have parentLt := materialized.canonical layout.parentColumn
  have lcLtModulus :
      lcEval assignment ((childColumns layout).zip powers) <
        goldilocksModulus := by
    simpa [goldilocksP, goldilocksModulus] using lcLt
  have parentLtModulus :
      assignment layout.parentColumn < goldilocksModulus := by
    simpa [goldilocksP, goldilocksModulus] using parentLt
  change lcEval assignment ((childColumns layout).zip powers) %
      goldilocksModulus =
    assignment layout.parentColumn % goldilocksModulus at values
  rw [Nat.mod_eq_of_lt lcLtModulus,
    Nat.mod_eq_of_lt parentLtModulus] at values
  exact values.symm

/-- Honest `splitScalar` values and the one deterministic sign auxiliary
materialize a satisfying witness for all seventeen rows. -/
theorem honest_complete_rows
    {layout : Layout} {assignment : Nat -> Nat} {parent : F}
    (bounded : centeredMagnitude parent < combinedBound)
    (materialized : HonestMaterialization layout assignment parent) :
    Satisfies (rows layout) assignment := by
  have recompositionHolds : RowHolds assignment
      (recompositionInstruction layout).row :=
    Nightstream.Implementation.R1CS.PiDecStrictSound.recompositionCheck_complete
      materialized.one powers_canonical
      (materialized_recomposes materialized)
  have centeredHolds := centeredInstruction_complete materialized.canonical
    materialized.one materialized.signDefinition
    (centered_of_materialized materialized)
  have digitHolds : Satisfies
      (CheckedProgram.rows (digitInstructions layout)) assignment := by
    intro row rowMember
    rcases List.mem_map.mp rowMember with
      ⟨instruction, instructionMember, rfl⟩
    rcases List.mem_ofFn.mp instructionMember with ⟨index, rfl⟩
    exact digitInstruction_complete index
      (digit_of_materialized bounded materialized index)
  intro row rowMember
  simp only [rows, instructions, CheckedProgram.rows, List.map_cons,
    List.mem_cons] at rowMember
  rcases rowMember with rfl | rowMember
  · exact recompositionHolds
  · have canonicalityHolds : Satisfies
        (CheckedProgram.rows (canonicalityInstructions layout)) assignment := by
      simpa [canonicalityInstructions, CheckedProgram.rows] using
        Nightstream.Implementation.R1CS.PiDecStrictSound.satisfies_append
          centeredHolds digitHolds
    exact canonicalityHolds row rowMember

/-! ## Exact model-level cost -/

def canonicalityRowCount : Nat := 16
def totalRowCount : Nat := 17
def currentIndependentAlphabetRowCount : Nat := 28
def currentTotalRowCount : Nat := 29
def rowsSavedPerCoordinate : Nat := 12

/-- Existing strict compiler schedule for the same fourteen digit columns.
The starting output column is irrelevant to the row census. -/
def currentIndependentAlphabetInstructions (layout : Layout) :
    List Instruction :=
  alphabetFrom layout.signOutputColumn (childColumns layout)

def currentInstructions (layout : Layout) : List Instruction :=
  recompositionInstruction layout ::
    currentIndependentAlphabetInstructions layout

theorem canonicality_rows_exact (layout : Layout) :
    (CheckedProgram.rows (canonicalityInstructions layout)).length =
      canonicalityRowCount := by
  change (centeredUnitInstructions layout.signColumn layout.signOutputColumn ++
    digitInstructions layout).length = canonicalityRowCount
  rw [List.length_append]
  simp [centeredUnitInstructions, digitInstructions, canonicalityRowCount,
    productionGlobalParams]

theorem total_rows_exact (layout : Layout) :
    (rows layout).length = totalRowCount := by
  have canonicalityLength :
      (canonicalityInstructions layout).length = canonicalityRowCount := by
    simpa [CheckedProgram.rows] using canonicality_rows_exact layout
  simp [rows, instructions, CheckedProgram.rows, canonicalityLength,
    canonicalityRowCount, totalRowCount]

theorem current_independent_alphabet_rows_exact (layout : Layout) :
    (CheckedProgram.rows
      (currentIndependentAlphabetInstructions layout)).length =
        currentIndependentAlphabetRowCount := by
  simp [currentIndependentAlphabetInstructions, childColumns, alphabetFrom,
    centeredUnitInstructions, CheckedProgram.rows, productionGlobalParams,
    currentIndependentAlphabetRowCount]

theorem current_total_rows_exact (layout : Layout) :
    (CheckedProgram.rows (currentInstructions layout)).length =
      currentTotalRowCount := by
  have currentLength :
      (currentIndependentAlphabetInstructions layout).length =
        currentIndependentAlphabetRowCount := by
    simpa [CheckedProgram.rows] using
      current_independent_alphabet_rows_exact layout
  simp [currentInstructions, CheckedProgram.rows, currentLength,
    currentIndependentAlphabetRowCount, currentTotalRowCount]

theorem exact_saving :
    currentIndependentAlphabetRowCount - canonicalityRowCount =
      rowsSavedPerCoordinate /\
    currentTotalRowCount - totalRowCount = rowsSavedPerCoordinate := by
  decide

/-- The saving is tied to the actual current and proposed instruction
schedules, not to an external measured circuit count. -/
theorem schedule_saving_exact (layout : Layout) :
    (CheckedProgram.rows
        (currentIndependentAlphabetInstructions layout)).length -
      (CheckedProgram.rows (canonicalityInstructions layout)).length =
        rowsSavedPerCoordinate /\
    (CheckedProgram.rows (currentInstructions layout)).length -
      (rows layout).length = rowsSavedPerCoordinate := by
  rw [current_independent_alphabet_rows_exact, canonicality_rows_exact,
    current_total_rows_exact, total_rows_exact]
  exact exact_saving

end Nightstream.Implementation.R1CS.PiDecStrictCanonicalX
