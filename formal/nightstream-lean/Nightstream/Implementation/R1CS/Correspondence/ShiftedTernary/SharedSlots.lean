import Nightstream.Implementation.R1CS.Core.LinearSubstitution
import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.ReducedCore
import Nightstream.Implementation.R1CS.Artifacts.ShiftedTernary

/-!
Contract: exact production bridge for the reduced balanced-ternary opening.

Owns: the structural field-to-digit alias, the exact 20 centered residual-pair
rows plus one odd tail, the exact 82 retained product rows, and their
equivalence to the 123-obligation ShiftedTernaryReducedCore model.

Does not own: source-row validation in Rust, witness materialization, SIS
commitment semantics, or any full-F' cost claim.

Emits constraints: no. This file proves what the generated production schedule
means; Rust emits and validates the rows.

Authority boundary: the field has no independent target slot. Its value is
decoded from the same 41 digit slots used by the centered gates. The alias is
therefore structural and verifier-checked, not a digest or witness assertion.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---:|
| `production_census` | `shifted_ternary.{centered,definition,transition,omitted}` | exact 20 pair + 1 tail physical schedule, 123 retained obligations, and exact row indices | generated schema-3 artifact | no, census only |
| `production_gate_polynomial` | `shifted_ternary.{centered,definition,transition}` | exact production arity/roles and pair, tail, product specializations | generated schema-3 artifact | no, semantics binding only |
| `artifactGateAccepts_iff_productionAccepts` | actual CCS gate polynomial and generated matrix rows | 20 residual-pair rows plus one tail are exactly the 41 logical centered obligations | `ProjectiveSevenNonresidue` and verifier-fixed one | no, bridge only |
| `production_decoded_sharedAlias` | `shifted_ternary.shared.field` | field and 41 digits decode from the same target slots | generated source/target columns and coefficients | reconstruction only |
| `productionAccepts_iff_canonicalRows` | logical reduced-opening model | 41 centered obligations plus 82 product obligations are equivalent to all 124 canonical rows | prime Goldilocks modulus and verifier-fixed one | logical row-removal only |
| `artifactGateAccepts_iff_canonicalRows` | complete physical reduced opening | the exact 103 production rows are equivalent to all 124 canonical rows | `ProjectiveSevenNonresidue`, prime Goldilocks modulus, and verifier-fixed one | exactly the four proved families |
-/

namespace Nightstream.Implementation.R1CS.ShiftedTernarySharedSlots

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler

set_option maxRecDepth 262144

/-- Physical target slots. The field deliberately has no independent slot. -/
structure Layout where
  oneColumn : Nat
  digitColumn : Fin digitCount → Nat
  negativeColumn : Fin digitCount → Nat
  borrowColumn : Fin (digitCount - 1) → Nat
  columnsDistinct :
    (oneColumn ::
      (List.finRange digitCount).map digitColumn ++
      (List.finRange digitCount).map negativeColumn ++
      (List.finRange (digitCount - 1)).map borrowColumn).Nodup

def digitColumns (layout : Layout) : List Nat :=
  (List.finRange digitCount).map layout.digitColumn

def negativeColumns (layout : Layout) : List Nat :=
  (List.finRange digitCount).map layout.negativeColumn

def borrowColumns (layout : Layout) : List Nat :=
  (List.finRange (digitCount - 1)).map layout.borrowColumn

def aliasColumn : List Nat → List Nat → Nat → Option Nat
  | source :: sourceTail, target :: targetTail, query =>
      if query = source then some target
      else aliasColumn sourceTail targetTail query
  | _, _, _ => none

/-- Weighted field decoding over the exact slots used by the digit aliases. -/
def fieldTerms (layout : Layout) : List (Nat × Nat) :=
  (List.finRange digitCount).map fun index =>
    (layout.digitColumn index, 3 ^ index.val % goldilocksP)

def sourceExpansion (layout : Layout) :
    LinearSubstitution.ColumnExpansion :=
  fun source =>
    if source = 0 then [(layout.oneColumn, 1)]
    else if source = ShiftedTernary.fieldCol then fieldTerms layout
    else
      match aliasColumn ShiftedTernary.digitCols (digitColumns layout) source with
      | some target => [(target, 1)]
      | none =>
          match aliasColumn ShiftedTernary.negativeCols
              (negativeColumns layout) source with
          | some target => [(target, 1)]
          | none =>
              match aliasColumn ShiftedTernary.borrowCols
                  (borrowColumns layout) source with
              | some target => [(target, 1)]
              | none => []

def decodedAssignment (layout : Layout) (encoded : Nat → Nat) : Nat → Nat :=
  LinearSubstitution.assignment (sourceExpansion layout) encoded

/-- Exact shared-slot layout exported by the Rust production lowerer. -/
def productionLayout : Layout where
  oneColumn := ShiftedTernarySharedSlotsArtifact.oneColumn
  digitColumn := fun index =>
    ShiftedTernarySharedSlotsArtifact.digitColumns.getD index.val 0
  negativeColumn := fun index =>
    ShiftedTernarySharedSlotsArtifact.negativeColumns.getD index.val 0
  borrowColumn := fun index =>
    ShiftedTernarySharedSlotsArtifact.borrowColumns.getD index.val 0
  columnsDistinct := by native_decide

/-- Schema 3 is the first artifact contract that distinguishes physical
centered residual pairs from the ordinary odd tail. -/
theorem production_schema :
    ShiftedTernarySharedSlotsArtifact.schemaVersion = 3 := by
  native_decide

theorem production_source_columns :
    ShiftedTernarySharedSlotsArtifact.sourceFieldCol = ShiftedTernary.fieldCol ∧
    ShiftedTernarySharedSlotsArtifact.sourceDigitCols = ShiftedTernary.digitCols ∧
    ShiftedTernarySharedSlotsArtifact.sourceNegativeCols = ShiftedTernary.negativeCols ∧
    ShiftedTernarySharedSlotsArtifact.sourceBorrowCols = ShiftedTernary.borrowCols := by
  native_decide

/-- Every source column used by the reduced core instantiates the checked
production alias map. -/
theorem production_sourceExpansion_instantiates :
    sourceExpansion productionLayout 0 =
      [(ShiftedTernarySharedSlotsArtifact.oneColumn, 1)] ∧
    sourceExpansion productionLayout
        ShiftedTernarySharedSlotsArtifact.sourceFieldCol =
      ShiftedTernarySharedSlotsArtifact.fieldTerms ∧
    ShiftedTernarySharedSlotsArtifact.sourceDigitCols.map
        (sourceExpansion productionLayout) =
      ShiftedTernarySharedSlotsArtifact.digitColumns.map
        (fun target => [(target, 1)]) ∧
    ShiftedTernarySharedSlotsArtifact.sourceNegativeCols.map
        (sourceExpansion productionLayout) =
      ShiftedTernarySharedSlotsArtifact.negativeColumns.map
        (fun target => [(target, 1)]) ∧
    ShiftedTernarySharedSlotsArtifact.sourceBorrowCols.map
        (sourceExpansion productionLayout) =
      ShiftedTernarySharedSlotsArtifact.borrowColumns.map
        (fun target => [(target, 1)]) := by
  native_decide

/-- Two sparse rows differ only by term order in their three LCs. -/
def RowPermutation (left right : Row) : Prop :=
  left.a.Perm right.a ∧ left.b.Perm right.b ∧ left.c.Perm right.c

instance (left right : Row) : Decidable (RowPermutation left right) := by
  unfold RowPermutation
  infer_instance

inductive RowsPermutation : List Row → List Row → Prop where
  | nil : RowsPermutation [] []
  | cons {leftHead rightHead : Row} {leftTail rightTail : List Row}
      (head : RowPermutation leftHead rightHead)
      (tail : RowsPermutation leftTail rightTail) :
      RowsPermutation (leftHead :: leftTail) (rightHead :: rightTail)

private def rowsPermutationDecidable :
    (left right : List Row) → Decidable (RowsPermutation left right)
  | [], [] => isTrue .nil
  | [], _ :: _ => isFalse fun permutation => by cases permutation
  | _ :: _, [] => isFalse fun permutation => by cases permutation
  | leftHead :: leftTail, rightHead :: rightTail =>
      if head : RowPermutation leftHead rightHead then
        match rowsPermutationDecidable leftTail rightTail with
        | isTrue tail => isTrue (.cons head tail)
        | isFalse notTail => isFalse fun permutation => by
            cases permutation with
            | cons _ actualTail => exact notTail actualTail
      else
        isFalse fun permutation => by
          cases permutation with
          | cons actualHead _ => exact head actualHead

instance (left right : List Row) : Decidable (RowsPermutation left right) :=
  rowsPermutationDecidable left right

theorem rowHolds_iff_of_permutation
    (encoded : Nat → Nat) {left right : Row}
    (permutation : RowPermutation left right) :
    RowHolds encoded left ↔ RowHolds encoded right := by
  unfold RowHolds
  rw [Program.lcEval_eq_of_perm encoded permutation.1,
    Program.lcEval_eq_of_perm encoded permutation.2.1,
    Program.lcEval_eq_of_perm encoded permutation.2.2]

private theorem satisfies_cons_iff
    (head : Row) (tail : List Row) (encoded : Nat → Nat) :
    Satisfies (head :: tail) encoded ↔
      RowHolds encoded head ∧ Satisfies tail encoded := by
  simp [Satisfies]

theorem satisfies_iff_of_rowsPermutation
    (encoded : Nat → Nat) {left right : List Row}
    (permutation : RowsPermutation left right) :
    Satisfies left encoded ↔ Satisfies right encoded := by
  induction permutation with
  | nil => simp [Satisfies]
  | cons rowPermutation _ inductionHypothesis =>
      rw [satisfies_cons_iff, satisfies_cons_iff,
        rowHolds_iff_of_permutation encoded rowPermutation,
        inductionHypothesis]

/-- The exact 82 source R1CS rows retained by production. -/
def retainedCoreRows : List Row :=
  ((List.range digitCount).map fun index =>
      negativeDefinitionRow
        (ShiftedTernary.digitCols.getD index 0)
        (ShiftedTernary.negativeCols.getD index 0)) ++
    (List.range digitCount).map borrowRow

def retainedRows (layout : Layout) : List Row :=
  retainedCoreRows.map (LinearSubstitution.row (sourceExpansion layout))

theorem retainedCoreRows_length : retainedCoreRows.length = 82 := by
  native_decide

theorem satisfies_retainedCoreRows_iff (assignment : Nat → Nat) :
    Satisfies retainedCoreRows assignment ↔
      (∀ index, index < digitCount →
        RowHolds assignment
          (negativeDefinitionRow
            (ShiftedTernary.digitCols.getD index 0)
            (ShiftedTernary.negativeCols.getD index 0))) ∧
      (∀ index, index < digitCount →
        RowHolds assignment (borrowRow index)) := by
  constructor
  · intro satisfies
    constructor
    · intro index indexLt
      apply satisfies
      unfold retainedCoreRows
      apply List.mem_append_left
      apply List.mem_map.mpr
      exact ⟨index, List.mem_range.mpr indexLt, rfl⟩
    · intro index indexLt
      apply satisfies
      unfold retainedCoreRows
      apply List.mem_append_right
      apply List.mem_map.mpr
      exact ⟨index, List.mem_range.mpr indexLt, rfl⟩
  · rintro ⟨definitions, transitions⟩ row member
    unfold retainedCoreRows at member
    rw [List.mem_append] at member
    rcases member with definition | transition
    · rcases List.mem_map.mp definition with ⟨index, indexMember, rfl⟩
      exact definitions index (List.mem_range.mp indexMember)
    · rcases List.mem_map.mp transition with ⟨index, indexMember, rfl⟩
      exact transitions index (List.mem_range.mp indexMember)

theorem satisfies_retainedRows_iff
    (layout : Layout) (encoded : Nat → Nat) :
    Satisfies (retainedRows layout) encoded ↔
      (∀ index, index < digitCount →
        RowHolds (decodedAssignment layout encoded)
          (negativeDefinitionRow
            (ShiftedTernary.digitCols.getD index 0)
            (ShiftedTernary.negativeCols.getD index 0))) ∧
      (∀ index, index < digitCount →
        RowHolds (decodedAssignment layout encoded) (borrowRow index)) := by
  rw [retainedRows, LinearSubstitution.satisfies_mapped_iff,
    satisfies_retainedCoreRows_iff]
  rfl

theorem reducedAccepts_iff
    (layout : Layout) (encoded : Nat → Nat) :
    ShiftedTernaryReducedCore.Accepts
        (decodedAssignment layout encoded) ↔
      (∀ index, index < digitCount →
        ShiftedTernaryReducedCore.CenteredUnitGateHolds
          (decodedAssignment layout encoded
            (ShiftedTernary.digitCols.getD index 0))) ∧
      Satisfies (retainedRows layout) encoded := by
  constructor
  · intro accepts
    exact ⟨accepts.centeredUnit,
      (satisfies_retainedRows_iff layout encoded).mpr
        ⟨accepts.negativeDefinition, accepts.borrowTransition⟩⟩
  · rintro ⟨centered, retained⟩
    have r1cs := (satisfies_retainedRows_iff layout encoded).mp retained
    exact ⟨centered, r1cs.1, r1cs.2⟩

def centeredPairRowIds : List Nat :=
  (ShiftedTernarySharedSlotsArtifact.centeredPairRows).map Prod.fst

def centeredTailRowIds : List Nat :=
  (ShiftedTernarySharedSlotsArtifact.centeredTailRows).map Prod.fst

def centeredPairCoordinates : List Nat :=
  (ShiftedTernarySharedSlotsArtifact.centeredPairRows).flatMap fun row =>
    [row.2.1, row.2.2]

def centeredTailCoordinates : List Nat :=
  (ShiftedTernarySharedSlotsArtifact.centeredTailRows).map Prod.snd

def centeredScheduledCoordinates : List Nat :=
  centeredPairCoordinates ++ centeredTailCoordinates

/-- Static census, exact row-index partition, and pair/tail schedule from the
generated artifact. The `Nodup` and `Disjoint` facts reject duplicate,
missing, or overlapping physical rows and centered coordinates. -/
theorem production_census :
    ShiftedTernarySharedSlotsArtifact.sourceRows =
      (List.range 124).map (fun index => index + 2) ∧
    ShiftedTernarySharedSlotsArtifact.retainedSourceRows =
      ShiftedTernarySharedSlotsArtifact.indicatorDefinitionSourceRows ++
        ShiftedTernarySharedSlotsArtifact.transitionSourceRows ∧
    ShiftedTernarySharedSlotsArtifact.omittedSourceRows =
      ShiftedTernarySharedSlotsArtifact.indicatorSupportSourceRows ++
        [ShiftedTernarySharedSlotsArtifact.reconstructionSourceRow] ∧
    ShiftedTernarySharedSlotsArtifact.retainedEncodedRows =
      (List.range 82).map (fun index => index + 1128) ∧
    ShiftedTernarySharedSlotsArtifact.retainedRowMap =
      ShiftedTernarySharedSlotsArtifact.retainedSourceRows.zip
        ShiftedTernarySharedSlotsArtifact.retainedEncodedRows ∧
    centeredPairRowIds.Nodup ∧
    centeredTailRowIds.Nodup ∧
    (centeredPairRowIds ++ centeredTailRowIds).Nodup ∧
    (centeredPairRowIds ++ centeredTailRowIds ++
      ShiftedTernarySharedSlotsArtifact.retainedEncodedRows).Nodup ∧
    centeredPairCoordinates.Nodup ∧
    centeredTailCoordinates.Nodup ∧
    (centeredPairCoordinates ++ centeredTailCoordinates).Nodup ∧
    centeredScheduledCoordinates =
      ShiftedTernarySharedSlotsArtifact.digitColumns ∧
    ShiftedTernarySharedSlotsArtifact.omittedNegativeBitnessColumns =
      ShiftedTernarySharedSlotsArtifact.negativeColumns ∧
    ShiftedTernarySharedSlotsArtifact.omittedBorrowBitnessColumns =
      ShiftedTernarySharedSlotsArtifact.borrowColumns ∧
    ShiftedTernarySharedSlotsArtifact.retainedSourceRows.length = 82 ∧
    ShiftedTernarySharedSlotsArtifact.omittedSourceRows.length = 42 ∧
    ShiftedTernarySharedSlotsArtifact.centeredPairRows.length = 20 ∧
    ShiftedTernarySharedSlotsArtifact.centeredTailRows.length = 1 ∧
    centeredScheduledCoordinates.length = 41 ∧
    ShiftedTernarySharedSlotsArtifact.omittedNegativeBitnessColumns.length = 41 ∧
    ShiftedTernarySharedSlotsArtifact.omittedBorrowBitnessColumns.length = 40 ∧
    ShiftedTernarySharedSlotsArtifact.retainedObligationCount = 123 ∧
    ShiftedTernarySharedSlotsArtifact.retainedPhysicalRowCount = 103 ∧
    ShiftedTernarySharedSlotsArtifact.omittedObligationCount = 123 ∧
    ShiftedTernarySharedSlotsArtifact.retainedSourceExpandedRows.length = 82 ∧
    ShiftedTernarySharedSlotsArtifact.retainedProductRows.length = 82 := by
  native_decide

/-- The schema-3 physical schedule covers all 41 digit coordinates exactly in
source order. Pair rows, the odd tail, and retained product rows have disjoint
row identities, so no missing, duplicate, or cross-family-overlapping row can
hide in the artifact census. -/
theorem centered_pair_tail_schedule_exact :
    centeredScheduledCoordinates =
      ShiftedTernarySharedSlotsArtifact.digitColumns ∧
    centeredScheduledCoordinates.length = digitCount ∧
    (centeredPairCoordinates ++ centeredTailCoordinates).Nodup ∧
    (centeredPairRowIds ++ centeredTailRowIds ++
      ShiftedTernarySharedSlotsArtifact.retainedEncodedRows).Nodup ∧
    ShiftedTernarySharedSlotsArtifact.centeredPairRows.length = 20 ∧
    ShiftedTernarySharedSlotsArtifact.centeredTailRows.length = 1 ∧
    ShiftedTernarySharedSlotsArtifact.retainedPhysicalRowCount = 103 := by
  native_decide

/-- The generated artifact records the exact production gate arity, role IDs,
and polynomial specializations used by the 20 centered pair rows, one centered
tail row, and 82 single-product rows. -/
theorem production_gate_polynomial :
    ShiftedTernarySharedSlotsArtifact.productionPolynomialArity = 56 ∧
    ShiftedTernarySharedSlotsArtifact.selectorRole = 0 ∧
    ShiftedTernarySharedSlotsArtifact.centeredUnitTailRole = 2 ∧
    ShiftedTernarySharedSlotsArtifact.centeredPairLeftRole = 46 ∧
    ShiftedTernarySharedSlotsArtifact.centeredPairRightRole = 47 ∧
    ShiftedTernarySharedSlotsArtifact.productLeftRole = 3 ∧
    ShiftedTernarySharedSlotsArtifact.productRightRole = 21 ∧
    ShiftedTernarySharedSlotsArtifact.productOutRole = 39 ∧
    ShiftedTernarySharedSlotsArtifact.centeredTailPolynomialTerms =
      [(1, [(0, 1), (2, 3)]),
        (goldilocksP - 1, [(0, 1), (2, 1)])] ∧
    ShiftedTernarySharedSlotsArtifact.centeredPairPolynomialTerms =
      [(1, [(0, 1), (46, 6)]),
        (goldilocksP - 2, [(0, 1), (46, 4)]),
        (1, [(0, 1), (46, 2)]),
        (goldilocksP - 7, [(0, 1), (47, 6)]),
        (14, [(0, 1), (47, 4)]),
        (goldilocksP - 7, [(0, 1), (47, 2)])] ∧
    ShiftedTernarySharedSlotsArtifact.singleProductPolynomialTerms =
      [(1, [(0, 1), (3, 1), (21, 1)]),
        (goldilocksP - 1, [(0, 1), (39, 1)])] := by
  native_decide

private instance : NeZero goldilocksP := ⟨by decide⟩

private abbrev GateField := Fin goldilocksP

private def gateResidue (value : Nat) : GateField :=
  ⟨value % goldilocksP, Nat.mod_lt _ (by decide)⟩

private theorem gateResidue_one : gateResidue 1 = (1 : GateField) := by
  apply Fin.ext
  native_decide

/-- Local evaluator for the sparse polynomial terms exported by the Rust
artifact. It intentionally models only the gate specialization needed here. -/
private def gateMonomialEval (values : Nat → GateField) :
    List (Nat × Nat) → GateField
  | [] => 1
  | power :: powers =>
      values power.1 ^ power.2 * gateMonomialEval values powers

private def gatePolynomialEval (values : Nat → GateField) :
    List (Nat × List (Nat × Nat)) → GateField
  | [] => 0
  | term :: terms =>
      gateResidue term.1 * gateMonomialEval values term.2 +
        gatePolynomialEval values terms

private def GatePolynomialHolds
    (terms : List (Nat × List (Nat × Nat)))
    (values : Nat → GateField) : Prop :=
  gatePolynomialEval values terms = 0

private def centeredTailPolynomialValues
    (selector centered : Nat) : Nat → GateField :=
  fun matrix =>
    if matrix = 0 then gateResidue selector
    else if matrix = 2 then gateResidue centered
    else 0

private def centeredPairPolynomialValues
    (selector left right : Nat) : Nat → GateField :=
  fun matrix =>
    if matrix = 0 then gateResidue selector
    else if matrix = 46 then gateResidue left
    else if matrix = 47 then gateResidue right
    else 0

private def singleProductPolynomialValues
    (selector left right output : Nat) : Nat → GateField :=
  fun matrix =>
    if matrix = 0 then gateResidue selector
    else if matrix = 3 then gateResidue left
    else if matrix = 21 then gateResidue right
    else if matrix = 39 then gateResidue output
    else 0

private theorem negOne_mul_add_self (value : GateField) :
    gateResidue (goldilocksP - 1) * value + value = 0 := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul, gateResidue]
  have raw : (goldilocksP - 1) * value.val + value.val =
      goldilocksP * value.val := by
    unfold goldilocksP
    omega
  have modular : ((goldilocksP - 1) * value.val + value.val) %
      goldilocksP = 0 := by
    rw [raw, Nat.mul_mod_right]
  simpa only [Nat.add_mod, Nat.mul_mod, Nat.mod_mod,
    Nat.mod_eq_of_lt value.isLt] using modular

private theorem gateField_add_assoc (left middle right : GateField) :
    (left + middle) + right = left + (middle + right) := by
  apply Fin.ext
  simp only [Fin.val_add]
  rw [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

private theorem gateField_add_comm (left right : GateField) :
    left + right = right + left := by
  apply Fin.ext
  simp only [Fin.val_add, Nat.add_comm]

private theorem gateField_add_reassociate_four
    (a b c d : GateField) :
    (a + b) + (c + d) = (a + (b + c)) + d := by
  apply Fin.ext
  simp only [Fin.val_add]
  simp [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

private theorem gateField_add_reassociate_six
    (a b c d e f : GateField) :
    a + (b + (c + (d + (e + f)))) =
      ((a + b) + c) + ((d + e) + f) := by
  apply Fin.ext
  simp only [Fin.val_add]
  simp [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

private theorem gateField_one_mul (value : GateField) :
    1 * value = value := by
  apply Fin.ext
  change (1 % goldilocksP * value.val) % goldilocksP = value.val
  have oneMod : 1 % goldilocksP = 1 := by native_decide
  rw [oneMod, Nat.one_mul, Nat.mod_eq_of_lt value.isLt]

private theorem gateField_mul_one (value : GateField) :
    value * 1 = value := by
  apply Fin.ext
  change (value.val * (1 % goldilocksP)) % goldilocksP = value.val
  have oneMod : 1 % goldilocksP = 1 := by native_decide
  rw [oneMod, Nat.mul_one, Nat.mod_eq_of_lt value.isLt]

private theorem gateField_add_zero (value : GateField) :
    value + 0 = value := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_zero, Nat.add_zero,
    Nat.mod_eq_of_lt value.isLt]

private theorem gateField_zero_add (value : GateField) :
    0 + value = value := by
  rw [gateField_add_comm, gateField_add_zero]

private theorem gateField_mul_assoc (left middle right : GateField) :
    (left * middle) * right = left * (middle * right) := by
  apply Fin.ext
  simp only [Fin.val_mul]
  rw [Nat.mod_mul_mod, Nat.mul_mod_mod, Nat.mul_assoc]

private theorem gateField_mul_comm (left right : GateField) :
    left * right = right * left := by
  apply Fin.ext
  simp only [Fin.val_mul, Nat.mul_comm]

private theorem gateField_mul_add (left middle right : GateField) :
    left * (middle + right) = left * middle + left * right := by
  apply Fin.ext
  simp only [Fin.val_mul, Fin.val_add]
  simp [Nat.mod_eq_of_lt middle.isLt, Nat.mod_eq_of_lt right.isLt,
    Nat.mul_mod_mod, Nat.mul_add, Nat.add_mod]

private theorem gateField_add_mul (left middle right : GateField) :
    (left + middle) * right = left * right + middle * right := by
  rw [gateField_mul_comm, gateField_mul_add]
  congr 1 <;> rw [gateField_mul_comm]

private theorem negOne_add_negOne :
    gateResidue (goldilocksP - 1) + gateResidue (goldilocksP - 1) =
      gateResidue (goldilocksP - 2) := by
  native_decide

private theorem negOne_mul_negOne :
    gateResidue (goldilocksP - 1) * gateResidue (goldilocksP - 1) = 1 := by
  native_decide

private theorem negSeven_mul_negTwo :
    gateResidue (goldilocksP - 7) * gateResidue (goldilocksP - 2) =
      gateResidue 14 := by
  native_decide

private theorem add_negOne_mul_eq_zero_iff
    (left right : GateField) :
    left + gateResidue (goldilocksP - 1) * right = 0 ↔ left = right := by
  constructor
  · intro zero
    have added := congrArg (fun value => value + right) zero
    dsimp at added
    rw [gateField_add_assoc, negOne_mul_add_self, gateField_add_zero,
      gateField_zero_add] at added
    exact added
  · intro equal
    rw [equal, gateField_add_comm, negOne_mul_add_self]

private theorem centered_field_eq_iff (value : Nat) :
    gateResidue value * gateResidue value * gateResidue value =
        gateResidue value ↔
      ShiftedTernaryReducedCore.CenteredUnitGateHolds value := by
  rw [← add_negOne_mul_eq_zero_iff]
  constructor
  · intro zero
    have values := congrArg Fin.val zero
    simpa [gateResidue, Fin.val_add, Fin.val_mul, Nat.pow_succ,
      Nat.add_mod, Nat.mul_mod, Nat.mod_mod,
      ShiftedTernaryReducedCore.CenteredUnitGateHolds] using values
  · intro holds
    apply Fin.ext
    simpa [gateResidue, Fin.val_add, Fin.val_mul, Nat.pow_succ,
      Nat.add_mod, Nat.mul_mod, Nat.mod_mod,
      ShiftedTernaryReducedCore.CenteredUnitGateHolds] using holds

private theorem product_field_eq_iff_rowHolds
    (encoded : Nat → Nat) (row : Row) :
    gateResidue (lcEval encoded row.a) *
        gateResidue (lcEval encoded row.b) =
      gateResidue (lcEval encoded row.c) ↔
        RowHolds encoded row := by
  have outputLt : lcEval encoded row.c < goldilocksP :=
    Nat.mod_lt _ (by decide)
  constructor
  · intro equal
    have values := congrArg Fin.val equal
    simpa [gateResidue, Fin.val_mul, Nat.mul_mod, Nat.mod_mod,
      Nat.mod_eq_of_lt outputLt, RowHolds] using values
  · intro holds
    apply Fin.ext
    simpa [gateResidue, Fin.val_mul, Nat.mul_mod, Nat.mod_mod,
      Nat.mod_eq_of_lt outputLt, RowHolds] using holds

private theorem centeredTailPolynomialEval (value : Nat) :
    gatePolynomialEval (centeredTailPolynomialValues 1 value)
      ShiftedTernarySharedSlotsArtifact.centeredTailPolynomialTerms =
      gateResidue value * gateResidue value * gateResidue value +
        gateResidue (goldilocksP - 1) * gateResidue value := by
  have terms :
      ShiftedTernarySharedSlotsArtifact.centeredTailPolynomialTerms =
        [(1, [(0, 1), (2, 3)]),
          (goldilocksP - 1, [(0, 1), (2, 1)])] := by
    native_decide
  rw [terms]
  simp [gatePolynomialEval, gateMonomialEval, centeredTailPolynomialValues,
    gateResidue_one, gateField_one_mul, gateField_mul_one]

private theorem singleProductPolynomialEval
    (left right output : Nat) :
    gatePolynomialEval (singleProductPolynomialValues 1 left right output)
        ShiftedTernarySharedSlotsArtifact.singleProductPolynomialTerms =
      gateResidue left * gateResidue right +
        gateResidue (goldilocksP - 1) * gateResidue output := by
  have terms :
      ShiftedTernarySharedSlotsArtifact.singleProductPolynomialTerms =
        [(1, [(0, 1), (3, 1), (21, 1)]),
          (goldilocksP - 1, [(0, 1), (39, 1)])] := by
    native_decide
  rw [terms]
  simp [gatePolynomialEval, gateMonomialEval,
    singleProductPolynomialValues, gateResidue_one, gateField_one_mul,
    gateField_mul_one]

/-- The exported `+S*C^3-S*C` specialization with `S=1` has exactly the
common centered-unit semantics used by `ProductionAccepts`. -/
theorem centeredTailPolynomialHolds_iff (value : Nat) :
    GatePolynomialHolds
        ShiftedTernarySharedSlotsArtifact.centeredTailPolynomialTerms
        (centeredTailPolynomialValues 1 value) ↔
      ShiftedTernaryReducedCore.CenteredUnitGateHolds value := by
  unfold GatePolynomialHolds
  rw [centeredTailPolynomialEval, add_negOne_mul_eq_zero_iff,
    centered_field_eq_iff]

private def centeredResidual (value : Nat) : GateField :=
  gateResidue value * gateResidue value * gateResidue value +
    gateResidue (goldilocksP - 1) * gateResidue value

private theorem centeredResidual_square_expansion (value : Nat) :
    centeredResidual value * centeredResidual value =
      gateResidue value * gateResidue value * gateResidue value *
            gateResidue value * gateResidue value * gateResidue value +
        gateResidue (goldilocksP - 2) *
            (gateResidue value * gateResidue value *
              gateResidue value * gateResidue value) +
        gateResidue value * gateResidue value := by
  unfold centeredResidual
  rw [gateField_add_mul, gateField_mul_add, gateField_mul_add]
  calc
    _ = gateResidue value * gateResidue value * gateResidue value *
            gateResidue value * gateResidue value * gateResidue value +
          gateResidue (goldilocksP - 1) *
              (gateResidue value * gateResidue value *
                gateResidue value * gateResidue value) +
          (gateResidue (goldilocksP - 1) *
              (gateResidue value * gateResidue value *
                gateResidue value * gateResidue value) +
            (gateResidue (goldilocksP - 1) *
                gateResidue (goldilocksP - 1)) *
              (gateResidue value * gateResidue value)) := by
        ac_rfl
    _ = (gateResidue value * gateResidue value * gateResidue value *
            gateResidue value * gateResidue value * gateResidue value +
          (gateResidue (goldilocksP - 1) *
              (gateResidue value * gateResidue value *
                gateResidue value * gateResidue value) +
            gateResidue (goldilocksP - 1) *
              (gateResidue value * gateResidue value *
                gateResidue value * gateResidue value))) +
          gateResidue value * gateResidue value := by
      rw [negOne_mul_negOne, gateField_one_mul]
      apply gateField_add_reassociate_four
    _ = _ := by
      have combine (term : GateField) :
          gateResidue (goldilocksP - 1) * term +
              gateResidue (goldilocksP - 1) * term =
            gateResidue (goldilocksP - 2) * term := by
        rw [← gateField_add_mul, negOne_add_negOne]
      rw [combine]

private theorem centeredResidual_zero_iff (value : Nat) :
    centeredResidual value = 0 ↔
      ShiftedTernaryReducedCore.CenteredUnitGateHolds value := by
  unfold centeredResidual
  rw [← centeredTailPolynomialEval]
  exact centeredTailPolynomialHolds_iff value

/-- Narrow algebraic instantiation boundary: seven is projectively
nonresidual over the canonical Goldilocks residue carrier. The unconditional
field-level theorem already exists as
`SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.ResidualPairFamilies.residualPairHolds_iff`,
using `KExt.w_not_square`. This no-Mathlib project deliberately carries the
Nat-residue instantiation as a visible premise. -/
def ProjectiveSevenNonresidue : Prop :=
  ∀ left right : GateField,
    left * left + gateResidue (goldilocksP - 7) * (right * right) = 0 →
      left = 0 ∧ right = 0

/-- The schema-3 six-term specialization is exactly the square of each
centered residual, with the right square scaled by negative seven. -/
private theorem centeredPairPolynomialEval (left right : Nat) :
    gatePolynomialEval (centeredPairPolynomialValues 1 left right)
        ShiftedTernarySharedSlotsArtifact.centeredPairPolynomialTerms =
      centeredResidual left * centeredResidual left +
        gateResidue (goldilocksP - 7) *
          (centeredResidual right * centeredResidual right) := by
  have terms :
      ShiftedTernarySharedSlotsArtifact.centeredPairPolynomialTerms =
        [(1, [(0, 1), (46, 6)]),
          (goldilocksP - 2, [(0, 1), (46, 4)]),
          (1, [(0, 1), (46, 2)]),
          (goldilocksP - 7, [(0, 1), (47, 6)]),
          (14, [(0, 1), (47, 4)]),
          (goldilocksP - 7, [(0, 1), (47, 2)])] := by
    native_decide
  rw [terms]
  simp [gatePolynomialEval, gateMonomialEval,
    centeredPairPolynomialValues, gateResidue_one, gateField_one_mul,
    gateField_mul_one]
  rw [centeredResidual_square_expansion left,
    centeredResidual_square_expansion right]
  rw [gateField_mul_add, gateField_mul_add]
  have middle :
      gateResidue (goldilocksP - 7) *
          (gateResidue (goldilocksP - 2) *
            (gateResidue right * gateResidue right *
              gateResidue right * gateResidue right)) =
        gateResidue 14 *
          (gateResidue right * gateResidue right *
            gateResidue right * gateResidue right) := by
    rw [← gateField_mul_assoc, negSeven_mul_negTwo]
  rw [middle]
  apply gateField_add_reassociate_six

/-- Under the named projective-seven premise, one exact schema-3 pair
specialization is equivalent to both centered obligations. Polynomial/role
binding is proved above and is not part of the premise. -/
theorem centeredPairPolynomialHolds_iff
    (projectiveNonresidue : ProjectiveSevenNonresidue)
    (left right : Nat) :
    GatePolynomialHolds
        ShiftedTernarySharedSlotsArtifact.centeredPairPolynomialTerms
        (centeredPairPolynomialValues 1 left right) ↔
      ShiftedTernaryReducedCore.CenteredUnitGateHolds left ∧
        ShiftedTernaryReducedCore.CenteredUnitGateHolds right := by
  unfold GatePolynomialHolds
  rw [centeredPairPolynomialEval]
  constructor
  · intro packed
    have residuals := projectiveNonresidue _ _ packed
    rwa [centeredResidual_zero_iff,
      centeredResidual_zero_iff] at residuals
  · rintro ⟨leftCentered, rightCentered⟩
    have leftZero := (centeredResidual_zero_iff left).mpr leftCentered
    have rightZero := (centeredResidual_zero_iff right).mpr rightCentered
    rw [leftZero, rightZero]
    native_decide

/-- The exported `+S*L0*R0-S*OUT` specialization with `S=1` has exactly the
R1CS product-row semantics used by `ProductionAccepts`. -/
theorem singleProductPolynomialHolds_iff
    (encoded : Nat → Nat) (row : Row) :
    GatePolynomialHolds
        ShiftedTernarySharedSlotsArtifact.singleProductPolynomialTerms
        (singleProductPolynomialValues 1
          (lcEval encoded row.a) (lcEval encoded row.b)
          (lcEval encoded row.c)) ↔
      RowHolds encoded row := by
  unfold GatePolynomialHolds
  rw [singleProductPolynomialEval, add_negOne_mul_eq_zero_iff,
    product_field_eq_iff_rowHolds]

/-- The generated pre-CSC retained rows are the exact generic reduced rows
after structural column substitution. -/
theorem retainedSourceExpandedRows_permutation :
    RowsPermutation
      ShiftedTernarySharedSlotsArtifact.retainedSourceExpandedRows
      (retainedRows productionLayout) := by
  native_decide

/-- The 82 actual product rows read from production CCS matrices match the
generated pre-CSC rows modulo sparse-term order. -/
theorem retainedProductRows_permutation :
    RowsPermutation
      ShiftedTernarySharedSlotsArtifact.retainedProductRows
      ShiftedTernarySharedSlotsArtifact.retainedSourceExpandedRows := by
  native_decide

theorem satisfies_retainedProductRows_iff
    (encoded : Nat → Nat) :
    Satisfies ShiftedTernarySharedSlotsArtifact.retainedProductRows encoded ↔
      Satisfies (retainedRows productionLayout) encoded := by
  rw [satisfies_iff_of_rowsPermutation encoded
      retainedProductRows_permutation,
    satisfies_iff_of_rowsPermutation encoded
      retainedSourceExpandedRows_permutation]

private theorem production_decoded_digit
    (encoded : Nat → Nat) {index : Nat} (indexLt : index < digitCount) :
    decodedAssignment productionLayout encoded
        (ShiftedTernary.digitCols.getD index 0) =
      encoded
        (ShiftedTernarySharedSlotsArtifact.digitColumns.getD index 0) %
          goldilocksP := by
  have mapped := production_sourceExpansion_instantiates.2.2.1
  have atIndex := congrArg
    (fun entries : List (List (Nat × Nat)) => entries.getD index [])
    mapped
  have sourceLength :
      ShiftedTernarySharedSlotsArtifact.sourceDigitCols.length =
        digitCount := by native_decide
  have targetLength :
      ShiftedTernarySharedSlotsArtifact.digitColumns.length =
        digitCount := by native_decide
  have expansion :
      sourceExpansion productionLayout
          (ShiftedTernarySharedSlotsArtifact.sourceDigitCols.getD index 0) =
        [(ShiftedTernarySharedSlotsArtifact.digitColumns.getD index 0, 1)] := by
    simpa [List.getD_eq_getElem?_getD, sourceLength, targetLength,
      indexLt] using atIndex
  rw [← production_source_columns.2.1]
  unfold decodedAssignment LinearSubstitution.assignment
  rw [expansion]
  simp [lcEval]

private theorem production_decoded_field (encoded : Nat → Nat) :
    decodedAssignment productionLayout encoded ShiftedTernary.fieldCol =
      ShiftedTernarySound.lowValue
          (fun index =>
            encoded
              (ShiftedTernarySharedSlotsArtifact.digitColumns.getD index 0))
          digitCount %
        goldilocksP := by
  have schedule :
      ShiftedTernarySharedSlotsArtifact.fieldTerms =
        (List.range digitCount).map fun index =>
          (ShiftedTernarySharedSlotsArtifact.digitColumns.getD index 0,
            3 ^ index % goldilocksP) := by
    native_decide
  rw [← production_source_columns.1]
  unfold decodedAssignment LinearSubstitution.assignment
  rw [production_sourceExpansion_instantiates.2.1, schedule]
  have folded := ShiftedTernarySound.foldl_range_eq_lowValue
    (fun index =>
      encoded
        (ShiftedTernarySharedSlotsArtifact.digitColumns.getD index 0))
    0 digitCount
  simpa [lcEval, List.foldl_map] using congrArg
    (fun value => value % goldilocksP) folded

/-- Reconstruction is a theorem of the checked slot map, not an emitted row. -/
theorem production_decoded_sharedAlias (encoded : Nat → Nat) :
    ShiftedTernaryReducedCore.SharedFieldDigitAlias
      (decodedAssignment productionLayout encoded) := by
  unfold ShiftedTernaryReducedCore.SharedFieldDigitAlias
  rw [production_decoded_field]
  simp only [Nat.mod_mod]
  apply ShiftedTernarySound.lowValue_mod_congr
  intro index indexLt
  unfold ShiftedTernarySound.centeredDigit
  rw [production_decoded_digit encoded indexLt]
  simp

/-- Semantic centered obligation for one generated digit column. -/
def CenteredColumnHolds (encoded : Nat → Nat) (column : Nat) : Prop :=
  ShiftedTernaryReducedCore.CenteredUnitGateHolds
    (encoded column % goldilocksP)

def CenteredColumnsAccept (encoded : Nat → Nat)
    (columns : List Nat) : Prop :=
  ∀ column ∈ columns, CenteredColumnHolds encoded column

/-- Exact 41 logical centered obligations in generated source order. Physical
cost is 20 residual-pair rows plus one ordinary odd tail. -/
def ProductionCenteredAccepts (encoded : Nat → Nat) : Prop :=
  CenteredColumnsAccept encoded
    ShiftedTernarySharedSlotsArtifact.digitColumns

def ArtifactCenteredSemantics (encoded : Nat → Nat) : Prop :=
  (∀ pair ∈ ShiftedTernarySharedSlotsArtifact.centeredPairRows,
    CenteredColumnHolds encoded pair.2.1 ∧
      CenteredColumnHolds encoded pair.2.2) ∧
  (∀ tail ∈ ShiftedTernarySharedSlotsArtifact.centeredTailRows,
    CenteredColumnHolds encoded tail.2)

theorem artifactCenteredSemantics_iff_columns (encoded : Nat → Nat) :
    ArtifactCenteredSemantics encoded ↔
      CenteredColumnsAccept encoded centeredScheduledCoordinates := by
  unfold ArtifactCenteredSemantics CenteredColumnsAccept
  constructor
  · rintro ⟨pairs, tails⟩ column member
    unfold centeredScheduledCoordinates at member
    rw [List.mem_append] at member
    rcases member with pairMember | tailMember
    · unfold centeredPairCoordinates at pairMember
      rcases List.mem_flatMap.mp pairMember with
        ⟨pair, pairIn, columnIn⟩
      have accepted := pairs pair pairIn
      simp only [List.mem_cons, List.not_mem_nil, or_false] at columnIn
      rcases columnIn with rfl | rfl
      · exact accepted.1
      · exact accepted.2
    · unfold centeredTailCoordinates at tailMember
      rcases List.mem_map.mp tailMember with ⟨tail, tailIn, rfl⟩
      exact tails tail tailIn
  · intro columns
    constructor
    · intro pair pairIn
      constructor
      · apply columns pair.2.1
        unfold centeredScheduledCoordinates centeredPairCoordinates
        apply List.mem_append_left
        exact List.mem_flatMap.mpr ⟨pair, pairIn, by simp⟩
      · apply columns pair.2.2
        unfold centeredScheduledCoordinates centeredPairCoordinates
        apply List.mem_append_left
        exact List.mem_flatMap.mpr ⟨pair, pairIn, by simp⟩
    · intro tail tailIn
      apply columns tail.2
      unfold centeredScheduledCoordinates centeredTailCoordinates
      apply List.mem_append_right
      exact List.mem_map.mpr ⟨tail, tailIn, rfl⟩

/-- Index view used by the reduced semantic model. -/
theorem productionCenteredAccepts_indexed (encoded : Nat → Nat) :
    ProductionCenteredAccepts encoded ↔
      ∀ index, index < digitCount →
    ShiftedTernaryReducedCore.CenteredUnitGateHolds
      (encoded
        (ShiftedTernarySharedSlotsArtifact.digitColumns.getD index 0) %
          goldilocksP) := by
  unfold ProductionCenteredAccepts CenteredColumnsAccept CenteredColumnHolds
  constructor
  · intro accepts index indexLt
    apply accepts
    have length :
        ShiftedTernarySharedSlotsArtifact.digitColumns.length = digitCount := by
      native_decide
    rw [List.mem_iff_getElem]
    exact ⟨index, by simpa [length], by
      simp [List.getD_eq_getElem?_getD, indexLt, length]⟩
  · intro accepts column member
    rw [List.mem_iff_getElem] at member
    rcases member with ⟨index, indexLt, rfl⟩
    have digitLt : index < digitCount := by
      simpa using indexLt
    have getD_eq :
        ShiftedTernarySharedSlotsArtifact.digitColumns.getD index 0 =
          ShiftedTernarySharedSlotsArtifact.digitColumns[index] := by
      simp [List.getD_eq_getElem?_getD, indexLt]
      rfl
    have accepted := accepts index digitLt
    rw [getD_eq] at accepted
    exact accepted

/-- Logical reduced-opening acceptance: 41 centered obligations plus the 82
generated product rows. The physical centered schedule is represented by
`ArtifactGateAccepts` and needs `ProjectiveSevenNonresidue` to refine this
predicate. -/
def ProductionAccepts (encoded : Nat → Nat) : Prop :=
  ProductionCenteredAccepts encoded ∧
    Satisfies ShiftedTernarySharedSlotsArtifact.retainedProductRows encoded

instance (encoded : Nat → Nat) : Decidable (ProductionAccepts encoded) := by
  unfold ProductionAccepts ProductionCenteredAccepts
    CenteredColumnsAccept CenteredColumnHolds
  infer_instance

/-- Acceptance of the exact 20 pair rows, one tail row, and 82 product rows
recorded from production matrices. -/
def ArtifactGateAccepts (encoded : Nat → Nat) : Prop :=
  (∀ pair ∈ ShiftedTernarySharedSlotsArtifact.centeredPairRows,
    GatePolynomialHolds
      ShiftedTernarySharedSlotsArtifact.centeredPairPolynomialTerms
      (centeredPairPolynomialValues
        (encoded ShiftedTernarySharedSlotsArtifact.oneColumn)
        (encoded pair.2.1 % goldilocksP)
        (encoded pair.2.2 % goldilocksP))) ∧
  (∀ tail ∈ ShiftedTernarySharedSlotsArtifact.centeredTailRows,
    GatePolynomialHolds
      ShiftedTernarySharedSlotsArtifact.centeredTailPolynomialTerms
      (centeredTailPolynomialValues
        (encoded ShiftedTernarySharedSlotsArtifact.oneColumn)
        (encoded tail.2 % goldilocksP))) ∧
  (∀ row ∈ ShiftedTernarySharedSlotsArtifact.retainedProductRows,
    GatePolynomialHolds
      ShiftedTernarySharedSlotsArtifact.singleProductPolynomialTerms
      (singleProductPolynomialValues
        (encoded ShiftedTernarySharedSlotsArtifact.oneColumn)
        (lcEval encoded row.a) (lcEval encoded row.b)
        (lcEval encoded row.c)))

/-- The actual exported 103-row polynomial/matrix specialization is exactly
the 123-obligation logical `ProductionAccepts` predicate, once the selector's
ONE slot is fixed and the named projective nonresidue premise is supplied. -/
theorem artifactGateAccepts_iff_productionAccepts
    (projectiveNonresidue : ProjectiveSevenNonresidue)
    (encoded : Nat → Nat)
    (one : encoded ShiftedTernarySharedSlotsArtifact.oneColumn = 1) :
    ArtifactGateAccepts encoded ↔ ProductionAccepts encoded := by
  constructor
  · rintro ⟨pairs, tails, products⟩
    constructor
    · unfold ProductionCenteredAccepts
      rw [← centered_pair_tail_schedule_exact.1]
      apply (artifactCenteredSemantics_iff_columns encoded).mp
      constructor
      · intro pair member
        apply (centeredPairPolynomialHolds_iff projectiveNonresidue _ _).mp
        simpa [one] using pairs pair member
      · intro tail member
        apply (centeredTailPolynomialHolds_iff _).mp
        simpa [one] using tails tail member
    · intro row member
      apply (singleProductPolynomialHolds_iff encoded row).mp
      simpa [one] using products row member
  · rintro ⟨centered, products⟩
    have semantic : ArtifactCenteredSemantics encoded := by
      apply (artifactCenteredSemantics_iff_columns encoded).mpr
      rw [centered_pair_tail_schedule_exact.1]
      exact centered
    rcases semantic with ⟨pairSemantics, tailSemantics⟩
    exact ⟨by
      intro pair member
      rw [one]
      exact (centeredPairPolynomialHolds_iff
        projectiveNonresidue _ _).mpr (pairSemantics pair member), by
      intro tail member
      rw [one]
      exact (centeredTailPolynomialHolds_iff _).mpr
        (tailSemantics tail member), by
      intro row member
      rw [one]
      exact (singleProductPolynomialHolds_iff encoded row).mpr
        (products row member)⟩

theorem productionCenteredAccepts_iff (encoded : Nat → Nat) :
    ProductionCenteredAccepts encoded ↔
      ∀ index, index < digitCount →
        ShiftedTernaryReducedCore.CenteredUnitGateHolds
          (decodedAssignment productionLayout encoded
            (ShiftedTernary.digitCols.getD index 0)) := by
  rw [productionCenteredAccepts_indexed]
  constructor <;> intro accepts index indexLt
  · rw [production_decoded_digit encoded indexLt]
    exact accepts index indexLt
  · have accepted := accepts index indexLt
    rw [production_decoded_digit encoded indexLt] at accepted
    exact accepted

/-- Logical equivalence between the 123 retained obligations and the generic
reduced model. This theorem does not claim that the 20 residual-pair rows
enforce their 40 centered obligations; that physical bridge is the conditional
`artifactGateAccepts_iff_productionAccepts` theorem above. -/
theorem productionAccepts_iff_reduced (encoded : Nat → Nat) :
    ProductionAccepts encoded ↔
      ShiftedTernaryReducedCore.Accepts
        (decodedAssignment productionLayout encoded) := by
  rw [reducedAccepts_iff]
  unfold ProductionAccepts
  rw [productionCenteredAccepts_iff,
    satisfies_retainedProductRows_iff]

theorem decodedAssignment_canonical
    (layout : Layout) (encoded : Nat → Nat) :
    ∀ source, decodedAssignment layout encoded source < goldilocksP := by
  intro source
  unfold decodedAssignment LinearSubstitution.assignment lcEval
  exact Nat.mod_lt _ (by native_decide)

theorem decodedAssignment_one
    {layout : Layout} {encoded : Nat → Nat}
    (one : encoded layout.oneColumn = 1) :
    decodedAssignment layout encoded 0 = 1 := by
  have oneLt : 1 < goldilocksP := by native_decide
  simp [decodedAssignment, LinearSubstitution.assignment, sourceExpansion,
    lcEval, one, Nat.mod_eq_of_lt oneLt]

/-- Logical isolated-opening theorem: under verifier-fixed ONE, the 123
retained obligations accept iff the old complete canonical row relation
accepts. The four omitted families are discharged by `ReducedCore`. Physical
residual-pair rows enter only through the conditional theorem below. -/
theorem productionAccepts_iff_canonicalRows
    (prime : EuclidPrime goldilocksP)
    {encoded : Nat → Nat}
    (one : encoded ShiftedTernarySharedSlotsArtifact.oneColumn = 1) :
    ProductionAccepts encoded ↔
      Satisfies canonicalRows
        (decodedAssignment productionLayout encoded) := by
  have canonical := decodedAssignment_canonical productionLayout encoded
  have decodedOne := decodedAssignment_one
    (layout := productionLayout) (encoded := encoded) one
  constructor
  · intro production
    apply (ShiftedTernaryReducedCore.reduced_iff_canonicalRows
      prime canonical decodedOne).mp
    exact ⟨(productionAccepts_iff_reduced encoded).mp production,
      production_decoded_sharedAlias encoded⟩
  · intro full
    apply (productionAccepts_iff_reduced encoded).mpr
    exact ((ShiftedTernaryReducedCore.reduced_iff_canonicalRows
      prime canonical decodedOne).mpr full).1

/-- Conditional physical-row theorem for the exact schema-3 artifact. The 103
production rows (20 residual pairs, one ordinary tail, and 82 products) accept
iff all 124 canonical shifted-ternary rows accept. The hard field-instantiation
premise is intentionally visible at this final row-equivalence boundary. -/
theorem artifactGateAccepts_iff_canonicalRows
    (projectiveNonresidue : ProjectiveSevenNonresidue)
    (prime : EuclidPrime goldilocksP)
    {encoded : Nat → Nat}
    (one : encoded ShiftedTernarySharedSlotsArtifact.oneColumn = 1) :
    ArtifactGateAccepts encoded ↔
      Satisfies canonicalRows
        (decodedAssignment productionLayout encoded) := by
  rw [artifactGateAccepts_iff_productionAccepts
      projectiveNonresidue encoded one,
    productionAccepts_iff_canonicalRows prime one]

theorem canonicalOpening_of_production
    (prime : EuclidPrime goldilocksP)
    {encoded : Nat → Nat}
    (one : encoded ShiftedTernarySharedSlotsArtifact.oneColumn = 1)
    (accepted : ProductionAccepts encoded) :
    CanonicalOpening (decodedAssignment productionLayout encoded) := by
  apply ShiftedTernarySound.canonicalOpening_of_canonicalRows prime
    (decodedAssignment_canonical productionLayout encoded)
    (decodedAssignment_one (layout := productionLayout)
      (encoded := encoded) one)
  exact (productionAccepts_iff_canonicalRows prime one).mp accepted

/-- Honest completeness for an already materialized production assignment. -/
theorem production_complete
    {encoded : Nat → Nat}
    (witness : ShiftedTernaryComplete.CanonicalWitness
      (decodedAssignment productionLayout encoded)) :
    ProductionAccepts encoded := by
  apply (productionAccepts_iff_reduced encoded).mpr
  exact
    (ShiftedTernaryReducedCore.CanonicalWitness.reducedCore_complete
      witness).1

end Nightstream.Implementation.R1CS.ShiftedTernarySharedSlots
