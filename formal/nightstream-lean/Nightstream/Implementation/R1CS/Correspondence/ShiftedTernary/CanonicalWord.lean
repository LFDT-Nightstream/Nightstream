import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.CenteredZero
import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.ShiftedTernaryComplete
import Nightstream.Implementation.R1CS.Ownership.Core.OwnerCertificate

/-!
Determinism of a canonical shifted-ternary word from its source field.

Assurance tier: proved arithmetic semantics plus generic column refinement.
This closes the gap between “some valid canonical opening exists” and “the
41 digits consumed by SIS are the unique native radix-three word of the
source field.”

Owns: bounded radix-three uniqueness for a `CanonicalOpening`; equality of
all accepted digit/negative coordinates to the executable native encoding;
and the exact owner column-map facts for source and digit columns.

Does not own: any particular generated owner; seeded-Phi81 coefficients;
Rust seed expansion; a digest; transcript authority; row necessity; row
removal; or cost totals.

Emits constraints: no.

Authority boundary: the source field and the mathematical canonical-opening
predicate determine the word. No generated witness, commitment output, or
digest is accepted as authority.

| Protocol-neutral phase | Constraint family | Theorem | Exact guarantee |
|---|---|---|---|
| canonical encoding | radix-three uniqueness | `opening_trit_eq_native` | all 41 trits equal the canonical digits of `(x+shift) mod p` |
| canonical encoding | digit witnesses | `opening_digitPair_eq_native` | centered digit and negative flag equal the native pair |
| canonical encoding | field-only word | `nativeDigit_eq_canonicalDigit` | the executable digit depends only on the source field value |
| owner refinement | source placement | `localAssignment_field` | local field coordinate is the named production source column |
| owner refinement | SIS word placement | `localAssignment_digit` | local digit coordinate `i` is production column `digitStart+i` |
-/

namespace Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.ShiftedTernaryComplete

/-- Assignment-free centered radix-three digit of one canonical field value.
This is the word consumed by the abstract SIS linear-map semantics. -/
def canonicalDigit (fieldValue index : Nat) : Nat :=
  match ((fieldValue + shift) % goldilocksP) / 3 ^ index % 3 with
  | 0 => goldilocksP - 1
  | 1 => 0
  | _ => 1

/-- The witness generator's digit function observes no assignment coordinate
other than the source field. -/
theorem nativeDigit_eq_canonicalDigit
    (assignment : Nat -> Nat) (index : Nat) :
    nativeDigit assignment index =
      canonicalDigit (assignment ShiftedTernary.fieldCol) index := by
  rfl

/-- A canonical opening has exactly the native little-endian radix-three
digits of its source field. -/
theorem opening_trit_eq_native
    {assignment : Nat -> Nat}
    (opening : CanonicalOpening assignment) :
    forall index, index < digitCount ->
      assignmentTrit assignment index = nativeTrit assignment index := by
  have lowValuesEqual :
      lowValue (assignmentTrit assignment) digitCount =
        lowValue (nativeTrit assignment) digitCount := by
    calc
      lowValue (assignmentTrit assignment) digitCount =
          encodedValue assignment := (encodedValue_eq_lowValue assignment).symm
      _ = targetValue assignment := by
        simpa [targetValue] using opening.fieldMatches.symm
      _ = lowValue (nativeTrit assignment) digitCount :=
        (lowValue_nativeTrit_full assignment).symm
  exact ShiftedTernaryCenteredZero.lowValue_injective
    (fun index indexLt =>
      Digit.tritValue_lt_three (opening.digits index indexLt))
    (fun index _ => nativeTrit_lt_three assignment index)
    lowValuesEqual

/-- Both coordinates of every centered digit pair are uniquely fixed by the
source field. -/
theorem opening_digitPair_eq_native
    {assignment : Nat -> Nat}
    (opening : CanonicalOpening assignment)
    (index : Nat) (indexLt : index < digitCount) :
    assignment (ShiftedTernary.digitCols.getD index 0) =
        nativeDigit assignment index /\
      assignment (ShiftedTernary.negativeCols.getD index 0) =
        nativeNegative assignment index := by
  have tritEqual := opening_trit_eq_native opening index indexLt
  have digit := opening.digits index indexLt
  cases digit with
  | neg valueEq negativeEq =>
      have nativeZero : nativeTrit assignment index = 0 := by
        unfold assignmentTrit at tritEqual
        rw [valueEq] at tritEqual
        simpa [tritValue, goldilocksP] using tritEqual.symm
      constructor
      · rw [valueEq]
        simp [nativeDigit, nativeZero]
      · rw [negativeEq]
        simp [nativeNegative, nativeZero]
  | zero valueEq negativeEq =>
      have nativeOne : nativeTrit assignment index = 1 := by
        unfold assignmentTrit at tritEqual
        rw [valueEq] at tritEqual
        simpa [tritValue, goldilocksP] using tritEqual.symm
      constructor
      · rw [valueEq]
        simp [nativeDigit, nativeOne]
      · rw [negativeEq]
        simp [nativeNegative, nativeOne]
  | pos valueEq negativeEq =>
      have nativeTwo : nativeTrit assignment index = 2 := by
        unfold assignmentTrit at tritEqual
        rw [valueEq] at tritEqual
        simpa [tritValue, goldilocksP] using tritEqual.symm
      constructor
      · rw [valueEq]
        simp [nativeDigit, nativeTwo]
      · rw [negativeEq]
        simp [nativeNegative, nativeTwo]

def localAssignment
    (assignment : Nat -> Nat) (fieldColumn digitStart : Nat) : Nat -> Nat :=
  Relabel.assignment
    (shiftedTernaryColumnMap fieldColumn digitStart) assignment

@[simp] theorem localAssignment_zero
    (assignment : Nat -> Nat) (fieldColumn digitStart : Nat) :
    localAssignment assignment fieldColumn digitStart 0 = assignment 0 := by
  simp [localAssignment, shiftedTernaryColumnMap,
    Relabel.assignment, Relabel.column]

@[simp] theorem localAssignment_field
    (assignment : Nat -> Nat) (fieldColumn digitStart : Nat) :
    localAssignment assignment fieldColumn digitStart ShiftedTernary.fieldCol =
      assignment fieldColumn := by
  simp [localAssignment, shiftedTernaryColumnMap,
    Relabel.assignment, Relabel.column, ShiftedTernary.fieldCol]

/-- The 41 columns consumed by a seeded-Phi81 word are exactly the local
canonical-opening digit coordinates. -/
theorem digitCols_getD (index : Nat) (indexLt : index < digitCount) :
    ShiftedTernary.digitCols.getD index 0 = 58 + index := by
  have columns : ShiftedTernary.digitCols =
      (List.range digitCount).map fun position => 58 + position := by
    decide
  rw [columns]
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
  rw [List.getElem?_range indexLt]
  simp

theorem columnMap_digit
    (fieldColumn digitStart index : Nat) (indexLt : index < digitCount) :
    Relabel.column (shiftedTernaryColumnMap fieldColumn digitStart)
        (58 + index) = digitStart + index := by
  let headColumns : List Nat := [0, fieldColumn] ++ List.replicate 56 0
  have mapShape : shiftedTernaryColumnMap fieldColumn digitStart =
      headColumns ++
        (List.range 122).map (fun position => digitStart + position) := by
    simp [shiftedTernaryColumnMap, headColumns]
  rw [mapShape]
  unfold Relabel.column
  simp only [List.getD_eq_getElem?_getD]
  rw [List.getElem?_append_right (by simp [headColumns])]
  have headLength : headColumns.length = 58 := by
    simp [headColumns]
  rw [headLength, show 58 + index - 58 = index by omega]
  simp only [List.getElem?_map]
  rw [List.getElem?_range
    (Nat.lt_trans indexLt (by decide : digitCount < 122))]
  simp

theorem localAssignment_digit
    (assignment : Nat -> Nat) (fieldColumn digitStart index : Nat)
    (indexLt : index < digitCount) :
    localAssignment assignment fieldColumn digitStart
        (ShiftedTernary.digitCols.getD index 0) =
      assignment (digitStart + index) := by
  rw [digitCols_getD index indexLt]
  unfold localAssignment Relabel.assignment
  rw [columnMap_digit fieldColumn digitStart index indexLt]

/-- A canonical local opening fixes every actual SIS input column to the
native digit word of its named source field. -/
theorem productionDigit_eq_native
    {assignment : Nat -> Nat} {fieldColumn digitStart : Nat}
    (opening : CanonicalOpening
      (localAssignment assignment fieldColumn digitStart))
    (index : Nat) (indexLt : index < digitCount) :
    assignment (digitStart + index) =
      nativeDigit (localAssignment assignment fieldColumn digitStart) index := by
  rw [← localAssignment_digit assignment fieldColumn digitStart index indexLt]
  exact (opening_digitPair_eq_native opening index indexLt).1

/-- Production form of the field-only word theorem. -/
theorem productionDigit_eq_canonicalDigit
    {assignment : Nat -> Nat} {fieldColumn digitStart : Nat}
    (opening : CanonicalOpening
      (localAssignment assignment fieldColumn digitStart))
    (index : Nat) (indexLt : index < digitCount) :
    assignment (digitStart + index) =
      canonicalDigit (assignment fieldColumn) index := by
  rw [productionDigit_eq_native opening index indexLt]
  rw [nativeDigit_eq_canonicalDigit]
  simp

end Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord
