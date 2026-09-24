import NightstreamFPrime.Layout.Stage1.SpartanValues
import NightstreamFPrime.Layout.Stage1.Wide.RunningTransitionValues
import NightstreamFPrime.Layout.R1CS.ColumnMap

/-! Place the candidate physical prefix in private/constant/public order.
The fixed pilot and PiCCS permutation is reused; only its public suffix moves
to the candidate endpoint. No discarded source interval becomes a column. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.SourceOrder

open NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1

def sourceWidth : Nat := RunningTransitionLayout.physicalEnd
def privateColumns : Nat := sourceWidth - Spartan.publicColumnCount
def constantColumn : Nat := privateColumns
def totalColumns : Nat := sourceWidth + 1

theorem sourceWidth_eq : sourceWidth = 27859538 := rfl
theorem privateColumns_eq : privateColumns = 27859260 := rfl
theorem totalColumns_eq : totalColumns = 27859539 := rfl

theorem sourceWidth_le_reference : sourceWidth ≤ Spartan.SourceColumnCount := by decide

def relocate (column : Nat) : Nat :=
  if Spartan.privateColumnCount ≤ column then
    privateColumns + (column - Spartan.privateColumnCount)
  else column

def expand (column : Nat) : Nat :=
  if privateColumns ≤ column then
    Spartan.privateColumnCount + (column - privateColumns)
  else column

def column (source : Nat) : Nat := relocate (Spartan.sourceToSpartan source)

/-- Every live source maps either into the retained private prefix or the
public suffix. The removed interval cannot contain a mapped source. -/
theorem reference_region (source : Nat) (bounded : source < sourceWidth) :
    Spartan.sourceToSpartan source < privateColumns ∨
      Spartan.privateColumnCount ≤ Spartan.sourceToSpartan source := by
  rw [sourceWidth_eq] at bounded
  rw [privateColumns_eq]
  unfold Spartan.sourceToSpartan
  split
  · unfold Spartan.liftPilotColumn
    split
    · left
      rename_i below
      norm_num [Spartan.pilotInputPrivateColumnCount] at below
      omega
    · split
      · left
        rename_i below
        norm_num [Spartan.pilotPrivateColumnCount, Spartan.proofInputColumnCount] at below ⊢
        omega
      · right
        omega
  · split
    · right
      unfold Spartan.expectedContextPublicStart
      omega
    · split
      · left
        rename_i below
        norm_num [Spartan.piCcsPhaseOffset, Spartan.pilotInputPrivateColumnCount,
          Spartan.proofInputSourceStart] at below ⊢
        omega
      · left
        norm_num [Spartan.piCcsLocalStart, Spartan.piCcsPhaseOffset]
        omega

theorem expand_relocate (value : Nat)
    (live : value < privateColumns ∨ Spartan.privateColumnCount ≤ value) :
    expand (relocate value) = value := by
  have smaller : privateColumns ≤ Spartan.privateColumnCount := by decide
  unfold expand relocate
  split_ifs <;> omega

theorem column_lt (source : Nat) (bounded : source < sourceWidth) :
    column source < totalColumns := by
  have upper := Spartan.sourceToSpartan_lt source (Nat.lt_of_lt_of_le bounded sourceWidth_le_reference)
  have live := reference_region source bounded
  rw [Spartan.spartanColumnCount_eq] at upper
  unfold column relocate
  rw [privateColumns_eq, totalColumns_eq, Spartan.privateColumnCount_eq]
  rw [privateColumns_eq, Spartan.privateColumnCount_eq] at live
  split_ifs <;> omega

theorem column_ne_constant (source : Nat) (bounded : source < sourceWidth) :
    column source ≠ constantColumn := by
  have different := Spartan.sourceToSpartan_ne_constant source
    (Nat.lt_of_lt_of_le bounded sourceWidth_le_reference)
  have live := reference_region source bounded
  change Spartan.sourceToSpartan source ≠ Spartan.privateColumnCount at different
  unfold column relocate constantColumn
  split_ifs <;> omega

theorem column_injective {left right : Nat} (leftBound : left < sourceWidth)
    (rightBound : right < sourceWidth) (same : column left = column right) : left = right := by
  have lifted := congrArg expand same
  change expand (relocate (Spartan.sourceToSpartan left)) =
    expand (relocate (Spartan.sourceToSpartan right)) at lifted
  rw [expand_relocate _ (reference_region left leftBound),
    expand_relocate _ (reference_region right rightBound)] at lifted
  exact Spartan.sourceToSpartan_injective
    (Nat.lt_of_lt_of_le leftBound sourceWidth_le_reference)
    (Nat.lt_of_lt_of_le rightBound sourceWidth_le_reference) lifted

def pullback (target : Env) : Env := fun source => target (column source)

def remapRow (row : R1CS.Row) : R1CS.Row := R1CS.mapRowColumns column row

theorem remapRow_holds (target : Env) (row : R1CS.Row) :
    (remapRow row).Holds target ↔ row.Holds (pullback target) :=
  R1CS.mapRowColumns_holds column row target

def remapRows (rows : List R1CS.Row) : List R1CS.Row := rows.map remapRow

theorem remapRows_holds (target : Env) (rows : List R1CS.Row) :
    R1CS.RowsHold target (remapRows rows) ↔ R1CS.RowsHold (pullback target) rows := by
  simp only [R1CS.RowsHold, remapRows, List.mem_map, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂, remapRow_holds]

end NightstreamFPrime.Layout.Stage1.Wide.SourceOrder
