import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row.Gating

namespace Tests.FPrimeFullHistorySelectiveCcsSelectorComposition

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Complement
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.RowPointGating
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Necessity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ArtifactRefinement
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Gating

def threeArmRows : Fin 3 → ResidualFamily
  | ⟨0, _⟩ => .ofList [1]
  | ⟨1, _⟩ => .ofList [0, 0]
  | ⟨2, _⟩ => .ofList [1, 0]

theorem bootstrap_rows_zero : RowsZero (threeArmRows 1) := by
  change ∀ row : Fin 2, ([0, 0].get row : F) = 0
  intro row
  rcases (by omega : row.val = 0 ∨ row.val = 1) with rowZero | rowOne
  · have equal : row = (0 : Fin 2) := Fin.eq_of_val_eq rowZero
    subst row
    rfl
  · have equal : row = (1 : Fin 2) := Fin.eq_of_val_eq rowOne
    subst row
    rfl

theorem zeroPairRows_zero : RowsZero (.ofList [0, 0]) := by
  change ∀ row : Fin 2, ([0, 0].get row : F) = 0
  intro row
  rcases (by omega : row.val = 0 ∨ row.val = 1) with rowZero | rowOne
  · have equal : row = (0 : Fin 2) := Fin.eq_of_val_eq rowZero
    subst row
    rfl
  · have equal : row = (1 : Fin 2) := Fin.eq_of_val_eq rowOne
    subst row
    rfl

def groupedArm : Fin 3 → Fin 2
  | ⟨0, _⟩ => 0
  | ⟨1, _⟩ => 1
  | ⟨2, _⟩ => 1

def groupedCommonRows : Fin 2 → ResidualFamily
  | ⟨0, _⟩ => .ofList [1]
  | ⟨1, _⟩ => .ofList [0, 0]

def scheduledKind : Fin 3 → Fin 2
  | ⟨0, _⟩ => 0
  | ⟨1, _⟩ => 1
  | ⟨2, _⟩ => 0

example :
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.Accepts
      groupedArm (unitWeights (1 : Fin 3))
      groupedCommonRows threeArmRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.accepts_complete
    groupedArm groupedCommonRows threeArmRows 1 zeroPairRows_zero
      bootstrap_rows_zero

example (noZeroProducts : NoZeroProducts) :
    (∃ weights,
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.Accepts
        groupedArm weights groupedCommonRows threeArmRows) ↔
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.Selected
        groupedArm groupedCommonRows threeArmRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.exists_accepts_iff_selected
    noZeroProducts groupedArm groupedCommonRows threeArmRows

example :
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.LinkedAccepts
      groupedArm (unitWeights (1 : Fin 3))
        (fun group =>
          Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.groupWeight
            groupedArm (unitWeights (1 : Fin 3)) group)
      groupedCommonRows threeArmRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.linkedAccepts_complete
    groupedArm groupedCommonRows threeArmRows 1 zeroPairRows_zero
      bootstrap_rows_zero

example (noZeroProducts : NoZeroProducts) :
    (∃ weights groupWeights,
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.LinkedAccepts
        groupedArm weights groupWeights groupedCommonRows threeArmRows) ↔
      Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.Selected
        groupedArm groupedCommonRows threeArmRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.exists_linkedAccepts_iff_selected
    noZeroProducts groupedArm groupedCommonRows threeArmRows

example :
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGrouped.LinkedAccepts
      groupedArm scheduledKind (unitWeights (1 : Fin 3))
        (fun group => groupWeight groupedArm (unitWeights (1 : Fin 3)) group)
        (fun kind => groupWeight scheduledKind (unitWeights (1 : Fin 3)) kind)
      groupedCommonRows groupedCommonRows threeArmRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGrouped.linkedAccepts_complete
    groupedArm scheduledKind groupedCommonRows groupedCommonRows threeArmRows 1
      zeroPairRows_zero zeroPairRows_zero bootstrap_rows_zero

example : Accepts (unitWeights (1 : Fin 3)) threeArmRows :=
  accepts_complete threeArmRows 1 bootstrap_rows_zero

example (noZeroProducts : NoZeroProducts) :
    (∃ weights, Accepts weights threeArmRows) ↔
      SelectedBranch threeArmRows :=
  exists_accepts_iff_selectedBranch noZeroProducts threeArmRows

example :
    ComplementAccepts 1 (.ofList [0, 0]) (.ofList [1]) :=
  complementAccepts_complete_base zeroPairRows_zero

example : ¬ SelectedBranch badRows :=
  no_bad_branch_selected

example :
    ∃ weights : Fin 2 → F,
      Accepts weights baseSelectedRows ∧
        ¬ InactiveAdviceZero weights nonzeroInactiveAdvice :=
  inactiveAdviceZero_not_required

example
    (assignment : Fin decodedTotalRow.columns → F)
    (constantOne : assignment totalConstantColumn = 1) :
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean.residual
        decodedTotalRow assignment = 0 ↔
      SelectorTotal (totalWeights assignment) :=
  generated_total_row_iff_selectorTotal assignment constantOne

def validatedGeneratedGate : ValidatedGateRow decodedGatedRow where
  gate := .general
  selectorColumn := gatedSelectorColumn
  shape := by decide

example (assignment : Fin decodedGatedRow.columns → F) :
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean.residual
        decodedGatedRow assignment =
      assignment gatedSelectorColumn *
        Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Semantics.evaluate
          (ungate .general
            (Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean.rowPoint
              decodedGatedRow assignment)) :=
  residual_eq_selector_mul_ungated decodedGatedRow validatedGeneratedGate
    assignment

#check generated_selector_rows_shape
#check generated_gated_row_residual
#check generated_gated_source_residual
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.accepts_sound
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.exists_accepts_iff_semantics
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.linkedAccepts_sound
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommon.exists_linkedAccepts_iff_semantics
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommonArtifact.decode_rawArtifact
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.GroupedCommonArtifact.linkRowsHold_iff
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGrouped.exists_linkedAccepts_iff_selected
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGrouped.exists_linkedAccepts_iff_semantics
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGroupedArtifact.decode_rawArtifact
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGroupedArtifact.linkRowsHold_iff
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGroupedArtifact.composedRowsHold_iff_linkedAccepts
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledLinkedOverlay.exists_linkedAccepts_iff_selected
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledLinkedOverlay.exists_linkedAccepts_iff_semantics
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledLinkedOverlayArtifact.decode_rawArtifact
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledLinkedOverlayArtifact.linkRowsHold_iff
#check Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledLinkedOverlayArtifact.composedRowsHold_iff_linkedAccepts_and_padding
#check evaluate_general_gated
#check evaluate_evaluation_gated
#check residualAt_general_gated
#check residualAt_evaluation_gated
#check ExactRowAction.residualAt_eq_selector_mul_ungated

end Tests.FPrimeFullHistorySelectiveCcsSelectorComposition
