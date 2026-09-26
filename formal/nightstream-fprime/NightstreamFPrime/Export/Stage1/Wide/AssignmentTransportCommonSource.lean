import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCommon
import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportSemantics

/-! Checked transport addresses recover the proved common source view.
Emission rejects private source values that this view deliberately discards. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCommonSource

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open Layout.Stage1 Layout.Stage1.Wide
open AssignmentTransport AssignmentTransportSemantics

private theorem constant_eq : PerApplicationPackage.basePackage.layout.constantColumn =
    Spartan.privateColumnCount := by
  rw [Spartan.privateColumnCount_eq]
  exact Package.circuitPackage_layout_values.2.2.1

private theorem pilot_source_bound (column source : Nat)
    (read : PilotSpartan.spartanToSource column = some source) : source < 14722512 := by
  simp only [PilotSpartan.spartanToSource, PilotSpartan.secondPrivateStart_value,
    PilotSpartan.witnessPrivateStart_value, PilotSpartan.privateColumnCount_value,
    PilotSpartan.constantColumn_value, PilotSpartan.secondPublicStart_value,
    PilotSpartan.firstPublicStart_value, PilotSpartan.spartanColumnCount_value,
    PilotSpartan.priorPublicStart_value, PilotSpartan.outputPreimageStart_value,
    PilotSpartan.outputDigestStart_value, PilotSpartan.witnessStart_value] at read
  split_ifs at read <;> simp only [Option.some.injEq] at read <;> omega

private theorem public_source_prefix (column source : Nat)
    (afterPrivate : Spartan.privateColumnCount ≤ column)
    (read : Spartan.spartanToSource column = some source) : source < SourceAssignment.prefixEnd := by
  have afterNumber : 28784740 ≤ column := by simpa using afterPrivate
  change source < 19513117
  unfold Spartan.spartanToSource at read
  rw [if_neg (by change ¬column < 98786; omega),
    if_neg (by change ¬column < 128074; omega),
    if_neg (by change ¬column < 14751526; omega),
    if_neg (Nat.not_lt.mpr afterPrivate)] at read
  by_cases constant : column = Spartan.constantColumn
  · rw [if_pos constant] at read
    cases read
  · rw [if_neg constant] at read
    by_cases pilot : column < Spartan.expectedContextPublicStart
    · rw [if_pos pilot] at read
      have bound := pilot_source_bound _ _ read
      omega
    · rw [if_neg pilot] at read
      by_cases bounded : column < Spartan.spartanColumnCount
      · rw [if_pos bounded] at read
        have equal := Option.some.inj read
        rw [Spartan.spartanColumnCount_eq] at bounded
        rw [Spartan.expectedContextPublicStart, Spartan.privateColumnCount_eq] at equal
        change 14722512 + (column - 28785015) = source at equal
        omega
      · rw [if_neg bounded] at read
        cases read

private theorem public_value (env : Env) (column : Nat)
    (afterPrivate : Spartan.privateColumnCount ≤ column) :
    prefixValues env (SourceOrder.privateColumns + (column - Spartan.privateColumnCount)) =
      SourceAssignment.targetEnv env column := by
  have expansion := SourceOrder.expand_relocate column (Or.inr afterPrivate)
  rw [SourceOrder.relocate, if_pos afterPrivate] at expansion
  unfold prefixValues SourceAssignment.targetEnv
  have sameConstant : SourceOrder.privateColumns + (column - Spartan.privateColumnCount) =
      SourceOrder.constantColumn ↔ column = Spartan.constantColumn := by
    change _ = SourceOrder.privateColumns ↔ column = Spartan.privateColumnCount
    omega
  by_cases isConstant : column = Spartan.constantColumn
  · rw [if_pos (sameConstant.mpr isConstant), if_pos isConstant]
  · rw [if_neg (fun same => isConstant (sameConstant.mp same)), if_neg isConstant, expansion]
    cases read : Spartan.spartanToSource column with
    | none => rfl
    | some source =>
      simp only [Option.map_some, Option.getD_some]
      exact (SourceAssignment.sourceEnv_prefix env source (public_source_prefix column source afterPrivate read)).symm

theorem finalColumn_private (source target : Nat) (before : source < Spartan.privateColumnCount)
    (mapped : finalColumn source = .ok target) :
    ∃ original current, Spartan.spartanToSource source = some original ∧
      SourceAssignment.source? original = some current ∧
      SourceOrder.column current = target ∧ target < SourceOrder.privateColumns := by
  rw [finalColumn, if_neg (Nat.not_le.mpr before)] at mapped
  cases original : Spartan.spartanToSource source with
  | none => rw [original] at mapped; cases mapped
  | some originalSource =>
    rw [original] at mapped
    dsimp only at mapped
    cases current : SourceAssignment.source? originalSource with
    | none => rw [current] at mapped; cases mapped
    | some currentSource =>
      rw [current] at mapped
      dsimp only at mapped
      split at mapped
      · rename_i bounded
        have equal := Except.ok.inj mapped
        exact ⟨originalSource, currentSource, rfl, current, equal, equal ▸ bounded⟩
      · cases mapped

theorem base_value (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (source : Fin (PiRLCProductPlan.baseSourceWidth program)) (target : Nat)
    (mapped : finalColumn source.val = .ok target) :
    physicalValues program env application target = (SourceAssignment.raw program env application).base source := by
  change _ = PerApplicationSourceAssignment.ofCompleted program (SourceAssignment.targetEnv env) application source
  by_cases suffix : Spartan.privateColumnCount ≤ source.val
  · rw [finalColumn, if_pos suffix] at mapped
    have equal := Except.ok.inj mapped
    subst target
    unfold physicalValues PerApplicationSourceAssignment.ofCompleted
    have oldBefore : ¬source.val < PerApplicationPackage.basePackage.layout.constantColumn := by
      rw [constant_eq]
      omega
    have newBefore : ¬SourceOrder.privateColumns + (source.val - Spartan.privateColumnCount) <
        SourceOrder.privateColumns := by omega
    rw [dif_neg newBefore, dif_neg oldBefore]
    by_cases inside : source.val < Spartan.privateColumnCount + PerApplicationPackage.addedPrivateColumnCount program
    · have oldInside : source.val < PerApplicationPackage.basePackage.layout.constantColumn +
          PerApplicationPackage.addedPrivateColumnCount program := by rwa [constant_eq]
      have newInside : SourceOrder.privateColumns + (source.val - Spartan.privateColumnCount) <
          SourceOrder.privateColumns + PerApplicationPackage.addedPrivateColumnCount program := by omega
      rw [dif_pos oldInside, dif_pos newInside]
      apply congrArg application
      apply Fin.ext
      dsimp only
      rw [constant_eq]
      omega
    · have oldInside : ¬source.val < PerApplicationPackage.basePackage.layout.constantColumn +
          PerApplicationPackage.addedPrivateColumnCount program := by rwa [constant_eq]
      have newInside : ¬SourceOrder.privateColumns + (source.val - Spartan.privateColumnCount) <
          SourceOrder.privateColumns + PerApplicationPackage.addedPrivateColumnCount program := by omega
      rw [dif_neg oldInside, dif_neg newInside]
      have relocated : SourceOrder.privateColumns + (source.val - Spartan.privateColumnCount) -
          PerApplicationPackage.addedPrivateColumnCount program =
          SourceOrder.privateColumns +
            (source.val - PerApplicationPackage.addedPrivateColumnCount program - Spartan.privateColumnCount) := by omega
      rw [relocated]
      exact public_value env _ (by omega)
  · have before : source.val < Spartan.privateColumnCount := Nat.lt_of_not_ge suffix
    obtain ⟨original, current, read, recovered, equal, bounded⟩ := finalColumn_private source.val target before mapped
    rw [← equal, physicalValues_private_source program env application current
      (SourceAssignment.source?_lt original current recovered) (by rwa [equal])]
    unfold PerApplicationSourceAssignment.ofCompleted
    rw [dif_pos (by rw [constant_eq]; exact before)]
    unfold SourceAssignment.targetEnv
    rw [if_neg (by change source.val ≠ Spartan.privateColumnCount; omega), read]
    simp only [Option.map_some, Option.getD_some, SourceAssignment.sourceEnv, recovered]

end NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCommonSource
