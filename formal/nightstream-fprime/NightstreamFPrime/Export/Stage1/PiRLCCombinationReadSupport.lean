import NightstreamFPrime.Export.Stage1.PiRLCCombinationWitnessReadSupport
import NightstreamFPrime.Export.Stage1.PerApplicationCachedShift

/-! Column-map support shared by the canonical stored-event owners. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationReadSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Package
open PiRLCCombinationWitnessReadSupport (Outside)

theorem shifted_outside (application : Stage1.Application.Program) (column : Nat)
    (outside : Outside column) :
    Outside (PerApplicationPackage.shiftColumn application column) := by
  unfold PerApplicationPackage.shiftColumn
  split_ifs with before
  · exact outside
  · right
    have constant := Package.circuitPackage_layout_values.2.2.1
    change PerApplicationPackage.basePackage.layout.constantColumn = 28784740 at constant
    rw [constant] at before
    change 28421264 ≤ _
    omega

theorem mapped_outside (column : Nat) (bounded : column < Spartan.SourceColumnCount)
    (outside : column < PiRLCStarts.commitmentFreshStart ∨
      PiRLCStarts.outputFreshStart ≤ column) : Outside (Spartan.sourceToSpartan column) := by
  by_contra failure
  have lower : 20572364 ≤ Spartan.sourceToSpartan column := by
    change ¬ (_ < 20572364 ∨ 28421264 ≤ _) at failure
    omega
  have upper : Spartan.sourceToSpartan column < 28421264 := by
    change ¬ (_ < 20572364 ∨ 28421264 ≤ _) at failure
    omega
  have inverse := Spartan.spartanToSource_sourceToSpartan column bounded
  unfold Spartan.spartanToSource at inverse
  rw [if_neg (by change ¬ _ < 98786; omega),
    if_neg (by change ¬ _ < 128074; omega),
    if_neg (by change ¬ _ < 14751526; omega),
    if_pos (by rw [Spartan.privateColumnCount_eq]; omega)] at inverse
  have coordinate := Option.some.inj inverse
  change 14751804 + (Spartan.sourceToSpartan column - 14751526) = column at coordinate
  change column < 20572642 ∨ 28421542 ≤ column at outside
  omega

theorem mapped_before (column : Nat) (before : column < PiRLCStarts.commitmentFreshStart) :
    Outside (Spartan.sourceToSpartan column) := by
  apply mapped_outside column _ (Or.inl before)
  exact Nat.lt_of_lt_of_le before
    (Nat.le_trans (by decide) Spartan.sourceColumnCount_ge_piDecPhaseOffset)

theorem shiftedSource_before (application : Stage1.Application.Program) (column : Nat)
    (before : column < PiRLCStarts.commitmentFreshStart) :
    Outside (PerApplicationPackage.shiftColumn application (Spartan.sourceToSpartan column)) :=
  shifted_outside application _ (mapped_before column before)

theorem mapCombination_supported (column : Nat → Nat)
    (allowed target : Nat → Prop) (combination : R1CS.LinearCombination)
    (support : combination.VarsSatisfy allowed)
    (maps : ∀ index, allowed index → target (column index)) :
    (R1CS.mapCombinationColumns column combination).VarsSatisfy target := by
  intro term member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  exact maps source.1 (support source sourceMember)

theorem shiftSparseCombination_supported (application : Stage1.Application.Program)
    (combination : SparseCombination) (support : combination.toR1CS.VarsSatisfy Outside) :
    (PerApplicationPackage.shiftSparseCombination application combination).toR1CS.VarsSatisfy
      Outside := by
  rw [PerApplicationPackage.shiftSparseCombination_toR1CS]
  exact mapCombination_supported _ Outside Outside _ support (shifted_outside application)

theorem cachedShiftSparseCombination_supported (context : PerApplicationCachedShift.Context)
    (combination : SparseCombination) (support : combination.toR1CS.VarsSatisfy Outside) :
    (PerApplicationCachedShift.shiftSparseCombination context combination).toR1CS.VarsSatisfy
      Outside := by
  rw [PerApplicationCachedShift.shiftSparseCombination_eq]
  exact shiftSparseCombination_supported context.program combination support

theorem remapExpr_supported (expression : Expr)
    (supported : expression.VarsSatisfy (fun column => Outside (Spartan.sourceToSpartan column))) :
    (WitnessProgram.remapExpr expression).VarsSatisfy Outside := by
  induction expression with
  | const => trivial
  | var => exact supported
  | add left right leftIH rightIH => exact ⟨leftIH supported.1, rightIH supported.2⟩
  | mul left right leftIH rightIH => exact ⟨leftIH supported.1, rightIH supported.2⟩

theorem remapBatch_supported (batch : WitnessBatch)
    (supported : batch.ReadsSatisfy (fun column => Outside (Spartan.sourceToSpartan column))) :
    (WitnessProgram.remapBatch batch).ReadsSatisfy Outside := by
  constructor
  · intro expression member
    rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
    exact remapExpr_supported source (supported.1 source sourceMember)
  · intro hint member
    rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
    have support := remapExpr_supported source.source (supported.2 source sourceMember)
    cases source <;> exact support

/-- The pilot lift has no image in the later product scratch interval. -/
theorem liftPilotColumn_outside (column : Nat) : Outside (Spartan.liftPilotColumn column) := by
  unfold Spartan.liftPilotColumn
  split_ifs with input privateColumn
  · left
    change column < 98786 at input
    change column < 20572364
    omega
  · left
    change column < 14722238 at privateColumn
    change column + 29288 < 20572364
    omega
  · right
    rw [Spartan.privateColumnCount_eq]
    change 28421264 ≤ 28784740 + _
    omega

theorem liftPilotExpr_supported (expression : Expr) :
    (Data.liftPilotExpr expression).VarsSatisfy Outside := by
  induction expression with
  | const => trivial
  | var column => exact liftPilotColumn_outside column
  | add left right leftIH rightIH => exact ⟨leftIH, rightIH⟩
  | mul left right leftIH rightIH => exact ⟨leftIH, rightIH⟩

theorem liftPilotBatch_supported (batch : WitnessBatch) :
    (Data.liftPilotBatch batch).ReadsSatisfy Outside := by
  constructor
  · intro expression member
    rcases List.mem_map.mp member with ⟨source, _, rfl⟩
    exact liftPilotExpr_supported source
  · intro hint member
    rcases List.mem_map.mp member with ⟨source, _, rfl⟩
    cases source <;> exact liftPilotExpr_supported _

theorem liftPilotCombination_supported (combination : SparseCombination) :
    (Data.liftPilotCombination combination).toR1CS.VarsSatisfy Outside := by
  intro term member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  rcases List.mem_map.mp sourceMember with ⟨original, _, rfl⟩
  exact liftPilotColumn_outside original.column

theorem instruction_supported (rows : List Rows.CompiledRow) (allowed : Nat → Prop)
    (supported : ∀ row ∈ rows.map Rows.CompiledRow.toR1CS, row.VarsSatisfy allowed)
    (instruction : WitnessInstruction) (member : instruction ∈ Rows.witnessInstructionsTR rows) :
    instruction.a.toR1CS.VarsSatisfy allowed ∧ instruction.b.toR1CS.VarsSatisfy allowed := by
  rw [Rows.witnessInstructionsTR_eq, Rows.witnessInstructions_member] at member
  have support := supported _ (List.mem_map.mpr ⟨.witness instruction, member, rfl⟩)
  exact ⟨support.1, support.2.1⟩

theorem shiftInstruction_supported (context : PerApplicationCachedShift.Context)
    (instruction : WitnessInstruction)
    (supported : instruction.a.toR1CS.VarsSatisfy Outside ∧
      instruction.b.toR1CS.VarsSatisfy Outside) :
    (PerApplicationCachedShift.shiftWitnessInstruction context instruction).a.toR1CS.VarsSatisfy
        Outside ∧
      (PerApplicationCachedShift.shiftWitnessInstruction context instruction).b.toR1CS.VarsSatisfy
        Outside :=
  ⟨cachedShiftSparseCombination_supported context instruction.a supported.1,
    cachedShiftSparseCombination_supported context instruction.b supported.2⟩

end NightstreamFPrime.Export.Stage1.PiRLCCombinationReadSupport
