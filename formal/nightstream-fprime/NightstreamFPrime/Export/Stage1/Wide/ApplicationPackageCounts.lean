import NightstreamFPrime.Export.Stage1.Wide.ApplicationPackage
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

/-! The selected hash-chain application's physical footprint is unchanged
when its caller-owned and private columns move to the wide source layout. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ApplicationPackageCounts

open NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open Lifecycle.Stage1

abbrev application := Poseidon2HashChainV1Package.application

private theorem privateCount_eq (program : ApplicationPackage.Program) (rowStart : Nat) :
    (ApplicationPackage.plan program rowStart).privateCount =
      localLength (Stage1.ApplicationPackage.operations program (ApplicationPackage.columns program)
        (Layout.Stage1.Wide.SourceOrder.privateColumns + program.witnessWordCount)) +
      R1CS.totalFreshCount (Stage1.ApplicationPackage.constraints program (ApplicationPackage.columns program)
        (Layout.Stage1.Wide.SourceOrder.privateColumns + program.witnessWordCount)) := rfl

private theorem rowCount_eq (program : ApplicationPackage.Program) (rowStart : Nat) :
    (ApplicationPackage.plan program rowStart).rowCount =
      (Stage1.ApplicationPackage.compiledRows program (ApplicationPackage.columns program)
        (Layout.Stage1.Wide.SourceOrder.privateColumns + program.witnessWordCount) rowStart).length := rfl

private theorem affine (start : Nat) :
    Layout.Poseidon2.HashInterfaceAffine
      (Poseidon2HashChainV1.hashInterface (ApplicationPackage.columns application).interface) start := by
  constructor
  · intro expression member
    simp only [Poseidon2HashChainV1.hashInterface, Poseidon2HashChainV1.inputExpressions,
      List.mem_append] at member
    rcases member with (tagMember | inputMember) | witnessMember
    · rcases List.mem_map.mp tagMember with ⟨value, _, rfl⟩
      exact R1CS.isAffine_const _
    · rw [List.mem_ofFn'] at inputMember
      rcases inputMember with ⟨index, rfl⟩
      exact R1CS.isAffine_var _
    · rw [List.mem_ofFn'] at witnessMember
      rcases witnessMember with ⟨index, rfl⟩
      exact R1CS.isAffine_var _
  · intro lane
    exact R1CS.isAffine_var _

theorem plan_privateCount (rowStart : Nat) :
    (ApplicationPackage.plan application rowStart).privateCount = 7696 := by
  rw [privateCount_eq]
  have locals := Poseidon2HashChainV1.program_localLength
    (ApplicationPackage.columns application).interface
    (Layout.Stage1.Wide.SourceOrder.privateColumns + application.witnessWordCount)
  have fresh := Layout.Poseidon2.hashConstraints_freshCount _ _
    (affine (Layout.Stage1.Wide.SourceOrder.privateColumns + application.witnessWordCount))
  change _ = 7696 at locals
  change R1CS.totalFreshCount (Stage1.ApplicationPackage.constraints application
    (ApplicationPackage.columns application) _) = 0 at fresh
  rw [fresh, Nat.add_zero]
  exact locals

theorem plan_rowCount (rowStart : Nat) :
    (ApplicationPackage.plan application rowStart).rowCount = 7700 := by
  rw [rowCount_eq]
  unfold Stage1.ApplicationPackage.compiledRows
  rw [Rows.compileRowsTR_length, Rows.lowerConstraintsTR_eq, R1CS.lowerConstraints_rows_length]
  change R1CS.totalRowCount (Layout.Poseidon2.hashConstraints
    (Poseidon2HashChainV1.hashInterface (ApplicationPackage.columns application).interface) _) = _
  rw [Layout.Poseidon2.hashConstraints_rowCount _ _ (affine _)]
  change (Gadgets.Poseidon2.Hash.inputChunks (Poseidon2HashChainV1.inputExpressions
    (ApplicationPackage.columns application).interface _)).length * 592 + 596 = 7700
  rw [Poseidon2HashChainV1.inputChunks_length]

theorem plan_counts (rowStart : Nat) :
    (ApplicationPackage.plan application rowStart).privateCount =
      (PerApplicationPackage.applicationPlan application).privateCount ∧
    (ApplicationPackage.plan application rowStart).rowCount =
      (PerApplicationPackage.applicationPlan application).rowCount := by
  rw [plan_privateCount, plan_rowCount, Poseidon2HashChainV1Package.applicationPlan_privateCount,
    Poseidon2HashChainV1Package.applicationPlan_rowCount]
  exact ⟨rfl, rfl⟩

private theorem bind_ok {α β : Type} (input : Except String α) (next : α → Except String β) (output : β) :
    (input >>= next) = .ok output ↔ ∃ value, input = .ok value ∧ next value = .ok output := by
  cases input <;> simp [Bind.bind, Except.bind]

private theorem physical_columns (before after : CircuitPackage)
    (built : PhysicalPackage.ofCommon before = .ok after) :
    after.layout.totalColumnCount = Layout.Stage1.Wide.SourceOrder.totalColumns := by
  unfold PhysicalPackage.ofCommon at built
  simp only [bind_ok] at built
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, result⟩ := built
  simp only [Pure.pure, Except.pure, Except.ok.injEq] at result
  subst after
  rfl

private theorem application_columns (program : ApplicationPackage.Program) (base final : CircuitPackage)
    (plan : Stage1.ApplicationPackage.Plan)
    (built : ApplicationPackage.ofBase program base = .ok (final, plan)) :
    final.layout.totalColumnCount = base.layout.totalColumnCount +
      (program.witnessWordCount + (ApplicationPackage.plan program base.layout.rowCount).privateCount) := by
  unfold ApplicationPackage.ofBase at built
  simp only [bind_ok] at built
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, result⟩ := built
  simp only [Pure.pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at result
  obtain ⟨result, _⟩ := result
  subst final
  rfl

theorem package_totalColumnCount (package : CircuitPackage) (plan : Stage1.ApplicationPackage.Plan)
    (built : ApplicationPackage.package application = .ok (package, plan)) :
    package.layout.totalColumnCount = Layout.Stage1.Wide.SourceOrder.totalColumns +
      PerApplicationPackage.addedPrivateColumnCount application := by
  unfold ApplicationPackage.package at built
  rw [bind_ok] at built
  obtain ⟨base, baseBuilt, extended⟩ := built
  have columns := physical_columns (PhysicalPackage.common ()) base baseBuilt
  rw [application_columns application base package plan extended, columns, plan_privateCount]
  unfold PerApplicationPackage.addedPrivateColumnCount
  rw [Poseidon2HashChainV1Package.applicationPlan_privateCount]

end NightstreamFPrime.Export.Stage1.Wide.ApplicationPackageCounts
