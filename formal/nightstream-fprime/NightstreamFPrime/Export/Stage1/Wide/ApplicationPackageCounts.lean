import NightstreamFPrime.Export.Stage1.Wide.ApplicationPackage
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

/-! The selected hash-chain application's physical footprint is unchanged
when its caller-owned and private columns move to the wide source layout. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ApplicationPackageCounts

open NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Export.Package
open Lifecycle.Stage1

abbrev application := Poseidon2HashChainV1Package.application

theorem plan_counts (rowStart : Nat) :
    (ApplicationPackage.plan application rowStart).privateCount =
      (PerApplicationPackage.applicationPlan application).privateCount ∧
    (ApplicationPackage.plan application rowStart).rowCount =
      (PerApplicationPackage.applicationPlan application).rowCount :=
  ApplicationRelocation.plan_counts application rowStart

theorem plan_privateCount (rowStart : Nat) :
    (ApplicationPackage.plan application rowStart).privateCount = 7696 :=
  (plan_counts rowStart).1.trans Poseidon2HashChainV1Package.applicationPlan_privateCount

theorem plan_rowCount (rowStart : Nat) :
    (ApplicationPackage.plan application rowStart).rowCount = 7700 :=
  (plan_counts rowStart).2.trans Poseidon2HashChainV1Package.applicationPlan_rowCount

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
