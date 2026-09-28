import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPreservation
import NightstreamFPrime.Export.Stage1.RunningTransitionPackage
import NightstreamFPrime.Export.Stage1.Stage1LoweringBridge
import NightstreamFPrime.Layout.Stage1.Preservation

/-!
Owns the reverse row-level bridge for the final package slices whose canonical
R1CS source lists are already explicit: PiDEC, running transition,
application, and NextPreimage.

The remaining Pilot, PiCCS, and PiRLC template families have separate package
representations and remain obligations of the complete prefix bridge. This
module adds no semantic predicate and does not use phase acceptance as a
substitute for physical row satisfaction.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationPhysicalPreservation

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

theorem basePackageRows_imply_piDecPhysical
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (env : Env) (holds : (Data.circuitPackage ()).RowsHold env) :
    NightstreamFPrime.Layout.PiDEC.v1_1.PhysicalHolds relation
      (PiDECInputs.interface Data.logicalWidth Data.publicFits)
      PiDECInputs.phaseOffset (Spartan.pullback env) := by
  have packageRows := Package.circuitPackage_implies_piDecArithmeticRows
    env holds
  have exactRows := PiDECArithmetic.Plan.rows_to_layout
    (PiDECArithmetic.canonicalPlan Data.logicalWidth Data.publicFits)
    (PiDECArithmetic.canonicalLayoutPlan relation)
    (PiDECArithmetic.canonicalPlan_matches relation)
  rw [exactRows] at packageRows
  have physical := (Spartan.remapRows_hold env _).mp packageRows
  simpa [NightstreamFPrime.Layout.PiDEC.v1_1.PhysicalHolds,
    PiDECArithmetic.canonicalLayoutPlan, PiDECArithmetic.phaseInterface]
    using physical

theorem basePackageRows_imply_runningPhysical
    (env : Env) (holds : (Data.circuitPackage ()).RowsHold env) :
    RunningTransitionLayout.PhysicalHolds Data.logicalWidth Data.publicFits
      (Spartan.pullback env) :=
  RunningTransitionPackage.circuitPackage_implies_physicalHolds env holds

theorem packageRows_imply_applicationPhysical
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (holds : (PerApplicationPackage.package program).RowsHold env) :
    Preservation.ApplicationPhysicalHolds program env := by
  let rows := ApplicationPackage.compiledRows program
    (ApplicationPackage.productionColumns program)
    (ApplicationInputs.localStart program)
    PerApplicationPackage.basePackage.layout.rowCount
  have instructions : ∀ instruction ∈ Rows.witnessInstructions rows,
      instruction.Holds env := by
    intro instruction member
    apply holds.2.2.2.1 instruction
    rw [PerApplicationPackage.package_witnessInstructions]
    apply List.mem_append_right
    simpa [PerApplicationPackage.applicationPlan,
      ApplicationPackage.productionPlan, ApplicationPackage.ofProgram,
      rows, Rows.witnessInstructionsTR_eq] using member
  have assertions : ∀ assertion ∈ Rows.assertionRows rows,
      assertion.Holds env := by
    intro assertion member
    apply holds.2.2.2.2 assertion
    rw [PerApplicationPackage.package_assertionRows]
    apply List.mem_append_left
    apply List.mem_append_right
    simpa [PerApplicationPackage.applicationPlan,
      ApplicationPackage.productionPlan, ApplicationPackage.ofProgram,
      rows, Rows.assertionRowsTR_eq] using member
  have compiled : R1CS.RowsHold env
      (rows.map Rows.CompiledRow.toR1CS) :=
    (Rows.compiledRows_hold_iff rows env).mpr ⟨instructions, assertions⟩
  unfold Preservation.ApplicationPhysicalHolds
  rw [Stage1LoweringBridge.applicationRows_eq_sourceRows]
  simpa [ApplicationDirectSource.sourceRows, rows] using compiled

theorem packageRows_imply_nextPreimagePhysical
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (holds : (PerApplicationPackage.package program).RowsHold env) :
    Preservation.NextPreimagePhysicalHolds env := by
  let rowStart := PerApplicationPackage.nextPreimageRowStart program
  let rows := NextPreimagePackage.compiledRows rowStart
  have instructions : ∀ instruction ∈ Rows.witnessInstructionsTR rows,
      instruction.Holds env := by
    intro instruction member
    have empty : Rows.witnessInstructionsTR rows = [] := by
      simpa [rows, rowStart, NextPreimagePackage.witnessInstructions] using
        NextPreimagePackage.witnessInstructions_eq_nil rowStart
    rw [empty] at member
    contradiction
  have assertions : ∀ assertion ∈ Rows.assertionRowsTR rows,
      assertion.Holds env := by
    intro assertion member
    exact holds.2.2.2.2 assertion (by
      rw [PerApplicationPackage.package_assertionRows]
      apply List.mem_append_right
      simpa [rows, rowStart, NextPreimagePackage.assertionRows] using member)
  have compiled : R1CS.RowsHold env
      (rows.map Rows.CompiledRow.toR1CS) :=
    (Rows.compiledRows_hold_iff rows env).mpr (by
      simpa [Rows.witnessInstructionsTR_eq, Rows.assertionRowsTR_eq] using
        And.intro instructions assertions)
  have sourceRows : R1CS.RowsHold env NextPreimagePackage.sourceRows := by
    rw [NextPreimagePackage.sourceRows_eq,
      ← NextPreimagePackage.compiledRows_toR1CS rowStart]
    exact compiled
  unfold Preservation.NextPreimagePhysicalHolds
  rw [Stage1LoweringBridge.nextPreimageRows_eq_sourceRows]
  exact sourceRows

structure PhysicalSlices
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) : Prop where
  piDec : NightstreamFPrime.Layout.PiDEC.v1_1.PhysicalHolds relation
    (PiDECInputs.interface Data.logicalWidth Data.publicFits)
    PiDECInputs.phaseOffset
    (Spartan.pullback (PerApplicationPackage.baseEnv program env))
  running : RunningTransitionLayout.PhysicalHolds Data.logicalWidth
    Data.publicFits
    (Spartan.pullback (PerApplicationPackage.baseEnv program env))
  application : Preservation.ApplicationPhysicalHolds program env
  nextPreimage : Preservation.NextPreimagePhysicalHolds env

theorem packageRows_imply_physicalSlices
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (holds : (PerApplicationPackage.package program).RowsHold env) :
    PhysicalSlices relation program env := by
  have base :=
    PerApplicationCanonicalPreservation.packageRows_imply_validatedPrefix
      program env holds
  exact {
    piDec := basePackageRows_imply_piDecPhysical relation
      (PerApplicationPackage.baseEnv program env) base
    running := basePackageRows_imply_runningPhysical
      (PerApplicationPackage.baseEnv program env) base
    application := packageRows_imply_applicationPhysical program env holds
    nextPreimage := packageRows_imply_nextPreimagePhysical program env holds }

end NightstreamFPrime.Export.Stage1.PerApplicationPhysicalPreservation
