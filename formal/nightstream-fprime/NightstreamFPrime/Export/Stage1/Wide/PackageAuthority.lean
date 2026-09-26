import NightstreamFPrime.Export.Stage1.Wide.AuthorityStream
import NightstreamFPrime.Export.Stage1.Wide.PhysicalSourceCustody
import NightstreamFPrime.Export.Stage1.Wide.CommonSchedule

/-! Connect the prepared sealed package directly to its proved CCS plan.
The source archive, matrix program and retained transport come from the same
successful constructor; no independent row-equality certificate is supplied. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PackageAuthority

open NightstreamFPrime.Layout NightstreamFPrime.Export.Package
open AuthorityStream

private theorem assembled_children
    (package : Except String (CircuitPackage × Stage1.ApplicationPackage.Plan))
    (program : Except String Layout.MatrixProgram.Program)
    (transportFor : CircuitPackage → Except String AssignmentTransport.Plan)
    (width : Nat) (assemble : CircuitPackage → Layout.MatrixProgram.Program →
      Stage1.ApplicationPackage.Plan → AssignmentTransport.Plan → Parts)
    (parts : Parts)
    (prepared : (do
      let (physical, applicationPlan) ← package
      let matrix ← program
      let transport ← transportFor physical
      unless transport.coordinateCount = width do throw "wide transport width differs from matrix width"
      return assemble physical matrix applicationPlan transport) = .ok parts) :
    ∃ physical applicationPlan matrix transport,
      package = .ok (physical, applicationPlan) ∧ program = .ok matrix ∧
      transportFor physical = .ok transport ∧ transport.coordinateCount = width ∧
      parts = assemble physical matrix applicationPlan transport := by
  cases package with
  | error message => simp only [Bind.bind, Except.bind, reduceCtorEq] at prepared
  | ok value =>
    rcases value with ⟨physical, applicationPlan⟩
    cases program with
    | error message => simp only [Bind.bind, Except.bind, reduceCtorEq] at prepared
    | ok matrix =>
      cases transported : transportFor physical with
      | error message =>
        simp only [Bind.bind, Except.bind, transported, reduceCtorEq] at prepared
      | ok transport =>
        simp only [Bind.bind, Except.bind, transported] at prepared
        by_cases count : transport.coordinateCount = width
        · simp only [count, ↓reduceIte, Pure.pure, Except.pure, Except.ok.injEq] at prepared
          exact ⟨physical, applicationPlan, matrix, transport, rfl, rfl, transported, count, prepared.symm⟩
        · simp only [count, ↓reduceIte, Except.bind, reduceCtorEq] at prepared

theorem prepared_children (compiled : PiRlcWideSampler.RangePlan.Compiled) (parts : Parts)
    (prepared : prepare compiled = .ok parts) :
    ∃ physical applicationPlan matrix transport,
      ApplicationPackage.package application = .ok (physical, applicationPlan) ∧
      PhysicalMatrixSource.program application compiled = .ok matrix ∧
      AssignmentTransport.plan application physical.layout.totalColumnCount = .ok transport ∧
      transport.coordinateCount = RetainedLayout.logicalWidth application ∧
      parts = ofChildren physical matrix applicationPlan transport :=
  assembled_children (ApplicationPackage.package application)
    (PhysicalMatrixSource.program application compiled)
    (fun physical => AssignmentTransport.plan application physical.layout.totalColumnCount)
    (RetainedLayout.logicalWidth application) ofChildren parts prepared

theorem matrix_exact (compiled : PiRlcWideSampler.RangePlan.Compiled) (parts : Parts)
    (prepared : prepare compiled = .ok parts) :
    Layout.MatrixProgram.Exact parts.matrix
      (FixedPoint.structuralPlan application compiled Poseidon2HashChainV1Package.fits)
      (PackageSourceRows.packageSourceRow? parts.package) := by
  obtain ⟨physical, applicationPlan, matrix, transport, built, emitted, _, _, rfl⟩ :=
    prepared_children compiled parts prepared
  exact PhysicalSourceCustody.exact compiled Poseidon2HashChainV1Package.fits
    physical applicationPlan built matrix emitted

theorem physical_width (compiled : PiRlcWideSampler.RangePlan.Compiled) (parts : Parts)
    (prepared : prepare compiled = .ok parts) :
    parts.package.layout.totalColumnCount = CommonSchedule.physicalWidth application := by
  obtain ⟨physical, applicationPlan, matrix, transport, built, _, _, _, rfl⟩ :=
    prepared_children compiled parts prepared
  dsimp only [ofChildren]
  rw [TerminalPackage.install_layout]
  unfold CommonSchedule.physicalWidth
  have counted := ApplicationPackageCounts.package_totalColumnCount physical applicationPlan built
  dsimp only [AuthorityStream.application, ApplicationPackageCounts.application] at counted ⊢
  exact counted

theorem transport_emitted (compiled : PiRlcWideSampler.RangePlan.Compiled) (parts : Parts)
    (prepared : prepare compiled = .ok parts) :
    AssignmentTransport.plan application (CommonSchedule.physicalWidth application) = .ok parts.transport := by
  obtain ⟨physical, applicationPlan, matrix, transport, built, _, emitted, _, rfl⟩ :=
    prepared_children compiled parts prepared
  rw [ApplicationPackageCounts.package_totalColumnCount physical applicationPlan built] at emitted
  dsimp only [ofChildren]
  unfold CommonSchedule.physicalWidth
  dsimp only [AuthorityStream.application, ApplicationPackageCounts.application] at emitted ⊢
  exact emitted

end NightstreamFPrime.Export.Stage1.Wide.PackageAuthority
