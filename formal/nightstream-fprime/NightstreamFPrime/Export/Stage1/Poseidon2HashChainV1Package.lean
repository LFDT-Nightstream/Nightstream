import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Layout.Poseidon2
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1

/-!
Owns the concrete package geometry and recursive fixed point for
`Poseidon2HashChainV1`. It does not select the semantic Ajtai key or authorize
a Rust loader or proof backend.
-/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

/-- The only application program selected by this package. -/
def application : Lifecycle.Stage1.Application.Program :=
  Lifecycle.Stage1.Poseidon2HashChainV1.program

private theorem inputExpressions_affine (offset : Nat) :
    Layout.Poseidon2.ListAffine
      (Lifecycle.Stage1.Poseidon2HashChainV1.inputExpressions
        (Layout.Stage1.ApplicationInputs.interface application) offset) := by
  intro expression member
  simp only [Lifecycle.Stage1.Poseidon2HashChainV1.inputExpressions,
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

theorem hashInterface_affine :
    Layout.Poseidon2.HashInterfaceAffine
      (Lifecycle.Stage1.Poseidon2HashChainV1.hashInterface
        (Layout.Stage1.ApplicationInputs.interface application))
      (Layout.Stage1.ApplicationInputs.localStart application) := by
  constructor
  · exact inputExpressions_affine _
  · intro lane
    exact R1CS.isAffine_var _

theorem constraints_eq_hashConstraints :
    ApplicationPackage.constraints application
        (ApplicationPackage.productionColumns application)
        (Layout.Stage1.ApplicationInputs.localStart application) =
      Layout.Poseidon2.hashConstraints
        (Lifecycle.Stage1.Poseidon2HashChainV1.hashInterface
          (Layout.Stage1.ApplicationInputs.interface application))
        (Layout.Stage1.ApplicationInputs.localStart application) := by
  rfl

theorem constraints_freshCount :
    R1CS.totalFreshCount
      (ApplicationPackage.constraints application
        (ApplicationPackage.productionColumns application)
        (Layout.Stage1.ApplicationInputs.localStart application)) = 0 := by
  rw [constraints_eq_hashConstraints]
  exact Layout.Poseidon2.hashConstraints_freshCount _ _ hashInterface_affine

theorem constraints_rowCount :
    R1CS.totalRowCount
      (ApplicationPackage.constraints application
        (ApplicationPackage.productionColumns application)
        (Layout.Stage1.ApplicationInputs.localStart application)) = 7700 := by
  rw [constraints_eq_hashConstraints,
    Layout.Poseidon2.hashConstraints_rowCount _ _ hashInterface_affine]
  change (NightstreamFPrime.Gadgets.Poseidon2.Hash.inputChunks
    (Lifecycle.Stage1.Poseidon2HashChainV1.inputExpressions
      (Layout.Stage1.ApplicationInputs.interface application)
      (Layout.Stage1.ApplicationInputs.localStart application))).length *
        592 + 596 = 7700
  rw [Lifecycle.Stage1.Poseidon2HashChainV1.inputChunks_length]

@[simp] theorem applicationPlan_rowCount :
    (PerApplicationPackage.applicationPlan application).rowCount = 7700 := by
  rw [PerApplicationPackage.applicationPlan,
    ApplicationPackage.productionPlan_rowCount]
  unfold ApplicationPackage.compiledRows
  rw [Rows.compileRowsTR_length, Rows.lowerConstraintsTR_eq,
    R1CS.lowerConstraints_rows_length]
  exact constraints_rowCount

theorem operations_localLength :
    localLength
      (ApplicationPackage.operations application
        (ApplicationPackage.productionColumns application)
        (Layout.Stage1.ApplicationInputs.localStart application)) = 7696 := by
  unfold ApplicationPackage.operations application
  exact Lifecycle.Stage1.Poseidon2HashChainV1.program_localLength _ _

@[simp] theorem applicationPlan_privateCount :
    (PerApplicationPackage.applicationPlan application).privateCount =
      7696 := by
  rw [PerApplicationPackage.applicationPlan,
    ApplicationPackage.productionPlan_privateCount]
  rw [constraints_freshCount, operations_localLength]

@[simp] theorem addedPrivateColumnCount :
    PerApplicationPackage.addedPrivateColumnCount application = 7700 := by
  rw [PerApplicationPackage.addedPrivateColumnCount,
    applicationPlan_privateCount]
  simp [application, Lifecycle.Stage1.Poseidon2HashChainV1.program,
    Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount]

theorem sourceWidth :
    ApplicationDirectSource.sourceWidth application =
      Layout.Stage1.ApplicationInputs.localStart application + 7696 := by
  unfold ApplicationDirectSource.sourceWidth ApplicationPackage.r1csFreshStart
  rw [constraints_freshCount, operations_localLength]

@[simp] theorem retainedLocalCount :
    ApplicationRetainedBlocks.localCount application = 7696 := by
  unfold ApplicationRetainedBlocks.localCount
    ApplicationRetainedBlocks.sourceWidth
  rw [sourceWidth]
  omega

@[simp] theorem retainedApplicationWordCount :
    application.witnessWordCount +
      ApplicationRetainedBlocks.localCount application = 7700 := by
  rw [retainedLocalCount]
  simp [application, Lifecycle.Stage1.Poseidon2HashChainV1.program,
    Lifecycle.Stage1.Poseidon2HashChainV1.messageWordCount]

/-- All physical-package, retained-carrier, and recursive-plan bounds for the
approved `2^28` profile. -/
def fits : PerApplicationFixedPoint.FitsTwoPow28 application :=
  PerApplicationFixedPoint.fitsTwoPow28OfApplicationBounds application
    (by rw [applicationPlan_rowCount]; norm_num)
    (by rw [addedPrivateColumnCount]; norm_num)
    (by rw [retainedApplicationWordCount]; norm_num)

@[simp] theorem logicalWidth :
    PerApplicationFixedPoint.logicalWidth application = 264627433 := by
  unfold PerApplicationFixedPoint.logicalWidth
  rw [ApplicationRetainedGeometry.completeLogicalWidth_eq_applicationCounts,
    retainedApplicationWordCount]

@[simp] theorem structuralRowCount :
    (PerApplicationFixedPoint.structuralPlan application fits).rowCount =
      6377559 := by
  rw [PerApplicationFixedPoint.structuralPlan_rowCount,
    applicationPlan_rowCount]

@[simp] theorem physicalPackageRowCount :
    (PerApplicationPackage.package application).layout.rowCount = 29225729 := by
  rw [PerApplicationPackage.package_rowCount,
    PerApplicationPackage.basePackage_rowCount_eq,
    applicationPlan_rowCount]

@[simp] theorem physicalPackageTotalColumnCount :
    (PerApplicationPackage.package application).layout.totalColumnCount =
      29344425 := by
  rw [PerApplicationPackage.package_totalColumnCount,
    PerApplicationPackage.basePackage_totalColumnCount_eq,
    addedPrivateColumnCount]

theorem plan_fixedPoint :
    DirectApplicationPrefixPlan.plan
        (PerApplicationFixedPoint.relation application fits)
        fits.package (PerApplicationFixedPoint.geometry application) =
      PerApplicationFixedPoint.structuralPlan application fits :=
  PerApplicationFixedPoint.plan_fixedPoint application fits

theorem jointDomain_le_twoPow28 :
    max (PerApplicationFixedPoint.structuralPlan application fits).rowCount
        (NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
          (PerApplicationFixedPoint.logicalWidth application)) ≤
      2 ^ Lifecycle.cubeVariables :=
  PerApplicationFixedPoint.jointDomain_le_twoPow28 application fits

/-- Canonical physical package with the self-derived recursive relation and
terminal metadata installed. The explicit argument prevents artifact-sized
construction during module initialization. -/
def package (_unit : Unit) : Export.Package.CircuitPackage :=
  PerApplicationCanonicalPackage.package application fits

@[simp] theorem package_relation :
    (package ()).relation =
      PerApplicationCanonicalPackage.recursiveRelation application fits := by
  exact PerApplicationCanonicalPackage.package_relation application fits

theorem matrixProgram_exact :
    PerApplicationMatrixProgramSemantics.Exact
      (PerApplicationMatrixProgram.matrixProgram application)
      (PerApplicationFixedPoint.structuralPlan application fits)
      (PerApplicationCanonicalPackage.sourceRow application fits) :=
  PerApplicationCanonicalPackage.matrixProgram_exact application fits

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
