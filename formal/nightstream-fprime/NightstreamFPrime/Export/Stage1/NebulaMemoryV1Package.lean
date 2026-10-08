import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Export.Stage1.NebulaMemoryPackage
import NightstreamFPrime.Lifecycle.Nebula.FirstPlan

/-!
Owns the package of the first Nebula memory application
(`Lifecycle/Nebula/FirstPlan.lean`): the polynomial rows' multiplication and
row counts, the Stage 1 size bounds, the recursive fixed point, and the `2^28`
domain. The plan-independent geometry is in `NebulaMemoryPackage`.
-/

namespace NightstreamFPrime.Export.Stage1.NebulaMemoryV1Package

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Nebula

/-- The only application program selected by this package. -/
def application : Lifecycle.Stage1.Application.Program :=
  NebulaMemoryPackage.application FirstPlan.plan FirstPlan.two

theorem polyRows_mulCount : ((Rows.polyRows FirstPlan.plan).map Layout.R1CS.mulCount).sum ≤ 100000 := by
  decide +kernel

theorem polyRows_length : (Rows.polyRows FirstPlan.plan).length ≤ 10000 := by
  rw [← Rows.names_count]
  decide

theorem rows_le :
    (PerApplicationPackage.applicationPlan (NebulaMemoryPackage.application FirstPlan.plan FirstPlan.two)).rowCount ≤ 256083468 :=
  NebulaMemoryPackage.rows_le FirstPlan.plan FirstPlan.two polyRows_mulCount polyRows_length (by decide +kernel)

theorem columns_le :
    PerApplicationPackage.addedPrivateColumnCount (NebulaMemoryPackage.application FirstPlan.plan FirstPlan.two) ≤ 255992239 :=
  NebulaMemoryPackage.columns_le FirstPlan.plan FirstPlan.two polyRows_mulCount (by decide +kernel)

theorem carrier_le :
    (NebulaMemoryPackage.application FirstPlan.plan FirstPlan.two).witnessWordCount +
      ApplicationRetainedBlocks.localCount (NebulaMemoryPackage.application FirstPlan.plan FirstPlan.two) ≤ 5340302 :=
  NebulaMemoryPackage.carrier_le FirstPlan.plan FirstPlan.two polyRows_mulCount (by decide +kernel)

/-- All physical-package, retained-carrier, and recursive-plan bounds for the
approved `2^28` profile. -/
def fits : PerApplicationFixedPoint.FitsTwoPow28 application :=
  PerApplicationFixedPoint.fitsTwoPow28OfApplicationBounds (NebulaMemoryPackage.application FirstPlan.plan FirstPlan.two) rows_le columns_le
    carrier_le

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

theorem matrixProgram_exact :
    PerApplicationMatrixProgramSemantics.Exact
      (PerApplicationMatrixProgram.matrixProgram application)
      (PerApplicationFixedPoint.structuralPlan application fits)
      (PerApplicationCanonicalPackage.sourceRow application fits) :=
  PerApplicationCanonicalPackage.matrixProgram_exact application fits

/-- The named assertion row ranges of this package (`NebulaMemoryPackage`). -/
def namedRowRanges : List (String × ℕ × ℕ) :=
  NebulaMemoryPackage.namedRowRanges FirstPlan.plan FirstPlan.two

end NightstreamFPrime.Export.Stage1.NebulaMemoryV1Package
