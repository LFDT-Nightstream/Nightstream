import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Export.Stage1.NebulaMemoryPackage
import NightstreamFPrime.Lifecycle.Nebula.SecondPlan

/-!
Owns the package of the second Nebula memory application
(`Lifecycle/Nebula/SecondPlan.lean`, segments of one step): the polynomial
rows' multiplication and row counts, the Stage 1 size bounds, the recursive
fixed point, and the `2^28` domain. The plan-independent geometry is in
`NebulaMemoryPackage`.
-/

namespace NightstreamFPrime.Export.Stage1.NebulaMemoryN1Package

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Nebula

/-- The only application program selected by this package. -/
def application : Lifecycle.Stage1.Application.Program :=
  NebulaMemoryPackage.application ⟨SecondPlan.geometry, SecondPlan.valid, SecondPlan.secure⟩ SecondPlan.app

theorem polyRows_mulCount : ((Rows.polyRows SecondPlan.geometry SecondPlan.app).map Layout.R1CS.mulCount).sum ≤ 300000 := by
  decide +kernel

theorem polyRows_length : (Rows.polyRows SecondPlan.geometry SecondPlan.app).length ≤ 10000 := by
  decide +kernel

theorem rows_le :
    (PerApplicationPackage.applicationPlan (MemoryApp.program SecondPlan.geometry SecondPlan.app)).rowCount ≤ 257057539 :=
  NebulaMemoryPackage.rows_le SecondPlan.geometry SecondPlan.app polyRows_mulCount polyRows_length (by decide +kernel)

theorem columns_le :
    PerApplicationPackage.addedPrivateColumnCount (MemoryApp.program SecondPlan.geometry SecondPlan.app) ≤ 256970581 :=
  NebulaMemoryPackage.columns_le SecondPlan.geometry SecondPlan.app polyRows_mulCount (by decide +kernel)

theorem carrier_le :
    (MemoryApp.program SecondPlan.geometry SecondPlan.app).witnessWordCount +
      ApplicationRetainedBlocks.localCount (MemoryApp.program SecondPlan.geometry SecondPlan.app) ≤ 5480402 :=
  NebulaMemoryPackage.carrier_le SecondPlan.geometry SecondPlan.app polyRows_mulCount (by decide +kernel)

/-- All physical-package, retained-carrier, and recursive-plan bounds for the
approved `2^28` profile. -/
theorem fits : PerApplicationFixedPoint.FitsTwoPow28 application :=
  PerApplicationFixedPoint.fitsTwoPow28OfApplicationBounds (MemoryApp.program SecondPlan.geometry SecondPlan.app) rows_le columns_le
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
  NebulaMemoryPackage.namedRowRanges SecondPlan.geometry SecondPlan.app

end NightstreamFPrime.Export.Stage1.NebulaMemoryN1Package
