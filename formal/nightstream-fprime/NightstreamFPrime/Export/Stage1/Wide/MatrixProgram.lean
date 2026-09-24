import NightstreamFPrime.Export.Stage1.Wide.ProductMatrixProgram
import NightstreamFPrime.Export.Stage1.Wide.ReusedMatrixPrograms
import NightstreamFPrime.Layout.PiRlcWideSampler.BatchMatrixProgram
import NightstreamFPrime.Export.Stage1.Wide.FixedPoint

/-! Whole candidate matrix program. Every phase has exact sparse-form
correspondence; source rows for reused phases come from the canonical package. -/

namespace NightstreamFPrime.Export.Stage1.Wide.MatrixProgram

open NightstreamFPrime.Spec NightstreamFPrime.Layout NightstreamFPrime.Lifecycle
open Layout.MatrixProgram ProductionRelation PaperAlgebra Spec.Folding.PiCCS.PaperJoint

abbrev ApplicationProgram := RetainedLayout.Program

def piRlc (application : ApplicationProgram) (compiled : PiRlcWideSampler.RangePlan.Compiled) : Layout.MatrixProgram.Program :=
  (PiRlcWideSampler.BatchMatrix.program compiled (PiRLCGeometry.sampler (Stage1Plan.piRlcInterface application))).append
    (ProductMatrix.matrixProgram application)

def program (application : ApplicationProgram) (compiled : PiRlcWideSampler.RangePlan.Compiled) : Layout.MatrixProgram.Program :=
  ((((((ReusedMatrixPrograms.prefixProgram application).append (piRlc application compiled)).append
    (ReusedMatrixPrograms.piDecProgram application)).append
    (ReusedMatrixPrograms.runningProgram application)).append
    (ReusedMatrixPrograms.applicationProgram application)).append
    (ReusedMatrixPrograms.nextProgram application)).append
    (ReusedMatrixPrograms.publicProgram application)

theorem piRlc_exact (application : ApplicationProgram) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (sourceRow : Nat → Option R1CS.Row) :
    Exact (piRlc application compiled) (Stage1Plan.piRlc application compiled) sourceRow :=
  (PiRlcWideSampler.BatchMatrix.exact compiled
    (PiRLCGeometry.sampler (Stage1Plan.piRlcInterface application)) sourceRow).append
    (ProductMatrix.exact application sourceRow) _

variable {relationWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}

/-- Every decoded candidate row equals the semantic plan, port by port. -/
theorem exact (application : ApplicationProgram) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : ReusedMatrixPrograms.SourceCustody application relation fits.package sourceRow) :
    Exact (program application compiled) (Stage1Plan.plan application compiled relation fits.package) sourceRow := by
  unfold program Stage1Plan.plan Stage1Plan.throughNextPreimage Stage1Plan.throughApplication
    Stage1Plan.beforeApplication Stage1Plan.throughPiDec Stage1Plan.throughPiRlc
  exact ((((((ReusedMatrixPrograms.prefix_exact application relation fits sourceRow custody).append
    (piRlc_exact application compiled sourceRow) _).append
    (ReusedMatrixPrograms.piDec_exact application relation fits sourceRow custody) _).append
    (ReusedMatrixPrograms.running_exact application fits sourceRow) _).append
    (ReusedMatrixPrograms.application_exact application relation fits sourceRow custody) _).append
    (ReusedMatrixPrograms.next_exact application relation fits sourceRow custody) _).append
    (ReusedMatrixPrograms.public_exact application fits sourceRow) _

/-- The canonical source archive supplies all reused rows. No caller-owned
row or old-sampler success premise is needed for whole-program correspondence. -/
theorem fixedPoint_exact (application : ApplicationProgram) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application) :
    Exact (program application compiled) (FixedPoint.structuralPlan application compiled fits)
      (PerApplicationPackageSourceCustody.sourceRow application) := by
  have result := exact application compiled (FixedPoint.relation application compiled fits) fits
    (PerApplicationPackageSourceCustody.sourceRow application)
    (ReusedMatrixPrograms.source_custody application (FixedPoint.relation application compiled fits) fits)
  simpa only [FixedPoint.plan_fixedPoint] using result

end NightstreamFPrime.Export.Stage1.Wide.MatrixProgram
