import NightstreamFPrime.Export.Stage1.Wide.MatrixProjection
import NightstreamFPrime.Export.Stage1.PerApplicationMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PerApplicationPackageSourceCustody

/-! Compact programs for the six reused candidate phases. Ordinary rows have
explicit source custody; no old sampler row is required by that custody. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ReusedMatrixPrograms

open NightstreamFPrime.Layout NightstreamFPrime.Spec NightstreamFPrime.Lifecycle
open Layout.Stage1 ProductionRelation MatrixProgram
open PaperAlgebra Spec.Folding.PiCCS.PaperJoint

abbrev ApplicationProgram := RetainedLayout.Program

variable {relationWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth relationWidth}

structure SourceCustody (application : ApplicationProgram)
    (relation : Lifecycle.ProductionKey.LogicalRelation relationWidth publicFits)
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row) : Prop where
  piCcsOrdinary : ∀ index : Fin 259963, ∀ sourceIndex,
    PiCCSOrdinaryMatrixProgram.rowSchedule.index? index.val =
        some sourceIndex →
      sourceRow sourceIndex = some
        (PerApplicationSourceProjection.basePackageRow application
          (PiCCSOrdinaryDirectSource.programRow relation
            index))
  pilotOrdinary : ∀ index : Fin 1330,
    sourceRow (PilotOrdinaryMatrixProgram.rowIndexAt index) =
      some (PerApplicationSourceProjection.pilotPackageRow application
        (PilotOrdinaryDirectSource.programRow index))
  piDecPublic : ∀ index : Fin 22680,
    sourceRow (PiDECStarts.publicInputRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.publicProgramRow
          relation index))
  piDecCommitment : ∀ index : Fin 1188,
    sourceRow (PiDECStarts.commitmentRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.commitmentProgramRow
          relation index))
  piDecEvalK : ∀ index : Fin 108,
    sourceRow (PiDECStarts.evalKRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.evalKProgramRow
          relation index))
  piDecEvalA : ∀ index : Fin 1512,
    sourceRow (PiDECStarts.evalARowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.evalAProgramRow
          relation index))
  applicationRows : application.compactHashChain = none → ∀ index :
      Fin (ApplicationDirectSource.program application fits).rowCount,
    sourceRow (PerApplicationPackage.basePackage.layout.rowCount + index.val) =
      some ((ApplicationDirectSource.program application fits).row
        index)
  nextPreimage : ∀ index : Fin NextPreimageDirectPlan.program.rowCount,
    sourceRow
        (PerApplicationPackage.nextPreimageRowStart application + index.val) =
      some (NextPreimageDirectPlan.program.row index)

def prefixProgram (application : ApplicationProgram) :=
  MatrixProjection.program application (PerApplicationMatrixProgram.piCcsCompleteProgram application)
def piDecProgram (application : ApplicationProgram) :=
  MatrixProjection.program application (PerApplicationMatrixProgram.piDecProgram application)
def runningProgram (application : ApplicationProgram) :=
  MatrixProjection.program application (PerApplicationMatrixProgram.runningTransitionProgram application)
def applicationProgram (application : ApplicationProgram) :=
  MatrixProjection.program application (PerApplicationMatrixProgram.applicationProgram application)
def nextProgram (application : ApplicationProgram) :=
  MatrixProjection.program application (PerApplicationMatrixProgram.nextPreimageProgram application)
def publicProgram (application : ApplicationProgram) :=
  MatrixProjection.program application (PerApplicationMatrixProgram.recursivePublicOutputProgram application)

variable (application : ApplicationProgram)
  (relation : ProductionKey.LogicalRelation relationWidth publicFits)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (sourceRow : Nat → Option R1CS.Row)

include fits

private theorem pilot_exact :
    Exact (PerApplicationMatrixProgram.pilotPoseidonProgram application)
      (DirectPiDECPrefixPlan.pilotPlan (Stage1Plan.piDecGeometry application)) sourceRow :=
  PerApplicationMatrixProgramSemantics.pilotPoseidonExact application fits sourceRow

private theorem c_hash_exact :
    Exact (PerApplicationMatrixProgram.piCcsPoseidonProgram application)
      (DirectPiDECPrefixPlan.piCcsPoseidonPlan (Stage1Plan.piDecGeometry application)) sourceRow :=
  PerApplicationMatrixProgramSemantics.piCcsPoseidonExact application fits sourceRow

private theorem c_ordinary_exact (custody : SourceCustody application relation fits.package sourceRow) :
    Exact (PerApplicationMatrixProgram.piCcsOrdinaryProgram application)
      (DirectPiDECPrefixPlan.piCcsOrdinaryPlan relation (Stage1Plan.piDecGeometry application)) sourceRow := by
  refine ⟨?_, ?_⟩
  · simp [PerApplicationMatrixProgram.piCcsOrdinaryProgram, DirectPiDECPrefixPlan.piCcsOrdinaryPlan]
  · intro row
    exact PiCCSOrdinaryMatrixProgram.matrixProgram_row? relation
      (Stage1Plan.piCcsGeometry application) sourceRow custody.piCcsOrdinary row

private theorem pilot_ordinary_exact (custody : SourceCustody application relation fits.package sourceRow) :
    Exact (PerApplicationMatrixProgram.pilotOrdinaryProgram application)
      (DirectPiDECPrefixPlan.pilotOrdinaryPlan (Stage1Plan.piDecGeometry application)) sourceRow := by
  refine ⟨?_, ?_⟩
  · simp [PerApplicationMatrixProgram.pilotOrdinaryProgram, DirectPiDECPrefixPlan.pilotOrdinaryPlan]
  · intro row
    exact PilotOrdinaryMatrixProgram.matrixProgram_row?
      (DirectPiDECPrefixPlan.pilotOrdinaryGeometry (Stage1Plan.piDecGeometry application))
      sourceRow custody.pilotOrdinary row

private theorem pilot_binding_exact :
    Exact (PerApplicationMatrixProgram.pilotDigestBindingProgram application)
      (DirectPiDECPrefixPlan.pilotBindingPlan (Stage1Plan.piDecGeometry application)) sourceRow :=
  PerApplicationMatrixProgramSemantics.pilotDigestBindingExact application fits sourceRow

private theorem endpoint_exact :
    Exact (PerApplicationMatrixProgram.piCcsEndpointProgram application)
      (DirectPiDECPrefixPlan.piCcsEndpointPlan (Stage1Plan.piDecGeometry application)) sourceRow :=
  PerApplicationMatrixProgramSemantics.piCcsEndpointExact application fits sourceRow

/-- Pilot and PiCCS matrix rows, after the checked coordinate projection. -/
theorem prefix_exact (custody : SourceCustody application relation fits.package sourceRow) :
    Exact (prefixProgram application) (Stage1Plan.prefixPlan application relation) sourceRow := by
  apply MatrixProjection.exact
  change Exact (PerApplicationMatrixProgram.piCcsCompleteProgram application)
    (DirectPiDECPrefixPlan.piCcsCompletePlan relation (Stage1Plan.piDecGeometry application)) sourceRow
  unfold PerApplicationMatrixProgram.piCcsCompleteProgram PerApplicationMatrixProgram.pilotBindingPrefixProgram
    PerApplicationMatrixProgram.pilotOrdinaryPrefixProgram PerApplicationMatrixProgram.piCcsCoreProgram
    PerApplicationMatrixProgram.piCcsPoseidonPrefixProgram DirectPiDECPrefixPlan.piCcsCompletePlan
    DirectPiDECPrefixPlan.pilotBindingPrefixPlan DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan
    DirectPiDECPrefixPlan.piCcsCorePlan DirectPiDECPrefixPlan.piCcsPoseidonPrefix
  exact (((((pilot_exact application fits sourceRow).append (c_hash_exact application fits sourceRow) _).append
    (c_ordinary_exact application relation fits sourceRow custody) _).append
    (pilot_ordinary_exact application relation fits sourceRow custody) _).append
    (pilot_binding_exact application fits sourceRow) _).append (endpoint_exact application fits sourceRow) _

/-- PiDEC reads the mapped candidate outputs through its unchanged source rows. -/
theorem piDec_exact (custody : SourceCustody application relation fits.package sourceRow) :
    Exact (piDecProgram application) (Stage1Plan.piDec application relation) sourceRow := by
  apply MatrixProjection.exact
  refine ⟨?_, ?_⟩
  · simp [PerApplicationMatrixProgram.piDecProgram, DirectPiDECPrefixPlan.piDecPlan]
  · intro row
    exact PiDECMatrixProgram.matrixProgram_row? relation (Stage1Plan.piDecGeometry application)
      sourceRow custody.piDecPublic custody.piDecCommitment custody.piDecEvalK custody.piDecEvalA row

theorem running_exact :
    Exact (runningProgram application) (Stage1Plan.running application) sourceRow := by
  apply MatrixProjection.exact
  exact PerApplicationMatrixProgramSemantics.runningTransitionExact application fits sourceRow

theorem application_exact (custody : SourceCustody application relation fits.package sourceRow) :
    Exact (applicationProgram application) (Stage1Plan.application application fits.package) sourceRow := by
  apply MatrixProjection.exact
  refine ⟨?_, ?_⟩
  · simp [PerApplicationMatrixProgram.applicationProgram]
  · intro row
    exact ApplicationMatrixProgram.matrixProgram_row? fits.package (Stage1Plan.referenceGeometry application)
      sourceRow custody.applicationRows row

theorem next_exact (custody : SourceCustody application relation fits.package sourceRow) :
    Exact (nextProgram application) (Stage1Plan.nextPreimage application) sourceRow := by
  apply MatrixProjection.exact
  refine ⟨?_, ?_⟩
  · simp [PerApplicationMatrixProgram.nextPreimageProgram, DirectApplicationPrefixPlan.nextPreimagePlan]
  · intro row
    exact NextPreimageMatrixProgram.matrixProgram_row? (Stage1Plan.piCcsGeometry application)
      sourceRow custody.nextPreimage row

theorem public_exact :
    Exact (publicProgram application) (Stage1Plan.publicOutput application) sourceRow := by
  apply MatrixProjection.exact
  exact PerApplicationMatrixProgramSemantics.recursivePublicOutputExact application fits sourceRow

/-- The existing canonical package supplies every reused source row for the
candidate relation. This proof does not request any old-sampler source row. -/
theorem source_custody :
    SourceCustody application relation fits.package
      (PerApplicationPackageSourceCustody.sourceRow application) := by
  refine {
    piCcsOrdinary := ?_
    pilotOrdinary := ?_
    piDecPublic := ?_
    piDecCommitment := ?_
    piDecEvalK := ?_
    piDecEvalA := ?_
    applicationRows := ?_
    nextPreimage := ?_ }
  · intro index sourceIndex selected
    exact PerApplicationPackageSourceRows.piCcsPackageSourceRow?_eq_some
      application relation index sourceIndex selected
  · intro index
    exact PerApplicationPackageSourceRows.pilotPackageSourceRowAt?_eq_some application index
  · intro index
    exact PerApplicationPackageSourceRows.piDecPublicPackageSourceRow?_eq_some application relation index
  · intro index
    exact PerApplicationPackageSourceRows.piDecCommitmentPackageSourceRow?_eq_some application relation index
  · intro index
    exact PerApplicationPackageSourceRows.piDecEvalKPackageSourceRow?_eq_some application relation index
  · intro index
    exact PerApplicationPackageSourceRows.piDecEvalAPackageSourceRow?_eq_some application relation index
  · intro _
    exact PerApplicationPackageSourceCustody.applicationSourceRow?_eq_some application fits
  · exact PerApplicationPackageSourceCustody.nextPreimageSourceRow?_eq_some application

end NightstreamFPrime.Export.Stage1.Wide.ReusedMatrixPrograms
