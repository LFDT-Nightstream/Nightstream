import NightstreamFPrime.Export.Stage1.ApplicationMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PerApplicationMatrixProgram
import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PiCCSPoseidonMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PiDECMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PilotOrdinaryMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PilotPoseidonMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.RunningTransitionMatrixProgramSemantics

/-!
Proves row-by-row equality between the complete compact per-application matrix
program and the exact Lean structural plan. Ordinary source rows remain an
explicit identity-checked input to this theorem.

This module does not claim package identity, Rust conformance, or production
closure.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationMatrixProgramSemantics

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1

abbrev ApplicationProgram := Lifecycle.Stage1.Application.Program

def relation (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application) :=
  PerApplicationProductionPlan.relation application fits

/-- Exact compact interpretation of one semantic plan. -/
structure Exact {logicalWidth : Nat}
    (matrixProgram : MatrixProgram.Program)
    (plan : ProductionRelation.Plan logicalWidth)
    (sourceRow : Nat → Option R1CS.Row) : Prop where
  rowCount : matrixProgram.rowCount = plan.rowCount
  row? : ∀ row : Fin plan.rowCount,
    matrixProgram.row? logicalWidth sourceRow row.val = some (plan.forms row)

theorem Exact.append {logicalWidth : Nat}
    {leftProgram rightProgram : MatrixProgram.Program}
    {leftPlan rightPlan : ProductionRelation.Plan logicalWidth}
    {sourceRow : Nat → Option R1CS.Row}
    (left : Exact leftProgram leftPlan sourceRow)
    (right : Exact rightProgram rightPlan sourceRow)
    (fits : leftPlan.rowCount + rightPlan.rowCount ≤
      2 ^ Lifecycle.cubeVariables) :
    Exact (leftProgram.append rightProgram)
      (ProductionRelation.Plan.append leftPlan rightPlan fits) sourceRow := by
  refine ⟨?_, ?_⟩
  · rw [MatrixProgram.Program.append_rowCount,
      ProductionRelation.Plan.append_rowCount, left.rowCount, right.rowCount]
  · intro global
    exact MatrixProgram.Program.append_plan_row? leftProgram rightProgram
      leftPlan rightPlan fits sourceRow left.rowCount left.row? right.row?
      global

/-- The final source-row accessor must return the exact Lean-selected source
row for every ordinary family. -/
structure SourceCustody (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row) : Prop where
  piCcsOrdinary : ∀ index : Fin 811669, ∀ sourceIndex,
    PiCCSOrdinaryMatrixProgram.rowSchedule.index? index.val =
        some sourceIndex →
      sourceRow sourceIndex = some
        (PerApplicationSourceProjection.basePackageRow application
          (PiCCSOrdinaryDirectSource.programRow (relation application fits)
            index))
  pilotOrdinary : ∀ index : Fin 1330,
    sourceRow (PilotOrdinaryMatrixProgram.rowIndexAt index) =
      some (PerApplicationSourceProjection.pilotPackageRow application
        (PilotOrdinaryDirectSource.programRow index))
  samplerOrdinary : ∀ index : Fin 220881, ∀ sourceIndex,
    PiRLCSamplerOrdinaryMatrixSchedule.rowSchedule.index? index.val =
        some sourceIndex →
      sourceRow sourceIndex = some
        (PerApplicationSourceProjection.basePackageRow application
          (PiRLCSamplerOrdinaryDirectSource.programRow
            (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
            (publicFits := PerApplicationFixedPoint.publicFits application)
            index))
  piDecPublic : ∀ index : Fin 22680,
    sourceRow (PiDECStarts.publicInputRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.publicProgramRow
          (relation application fits) index))
  piDecCommitment : ∀ index : Fin 1188,
    sourceRow (PiDECStarts.commitmentRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.commitmentProgramRow
          (relation application fits) index))
  piDecEvalK : ∀ index : Fin 108,
    sourceRow (PiDECStarts.evalKRowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.evalKProgramRow
          (relation application fits) index))
  piDecEvalA : ∀ index : Fin 1512,
    sourceRow (PiDECStarts.evalARowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        (PiDECOrdinaryDirectSource.evalAProgramRow
          (relation application fits) index))
  runningTransition : ∀ index :
      Fin (RunningTransitionDirectSource.program
        (relation application fits)).rowCount,
    sourceRow (RunningTransitionArithmetic.rowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        ((RunningTransitionDirectSource.program
          (relation application fits)).row index))
  applicationRows : ∀ index :
      Fin (ApplicationDirectSource.program application fits.package).rowCount,
    sourceRow (PerApplicationPackage.basePackage.layout.rowCount + index.val) =
      some ((ApplicationDirectSource.program application fits.package).row
        index)
  nextPreimage : ∀ index : Fin NextPreimageDirectPlan.program.rowCount,
    sourceRow
        (PerApplicationPackage.nextPreimageRowStart application + index.val) =
      some (NextPreimageDirectPlan.program.row index)

theorem piCcsPoseidonExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row) :
    Exact (PerApplicationMatrixProgram.blockProgram application
        .piCcsPoseidon)
      (PerApplicationProductionPlan.BlockKind.piCcsPoseidon.plan application
        fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .piCcsPoseidon, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.piCcsPoseidonProgram,
    PerApplicationMatrixProgram.poseidonGeometry,
    PerApplicationMatrixProgram.piDecGeometry,
    PerApplicationProductionPlan.BlockKind.plan] using
      PiCCSPoseidonMatrixProgram.matrixProgram_row?
        (PerApplicationMatrixProgram.poseidonGeometry application)
        sourceRow row

theorem samplerPoseidonExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row) :
    Exact (PerApplicationMatrixProgram.blockProgram application
        .samplerPoseidon)
      (PerApplicationProductionPlan.BlockKind.samplerPoseidon.plan application
        fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .samplerPoseidon, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.samplerPoseidonProgram,
    PerApplicationMatrixProgram.poseidonGeometry,
    PerApplicationMatrixProgram.piDecGeometry,
    PerApplicationProductionPlan.BlockKind.plan,
    DirectPiDECPrefixPlan.samplerPlan, DirectPrefixPlan.samplerPlan,
    PiRLCSamplerPoseidonPlan.plan, PoseidonSboxFamilyPlan.plan,
    ProductionRelation.Plan.indexed] using
      PiRLCSamplerPoseidonMatrixProgram.matrixProgram_row?
        (PerApplicationMatrixProgram.poseidonGeometry application)
        sourceRow row

theorem piRlcExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row) :
    Exact (PerApplicationMatrixProgram.blockProgram application .piRlc)
      (PerApplicationProductionPlan.BlockKind.piRlc.plan application fits)
      sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .piRlc, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.piRlcProgram,
    PerApplicationMatrixProgram.piRlcGeometry,
    PerApplicationMatrixProgram.poseidonGeometry,
    PerApplicationMatrixProgram.piDecGeometry,
    PerApplicationProductionPlan.BlockKind.plan,
    DirectPiRLCSamplerCompletePrefixPlan.piRlcPlan,
    DirectPiDECPrefixPlan.piRlcPlan, DirectPrefixPlan.piRlcPlan] using
      PiRLCMatrixProgram.matrixProgram_row?
        (PerApplicationMatrixProgram.piRlcGeometry application) sourceRow row

theorem pilotDigestBindingExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row) :
    Exact (PerApplicationMatrixProgram.blockProgram application
        .pilotDigestBinding)
      (PerApplicationProductionPlan.BlockKind.pilotDigestBinding.plan
        application fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .pilotDigestBinding, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.pilotDigestBindingProgram,
    PerApplicationMatrixProgram.piDecGeometry,
    PerApplicationProductionPlan.BlockKind.plan] using
      PinMatrixPrograms.pilotDigestBindingProgram_plan_row?
        (PerApplicationMatrixProgram.piDecGeometry application) sourceRow row

theorem piCcsEndpointExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row) :
    Exact (PerApplicationMatrixProgram.blockProgram application
        .piCcsEndpoint)
      (PerApplicationProductionPlan.BlockKind.piCcsEndpoint.plan application
        fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .piCcsEndpoint, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.piCcsEndpointProgram,
    PerApplicationMatrixProgram.piDecGeometry,
    PerApplicationProductionPlan.BlockKind.plan] using
      PinMatrixPrograms.piCcsEndpointProgram_plan_row?
        (PerApplicationMatrixProgram.piDecGeometry application) sourceRow row

theorem pilotPoseidonExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row) :
    Exact (PerApplicationMatrixProgram.blockProgram application
        .pilotPoseidon)
      (PerApplicationProductionPlan.BlockKind.pilotPoseidon.plan application
        fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .pilotPoseidon, ?_⟩
  intro row
  let geometry := PerApplicationMatrixProgram.pilotGeometry application
  let priorProgram : MatrixProgram.Program :=
    MatrixProgram.Program.mk
      [.poseidon (PilotPoseidonMatrixProgram.priorBlock geometry)]
  let outputProgram : MatrixProgram.Program :=
    MatrixProgram.Program.mk
      [.poseidon (PilotPoseidonMatrixProgram.outputBlock geometry)]
  have priorCount : priorProgram.rowCount =
      (PilotPoseidonPlan.priorPlan geometry).rowCount := by
    rw [show priorProgram = MatrixProgram.Program.mk
      [.poseidon (PilotPoseidonMatrixProgram.priorBlock geometry)] by rfl]
    rw [MatrixProgram.Program.singleton_rowCount]
    change (PilotPoseidonMatrixProgram.priorBlock geometry).rowCount = _
    rw [PilotPoseidonMatrixProgram.priorBlock_rowCount]
    simp [PilotPoseidonPlan.priorPlan, PoseidonSboxFamilyPlan.plan,
      PilotPoseidonPlan.invocationCount_eq]
  have exactRow := MatrixProgram.Program.append_plan_row?
    priorProgram outputProgram
    (PilotPoseidonPlan.priorPlan geometry)
    (PilotPoseidonPlan.outputPlan geometry)
    (PilotPoseidonPlan.combinedRowCount_le geometry) sourceRow priorCount
    (PilotPoseidonMatrixProgram.priorProgram_row? geometry sourceRow)
    (PilotPoseidonMatrixProgram.outputProgram_row? geometry sourceRow)
  simpa [geometry, priorProgram, outputProgram,
    PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.pilotPoseidonProgram,
    PerApplicationMatrixProgram.pilotGeometry,
    PerApplicationMatrixProgram.poseidonGeometry,
    PerApplicationMatrixProgram.piDecGeometry,
    PerApplicationProductionPlan.BlockKind.plan,
    DirectPiDECPrefixPlan.pilotPlan, DirectPrefixPlan.pilotPlan,
    PilotPoseidonMatrixProgram.matrixProgram, PilotPoseidonPlan.plan] using
      exactRow row

theorem piCcsOrdinaryExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : SourceCustody application fits sourceRow) :
    Exact (PerApplicationMatrixProgram.blockProgram application
        .piCcsOrdinary)
      (PerApplicationProductionPlan.BlockKind.piCcsOrdinary.plan application
        fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .piCcsOrdinary, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.piCcsOrdinaryProgram,
    PerApplicationMatrixProgram.piCcsOrdinaryGeometry,
    PerApplicationMatrixProgram.piDecGeometry,
    PerApplicationProductionPlan.BlockKind.plan,
    DirectPiDECPrefixPlan.piCcsOrdinaryPlan] using
      PiCCSOrdinaryMatrixProgram.matrixProgram_row?
        (relation application fits)
        (PerApplicationMatrixProgram.piCcsOrdinaryGeometry application)
        sourceRow custody.piCcsOrdinary row

theorem pilotOrdinaryExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : SourceCustody application fits sourceRow) :
    Exact (PerApplicationMatrixProgram.blockProgram application
        .pilotOrdinary)
      (PerApplicationProductionPlan.BlockKind.pilotOrdinary.plan application
        fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .pilotOrdinary, ?_⟩
  intro row
  change (PilotOrdinaryMatrixProgram.matrixProgram
      (PerApplicationMatrixProgram.pilotOrdinaryGeometry application)).row?
        (PerApplicationFixedPoint.logicalWidth application) sourceRow row.val =
    some ((PilotOrdinaryDirectPlan.plan
      (PerApplicationMatrixProgram.pilotOrdinaryGeometry application)).forms row)
  exact PilotOrdinaryMatrixProgram.matrixProgram_row?
    (PerApplicationMatrixProgram.pilotOrdinaryGeometry application)
    sourceRow custody.pilotOrdinary row

theorem samplerOrdinaryExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : SourceCustody application fits sourceRow) :
    Exact (PerApplicationMatrixProgram.blockProgram application
        .samplerOrdinary)
      (PerApplicationProductionPlan.BlockKind.samplerOrdinary.plan application
        fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .samplerOrdinary, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.samplerOrdinaryProgram,
    PerApplicationMatrixProgram.samplerGeometry,
    PerApplicationProductionPlan.BlockKind.plan,
    DirectPiRLCSamplerCompletePrefixPlan.samplerOrdinaryPlan] using
      PiRLCSamplerOrdinaryMatrixProgram.matrixProgram_row?
        (relation application fits)
        (PerApplicationMatrixProgram.samplerGeometry application)
        sourceRow custody.samplerOrdinary row

theorem piDecExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : SourceCustody application fits sourceRow) :
    Exact (PerApplicationMatrixProgram.blockProgram application .piDec)
      (PerApplicationProductionPlan.BlockKind.piDec.plan application fits)
      sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .piDec, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.piDecProgram,
    PerApplicationMatrixProgram.piDecGeometry,
    PerApplicationProductionPlan.BlockKind.plan,
    DirectPiRLCSamplerCompletePrefixPlan.piDecPlan,
    DirectPiDECPrefixPlan.piDecPlan] using
      PiDECMatrixProgram.matrixProgram_row? (relation application fits)
        (PerApplicationMatrixProgram.piDecGeometry application) sourceRow
        custody.piDecPublic custody.piDecCommitment custody.piDecEvalK
        custody.piDecEvalA row

theorem runningTransitionExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : SourceCustody application fits sourceRow) :
    Exact (PerApplicationMatrixProgram.blockProgram application
        .runningTransition)
      (PerApplicationProductionPlan.BlockKind.runningTransition.plan
        application fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .runningTransition, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.runningTransitionProgram,
    PerApplicationMatrixProgram.runningGeometry,
    PerApplicationMatrixProgram.piDecGeometry,
    PerApplicationProductionPlan.BlockKind.plan,
    DirectPiRLCSamplerCompletePrefixPlan.transitionPlan,
    DirectPiDECPrefixPlan.transitionPlan] using
      RunningTransitionMatrixProgram.matrixProgram_row?
        (relation application fits)
        (PerApplicationMatrixProgram.runningGeometry application) sourceRow
        custody.runningTransition row

theorem applicationExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : SourceCustody application fits sourceRow) :
    Exact (PerApplicationMatrixProgram.blockProgram application .application)
      (PerApplicationProductionPlan.BlockKind.application.plan application
        fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .application, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.applicationProgram,
    PerApplicationMatrixProgram.applicationGeometry,
    PerApplicationProductionPlan.BlockKind.plan,
    DirectApplicationPrefixPlan.applicationPlan] using
      ApplicationMatrixProgram.matrixProgram_row? fits.package
        (PerApplicationMatrixProgram.applicationGeometry application)
        sourceRow custody.applicationRows row

theorem nextPreimageExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : SourceCustody application fits sourceRow) :
    Exact (PerApplicationMatrixProgram.blockProgram application .nextPreimage)
      (PerApplicationProductionPlan.BlockKind.nextPreimage.plan application
        fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .nextPreimage, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.nextPreimageProgram,
    PerApplicationMatrixProgram.piCcsOrdinaryGeometry,
    PerApplicationMatrixProgram.piDecGeometry,
    PerApplicationProductionPlan.BlockKind.plan,
    DirectApplicationPrefixPlan.nextPreimagePlan,
    DirectApplicationPrefixPlan.piCcsOrdinaryGeometry] using
      NextPreimageMatrixProgram.matrixProgram_row?
        (PerApplicationMatrixProgram.piCcsOrdinaryGeometry application)
        sourceRow custody.nextPreimage row

theorem recursivePublicOutputExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row) :
    Exact (PerApplicationMatrixProgram.blockProgram application
        .recursivePublicOutput)
      (PerApplicationProductionPlan.BlockKind.recursivePublicOutput.plan
        application fits) sourceRow := by
  refine ⟨PerApplicationMatrixProgram.blockProgram_rowCount application fits
    .recursivePublicOutput, ?_⟩
  intro row
  simpa [PerApplicationMatrixProgram.blockProgram,
    PerApplicationMatrixProgram.recursivePublicOutputProgram,
    PerApplicationMatrixProgram.applicationGeometry,
    PerApplicationProductionPlan.BlockKind.plan,
    DirectApplicationPrefixPlan.publicOutputPlan] using
      PinMatrixPrograms.recursivePublicOutputProgram_plan_row?
        (PerApplicationMatrixProgram.applicationGeometry application)
        sourceRow row

theorem blockExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : SourceCustody application fits sourceRow)
    (kind : PerApplicationProductionPlan.BlockKind) :
    Exact (PerApplicationMatrixProgram.blockProgram application kind)
      (kind.plan application fits) sourceRow := by
  cases kind with
  | pilotPoseidon => exact pilotPoseidonExact application fits sourceRow
  | piCcsPoseidon => exact piCcsPoseidonExact application fits sourceRow
  | piCcsOrdinary =>
      exact piCcsOrdinaryExact application fits sourceRow custody
  | pilotOrdinary =>
      exact pilotOrdinaryExact application fits sourceRow custody
  | pilotDigestBinding =>
      exact pilotDigestBindingExact application fits sourceRow
  | piCcsEndpoint => exact piCcsEndpointExact application fits sourceRow
  | samplerPoseidon => exact samplerPoseidonExact application fits sourceRow
  | samplerOrdinary =>
      exact samplerOrdinaryExact application fits sourceRow custody
  | piRlc => exact piRlcExact application fits sourceRow
  | piDec => exact piDecExact application fits sourceRow custody
  | runningTransition =>
      exact runningTransitionExact application fits sourceRow custody
  | application => exact applicationExact application fits sourceRow custody
  | nextPreimage =>
      exact nextPreimageExact application fits sourceRow custody
  | recursivePublicOutput =>
      exact recursivePublicOutputExact application fits sourceRow

/-- Any successfully compiled Lean production tree has an exact compact
matrix interpretation. -/
theorem compileMatrixExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : SourceCustody application fits sourceRow)
    (program : PerApplicationProductionPlan.Program)
    (plan : ProductionRelation.Plan
      (PerApplicationFixedPoint.logicalWidth application))
    (compiled : program.compile application fits = some plan) :
    Exact (PerApplicationMatrixProgram.compileMatrix application program)
      plan sourceRow := by
  induction program generalizing plan with
  | leaf kind =>
      simp only [PerApplicationProductionPlan.Program.compile] at compiled
      cases compiled
      simpa [PerApplicationMatrixProgram.compileMatrix] using
        blockExact application fits sourceRow custody kind
  | append left right leftInduction rightInduction =>
      simp only [PerApplicationProductionPlan.Program.compile] at compiled
      cases leftCompiled : left.compile application fits with
      | none => simp [leftCompiled] at compiled
      | some leftPlan =>
          cases rightCompiled : right.compile application fits with
          | none => simp [leftCompiled, rightCompiled] at compiled
          | some rightPlan =>
              by_cases bounded : leftPlan.rowCount + rightPlan.rowCount ≤
                  2 ^ Lifecycle.cubeVariables
              · simp [leftCompiled, rightCompiled, bounded] at compiled
                subst plan
                simpa [PerApplicationMatrixProgram.compileMatrix] using
                  Exact.append
                    (leftInduction leftPlan leftCompiled)
                    (rightInduction rightPlan rightCompiled) bounded
              · simp [leftCompiled, rightCompiled, bounded] at compiled

/-- The canonical compact matrix program is exactly the self-derived
per-application structural plan, row for row. -/
theorem matrixProgramExact (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (sourceRow : Nat → Option R1CS.Row)
    (custody : SourceCustody application fits sourceRow) :
    Exact (PerApplicationMatrixProgram.matrixProgram application)
      (PerApplicationFixedPoint.structuralPlan application fits) sourceRow := by
  simpa using compileMatrixExact application fits sourceRow custody
    PerApplicationProductionPlan.canonical
    (PerApplicationFixedPoint.structuralPlan application fits)
    (PerApplicationProductionPlan.compile_canonical application fits)

end NightstreamFPrime.Export.Stage1.PerApplicationMatrixProgramSemantics
