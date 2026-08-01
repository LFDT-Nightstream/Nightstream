import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Deployment
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStepCost
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.Profile

/-!
Contract: exact recursive dimensions and physical cost for the native
four-matrix CCS version of the 42-times-6 WASM benchmark.

Assurance tier: model-level.

Owns: the matrix-count-four dimensions, exact intrinsic NIFS cost, exact
complete native Step cost, and the row/column fixed-point equations.

Does not own: the four sparse matrices, an Ajtai setup value, JSON, Rust
decoding, or a production WASM application.

Emits constraints: none. It computes existing receipt-derived resources.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointCost

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ApplicationStepCostSplit
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-- Least recursive dimensions for the native benchmark program. The
thirteen logical completion coordinates remain part of the 270-coordinate
carrier; they are not activation residuals. -/
def dimensions : Dimensions where
  rowVariables := 23
  legacyLogicalWidth := 5_354_522
  matrixCount := NativeCcsSelector.matrixCount
  legacyPublicFits := by decide

abbrev commitmentRows : Nat :=
  productionProfile.commitmentWidth

abbrev shape : SemanticShape :=
  ConcreteNifsPlain270Profile.Shape dimensions

def publicFits :
    ringDegree * publicRingColumns ≤ shape.carrierWidth :=
  ConcreteNifsPlain270Profile.publicFits dimensions

private theorem cost_ext
    {left right : Cost}
    (rows : left.recurringRows = right.recurringRows)
    (committed : left.committedColumns = right.committedColumns)
    (publicEq : left.publicColumns = right.publicColumns)
    (auxiliary : left.auxiliaryColumns = right.auxiliaryColumns) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem cost_add_right_cancel
    {left right extra : Cost}
    (equal : left + extra = right + extra) :
    left = right := by
  apply cost_ext
  · exact Nat.add_right_cancel (congrArg Cost.recurringRows equal)
  · exact Nat.add_right_cancel (congrArg Cost.committedColumns equal)
  · exact Nat.add_right_cancel (congrArg Cost.publicColumns equal)
  · exact Nat.add_right_cancel (congrArg Cost.auxiliaryColumns equal)

private def rowDegree : Nat :=
  SumCheck.Fe.Drow
    (KSplitNcStaticInput.layoutInput
      (shape := shape) NativeCcsSelector.constraintPolynomial)

private noncomputable def statementFields : Nat :=
  ConcreteNifsStaticFootprint.statementFieldCount
    shape publicRingColumns commitmentRows publicFits

private def outputFields : Nat :=
  ConcreteNifsStaticFootprint.outputFieldCount shape

private noncomputable def transcriptControl : SymbolicDuplexCount.Control :=
  KSplitNcTranscriptCount.afterOutput dimensions.rowVariables rowDegree
    PiCcsDomains.production.laneVariables
    PiCcsDomains.production.blockVariables
    0 statementFields outputFields

private noncomputable def transcriptCost : Cost :=
  KSplitNcTranscriptCount.cost dimensions.rowVariables rowDegree
    PiCcsDomains.production.laneVariables
    PiCcsDomains.production.blockVariables
    0 statementFields outputFields

private noncomputable def blockLaneCost : Cost :=
  KSplitNcBlockLaneRows.cost
    (KSplitNcTranscript.numericColumns
      (ConcreteNifsStaticFootprint.compactTranscriptInput
        shape NativeCcsSelector.constraintPolynomial publicRingColumns
        commitmentRows publicFits))

private noncomputable def endpointCost : Cost :=
  KSplitNcOperationalRows.endpointCost
    (ConcreteNifsStaticFootprint.compactOperationalInput
      shape NativeCcsSelector.constraintPolynomial publicRingColumns
      commitmentRows publicFits)

private def piRlcActionCost : Cost :=
  ConcreteNifsPiRlcActionRows.cost shape publicRingColumns commitmentRows

private theorem rowDegree_exact : rowDegree = 4 := by
  rfl

private theorem statementFields_exact : statementFields = 33_146 := by
  rfl

private theorem outputFields_exact : outputFields = 8_103 := by
  rfl

private theorem transcriptPermutationCount_exact :
    transcriptControl.entries = 10_581 := by
  rfl

private theorem transcriptCost_exact :
    transcriptCost = ⟨3_724_512, 0, 0, 3_724_512⟩ := by
  apply cost_ext <;> rfl

private theorem blockLaneCost_exact :
    blockLaneCost = ⟨726, 0, 0, 612⟩ := by
  apply cost_ext <;> rfl

private theorem endpointCost_exact :
    endpointCost = ⟨25_102, 0, 0, 25_094⟩ := by
  apply cost_ext <;> rfl

private theorem compactOperationalCost_exact :
    ConcreteNifsStaticFootprint.compactOperationalCost
        shape NativeCcsSelector.constraintPolynomial publicRingColumns
          commitmentRows publicFits =
      ⟨3_750_340, 0, 0, 3_750_218⟩ := by
  unfold ConcreteNifsStaticFootprint.compactOperationalCost
  change transcriptCost + blockLaneCost + endpointCost =
    ⟨3_750_340, 0, 0, 3_750_218⟩
  rw [transcriptCost_exact, blockLaneCost_exact, endpointCost_exact]
  rfl

private theorem piRlcSamplerCost_exact :
    PiRlcCanonicalSamplerProgram.cost =
      ⟨105_930, 0, 0, 99_885⟩ := by
  apply cost_ext <;> rfl

private theorem piRlcChallengeCost_exact :
    ConcreteNifsOperationalSampler.challengeCost =
      ⟨810, 0, 0, 0⟩ := by
  apply cost_ext <;> rfl

private theorem samplerCost_exact :
    ConcreteNifsStaticFootprint.samplerCost
        shape NativeCcsSelector.constraintPolynomial publicRingColumns
          commitmentRows publicFits =
      ⟨3_857_080, 0, 0, 3_850_103⟩ := by
  unfold ConcreteNifsStaticFootprint.samplerCost
  rw [compactOperationalCost_exact, piRlcSamplerCost_exact,
    piRlcChallengeCost_exact]
  rfl

private theorem piRlcActionCost_exact :
    piRlcActionCost = ⟨1_357_614, 0, 0, 1_355_940⟩ := by
  apply cost_ext <;> rfl

/-- Exact selected verifier cost before any activation wrapper. -/
theorem intrinsicCost_exact :
    ConcreteNifsStaticFootprint.intrinsicCost shape
        NativeCcsSelector.constraintPolynomial publicRingColumns
        commitmentRows publicFits =
      ⟨5_242_820, 0, 0, 5_206_053⟩ := by
  unfold ConcreteNifsStaticFootprint.intrinsicCost
  rw [samplerCost_exact]
  change
    ConcreteNifsRawProgram.claimedValueCost +
          ConcreteNifsProofCanonicalityRows.cost +
        ConcreteNifsRunningAuthorityRows.cost
          shape publicRingColumns commitmentRows +
      ⟨3_857_080, 0, 0, 3_850_103⟩ +
      ConcreteNifsPiRlcPointRows.cost shape.rowVariables +
      piRlcActionCost +
      ConcreteNifsPiDecRows.cost shape publicRingColumns commitmentRows +
      ConcreteNifsOutputRows.cost shape publicRingColumns commitmentRows =
        ⟨5_242_820, 0, 0, 5_206_053⟩
  rw [piRlcActionCost_exact]
  apply cost_ext <;> rfl

/-- The legacy compatibility receipt at the native recursive shape still
contains the residual activation wrapper. This value is used only to prove
the exact removal equation. -/
theorem activatedCost_exact :
    ConcreteNifsStaticFootprint.cost shape
        NativeCcsSelector.constraintPolynomial publicRingColumns
        commitmentRows publicFits =
      ⟨10_485_640, 0, 0, 10_448_873⟩ := by
  unfold ConcreteNifsStaticFootprint.cost ActivatedRawProgram.cost
  rw [intrinsicCost_exact]
  rfl

noncomputable def parameters
    (setup : RelationSetup dimensions commitmentRows) :=
  ConcreteNifsPlain270Profile.selected dimensions
    (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup)
    (defaultRunning dimensions commitmentRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions commitmentRows)
    (terminalChecks dimensions commitmentRows)
    (widths setup)
    (selectedFootprints setup)

noncomputable def sourceCertificate
    (setup : RelationSetup dimensions commitmentRows) :=
  ConcreteNifsCanonicalCertification.complete
    setup
    (defaultRunning dimensions commitmentRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions commitmentRows)
    (terminalChecks dimensions commitmentRows)
    (widths setup)
    (selectedFootprints setup)
    (deployment setup)

noncomputable def nifsCertificate
    (setup : RelationSetup dimensions commitmentRows) :=
  ConcreteNifsCanonicalCertification.nifs
    setup
    (defaultRunning dimensions commitmentRows)
    (machine benchmarkHashPlan)
    (terminalRelations dimensions commitmentRows)
    (terminalChecks dimensions commitmentRows)
    (widths setup)
    (selectedFootprints setup)
    (deployment setup)

noncomputable def nifsFrame
    (setup : RelationSetup dimensions commitmentRows) :=
  (ConcreteNifsNativeCcsStep.invokePlan
    (deployment setup).application.phase4
    (nifsCertificate setup)
    (deployment setup).step
    (deployment setup).defaultRunningAdmissible).frame

noncomputable def nativeProgram
    (setup : RelationSetup dimensions commitmentRows) :
    NativeCcsProgram.Program :=
  ConcreteNifsNativeCcsStep.program
    (deployment setup).application.phase4
    (nifsCertificate setup)
    (deployment setup).step
    (deployment setup).defaultRunningAdmissible

private theorem runningWidth_exact
    (setup : RelationSetup dimensions commitmentRows) :
    (parameters setup).widths.running = 25_800 := by
  rfl

private theorem freshWidth_exact
    (setup : RelationSetup dimensions commitmentRows) :
    (parameters setup).widths.fresh = 1_242 := by
  rfl

private theorem proofWidth_exact
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        NativeCcsSelector.constraintPolynomial) :
    (parameters setup).widths.nifsProof = 38_964 := by
  unfold parameters widths proofCodec
  rw [polynomialExact]
  rfl

private theorem stepInputCost_exact
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        NativeCcsSelector.constraintPolynomial) :
    (stepInputSchema (parameters setup)).cost =
      ⟨0, 66_021, 0, 0⟩ := by
  rw [stepInputSchema_cost_exact]
  rw [runningWidth_exact, freshWidth_exact,
    proofWidth_exact setup polynomialExact]
  rfl

private theorem nifsCallCost_exact
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        NativeCcsSelector.constraintPolynomial) :
    (signature (parameters setup)).callCost Call.nifsVerify =
      ⟨10_485_640, 25_800, 0, 10_448_873⟩ := by
  rw [nifsVerify_callCost_exact]
  have footprintExact :
      (parameters setup).footprints.nifsVerify =
        nifsFootprint setup := by
    rfl
  rw [footprintExact]
  unfold nifsFootprint ConcreteNifsStaticFootprint.footprint
    CallFootprint.cost
  simp only [List.map_cons, List.map_nil, Cost.sum,
    auxiliaryLayout_cost_exact, committedLayout_cost_exact]
  rw [polynomialExact, activatedCost_exact, runningWidth_exact]
  rfl

private theorem compactCost_exact
    (selected : Parameters)
    (inputCost :
      (stepInputSchema selected).cost = ⟨0, 66_021, 0, 0⟩)
    (runningWidth : selected.widths.running = 25_800)
    (step :
      (signature selected).callCost Call.step = ⟨11, 7, 0, 4⟩)
    (iterationZero :
      (signature selected).callCost Call.iterationZero =
        ⟨3, 0, 0, 3⟩)
    (stateEqual :
      (signature selected).callCost Call.stateEqual =
        ⟨21, 0, 0, 21⟩)
    (hashPrior :
      (signature selected).callCost Call.hashPrior =
        ⟨2503, 0, 0, 2499⟩)
    (freshPublic :
      (signature selected).callCost Call.freshPublic =
        ⟨5, 0, 0, 5⟩)
    (encodeInstance :
      (signature selected).callCost Call.encodeInstance =
        ⟨5, 0, 0, 5⟩)
    (encodedEqual :
      (signature selected).callCost Call.encodedEqual =
        ⟨15, 0, 0, 15⟩)
    (nifsVerify :
      (signature selected).callCost Call.nifsVerify =
        ⟨10_485_640, 25_800, 0, 10_448_873⟩)
    (hashNext :
      (signature selected).callCost Call.hashNext =
        ⟨2503, 0, 5, 2494⟩) :
    compactStepCost selected =
      ⟨10_542_310, 143_428, 6, 10_453_921⟩ := by
  have runningJoin :
      (Ports.committedRunning selected).layout.cost +
          ⟨selected.widths.running, 0, 0, 0⟩ =
        ⟨25_800, 25_800, 0, 0⟩ := by
    simp only [Ports.committedRunning, dataPort,
      committedLayout_cost_exact]
    rw [runningWidth]
    rfl
  unfold compactStepCost
  rw [inputCost]
  unfold stepBodyCosts
  rw [step, iterationZero, stateEqual, runningJoin, hashPrior,
    freshPublic, encodeInstance, encodedEqual, nifsVerify, hashNext]
  rfl

private theorem compactSourceCost_exact
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        NativeCcsSelector.constraintPolynomial) :
    ApplicationStepCostSplit.compactStepCost (parameters setup) =
      ⟨10_542_310, 143_428, 6, 10_453_921⟩ := by
  apply compactCost_exact
  · exact stepInputCost_exact setup polynomialExact
  · exact runningWidth_exact setup
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact nifsCallCost_exact setup polynomialExact
  · rfl

/-- Exact cost of the temporary compatibility Step before removing the
activation residual wrapper. -/
theorem sourceCost_exact
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        NativeCcsSelector.constraintPolynomial) :
    (sourceCertificate setup).stepCost =
      ⟨10_542_310, 143_428, 6, 10_453_921⟩ := by
  rw [CompleteApplicationCertification.stepCost_eq_compact]
  exact compactSourceCost_exact setup polynomialExact

private theorem overhead_exact
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        NativeCcsSelector.constraintPolynomial) :
    ConcreteNifsNativeCcsStepCost.overhead
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible =
      ⟨5_242_820, 0, 0, 5_242_820⟩ := by
  unfold ConcreteNifsNativeCcsStepCost.overhead
  change
    ActivatedRawProgram.overheadCost
        (ConcreteNifsRawProgram.cost
          (deployment setup).application.phase4.profile
          (nifsCertificate setup).operational
          (nifsFrame setup)
        ).recurringRows =
      ⟨5_242_820, 0, 0, 5_242_820⟩
  have rawCostExact :
      ConcreteNifsRawProgram.cost
          (deployment setup).application.phase4.profile
          (nifsCertificate setup).operational
          (nifsFrame setup) =
        ⟨5_242_820, 0, 0, 5_206_053⟩ := by
    have static :
        ConcreteNifsRawProgram.cost
            (deployment setup).application.phase4.profile
            (ConcreteNifsCanonicalOperationalProfile.operational
              setup
              (defaultRunning dimensions commitmentRows)
              (machine benchmarkHashPlan)
              (terminalRelations dimensions commitmentRows)
              (terminalChecks dimensions commitmentRows)
              (widths setup)
              (selectedFootprints setup)
              (deployment setup).application)
            (nifsFrame setup) =
          ⟨5_242_820, 0, 0, 5_206_053⟩ := by
      calc
        _ =
            ConcreteNifsStaticFootprint.intrinsicCost shape
              setup.system.constraintPolynomial publicRingColumns
              commitmentRows publicFits :=
          ConcreteNifsStaticFootprint.intrinsicCost_eq
            setup
            (defaultRunning dimensions commitmentRows)
            (machine benchmarkHashPlan)
            (terminalRelations dimensions commitmentRows)
            (terminalChecks dimensions commitmentRows)
            (widths setup)
            (selectedFootprints setup)
            (deployment setup).application
            (nifsFrame setup)
        _ = ⟨5_242_820, 0, 0, 5_206_053⟩ := by
          rw [polynomialExact]
          exact intrinsicCost_exact
    change
      ConcreteNifsRawProgram.cost
          (deployment setup).application.phase4.profile
          (ConcreteNifsCanonicalOperationalProfile.operational
            setup
            (defaultRunning dimensions commitmentRows)
            (machine benchmarkHashPlan)
            (terminalRelations dimensions commitmentRows)
            (terminalChecks dimensions commitmentRows)
            (widths setup)
            (selectedFootprints setup)
            (deployment setup).application)
          (nifsFrame setup) =
        ⟨5_242_820, 0, 0, 5_206_053⟩
    exact static
  rw [rawCostExact]
  rfl

/-- Exact native four-matrix Step cost at the recursive fixed point. -/
theorem nativeCost_exact
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        NativeCcsSelector.constraintPolynomial) :
    (nativeProgram setup).cost =
      ⟨5_299_490, 143_428, 6, 5_211_101⟩ := by
  have split :=
    ConcreteNifsNativeCcsStepCost.sourceCost_eq_nativeCost_add_overhead
      (deployment setup).application.phase4
      (nifsCertificate setup)
      (deployment setup).step
      (deployment setup).defaultRunningAdmissible
  change
    (sourceCertificate setup).stepCost =
      (nativeProgram setup).cost +
        ConcreteNifsNativeCcsStepCost.overhead
          (deployment setup).application.phase4
          (nifsCertificate setup)
          (deployment setup).step
          (deployment setup).defaultRunningAdmissible at split
  rw [sourceCost_exact setup polynomialExact,
    overhead_exact setup polynomialExact] at split
  apply cost_add_right_cancel
  exact split.symm.trans (by rfl)

theorem nativeRows_exact
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        NativeCcsSelector.constraintPolynomial) :
    (nativeProgram setup).rows.length = 5_299_490 := by
  rw [NativeCcsProgram.Program.rows_length,
    nativeCost_exact setup polynomialExact]

theorem nativeColumns_exact
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        NativeCcsSelector.constraintPolynomial) :
    (nativeProgram setup).columnIds.length = 5_354_535 := by
  rw [NativeCcsProgram.Program.columnIds_length_eq_cost_columns,
    nativeCost_exact setup polynomialExact]
  rfl

/-- The row cube is the least power-of-two domain that covers the native
program. -/
theorem rowVariables_fixed
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        NativeCcsSelector.constraintPolynomial) :
    dimensions.rowVariables =
      Profile.rowVariables (nativeProgram setup).rows.length := by
  rw [nativeRows_exact setup polynomialExact]
  rfl

/-- The recursive logical width is exactly the emitted native allocation
width. -/
theorem logicalWidth_fixed
    (setup : RelationSetup dimensions commitmentRows)
    (polynomialExact :
      setup.system.constraintPolynomial =
        NativeCcsSelector.constraintPolynomial) :
    dimensions.alignedLogicalWidth =
      (nativeProgram setup).columnIds.length := by
  rw [nativeColumns_exact setup polynomialExact]
  rfl

theorem matrixCount_fixed :
    dimensions.matrixCount = NativeCcsSelector.matrixCount :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointCost
