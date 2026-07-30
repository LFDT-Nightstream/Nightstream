import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4NifsRows

/-!
Contract: prove physical row stability for the action receipts that contain
the selected NIFS call.

Assurance tier: model-level.

Owns: exact action-row equality after the selected relation polynomial is
fixed.

Does not own: the other Step receipts, fixed-point compilation, production
selection, Rust equality, or a security reduction.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ActionRows

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4CarrierLocations
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperandFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ProofFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4NifsRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private theorem carriedF_eq_of_source_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (leftSource rightSource : List (Nat × Nat))
    (sourceEqual : leftSource = rightSource) :
    ConcreteNifsPiRlcActionRows.carriedF
        (application (template.withSystem left))
        (invokePlan (template.withSystem left)).frame leftSource =
      ConcreteNifsPiRlcActionRows.carriedF
        (application (template.withSystem right))
        (invokePlan (template.withSystem right)).frame rightSource := by
  unfold ConcreteNifsPiRlcActionRows.carriedF
  subst rightSource
  exact congrArg
    (fun columnMap => terms columnMap leftSource)
    (columnMap_eq_of_constraintPolynomial_eq template left right same)

theorem challenge_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (source : Fin FixedActive.arity.total) :
    ConcreteNifsPiRlcActionRows.challenge
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame source =
      ConcreteNifsPiRlcActionRows.challenge
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame source := by
  funext lane
  unfold ConcreteNifsPiRlcActionRows.challenge
  apply carriedF_eq_of_source_eq template left right same
  apply fCarried_eq_of_numeric_eq
  apply proofFNumeric_eq_of_ids_and_index
  · simpa only [orderedIds] using
      orderedIds_eq_of_constraintPolynomial_eq template left right same
  · exact proofOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  ·
    cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl

theorem commitmentValue_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (row : Fin verifierRows) :
    ConcreteNifsPiRlcActionRows.commitmentValue
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame row =
  ConcreteNifsPiRlcActionRows.commitmentValue
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame row := by
  unfold ConcreteNifsPiRlcActionRows.commitmentValue
  funext source
  refine Fin.addCases ?_ ?_ source
  · intro fresh
    simp only [Fin.addCases_left]
    funext lane
    apply carriedF_eq_of_source_eq template left right same
    apply fCarried_eq_of_numeric_eq
    apply freshFNumeric_eq_of_ids_and_index
    · simpa only [orderedIds] using
        orderedIds_eq_of_constraintPolynomial_eq template left right same
    · exact freshOperandIds_eq_of_constraintPolynomial_eq
        template left right same
    ·
      cases left with
      | mk leftMatrices polynomial =>
          cases right with
          | mk rightMatrices rightPolynomial =>
              simp only at same
              subst rightPolynomial
              rfl
  · intro child
    simp only [Fin.addCases_right]
    funext lane
    apply carriedF_eq_of_source_eq template left right same
    apply fCarried_eq_of_numeric_eq
    apply runningFNumeric_eq_of_ids_and_index
    · simpa only [orderedIds] using
        orderedIds_eq_of_constraintPolynomial_eq template left right same
    · exact runningOperandIds_eq_of_constraintPolynomial_eq
        template left right same
    ·
      cases left with
      | mk leftMatrices polynomial =>
          cases right with
          | mk rightMatrices rightPolynomial =>
              simp only at same
              subst rightPolynomial
              rfl

theorem publicValue_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (block : Fin publicRingColumns) :
    ConcreteNifsPiRlcActionRows.publicValue
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame block =
  ConcreteNifsPiRlcActionRows.publicValue
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame block := by
  unfold ConcreteNifsPiRlcActionRows.publicValue
  funext source
  refine Fin.addCases ?_ ?_ source
  · intro fresh
    simp only [Fin.addCases_left]
    funext lane
    apply carriedF_eq_of_source_eq template left right same
    apply fCarried_eq_of_numeric_eq
    apply freshFNumeric_eq_of_ids_and_index
    · simpa only [orderedIds] using
        orderedIds_eq_of_constraintPolynomial_eq template left right same
    · exact freshOperandIds_eq_of_constraintPolynomial_eq
        template left right same
    ·
      cases left with
      | mk leftMatrices polynomial =>
          cases right with
          | mk rightMatrices rightPolynomial =>
              simp only at same
              subst rightPolynomial
              rfl
  · intro child
    simp only [Fin.addCases_right]
    funext lane
    apply carriedF_eq_of_source_eq template left right same
    apply fCarried_eq_of_numeric_eq
    apply runningFNumeric_eq_of_ids_and_index
    · simpa only [orderedIds] using
        orderedIds_eq_of_constraintPolynomial_eq template left right same
    · exact runningOperandIds_eq_of_constraintPolynomial_eq
        template left right same
    ·
      cases left with
      | mk leftMatrices polynomial =>
          cases right with
          | mk rightMatrices rightPolynomial =>
              simp only at same
              subst rightPolynomial
              rfl

theorem evaluationValueLow_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (matrix : Fin dimensions.matrixCount) :
    ConcreteNifsPiRlcActionRows.evaluationValueLow
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame matrix =
      ConcreteNifsPiRlcActionRows.evaluationValueLow
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame matrix := by
  funext source lane
  unfold ConcreteNifsPiRlcActionRows.evaluationValueLow
  apply carriedF_eq_of_source_eq template left right same
  exact congrArg (fun carried => carried.low)
    (carried_eq_of_numeric_eq _ _
      (outputYRing_numeric_eq_of_constraintPolynomial_eq
        template left right same _ matrix lane))

theorem evaluationValueHigh_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (matrix : Fin dimensions.matrixCount) :
    ConcreteNifsPiRlcActionRows.evaluationValueHigh
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame matrix =
      ConcreteNifsPiRlcActionRows.evaluationValueHigh
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame matrix := by
  funext source lane
  unfold ConcreteNifsPiRlcActionRows.evaluationValueHigh
  apply carriedF_eq_of_source_eq template left right same
  exact congrArg (fun carried => carried.high)
    (carried_eq_of_numeric_eq _ _
      (outputYRing_numeric_eq_of_constraintPolynomial_eq
        template left right same _ matrix lane))

theorem commitmentOutput_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (row : Fin verifierRows) :
    ConcreteNifsPiRlcActionRows.commitmentOutput
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame row =
      ConcreteNifsPiRlcActionRows.commitmentOutput
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame row := by
  funext lane
  unfold ConcreteNifsPiRlcActionRows.commitmentOutput
  apply carriedF_eq_of_source_eq template left right same
  simpa only [ConcreteNifsCarrierViews.RunningCoordinate.view] using
    outputRunningFCarried_eq_of_constraintPolynomial_eq
      template left right same (.parentCommitment row lane)

theorem publicOutput_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (block : Fin publicRingColumns) :
    ConcreteNifsPiRlcActionRows.publicOutput
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame block =
      ConcreteNifsPiRlcActionRows.publicOutput
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame block := by
  funext lane
  unfold ConcreteNifsPiRlcActionRows.publicOutput
  apply carriedF_eq_of_source_eq template left right same
  simpa only [ConcreteNifsCarrierViews.RunningCoordinate.view] using
    outputRunningFCarried_eq_of_constraintPolynomial_eq
      template left right same (.parentPublic
        (ConcreteNifsPiRlcActionRows.publicCoordinate block lane))

theorem evaluationOutputLow_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (matrix : Fin dimensions.matrixCount) :
    ConcreteNifsPiRlcActionRows.evaluationOutputLow
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame matrix =
      ConcreteNifsPiRlcActionRows.evaluationOutputLow
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame matrix := by
  funext lane
  unfold ConcreteNifsPiRlcActionRows.evaluationOutputLow
  apply carriedF_eq_of_source_eq template left right same
  exact congrArg
    (fun coordinate => coordinate.parent.low)
    (piDecEvaluationCoordinate_eq_of_constraintPolynomial_eq
      template left right same matrix lane)

theorem evaluationOutputHigh_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (matrix : Fin dimensions.matrixCount) :
    ConcreteNifsPiRlcActionRows.evaluationOutputHigh
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame matrix =
      ConcreteNifsPiRlcActionRows.evaluationOutputHigh
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame matrix := by
  funext lane
  unfold ConcreteNifsPiRlcActionRows.evaluationOutputHigh
  apply carriedF_eq_of_source_eq template left right same
  exact congrArg
    (fun coordinate => coordinate.parent.high)
    (piDecEvaluationCoordinate_eq_of_constraintPolynomial_eq
      template left right same matrix lane)

theorem actionBase_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ConcreteNifsRawProgram.actionBase
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame =
      ConcreteNifsRawProgram.actionBase
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame := by
  unfold ConcreteNifsRawProgram.actionBase
  have baseEqual :=
    samplerBase_eq_of_constraintPolynomial_eq
      template left right same
  unfold samplerBase at baseEqual
  exact congrArg
    (fun base =>
      base +
        Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram.cost.auxiliaryColumns)
    baseEqual

private theorem frame_ext
    {count : Nat}
    (left right : Phi81RingAction.Frame count)
    (ownerEqual : left.owner = right.owner)
    (ordinalEqual : left.firstOrdinal = right.firstOrdinal)
    (oneEqual : left.one = right.one)
    (challengeEqual : left.challenges = right.challenges)
    (valuesEqual : left.values = right.values)
    (outputEqual : left.output = right.output)
    (productEqual : left.productColumn = right.productColumn) :
    left = right := by
  cases left
  cases right
  simp only at ownerEqual ordinalEqual oneEqual challengeEqual valuesEqual outputEqual productEqual
  cases ownerEqual
  cases ordinalEqual
  cases oneEqual
  cases challengeEqual
  cases valuesEqual
  cases outputEqual
  cases productEqual
  rfl

private theorem actionFrame_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (leftBase rightBase target : Nat)
    (baseEqual : leftBase = rightBase)
    (leftValues rightValues :
      Fin FixedActive.arity.total → Phi81RingAction.CarriedRing)
    (valuesEqual : leftValues = rightValues)
    (leftOutput rightOutput : Phi81RingAction.CarriedRing)
    (outputEqual : leftOutput = rightOutput) :
    ConcreteNifsPiRlcActionRows.actionFrame
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame
        leftBase target leftValues leftOutput =
      ConcreteNifsPiRlcActionRows.actionFrame
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame
        rightBase target rightValues rightOutput := by
  apply frame_ext
  ·
    cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl
  · rfl
  · exact one_eq_of_constraintPolynomial_eq template left right same
  ·
    funext source
    exact challenge_eq_of_constraintPolynomial_eq
      template left right same source
  · exact valuesEqual
  · exact outputEqual
  ·
    funext source leftIndex rightIndex
    unfold ConcreteNifsPiRlcActionRows.actionFrame
    rw [baseEqual]
    exact congrArg
      (fun columnMap =>
        columnMap
          (ConcreteNifsPiRlcActionRows.productSource
            rightBase target source leftIndex rightIndex))
      (columnMap_eq_of_constraintPolynomial_eq
        template left right same)

theorem commitmentFrame_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (row : Fin verifierRows) :
    ConcreteNifsPiRlcActionRows.commitmentFrame
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame
        (ConcreteNifsRawProgram.actionBase
          (application (template.withSystem left))
          (operational (template.withSystem left))
          (invokePlan (template.withSystem left)).frame)
        row =
      ConcreteNifsPiRlcActionRows.commitmentFrame
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame
        (ConcreteNifsRawProgram.actionBase
          (application (template.withSystem right))
          (operational (template.withSystem right))
          (invokePlan (template.withSystem right)).frame)
        row := by
  unfold ConcreteNifsPiRlcActionRows.commitmentFrame
  apply actionFrame_eq_of_constraintPolynomial_eq
    template left right same
  · exact actionBase_eq_of_constraintPolynomial_eq
      template left right same
  · exact commitmentValue_eq_of_constraintPolynomial_eq
      template left right same row
  · exact commitmentOutput_eq_of_constraintPolynomial_eq
      template left right same row

theorem publicFrame_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (block : Fin publicRingColumns) :
    ConcreteNifsPiRlcActionRows.publicFrame
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame
        (ConcreteNifsRawProgram.actionBase
          (application (template.withSystem left))
          (operational (template.withSystem left))
          (invokePlan (template.withSystem left)).frame)
        block =
      ConcreteNifsPiRlcActionRows.publicFrame
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame
        (ConcreteNifsRawProgram.actionBase
          (application (template.withSystem right))
          (operational (template.withSystem right))
          (invokePlan (template.withSystem right)).frame)
        block := by
  unfold ConcreteNifsPiRlcActionRows.publicFrame
  apply actionFrame_eq_of_constraintPolynomial_eq
    template left right same
  · exact actionBase_eq_of_constraintPolynomial_eq
      template left right same
  · exact publicValue_eq_of_constraintPolynomial_eq
      template left right same block
  · exact publicOutput_eq_of_constraintPolynomial_eq
      template left right same block

theorem evaluationLowFrame_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (matrix : Fin dimensions.matrixCount) :
    ConcreteNifsPiRlcActionRows.evaluationLowFrame
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame
        (ConcreteNifsRawProgram.actionBase
          (application (template.withSystem left))
          (operational (template.withSystem left))
          (invokePlan (template.withSystem left)).frame)
        matrix =
      ConcreteNifsPiRlcActionRows.evaluationLowFrame
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame
        (ConcreteNifsRawProgram.actionBase
          (application (template.withSystem right))
          (operational (template.withSystem right))
          (invokePlan (template.withSystem right)).frame)
        matrix := by
  unfold ConcreteNifsPiRlcActionRows.evaluationLowFrame
  apply actionFrame_eq_of_constraintPolynomial_eq
    template left right same
  · exact actionBase_eq_of_constraintPolynomial_eq
      template left right same
  · exact evaluationValueLow_eq_of_constraintPolynomial_eq
      template left right same matrix
  · exact evaluationOutputLow_eq_of_constraintPolynomial_eq
      template left right same matrix

theorem evaluationHighFrame_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (matrix : Fin dimensions.matrixCount) :
    ConcreteNifsPiRlcActionRows.evaluationHighFrame
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame
        (ConcreteNifsRawProgram.actionBase
          (application (template.withSystem left))
          (operational (template.withSystem left))
          (invokePlan (template.withSystem left)).frame)
        matrix =
      ConcreteNifsPiRlcActionRows.evaluationHighFrame
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame
        (ConcreteNifsRawProgram.actionBase
          (application (template.withSystem right))
          (operational (template.withSystem right))
          (invokePlan (template.withSystem right)).frame)
        matrix := by
  unfold ConcreteNifsPiRlcActionRows.evaluationHighFrame
  apply actionFrame_eq_of_constraintPolynomial_eq
    template left right same
  · exact actionBase_eq_of_constraintPolynomial_eq
      template left right same
  · exact evaluationValueHigh_eq_of_constraintPolynomial_eq
      template left right same matrix
  · exact evaluationOutputHigh_eq_of_constraintPolynomial_eq
      template left right same matrix

theorem frames_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ConcreteNifsPiRlcActionRows.frames
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame
        (ConcreteNifsRawProgram.actionBase
          (application (template.withSystem left))
          (operational (template.withSystem left))
          (invokePlan (template.withSystem left)).frame) =
      ConcreteNifsPiRlcActionRows.frames
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame
        (ConcreteNifsRawProgram.actionBase
          (application (template.withSystem right))
          (operational (template.withSystem right))
          (invokePlan (template.withSystem right)).frame) := by
  unfold ConcreteNifsPiRlcActionRows.frames
  have commitmentEqual :
      List.ofFn
          (ConcreteNifsPiRlcActionRows.commitmentFrame
            (application (template.withSystem left))
            (operational (template.withSystem left))
            (invokePlan (template.withSystem left)).frame
            (ConcreteNifsRawProgram.actionBase
              (application (template.withSystem left))
              (operational (template.withSystem left))
              (invokePlan (template.withSystem left)).frame)) =
        List.ofFn
          (ConcreteNifsPiRlcActionRows.commitmentFrame
            (application (template.withSystem right))
            (operational (template.withSystem right))
            (invokePlan (template.withSystem right)).frame
            (ConcreteNifsRawProgram.actionBase
              (application (template.withSystem right))
              (operational (template.withSystem right))
              (invokePlan (template.withSystem right)).frame)) := by
    apply congrArg List.ofFn
    funext row
    exact commitmentFrame_eq_of_constraintPolynomial_eq
      template left right same row
  have publicEqual :
      List.ofFn
          (ConcreteNifsPiRlcActionRows.publicFrame
            (application (template.withSystem left))
            (operational (template.withSystem left))
            (invokePlan (template.withSystem left)).frame
            (ConcreteNifsRawProgram.actionBase
              (application (template.withSystem left))
              (operational (template.withSystem left))
              (invokePlan (template.withSystem left)).frame)) =
        List.ofFn
          (ConcreteNifsPiRlcActionRows.publicFrame
            (application (template.withSystem right))
            (operational (template.withSystem right))
            (invokePlan (template.withSystem right)).frame
            (ConcreteNifsRawProgram.actionBase
              (application (template.withSystem right))
              (operational (template.withSystem right))
              (invokePlan (template.withSystem right)).frame)) := by
    apply congrArg List.ofFn
    funext block
    exact publicFrame_eq_of_constraintPolynomial_eq
      template left right same block
  have evaluationLowEqual :
      List.ofFn
          (ConcreteNifsPiRlcActionRows.evaluationLowFrame
            (application (template.withSystem left))
            (operational (template.withSystem left))
            (invokePlan (template.withSystem left)).frame
            (ConcreteNifsRawProgram.actionBase
              (application (template.withSystem left))
              (operational (template.withSystem left))
              (invokePlan (template.withSystem left)).frame)) =
        List.ofFn
          (ConcreteNifsPiRlcActionRows.evaluationLowFrame
            (application (template.withSystem right))
            (operational (template.withSystem right))
            (invokePlan (template.withSystem right)).frame
            (ConcreteNifsRawProgram.actionBase
              (application (template.withSystem right))
              (operational (template.withSystem right))
              (invokePlan (template.withSystem right)).frame)) := by
    apply congrArg List.ofFn
    funext matrix
    exact evaluationLowFrame_eq_of_constraintPolynomial_eq
      template left right same matrix
  have evaluationHighEqual :
      List.ofFn
          (ConcreteNifsPiRlcActionRows.evaluationHighFrame
            (application (template.withSystem left))
            (operational (template.withSystem left))
            (invokePlan (template.withSystem left)).frame
            (ConcreteNifsRawProgram.actionBase
              (application (template.withSystem left))
              (operational (template.withSystem left))
              (invokePlan (template.withSystem left)).frame)) =
        List.ofFn
          (ConcreteNifsPiRlcActionRows.evaluationHighFrame
            (application (template.withSystem right))
            (operational (template.withSystem right))
            (invokePlan (template.withSystem right)).frame
            (ConcreteNifsRawProgram.actionBase
              (application (template.withSystem right))
              (operational (template.withSystem right))
              (invokePlan (template.withSystem right)).frame)) := by
    apply congrArg List.ofFn
    funext matrix
    exact evaluationHighFrame_eq_of_constraintPolynomial_eq
      template left right same matrix
  rw [commitmentEqual, publicEqual, evaluationLowEqual,
    evaluationHighEqual]

theorem actionRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    actionRows (template.withSystem left) =
      actionRows (template.withSystem right) := by
  unfold actionRows ConcreteNifsPiRlcActionRows.rows
  rw [frames_eq_of_constraintPolynomial_eq
    template left right same]

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ActionRows
