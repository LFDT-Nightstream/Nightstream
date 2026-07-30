import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperationalInputEquality
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4CarrierLocations
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ProofRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RunningRows

/-!
Contract: prove physical row stability for the complete selected NIFS call
when only the relation matrix payload changes.

Assurance tier: model-level.

Owns: exact row equality for the running-authority, Fiat-Shamir sampler,
Pi-RLC point, Pi-RLC action, Pi-DEC, output, and Split-NC parts of the
benchmark NIFS program.

Does not own: Step assembly, fixed-point compilation, a production deployment
selection, Rust equality, or a security reduction.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4NifsRows

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperationalInput
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperationalInputEquality
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4CarrierLocations
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperandFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4ProofFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RunningFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

noncomputable def operationalRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsOperationalOccurrence.rows
    (application setup) (operational setup) (invokePlan setup).frame

theorem operationalRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    operationalRows (template.withSystem left) =
      operationalRows (template.withSystem right) := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          unfold operationalRows
            ConcreteNifsOperationalOccurrence.rows
          rw [show
            ConcreteNifsOperationalOccurrence.input
                (application
                  (template.withSystem
                    { matrices := leftMatrices
                      constraintPolynomial := polynomial }))
                (operational
                  (template.withSystem
                    { matrices := leftMatrices
                      constraintPolynomial := polynomial }))
                (invokePlan
                  (template.withSystem
                    { matrices := leftMatrices
                      constraintPolynomial := polynomial })).frame =
              ConcreteNifsOperationalOccurrence.input
                (application
                  (template.withSystem
                    { matrices := rightMatrices
                      constraintPolynomial := polynomial }))
                (operational
                  (template.withSystem
                    { matrices := rightMatrices
                      constraintPolynomial := polynomial }))
                (invokePlan
                  (template.withSystem
                    { matrices := rightMatrices
                      constraintPolynomial := polynomial })).frame
            from eq_of_heq
              (input_heq_of_constraintPolynomial_eq
                template
                { matrices := leftMatrices
                  constraintPolynomial := polynomial }
                { matrices := rightMatrices
                  constraintPolynomial := polynomial }
                rfl)]
          rfl

noncomputable def samplerRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsOperationalSampler.rows
    (application setup) (operational setup) (invokePlan setup).frame

noncomputable def samplerBase
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsOperationalSampler.samplerBase
    (application setup) (operational setup) (invokePlan setup).frame

noncomputable def samplerLanes
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsOperationalSampler.samplerLanes
    (application setup) (operational setup) (invokePlan setup).frame

noncomputable def intrinsicSamplerRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsOperationalSampler.samplerRows
    (application setup) (operational setup) (invokePlan setup).frame

noncomputable def challengeRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsOperationalSampler.challengeRows
    (application setup) (operational setup) (invokePlan setup).frame

theorem samplerBase_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    samplerBase (template.withSystem left) =
      samplerBase (template.withSystem right) := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          unfold samplerBase
            ConcreteNifsOperationalSampler.samplerBase
          exact congrArg KSplitNcOperationalRows.afterAllocation
            (eq_of_heq
              (input_heq_of_constraintPolynomial_eq
                template
                { matrices := leftMatrices
                  constraintPolynomial := polynomial }
                { matrices := rightMatrices
                  constraintPolynomial := polynomial }
                rfl))

theorem samplerLanes_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    samplerLanes (template.withSystem left) =
      samplerLanes (template.withSystem right) := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          unfold samplerLanes
            ConcreteNifsOperationalSampler.samplerLanes
          exact congrArg
            (fun transcript =>
              (KSplitNcTranscript.outputBuilder transcript).lanes)
            (eq_of_heq
              (transcript_heq_of_constraintPolynomial_eq
                template
                { matrices := leftMatrices
                  constraintPolynomial := polynomial }
                { matrices := rightMatrices
                  constraintPolynomial := polynomial }
                rfl))

theorem constants_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (operational (template.withSystem left)).constants =
      (operational (template.withSystem right)).constants := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          rfl

theorem intrinsicSamplerRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    intrinsicSamplerRows (template.withSystem left) =
    intrinsicSamplerRows (template.withSystem right) := by
  unfold intrinsicSamplerRows
    ConcreteNifsOperationalSampler.samplerRows
  have baseEqual :=
    samplerBase_eq_of_constraintPolynomial_eq
      template left right same
  have lanesEqual :=
    samplerLanes_eq_of_constraintPolynomial_eq
      template left right same
  unfold samplerBase at baseEqual
  unfold samplerLanes at lanesEqual
  rw [baseEqual,
    constants_eq_of_constraintPolynomial_eq template left right same,
    lanesEqual]

theorem challengeCarried_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (coordinate :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total)
    (position : Fin ringDegree) :
    (ConcreteNifsOperationalSampler.challengeLocation
      (application (template.withSystem left))
      (operational (template.withSystem left))
      (invokePlan (template.withSystem left)).frame
      coordinate position).carried =
    (ConcreteNifsOperationalSampler.challengeLocation
      (application (template.withSystem right))
      (operational (template.withSystem right))
      (invokePlan (template.withSystem right)).frame
      coordinate position).carried := by
  apply fCarried_eq_of_numeric_eq
  unfold ConcreteNifsOperationalSampler.challengeLocation
    ConcreteNifsOperationalOccurrence.proofFieldLocation
  apply proofFNumeric_eq_of_ids_and_index
  · exact orderedIds_eq_of_constraintPolynomial_eq
      template left right same
  · exact proofOperandIds_eq_of_constraintPolynomial_eq
      template left right same
  · cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl

theorem challengeRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    challengeRows (template.withSystem left) =
      challengeRows (template.withSystem right) := by
  unfold challengeRows
    ConcreteNifsOperationalSampler.challengeRows
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext coordinate
  apply congrArg List.ofFn
  funext position
  unfold ConcreteNifsOperationalSampler.challengeRow
  have baseEqual :=
    samplerBase_eq_of_constraintPolynomial_eq
      template left right same
  unfold samplerBase at baseEqual
  rw [baseEqual,
    challengeCarried_eq_of_constraintPolynomial_eq
      template left right same coordinate position]

theorem samplerRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    samplerRows (template.withSystem left) =
      samplerRows (template.withSystem right) := by
  unfold samplerRows ConcreteNifsOperationalSampler.rows
  have operationalEqual :=
    operationalRows_eq_of_constraintPolynomial_eq
      template left right same
  change
    ConcreteNifsOperationalOccurrence.rows
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame =
      ConcreteNifsOperationalOccurrence.rows
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame
    at operationalEqual
  rw [operationalEqual]
  have intrinsicEqual :=
    intrinsicSamplerRows_eq_of_constraintPolynomial_eq
      template left right same
  have challengeEqual :=
    challengeRows_eq_of_constraintPolynomial_eq
      template left right same
  unfold intrinsicSamplerRows at intrinsicEqual
  unfold challengeRows at challengeEqual
  rw [intrinsicEqual, challengeEqual]

noncomputable def pointRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsPiRlcPointRows.rows
    (application setup) (operational setup) (invokePlan setup).frame

noncomputable def actionRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsPiRlcActionRows.rows
    (application setup) (operational setup) (invokePlan setup).frame
    (ConcreteNifsRawProgram.actionBase
      (application setup) (operational setup) (invokePlan setup).frame)

noncomputable def piDecRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsPiDecRows.rows
    (application setup) (operational setup) (invokePlan setup).frame

noncomputable def outputRows
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :=
  ConcreteNifsOutputRows.rows
    (application setup) (operational setup) (invokePlan setup).frame

theorem pointTranscriptCoordinate_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (coordinate : Fin dimensions.rowVariables) :
    ConcreteNifsPiRlcPointRows.transcriptCoordinate
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame coordinate =
      ConcreteNifsPiRlcPointRows.transcriptCoordinate
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame coordinate := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          exact congrArg
            (fun input =>
              KSplitNcEndpoints.feRowPoint
                (KSplitNcOperationalRows.endpointInput input) coordinate)
            (eq_of_heq
              (input_heq_of_constraintPolynomial_eq
                template
                { matrices := leftMatrices
                  constraintPolynomial := polynomial }
                { matrices := rightMatrices
                  constraintPolynomial := polynomial }
                rfl))

theorem pointOutputCoordinate_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (coordinate : Fin dimensions.rowVariables) :
    ConcreteNifsPiRlcPointRows.outputCoordinate
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame coordinate =
      ConcreteNifsPiRlcPointRows.outputCoordinate
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame coordinate := by
  apply carried_eq_of_numeric_eq
  apply outputKNumeric_eq_of_ids_and_indices
  · simpa only [orderedIds] using
      orderedIds_eq_of_constraintPolynomial_eq template left right same
  · simpa only [outputIds,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.unaryOutput_ids] using
      outputIds_eq_of_constraintPolynomial_eq template left right same
  ·
    cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl
  ·
    cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl

theorem pointRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    pointRows (template.withSystem left) =
      pointRows (template.withSystem right) := by
  unfold pointRows ConcreteNifsPiRlcPointRows.rows
  apply congrArg
    ((List.ofFn fun coordinate : Fin dimensions.rowVariables =>
      coordinate).flatMap ·)
  funext coordinate
  unfold ConcreteNifsPiRlcPointRows.coordinateRows
  rw [pointTranscriptCoordinate_eq_of_constraintPolynomial_eq
      template left right same coordinate,
    pointOutputCoordinate_eq_of_constraintPolynomial_eq
      template left right same coordinate]

private theorem fCoordinate_ext
    (left right : Phi81RadixRows.FCoordinate)
    (children : ∀ child, left.children child = right.children child)
    (parent : left.parent = right.parent) :
    left = right := by
  cases left
  cases right
  simp only at children parent
  have childrenEqual := funext children
  cases childrenEqual
  cases parent
  rfl

private theorem kCoordinate_ext
    (left right : Phi81RadixRows.KCoordinate)
    (children : ∀ child, left.children child = right.children child)
    (parent : left.parent = right.parent) :
    left = right := by
  cases left
  cases right
  simp only at children parent
  have childrenEqual := funext children
  cases childrenEqual
  cases parent
  rfl

theorem piDecCommitmentCoordinate_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (row : Fin verifierRows)
    (lane : Fin ringDegree) :
    ConcreteNifsPiDecRows.commitmentCoordinate
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame row lane =
      ConcreteNifsPiDecRows.commitmentCoordinate
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame row lane := by
  apply fCoordinate_ext
  · intro child
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
  · apply fCarried_eq_of_numeric_eq
    apply outputFNumeric_eq_of_ids_and_index
    · simpa only [orderedIds] using
        orderedIds_eq_of_constraintPolynomial_eq template left right same
    · simpa only [outputIds,
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.unaryOutput_ids] using
        outputIds_eq_of_constraintPolynomial_eq template left right same
    ·
      cases left with
      | mk leftMatrices polynomial =>
          cases right with
          | mk rightMatrices rightPolynomial =>
              simp only at same
              subst rightPolynomial
              rfl

theorem piDecPublicCoordinate_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (column : Fin (ringDegree * publicRingColumns)) :
    ConcreteNifsPiDecRows.publicCoordinate
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame column =
      ConcreteNifsPiDecRows.publicCoordinate
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame column := by
  apply fCoordinate_ext
  · intro child
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
  · apply fCarried_eq_of_numeric_eq
    apply outputFNumeric_eq_of_ids_and_index
    · simpa only [orderedIds] using
        orderedIds_eq_of_constraintPolynomial_eq template left right same
    · simpa only [outputIds,
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.unaryOutput_ids] using
        outputIds_eq_of_constraintPolynomial_eq template left right same
    ·
      cases left with
      | mk leftMatrices polynomial =>
          cases right with
          | mk rightMatrices rightPolynomial =>
              simp only at same
              subst rightPolynomial
              rfl

theorem piDecEvaluationCoordinate_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (matrix : Fin dimensions.matrixCount)
    (lane : Fin ringDegree) :
    ConcreteNifsPiDecRows.evaluationCoordinate
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame matrix lane =
      ConcreteNifsPiDecRows.evaluationCoordinate
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame matrix lane := by
  apply kCoordinate_ext
  · intro child
    apply carried_eq_of_numeric_eq
    apply proofNumeric_eq_of_ids_and_indices
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
    ·
      cases left with
      | mk leftMatrices polynomial =>
          cases right with
          | mk rightMatrices rightPolynomial =>
              simp only at same
              subst rightPolynomial
              rfl
  · apply carried_eq_of_numeric_eq
    apply outputKNumeric_eq_of_ids_and_indices
    · simpa only [orderedIds] using
        orderedIds_eq_of_constraintPolynomial_eq template left right same
    · simpa only [outputIds,
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.unaryOutput_ids] using
        outputIds_eq_of_constraintPolynomial_eq template left right same
    ·
      cases left with
      | mk leftMatrices polynomial =>
          cases right with
          | mk rightMatrices rightPolynomial =>
              simp only at same
              subst rightPolynomial
              rfl
    ·
      cases left with
      | mk leftMatrices polynomial =>
          cases right with
          | mk rightMatrices rightPolynomial =>
              simp only at same
              subst rightPolynomial
              rfl

theorem piDecRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    piDecRows (template.withSystem left) =
      piDecRows (template.withSystem right) := by
  unfold piDecRows ConcreteNifsPiDecRows.rows
    ConcreteNifsPiDecRows.fCoordinates
    ConcreteNifsPiDecRows.commitmentCoordinates
    ConcreteNifsPiDecRows.publicCoordinates
    ConcreteNifsPiDecRows.evaluationCoordinates
  apply congrArg₂ Phi81RadixRows.rows
  · apply congrArg₂ (· ++ ·)
    · apply congrArg List.flatten
      rw [List.map_ofFn, List.map_ofFn]
      apply congrArg List.ofFn
      funext row
      apply congrArg List.ofFn
      funext lane
      exact piDecCommitmentCoordinate_eq_of_constraintPolynomial_eq
        template left right same row lane
    · apply congrArg List.ofFn
      funext column
      exact piDecPublicCoordinate_eq_of_constraintPolynomial_eq
        template left right same column
  · apply congrArg List.flatten
    rw [List.map_ofFn, List.map_ofFn]
    apply congrArg List.ofFn
    funext matrix
    apply congrArg List.ofFn
    funext lane
    exact piDecEvaluationCoordinate_eq_of_constraintPolynomial_eq
      template left right same matrix lane

theorem outputRunningFCarried_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (coordinate : RunningCoordinate dimensions verifierRows) :
    (ConcreteNifsCarrierFrame.outputFLocation
      (application (template.withSystem left)).family
      (invokePlan (template.withSystem left)).frame
      (ConcreteNifsCarrierViews.RunningCoordinate.view
        (operational (template.withSystem left)).runningViews
        coordinate)).carried =
    (ConcreteNifsCarrierFrame.outputFLocation
      (application (template.withSystem right)).family
      (invokePlan (template.withSystem right)).frame
      (ConcreteNifsCarrierViews.RunningCoordinate.view
        (operational (template.withSystem right)).runningViews
        coordinate)).carried := by
  apply fCarried_eq_of_numeric_eq
  apply outputFNumeric_eq_of_ids_and_index
  · simpa only [orderedIds] using
      orderedIds_eq_of_constraintPolynomial_eq template left right same
  · simpa only [outputIds,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.unaryOutput_ids] using
      outputIds_eq_of_constraintPolynomial_eq template left right same
  ·
    cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            cases coordinate <;> rfl

theorem outputChildPoint_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (child : Fin productionGlobalParams.k)
    (coordinate : Fin dimensions.rowVariables) :
    ConcreteNifsOutputRows.outputChildPoint
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame child coordinate =
      ConcreteNifsOutputRows.outputChildPoint
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame child coordinate := by
  apply carried_eq_of_numeric_eq
  apply outputKNumeric_eq_of_ids_and_indices
  · simpa only [orderedIds] using
      orderedIds_eq_of_constraintPolynomial_eq template left right same
  · simpa only [outputIds,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.unaryOutput_ids] using
      outputIds_eq_of_constraintPolynomial_eq template left right same
  ·
    cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl
  ·
    cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl

theorem outputChildEvaluation_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (child : Fin productionGlobalParams.k)
    (matrix : Fin dimensions.matrixCount)
    (lane : Fin ringDegree) :
    ConcreteNifsOutputRows.outputChildEvaluation
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame child matrix lane =
      ConcreteNifsOutputRows.outputChildEvaluation
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame child matrix lane := by
  apply carried_eq_of_numeric_eq
  apply outputKNumeric_eq_of_ids_and_indices
  · simpa only [orderedIds] using
      orderedIds_eq_of_constraintPolynomial_eq template left right same
  · simpa only [outputIds,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.unaryOutput_ids] using
      outputIds_eq_of_constraintPolynomial_eq template left right same
  ·
    cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl
  ·
    cases left with
    | mk leftMatrices polynomial =>
        cases right with
        | mk rightMatrices rightPolynomial =>
            simp only at same
            subst rightPolynomial
            rfl

theorem outputChildRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (child : Fin productionGlobalParams.k) :
    ConcreteNifsOutputRows.childRows
        (application (template.withSystem left))
        (operational (template.withSystem left))
        (invokePlan (template.withSystem left)).frame child =
      ConcreteNifsOutputRows.childRows
        (application (template.withSystem right))
        (operational (template.withSystem right))
        (invokePlan (template.withSystem right)).frame child := by
  have commitmentEqual :
      ConcreteNifsOutputRows.commitmentRows
          (application (template.withSystem left))
          (operational (template.withSystem left))
          (invokePlan (template.withSystem left)).frame child =
        ConcreteNifsOutputRows.commitmentRows
          (application (template.withSystem right))
          (operational (template.withSystem right))
          (invokePlan (template.withSystem right)).frame child := by
    unfold ConcreteNifsOutputRows.commitmentRows
    apply congrArg List.flatten
    rw [List.map_ofFn, List.map_ofFn]
    apply congrArg List.ofFn
    funext row
    simp only [Function.comp_apply]
    rw [List.map_ofFn, List.map_ofFn]
    apply congrArg List.ofFn
    funext lane
    unfold ConcreteNifsOutputRows.outputChildCommitment
      ConcreteNifsOutputRows.proofChildCommitment
    apply congrArg₂ KEquality.equalityRow
    · simpa only [ConcreteNifsCarrierViews.RunningCoordinate.view] using
        outputRunningFCarried_eq_of_constraintPolynomial_eq
          template left right same (.childCommitment child row lane)
    · exact congrArg
        (fun coordinate => coordinate.children child)
        (piDecCommitmentCoordinate_eq_of_constraintPolynomial_eq
          template left right same row lane)
  have publicEqual :
      ConcreteNifsOutputRows.publicRows
          (application (template.withSystem left))
          (operational (template.withSystem left))
          (invokePlan (template.withSystem left)).frame child =
        ConcreteNifsOutputRows.publicRows
          (application (template.withSystem right))
          (operational (template.withSystem right))
          (invokePlan (template.withSystem right)).frame child := by
    unfold ConcreteNifsOutputRows.publicRows
    rw [List.map_ofFn, List.map_ofFn]
    apply congrArg List.ofFn
    funext column
    simp only [Function.comp_apply]
    unfold ConcreteNifsOutputRows.outputChildPublic
      ConcreteNifsOutputRows.proofChildPublic
    apply congrArg₂ KEquality.equalityRow
    · simpa only [ConcreteNifsCarrierViews.RunningCoordinate.view] using
        outputRunningFCarried_eq_of_constraintPolynomial_eq
          template left right same (.childPublic child column)
    · exact congrArg
        (fun coordinate => coordinate.children child)
        (piDecPublicCoordinate_eq_of_constraintPolynomial_eq
          template left right same column)
  have pointEqual :
      ConcreteNifsOutputRows.pointRows
          (application (template.withSystem left))
          (operational (template.withSystem left))
          (invokePlan (template.withSystem left)).frame child =
        ConcreteNifsOutputRows.pointRows
          (application (template.withSystem right))
          (operational (template.withSystem right))
          (invokePlan (template.withSystem right)).frame child := by
    unfold ConcreteNifsOutputRows.pointRows
    apply congrArg List.flatten
    rw [List.map_ofFn, List.map_ofFn]
    apply congrArg List.ofFn
    funext coordinate
    simp only [Function.comp_apply]
    apply congrArg₂ KEquality.rows
    · exact outputChildPoint_eq_of_constraintPolynomial_eq
        template left right same child coordinate
    · change
        ConcreteNifsPiRlcPointRows.outputCoordinate
            (application (template.withSystem left))
            (operational (template.withSystem left))
            (invokePlan (template.withSystem left)).frame coordinate =
          ConcreteNifsPiRlcPointRows.outputCoordinate
            (application (template.withSystem right))
            (operational (template.withSystem right))
            (invokePlan (template.withSystem right)).frame coordinate
      exact pointOutputCoordinate_eq_of_constraintPolynomial_eq
        template left right same coordinate
  have evaluationEqual :
      ConcreteNifsOutputRows.evaluationRows
          (application (template.withSystem left))
          (operational (template.withSystem left))
          (invokePlan (template.withSystem left)).frame child =
        ConcreteNifsOutputRows.evaluationRows
          (application (template.withSystem right))
          (operational (template.withSystem right))
          (invokePlan (template.withSystem right)).frame child := by
    unfold ConcreteNifsOutputRows.evaluationRows
    apply congrArg List.flatten
    rw [List.map_ofFn, List.map_ofFn]
    apply congrArg List.ofFn
    funext matrix
    simp only [Function.comp_apply]
    apply congrArg List.flatten
    rw [List.map_ofFn, List.map_ofFn]
    apply congrArg List.ofFn
    funext lane
    simp only [Function.comp_apply]
    apply congrArg₂ KEquality.rows
    · exact outputChildEvaluation_eq_of_constraintPolynomial_eq
        template left right same child matrix lane
    · exact congrArg
        (fun coordinate => coordinate.children child)
        (piDecEvaluationCoordinate_eq_of_constraintPolynomial_eq
          template left right same matrix lane)
  unfold ConcreteNifsOutputRows.childRows
  rw [commitmentEqual, publicEqual, pointEqual, evaluationEqual]

theorem outputRows_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    outputRows (template.withSystem left) =
      outputRows (template.withSystem right) := by
  unfold outputRows ConcreteNifsOutputRows.rows
  apply congrArg List.flatten
  rw [List.map_ofFn, List.map_ofFn]
  apply congrArg List.ofFn
  funext child
  exact outputChildRows_eq_of_constraintPolynomial_eq
    template left right same child

theorem translate_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (leftSource rightSource :
      List Nightstream.Implementation.R1CS.Row)
    (sourceEqual : leftSource = rightSource) :
    ConcreteNifsRawProgram.translate
        (application (template.withSystem left))
        (invokePlan (template.withSystem left)).frame
        leftSource =
      ConcreteNifsRawProgram.translate
        (application (template.withSystem right))
        (invokePlan (template.withSystem right)).frame
        rightSource := by
  subst rightSource
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4NifsRows
