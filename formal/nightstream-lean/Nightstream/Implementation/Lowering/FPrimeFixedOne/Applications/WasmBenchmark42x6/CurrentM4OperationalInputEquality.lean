import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperationalInput

/-!
Contract: compose the current WASM benchmark's compact physical-frame
stability lemmas into equality of the complete operational Split-NC input.

Assurance tier: model-level.

Owns: named physical transcript projections, stability of every FE and NC
message family, equality of the complete dependent transcript input, and
equality of the joined operational input when two relation structures select
the same constraint polynomial.

Does not own: emitted rows, later NIFS row families, activation, the recursive
fixed point, Rust, or generated artifacts.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperationalInputEquality

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperationalInput
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4TranscriptMessages
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

noncomputable def feRowRounds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    Fin (ConcreteNifsPlain270Profile.Shape dimensions).rowVariables →
      KFixedPhaseSemanticOccurrence.RoundColumns
        (Verifier.SumCheck.Fe.Drow
          (KSplitNcStaticInput.layoutInput
            (operational setup).constraintPolynomial)) :=
  fun round => {
    coefficients :=
      List.ofFn fun slot =>
        ConcreteNifsOperationalOccurrence.proofColumns
          (application setup).family
          (invokePlan setup).frame
          ((operational setup).messageViews.feRow round slot)
    coefficients_length := by simp
  }

noncomputable def feLaneRounds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.laneVariables →
      KFixedPhaseSemanticOccurrence.RoundColumns 2 :=
  fun round => {
    coefficients :=
      List.ofFn fun slot =>
        ConcreteNifsOperationalOccurrence.proofColumns
          (application setup).family
          (invokePlan setup).frame
          ((operational setup).messageViews.feLane round slot)
    coefficients_length := by simp
  }

noncomputable def ncBlockRounds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.blockVariables →
      KFixedPhaseSemanticOccurrence.RoundColumns 4 :=
  fun round => {
    coefficients :=
      List.ofFn fun slot =>
        ConcreteNifsOperationalOccurrence.proofColumns
          (application setup).family
          (invokePlan setup).frame
          ((operational setup).messageViews.nc (Fin.castAdd _ round) slot)
    coefficients_length := by simp
  }

noncomputable def ncLaneRounds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.laneVariables →
      KFixedPhaseSemanticOccurrence.RoundColumns 4 :=
  fun round => {
    coefficients :=
      List.ofFn fun slot =>
        ConcreteNifsOperationalOccurrence.proofColumns
          (application setup).family
          (invokePlan setup).frame
          ((operational setup).messageViews.nc (Fin.natAdd _ round) slot)
    coefficients_length := by simp
  }

noncomputable def temporaryColumns
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows)
    (index : Nat) :=
  ConcreteNifsOperationalOccurrence.temporaryK
    (application setup).family
    (invokePlan setup).frame
    index

theorem transcript_feRowRounds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (transcript setup).fe.rowRounds = feRowRounds setup := by
  rfl

theorem transcript_feLaneRounds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (transcript setup).fe.laneRounds = feLaneRounds setup := by
  rfl

theorem transcript_ncBlockRounds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (transcript setup).nc.blockRounds = ncBlockRounds setup := by
  rfl

theorem transcript_ncLaneRounds
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (transcript setup).nc.laneRounds = ncLaneRounds setup := by
  rfl

theorem transcript_feInitial
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (transcript setup).fe.initial = temporaryColumns setup 0 := by
  rfl

theorem transcript_feBoundary
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (transcript setup).fe.boundary = temporaryColumns setup 1 := by
  rfl

theorem transcript_feTerminal
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (transcript setup).fe.terminal = temporaryColumns setup 2 := by
  rfl

theorem transcript_ncInitial
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (transcript setup).nc.initial = temporaryColumns setup 3 := by
  rfl

theorem transcript_ncTerminal
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (setup : RelationSetup dimensions verifierRows) :
    (transcript setup).nc.terminal = temporaryColumns setup 4 := by
  rfl

private theorem roundColumns_ext
    {degree : Nat}
    (left right :
      KFixedPhaseSemanticOccurrence.RoundColumns degree)
    (coefficients : left.coefficients = right.coefficients) :
    left = right := by
  cases left
  cases right
  simp only at coefficients
  cases coefficients
  rfl

private theorem feColumns_ext
    {shape : SemanticShape}
    {polynomial :
      Verifier.PublicInput shape}
    {domains :
      Verifier.Protocol.TranscriptAuthority.BlockLane.Domains}
    (left right : KSplitNcTranscript.FeColumns polynomial domains)
    (initial : left.initial = right.initial)
    (rowRounds : left.rowRounds = right.rowRounds)
    (boundary : left.boundary = right.boundary)
    (laneRounds : left.laneRounds = right.laneRounds)
    (terminal : left.terminal = right.terminal) :
    left = right := by
  cases left
  cases right
  simp only at *
  cases initial
  cases rowRounds
  cases boundary
  cases laneRounds
  cases terminal
  rfl

private theorem ncColumns_ext
    {domains :
      Verifier.Protocol.TranscriptAuthority.BlockLane.Domains}
    (left right : KSplitNcTranscript.NcColumns domains)
    (initial : left.initial = right.initial)
    (blockRounds : left.blockRounds = right.blockRounds)
    (laneRounds : left.laneRounds = right.laneRounds)
    (terminal : left.terminal = right.terminal) :
    left = right := by
  cases left
  cases right
  simp only at *
  cases initial
  cases blockRounds
  cases laneRounds
  cases terminal
  rfl

private theorem transcript_ext
    {shape : SemanticShape}
    {polynomial :
      Verifier.PublicInput shape}
    {domains :
      Verifier.Protocol.TranscriptAuthority.BlockLane.Domains}
    (left right : KSplitNcTranscript.Input polynomial domains)
    (transcriptBase :
      left.transcriptBase = right.transcriptBase)
    (priorLanes : left.priorLanes = right.priorLanes)
    (priorAbsorbed : left.priorAbsorbed = right.priorAbsorbed)
    (statementFields :
      left.statementFields = right.statementFields)
    (outputFields : left.outputFields = right.outputFields)
    (fe : left.fe = right.fe)
    (nc : left.nc = right.nc) :
    left = right := by
  cases left
  cases right
  simp only at *
  cases transcriptBase
  cases priorLanes
  cases priorAbsorbed
  cases statementFields
  cases outputFields
  cases fe
  cases nc
  rfl

private theorem operationalInput_ext
    {shape : SemanticShape}
    {polynomial :
      Verifier.PublicInput shape}
    {domains :
      Verifier.Protocol.TranscriptAuthority.BlockLane.Domains}
    (left right : KSplitNcOperationalRows.Input polynomial domains)
    (transcript : left.transcript = right.transcript)
    (authority : left.authority = right.authority) :
    left = right := by
  cases left
  cases right
  simp only at *
  cases transcript
  cases authority
  rfl

theorem feRowRounds_heq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    HEq
      (feRowRounds (template.withSystem left))
      (feRowRounds (template.withSystem right)) := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          apply heq_of_eq
          apply funext
          intro round
          apply roundColumns_ext
          apply congrArg List.ofFn
          funext slot
          exact
            feRowCoefficient_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl round slot

theorem feLaneRounds_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    feLaneRounds (template.withSystem left) =
      feLaneRounds (template.withSystem right) := by
  apply funext
  intro round
  apply roundColumns_ext
  simp only [feLaneRounds]
  apply congrArg List.ofFn
  funext slot
  exact
    feLaneCoefficient_eq_of_constraintPolynomial_eq
      template left right same round slot

theorem ncBlockRounds_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ncBlockRounds (template.withSystem left) =
      ncBlockRounds (template.withSystem right) := by
  apply funext
  intro round
  apply roundColumns_ext
  simp only [ncBlockRounds]
  apply congrArg List.ofFn
  funext slot
  exact
    ncCoefficient_eq_of_constraintPolynomial_eq
      template left right same (Fin.castAdd _ round) slot

theorem ncLaneRounds_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    ncLaneRounds (template.withSystem left) =
      ncLaneRounds (template.withSystem right) := by
  apply funext
  intro round
  apply roundColumns_ext
  simp only [ncLaneRounds]
  apply congrArg List.ofFn
  funext slot
  exact
    ncCoefficient_eq_of_constraintPolynomial_eq
      template left right same (Fin.natAdd _ round) slot

theorem temporaryColumns_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial)
    (index : Nat) :
    temporaryColumns (template.withSystem left) index =
      temporaryColumns (template.withSystem right) index := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          rfl

theorem fe_heq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    HEq
      (transcript (template.withSystem left)).fe
      (transcript (template.withSystem right)).fe := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          apply heq_of_eq
          apply feColumns_ext
          · rw [transcript_feInitial, transcript_feInitial]
            exact temporaryColumns_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl 0
          · rw [transcript_feRowRounds, transcript_feRowRounds]
            exact eq_of_heq
              (feRowRounds_heq_of_constraintPolynomial_eq
                template
                { matrices := leftMatrices
                  constraintPolynomial := polynomial }
                { matrices := rightMatrices
                  constraintPolynomial := polynomial }
                rfl)
          · rw [transcript_feBoundary, transcript_feBoundary]
            exact temporaryColumns_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl 1
          · rw [transcript_feLaneRounds, transcript_feLaneRounds]
            exact feLaneRounds_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl
          · rw [transcript_feTerminal, transcript_feTerminal]
            exact temporaryColumns_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl 2

theorem nc_eq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (transcript (template.withSystem left)).nc =
      (transcript (template.withSystem right)).nc := by
  apply ncColumns_ext
  · rw [transcript_ncInitial, transcript_ncInitial]
    exact temporaryColumns_eq_of_constraintPolynomial_eq
      template left right same 3
  · rw [transcript_ncBlockRounds, transcript_ncBlockRounds]
    exact ncBlockRounds_eq_of_constraintPolynomial_eq
      template left right same
  · rw [transcript_ncLaneRounds, transcript_ncLaneRounds]
    exact ncLaneRounds_eq_of_constraintPolynomial_eq
      template left right same
  · rw [transcript_ncTerminal, transcript_ncTerminal]
    exact temporaryColumns_eq_of_constraintPolynomial_eq
      template left right same 4

theorem transcript_heq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    HEq
      (transcript (template.withSystem left))
      (transcript (template.withSystem right)) := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          apply heq_of_eq
          apply transcript_ext
          · exact transcriptBase_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl
          · exact priorLanes_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl
          · exact priorAbsorbed_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl
          · exact statementFields_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl
          · exact outputFields_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl
          · exact eq_of_heq
              (fe_heq_of_constraintPolynomial_eq
                template
                { matrices := leftMatrices
                  constraintPolynomial := polynomial }
                { matrices := rightMatrices
                  constraintPolynomial := polynomial }
                rfl)
          · exact nc_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl

theorem input_heq_of_constraintPolynomial_eq
    {dimensions : Dimensions}
    {verifierRows : Nat}
    (template : SetupTemplate dimensions verifierRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    HEq
      (input (template.withSystem left))
      (input (template.withSystem right)) := by
  cases left with
  | mk leftMatrices polynomial =>
      cases right with
      | mk rightMatrices rightPolynomial =>
          simp only at same
          subst rightPolynomial
          apply heq_of_eq
          apply operationalInput_ext
          · exact eq_of_heq
              (transcript_heq_of_constraintPolynomial_eq
                template
                { matrices := leftMatrices
                  constraintPolynomial := polynomial }
                { matrices := rightMatrices
                  constraintPolynomial := polynomial }
                rfl)
          · exact authority_eq_of_constraintPolynomial_eq
              template
              { matrices := leftMatrices
                constraintPolynomial := polynomial }
              { matrices := rightMatrices
                constraintPolynomial := polynomial }
              rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperationalInputEquality
