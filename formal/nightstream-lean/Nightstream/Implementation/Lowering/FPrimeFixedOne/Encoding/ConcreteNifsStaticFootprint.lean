import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsActivatedProgram
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalOperationalProfile
import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptControl
import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptCount

/-!
Contract: derive one frame-independent selected-NIFS footprint from the
Lean-owned activated verifier program.

Assurance tier: model-level.

Owns: a value-free control input, its exact cost, and equality between that
cost and every canonical physical `nifsVerify` occurrence.

Does not own: an application step, recursive fixed-point dimensions, Rust,
or generated artifacts.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsStaticFootprint

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private abbrev TranscriptState := Poseidon2Duplex.State
private abbrev KColumns :=
  Nightstream.Implementation.R1CS.ProjectionProgram.KColumns

/-- Number of fields in the canonical selected statement. -/
noncomputable def statementFieldCount
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) : Nat :=
  10 +
    (ConcreteNifsCanonicalRunningCodec.runningCodec
      shape publicRingColumns verifierRows publicFits).width +
    (ConcreteNifsCanonicalRunningCodec.freshCodec
      shape publicRingColumns verifierRows publicFits).width +
    shape.rowVariables * 2 +
    shape.runningCount * shape.matrixCount * ringDegree * 2

/-- Number of fields in the canonical selected output. -/
def outputFieldCount (shape : SemanticShape) : Nat :=
  3 +
    shape.sourceCount * shape.matrixCount * ringDegree * 2 +
    shape.sourceCount * ringDegree * 2

private def zeroColumns : KColumns :=
  ⟨0, 0⟩

private def zeroRound (degree : Nat) :
    KFixedPhaseSemanticOccurrence.RoundColumns degree where
  coefficients := List.replicate (degree + 1) zeroColumns
  coefficients_length := by simp

/-- Value-free transcript input with the exact selected control shape.
Physical values and columns are zero because only the control path is priced.
-/
noncomputable def transcriptInput
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    KSplitNcTranscript.Input
      (KSplitNcStaticInput.layoutInput constraintPolynomial)
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production where
  transcriptBase := 0
  priorLanes := fun _ => []
  priorAbsorbed := 0
  statementFields :=
    List.replicate
      (statementFieldCount shape publicRingColumns verifierRows publicFits) []
  outputFields := List.replicate (outputFieldCount shape) []
  fe := {
    initial := zeroColumns
    rowRounds := fun _ =>
      zeroRound
        (SumCheck.Fe.Drow
          (KSplitNcStaticInput.layoutInput constraintPolynomial))
    boundary := zeroColumns
    laneRounds := fun _ => zeroRound 2
    terminal := zeroColumns
  }
  nc := {
    initial := zeroColumns
    blockRounds := fun _ => zeroRound 4
    laneRounds := fun _ => zeroRound 4
    terminal := zeroColumns
  }

/-- Value-free endpoint authority with the exact selected index domains. -/
def authority (shape : SemanticShape) :
    KSplitNcEndpoints.AuthorityColumns shape where
  priorPoint := fun _ => KLinear.zeroCarried
  claimedYRing := fun _ _ _ => KLinear.zeroCarried
  outputYRing := fun _ _ _ => KLinear.zeroCarried
  outputYZcol := fun _ _ => KLinear.zeroCarried

/-- Complete value-free operational input used only to derive static cost. -/
noncomputable def operationalInput
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    KSplitNcOperationalRows.Input
      (KSplitNcStaticInput.layoutInput constraintPolynomial)
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production where
  transcript :=
    transcriptInput shape constraintPolynomial publicRingColumns verifierRows
      publicFits
  authority := authority shape

/-- Small control witness used to price non-transcript operational rows.
Statement and output values do not affect their row or allocation count. -/
noncomputable def compactTranscriptInput
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    KSplitNcTranscript.Input
      (KSplitNcStaticInput.layoutInput constraintPolynomial)
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production :=
  { transcriptInput shape constraintPolynomial publicRingColumns verifierRows
      publicFits with
    statementFields := []
    outputFields := []
  }

/-- Small operational witness for the value-independent numeric and endpoint
costs. -/
noncomputable def compactOperationalInput
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    KSplitNcOperationalRows.Input
      (KSplitNcStaticInput.layoutInput constraintPolynomial)
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production where
  transcript :=
    compactTranscriptInput shape constraintPolynomial publicRingColumns
      verifierRows publicFits
  authority := authority shape

/-- Compact operational ΠCCS cost. The transcript count uses only field
counts; the small witness prices the remaining numeric and endpoint rows. -/
noncomputable def compactOperationalCost
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) : Cost :=
  KSplitNcTranscriptCount.cost shape.rowVariables
      (SumCheck.Fe.Drow
        (KSplitNcStaticInput.layoutInput constraintPolynomial))
      PiCcsDomains.production.laneVariables
      PiCcsDomains.production.blockVariables
      0
      (statementFieldCount shape publicRingColumns verifierRows publicFits)
      (outputFieldCount shape) +
    KSplitNcBlockLaneRows.cost
      (KSplitNcTranscript.numericColumns
        (compactTranscriptInput shape constraintPolynomial publicRingColumns
          verifierRows publicFits)) +
    KSplitNcOperationalRows.endpointCost
      (compactOperationalInput shape constraintPolynomial publicRingColumns
        verifierRows publicFits)

/-- Static operational ΠCCS and ΠRLC sampler cost. -/
noncomputable def samplerCost
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) : Cost :=
  compactOperationalCost shape constraintPolynomial publicRingColumns
      verifierRows publicFits +
    PiRlcCanonicalSamplerProgram.cost +
    ConcreteNifsOperationalSampler.challengeCost

/-- Static intrinsic selected-NIFS cost before activation. -/
noncomputable def intrinsicCost
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) : Cost :=
  ConcreteNifsRawProgram.claimedValueCost +
    ConcreteNifsProofCanonicalityRows.cost +
    ConcreteNifsRunningAuthorityRows.cost
      shape publicRingColumns verifierRows +
    samplerCost shape constraintPolynomial publicRingColumns verifierRows
      publicFits +
    ConcreteNifsPiRlcPointRows.cost shape.rowVariables +
    ConcreteNifsPiRlcActionRows.cost
      shape publicRingColumns verifierRows +
    ConcreteNifsPiDecRows.cost shape publicRingColumns verifierRows +
    ConcreteNifsOutputRows.cost shape publicRingColumns verifierRows

/-- Static selected-NIFS cost after activation lowering. -/
noncomputable def cost
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) : Cost :=
  ActivatedRawProgram.cost
    (intrinsicCost shape constraintPolynomial publicRingColumns verifierRows
      publicFits)

/-- Exact vocabulary footprint derived without a call frame. -/
noncomputable def footprint
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    CallFootprint where
  recurringRows :=
    (cost shape constraintPolynomial publicRingColumns verifierRows
      publicFits).recurringRows
  temporaries :=
    [auxiliaryLayout
      (cost shape constraintPolynomial publicRingColumns verifierRows
        publicFits).auxiliaryColumns]

@[simp] theorem transcriptInput_statement_length
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (transcriptInput shape constraintPolynomial publicRingColumns verifierRows
      publicFits).statementFields.length =
      statementFieldCount shape publicRingColumns verifierRows publicFits := by
  simp [transcriptInput]

@[simp] theorem transcriptInput_output_length
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    (transcriptInput shape constraintPolynomial publicRingColumns verifierRows
      publicFits).outputFields.length = outputFieldCount shape := by
  simp [transcriptInput]

private theorem endpointAllocation_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (left right : KSplitNcEndpoints.Input polynomialInput domains) :
    KSplitNcEndpoints.allocationWidth left =
      KSplitNcEndpoints.allocationWidth right := by
  rfl

private theorem endpointCost_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (left right : KSplitNcOperationalRows.Input polynomialInput domains) :
    KSplitNcOperationalRows.endpointCost left =
      KSplitNcOperationalRows.endpointCost right := by
  have allocationEq :
      KSplitNcEndpoints.allocationWidth
          (KSplitNcOperationalRows.endpointInput left) =
        KSplitNcEndpoints.allocationWidth
          (KSplitNcOperationalRows.endpointInput right) :=
    endpointAllocation_eq
      (KSplitNcOperationalRows.endpointInput left)
      (KSplitNcOperationalRows.endpointInput right)
  have rowsEq :
      (KSplitNcOperationalRows.endpointRows left).length =
        (KSplitNcOperationalRows.endpointRows right).length := by
    unfold KSplitNcOperationalRows.endpointRows
    rw [KSplitNcEndpoints.rows_length_eq_allocationWidth_add_eight,
      KSplitNcEndpoints.rows_length_eq_allocationWidth_add_eight,
      allocationEq]
  unfold KSplitNcOperationalRows.endpointCost
  rw [rowsEq, allocationEq]

private theorem operationalCost_eq
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (left right : KSplitNcOperationalRows.Input polynomialInput domains)
    (priorAbsorbed :
      left.transcript.priorAbsorbed = right.transcript.priorAbsorbed)
    (statementLength :
      left.transcript.statementFields.length =
        right.transcript.statementFields.length)
    (outputLength :
      left.transcript.outputFields.length =
        right.transcript.outputFields.length) :
    KSplitNcOperationalRows.cost left =
      KSplitNcOperationalRows.cost right := by
  unfold KSplitNcOperationalRows.cost
  rw [KSplitNcTranscriptControl.cost_eq
      left.transcript right.transcript priorAbsorbed statementLength
        outputLength,
    KSplitNcTranscriptControl.numericCost_eq left.transcript right.transcript,
    endpointCost_eq left right]

/-- The large value-free operational witness has the compact arithmetic
cost. No statement or output list is traversed by the right-hand side. -/
theorem operationalInput_cost_eq_compact
    (shape : SemanticShape)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    KSplitNcOperationalRows.cost
        (operationalInput shape constraintPolynomial publicRingColumns
          verifierRows publicFits) =
      compactOperationalCost shape constraintPolynomial publicRingColumns
        verifierRows publicFits := by
  let expanded :=
    operationalInput shape constraintPolynomial publicRingColumns
      verifierRows publicFits
  let compact :=
    compactOperationalInput shape constraintPolynomial publicRingColumns
      verifierRows publicFits
  unfold KSplitNcOperationalRows.cost compactOperationalCost
  rw [KSplitNcTranscriptCount.cost_eq expanded.transcript]
  rw [KSplitNcTranscriptControl.numericCost_eq
    expanded.transcript compact.transcript]
  rw [endpointCost_eq expanded compact]
  have priorAbsorbed :
      expanded.transcript.priorAbsorbed = 0 := by
    rfl
  have statementLength :
      expanded.transcript.statementFields.length =
        statementFieldCount shape publicRingColumns verifierRows publicFits := by
    exact
      transcriptInput_statement_length shape constraintPolynomial
        publicRingColumns verifierRows publicFits
  have outputLength :
      expanded.transcript.outputFields.length = outputFieldCount shape := by
    exact
      transcriptInput_output_length shape constraintPolynomial
        publicRingColumns verifierRows publicFits
  rw [priorAbsorbed, statementLength, outputLength]
  rfl

section CanonicalApplication

variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {dimensions : Dimensions}
variable {verifierRows : Nat}
variable (setup : RelationSetup dimensions verifierRows)
variable (defaultRunning : Running dimensions verifierRows)
variable
  (machine :
    Machine
      (Key dimensions TranscriptState verifierRows)
      Digest AppState Witness
      (Running dimensions verifierRows)
      (Fresh dimensions verifierRows)
      Encoded 1)
variable
  (terminalRelations :
    TerminalRelations
      (Key dimensions TranscriptState verifierRows)
      (Running dimensions verifierRows)
      RunningWitness
      (Fresh dimensions verifierRows)
      FreshWitness 1)
variable
  (terminalChecks :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      terminalRelations)
variable (widths : Widths) (footprints : Footprints)

local notation "Selected" =>
  ConcreteNifsPlain270Profile.selected dimensions
    (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup)
    defaultRunning machine terminalRelations terminalChecks widths footprints

private abbrev CanonicalApplication :=
  ConcreteNifsCanonicalOperationalProfile.Application
    setup defaultRunning machine terminalRelations terminalChecks
      widths footprints

private noncomputable abbrev FamilyFor
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints) :=
  application.phase4.profile.family Selected

private abbrev FrameFor
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)} :=
  CallFrame (signature := signature Selected)
    (FamilyFor setup defaultRunning machine terminalRelations terminalChecks
      widths footprints application)
    Call.nifsVerify
    (Refs.cons runningRef
      (Refs.cons freshRef (Refs.cons proofRef .nil)))

private noncomputable abbrev operationalProfile
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints) :=
  ConcreteNifsCanonicalOperationalProfile.operational
    setup defaultRunning machine terminalRelations terminalChecks
      widths footprints application

/-- Every canonical call frame has the static intrinsic cost. -/
theorem intrinsicCost_eq
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ConcreteNifsRawProgram.cost application.phase4.profile
        (operationalProfile setup defaultRunning machine terminalRelations
          terminalChecks widths footprints application)
        frame =
      intrinsicCost
        (ConcreteNifsPlain270Profile.Shape dimensions)
        setup.system.constraintPolynomial publicRingColumns verifierRows
        (publicFits dimensions) := by
  let profile :=
    operationalProfile setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application
  let actual :=
    ConcreteNifsOperationalOccurrence.input
      application.phase4.profile profile frame
  let static :=
    operationalInput
      (ConcreteNifsPlain270Profile.Shape dimensions)
      profile.constraintPolynomial publicRingColumns verifierRows
      (publicFits dimensions)
  have statementLength :
      actual.transcript.statementFields.length =
        static.transcript.statementFields.length := by
    simp only [actual, static,
      ConcreteNifsOperationalOccurrence.input,
      ConcreteNifsOperationalOccurrence.transcriptInput,
      operationalInput, transcriptInput, List.length_map,
      List.length_replicate]
    rw [
      ConcreteNifsCanonicalOperationalProfile.operational_statementSources_length
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints application]
    rfl
  have outputLength :
      actual.transcript.outputFields.length =
        static.transcript.outputFields.length := by
    simp only [actual, static,
      ConcreteNifsOperationalOccurrence.input,
      ConcreteNifsOperationalOccurrence.transcriptInput,
      operationalInput, transcriptInput, List.length_map,
      List.length_replicate]
    rw [
      ConcreteNifsCanonicalOperationalProfile.operational_outputSources_length
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints application]
    rfl
  have selectedCost :
      KSplitNcOperationalRows.cost actual =
        KSplitNcOperationalRows.cost static := by
    apply operationalCost_eq
    · exact
        ConcreteNifsCanonicalOperationalProfile.operational_priorAbsorbed
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints application
    · exact statementLength
    · exact outputLength
  calc
    ConcreteNifsRawProgram.cost application.phase4.profile profile frame =
        intrinsicCost
          (ConcreteNifsPlain270Profile.Shape dimensions)
          profile.constraintPolynomial publicRingColumns verifierRows
          (publicFits dimensions) := by
      simp only [ConcreteNifsRawProgram.cost, intrinsicCost,
        ConcreteNifsOperationalSampler.cost, samplerCost]
      rw [selectedCost,
        operationalInput_cost_eq_compact
          (ConcreteNifsPlain270Profile.Shape dimensions)
          profile.constraintPolynomial publicRingColumns verifierRows
          (publicFits dimensions)]
    _ =
        intrinsicCost
          (ConcreteNifsPlain270Profile.Shape dimensions)
          setup.system.constraintPolynomial publicRingColumns verifierRows
          (publicFits dimensions) := by
      rw [
        ConcreteNifsCanonicalOperationalProfile.operational_constraintPolynomial
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints application]

/-- Every activated canonical call frame has the frame-independent static
cost. -/
theorem cost_eq
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ConcreteNifsActivatedProgram.cost application.phase4.profile
        (operationalProfile setup defaultRunning machine terminalRelations
          terminalChecks widths footprints application)
        frame =
      cost
        (ConcreteNifsPlain270Profile.Shape dimensions)
        setup.system.constraintPolynomial publicRingColumns verifierRows
        (publicFits dimensions) := by
  unfold ConcreteNifsActivatedProgram.cost
    ConcreteNifsActivatedProgram.intrinsicCost cost
  rw [intrinsicCost_eq setup defaultRunning machine terminalRelations
    terminalChecks widths footprints application frame]

/-- The static footprint equals the footprint emitted for every canonical
physical occurrence. -/
theorem footprint_eq
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    footprint
        (ConcreteNifsPlain270Profile.Shape dimensions)
        setup.system.constraintPolynomial publicRingColumns verifierRows
        (publicFits dimensions) =
      ConcreteNifsActivatedProgram.footprint application.phase4.profile
        (operationalProfile setup defaultRunning machine terminalRelations
          terminalChecks widths footprints application)
        frame := by
  unfold footprint ConcreteNifsActivatedProgram.footprint
  rw [cost_eq setup defaultRunning machine terminalRelations terminalChecks
    widths footprints application frame]

end CanonicalApplication

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsStaticFootprint
