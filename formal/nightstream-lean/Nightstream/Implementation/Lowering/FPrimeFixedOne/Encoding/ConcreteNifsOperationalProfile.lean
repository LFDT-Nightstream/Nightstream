import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalFrame
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23ApplicationProfile
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalMachine
import Nightstream.Implementation.R1CS.Canonical.KSplitNcStaticInput
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCursor
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains

/-!
Contract: the finite serialization profile needed to lower the selected
fixed-active NIFS `PiCCS` occurrence.

The profile contains only representation data:

* one Lean-selected Poseidon2 schedule and its concrete constants;
* codec projections for every prior-state lane and raw prover message;
* a fixed verifier-owned constraint polynomial and absorb cursor; and
* exact serialization equations for the complete typed statement and output.

Successful proof decoding supplies the two static invariants.  No field below
can carry verifier acceptance, a SumCheck equation, a challenge, an output
claim, source authority, or a paper-event branch.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 800000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCodecProjection
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private abbrev TranscriptState := Poseidon2Duplex.State

/-- One base-field word in a complete statement or output serialization.
Constants are verifier-owned words; all other cases are direct codec
coordinates of one authoritative call operand. -/
inductive FieldSource
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (runningCodec :
      Codec
        (SelectedRunning shape publicRingColumns publicFits verifierRows))
    (freshCodec :
      Codec
        (SelectedFresh shape publicRingColumns publicFits verifierRows))
    (proofCodec :
      Codec
        (SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows)) where
  | constant (value : Field)
  | running
      (value :
        SelectedRunning shape publicRingColumns publicFits verifierRows →
          Field)
      (view : FView runningCodec value)
  | fresh
      (value :
        SelectedFresh shape publicRingColumns publicFits verifierRows →
          Field)
      (view : FView freshCodec value)
  | proof
      (value :
        SelectedProof shape TranscriptState publicRingColumns publicFits
            verifierRows →
          Field)
      (view : FView proofCodec value)

namespace FieldSource

/-- Value-level word selected by one source. -/
def value
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {runningCodec :
      Codec
        (SelectedRunning shape publicRingColumns publicFits verifierRows)}
    {freshCodec :
      Codec
        (SelectedFresh shape publicRingColumns publicFits verifierRows)}
    {proofCodec :
      Codec
        (SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows)}
    (source :
      FieldSource
        (shape := shape) (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        runningCodec freshCodec proofCodec)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : Field :=
  match source with
  | .constant value => value
  | .running value _ => value running
  | .fresh value _ => value fresh
  | .proof value _ => value proof

end FieldSource

/-- Exact proof-codec views of every physical FE/NC message coefficient.
The row-phase slot count is indexed by the selected static polynomial; lane
and NC widths are the protocol-fixed three and five extension values. -/
structure MessageViews
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (proofCodec :
      Codec
        (SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows))
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount) where
  feRow :
    ∀ (round : Fin shape.rowVariables)
      (slot :
        Fin
          (SumCheck.Fe.Drow
            (KSplitNcStaticInput.layoutInput constraintPolynomial) + 1)),
      KView proofCodec
        (fun proof =>
          (proof.certificate.piCcs.fe.rowRounds round).coefficients.getD
            slot.val K.zero)
  feLane :
    ∀ (round :
        Fin
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.fe.laneVariables)
      (slot : Fin 3),
      KView proofCodec
        (fun proof =>
          (proof.certificate.piCcs.fe.laneRounds round).coefficients.getD
            slot.val K.zero)
  nc :
    ∀ (round :
        Fin
          (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.blockVariables +
            Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production.nc.laneVariables))
      (slot : Fin 5),
      KView proofCodec
        (fun proof =>
          (proof.certificate.piCcs.nc.rounds round).coefficients.getD
            slot.val K.zero)

/-- Exact proof-codec coordinates of the carried ΠRLC challenge vector.
These are representation laws only: the canonical sampler rows compute the
values independently before equality rows bind them to these coordinates. -/
structure SamplerViews
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (proofCodec :
      Codec
        (SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows)) where
  challenge :
    ∀ (coordinate :
        Fin
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total)
      (position : Fin ringDegree),
      FView proofCodec
        (fun proof =>
          proof.certificate.piRlcChallenges coordinate position)

/-- Complete finite profile for one selected operational `PiCCS` occurrence.
Every proof field is a codec-domain or serialization law. -/
structure Profile
    {shape : SemanticShape}
    {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
    [DecidableEq AppState] [DecidableEq Encoded]
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {keys : Fin 1 →
      SelectedKey shape TranscriptState publicRingColumns publicFits
        verifierRows}
    {defaultRunning :
      SelectedRunning shape publicRingColumns publicFits verifierRows}
    {machine :
      Machine
        (SelectedKey shape TranscriptState publicRingColumns publicFits
          verifierRows)
        Digest AppState Witness
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        Encoded 1}
    {terminalRelations :
      TerminalRelations
        (SelectedKey shape TranscriptState publicRingColumns publicFits
          verifierRows)
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        RunningWitness
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        FreshWitness 1}
    {terminalChecks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        terminalRelations}
    {widths : Widths} {footprints : Footprints}
    (application :
      Poseidon23ApplicationProfile
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)) where
  constants : Poseidon2Schedule.Constants
  serialization :
    KSplitNcPoseidonSchedule.Serialization
      (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.VerifierKey
        shape publicRingColumns publicFits verifierRows)
      (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.StatementInput
        shape publicRingColumns publicFits verifierRows
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity)
      shape
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial F shape.matrixCount
  priorAbsorbed : Nat
  proofAdmissiblePolynomial :
    ∀ proof,
      (application.family
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)).codecFor
          (.data .nifsProof) |>.Admissible proof →
      proof.piCcsInput.constraintPolynomial = constraintPolynomial
  proofAdmissibleCursor :
    ∀ proof,
      (application.family
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)).codecFor
          (.data .nifsProof) |>.Admissible proof →
      proof.priorState.absorbed = priorAbsorbed
  proofAdmissiblePriorState :
    ∀ proof,
      (application.family
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)).codecFor
          (.data .nifsProof) |>.Admissible proof →
      proof.priorState = Poseidon2Duplex.empty
  proofAdmissibleLanes :
    ∀ proof,
      (application.family
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)).codecFor
          (.data .nifsProof) |>.Admissible proof →
      ∀ lane,
        proof.priorState.lanes lane <
          Nightstream.Implementation.R1CS.goldilocksP
  selectedSchedule :
    (keys
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).template.piCcsSchedule =
      KSplitNcPoseidonSchedule.schedule
        (domains :=
          Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains.production)
        constants serialization
  selectedSamplerMachine :
    (keys
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).template.piRlcMachine =
      PiRlcCanonicalMachine.machine constants
  priorLane :
    ∀ lane : Fin 8,
      FView
        ((application.family
          (ConcreteNifsParameters.selected keys defaultRunning machine
            terminalRelations terminalChecks widths footprints)).codecFor
              (.data .nifsProof))
        (fun proof =>
          Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue
            (proof.priorState.lanes lane))
  statementSources :
    List
      (FieldSource
        (shape := shape) (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        ((application.family
          (ConcreteNifsParameters.selected keys defaultRunning machine
            terminalRelations terminalChecks widths footprints)).codecFor
              (.data .running))
        ((application.family
          (ConcreteNifsParameters.selected keys defaultRunning machine
            terminalRelations terminalChecks widths footprints)).codecFor
              (.data .fresh))
        ((application.family
          (ConcreteNifsParameters.selected keys defaultRunning machine
            terminalRelations terminalChecks widths footprints)).codecFor
              (.data .nifsProof)))
  statementExact :
    ∀ running fresh proof,
      statementSources.map
          (fun source => (source.value running fresh proof).val) =
        serialization.statementFields
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize.piCcsStatement
  statementLength :
    statementSources.length =
      10 +
        ((application.family
          (ConcreteNifsParameters.selected keys defaultRunning machine
            terminalRelations terminalChecks widths footprints)).codecFor
              (.data .running)).width +
        ((application.family
          (ConcreteNifsParameters.selected keys defaultRunning machine
            terminalRelations terminalChecks widths footprints)).codecFor
              (.data .fresh)).width +
        shape.rowVariables * 2 +
        shape.runningCount * shape.matrixCount * ringDegree * 2
  outputSources :
    List
      (FieldSource
        (shape := shape) (publicRingColumns := publicRingColumns)
        (verifierRows := verifierRows) (publicFits := publicFits)
        ((application.family
          (ConcreteNifsParameters.selected keys defaultRunning machine
            terminalRelations terminalChecks widths footprints)).codecFor
              (.data .running))
        ((application.family
          (ConcreteNifsParameters.selected keys defaultRunning machine
            terminalRelations terminalChecks widths footprints)).codecFor
              (.data .fresh))
        ((application.family
          (ConcreteNifsParameters.selected keys defaultRunning machine
            terminalRelations terminalChecks widths footprints)).codecFor
              (.data .nifsProof)))
  outputExact :
    ∀ running fresh proof,
      outputSources.map
          (fun source => (source.value running fresh proof).val) =
        serialization.outputFields proof.certificate.piCcs.output
  outputLength :
    outputSources.length =
      3 +
        shape.sourceCount * shape.matrixCount * ringDegree * 2 +
        shape.sourceCount * ringDegree * 2
  outputCursorOne :
    SymbolicDuplexCursor.after 0 (2 + outputSources.length) = 1
  messageViews :
    MessageViews
      (shape := shape) (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows) (publicFits := publicFits)
      ((application.family
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)).codecFor
            (.data .nifsProof))
      constraintPolynomial
  samplerViews :
    SamplerViews
      (shape := shape) (publicRingColumns := publicRingColumns)
      (verifierRows := verifierRows) (publicFits := publicFits)
      ((application.family
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)).codecFor
            (.data .nifsProof))
  endpointViews :
    ConcreteNifsOperationalFrame.ProofViews
      ((application.family
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)).codecFor
            (.data .nifsProof))
  runningViews :
    ConcreteNifsCarrierViews.RunningViews
      ((application.family
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)).codecFor
            (.data .running))
  runningCoverage :
    ConcreteNifsCarrierViews.RunningCodecCoverage
      ((application.family
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)).codecFor
            (.data .running))
      runningViews
  freshViews :
    ConcreteNifsCarrierViews.FreshViews
      ((application.family
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)).codecFor
            (.data .fresh))
  payloadViews :
    ConcreteNifsCarrierViews.PayloadViews
      ((application.family
        (ConcreteNifsParameters.selected keys defaultRunning machine
          terminalRelations terminalChecks widths footprints)).codecFor
            (.data .nifsProof))

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
