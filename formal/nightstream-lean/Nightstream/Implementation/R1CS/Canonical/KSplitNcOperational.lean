import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPhases
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.BlockLane

/-!
Contract: exact operational composition of the physical Split-NC transcript
and claimed-chain rows.

This module does not compute the four verifier-owned endpoint scalars.  It
names those equations explicitly so their dedicated row programs can be
composed without ever moving an acceptance result, challenge point, or
transcript state into a premise.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcOperational

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
open Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptSemantics
open Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptPhases
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

/-- The message-only certificate decoded from the physical round columns.
The raw output message remains an explicit semantic value until its dedicated
codec rows are composed at the call-frame boundary. -/
def certificate
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (message : OutputMessage shape) :
    Protocol.BlockLane.Certificate polynomialInput domains where
  fe := feCertificate assignment input
  nc := ncCertificate assignment input
  output := message

/-- The exact four endpoint equations still to be supplied by arithmetic row
programs.  No equation here is a claimed-chain equation or acceptance result. -/
structure EndpointAgrees
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (message : OutputMessage shape) : Prop where
  feInitial :
    decodedColumns assignment input.fe.initial =
      semanticFeInitial profile constants assignment input
  feTerminal :
    decodedColumns assignment input.fe.terminal =
      Polynomial.Fe.terminalFromMessage profile polynomialInput
        (semanticPre constants assignment input).challenges.feCoins
        (semanticFeExecution profile constants assignment input).challengePoint
        message
  ncInitial :
    decodedColumns assignment input.nc.initial =
      Polynomial.Nc.BlockLane.InitialSum.claimedInitial
  ncTerminal :
    decodedColumns assignment input.nc.terminal =
      Polynomial.Nc.BlockLane.Terminal.terminalFromMessage message
        (semanticPre constants assignment input).challenges.ncCoins
        (semanticNcExecution profile constants assignment input).challengePoint

/-- Satisfying the transcript rows and numeric claimed-chain rows, together
with the four endpoint computations, is exactly enough for the selected
deterministic block×lane verifier to accept.

The conclusion is the unchanged operational relation.  In particular no
paper event is inserted into the physical call contract. -/
theorem accepted_of_rows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (profile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (input : KSplitNcTranscript.Input polynomialInput domains)
    (message : OutputMessage shape)
    (transcriptValid :
      SymbolicDuplexSemantics.Valid input.transcriptBase constants assignment
        (KSplitNcTranscript.outputBuilder input))
    (endpoints :
      EndpointAgrees profile constants assignment input message)
    (numericBase : Nat)
    (numericSatisfied :
      Satisfies
        (KSplitNcBlockLaneRows.rows
          (KSplitNcTranscript.numericColumns input) numericBase)
        assignment) :
    Protocol.BlockLane.Accepted
      (fun _ : Unit => polynomialInput)
      (valueSchedule constants assignment input)
      (priorState assignment input)
      profile unitStatement
      (certificate assignment input message) := by
  have feReplay :=
    decoded_fe profile constants assignment constantWire input
      transcriptValid endpoints.feInitial
  have ncReplay :=
    decoded_nc profile constants assignment constantWire input
      transcriptValid endpoints.feInitial
  have chains :=
    KSplitNcBlockLaneRows.accepted_of_rows
      (KSplitNcTranscript.numericColumns input)
      numericBase assignment constantWire
      (decodedColumns assignment input.fe.initial)
      (decodedColumns assignment input.fe.terminal)
      (decodedColumns assignment input.nc.initial)
      (decodedColumns assignment input.nc.terminal)
      (decodedFePoint assignment input)
      (feCertificate assignment input)
      (decodedNcPoint assignment input)
      (ncCertificate assignment input)
      (feAgrees assignment input)
      (ncAgrees assignment input)
      numericSatisfied
  rcases chains with ⟨feAccepted, ncAccepted⟩
  unfold Protocol.BlockLane.Accepted
  change
    SumCheck.Fe.Accepted
        (Polynomial.Fe.initial profile polynomialInput
          (semanticPre constants assignment input).challenges.feCoins)
        (Polynomial.Fe.terminalFromMessage profile polynomialInput
          (semanticPre constants assignment input).challenges.feCoins
          (semanticFeExecution profile constants assignment input).challengePoint
          message)
        (semanticFeExecution profile constants assignment input).challengePoint
        (feCertificate assignment input) ∧
      SumCheck.Nc.Accepted
        Polynomial.Nc.BlockLane.InitialSum.claimedInitial
        (semanticNcExecution profile constants assignment input).challengePoint.coordinates
        (Polynomial.Nc.BlockLane.Terminal.terminalFromMessage message
          (semanticPre constants assignment input).challenges.ncCoins
          (semanticNcExecution profile constants assignment input).challengePoint)
        (ncCertificate assignment input).toSumCheck
  constructor
  · have bound := feAccepted
    rw [endpoints.feInitial, endpoints.feTerminal, feReplay.point] at bound
    simpa only [semanticFeInitial] using bound
  · have bound := ncAccepted
    rw [endpoints.ncInitial, endpoints.ncTerminal, ncReplay.point] at bound
    exact bound

end Nightstream.Implementation.R1CS.Canonical.KSplitNcOperational
